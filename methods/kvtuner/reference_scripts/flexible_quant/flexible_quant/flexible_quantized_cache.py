import torch
from typing import Any, Dict, List, Optional, Tuple
from transformers.cache_utils import DynamicCache, CacheConfig, QuantizedCacheConfig, is_optimum_quanto_available
from transformers.utils import is_hqq_available

from .vanilla_quantizer import VanillaQuantizer

if is_hqq_available():
    from hqq.core.quantize import Quantizer as HQQQuantizer

'''
per_head_config = {
    {n_layer}: {
        {head_idx}: {
            'nbits_key': 4,
            'nbits_value': 4,
        }
    }
'''

class FlexibleQuantizedCacheConfig(QuantizedCacheConfig):
    """
    Configuration for flexible quantized cache.

    Attributes:
        backend (str): Backend for quantization. Options: "quanto", "hqq", "vanilla".
        nbits (Optional[int]): Precision for both key and value. Used if `nbits_key` and `nbits_value` are not set.
                               For per-layer or per-head quantization, set `nbits` to -1.
        nbits_key (Optional[int]): Precision for key quantization. For per-layer or per-head quantization, set to -1.
        nbits_value (Optional[int]): Precision for value quantization. For per-layer or per-head quantization, set to -1.
        axis_key (Optional[int]): Axis for key quantization. In Vanilla mode:
                                  - 0: Per-token quantization
                                  - 1: Per-channel quantization
        axis_value (Optional[int]): Axis for value quantization. In Vanilla mode:
                                    - 0: Per-token quantization
                                    - 1: Per-channel quantization
        asym (Optional[bool]): Whether to use asymmetric quantization. Works only for Vanilla mode.
        q_group_size (Optional[int]): Group size for quantization. Use -1 for per-token quantization.
        residual_length (Optional[int]): Length of residual tokens that are not quantized.
                                         Must be a multiple of `q_group_size`. Use 0 for per-token quantization.
        compute_dtype (Optional[torch.dtype]): Compute dtype for the model. Default: `torch.float16`.
        device (Optional[str]): Device for the cache. Default: `"cpu"`.
        force_quant (Optional[bool]): Whether to quantize the cache during the pre-filling stage.
        per_layer_quant (Optional[bool]): Whether to use per-layer quantization.
        per_layer_config (Optional[Dict[str, Any]]): If `per_layer_quant` is True, provides the quantization config
                                                     for each layer. Alternatively, use `per_layer_config_path`.
        per_layer_config_path (Optional[str]): Path to the quantization config for each layer.
                                               Used if `per_layer_quant` is True.
        per_head_quant (Optional[bool]): Whether to use per-head quantization.
        per_head_config (Optional[Dict[str, Any]]): If `per_head_quant` is True, provides the quantization config
                                                    for each head. Alternatively, use `per_head_config_path`.
        per_head_config_path (Optional[str]): Path to the quantization config for each head.
                                              Used if `per_head_quant` is True.
    """

    cache_implementation = "flexible"
    
    def __init__(
        self,
        backend: str = "quanto",
        nbits: Optional[int] = 4,
        nbits_key: Optional[int] = 0,
        nbits_value: Optional[int] = 0,
        axis_key: Optional[int] = 0,
        axis_value: Optional[int] = 0,
        asym: Optional[bool] = False,
        q_group_size: Optional[int] = 64,
        residual_length: Optional[int] = 128,
        compute_dtype: Optional[torch.dtype] = torch.float16,
        device: Optional[str] = "cpu",
        force_quant: Optional[bool] = False,
        per_layer_quant: Optional[bool] = False,
        per_layer_config: Optional[Dict[str, Any]] = None,
        per_layer_config_path: Optional[str] = None,
        per_head_quant: Optional[bool] = False,
        per_head_config: Optional[Dict[str, Any]] = None,
        per_head_config_path: Optional[str] = None,
    ):
        super().__init__(
            backend=backend,
            nbits=nbits,
            axis_key=axis_key,
            axis_value=axis_value,
            q_group_size=q_group_size,
            residual_length=residual_length,
            compute_dtype=compute_dtype,
            device=device,
        )
        self.nbits_key = nbits_key if nbits_key else nbits
        self.nbits_value = nbits_value if nbits_value else nbits
        self.asym = asym
        self.force_quant = force_quant
        if per_head_quant and per_layer_quant:
            raise ValueError("per_layer_quant and per_head_quant cannot be Enabled at the same time.")
        self.per_layer_quant = per_layer_quant
        if per_layer_quant:
            if per_layer_config is not None:
                self.per_layer_config = per_layer_config
            elif per_layer_config_path is not None:
                import yaml
                with open(per_layer_config_path, 'r') as f:
                    self.per_layer_config = yaml.safe_load(f)
            else:
                raise ValueError("per_layer_quant is set to True but per_layer_config or per_layer_config_path is not provided.")
        self.per_head_quant = per_head_quant
        if per_head_quant:
            if per_head_config is not None:
                self.per_head_config = per_head_config
            elif per_head_config_path is not None:
                import yaml
                with open(per_head_config_path, 'r') as f:
                    self.per_head_config = yaml.safe_load(f)
            else:
                raise ValueError("per_head_quant is set to True but per_head_config or per_head_config_path is not provided.")


class FlexibleQuantizedCache(DynamicCache):
    """
    A quantizer cache similar to what is described in the [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://arxiv.org/abs/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for Key and Value cache by applying quantization.

    The cache has two types of storage, one for original precision and one for the quantized cache. A `residual length` is set as a maximum capacity for the
    original precision cache. When the length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache. The
    quantization is done per-channel with a set `q_group_size` for both Keys and Values, in contrast to what was described in the paper.

    It stores Keys and Values a list of quantized tensors (tuples in case we need to store metadata), one for each layer. Additionally, it stores the Key and
    Value in original precision states as a list of tensors, one for each layer. The size of each tensor
    is `[batch_size, num_heads, seq_len - residual_length, head_dim]`
    """

    def __init__(self, cache_config: FlexibleQuantizedCacheConfig) -> None:
        super().__init__()
        self._quantized_key_cache: List[torch.Tensor] = []
        self._quantized_value_cache: List[torch.Tensor] = []

        self.nbits = cache_config.nbits
        self.nbits_key = cache_config.nbits_key
        self.nbits_value = cache_config.nbits_value
        self.residual_length = cache_config.residual_length
        self.q_group_size = cache_config.q_group_size
        self.axis_key = cache_config.axis_key
        self.axis_value = cache_config.axis_value
        self.asym = cache_config.asym
        self.compute_dtype = cache_config.compute_dtype
        self.device = cache_config.device
        self.force_quant = cache_config.force_quant
        self.per_layer_quant = cache_config.per_layer_quant
        if self.per_layer_quant:
            self.per_layer_config = cache_config.per_layer_config
        self.per_head_quant = cache_config.per_head_quant
        if self.per_head_quant:
            self.per_head_config = cache_config.per_head_config

        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        # print('Called Update, {}, {}'.format(self.nbits_key, self.nbits_value))
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        if len(self.key_cache) < layer_idx:
            raise ValueError("QuantizedCache does not support model usage where layers are skipped. Use DynamicCache.")

        if not self.per_head_quant:
            nbits_key = self.nbits_key if not self.per_layer_quant else self.per_layer_config[layer_idx]['nbits_key']
            nbits_value = self.nbits_value if not self.per_layer_quant else self.per_layer_config[layer_idx]['nbits_value']
            if len(self.key_cache) == layer_idx:
                if self.force_quant:
                    # quirk: use dequantized key/value instead of original key/value
                    if self.residual_length:
                        tokens_to_keep = key_states.shape[-2] % self.residual_length
                        # keep tokens_to_keep by slicing the cache in axis -2
                        self._quantized_key_cache.append(self._quantize(key_states[..., :-tokens_to_keep, :], axis=self.axis_key, nbits=nbits_key))
                        self._quantized_value_cache.append(self._quantize(value_states[..., :-tokens_to_keep, :], axis=self.axis_value, nbits=nbits_value))
                        self.key_cache.append(key_states[..., -tokens_to_keep:, :])
                        self.value_cache.append(value_states[..., -tokens_to_keep:, :])
                    else:
                        self._quantized_key_cache.append(self._quantize(key_states, axis=self.axis_key, nbits=nbits_key))
                        self._quantized_value_cache.append(self._quantize(value_states, axis=self.axis_value, nbits=nbits_value))
                        self.key_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
                        self.value_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
                    keys_to_return = [self._dequantize(self._quantized_key_cache[layer_idx]), self.key_cache[layer_idx]]
                    values_to_return = [self._dequantize(self._quantized_value_cache[layer_idx]), self.value_cache[layer_idx]]
                    keys_to_return = torch.cat(keys_to_return, dim=-2)
                    values_to_return = torch.cat(values_to_return, dim=-2)
                else:
                    self._quantized_key_cache.append(self._quantize(key_states.contiguous(), axis=self.axis_key, nbits=nbits_key))
                    self._quantized_value_cache.append(self._quantize(value_states.contiguous(), axis=self.axis_value, nbits=nbits_value))
                    self.key_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
                    self.value_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
                    keys_to_return, values_to_return = key_states, value_states
            else:
                dequant_key = self._dequantize(self._quantized_key_cache[layer_idx])
                dequant_value = self._dequantize(self._quantized_value_cache[layer_idx])
                keys_to_return = [dequant_key, self.key_cache[layer_idx], key_states]
                values_to_return = [dequant_value, self.value_cache[layer_idx], value_states]

                keys_to_return = torch.cat(keys_to_return, dim=-2)
                values_to_return = torch.cat(values_to_return, dim=-2)
                if (
                    self.key_cache[layer_idx].dim() == 4
                    and self.key_cache[layer_idx].shape[-2] + 1 >= self.residual_length
                ):
                    self._quantized_key_cache[layer_idx] = self._quantize(keys_to_return.contiguous(), axis=self.axis_key, nbits=nbits_key)
                    self._quantized_value_cache[layer_idx] = self._quantize(
                        values_to_return.contiguous(), axis=self.axis_value, nbits=nbits_value
                    )
                    self.key_cache[layer_idx] = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
                    self.value_cache[layer_idx] = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
                else:
                    self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                    self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        else: # per head quant
            assert key_states.dim() == 4 and value_states.dim() == 4
            key_states, value_states = key_states.transpose(0, 1).contiguous(), value_states.transpose(0, 1).contiguous()
            assert key_states.shape[0]  == value_states.shape[0]
            if key_states.shape[0] != len(self.per_head_config[layer_idx]):
                raise ValueError("Number of heads in the model does not match the number of heads in the per_head_config, you may loaded a invalid config file.")
            keys_to_return, values_to_return = [], []
            if len(self.key_cache) == layer_idx:
                self._quantized_key_cache.append([])
                self._quantized_value_cache.append([])
                self.key_cache.append([])
                self.value_cache.append([])
                for head_idx, config in self.per_head_config[layer_idx].items():
                    if self.force_quant:
                        if self.residual_length:
                            tokens_to_keep = key_states.shape[-2] % self.residual_length
                            self._quantized_key_cache[layer_idx].append(self._quantize(key_states[head_idx][..., :-tokens_to_keep, :], axis=self.axis_key, nbits=config['nbits_key']))
                            self._quantized_value_cache[layer_idx].append(self._quantize(value_states[head_idx][..., :-tokens_to_keep, :], axis=self.axis_value, nbits=config['nbits_value']))
                            self.key_cache[layer_idx].append(key_states[head_idx][..., -tokens_to_keep:, :])
                            self.value_cache[layer_idx].append(value_states[head_idx][..., -tokens_to_keep:, :])
                        else:
                            self._quantized_key_cache[layer_idx].append(self._quantize(key_states[head_idx], axis=self.axis_key, nbits=config['nbits_key']))
                            self._quantized_value_cache[layer_idx].append(self._quantize(value_states[head_idx], axis=self.axis_value, nbits=config['nbits_value']))
                            self.key_cache[layer_idx].append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
                            self.value_cache[layer_idx].append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
                        keys_to_return.append(torch.cat([self._dequantize(self._quantized_key_cache[layer_idx][-1]), self.key_cache[layer_idx][-1]], dim=-2))
                        values_to_return.append(torch.cat([self._dequantize(self._quantized_value_cache[layer_idx][-1]), self.value_cache[layer_idx][-1]], dim=-2))
                    else:
                        self._quantized_key_cache[layer_idx].append(self._quantize(key_states[head_idx].contiguous(), axis=self.axis_key, nbits=config['nbits_key']))
                        self._quantized_value_cache[layer_idx].append(self._quantize(value_states[head_idx].contiguous(), axis=self.axis_value, nbits=config['nbits_value']))
                        self.key_cache[layer_idx].append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
                        self.value_cache[layer_idx].append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
                        keys_to_return.append(key_states[head_idx])
                        values_to_return.append(value_states[head_idx])
            else:
                for head_idx, config in self.per_head_config[layer_idx].items():
                    dequant_key = self._dequantize(self._quantized_key_cache[layer_idx][head_idx])
                    dequant_value = self._dequantize(self._quantized_value_cache[layer_idx][head_idx])
                    keys_to_return.append(torch.cat([dequant_key, self.key_cache[layer_idx][head_idx], key_states[head_idx]], dim=-2))
                    values_to_return.append(torch.cat([dequant_value, self.value_cache[layer_idx][head_idx], value_states[head_idx]], dim=-2))
                    if (
                        self.key_cache[layer_idx][head_idx].dim() == 3
                        and self.key_cache[layer_idx][head_idx].shape[-2] + 1 >= self.residual_length
                    ):
                        self._quantized_key_cache[layer_idx][head_idx] = self._quantize(keys_to_return[head_idx].contiguous(), axis=self.axis_key, nbits=config['nbits_key'])
                        self._quantized_value_cache[layer_idx][head_idx] = self._quantize(values_to_return[head_idx].contiguous(), axis=self.axis_value, nbits=config['nbits_value'])
                        self.key_cache[layer_idx][head_idx] = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
                        self.value_cache[layer_idx][head_idx] = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
                    else:
                        self.key_cache[layer_idx][head_idx] = torch.cat([self.key_cache[layer_idx][head_idx], key_states[head_idx]], dim=-2)
                        self.value_cache[layer_idx][head_idx] = torch.cat([self.value_cache[layer_idx][head_idx], value_states[head_idx]], dim=-2)
            keys_to_return, values_to_return = torch.stack(keys_to_return), torch.stack(values_to_return)
            keys_to_return, values_to_return = keys_to_return.transpose(0, 1).contiguous(), values_to_return.transpose(0, 1).contiguous()
        return keys_to_return, values_to_return

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        # since we cannot get the seq_length of each layer directly and rely on `_seen_tokens` which is
        # updated every "layer_idx" == 0, this is a hack to get the actual seq_length for the given layer_idx
        # this part of code otherwise fails when used to verify attn_weight shape in some models
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1

    def _quantize(self, tensor, axis, nbits):
        """Quantizes a key/value using a defined quantization method."""
        raise NotImplementedError("Make sure to implement `_quantize` in a subclass.")

    def _dequantize(self, q_tensor):
        """Dequantizes back the tensor that was quantized by `self._quantize()`"""
        raise NotImplementedError("Make sure to implement `_dequantize` in a subclass.")

class FlexibleQuantoQuantizedCache(FlexibleQuantizedCache):
    """
    Quantized Cache class that uses `quanto` as a backend to perform quantization. Current implementation supports `int2` and `int4` dtypes only.

    Parameters:
        cache_config (`QuantizedCacheConfig`):
            A configuration containing all the arguments to be used by the quantizer, including axis, qtype and group size.

    Example:

        ```python
        >>> # Run pip install quanto first if you don't have it yet
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoQuantizedCache, QuantizedCacheConfig

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> cache_config = QuantizedCacheConfig(nbits=4)
        >>> past_key_values = QuantoQuantizedCache(cache_config=cache_config)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        QuantoQuantizedCache()
        ```
    """

    def __init__(self, cache_config: CacheConfig) -> None:
        super().__init__(cache_config)

        if is_optimum_quanto_available():
            from optimum.quanto import MaxOptimizer, qint2, qint4
        # elif is_quanto_available():
        #     logger.warning_once(
        #         "Importing from quanto will be deprecated in v4.47. Please install optimum-quanto instead `pip install optimum-quanto`"
        #     )
        #     quanto_version = version.parse(importlib.metadata.version("quanto"))
        #     if quanto_version < version.parse("0.2.0"):
        #         raise ImportError(
        #             f"You need quanto package version to be greater or equal than 0.2.0 to use `QuantoQuantizedCache`. Detected version {quanto_version}. "
        #             f"Since quanto will be deprecated, please install optimum-quanto instead with `pip install -U optimum-quanto`"
        #         )
        #     from quanto import MaxOptimizer, qint2, qint4, qint8

        if self.nbits not in [2, 4, 8]:
            raise ValueError(f"`nbits` for `quanto` backend has to be one of [`2`, `4`, `8`] but got {self.nbits}")

        if self.axis_key not in [0, -1]:
            raise ValueError(f"`axis_key` for `quanto` backend has to be one of [`0`, `-1`] but got {self.axis_key}")

        if self.axis_value not in [0, -1]:
            raise ValueError(
                f"`axis_value` for `quanto` backend has to be one of [`0`, `-1`] but got {self.axis_value}"
            )

        # self.qtype = qint4 if self.nbits == 4 else qint2
        self.optimizer = MaxOptimizer()  # hardcode as it's the only one for per-channel quantization
    
    def get_qtype(self, nbits):
        if is_optimum_quanto_available():
            from optimum.quanto import qint2, qint4, qint8
        # elif is_quanto_available():
        #     from quanto import qint2, qint4, qint8
        if nbits == 2:
            return qint2
        elif nbits == 4:
            return qint4
        elif nbits == 8:
            return qint8

    def _quantize(self, tensor, axis, nbits):
        # We have two different API since in optimum-quanto, we don't use AffineQuantizer anymore
        if is_optimum_quanto_available():
            from optimum.quanto import quantize_weight

            qtensor = quantize_weight(tensor, self.get_qtype(nbits), axis, self.q_group_size)
            return qtensor
        # elif is_quanto_available():
        #     logger.warning_once(
        #         "Importing from quanto will be deprecated in v4.47. Please install optimum-quanto instead `pip install optimum-quanto`"
        #     )
        #     from quanto import AffineQuantizer

        #     scale, zeropoint = self.optimizer(tensor, nbits, axis, self.q_group_size)
        #     qtensor = AffineQuantizer.apply(tensor, self.get_qtype(nbits), axis, self.q_group_size, scale, zeropoint)

        return qtensor

    def _dequantize(self, qtensor):
        return qtensor.dequantize()

class FlexibleHQQQuantizedCache(FlexibleQuantizedCache):
    """
    Quantized Cache class that uses `HQQ` as a backend to perform quantization. Current implementation supports `int2`, `int4`, `int8` dtypes.

    Parameters:
        cache_config (`QuantizedCacheConfig`):
            A configuration containing all the arguments to be used by the quantizer, including axis, qtype and group size.

    Example:

        ```python
        >>> # Run pip install hqq first if you don't have it yet
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, HQQQuantizedCache, QuantizedCacheConfig

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> cache_config = QuantizedCacheConfig(nbits=4, axis_key=1, axis_value=1)
        >>> past_key_values = HQQQuantizedCache(cache_config=cache_config)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        HQQQuantizedCache()
        ```
    """

    def __init__(self, cache_config: CacheConfig) -> None:
        super().__init__(cache_config)
        if self.nbits not in [1, 2, 3, 4, 8]:
            raise ValueError(
                f"`nbits` for `HQQ` backend has to be one of [`1`, `2`, `3`, `4`, `8`] but got {self.nbits}"
            )

        if self.axis_key not in [0, 1]:
            raise ValueError(f"`axis_key` for `HQQ` backend has to be one of [`0`, `1`] but got {self.axis_key}")

        if self.axis_value not in [0, 1]:
            raise ValueError(f"`axis_value` for `HQQ` backend has to be one of [`0`, `1`] but got {self.axis_value}")

        self.quantizer = HQQQuantizer

    def _quantize(self, tensor, axis, nbits):
        qtensor, meta = self.quantizer.quantize(
            tensor,
            axis=axis,
            device=self.device,
            compute_dtype=self.compute_dtype,
            nbits=nbits,
            group_size=self.q_group_size,
            channel_wise=True,
        )
        meta["compute_dtype"] = self.compute_dtype
        self.quantizer.cuda(qtensor, meta=meta, device=self.device)  # Move to device and cast to dtype
        return qtensor, meta

    def _dequantize(self, qtensor):
        quant_tensor, meta = qtensor
        tensor = self.quantizer.dequantize(quant_tensor, meta)
        return tensor

class FlexibleVanillaQuantizedCache(FlexibleQuantizedCache):
    def __init__(self, cache_config: CacheConfig) -> None:
        super().__init__(cache_config)

        if self.axis_key not in [0, 1]:
            raise ValueError(f"`axis_key` for `Vanilla` backend has to be one of [`0`, `1`] but got {self.axis_key}")

        if self.axis_value not in [0, 1]:
            raise ValueError(f"`axis_value` for `Vanilla` backend has to be one of [`0`, `1`] but got {self.axis_value}")
        
        self.quantilizers = {}

    def _quantize(self, tensor, axis, nbits):
        if (nbits, axis) not in self.quantilizers:
            self.quantilizers[(nbits, axis)] = VanillaQuantizer(nbits, self.asym, self.compute_dtype)
        quantilizer = self.quantilizers[(nbits, axis)]        
        return quantilizer.quantize(tensor, self.q_group_size, axis)
    
    def _dequantize(self, qtensor):
        return qtensor.dequantize()
    