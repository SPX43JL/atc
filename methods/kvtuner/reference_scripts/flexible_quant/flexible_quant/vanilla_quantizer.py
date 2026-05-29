
import torch


def quant_sym(x: torch.tensor, scaling: torch.tensor, nbits: int):
    q_max, q_min = 2 ** (nbits - 1) - 1, -2 ** (nbits - 1)
    return torch.round(x / scaling.unsqueeze(1)).clip(q_min, q_max).to(torch.int8)

def dequant_sym(x: torch.tensor, scaling: torch.tensor, target_dtype: torch.dtype):
    return x * scaling.unsqueeze(1).to(target_dtype)

def quant_asym(x: torch.tensor, scaling: torch.tensor, zeros: torch.tensor, nbits: int):
    q_max, q_min = 2 ** (nbits - 1) - 1, -2 ** (nbits - 1)
    return (torch.round(x / scaling.unsqueeze(1) - zeros.unsqueeze(1))).clip(q_min, q_max).to(torch.int8)
    
def dequant_asym(x: torch.tensor, scaling: torch.tensor, zeros: torch.tensor, target_dtype: torch.dtype):
    return (x + zeros.unsqueeze(1)) * scaling.unsqueeze(1).to(target_dtype)


class VanillaQuantizeMeta:
    def __init__(self, nbits, asym, compute_dtype):
        self.nbits = nbits
        # self.group_size = group_size
        # self.axis = axis # 1 for per-channel, 0 for per-token
        self.asym = asym
        self.compute_dtype = compute_dtype

    
class VanillaQuantizedTensor:
    def __init__(self, tensor, scaling, zeros, original_shape, axis, meta: VanillaQuantizeMeta):
        self.tensor = tensor
        self.scaling = scaling
        self.zeros = zeros
        self.original_shape = original_shape
        self.axis = axis
        self.meta = meta

    def dequantize(self):
        if self.meta.asym:
            dequant = dequant_asym(self.tensor, self.scaling, self.zeros, self.meta.compute_dtype)
        else:
            dequant = dequant_sym(self.tensor, self.scaling, self.meta.compute_dtype)
        dequant = dequant.view(self.original_shape)
        if self.axis == 1:
            max_dim = len(self.original_shape) - 1
            dequant = dequant.transpose(max_dim - 1, max_dim)
        return dequant

class VanillaQuantizer:
    def __init__(self, nbits, asym, compute_dtype):
        self.meta = VanillaQuantizeMeta(nbits, asym, compute_dtype)
    
    def quantize(self, tensor, q_group_size, axis):
        if axis == 1:
            max_dim = len(tensor.shape) - 1
            tensor = tensor.transpose(max_dim - 1, max_dim)
        if q_group_size == -1:
            assert axis == 0 # must be per-token
            q_group_size = tensor.shape[-1] # take the last dimension
        rs = tensor.reshape(-1, q_group_size)
        
        q_max, q_min = 2 ** (self.meta.nbits - 1) - 1, -2 ** (self.meta.nbits - 1)
        
        if self.meta.asym:
            _max, _min = rs.max(dim=1).values, rs.min(dim=1).values
            scale = (_max - _min).clamp(min=1e-5).div(q_max - q_min)
            zeros = (_min / scale).round() - q_min
            quant = quant_asym(rs, scale, zeros, self.meta.nbits)
        else:
            scale = rs.abs().max(dim=1).values.clamp(min=1e-5).div(q_max)
            zeros = None
            quant = quant_sym(rs, scale, self.meta.nbits)
        
        return VanillaQuantizedTensor(quant, scale, zeros, tensor.shape, axis, self.meta)
