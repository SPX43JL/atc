import os
import yaml

def merge_kv(nbits_key, nbits_value):
    if nbits_key != nbits_value:
        return f"K{nbits_key}V{nbits_value}"
    return f"KV{nbits_key}"

kv_to_layer = {
}
def get_precision(filename: str):
    with open(filename, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    ret = 0
    for layer_id, v in data.items():
        ret += v['nbits_key'] + v['nbits_value']
        if merge_kv(v['nbits_key'], v['nbits_value']) not in kv_to_layer:
            kv_to_layer[merge_kv(v['nbits_key'], v['nbits_value'])] = []
        kv_to_layer[merge_kv(v['nbits_key'], v['nbits_value'])].append(layer_id)
    ret /= len(data) * 2
    return ret

calibration_presets = os.listdir('./calibration_presets')

for preset in calibration_presets:
    kv_to_layer = {}
    if 'Mistral' in preset:
        continue
    print(f'Precision for {preset}: {get_precision(os.path.join("./calibration_presets", preset))}')
    print(kv_to_layer)