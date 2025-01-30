import torch

def round_nearest_power_of_2(awq_results: dict) -> None:
    """sets scale values to pwoer of 2"""
    awq_scales: list[torch.tensor] = awq_results['scale']
    # make power of 2 list
    biggest_max = 0
    for layer_in, layer_out, scales in awq_scales:
        biggest_max = max(torch.max(scales), biggest_max)
    power_of_2 = [1]
    while power_of_2[-1] < biggest_max:
        power_of_2.append(power_of_2[-1]*2)
    power_of_2 = torch.tensor(power_of_2)
    # round to nearest power of 2
    for i, (layer_in, layer_out, scales) in enumerate(awq_scales):
        difs = torch.abs(scales.unsqueeze(-1) - power_of_2)
        nearest_indices = torch.argmin(difs, dim=-1)
        new_scales = power_of_2[nearest_indices]
        awq_scales[i] = (layer_in, layer_out, new_scales)
    awq_results['scale'] = awq_scales

def set_all_ones(awq_results: dict) -> None:
    """sets scale values to all ones"""
    awq_scales: list[torch.tensor] = awq_results['scale']
    for i, (layer_in, layer_out, scales) in enumerate(awq_scales):
        new_scales = torch.ones_like(scales)
        awq_scales[i] = (layer_in, layer_out, new_scales)
    awq_results['scale'] = awq_scales