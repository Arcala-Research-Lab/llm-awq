import torch
from pathlib import Path
import csv

paths = [Path('/pub/oyahia/out/scales'), Path('/pub/oyahia/out/wanda_scales')]

with open('/pub/oyahia/out/llama_7b_layers.txt') as f:
    layers = [line.strip() for line in f]

for path in paths:
    with open(f'out/csv_out/{path.stem}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Layer', 'Mean', 'Median', 'Standard Deviation', 'Min', 'Max', 'Scale List Values-->'])
        scale_list = torch.load(path)
        for layer, scales in zip(layers, scale_list):
            scales = scales.cuda()
            scales_python_list = scales.tolist()
            scales_python_list.insert(0, torch.max(scales).item())
            scales_python_list.insert(0, torch.min(scales).item())
            scales_python_list.insert(0, torch.std(scales).item())
            scales_python_list.insert(0, torch.median(scales).item())
            scales_python_list.insert(0, torch.mean(scales).item())
            scales_python_list.insert(0, layer)
            writer.writerow(scales_python_list)
