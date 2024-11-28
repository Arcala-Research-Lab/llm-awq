import sys
import csv
import torch
from pathlib import Path

if len(sys.argv) <= 1:
    print('Please specify path to 1+ awq results dumps when running file')
    print('Usage: python export_scale_list.py [path_to_awq_result1] [path_to_awq_result2] ...')
paths = sys.argv[1:]
for path in paths:
    path = Path(path)
    with open(f'{path.parent.stem + "".join(path.parent.suffixes)}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['In Layer(s)', 'Out Layer(s)', 'Mean', 'Median', 'Standard Deviation', 'Min', 'Max', 'Scale List Values-->'])
        scale_list = torch.load(path)['scale']
        for in_layers, out_layers, scales in scale_list:
            scales = scales.cuda()
            scales_python_list = scales.tolist()
            scales_python_list.insert(0, torch.max(scales).item())
            scales_python_list.insert(0, torch.min(scales).item())
            scales_python_list.insert(0, torch.std(scales).item())
            scales_python_list.insert(0, torch.median(scales).item())
            scales_python_list.insert(0, torch.mean(scales).item())
            scales_python_list.insert(0, out_layers)
            scales_python_list.insert(0, in_layers)
            writer.writerow(scales_python_list)

