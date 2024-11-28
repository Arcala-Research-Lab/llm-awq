import sys
import torch
from pathlib import Path

if len(sys.argv) <= 1:
    print('Please specify path to 1+ awq results dumps when running file')
    print('Usage: python export_scale_list.py [path_to_awq_result1] [path_to_awq_result2] ...')
paths = sys.argv[1:]
for path in paths:
    path = Path(path)
    scale_list = torch.load(path)['scale']
    n_nonone = 0
    n_total = 0
    for _, _, scales in scale_list:
        n_nonone += sum(scales != 1)
        n_total += len(scales)
    print(f'{path} percent: {n_nonone / n_total}')

