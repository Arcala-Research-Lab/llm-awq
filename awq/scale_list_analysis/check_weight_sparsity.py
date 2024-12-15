import sys
import torch
from pathlib import Path
import matplotlib.pyplot as plt

if len(sys.argv) <= 1:
    print('Please specify path to 1+ awq results dumps when running file')
    print('Usage: python export_scale_list.py [path_to_awq_result1] [path_to_awq_result2] ...')
paths = sys.argv[1:]
for path in paths:
    path = Path(path)
    weight_list = torch.load(path)
    n_zero = 0
    n_total = 0
    plt.hist(weight_list[0].flatten().cpu().numpy())
    plt.savefig('out/plot.png', dpi=300)

    for weight in weight_list:
        zeroes = torch.sum(weight == 0)
        total = weight.numel()
        print(zeroes.item(), total, zeroes/total)
        print(weight.shape)
        n_zero += zeroes
        n_total += total
    print(n_zero.item(), n_total, (n_zero/n_total).item())
