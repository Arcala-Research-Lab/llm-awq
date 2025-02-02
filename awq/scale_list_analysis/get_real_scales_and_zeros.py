import argparse
import torch
from pathlib import Path
import csv
from tqdm import tqdm
from typing import Union
import matplotlib.pyplot as plt
import os
from pathlib import Path

def _write_csv_and_plt(tensor_dict, path):
    total = []
    with open(f'{path}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Layer', 'Mean', 'Median', 'Standard Deviation', 'Min', 'Max', 'N'])
        for layer, tensor in tqdm(tensor_dict.items(), total=len(tensor_dict), desc='making scales csv'):
            tensor = tensor.cuda()
            stats = [
                layer,
                torch.mean(tensor).item(),
                torch.median(tensor).item(),
                torch.std(tensor).item(),
                torch.min(tensor).item(),
                torch.max(tensor).item(),
                tensor.numel()
            ]
            writer.writerow(stats)
            total.extend(tensor.view(-1).tolist())
        print('getting total histogram')
        plt.figure()
        plt.title(f'histogram (log scale)')
        plt.hist(total, log=True)
        plt.savefig(f'{path}.png')
        plt.close()
        print('getting total stats')
        total = torch.tensor(total)
        stats = [
            'total',
            torch.mean(total).item(),
            torch.median(total).item(),
            torch.std(total).item(),
            torch.min(total).item(),
            torch.max(total).item(),
            total.numel()
        ]
        writer.writerow(stats)

def export_scales_zeros(scales: dict[str, torch.tensor], zeros: dict[str, torch.tensor], scale_path: Union[str, Path], zero_path: Union[str, Path]) -> None:
    _write_csv_and_plt(scales, scale_path)
    _write_csv_and_plt(zeros, zero_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_paths", type=str, nargs='+', help="paths to awq quant dump(s)", required=True)
    parser.add_argument("--output_dir", type=str, help="path to output directory", required=True)
    args = parser.parse_args()

    args.output_dir = Path(args.output_dir)
    for i, path in enumerate(args.input_paths):
        path = Path(path)
        x = torch.load(path)
        scales = {}
        zeros = {}
        for k in x.keys():
            if k.endswith('scales'):
                scales[k] = x[k]
            elif k.endswith('scaled_zeros'):
                zeros[k] = x[k]
    export_scales_zeros(scales, zeros, args.output_dir / f'real_scales{i}', args.output_dir / f'real_scaled_zeros{i}')



