
from __future__ import annotations
import os
import csv
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

def read_tsv(path: str) -> Dict[str, List[float]]:
    cols = {}
    with open(path, 'r', encoding='utf-8') as f:
        rdr = csv.DictReader(f, delimiter='\t')
        for row in rdr:
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return cols

def plot_bpf(tsv_paths: Dict[str, str], out_path: str) -> None:
    plt.figure()
    for label, path in tsv_paths.items():
        data = read_tsv(path)
        B = data['B']
        mean = data['mean']
        lo = data['ci_lo']
        hi = data['ci_hi']
        plt.plot(B, mean, label=label)
        # error bars
        plt.fill_between(B, lo, hi, alpha=0.2)
    plt.xlabel('Budget B')
    plt.ylabel('Performance P(B)')
    plt.title('Budget–Performance Frontier (with 95% CIs)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def plot_mvc(tsv_paths: Dict[str, str], mvc_paths: Dict[str, str], out_path: str) -> None:
    plt.figure()
    for label, path in tsv_paths.items():
        data = read_tsv(path)
        B = data['B']
        mvc_data = read_tsv(mvc_paths[label])
        B2 = mvc_data['B']
        slope = mvc_data['slope']
        lo = mvc_data['ci_lo']
        hi = mvc_data['ci_hi']
        # Align plotting
        plt.plot(B2, slope, label=label)
        plt.fill_between(B2, lo, hi, alpha=0.2)
    plt.xlabel('Budget B')
    plt.ylabel('MVC (ΔP / ΔB)')
    plt.title('Marginal Value-of-Compute (with 95% CIs)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
