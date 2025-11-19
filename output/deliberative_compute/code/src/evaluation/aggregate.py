
from __future__ import annotations
import argparse, os, csv, glob, json
from .stats import hedges_g, welch_t, bh_fdr
from .plotting import plot_bpf, plot_mvc

def read_tsv(path: str):
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        rdr = csv.DictReader(f, delimiter='\t')
        for row in rdr:
            B = int(float(row['B']))
            out.append({
                'B': B,
                'mean': float(row['mean']),
                'sd': float(row['sd']),
                'n': int(float(row['n'])),
                'ci_lo': float(row['ci_lo']),
                'ci_hi': float(row['ci_hi']),
            })
    return out

def write_effects(out_path: str, ref_path: str, tgt_path: str) -> None:
    ref = read_tsv(ref_path)
    tgt = read_tsv(tgt_path)
    byB = {r['B']: r for r in ref}
    rows = []
    pvals = []
    for r in tgt:
        B = r['B']
        if B == 0:  # skip degenerate
            continue
        if B not in byB:
            continue
        R = byB[B]
        g = hedges_g(r['mean'], r['sd'], r['n'], R['mean'], R['sd'], R['n'])
        t, p = welch_t(r['mean'], r['sd'], r['n'], R['mean'], R['sd'], R['n'])
        pvals.append(p)
        rows.append({'B': B, 'g': g, 't': t, 'p': p})
    # BH-FDR
    rejs = bh_fdr(pvals, alpha=0.05) if pvals else []
    for i,rej in enumerate(rejs):
        rows[i]['bh_fdr_reject_0.05'] = bool(rej)
    # write
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', type=str, required=True, help='Where *_summary.tsv files live')
    ap.add_argument('--out_dir', type=str, required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Collect summaries
    summaries = sorted(glob.glob(os.path.join(args.in_dir, '*_summary.tsv')))
    mvc = sorted(glob.glob(os.path.join(args.in_dir, '*_mvc.tsv')))
    label_from = lambda p: os.path.basename(p).replace('_summary.tsv','')

    # Plots
    tsv_paths = {label_from(p): p for p in summaries}
    mvc_paths = {label_from(p): p.replace('_summary.tsv','_mvc.tsv') for p in summaries}
    plot_bpf(tsv_paths, os.path.join(args.out_dir, 'bpf.png'))
    plot_mvc(tsv_paths, mvc_paths, os.path.join(args.out_dir, 'mvc.png'))

    # Effects and stats: compare IGD vs CoT by default if available
    if 'igd' in tsv_paths and 'cot' in tsv_paths:
        write_effects(os.path.join(args.out_dir, 'effects_igd_vs_cot.tsv'), tsv_paths['cot'], tsv_paths['igd'])

if __name__ == '__main__':
    main()
