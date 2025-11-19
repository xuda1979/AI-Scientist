
from __future__ import annotations
import argparse, os, json, math, csv
import numpy as np

from ..algorithms.baselines import CoT, SelfConsistency, ToT_BFS
from ..algorithms.igd import IGD
from ..algorithms.rs_mctt import RS_MCTT
from ..algorithms.extras.adr import ADR
from ..algorithms.extras.apt import APT
from ..algorithms.extras.csc import CSC
from ..algorithms.extras.psv import PSV
from ..algorithms.extras.dpg import DPG

from ..synthetic.env import make_synthetic_instance
from ..evaluation.compute_accounting import ComputeAccountant
from ..evaluation.stats import ci95_from_mean_sd_n, mvc_from_series

ALGOS = {
    'cot': lambda: CoT(),
    'self_consistency': lambda: SelfConsistency(chains=5),
    'tot_bfs': lambda: ToT_BFS(breadth=3),
    'igd': lambda: IGD(discount=0.95, risk_k=0.5),
    'rs_mctt': lambda: RS_MCTT(eta_start=0.0, eta_end=-3.0, exploration=1.0),
    # extras (toy/abridged):
    'adr': lambda: ADR(verify_every=5, verify_weight=0.2),
    'apt': lambda: APT(population=8, epochs=5, beta_max=3.0),
    'csc': lambda: CSC(chains=7, constraint_strength=0.15),
    'psv': lambda: PSV(max_checks=5, logodds_thresh=0.1),
    'dpg': lambda: DPG(theta=0.6, discount=0.95, risk_k=0.5),
}

def run_once(algo_name: str, budget: int, seed: int, m_threads: int) -> dict:
    rng = np.random.default_rng(seed)
    env = make_synthetic_instance(m_threads, rng)
    algo = ALGOS[algo_name]()
    acc = ComputeAccountant()
    res = algo.run(env, budget, rng, acc)
    out = {
        'algo': algo_name,
        'budget': int(budget),
        'seed': int(seed),
        'utility': float(res.utility),
        **{f'cc_{k}': v for k, v in res.accountant_snapshot.items()},
    }
    return out

def aggregate(rows):
    # returns dict: B -> (mean, sd, n)
    byB = {}
    for r in rows:
        B = int(r['budget'])
        byB.setdefault(B, []).append(r['utility'])
    agg = {}
    for B, arr in sorted(byB.items()):
        n = len(arr)
        m = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        lo, hi = ci95_from_mean_sd_n(m, sd, n)
        agg[B] = (m, sd, n, lo, hi)
    return agg

def write_tsv(path, agg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['B','mean','sd','n','ci_lo','ci_hi'])
        for B,(m,sd,n,lo,hi) in sorted(agg.items()):
            w.writerow([B,m,sd,n,lo,hi])

def write_mvc(path, agg):
    budgets = sorted(agg.keys())
    means = [agg[B][0] for B in budgets]
    vars_ = [(agg[B][1]**2)/max(agg[B][2],1) for B in budgets]
    mvc, ci = mvc_from_series(budgets, means, vars_)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['B','slope','ci_lo','ci_hi'])
        for B, s, (lo,hi) in zip(budgets, mvc, ci):
            w.writerow([B,s,lo,hi])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--budgets', type=int, nargs='+', default=[0,10,20,30,40,50])
    ap.add_argument('--algos', type=str, nargs='+', default=['cot','self_consistency','tot_bfs','igd','rs_mctt'])
    ap.add_argument('--n_eval_seeds', type=int, default=50)
    ap.add_argument('--pilot_seeds', type=int, default=10)  # reserved, not used in eval
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--save_per_run', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    per_run_rows = {a: [] for a in args.algos}
    for algo in args.algos:
        for B in args.budgets:
            for s in range(args.pilot_seeds, args.pilot_seeds + args.n_eval_seeds):
                row = run_once(algo, B, seed=s, m_threads=args.threads)
                per_run_rows[algo].append(row)

    # aggregate + save
    for algo, rows in per_run_rows.items():
        agg = aggregate(rows)
        tsv_path = os.path.join(args.out_dir, f'{algo}_summary.tsv')
        mvc_path = os.path.join(args.out_dir, f'{algo}_mvc.tsv')
        write_tsv(tsv_path, agg)
        write_mvc(mvc_path, agg)
        if args.save_per_run:
            pr = os.path.join(args.out_dir, f'{algo}_per_run.csv')
            import csv
            with open(pr, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
                w.writeheader()
                for r in rows:
                    w.writerow(r)

if __name__ == '__main__':
    main()
