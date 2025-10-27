import csv
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Matplotlib is optional at runtime; plotting proceeds only if available.
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

# Global deterministic seed for any optional randomness (e.g., jitter)
GLOBAL_SEED = 12345
random.seed(GLOBAL_SEED)

@dataclass
class SummaryRow:
    B: int
    mean: float
    sd: float
    n: int
    ci_lo: float
    ci_hi: float

def read_results_tsv(path: str) -> List[SummaryRow]:
    rows: List[SummaryRow] = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f, delimiter=' ', skipinitialspace=True)
        for r in reader:
            if not r or 'B' not in r:
                continue
            rows.append(SummaryRow(
                B=int(r['B']),
                mean=float(r['mean']),
                sd=float(r['sd']),
                n=int(r['n']),
                ci_lo=float(r['ci_lo']),
                ci_hi=float(r['ci_hi'])
            ))
    return rows

def trapezoid_area(budgets: List[int], means: List[float]) -> float:
    area = 0.0
    for i in range(len(budgets) - 1):
        w = budgets[i+1] - budgets[i]
        area += 0.5 * (means[i] + means[i+1]) * w
    return area

def mvc_slopes(budgets: List[int], means: List[float]) -> Dict[int, float]:
    # Finite differences relative to previous budget
    slopes: Dict[int, float] = {}
    for i in range(1, len(budgets)):
        dB = budgets[i] - budgets[i-1]
        slopes[budgets[i]] = (means[i] - means[i-1]) / dB
    return slopes

def mvc_ci_delta_method(rows: List[SummaryRow]) -> Dict[int, Tuple[float, float]]:
    """
    Approximate 95% CI for MVC slopes using a delta method:
    slope = (m_i - m_{i-1}) / dB, var(slope) ≈ (var(m_i)+var(m_{i-1})) / dB^2
    where var(m) ≈ sd^2 / n assuming independence across runs at adjacent budgets.
    Returns dict mapping budget -> (ci_lo, ci_hi) for slope at that budget.
    """
    byB = {r.B: r for r in rows}
    budgets = sorted(byB.keys())
    cis: Dict[int, Tuple[float, float]] = {}
    z = 1.96
    for i in range(1, len(budgets)):
        b_prev, b_cur = budgets[i-1], budgets[i]
        dB = b_cur - b_prev
        r_prev, r_cur = byB[b_prev], byB[b_cur]
        slope = (r_cur.mean - r_prev.mean) / dB
        var_prev = (r_prev.sd ** 2) / max(r_prev.n, 1)
        var_cur = (r_cur.sd ** 2) / max(r_cur.n, 1)
        var_slope = (var_prev + var_cur) / (dB ** 2)
        se = math.sqrt(max(var_slope, 1e-12))
        cis[b_cur] = (slope - z * se, slope + z * se)
    return cis

def hedges_g_from_summaries(m1: float, s1: float, n1: int, m2: float, s2: float, n2: int) -> float:
    # Pooled SD (unbiased) and small-sample correction
    s1sq = s1**2
    s2sq = s2**2
    sp2 = ((n1-1)*s1sq + (n2-1)*s2sq) / max((n1 + n2 - 2), 1)
    sp = math.sqrt(max(sp2, 1e-12))
    d = (m1 - m2) / sp if sp > 0 else 0.0
    # Hedges' g correction J
    J = 1.0 - (3.0 / (4.0*(n1 + n2) - 9.0)) if (n1 + n2) > 2 else 1.0
    return J * d

def welch_t_p_from_summaries(m1: float, s1: float, n1: int, m2: float, s2: float, n2: int) -> Tuple[float, float]:
    # Two-sided Welch's t-test using summary stats (normal approximation for p)
    se = math.sqrt((s1**2)/max(n1, 1) + (s2**2)/max(n2, 1))
    if se == 0:
        return float('inf'), 0.0
    t = (m1 - m2) / se
    # Degrees of freedom (Welch-Satterthwaite) -- not used in normal approx for p here
    v1 = (s1**2)/max(n1, 1)
    v2 = (s2**2)/max(n2, 1)
    _ = (v1 + v2)**2 / ((v1**2)/max(n1-1, 1) + (v2**2)/max(n2-1, 1))
    # Two-sided p-value via normal approximation (adequate for high n)
    p = 2.0 * (1.0 - normal_cdf(abs(t)))
    return t, p

def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def bh_fdr(pvals: List[float]) -> List[float]:
    m = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adj = [0.0] * m
    prev = 1.0
    for rank, (i, p) in enumerate(indexed, start=1):
        val = p * m / rank
        prev = min(prev, val)
        adj[i] = min(1.0, prev)
    return adj

def plot_bpf(curves: Dict[str, List[SummaryRow]], out_path: str) -> None:
    if plt is None:
        print("Matplotlib not available; skipping plot:", out_path)
        return
    plt.figure(figsize=(7.6, 3.6))
    for label, rows in curves.items():
        xs = [r.B for r in rows]
        ys = [r.mean for r in rows]
        lo = [r.ci_lo for r in rows]
        hi = [r.ci_hi for r in rows]
        plt.plot(xs, ys, marker='o', label=label)
        # CI ribbon
        plt.fill_between(xs, lo, hi, alpha=0.15)
    plt.xlabel('Budget B')
    plt.ylabel('Performance P(B)')
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print("Saved", out_path)

def plot_mvc(curves: Dict[str, List[SummaryRow]], out_path: str) -> None:
    if plt is None:
        print("Matplotlib not available; skipping plot:", out_path)
        return
    plt.figure(figsize=(7.6, 3.6))
    for label, rows in curves.items():
        xs = [r.B for r in rows]
        ys = [r.mean for r in rows]
        slopes = mvc_slopes(xs, ys)
        xs2 = sorted(slopes.keys())
        ys2 = [slopes[x] for x in xs2]
        cis = mvc_ci_delta_method(rows)
        lo = [cis[x][0] for x in xs2]
        hi = [cis[x][1] for x in xs2]
        plt.plot(xs2, ys2, marker='o', label=label)
        plt.fill_between(xs2, lo, hi, alpha=0.12)
    plt.xlabel('Budget B')
    plt.ylabel('MVC ΔP/ΔB')
    plt.ylim(0, 0.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print("Saved", out_path)

def read_effects_tsv(path: str) -> List[Tuple[int, float, float, float]]:
    rows: List[Tuple[int, float, float, float]] = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f, delimiter=' ', skipinitialspace=True)
        for r in reader:
            if not r or 'Budget' not in r:
                continue
            B = int(r['Budget'])
            g = float(r['g_IGD_vs_CoT'])
            p = float(r['p_uncorr'])
            p_bh = float(r['p_BH'])
            rows.append((B, g, p, p_bh))
    return rows

def print_ablation_table(path: str) -> None:
    """
    Pretty-print ablation TSVs. Supports headers:
    - Variant mean sd ci_lo ci_hi
    - Variant mean30 sd30 ci_lo30 ci_hi30
    """
    try:
        with open(path, newline='') as f:
            reader = csv.DictReader(f, delimiter=' ', skipinitialspace=True)
            for r in reader:
                if not r:
                    continue
                variant = (r.get('Variant') or r.get('variant') or r.get('name') or r.get('Model') or r.get('model') or '').strip()
                if 'mean30' in r:
                    mean = r.get('mean30', '')
                    sd = r.get('sd30', '')
                    ci_lo = r.get('ci_lo30', '')
                    ci_hi = r.get('ci_hi30', '')
                else:
                    mean = r.get('mean', '')
                    sd = r.get('sd', '')
                    ci_lo = r.get('ci_lo', '')
                    ci_hi = r.get('ci_hi', '')
                print(f"{variant:28s} mean={mean}  sd={sd}  ci_lo={ci_lo}  ci_hi={ci_hi}")
    except FileNotFoundError:
        print("Ablation TSV not found:", path)

def write_mvc_tsv(curves: Dict[str, List[SummaryRow]]) -> None:
    """
    Write MVC summaries with CI to separate TSV files per method: mvc_{label}.tsv
    Columns: B slope ci_lo ci_hi
    """
    for label, rows in curves.items():
        slopes = mvc_slopes([r.B for r in rows], [r.mean for r in rows])
        cis = mvc_ci_delta_method(rows)
        out = f"mvc_{label.replace(' ', '').replace('-', '').replace('/', '')}.tsv"
        with open(out, 'w', newline='') as f:
            w = csv.writer(f, delimiter=' ')
            w.writerow(['B', 'slope', 'ci_lo', 'ci_hi'])
            for B in sorted(slopes.keys()):
                lo, hi = cis[B]
                w.writerow([B, f"{slopes[B]:.6f}", f"{lo:.6f}", f"{hi:.6f}"])
        print("Wrote", out)

def main():
    # Deterministic behavior: no randomness in aggregation
    files = {
        'CoT': 'results_cot.tsv',
        'ToT-BFS': 'results_tot.tsv',
        'Self-Consistency': 'results_sc.tsv',
        'IGD': 'results_igd.tsv',
        'RS-MCTT': 'results_rsmctt.tsv',
    }
    curves = {name: read_results_tsv(path) for name, path in files.items()}

    print("== Frontier Area (FA) and MVC ==")
    for name, rows in curves.items():
        budgets = [r.B for r in rows]
        means = [r.mean for r in rows]
        fa = trapezoid_area(budgets, means)
        slopes = mvc_slopes(budgets, means)
        print(f"{name:>16}  FA={fa:.2f}   MVC@10={slopes.get(10, float('nan')):.3f}  "
              f"MVC@20={slopes.get(20, float('nan')):.3f}  MVC@30={slopes.get(30, float('nan')):.3f}")

    # Effect size and Welch test from summaries: IGD vs CoT (exclude B=0)
    igd = {r.B: r for r in curves['IGD']}
    cot = {r.B: r for r in curves['CoT']}
    budgets = sorted(igd.keys())
    budgets_eff = [B for B in budgets if B > 0]
    pvals = []
    stats_rows = []
    for B in budgets_eff:
        r1, r2 = igd[B], cot[B]
        g = hedges_g_from_summaries(r1.mean, r1.sd, r1.n, r2.mean, r2.sd, r2.n)
        t, p = welch_t_p_from_summaries(r1.mean, r1.sd, r1.n, r2.mean, r2.sd, r2.n)
        stats_rows.append((B, g, p))
        pvals.append(p)
    p_bh = bh_fdr(pvals)
    print("\n== IGD vs CoT effect sizes and p-values (Welch, BH FDR) ==")
    for (B, g, p), p_adj in zip(stats_rows, p_bh):
        print(f"B={B:>2}  Hedges g={g:.2f}  p={p:.2e}  p_BH={p_adj:.2e}")

    # Ablations table (read and echo)
    print("\n== Ablations at B=30 ==")
    print_ablation_table('ablation_igd.tsv')

    # Plots with uncertainty ribbons
    plot_bpf(curves, out_path='fig_bpf.pdf')
    plot_mvc(curves, out_path='fig_mvc.pdf')

    # Also produce MVC TSVs for pgfplots if desired
    write_mvc_tsv(curves)

    # Optional: plot effect size curve if matplotlib available
    if plt is not None:
        rows = read_effects_tsv('results_summary.tsv')
        xs = [r[0] for r in rows]
        gs = [r[1] for r in rows]
        plt.figure(figsize=(5.2, 3.6))
        plt.plot(xs, gs, marker='o', color='purple')
        plt.xlabel('Budget B')
        plt.ylabel("Hedges' g (IGD vs CoT)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('fig_effect.pdf')
        print("Saved fig_effect.pdf")

if __name__ == '__main__':
    main()