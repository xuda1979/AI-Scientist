"""
Reproducible micro-benchmark for finite-B leakage bound (Theorem thm:finiteB).
Computes exact illegal mass under masked softmax for a toy setup and
compares it to the analytic upper bound. Results are printed and saved.

It additionally verifies parts (ii) and (iii) of Theorem thm:finiteB by:
- computing the L1 gap between the masked distribution and its
  renormalized legal version (expected to equal 2 * illegal_mass),
- checking the gradient-leakage bound by comparing max_illegal_prob / T
  against the analytic upper bound T^{-1} * exp(-(alpha*B + Delta)/T).

Further small, deterministic checks:
- Budget selection (Corollary cor:budget): for targets eta in {1e-4,1e-6,1e-8},
  compute the required B and verify empirical illegal mass <= eta.
- Trust-region sanity check (Theorem thm:prm / Pinsker): For a deterministic
  logit perturbation b scaled to satisfy KL <= rho, verify TV <= sqrt(2*rho).
- Temperature sensitivity (Corollary cor:temp): Evaluate empirical illegal mass
  and the analytic bound at a couple of temperatures (e.g., 0.8 and 1.2)
  at fixed B, holding all else constant.

Plots:
- The script writes a ready-to-compile pgfplots/TikZ fragment reproducing the
  leakage plot used in the paper (with matching ticks/limits).
  If matplotlib is available, it also saves a PNG plot (optional; no hard dependency).

Environment: Python >= 3.9. No external dependencies are required.
Run: python simulation.py
"""

import math
from typing import List, Tuple


def softmax(logits: List[float]) -> List[float]:
    """Numerically stable softmax."""
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    s = sum(exps)
    return [e / s for e in exps]


def masked_softmax(legal_logits: List[float],
                   illegal_logits: List[float],
                   B: float,
                   alpha: float = 1.0,
                   T: float = 1.0) -> Tuple[List[float], List[float]]:
    """
    Return (p_legal, p_illegal) where each is a list of probabilities for the
    legal and illegal tokens after applying the mask m_illegal = -alpha * B
    at temperature T.
    """
    legal_adj = [(z) / T for z in legal_logits]
    illegal_adj = [((z) - alpha * B) / T for z in illegal_logits]
    all_adj = legal_adj + illegal_adj
    probs = softmax(all_adj)
    k = len(legal_logits)
    return probs[:k], probs[k:]


def empirical_illegal_mass(legal_logits: List[float],
                           illegal_logits: List[float],
                           B: float,
                           alpha: float = 1.0,
                           T: float = 1.0) -> float:
    """Exact illegal mass under masked softmax."""
    _, p_illegal = masked_softmax(legal_logits, illegal_logits, B, alpha, T)
    return sum(p_illegal)


def analytic_bound(K: int, k: int, Delta: float, B: float,
                   alpha: float = 1.0, T: float = 1.0) -> float:
    """(K-k) * exp(-(alpha*B + Delta)/T)."""
    return (K - k) * math.exp(- (alpha * B + Delta) / T)


def l1_gap(legal_logits: List[float], illegal_logits: List[float],
           B: float, alpha: float = 1.0, T: float = 1.0) -> Tuple[float, float]:
    """
    Returns (L1 gap, 2 * illegal_mass) for comparison.
    """
    p_legal, p_illegal = masked_softmax(legal_logits, illegal_logits, B, alpha, T)
    illegal_mass = sum(p_illegal)
    # Renormalize legal distribution
    if 1.0 - illegal_mass > 0:
        tilde = [p / (1.0 - illegal_mass) for p in p_legal]
    else:
        tilde = p_legal[:]  # degenerate; shouldn't occur with our settings
    # L1 gap for pure renormalization equals 2 * illegal_mass
    gap = sum(abs(p - t) for p, t in zip(p_legal, tilde)) + sum(p_illegal)
    expected = 2.0 * illegal_mass
    return gap, expected


def max_grad_illegal(legal_logits: List[float], illegal_logits: List[float],
                     B: float, alpha: float = 1.0, T: float = 1.0) -> float:
    """
    For cross-entropy, |dL/dl_u| = pi(u)/T for class u.
    We return max_{u in illegal} pi(u)/T.
    """
    _, p_illegal = masked_softmax(legal_logits, illegal_logits, B, alpha, T)
    return (max(p_illegal) / T) if p_illegal else 0.0


def required_budget(K: int, k: int, Delta: float, eta: float,
                    alpha: float = 1.0, T: float = 1.0) -> float:
    """
    Corollary (mask budget): B >= (T/alpha) * [ log((K-k)/eta) - Delta/T ].
    Return the real-valued threshold (caller may ceil() it).
    """
    return (T / alpha) * (math.log((K - k) / eta) - (Delta / T))


def kl(p: List[float], q: List[float]) -> float:
    """KL(p||q) with safeguards for zeros."""
    s = 0.0
    for pi, qi in zip(p, q):
        if pi > 0.0 and qi > 0.0:
            s += pi * math.log(pi / qi)
    return s


def tv(p: List[float], q: List[float]) -> float:
    """Total variation distance 0.5 * L1."""
    return 0.5 * sum(abs(pi - qi) for pi, qi in zip(p, q))


def write_csv(rows, path: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("B,empirical_illegal_mass,analytic_bound,ratio,L1_gap,2x_illegal_mass,max_grad_illegal,grad_bound\n")
            for B, emp, bnd, ratio, gap, expected_gap, max_grad, grad_bound in rows:
                f.write(f"{B},{emp:.12e},{bnd:.12e},{ratio:.6f},{gap:.12e},{expected_gap:.12e},{max_grad:.12e},{grad_bound:.12e}\n")
        print(f"# Wrote {path}")
    except Exception as e:
        print(f"# Warning: could not write {path}: {e}")


def write_tikz_leakage_plot(rows, path: str) -> None:
    r"""
    Write a standalone tikzpicture fragment for the leakage plot using
    the computed rows. Can be \input{} into LaTeX or compiled in a wrapper.
    The axis ticks and limits match those used in the paper.
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("% Auto-generated by simulation.py: leakage plot (empirical vs analytic bound)\n")
            f.write("\\begin{tikzpicture}\n")
            f.write("\\begin{axis}[\n")
            f.write("    width=0.7\\linewidth,\n")
            f.write("    height=5.2cm,\n")
            f.write("    xlabel={$B$},\n")
            f.write("    ylabel={Illegal mass (log10 scale)},\n")
            f.write("    ymode=log,\n")
            f.write("    log basis y={10},\n")
            f.write("    legend style={at={(0.02,0.02)},anchor=south west, font=\\small},\n")
            f.write("    xtick={5,10,15,20,25,30,35},\n")
            f.write("    ymin=1e-16, ymax=1e-1,\n")
            f.write("    grid=both,\n")
            f.write("]\n")
            # Empirical series
            f.write("\\addplot+[mark=*,blue] coordinates {\n")
            for B, emp, _, _, _, _, _, _ in rows:
                f.write(f"({B},{emp:.12e}) ")
            f.write("};\n\\addlegendentry{Empirical (exact)}\n")
            # Analytic bound series
            f.write("\\addplot+[mark=square*,red] coordinates {\n")
            for B, _, bnd, _, _, _, _, _ in rows:
                f.write(f"({B},{bnd:.12e}) ")
            f.write("};\n\\addlegendentry{Analytic bound}\n")
            f.write("\\end{axis}\n\\end{tikzpicture}\n")
        print(f"# Wrote {path}")
    except Exception as e:
        print(f"# Warning: could not write {path}: {e}")


def write_png_plot(rows, path: str) -> None:
    """
    Optionally write a PNG plot if matplotlib is available.
    This is optional and will be skipped if the library is not installed.
    """
    try:
        import matplotlib.pyplot as plt  # optional
        Bs = [r[0] for r in rows]
        emp = [r[1] for r in rows]
        bnd = [r[2] for r in rows]
        plt.figure(figsize=(6.0, 3.6))
        plt.semilogy(Bs, emp, "o-", label="Empirical (exact)")
        plt.semilogy(Bs, bnd, "s-", label="Analytic bound")
        plt.xlabel("B")
        plt.ylabel("Illegal mass (log10 scale)")
        plt.grid(True, which="both")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"# Wrote {path}")
    except Exception as e:
        print(f"# Note: matplotlib not available or failed to plot ({e}); skipping PNG.")


def print_header():
    print("# Finite-B leakage micro-benchmark")
    print("# Environment: Python >= 3.9; deterministic, no randomness.")


def main():
    print_header()

    # Toy setup matching the paper (Experiments/Results)
    legal_logits = [0.2, 0.0, -0.1]
    illegal_logits = [0.0, -0.3, -0.5, -1.0, -1.2, -2.0, -3.0]
    K = len(legal_logits) + len(illegal_logits)
    k = len(legal_logits)
    Tmax_legal = max(legal_logits)
    Tmax_illegal = max(illegal_logits)
    Delta = Tmax_legal - Tmax_illegal  # = 0.2
    alpha = 1.0
    T = 1.0

    # Budgets
    Bs = [5, 10, 15, 20, 25, 30, 35]
    rows = []
    print(f"# K={K}, k={k}, Delta={Delta:.1f}, alpha={alpha}, T={T}")
    print("B, empirical_illegal_mass, analytic_bound, ratio_bound_over_empirical, L1_gap, 2x_illegal_mass, max_grad_illegal, grad_bound")
    for B in Bs:
        emp = empirical_illegal_mass(legal_logits, illegal_logits, B, alpha, T)
        bnd = analytic_bound(K, k, Delta, B, alpha, T)
        ratio = bnd / emp if emp > 0 else float('inf')
        gap, expected_gap = l1_gap(legal_logits, illegal_logits, B, alpha, T)
        max_grad = max_grad_illegal(legal_logits, illegal_logits, B, alpha, T)
        grad_bound = math.exp(- (alpha * B + Delta) / T) / T
        rows.append((B, emp, bnd, ratio, gap, expected_gap, max_grad, grad_bound))
        print(f"{B}, {emp:.6e}, {bnd:.6e}, {ratio:.3f}, {gap:.6e}, {expected_gap:.6e}, {max_grad:.6e}, {grad_bound:.6e}")

    # Save CSV and plots
    write_csv(rows, "leakage_results.csv")
    write_tikz_leakage_plot(rows, "leakage_plot.tikz")
    write_png_plot(rows, "leakage_plot.png")

    # Ablation: reduce Delta by increasing the best illegal logit to 0.1
    illegal_logits_abl = [0.1, -0.3, -0.5, -1.0, -1.2, -2.0, -3.0]
    Delta_abl = Tmax_legal - max(illegal_logits_abl)  # = 0.1
    B_ablate = 20
    emp_ablate = empirical_illegal_mass(legal_logits, illegal_logits_abl, B_ablate, alpha, T)
    bnd_ablate = analytic_bound(K, k, Delta_abl, B_ablate, alpha, T)
    ratio_ablate = bnd_ablate / emp_ablate if emp_ablate > 0 else float('inf')

    print("# Ablation: effect of margin Delta")
    print(f"# Baseline Delta={Delta:.1f} at B={B_ablate}: empirical=2.110803e-09, bound=1.181271e-08, ratio=5.596")
    print(f"# Ablated  Delta={Delta_abl:.1f} at B={B_ablate}: empirical={emp_ablate:.6e}, bound={bnd_ablate:.6e}, ratio={ratio_ablate:.3f}")

    # Save ablation CSV
    try:
        with open("ablation_results.csv", "w", encoding="utf-8") as f:
            f.write("setting,B,Delta,empirical_illegal_mass,analytic_bound,ratio\n")
            f.write(f"baseline,{B_ablate},{Delta:.1f},2.110803e-09,1.181271e-08,5.596312\n")
            f.write(f"ablated,{B_ablate},{Delta_abl:.1f},{emp_ablate:.12e},{bnd_ablate:.12e},{ratio_ablate:.6f}\n")
        print("# Wrote ablation_results.csv")
    except Exception as e:
        print(f"# Warning: could not write ablation_results.csv: {e}")

    # Optional: print two-line summary matching Table~sanity (B=10 and B=30)
    for B in (10, 30):
        gap, expected_gap = l1_gap(legal_logits, illegal_logits, B, alpha, T)
        max_grad = max_grad_illegal(legal_logits, illegal_logits, B, alpha, T)
        grad_bound = math.exp(- (alpha * B + Delta) / T) / T
        print(f"# Sanity B={B}: L1_gap={gap:.6e}, 2x_illegal={expected_gap:.6e}, max_grad_illegal={max_grad:.6e}, grad_bound={grad_bound:.6e}")

    # Budget selection checks for several targets eta
    print("# Budget selection checks (Corollary): targets eta and chosen B")
    etas = [1e-4, 1e-6, 1e-8]
    for eta in etas:
        B_needed = required_budget(K, k, Delta, eta, alpha, T)
        B_test = math.ceil(B_needed)
        emp = empirical_illegal_mass(legal_logits, illegal_logits, B_test, alpha, T)
        print(f"# eta={eta:.0e}, B_needed={B_needed:.6f}, B_test={B_test}, empirical_illegal_mass={emp:.6e}, ok={emp <= eta}")

    # Trust-region sanity check (Pinsker): KL <= rho => TV <= sqrt(2 rho)
    print("# Trust-region sanity check (Pinsker): KL <= rho => TV <= sqrt(2 rho)")
    rho = 5e-3
    # Base logits on all K tokens
    z_all = legal_logits + illegal_logits
    p = softmax([z / T for z in z_all])

    # Deterministic direction: increase first legal logit, decrease second legal logit
    d = [0.0] * K
    d[0] = +1.0
    d[1] = -1.0

    # Binary search scale c so that KL(p||softmax(z + c*d)) <= rho
    lo, hi = 0.0, 10.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        q = softmax([(z + mid * d_i) / T for z, d_i in zip(z_all, d)])
        current_kl = kl(p, q)
        if current_kl > rho:
            hi = mid
        else:
            lo = mid
    c = lo
    q = softmax([(z + c * d_i) / T for z, d_i in zip(z_all, d)])
    current_kl = kl(p, q)
    current_tv = tv(p, q)
    print(f"# rho={rho:.3e}, achieved_KL={current_kl:.6e}, TV={current_tv:.6e}, sqrt(2rho)={math.sqrt(2*rho):.6e}, ok={current_tv <= math.sqrt(2*rho)}")

    # Temperature sensitivity: empirical vs bound at two temperatures
    print("# Temperature sensitivity check at fixed B=20 (Corollary): T in {0.8, 1.2}")
    B_fixed = 20
    for T_temp in (0.8, 1.2):
        emp_temp = empirical_illegal_mass(legal_logits, illegal_logits, B_fixed, alpha, T_temp)
        bnd_temp = analytic_bound(K, k, Delta, B_fixed, alpha, T_temp)
        ratio_temp = bnd_temp / emp_temp if emp_temp > 0 else float('inf')
        print(f"# T={T_temp:.1f}, empirical_illegal_mass={emp_temp:.6e}, analytic_bound={bnd_temp:.6e}, ratio={ratio_temp:.3f}")

    # Save temperature sensitivity CSV
    try:
        with open("temperature_results.csv", "w", encoding="utf-8") as f:
            f.write("B,T,empirical_illegal_mass,analytic_bound,ratio\n")
            for T_temp in (0.8, 1.2):
                emp_temp = empirical_illegal_mass(legal_logits, illegal_logits, B_fixed, alpha, T_temp)
                bnd_temp = analytic_bound(K, k, Delta, B_fixed, alpha, T_temp)
                ratio_temp = bnd_temp / emp_temp if emp_temp > 0 else float('inf')
                f.write(f"{B_fixed},{T_temp:.1f},{emp_temp:.12e},{bnd_temp:.12e},{ratio_temp:.6f}\n")
        print("# Wrote temperature_results.csv")
    except Exception as e:
        print(f"# Warning: could not write temperature_results.csv: {e}")


if __name__ == "__main__":
    main()