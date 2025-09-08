"""
Reproducible micro-benchmark for finite-B leakage bound (Theorem 3).
Computes exact illegal mass under masked softmax for a toy setup and
compares it to the analytic upper bound. Results are printed and saved.

Environment: Python >= 3.9. No external dependencies.
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
    # Apply temperature and mask
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


def main():
    # Toy setup matching the paper (Sec. Micro-benchmark)
    legal_logits = [0.2, 0.0, -0.1]
    illegal_logits = [0.0, -0.3, -0.5, -1.0, -1.2, -2.0, -3.0]
    K = len(legal_logits) + len(illegal_logits)
    k = len(legal_logits)
    Tmax_legal = max(legal_logits)
    Tmax_illegal = max(illegal_logits)
    Delta = Tmax_legal - Tmax_illegal  # = 0.2
    alpha = 1.0
    T = 1.0

    Bs = [10, 20, 30]
    rows = []
    print("# Finite-B leakage micro-benchmark")
    print(f"# K={K}, k={k}, Delta={Delta:.1f}, alpha={alpha}, T={T}")
    print("B, empirical_illegal_mass, analytic_bound, ratio_bound_over_empirical")
    for B in Bs:
        emp = empirical_illegal_mass(legal_logits, illegal_logits, B, alpha, T)
        bnd = analytic_bound(K, k, Delta, B, alpha, T)
        ratio = bnd / emp if emp > 0 else float('inf')
        rows.append((B, emp, bnd, ratio))
        print(f"{B}, {emp:.6e}, {bnd:.6e}, {ratio:.3f}")

    # Save CSV for convenience
    try:
        with open("leakage_results.csv", "w", encoding="utf-8") as f:
            f.write("B,empirical_illegal_mass,analytic_bound,ratio\n")
            for B, emp, bnd, ratio in rows:
                f.write(f"{B},{emp:.12e},{bnd:.12e},{ratio:.6f}\n")
        print("# Wrote leakage_results.csv")
    except Exception as e:
        print(f"# Warning: could not write leakage_results.csv: {e}")


if __name__ == "__main__":
    main()