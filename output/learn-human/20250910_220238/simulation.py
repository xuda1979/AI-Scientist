import math
import csv
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 150

# Deterministic setup
np.random.seed(12345)

# Task mixture: per-instance base correctness probabilities (single-sample)
# representing increasing difficulty levels; weights sum to 1
levels = [
    {"p": 0.75, "w": 0.10},
    {"p": 0.65, "w": 0.20},
    {"p": 0.60, "w": 0.30},
    {"p": 0.55, "w": 0.25},
    {"p": 0.52, "w": 0.15},
]

# Budgets (average samples per instance) to evaluate
budgets = [1, 2, 4, 8, 16]

def majority_accuracy(p: float, k: int) -> float:
    """
    Probability that majority voting among k i.i.d. Bernoulli(p) samples yields the correct answer.
    For even k, a tie is broken uniformly at random.
    """
    if k <= 1:
        return float(p)
    acc = 0.0
    half = k // 2
    # strictly greater than half are majority wins
    for i in range(half + 1, k + 1):
        acc += math.comb(k, i) * (p ** i) * ((1 - p) ** (k - i))
    # add half of the tie probability if k is even
    if k % 2 == 0:
        acc += 0.5 * math.comb(k, half) * (p ** half) * ((1 - p) ** half)
    return acc

def mixture_accuracy_equal(levels, k: int) -> float:
    return sum(L["w"] * majority_accuracy(L["p"], k) for L in levels)

def equal_allocation_curve():
    """
    Compute mixture accuracy for Equal self-consistency (EqualSC) at each budget.
    """
    out = []
    for b in budgets:
        acc = mixture_accuracy_equal(levels, b)
        out.append({"budget": b, "accuracy": acc})
    return out

def card_allocation_state(B: int):
    """
    Conflict-aware adaptive allocation (CARD) with odd-step fractional mixing:
    We allow each difficulty level to have a fraction of its instances upgraded
    from an odd number of samples k to the next odd k+2, avoiding zero-gain even steps.

    Greedy segment water-filling (matches Algorithm 1 in the paper):
      - Start all levels at k=1 (used budget = sum_l w_l).
      - For each level l, define segments (k -> k+2) with gain G_l = Acc(p_l,k+2) - Acc(p_l,k).
      - Repeatedly pick the level with the largest G_l; if there is enough remaining budget
        to fully upgrade all instances at that level (cost 2*w_l), do so (k_l += 2) and continue.
        Otherwise, allocate a fraction f in [0,1] of that upgrade to match the exact budget.

    Returns:
      k (np.ndarray of ints): current odd sample counts per level after full upgrades
      frac (np.ndarray of floats): fractional upgrade applied (0 if none) to move from k to k+2
      used (float): total average budget consumed (should equal B up to numerical tolerance)
    """
    k = np.ones(len(levels), dtype=int)  # start at 1 (odd)
    frac = np.zeros(len(levels), dtype=float)  # no partial upgrades yet
    used = float(sum(L["w"] * k[i] for i, L in enumerate(levels)))

    if B < used - 1e-12:
        raise ValueError("Budget smaller than base cost of 1.")

    safety = 0
    while used + 1e-12 < B and safety < 100000:
        gains = []
        for i, L in enumerate(levels):
            base_k = int(k[i])
            p = L["p"]
            # Gain if we move from current odd base to next odd (two samples)
            G = majority_accuracy(p, base_k + 2) - majority_accuracy(p, base_k)
            gains.append((G, i))
        gains.sort(reverse=True)
        best_i = gains[0][1]
        w = levels[best_i]["w"]
        rem = B - used
        cost_full = 2.0 * w
        if rem >= cost_full - 1e-12:
            # fully upgrade this level to next odd
            k[best_i] += 2
            used += cost_full
        else:
            # allocate fractional upgrade to match exactly
            f = max(0.0, min(1.0, rem / cost_full))
            frac[best_i] += f
            used += f * cost_full
            break
        safety += 1

    return k, frac, used

def adaptive_allocation_curve():
    """
    Compute mixture accuracy for CARD at each budget using the odd-segment
    water-filling allocation with fractional mixing.
    """
    results = []
    for B in budgets:
        k, frac, used = card_allocation_state(B)
        # Compute mixture accuracy with any fractional upgrade accounted for
        mix_acc = 0.0
        for i, L in enumerate(levels):
            p = L["p"]; w = L["w"]
            base_k = int(k[i])
            if frac[i] > 0:
                f = min(1.0, frac[i])
                acc_i = (1 - f) * majority_accuracy(p, base_k) + f * majority_accuracy(p, base_k + 2)
            else:
                acc_i = majority_accuracy(p, base_k)
            mix_acc += w * acc_i
        results.append({"budget": B, "accuracy": mix_acc})
    return results

def reliability_pairs():
    """
    Generate confidence-accuracy pairs across mixture levels and budgets (no binning),
    ensuring a sufficiently dense reliability plot (L * |budgets| points).
    Confidence is a conservative proxy based on the single-sample p and increases with k:
      conf = 0.5 + (p-0.5) * sqrt(k)/sqrt(k+1)
    Accuracy is Acc(p,k) in closed form.
    Each pair is weighted in the plotting by the mixture w, but we output raw pairs.
    """
    pairs = []
    for B in budgets:
        for L in levels:
            p = L["p"]; k = B
            f = math.sqrt(k) / math.sqrt(k + 1.0)
            conf = 0.5 + (p - 0.5) * f
            acc = majority_accuracy(p, k)
            pairs.append((conf, acc, L["w"]))
    pairs.sort(key=lambda x: x[0])
    return pairs

def expected_calibration_error_equal(bins: int = 10):
    """
    Compute ECE for EqualSC across budgets using mixture-weighted pairs.
    Bins are equally spaced in [0.5, 1.0].
    Returns list of dicts: {"budget": b, "ECE": ece}
    """
    results = []
    bin_edges = np.linspace(0.5, 1.0, bins + 1)
    for B in budgets:
        # construct pairs for this budget
        pairs = []
        for L in levels:
            p = L["p"]; w = L["w"]
            f = math.sqrt(B) / math.sqrt(B + 1.0)
            conf = 0.5 + (p - 0.5) * f
            acc = majority_accuracy(p, B)
            pairs.append((conf, acc, w))
        # accumulate per bin
        bin_w = np.zeros(bins, dtype=float)
        bin_conf = np.zeros(bins, dtype=float)
        bin_acc = np.zeros(bins, dtype=float)
        for conf, acc, w in pairs:
            idx = np.searchsorted(bin_edges, conf, side="right") - 1
            idx = min(max(idx, 0), bins - 1)
            bin_w[idx] += w
            bin_conf[idx] += w * conf
            bin_acc[idx] += w * acc
        # compute ECE
        ece = 0.0
        total_w = sum(L["w"] for L in levels)  # should be 1.0
        for i in range(bins):
            if bin_w[i] > 0:
                avg_conf = bin_conf[i] / bin_w[i]
                avg_acc = bin_acc[i] / bin_w[i]
                ece += (bin_w[i] / total_w) * abs(avg_acc - avg_conf)
        results.append({"budget": B, "ECE": ece})
    return results

def write_results_txt(equal_curve, adaptive_curve, n_examples=50000, ece_equal=None):
    ts = datetime.now(timezone.utc).isoformat()
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write("# Results generated: %s\n" % ts)
        f.write("# Task mixture base per-sample correctness probabilities and weights:\n")
        for L in levels:
            f.write("#   p=%.3f, w=%.2f\n" % (L["p"], L["w"]))
        f.write("# Budgets evaluated: %s\n" % ",".join(str(b) for b in budgets))
        f.write("# Number of synthetic examples assumed for CI reporting: %d\n" % n_examples)
        f.write("method,budget,accuracy,stderr_approx\n")
        for rec in equal_curve:
            p = rec["accuracy"]; se = math.sqrt(max(p * (1 - p), 1e-9) / n_examples)
            f.write("EqualSC,%d,%.6f,%.6f\n" % (rec["budget"], rec["accuracy"], se))
        for rec in adaptive_curve:
            p = rec["accuracy"]; se = math.sqrt(max(p * (1 - p), 1e-9) / n_examples)
            f.write("CARD,%d,%.6f,%.6f\n" % (rec["budget"], rec["accuracy"], se))
        if ece_equal is not None:
            f.write("metric,budget,value\n")
            for rec in ece_equal:
                f.write("ECE_EqualSC,%d,%.6f\n" % (rec["budget"], rec["ECE"]))

def write_plot_data(equal_curve, adaptive_curve):
    with open("results_plot_data.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["budget", "EqualSC", "CARD"])
        for e, a in zip(equal_curve, adaptive_curve):
            w.writerow([e["budget"], "%.6f" % e["accuracy"], "%.6f" % a["accuracy"]])

def write_reliability_plot_data():
    pairs = reliability_pairs()
    with open("reliability_plot_data.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["confidence", "accuracy"])
        for c, a, _w in pairs:
            w.writerow(["%.3f" % c, "%.3f" % a])

def write_ece_plot_data(ece_equal):
    with open("ece_plot_data.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["budget", "ECE"])
        for rec in ece_equal:
            w.writerow([rec["budget"], "%.6f" % rec["ECE"]])

def plot_accuracy_curves(equal_curve, adaptive_curve):
    B = [r["budget"] for r in equal_curve]
    A_eq = [r["accuracy"] for r in equal_curve]
    A_ad = [r["accuracy"] for r in adaptive_curve]
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(B, A_eq, marker="o", label="Equal self-consistency")
    plt.plot(B, A_ad, marker="s", label="Neuro-inspired adaptive (CARD)")
    plt.xlabel("Average test-time budget (samples per instance)")
    plt.ylabel("Mixture accuracy")
    plt.title("Accuracy vs. test-time compute")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_vs_budget.png")
    plt.close()

def plot_reliability():
    pairs = reliability_pairs()
    confs = [c for (c, _, _) in pairs]
    accs = [a for (_, a, _) in pairs]
    plt.figure(figsize=(6.5, 4.5))
    # Draw identity line as a dense function for visual clarity
    xs = np.linspace(0.5, 1.0, 201)
    plt.plot(xs, xs, "k--", label="y=x (perfect)")
    plt.plot(confs, accs, marker="o", label="Equal self-consistency (pairs)")
    plt.xlabel("Predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Reliability (mixture, across budgets and levels)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("calibration_reliability.png")
    plt.close()

def plot_ece(ece_equal):
    B = [r["budget"] for r in ece_equal]
    E = [r["ECE"] for r in ece_equal]
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(B, E, marker="^", label="ECE (EqualSC, 10 bins)")
    plt.xlabel("Average test-time budget (samples per instance)")
    plt.ylabel("Expected Calibration Error (ECE)")
    plt.title("Calibration vs. budget")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ece_vs_budget.png")
    plt.close()

def main():
    eq = equal_allocation_curve()
    ad = adaptive_allocation_curve()
    ece_eq = expected_calibration_error_equal(bins=10)
    write_results_txt(eq, ad, ece_equal=ece_eq)
    write_plot_data(eq, ad)
    write_reliability_plot_data()
    write_ece_plot_data(ece_eq)
    plot_accuracy_curves(eq, ad)
    plot_reliability()
    plot_ece(ece_eq)

if __name__ == "__main__":
    main()