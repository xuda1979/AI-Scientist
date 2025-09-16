# Deterministic, comprehensive simulation of key LLM problem dimensions
# Generates a results report and several publication-quality figures.

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone

rng = np.random.default_rng(42)

# 1) Test-time compute scaling via self-consistency (k candidates)
k_values = np.array([1, 2, 4, 8, 16], dtype=int)
base_acc = 0.62
acc_gain = 0.14
acc_k = base_acc + acc_gain * (np.log2(k_values) / np.log2(16))
latency_k = np.sqrt(k_values)  # normalized latency

# 2) Long-context degradation
ctx_k = np.array([1, 4, 16, 32, 64], dtype=int)  # in thousands of tokens
acc_ctx = 0.74 - 0.06 * np.log2(ctx_k)  # 0.74 at 1k, 0.62 at 4k, ..., 0.38 at 64k

# 3) Calibration (fixed-bin synthetic reliability diagram with exact ECE_eq=0.04)
bins = np.linspace(0.05, 0.95, 10)
# Differences symmetric around 0.04 average absolute deviation
diffs = np.array([0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.03, 0.04, 0.05, 0.06])
# sign(bins - 0.5) ==> underconfident below 0.5, overconfident above
acc_bins = np.clip(bins - np.sign(bins - 0.5) * diffs, 0, 1)  # empirical accuracies per bin
# Equal-weight ECE (used for exact reproducibility in paper)
ece_eq = float(np.mean(np.abs(acc_bins - bins)))
# Standard ECE with per-bin weights (here equal mass per bin for determinism)
weights = np.ones_like(bins) / len(bins)
ece_std = float(np.sum(weights * np.abs(acc_bins - bins)))

# 3b) Selective prediction (abstention) trade-offs via confidence thresholds
thresholds = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
coverage = []
acc_covered = []
for t in thresholds:
    mask = bins >= t
    cov = float(np.mean(mask))  # equal mass per bin
    coverage.append(cov)
    if np.any(mask):
        acc_cov = float(np.mean(acc_bins[mask]))
    else:
        acc_cov = float('nan')
    acc_covered.append(acc_cov)
coverage = np.array(coverage)
acc_covered = np.array(acc_covered)

# 4) Prompt-injection attack success vs. attack strength; defense shifts logit
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

attack_strength = np.linspace(0, 1, 9)
logit_scale = 3.0
defense_shift = 1.5
success_no_def = sigmoid(logit_scale * (attack_strength - 0.5))
success_def = sigmoid(logit_scale * (attack_strength - 0.5) - defense_shift)

# Snapshot at medium-strength attack for tabulation
s_mid = 0.5
succ_mid_no = float(sigmoid(logit_scale * (s_mid - 0.5)))
succ_mid_def = float(sigmoid(logit_scale * (s_mid - 0.5) - defense_shift))

# 5) Toxicity under red-teaming pressure and partial alignment
alignment_strength = 0.6  # 0..1
toxicity_rate = 0.2 * (1 - alignment_strength) + 0.002 * np.mean(attack_strength)  # deterministic
# equals 0.08 + 0.001 = 0.081

# 6) Memorization exposure vs. duplication count (canary-analysis inspired)
dup_counts = np.array([1, 2, 4, 8, 16], dtype=int)
lambda_mem = 0.08
exposure = 1 - np.exp(-lambda_mem * dup_counts)

# 7) Optional paraphrase robustness curve (not plotted, saved numerically)
paraphrase_level = np.linspace(0, 1, 6)
acc_paraphrase = 0.68 - 0.14 * paraphrase_level

# Write results with clear, human-readable structure
RESULTS_PATH = "results.txt"
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    f.write(f"timestamp: {datetime.now(timezone.utc).isoformat()}\n")
    f.write("=== Test-time compute scaling (self-consistency) ===\n")
    f.write(f"k_values: {','.join(map(str, k_values))}\n")
    f.write(f"accuracy: {','.join(f'{a:.3f}' for a in acc_k)}\n")
    f.write(f"latency: {','.join(f'{l:.3f}' for l in latency_k)}\n")
    f.write("=== Long-context degradation ===\n")
    f.write(f"context_k_tokens: {','.join(map(str, ctx_k))}\n")
    f.write(f"accuracy: {','.join(f'{a:.3f}' for a in acc_ctx)}\n")
    f.write("=== Calibration ===\n")
    f.write(f"bins_conf: {','.join(f'{b:.2f}' for b in bins)}\n")
    f.write(f"acc_per_bin: {','.join(f'{a:.3f}' for a in acc_bins)}\n")
    f.write(f"ece_eq: {ece_eq:.3f}\n")
    f.write(f"ece_std: {ece_std:.3f}\n")
    f.write("=== Selective prediction (abstention) ===\n")
    f.write(f"thresholds: {','.join(f'{t:.2f}' for t in thresholds)}\n")
    f.write(f"coverage: {','.join(f'{c:.3f}' for c in coverage)}\n")
    f.write(f"accuracy_covered: {','.join(f'{a:.3f}' for a in acc_covered)}\n")
    f.write("=== Prompt injection ===\n")
    f.write(f"attack_strengths: {','.join(f'{s:.2f}' for s in attack_strength)}\n")
    f.write(f"success_no_defense: {','.join(f'{s:.3f}' for s in success_no_def)}\n")
    f.write(f"success_with_defense: {','.join(f'{s:.3f}' for s in success_def)}\n")
    f.write(f"success_at_strength_0.50_no_defense: {succ_mid_no:.3f}\n")
    f.write(f"success_at_strength_0.50_with_defense: {succ_mid_def:.3f}\n")
    f.write("=== Safety (toxicity) ===\n")
    f.write(f"alignment_strength: {alignment_strength:.3f}\n")
    f.write(f"toxicity_rate: {toxicity_rate:.3f}\n")
    f.write("=== Memorization exposure ===\n")
    f.write(f"dup_counts: {','.join(map(str, dup_counts))}\n")
    f.write(f"exposure: {','.join(f'{e:.3f}' for e in exposure)}\n")
    f.write("=== Robustness to paraphrase ===\n")
    f.write(f"paraphrase_level: {','.join(f'{p:.2f}' for p in paraphrase_level)}\n")
    f.write(f"accuracy: {','.join(f'{a:.3f}' for a in acc_paraphrase)}\n")

# Plot 1: Test-time compute scaling
plt.figure(figsize=(6,4), dpi=150)
plt.plot(k_values, acc_k, marker='o', label='Accuracy')
plt.xlabel('Candidates (k)')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)
ax2 = plt.gca().twinx()
ax2.plot(k_values, latency_k, color='tab:red', marker='s', linestyle='--', label='Latency (normalized)')
ax2.set_ylabel('Latency (normalized)')
lines1, labels1 = plt.gca().get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
plt.title('Test-time compute scaling via self-consistency')
plt.tight_layout()
plt.savefig("fig_scaling.pdf")
plt.close()

# Plot 2: Long-context degradation
plt.figure(figsize=(6,4), dpi=150)
plt.plot(ctx_k, acc_ctx, marker='o')
plt.xlabel('Context length (thousands of tokens)')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)
plt.title('Long-context performance degradation')
plt.tight_layout()
plt.savefig("fig_context.pdf")
plt.close()

# Plot 3: Calibration reliability diagram
plt.figure(figsize=(6,4), dpi=150)
width = 0.07
plt.bar(bins, acc_bins, width=width, alpha=0.8, label='Empirical accuracy')
xline = np.linspace(0,1,100)
plt.plot(xline, xline, 'k--', label='Perfect calibration')
plt.xlabel('Predicted confidence')
plt.ylabel('Empirical accuracy')
plt.title(f'Reliability diagram (ECE_eq = {ece_eq:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fig_calibration.pdf")
plt.close()

# Plot 4: Prompt-injection attack success
plt.figure(figsize=(6,4), dpi=150)
plt.plot(attack_strength, success_no_def, marker='o', label='No defense')
plt.plot(attack_strength, success_def, marker='s', linestyle='--', label='With defense')
plt.xlabel('Attack strength')
plt.ylabel('Attack success rate')
plt.title('Prompt-injection vulnerability curve')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("fig_attack.pdf")
plt.close()

# Plot 5: Memorization exposure vs. duplication
plt.figure(figsize=(6,4), dpi=150)
plt.plot(dup_counts, exposure, marker='o')
plt.xlabel('Duplication count')
plt.ylabel('Exposure')
plt.title('Memorization exposure vs. duplication')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fig_memorization.pdf")
plt.close()