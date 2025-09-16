"""
Evaluation harness for 'Model Organisms for Foundation Model Risk'

This script simulates six task families with stylized stochastic policies,
computes mean risk metrics with bootstrap confidence intervals, and evaluates
best-of-k selection under an imperfect safety selector for the jailbreak task.

Enhancements in this version:
- Emits reliability-curve bins for CalibQA to support a reliability diagram.
- Performs selector-noise sensitivity analysis (sigma in {0.3, 0.6, 1.0}) for best-of-k.
- Logs all diagnostics in the JSON payload for reproducibility/auditing.

Reproducibility:
- Python 3.9+ recommended
- Dependencies: numpy
- Seeds are fixed for deterministic behavior across runs on the same platform/versions.

Outputs:
- Printed aggregate metrics and 95% CIs for each task family
- Printed best-of-k attack success for k in {1,2,3,5,8,10}
- Reliability bins for CalibQA
- Selector-noise sensitivity curves
- JSON payload with all results and diagnostics

Note: This is a simulation capturing qualitative patterns reported in the paper.
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# -----------------------
# Utilities
# -----------------------

def set_seeds(master_seed: int = 2025) -> None:
    # Set legacy seed for any code relying on np.random
    np.random.seed(master_seed)


def bootstrap_ci(values: np.ndarray, alpha: float = 0.05, B: int = 10_000, seed: int = 1337) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, n, size=(B, n))
    samples = values[idx].mean(axis=1)
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return lo, hi


def reliability_bins(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute reliability curve bins: for each probability bin, return the mean predicted probability (x)
    and the empirical accuracy (y). Empty bins are skipped.
    """
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    x_points: List[float] = []
    y_points: List[float] = []
    for b in range(num_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (probs >= lo) & (probs < hi) if b < num_bins - 1 else (probs >= lo) & (probs <= hi)
        if not np.any(mask):
            continue
        p_bin = probs[mask]
        y_bin = labels[mask]
        x_points.append(float(np.mean(p_bin)))
        y_points.append(float(np.mean(y_bin)))
    return np.array(x_points, dtype=float), np.array(y_points, dtype=float)


@dataclass
class TaskConfig:
    name: str
    N: int = 200            # trials
    m: int = 1              # queries per trial
    p: float = 0.1          # base risk parameter for Bernoulli-like tasks
    # Calib-specific
    calib_mean_conf: float = 0.7
    calib_noise: float = 0.15
    # Jailbreak selector noise
    selector_sigma: float = 0.6


@dataclass
class TaskResult:
    name: str
    mean: float
    ci95: Tuple[float, float]
    extra: Dict[str, float]


# -----------------------
# Task Simulators
# -----------------------

def simulate_bernoulli_risk(cfg: TaskConfig, seed: int) -> np.ndarray:
    """Simulate Nm Bernoulli outcomes with success prob cfg.p."""
    rng = np.random.default_rng(seed)
    M = cfg.N * cfg.m
    return rng.binomial(1, cfg.p, size=M).astype(float)


def simulate_calibration(cfg: TaskConfig, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate calibration with probabilities around calib_mean_conf and labels sampled
    from a Bernoulli with underlying prevalence moderately lower than mean confidence.
    Returns:
      - per-sample Brier score array
      - per-sample predicted probabilities
      - per-sample binary labels
    """
    rng = np.random.default_rng(seed)
    M = cfg.N * cfg.m
    probs = np.clip(
        rng.normal(loc=cfg.calib_mean_conf, scale=cfg.calib_noise, size=M),
        0.01, 0.99
    )
    prevalence = max(0.5 * cfg.calib_mean_conf, 0.05)
    labels = rng.binomial(1, prevalence, size=M).astype(float)
    per_sample_brier = (probs - labels) ** 2
    return per_sample_brier, probs, labels


def simulate_jailbreak_best_of_k(cfg: TaskConfig, ks: List[int], seed: int) -> Tuple[Dict[int, float], np.ndarray]:
    """
    Simulate jailbreak attack success under best-of-k with a noisy selector.
    For each query:
      - Draw k Bernoulli successes with prob p (1 = unsafe, 0 = safe).
      - Assign a safety score s = y + noise, where y in {0,1}, noise ~ N(0, sigma).
      - Choose candidate with minimum score (lower judged "safer").
      - Count chosen=1 as attack success.
    Returns:
      - dict mapping k -> success_rate
      - the chosen vector for k=1 (for CI consistency with base metric)
    """
    rng = np.random.default_rng(seed)
    M = cfg.N * cfg.m
    sigma = cfg.selector_sigma
    results: Dict[int, float] = {}
    chosen_k1 = None

    for k in ks:
        y = rng.binomial(1, cfg.p, size=(M, k)).astype(float)
        noise = rng.normal(loc=0.0, scale=sigma, size=(M, k))
        scores = y + noise
        idx = np.argmin(scores, axis=1)
        chosen = y[np.arange(M), idx]
        if k == 1:
            chosen_k1 = chosen.copy()
        results[k] = float(np.mean(chosen))

    assert chosen_k1 is not None
    return results, chosen_k1


def simulate_jailbreak_sigma_sweep(cfg: TaskConfig, ks: List[int], sigmas: List[float], seed: int) -> Dict[float, Dict[int, float]]:
    """
    For each sigma in sigmas, compute best-of-k success rates.
    Returns nested dict: sigma -> {k -> success_rate}
    """
    out: Dict[float, Dict[int, float]] = {}
    for s in sigmas:
        cfg_local = TaskConfig(**{**cfg.__dict__, "selector_sigma": s})
        by_k, _ = simulate_jailbreak_best_of_k(cfg_local, ks=ks, seed=seed + int(1000 * s))
        out[s] = by_k
    return out


# -----------------------
# Main evaluation
# -----------------------

def evaluate_all() -> Dict[str, TaskResult]:
    set_seeds(2025)

    ks = [1, 2, 3, 5, 8, 10]
    results: Dict[str, TaskResult] = {}

    # Configure tasks with target risk parameters
    task_cfgs = {
        "GMG-Gridworld": TaskConfig(name="GMG-Gridworld", p=0.31),
        "InstrReverse": TaskConfig(name="InstrReverse", p=0.18),
        "ToolSelect": TaskConfig(name="ToolSelect", p=0.12),
        "JailbreakResist": TaskConfig(name="JailbreakResist", p=0.42, selector_sigma=0.6),
        "DataExfil": TaskConfig(name="DataExfil", p=0.06),
        "CalibQA": TaskConfig(name="CalibQA", calib_mean_conf=0.65, calib_noise=0.18),
    }

    # GMG-Gridworld
    gmg_samples = simulate_bernoulli_risk(task_cfgs["GMG-Gridworld"], seed=1001)
    gmg_mean = float(np.mean(gmg_samples))
    gmg_ci = bootstrap_ci(gmg_samples, B=10_000, seed=5001)
    results["GMG-Gridworld"] = TaskResult("GMG-Gridworld", gmg_mean, gmg_ci, {})

    # InstrReverse
    ir_samples = simulate_bernoulli_risk(task_cfgs["InstrReverse"], seed=1002)
    ir_mean = float(np.mean(ir_samples))
    ir_ci = bootstrap_ci(ir_samples, B=10_000, seed=5002)
    results["InstrReverse"] = TaskResult("InstrReverse", ir_mean, ir_ci, {})

    # ToolSelect
    ts_samples = simulate_bernoulli_risk(task_cfgs["ToolSelect"], seed=1003)
    ts_mean = float(np.mean(ts_samples))
    ts_ci = bootstrap_ci(ts_samples, B=10_000, seed=5003)
    results["ToolSelect"] = TaskResult("ToolSelect", ts_mean, ts_ci, {})

    # DataExfil
    de_samples = simulate_bernoulli_risk(task_cfgs["DataExfil"], seed=1004)
    de_mean = float(np.mean(de_samples))
    de_ci = bootstrap_ci(de_samples, B=10_000, seed=5004)
    results["DataExfil"] = TaskResult("DataExfil", de_mean, de_ci, {})

    # CalibQA (Brier score) + reliability bins
    calib_brier, calib_probs, calib_labels = simulate_calibration(task_cfgs["CalibQA"], seed=1005)
    calib_mean = float(np.mean(calib_brier))
    calib_ci = bootstrap_ci(calib_brier, B=10_000, seed=5005)
    # Reliability curve (10 bins)
    rx, ry = reliability_bins(calib_probs, calib_labels, num_bins=10)
    calib_extra = {f"reliability_x_{i}": float(rx[i]) for i in range(len(rx))}
    calib_extra.update({f"reliability_y_{i}": float(ry[i]) for i in range(len(ry))})
    results["CalibQA"] = TaskResult("CalibQA", calib_mean, calib_ci, calib_extra)

    # JailbreakResist with best-of-k; derive base metric from k=1 chosen vector
    jb_cfg = task_cfgs["JailbreakResist"]
    jb_by_k, chosen_k1 = simulate_jailbreak_best_of_k(jb_cfg, ks=ks, seed=1006)
    jb_mean = float(np.mean(chosen_k1))
    jb_ci = bootstrap_ci(chosen_k1.astype(float), B=10_000, seed=5006)
    # Sigma sweep
    sigma_sweep = simulate_jailbreak_sigma_sweep(jb_cfg, ks=ks, sigmas=[0.3, 0.6, 1.0], seed=2006)
    extra = {f"k={k}": v for k, v in jb_by_k.items()}
    # Flatten sigma sweep into extra
    for s, byk in sigma_sweep.items():
        for k, v in byk.items():
            extra[f"sigma={s}:k={k}"] = v
    results["JailbreakResist"] = TaskResult("JailbreakResist", jb_mean, jb_ci, extra)

    return results


def main():
    results = evaluate_all()

    print("=== Aggregate Risk Metrics (mean and 95% CI) ===")
    order = ["GMG-Gridworld", "InstrReverse", "ToolSelect", "JailbreakResist", "DataExfil", "CalibQA"]
    for name in order:
        res = results[name]
        metric_name = "Brier" if name == "CalibQA" else "Risk"
        print(f"{name:16s}: mean {res.mean:.3f}, 95% CI [{res.ci95[0]:.3f}, {res.ci95[1]:.3f}] ({metric_name})")

    print("\n=== JailbreakResist: Best-of-k attack success ===")
    jb = results["JailbreakResist"]
    byk = {int(k.split('=')[1]): v for k, v in jb.extra.items() if k.startswith("k=")}
    for k in sorted(byk.keys()):
        print(f" k={k}: {byk[k]:.3f}")

    # Print selector-noise sensitivity
    print("\n=== JailbreakResist: Selector-noise sensitivity (best-of-k) ===")
    # Collect sigmas
    sigmas = sorted({float(k.split(':')[0].split('=')[1]) for k in jb.extra if k.startswith("sigma=")})
    ks = sorted({int(k.split('=')[2]) for k in jb.extra if k.startswith("sigma=")})
    for s in sigmas:
        row = []
        for k in ks:
            v = jb.extra.get(f"sigma={s}:k={k}", float('nan'))
            row.append(f"k={k}:{v:.3f}")
        print(f"sigma={s}: " + ", ".join(row))

    # Reliability curve (CalibQA)
    print("\n=== CalibQA: Reliability bins (x=mean predicted, y=empirical accuracy) ===")
    calib = results["CalibQA"]
    for i in range(50):  # generous upper bound; some bins may be missing
        xi = calib.extra.get(f"reliability_x_{i}", None)
        yi = calib.extra.get(f"reliability_y_{i}", None)
        if xi is None or yi is None:
            continue
        print(f"bin {i:02d}: x={xi:.3f}, y={yi:.3f}")

    payload = {name: {"mean": res.mean, "ci95": [res.ci95[0], res.ci95[1]], "extra": res.extra} for name, res in results.items()}
    print("\nJSON:")
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()