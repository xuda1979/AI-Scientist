#!/usr/bin/env python3
"""
simulation.py — Anytime Conformal Self-Consistency (ACSC) simulation and reference implementation.

Usage:
  python simulation.py --alpha 0.1 --Kmax 16 --n_total 1500 --n_calib 300 --C 8 --seed 42 --gamma 1.2 --noise 0.6

It will print metrics, and save: results.csv, results.json, samples_used_hist.png
"""
import argparse, json, os
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class Instance:
    difficulty: float
    true_label: int

class NoisyLLMSolver:
    def __init__(self, C: int = 10, gamma: float = 1.0, noise: float = 0.6, seed: int = 0):
        self.C = C
        self.gamma = gamma
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def _sigmoid(self, z: float) -> float:
        return 1.0 / (1.0 + np.exp(-z))

    def sample_answer(self, inst: Instance) -> Tuple[int, float]:
        eps = self.rng.normal(0, self.noise)
        p_correct = self._sigmoid(self.gamma - inst.difficulty + eps)
        if self.rng.random() < p_correct:
            yhat = inst.true_label
        else:
            offsets = np.arange(self.C)
            dists = np.minimum(np.abs(offsets - inst.true_label),
                               self.C - np.abs(offsets - inst.true_label))
            probs = np.exp(-(dists + 1.0))
            probs[inst.true_label] = 0.0
            probs /= probs.sum()
            yhat = self.rng.choice(self.C, p=probs)
        a = 1.0 + 6.0 * p_correct
        b = 1.0 + 6.0 * (1.0 - p_correct)
        conf = self.rng.beta(a, b)
        return int(yhat), float(conf)

def make_dataset(n: int, C: int = 10, seed: int = 0) -> List[Instance]:
    rng = np.random.default_rng(seed)
    difficulties = rng.uniform(-2.0, 2.0, size=n)
    labels = rng.integers(0, C, size=n)
    return [Instance(float(d), int(y)) for d, y in zip(difficulties, labels)]

def vote_shares(samples: List[int], C: int) -> Dict[int, float]:
    counts = np.zeros(C, dtype=float)
    for s in samples:
        counts[s] += 1.0
    if len(samples) == 0:
        return {i: 0.0 for i in range(C)}
    shares = counts / len(samples)
    return {i: float(shares[i]) for i in range(C)}

def calibrate_tau(dataset: List[Instance], solver: NoisyLLMSolver,
                  Kmax: int, alpha: float) -> Tuple[float, Dict[str, float]]:
    C = solver.C
    vs_correct = []
    for inst in dataset:
        samples = [solver.sample_answer(inst)[0] for _ in range(Kmax)]
        shares = vote_shares(samples, C)
        vs_correct.append(shares[inst.true_label])
    vs_correct = np.array(vs_correct)
    n = len(vs_correct)
    k = int(np.floor((n + 1) * alpha))
    k = max(0, min(k, n - 1))
    tau = float(np.sort(vs_correct)[k])
    stats = {
        "mean_vs_correct": float(np.mean(vs_correct)),
        "median_vs_correct": float(np.median(vs_correct)),
        "tau": tau
    }
    return tau, stats

def anytime_conformal_predict(inst: Instance, solver: NoisyLLMSolver,
                              tau: float, Kmax: int):
    C = solver.C
    samples = []
    for t in range(1, Kmax + 1):
        yhat, _ = solver.sample_answer(inst)
        samples.append(yhat)
        shares = vote_shares(samples, C)
        S = [y for y, s in shares.items() if s >= tau]
        if len(S) > 0:
            return S, t, samples
    shares = vote_shares(samples, C)
    max_share = max(shares.values())
    S = [y for y, s in shares.items() if s == max_share]
    return S, Kmax, samples

def fixedK_majority_predict(inst: Instance, solver: NoisyLLMSolver, K: int):
    C = solver.C
    samples = [solver.sample_answer(inst)[0] for _ in range(K)]
    shares = vote_shares(samples, C)
    top = max(shares, key=shares.get)
    return int(top), samples

def run_experiment(seed: int = 42, n_total: int = 1500, n_calib: int = 300,
                   C: int = 8, alpha: float = 0.1, Kmax: int = 16,
                   gamma: float = 1.2, noise: float = 0.6):
    data = make_dataset(n_total, C=C, seed=seed)
    calib = data[:n_calib]
    test = data[n_calib:]
    solver = NoisyLLMSolver(C=C, gamma=gamma, noise=noise, seed=seed + 1)
    tau, stats = calibrate_tau(calib, solver, Kmax, alpha)
    cov, set_sizes, samples_used = [], [], []
    for inst in test:
        S, t, _ = anytime_conformal_predict(inst, solver, tau, Kmax)
        cov.append(int(inst.true_label in S))
        set_sizes.append(len(S))
        samples_used.append(t)
    acc_mv = []
    for inst in test:
        top, _ = fixedK_majority_predict(inst, solver, Kmax)
        acc_mv.append(int(top == inst.true_label))
    results = {
        "alpha": alpha,
        "n_total": n_total,
        "n_calib": n_calib,
        "n_test": len(test),
        "Kmax": Kmax,
        "C": C,
        "tau": stats["tau"],
        "calib_mean_vs_correct": stats["mean_vs_correct"],
        "calib_median_vs_correct": stats["median_vs_correct"],
        "coverage_anytime": float(np.mean(cov)),
        "avg_set_size_anytime": float(np.mean(set_sizes)),
        "avg_samples_used_anytime": float(np.mean(samples_used)),
        "median_samples_used_anytime": float(np.median(samples_used)),
        "accuracy_fixedK_majority": float(np.mean(acc_mv))
    }
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_total", type=int, default=1500)
    parser.add_argument("--n_calib", type=int, default=300)
    parser.add_argument("--C", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--Kmax", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=1.2)
    parser.add_argument("--noise", type=float, default=0.6)
    args = parser.parse_args()

    res = run_experiment(seed=args.seed, n_total=args.n_total, n_calib=args.n_calib,
                         C=args.C, alpha=args.alpha, Kmax=args.Kmax,
                         gamma=args.gamma, noise=args.noise)

    df = pd.DataFrame([res])
    df.to_csv("results.csv", index=False)
    with open("results.json", "w") as f:
        json.dump(res, f, indent=2)

    # Diagnostic histogram
    def collect_samples_used(seed: int, n_total: int, n_calib: int, **kwargs):
        data = make_dataset(n_total, C=kwargs.get("C", 10), seed=seed)
        calib = data[:n_calib]
        test = data[n_calib:]
        solver = NoisyLLMSolver(C=kwargs.get("C", 10),
                                gamma=kwargs.get("gamma", 1.0),
                                noise=kwargs.get("noise", 0.6),
                                seed=seed + 1)
        tau, _ = calibrate_tau(calib, solver, kwargs.get("Kmax", 16), kwargs.get("alpha", 0.1))
        samples_used = []
        for inst in test:
            _, t, _ = anytime_conformal_predict(inst, solver, tau, kwargs.get("Kmax", 16))
            samples_used.append(t)
        return np.array(samples_used)

    samples_used = collect_samples_used(seed=args.seed, n_total=args.n_total, n_calib=args.n_calib,
                                        C=args.C, alpha=args.alpha, Kmax=args.Kmax,
                                        gamma=args.gamma, noise=args.noise)
    plt.figure()
    plt.hist(samples_used, bins=np.arange(0.5, args.Kmax + 0.6, 1.0))
    plt.xlabel("Samples used (anytime conformal)")
    plt.ylabel("Count")
    plt.title("Compute adaptivity: distribution of samples used")
    plt.tight_layout()
    plt.savefig("samples_used_hist.png")
    plt.close()

    print(json.dumps(res, indent=2))
    print("\\nSaved: results.csv, results.json, samples_used_hist.png")

if __name__ == "__main__":
    main()
