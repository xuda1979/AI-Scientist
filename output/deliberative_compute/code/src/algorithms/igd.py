
from __future__ import annotations
import numpy as np
from typing import Any, Dict, List
from .base import Algorithm, AlgoResult, Environment
from ..evaluation.compute_accounting import ComputeAccountant, overhead_timer

class IGD(Algorithm):
    """Index-Guided Deliberation (lightweight implementation).
    We maintain per-thread priors over discounted incremental gains and choose the thread with the
    largest conservative index (mean - k * std)/cost until budget is exhausted.
    """
    name = "igd"

    def __init__(self, discount: float = 0.95, risk_k: float = 0.5):
        self.discount = float(discount)
        self.risk_k = float(risk_k)

    def run(self, env: Environment, budget: int, rng: np.random.Generator, acc: ComputeAccountant) -> AlgoResult:
        env = env.clone()
        m = env.n_threads()
        # Posterior summaries: mean and variance of *one-step* gains per thread; we approximate discounted chain
        means = np.zeros(m, dtype=float)
        vars_ = np.ones(m, dtype=float) * 0.05
        counts = np.ones(m, dtype=float) * 1e-6  # avoid div by zero
        total_actions = 0

        for _ in range(budget):
            with overhead_timer(acc):
                # Conservative index ≈ (mean - k*std) / (1 - discount) per unit cost (cost=1)
                std = np.sqrt(np.maximum(vars_ / counts, 1e-12))
                index = (means - self.risk_k * std) / max(1e-6, (1.0 - self.discount))
                i = int(np.argmax(index))
                acc.count_decision(1)

            acc.charge("generation", 1)
            gain = env.step_extend_thread(i, rng)
            total_actions += 1
            # Update posterior (online mean/var)
            delta = gain - means[i]
            counts[i] += 1.0
            means[i] += delta / counts[i]
            vars_[i] += delta * (gain - means[i])  # Welford-like

        u = env.utility()
        details = {
            "means": means.tolist(),
            "counts": counts.tolist(),
            "total_actions": total_actions,
        }
        return AlgoResult(u, details, acc.snapshot())
