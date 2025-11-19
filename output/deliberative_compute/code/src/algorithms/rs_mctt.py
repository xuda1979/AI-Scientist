
from __future__ import annotations
import numpy as np
from typing import Any, Dict
from .base import Algorithm, AlgoResult, Environment
from ..evaluation.compute_accounting import ComputeAccountant, overhead_timer

def entropic_risk(xs: np.ndarray, eta: float) -> float:
    if xs.size == 0:
        return 0.0
    if abs(eta) < 1e-9:
        return float(np.mean(xs))
    # numerically stable log-mean-exp
    m = np.max(eta * xs)
    return float((np.log(np.mean(np.exp(eta * xs - m))) + m) / eta)

class RS_MCTT(Algorithm):
    """Risk-Sensitive Monte Carlo Tree-of-Thought (bandit-style approximation).
    We approximate UCT selection with an entropic-risk transform over per-thread returns and a
    log-UCT exploration bonus. This avoids full tree explosion while preserving risk-annealing.
    """
    name = "rs_mctt"

    def __init__(self, eta_start: float = 0.0, eta_end: float = -3.0, exploration: float = 1.0):
        self.eta_start = float(eta_start)
        self.eta_end = float(eta_end)
        self.exploration = float(exploration)

    def run(self, env: Environment, budget: int, rng: np.random.Generator, acc: ComputeAccountant) -> AlgoResult:
        env = env.clone()
        m = env.n_threads()
        # Maintain per-thread observed returns (gains) history to compute entropic risk
        returns = [list() for _ in range(m)]
        pulls = np.zeros(m, dtype=int)
        total = 0

        for t in range(1, budget + 1):
            # Depth-dependent risk temperature
            frac = t / max(budget, 1)
            eta = self.eta_start + frac * (self.eta_end - self.eta_start)
            with overhead_timer(acc):
                scores = np.zeros(m, dtype=float)
                ln_total = np.log(max(1, total))
                for i in range(m):
                    erisk = entropic_risk(np.array(returns[i], dtype=float), eta) if pulls[i] > 0 else 0.0
                    bonus = self.exploration * np.sqrt(max(0.0, (ln_total - np.log(max(1, pulls[i]))) if pulls[i] > 0 else ln_total))
                    scores[i] = erisk + bonus
                a = int(np.argmax(scores))
                acc.count_decision(1)

            acc.charge("generation", 1)
            g = env.step_extend_thread(a, rng)
            total += 1
            pulls[a] += 1
            returns[a].append(g)

        u = env.utility()
        details = {"eta_start": self.eta_start, "eta_end": self.eta_end, "pulls": pulls.tolist()}
        return AlgoResult(u, details, acc.snapshot())
