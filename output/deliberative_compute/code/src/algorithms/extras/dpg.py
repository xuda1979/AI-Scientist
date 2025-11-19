
from __future__ import annotations
import numpy as np
from typing import Dict, Any
from ..base import Algorithm, AlgoResult, Environment
from ...evaluation.compute_accounting import ComputeAccountant, overhead_timer
from ..baselines import CoT
from ..igd import IGD

class DPG(Algorithm):
    """Dual-Process Gating: fast path (CoT) escalates to IGD if confidence proxy is low."""
    name = "dpg"
    def __init__(self, theta: float = 0.6, discount: float = 0.95, risk_k: float = 0.5):
        self.theta = float(theta)
        self.igd = IGD(discount=discount, risk_k=risk_k)
        self.fast = CoT()

    def run(self, env: Environment, budget: int, rng: np.random.Generator, acc: ComputeAccountant) -> AlgoResult:
        # Do a quick probe to estimate confidence proxy
        env_probe = env.clone()
        probe_steps = max(1, budget // 10)
        for _ in range(probe_steps):
            with overhead_timer(acc):
                t = rng.integers(0, env_probe.n_threads())
                acc.count_decision(1)
            acc.charge("generation", 1)
            env_probe.step_extend_thread(int(t), rng)
        q = env_probe.utility()  # proxy for confidence
        remaining = max(0, budget - probe_steps)
        if q >= self.theta or remaining == 0:
            # stay fast
            return self.fast.run(env, remaining, rng, acc)
        else:
            # escalate
            return self.igd.run(env, remaining, rng, acc)
