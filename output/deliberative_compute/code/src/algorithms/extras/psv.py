
from __future__ import annotations
import numpy as np
from typing import Dict, Any
from ..base import Algorithm, AlgoResult, Environment
from ...evaluation.compute_accounting import ComputeAccountant, overhead_timer

class PSV(Algorithm):
    """Probabilistic Self-Verification (toy):
    Interleave computation with light-weight checks; stop when log-odds improvement falls below threshold.
    """
    name = "psv"

    def __init__(self, max_checks: int = 5, logodds_thresh: float = 0.1):
        self.max_checks = int(max_checks)
        self.logodds_thresh = float(logodds_thresh)

    def run(self, env: Environment, budget: int, rng: np.random.Generator, acc: ComputeAccountant) -> AlgoResult:
        env = env.clone()
        logodds = 0.0
        used = 0
        while used < budget:
            with overhead_timer(acc):
                t = rng.integers(0, env.n_threads())
                acc.count_decision(1)
            acc.charge("generation", 1)
            env.step_extend_thread(int(t), rng)
            used += 1
            # probabilistic check
            if used % max(1, budget // self.max_checks) == 0 and self.max_checks > 0:
                with overhead_timer(acc):
                    # toy likelihood ratio from recent improvement
                    lr = 1.0 + rng.random() * 0.5
                    logodds += np.log(lr)
                acc.charge("verification", 1)
                if logodds < self.logodds_thresh:
                    break
        return AlgoResult(env.utility(), {"logodds": logodds, "used": used}, acc.snapshot())
