
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List
from ..base import Algorithm, AlgoResult, Environment
from ...evaluation.compute_accounting import ComputeAccountant, overhead_timer

class CSC(Algorithm):
    """Counterfactual Self-Consistency (toy): run k chains, filter chains by pseudo-constraints.

    Constraints are simulated by a simple rule on partial gains; noisy checks reject a fraction of
    brittle chains while keeping most stable ones.
    """
    name = "csc"

    def __init__(self, chains: int = 7, constraint_strength: float = 0.15):
        self.chains = int(chains)
        self.constraint_strength = float(constraint_strength)

    def run(self, env: Environment, budget: int, rng: np.random.Generator, acc: ComputeAccountant) -> AlgoResult:
        chain_b = max(1, budget // self.chains)
        utils = []
        keep = []
        for _ in range(self.chains):
            env_i = env.clone()
            partials = []
            for _ in range(chain_b):
                with overhead_timer(acc):
                    t = rng.integers(0, env_i.n_threads())
                    acc.count_decision(1)
                acc.charge("generation", 1)
                g = env_i.step_extend_thread(int(t), rng)
                partials.append(g)
            # constraint check (noisy): filter if variance is too high (brittle) with probability
            v = np.var(partials) if len(partials) > 1 else 0.0
            prob_reject = min(0.7, self.constraint_strength * v * len(partials))
            keep_flag = (rng.random() > prob_reject)
            keep.append(keep_flag)
            if keep_flag:
                utils.append(env_i.utility())
        if len(utils) == 0:
            # Fall back to mean of all chains
            utils = [env.clone().utility() for _ in range(self.chains)]
        u = float(np.mean(utils))
        return AlgoResult(u, {"kept": int(sum(keep)), "total": self.chains}, acc.snapshot())
