
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List
from ..base import Algorithm, AlgoResult, Environment
from ...evaluation.compute_accounting import ComputeAccountant, overhead_timer

class APT(Algorithm):
    """Annealed Population of Thoughts (toy implementation).
    Maintain M candidate partial trajectories as vectors of (thread_id, gain).
    Epoch loop: mutate (random extension), reweight by exp(beta * score), resample.
    """
    name = "apt"

    def __init__(self, population: int = 8, epochs: int = 5, beta_max: float = 3.0):
        self.population = int(population)
        self.epochs = int(epochs)
        self.beta_max = float(beta_max)

    def run(self, env: Environment, budget: int, rng: np.random.Generator, acc: ComputeAccountant) -> AlgoResult:
        env0 = env.clone()
        m = env0.n_threads()
        # Each individual keeps its own cloned environment; this is simplified but effective for synthetic
        envs = [env0.clone() for _ in range(self.population)]
        scores = np.zeros(self.population, dtype=float)

        b_per_epoch = max(1, budget // max(1, self.epochs))
        step_count = 0
        for e in range(self.epochs):
            beta = (e + 1) / self.epochs * self.beta_max
            # mutate: extend each env by a few steps
            for i in range(self.population):
                steps = min(b_per_epoch, budget - step_count)
                for _ in range(steps):
                    with overhead_timer(acc):
                        t = int(rng.integers(0, m))
                        acc.count_decision(1)
                    acc.charge("generation", 1)
                    envs[i].step_extend_thread(t, rng)
                    step_count += 1
                    if step_count >= budget:
                        break
                scores[i] = envs[i].utility()
            if step_count >= budget:
                break
            # reweight + resample
            with overhead_timer(acc):
                w = np.exp(beta * (scores - scores.max()))
                if w.sum() <= 0:
                    break
                probs = w / w.sum()
                idx = rng.choice(self.population, size=self.population, replace=True, p=probs)
                envs = [envs[j].clone() for j in idx]
                scores = scores[idx]

        best = int(np.argmax(scores))
        return AlgoResult(envs[best].utility(), {"best_idx": best}, acc.snapshot())
