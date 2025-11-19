
from __future__ import annotations
import numpy as np
from typing import Dict, Any
from .base import Algorithm, AlgoResult, Environment, clip01
from ..evaluation.compute_accounting import ComputeAccountant, overhead_timer

class CoT(Algorithm):
    name = "cot"
    def run(self, env: Environment, budget: int, rng: np.random.Generator, acc: ComputeAccountant) -> AlgoResult:
        env = env.clone()
        # Single chain: extend the currently best-looking thread greedily by local expected gain
        # Here we just pick the best thread by a cheap probe at each step (synth env exposes a probe via step with rng)
        last_gain = 0.0
        for _ in range(budget):
            with overhead_timer(acc):
                # naive: pick thread uniformly at random (represents linear chain growth)
                thread = rng.integers(0, env.n_threads())
            # generation unit
            acc.charge("generation", 1)
            gain = env.step_extend_thread(int(thread), rng)
            last_gain = gain
        return AlgoResult(env.utility(), {"last_gain": last_gain}, acc.snapshot())

class SelfConsistency(Algorithm):
    name = "self_consistency"
    def __init__(self, chains: int = 5):
        self.chains = chains

    def run(self, env: Environment, budget: int, rng: np.random.Generator, acc: ComputeAccountant) -> AlgoResult:
        # Split budget across independent chains, vote by final utility
        chain_b = max(1, budget // self.chains)
        utils = []
        for _ in range(self.chains):
            env_i = env.clone()
            for _ in range(chain_b):
                with overhead_timer(acc):
                    t = rng.integers(0, env_i.n_threads())
                acc.charge("generation", 1)
                env_i.step_extend_thread(int(t), rng)
            utils.append(env_i.utility())
        # Vote: take mean as proxy for probability (soft vote)
        u = float(np.mean(utils))
        return AlgoResult(u, {"chain_utils": utils}, acc.snapshot())

class ToT_BFS(Algorithm):
    name = "tot_bfs"
    def __init__(self, breadth: int = 3):
        self.breadth = breadth

    def run(self, env: Environment, budget: int, rng: np.random.Generator, acc: ComputeAccountant) -> AlgoResult:
        env = env.clone()
        # BFS-like: cycle through threads deterministically for coverage
        order = list(range(env.n_threads()))
        idx = 0
        for _ in range(budget):
            with overhead_timer(acc):
                thread = order[idx % len(order)]
                idx += 1
            acc.charge("generation", 1)
            env.step_extend_thread(int(thread), rng)
        return AlgoResult(env.utility(), {"breadth": self.breadth}, acc.snapshot())
