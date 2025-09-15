# (existing imports)
import os
from typing import Optional
try:
    from poker_ai.utils.seed import seed_everything, forked_worker_seed
except Exception:
    seed_everything = None
    forked_worker_seed = None

def _maybe_seed():
    base = int(os.environ.get("POKER_AI_SEED", "0"))
    if base and seed_everything is not None:
        seed_everything(base)

def _worker_init(rank: int):
    if forked_worker_seed is not None:
        s = forked_worker_seed(int(os.environ.get("POKER_AI_SEED", "0")), rank)
        seed_everything(s)

def main():
    _maybe_seed()
    # When launching parallel self-play, ensure each worker calls:
    # _worker_init(rank)
    # (hook this into your mp.Process / DataLoader num_workers init_fn)
    # ... existing logic ...

if __name__ == "__main__":
    main()
