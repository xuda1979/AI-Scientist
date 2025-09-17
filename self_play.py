# (existing imports)
import logging
import os
import warnings

logger = logging.getLogger(__name__)

try:
    from poker_ai.utils.seed import seed_everything, forked_worker_seed
except ImportError as exc:
    warnings.warn(
        f"Unable to import poker_ai.utils.seed utilities: {exc}; seeding will be disabled.",
        stacklevel=2,
    )
    seed_everything = None
    forked_worker_seed = None
except Exception:
    logger.exception("Unexpected error while importing poker_ai.utils.seed utilities")
    raise


def _maybe_seed() -> None:
    base = int(os.environ.get("POKER_AI_SEED", "0"))
    if not base:
        return
    if seed_everything is None:
        warnings.warn(
            "POKER_AI_SEED is set but seed_everything is unavailable; skipping global seeding.",
            stacklevel=2,
        )
        return
    seed_everything(base)


def _worker_init(rank: int) -> None:
    base = int(os.environ.get("POKER_AI_SEED", "0"))
    if not base:
        return

    missing = []
    if forked_worker_seed is None:
        missing.append("forked_worker_seed")
    if seed_everything is None:
        missing.append("seed_everything")

    if missing:
        warnings.warn(
            "POKER_AI_SEED is set but the following seeding utilities are unavailable: "
            f"{', '.join(missing)}; skipping worker seeding.",
            stacklevel=2,
        )
        return

    s = forked_worker_seed(base, rank)
    seed_everything(s)


def main() -> None:
    _maybe_seed()
    # When launching parallel self-play, ensure each worker calls:
    # _worker_init(rank)
    # (hook this into your mp.Process / DataLoader num_workers init_fn)
    # ... existing logic ...


if __name__ == "__main__":
    main()
