# (existing imports)
import logging
import os
import warnings

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None
    warnings.warn("PyTorch is not installed; torch-specific features are disabled.", stacklevel=2)
except Exception:
    logger.exception("Unexpected error while importing torch")
    raise

try:
    from poker_ai.utils.seed import seed_everything
except ImportError as exc:
    warnings.warn(
        f"Unable to import poker_ai.utils.seed.seed_everything: {exc}; seeding will be disabled.",
        stacklevel=2,
    )
    seed_everything = None
except Exception:
    logger.exception("Unexpected error while importing poker_ai.utils.seed")
    raise


def main() -> None:
    # Respect deterministic runs if env var set
    base_seed = int(os.environ.get("POKER_AI_SEED", "0"))
    if base_seed:
        if seed_everything is not None:
            seed_everything(base_seed)
        else:
            warnings.warn(
                "POKER_AI_SEED is set but seed_everything is unavailable; skipping global seeding.",
                stacklevel=2,
            )
    # Optional: torch.compile for speed on PyTorch 2+
    if torch is not None and hasattr(torch, "compile"):
        try:
            # ensure your model creation path checks for this flag
            os.environ.setdefault("POKER_AI_USE_TORCH_COMPILE", "1")
        except Exception as e:
            warnings.warn(f"torch.compile not enabled: {e}")
    # ... existing CLI + training startup ...


if __name__ == "__main__":
    main()
