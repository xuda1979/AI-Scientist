# (existing imports)
import os
import warnings
try:
    import torch
except Exception:
    torch = None
try:
    from poker_ai.utils.seed import seed_everything
except Exception:
    seed_everything = None

def main():
    # Respect deterministic runs if env var set
    base_seed = int(os.environ.get("POKER_AI_SEED", "0"))
    if base_seed and seed_everything is not None:
        seed_everything(base_seed)
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
