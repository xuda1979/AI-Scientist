from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerPolicyHead(nn.Module):
    """
    Drop-in policy head that masks illegal actions before softmax.
    Expects:
      - x: [B, H] encoded state
      - logits: produced by a preceding MLP from x
      - legal_mask: [B, A] with 1 for legal, 0 for illegal actions
    """
    def __init__(self, in_dim: int, num_actions: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(x)
        # avoid underflow/NaNs on fully-illegal edge cases by clamping mask
        mask = (legal_mask > 0).to(dtype=logits.dtype)
        # Set illegal logits to very negative
        masked_logits = logits + (mask - 1) * 1e9
        # If a row were all illegal (should not happen), fall back to uniform
        if (mask.sum(dim=-1) == 0).any():
            uniform = torch.full_like(masked_logits, fill_value=-1e2)
            masked_logits = torch.where(mask.bool(), masked_logits, uniform)
        return F.log_softmax(masked_logits, dim=-1)
