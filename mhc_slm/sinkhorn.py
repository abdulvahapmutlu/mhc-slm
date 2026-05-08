from __future__ import annotations

import torch
import torch.nn.functional as F


def _safe_exp(logits: torch.Tensor) -> torch.Tensor:
    # Stable exp: exp(x - max(x)) to avoid overflow
    logits = logits - logits.amax(dim=-1, keepdim=True)
    return torch.exp(logits)


@torch.no_grad()
def check_doubly_stochastic(H: torch.Tensor, atol: float = 1e-3) -> dict:
    """
    Diagnostics helper. Works for:
      - H: (n, n)
      - H: (..., n, n)
    """
    row_sum = H.sum(dim=-1)
    col_sum = H.sum(dim=-2)
    return {
        "row_sum_mean_abs_err": (row_sum - 1.0).abs().mean().item(),
        "col_sum_mean_abs_err": (col_sum - 1.0).abs().mean().item(),
        "min_entry": H.min().item(),
        "max_entry": H.max().item(),
        "row_sum_max_abs_err": (row_sum - 1.0).abs().max().item(),
        "col_sum_max_abs_err": (col_sum - 1.0).abs().max().item(),
        "ok": bool(
            (row_sum - 1.0).abs().max().item() < atol
            and (col_sum - 1.0).abs().max().item() < atol
            and H.min().item() >= -atol
        ),
    }


def sinkhorn_knopp(
    logits: torch.Tensor,
    iters: int = 8,
    eps: float = 1e-8,
    temperature: float = 1.0,
    make_doubly_stochastic: bool = True,
) -> torch.Tensor:
    """
    Sinkhorn–Knopp projection to (approximately) doubly-stochastic matrices.

    Supports:
      - logits: (n, n)
      - logits: (..., n, n) (e.g. (B,T,n,n) for token-adaptive routing)

    Returns:
      H: same shape, non-negative, rows/cols sum to ~1.

    Notes:
      - We apply exp(logits / temperature) then alternate row/col normalization.
      - For performance: keep iters small (e.g., 4..10).
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    # Positive matrix
    P = _safe_exp(logits / temperature)

    if not make_doubly_stochastic:
        # If you only want row-stochastic, just normalize rows.
        return P / (P.sum(dim=-1, keepdim=True) + eps)

    for _ in range(iters):
        # Row normalize
        P = P / (P.sum(dim=-1, keepdim=True) + eps)
        # Col normalize
        P = P / (P.sum(dim=-2, keepdim=True) + eps)

    return P


def simplex_weights(
    logits: torch.Tensor,
    dim: int = -1,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Non-negative weights summing to 1: softmax(logits / temperature)
    """
    return F.softmax(logits / temperature, dim=dim)
