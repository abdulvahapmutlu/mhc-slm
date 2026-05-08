from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sinkhorn import sinkhorn_knopp, simplex_weights
from .ssm import SimpleSSMBlock, RMSNorm


class BaselineSLMBlock(nn.Module):
    """
    Standard residual SLM block:
      x_{l+1} = x_l + SSM(Norm(x_l))
    """
    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        conv_kernel: int = 3,
        norm_type: str = "rms",
    ):
        super().__init__()
        self.ssm = SimpleSSMBlock(
            dim=dim,
            conv_kernel=conv_kernel,
            dropout=dropout,
            norm_type=norm_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(x)


class MHCSLMBlock(nn.Module):
    """
    mHC-SSM block operating on n streams.

    NEW: stream-specialized adapters (SSA)
      - pre-adapter: X <- X + Adapter_pre(X)   (per-stream FiLM in bottleneck)
      - post-adapter: y_i <- y + Adapter_post_i(y) before scattering

    No state-adaptive routing here.
    """
    def __init__(
        self,
        dim: int,
        n_streams: int,
        sinkhorn_iters: int = 8,
        sinkhorn_temp: float = 1.0,
        dropout: float = 0.0,
        conv_kernel: int = 3,
        norm_type: str = "rms",
        adaptive_routing: bool = False,  # kept for compatibility, but not used
        route_hidden: int = 256,
        route_from: str = "u",
        # NEW:
        stream_adapters: bool = False,
        adapter_rank: int = 64,
        adapter_dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.n = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_temp = sinkhorn_temp

        # Stream mixing weights for pre/post. Use logits -> softmax => simplex weights.
        self.hpre_logits = nn.Parameter(torch.zeros(n_streams))
        self.hpost_logits = nn.Parameter(torch.zeros(n_streams))

        # Residual mixing logits (static)
        self.hres_logits = nn.Parameter(torch.randn(n_streams, n_streams) * 0.02)

        # Core SSM unit computes on (B,T,d)
        self.ssm = SimpleSSMBlock(
            dim=dim,
            conv_kernel=conv_kernel,
            dropout=dropout,
            norm_type=norm_type,
        )

        # ---- NEW: Stream-specialized adapters ----
        self.stream_adapters = bool(stream_adapters)
        self.adapter_rank = int(adapter_rank)
        self.adapter_dropout = float(adapter_dropout)

        if self.stream_adapters:
            r = self.adapter_rank
            # shared weights
            self.ad_down = nn.Linear(dim, r, bias=True)
            self.ad_up = nn.Linear(r, dim, bias=True)

            # per-stream FiLM scaling in bottleneck (pre and post separately)
            self.gamma_pre = nn.Parameter(torch.ones(n_streams, r))
            self.gamma_post = nn.Parameter(torch.ones(n_streams, r))

            self.ad_act = nn.SiLU()
            self.ad_drop = nn.Dropout(self.adapter_dropout)

            self.ad_norm = RMSNorm(dim) if norm_type == "rms" else nn.LayerNorm(dim)
        else:
            self.ad_down = None
            self.ad_up = None
            self.gamma_pre = None
            self.gamma_post = None
            self.ad_act = None
            self.ad_drop = None
            self.ad_norm = None

    def _compute_pre_post(self, device: torch.device, dtype: torch.dtype):
        w_pre = simplex_weights(self.hpre_logits.to(device=device, dtype=dtype), dim=-1)   # (n,)
        w_post = simplex_weights(self.hpost_logits.to(device=device, dtype=dtype), dim=-1) # (n,)
        return w_pre, w_post

    def _static_hres(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        logits = self.hres_logits.to(device=device, dtype=dtype)  # (n,n)
        H = sinkhorn_knopp(logits, iters=self.sinkhorn_iters, temperature=self.sinkhorn_temp)
        return H  # (n,n)

    def _apply_pre_adapter(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B,T,n,d)
        Xn = self.ad_norm(X)  # stable
        z = self.ad_down(Xn)  # (B,T,n,r)
        z = self.ad_act(z)
        z = z * self.gamma_pre.view(1, 1, self.n, self.adapter_rank)
        dz = self.ad_up(z)    # (B,T,n,d)
        dz = self.ad_drop(dz)
        return X + dz

    def _post_adapter_on_y(self, y: torch.Tensor) -> torch.Tensor:
        # y: (B,T,d)  -> returns (B,T,n,d)  with stream-specific adapter added
        yn = self.ad_norm(y)          # (B,T,d)
        y_r = self.ad_down(yn)        # (B,T,r)
        y_r = self.ad_act(y_r)        # (B,T,r)
        z = y_r.unsqueeze(2) * self.gamma_post.view(1, 1, self.n, self.adapter_rank)  # (B,T,n,r)
        dy = self.ad_up(z)            # (B,T,n,d)
        dy = self.ad_drop(dy)
        return y.unsqueeze(2) + dy    # (B,T,n,d)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B,T,n,d)
        returns: (B,T,n,d)
        """
        B, T, n, d = X.shape
        assert n == self.n and d == self.dim

        # NEW: stream-specialized pre-adapter
        if self.stream_adapters:
            X = self._apply_pre_adapter(X)

        w_pre, w_post = self._compute_pre_post(X.device, X.dtype)  # (n,), (n,)

        # Pre-mix to single stream: u[b,t,d] = sum_i w_pre[i] * X[b,t,i,d]
        u = torch.einsum("i,btid->btd", w_pre, X)

        # Core SSM
        y = self.ssm(u)  # (B,T,D)

        # NEW: stream-specialized post-adapter on y
        if self.stream_adapters:
            y_stream = self._post_adapter_on_y(y)  # (B,T,n,d)
        else:
            y_stream = y.unsqueeze(2)              # (B,T,1,d) broadcast ok

        # Post-mix scatter back to n streams: dX[b,t,i,d] = w_post[i] * y_stream[b,t,i,d]
        dX = y_stream * w_post.view(1, 1, n, 1)

        # Residual stream mixing (static mHC)
        H = self._static_hres(X.device, X.dtype)  # (n,n)
        X_mixed = torch.einsum("ij,btjd->btid", H, X)

        return X_mixed + dX
