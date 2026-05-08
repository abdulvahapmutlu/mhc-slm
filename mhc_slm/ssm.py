from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class DiagonalSSM(nn.Module):
    """
    Stable diagonal state-space layer (scan recurrence):

      s_t = a * s_{t-1} + b * u_t
      y_t = c * s_t     + d * u_t

    Key stability choices:
      - a in (0,1) via sigmoid, and clamped away from 0/1
      - run recurrence in float32 for numerical safety (even under amp)
      - no exp(a^{-t}) / closed-form that can overflow
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # a in (0,1)
        self.logit_a = nn.Parameter(torch.randn(dim) * 0.02)
        self.b = nn.Parameter(torch.randn(dim) * 0.02)
        self.c = nn.Parameter(torch.randn(dim) * 0.02)
        self.d = nn.Parameter(torch.zeros(dim))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (B, T, D)
        returns y: (B, T, D)
        """
        B, T, D = u.shape
        assert D == self.dim

        # Compute recurrence in float32 for stability
        u32 = u.float()

        a = torch.sigmoid(self.logit_a.float()).clamp(1e-4, 1.0 - 1e-4)  # (D,)
        b = self.b.float()
        c = self.c.float()
        d = self.d.float()

        s = torch.zeros(B, D, device=u.device, dtype=torch.float32)
        y = torch.empty(B, T, D, device=u.device, dtype=torch.float32)

        # scan over time (stable)
        for t in range(T):
            ut = u32[:, t, :]
            s = a * s + b * ut
            y[:, t, :] = c * s + d * ut

        return y.to(dtype=u.dtype)


class SimpleSSMBlock(nn.Module):
    """
    Practical SSM block (SLM unit):
      x -> norm -> in_proj -> (u, gate)
      u -> causal depthwise conv -> silu
      u -> diagonal SSM (stable scan) -> y
      y -> sigmoid(gate) * y -> out_proj
    """
    def __init__(
        self,
        dim: int,
        conv_kernel: int = 3,
        dropout: float = 0.0,
        norm_type: str = "rms",
    ):
        super().__init__()
        self.dim = dim
        self.dropout = dropout

        if norm_type == "rms":
            self.norm = RMSNorm(dim)
        elif norm_type == "ln":
            self.norm = nn.LayerNorm(dim)
        else:
            raise ValueError("norm_type must be 'rms' or 'ln'")

        self.in_proj = nn.Linear(dim, 2 * dim, bias=True)

        # causal depthwise conv: pad left (k-1)
        self.conv_kernel = conv_kernel
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=conv_kernel, padding=conv_kernel - 1, groups=dim, bias=True
        )

        self.ssm = DiagonalSSM(dim)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,D)
        returns: (B,T,D)
        """
        x = self.norm(x)
        u, gate = self.in_proj(x).chunk(2, dim=-1)  # (B,T,D), (B,T,D)

        # depthwise conv over time (causal)
        u_t = u.transpose(1, 2)  # (B,D,T)
        u_t = self.dwconv(u_t)
        u_t = u_t[..., : u.shape[1]]  # crop back to T
        u = u_t.transpose(1, 2)  # (B,T,D)

        u = F.silu(u)

        y = self.ssm(u)  # (B,T,D)
        y = y * torch.sigmoid(gate)

        y = self.out_proj(y)
        if self.dropout > 0:
            y = F.dropout(y, p=self.dropout, training=self.training)
        return y
