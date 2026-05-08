from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import BaselineSLMBlock, MHCSLMBlock
from .ssm import RMSNorm


@dataclass
class ModelConfig:
    vocab_size: int
    dim: int = 512
    n_layers: int = 8
    dropout: float = 0.0
    conv_kernel: int = 3
    norm_type: str = "rms"

    # mHC
    n_streams: int = 4
    sinkhorn_iters: int = 8
    sinkhorn_temp: float = 1.0

    # kept for compatibility (we won't use adaptive routing)
    adaptive_routing: bool = False
    route_hidden: int = 256
    route_from: str = "u"

    # NEW: stream adapters
    stream_adapters: bool = False
    adapter_rank: int = 64
    adapter_dropout: float = 0.0


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.emb(idx)


class LearnedPosEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, dim: int):
        super().__init__()
        self.pos = nn.Embedding(max_seq_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        pos_ids = torch.arange(T, device=x.device)
        return x + self.pos(pos_ids)[None, :, :]


class BaselineSLM(nn.Module):
    def __init__(self, cfg: ModelConfig, max_seq_len: int):
        super().__init__()
        self.cfg = cfg
        self.max_seq_len = max_seq_len

        self.tok_emb = TokenEmbedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = LearnedPosEmbedding(max_seq_len, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            BaselineSLMBlock(
                dim=cfg.dim,
                dropout=cfg.dropout,
                conv_kernel=cfg.conv_kernel,
                norm_type=cfg.norm_type,
            )
            for _ in range(cfg.n_layers)
        ])

        self.norm_f = RMSNorm(cfg.dim) if cfg.norm_type == "rms" else nn.LayerNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.emb.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.tok_emb(idx)
        x = self.pos_emb(x)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss


class StreamExpander(nn.Module):
    def __init__(self, dim: int, n_streams: int):
        super().__init__()
        self.dim = dim
        self.n = n_streams
        self.proj = nn.Linear(dim, n_streams * dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        return self.proj(x).view(B, T, self.n, D)


class StreamAggregator(nn.Module):
    def __init__(self, n_streams: int):
        super().__init__()
        self.w_logits = nn.Parameter(torch.zeros(n_streams))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.w_logits, dim=-1).to(device=X.device, dtype=X.dtype)
        return torch.einsum("i,btid->btd", w, X)


def stream_diversity_metric(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    X: (B,T,n,D)
    Returns a scalar measuring off-diagonal cosine similarity between stream summaries.
    Lower is better (streams are more diverse).
    """
    # stream summaries: (n,D)
    S = X.mean(dim=(0, 1))  # (n,D)
    S = F.normalize(S.float(), dim=-1, eps=eps)
    G = S @ S.t()  # (n,n)

    n = G.size(0)
    off = G - torch.eye(n, device=G.device, dtype=G.dtype)
    # mean absolute off-diagonal similarity
    return off.abs().mean()


class MHCSLM(nn.Module):
    def __init__(self, cfg: ModelConfig, max_seq_len: int):
        super().__init__()
        self.cfg = cfg
        self.max_seq_len = max_seq_len

        self.tok_emb = TokenEmbedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = LearnedPosEmbedding(max_seq_len, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)

        self.expand = StreamExpander(cfg.dim, cfg.n_streams)

        self.blocks = nn.ModuleList([
            MHCSLMBlock(
                dim=cfg.dim,
                n_streams=cfg.n_streams,
                sinkhorn_iters=cfg.sinkhorn_iters,
                sinkhorn_temp=cfg.sinkhorn_temp,
                dropout=cfg.dropout,
                conv_kernel=cfg.conv_kernel,
                norm_type=cfg.norm_type,
                adaptive_routing=False,  # not used
                route_hidden=cfg.route_hidden,
                route_from=cfg.route_from,
                stream_adapters=cfg.stream_adapters,
                adapter_rank=cfg.adapter_rank,
                adapter_dropout=cfg.adapter_dropout,
            )
            for _ in range(cfg.n_layers)
        ])

        self.aggregate = StreamAggregator(cfg.n_streams)
        self.norm_f = RMSNorm(cfg.dim) if cfg.norm_type == "rms" else nn.LayerNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.emb.weight

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        x = self.tok_emb(idx)
        x = self.pos_emb(x)
        x = self.drop(x)

        X = self.expand(x)  # (B,T,n,D)

        for blk in self.blocks:
            X = blk(X)

        aux = {}
        if return_aux:
            aux["stream_diversity"] = stream_diversity_metric(X)

        x_out = self.aggregate(X)
        x_out = self.norm_f(x_out)
        logits = self.lm_head(x_out)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        if return_aux:
            return logits, loss, aux
        return logits, loss
