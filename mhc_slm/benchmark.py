from __future__ import annotations

import time
import math
import argparse
from contextlib import nullcontext
import copy

import torch
from tqdm import tqdm

from .data import load_wikitext2_tokens, make_dataloaders
from .model import ModelConfig, BaselineSLM, MHCSLM


def build_amp(device: torch.device, requested_amp: bool):
    if requested_amp and device.type != "cuda":
        print("[warn] --amp was set but CUDA is not available. AMP disabled.")
        requested_amp = False

    amp_enabled = bool(requested_amp and device.type == "cuda")
    if not amp_enabled:
        return (lambda: nullcontext()), None, False

    try:
        from torch.amp import autocast, GradScaler
        autocast_ctx = lambda: autocast(device_type="cuda", enabled=True)
        scaler = GradScaler("cuda", enabled=True)
        return autocast_ctx, scaler, True
    except Exception:
        from torch.cuda.amp import autocast, GradScaler
        autocast_ctx = lambda: autocast(enabled=True)
        scaler = GradScaler(enabled=True)
        return autocast_ctx, scaler, True


def _torch_load_safe(path: str, device: torch.device):
    """
    Prefer weights_only=True if supported by this torch version to avoid pickle risks.
    Falls back to default torch.load signature if not supported.
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_checkpoint_if_any(model, ckpt_path: str | None, device: torch.device):
    if not ckpt_path:
        return None
    ckpt = _torch_load_safe(ckpt_path, device=device)
    if "model" not in ckpt:
        raise KeyError("Checkpoint missing key 'model'.")
    model.load_state_dict(ckpt["model"], strict=True)
    return ckpt


@torch.inference_mode()
def eval_ppl(model, val_loader, device, autocast_ctx, amp_enabled: bool, max_batches: int | None, is_mhc: bool):
    model.eval()
    losses = []

    for i, (x, y) in enumerate(tqdm(val_loader, desc="ppl-eval", leave=False)):
        if max_batches is not None and i >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if amp_enabled:
            with autocast_ctx():
                out = model(x, y, return_aux=False) if is_mhc else model(x, y)
        else:
            out = model(x, y, return_aux=False) if is_mhc else model(x, y)

        loss = out[1]
        losses.append(float(loss.item()))

    mean_loss = sum(losses) / max(1, len(losses))
    ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    return mean_loss, ppl


def benchmark_steps(
    model,
    device,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    steps: int,
    warmup: int,
    autocast_ctx,
    scaler,
    amp_enabled: bool,
    is_mhc: bool,
):
    """
    NOTE: This function performs optimizer steps on synthetic random data
    to include backward+step in the timing. This *will* change weights.

    Caller must restore weights if they also want to run perplexity eval
    on the original checkpoint.
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    # Warmup steps
    for _ in range(warmup):
        opt.zero_grad(set_to_none=True)
        if amp_enabled:
            with autocast_ctx():
                out = model(x, y, return_aux=False) if is_mhc else model(x, y)
                loss = out[1]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            out = model(x, y, return_aux=False) if is_mhc else model(x, y)
            loss = out[1]
            loss.backward()
            opt.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed steps
    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        if amp_enabled:
            with autocast_ctx():
                out = model(x, y, return_aux=False) if is_mhc else model(x, y)
                loss = out[1]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            out = model(x, y, return_aux=False) if is_mhc else model(x, y)
            loss = out[1]
            loss.backward()
            opt.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    total_time = t1 - t0
    tok = batch_size * seq_len * steps
    toks_per_sec = tok / max(1e-9, total_time)

    peak_mem = None
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)

    return {
        "total_time_sec": total_time,
        "tokens_total": tok,
        "tokens_per_sec": toks_per_sec,
        "peak_mem_mb": peak_mem,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="mhc", choices=["baseline", "mhc"])

    ap.add_argument("--n_streams", type=int, default=4)
    ap.add_argument("--sinkhorn_iters", type=int, default=8)
    ap.add_argument("--sinkhorn_temp", type=float, default=1.0)

    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--conv_kernel", type=int, default=3)
    ap.add_argument("--norm", type=str, default="rms", choices=["rms", "ln"])

    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)

    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--ppl_eval", action="store_true")
    ap.add_argument("--ppl_batches", type=int, default=50)

    ap.add_argument("--ckpt", type=str, default="", help="Path to a saved checkpoint")

    ap.add_argument("--tokenizer", type=str, default="gpt2")
    ap.add_argument("--num_workers", type=int, default=0)

    # adapter flags so architecture matches ckpt
    ap.add_argument("--stream_adapters", action="store_true")
    ap.add_argument("--adapter_rank", type=int, default=64)
    ap.add_argument("--adapter_dropout", type=float, default=0.0)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")

    autocast_ctx, scaler, amp_enabled = build_amp(device, args.amp)
    print(f"[info] AMP enabled: {amp_enabled}")

    tok, train_tokens, val_tokens = load_wikitext2_tokens(args.tokenizer)
    _, val_loader = make_dataloaders(
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        dim=args.dim,
        n_layers=args.layers,
        dropout=args.dropout,
        conv_kernel=args.conv_kernel,
        norm_type=args.norm,
        n_streams=args.n_streams,
        sinkhorn_iters=args.sinkhorn_iters,
        sinkhorn_temp=args.sinkhorn_temp,
        adaptive_routing=False,
        stream_adapters=args.stream_adapters,
        adapter_rank=args.adapter_rank,
        adapter_dropout=args.adapter_dropout,
    )

    is_mhc = (args.model == "mhc")
    if args.model == "baseline":
        model = BaselineSLM(cfg, max_seq_len=args.seq_len)
    else:
        model = MHCSLM(cfg, max_seq_len=args.seq_len)

    model.to(device)

    ckpt_path = args.ckpt.strip()
    if ckpt_path:
        ckpt = load_checkpoint_if_any(model, ckpt_path, device)
        print(f"[info] Loaded checkpoint: {ckpt_path} (step={ckpt.get('step', 'NA')})")
    else:
        ckpt = None

    # IMPORTANT: benchmark_steps modifies weights. Save/restore so ppl_eval is correct.
    state_before = None
    if args.ppl_eval:
        # deep copy is safest; benchmark modifies weights in-place
        state_before = copy.deepcopy(model.state_dict())

    stats = benchmark_steps(
        model=model,
        device=device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=cfg.vocab_size,
        steps=args.steps,
        warmup=args.warmup,
        autocast_ctx=autocast_ctx,
        scaler=scaler,
        amp_enabled=amp_enabled,
        is_mhc=is_mhc,
    )

    # Restore checkpoint weights before ppl eval
    if args.ppl_eval and state_before is not None:
        model.load_state_dict(state_before, strict=True)

    print("\n=== Benchmark Results ===")
    print(f"model: {args.model} (stream_adapters={args.stream_adapters})")
    print(f"batch_size: {args.batch_size} seq_len: {args.seq_len}")
    print(f"steps: {args.steps} warmup: {args.warmup} amp: {amp_enabled}")
    print(f"total_time_sec: {stats['total_time_sec']:.3f}")
    print(f"tokens_per_sec: {stats['tokens_per_sec']:.2f}")
    if stats["peak_mem_mb"] is not None:
        print(f"peak_mem_mb: {stats['peak_mem_mb']:.2f}")

    if args.ppl_eval:
        if not ckpt_path:
            print("\n[warn] --ppl_eval without --ckpt evaluates RANDOM weights. Provide --ckpt for real ppl.")
        val_loss, ppl = eval_ppl(
            model, val_loader, device, autocast_ctx, amp_enabled, args.ppl_batches, is_mhc=is_mhc
        )
        print("\n=== Perplexity (quick) ===")
        print(f"val_loss: {val_loss:.4f}")
        print(f"ppl: {ppl:.2f}")


if __name__ == "__main__":
    main()
