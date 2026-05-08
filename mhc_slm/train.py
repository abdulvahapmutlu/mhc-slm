from __future__ import annotations

import os
import time
import math
import argparse
from dataclasses import asdict
from contextlib import nullcontext

import torch
from torch.optim import AdamW
from tqdm import tqdm

from .data import load_wikitext2_tokens, make_dataloaders
from .model import ModelConfig, BaselineSLM, MHCSLM


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


@torch.no_grad()
def evaluate(model, val_loader, device, autocast_ctx, amp_enabled: bool, is_mhc: bool):
    model.eval()
    losses = []

    for x, y in tqdm(val_loader, desc="eval", leave=False):
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


def save_ckpt(path: str, arch: str, model, opt, scaler, step: int, cfg: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "arch": arch,
            "step": step,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "cfg": cfg,
        },
        path,
    )


def auto_run_dir(args) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.model == "baseline":
        name = f"baseline_d{args.dim}_L{args.layers}_T{args.seq_len}_B{args.batch_size}_{ts}"
    else:
        extra = []
        if args.stream_adapters:
            extra.append(f"ad{args.adapter_rank}")
        if args.stream_diversity_coeff > 0:
            extra.append(f"div{args.stream_diversity_coeff:g}")
        extra_s = ("_" + "_".join(extra)) if extra else ""
        name = (
            f"mhc_static_d{args.dim}_L{args.layers}_T{args.seq_len}_B{args.batch_size}"
            f"_n{args.n_streams}_sk{args.sinkhorn_iters}{extra_s}_{ts}"
        )
    if args.run_name:
        name = f"{args.run_name}_{ts}"
    return os.path.join(args.out_dir, name)


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

    ap.add_argument("--epochs", type=int, default=10)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=1000)

    ap.add_argument("--out_dir", type=str, default="runs/mhc_slm")
    ap.add_argument("--run_name", type=str, default="")

    ap.add_argument("--tokenizer", type=str, default="gpt2")
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--nan_guard", action="store_true")

    # ---- NEW flags ----
    ap.add_argument("--stream_adapters", action="store_true")
    ap.add_argument("--adapter_rank", type=int, default=64)
    ap.add_argument("--adapter_dropout", type=float, default=0.1)

    ap.add_argument("--stream_diversity_coeff", type=float, default=0.0)

    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")

    autocast_ctx, scaler, amp_enabled = build_amp(device, args.amp)
    print(f"[info] AMP enabled: {amp_enabled}")

    tok, train_tokens, val_tokens = load_wikitext2_tokens(args.tokenizer)

    pin_memory = (device.type == "cuda")
    train_loader, val_loader = make_dataloaders(
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
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

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_dir = auto_run_dir(args)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[info] Run dir: {run_dir}")

    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"train epoch {epoch+1}/{args.epochs}")

        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            t0 = time.time()

            need_aux = bool(is_mhc and args.stream_diversity_coeff > 0)

            if amp_enabled:
                with autocast_ctx():
                    if need_aux:
                        logits, loss, aux = model(x, y, return_aux=True)
                        div = aux["stream_diversity"]
                        loss_total = loss + (args.stream_diversity_coeff * div)
                    else:
                        logits, loss = model(x, y, return_aux=False) if is_mhc else model(x, y)
                        loss_total = loss

                scaler.scale(loss_total).backward()

                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(opt)
                scaler.update()
            else:
                if need_aux:
                    logits, loss, aux = model(x, y, return_aux=True)
                    div = aux["stream_diversity"]
                    loss_total = loss + (args.stream_diversity_coeff * div)
                else:
                    logits, loss = model(x, y, return_aux=False) if is_mhc else model(x, y)
                    loss_total = loss

                loss_total.backward()

                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                opt.step()

            dt = time.time() - t0
            global_step += 1

            if args.nan_guard and (not torch.isfinite(loss_total)):
                print("\n[error] Loss became NaN/Inf. Stopping for debugging.")
                print(f"step={global_step} loss_total={float(loss_total.item())} base_loss={float(loss.item())}")
                return

            postfix = {"loss": float(loss.item()), "step": global_step, "sec/step": f"{dt:.3f}"}
            if need_aux:
                postfix["div"] = f"{float(div.detach().item()):.4f}"
            pbar.set_postfix(**postfix)

            if global_step % args.eval_every == 0:
                val_loss, ppl = evaluate(model, val_loader, device, autocast_ctx, amp_enabled, is_mhc=is_mhc)
                print(f"\n[eval] step={global_step} val_loss={val_loss:.4f} ppl={ppl:.2f}\n")
                model.train()

            if global_step % args.save_every == 0:
                ckpt_path = os.path.join(run_dir, f"{args.model}_step{global_step}.pt")
                save_ckpt(ckpt_path, args.model, model, opt, scaler, global_step, cfg=asdict(cfg))

    val_loss, ppl = evaluate(model, val_loader, device, autocast_ctx, amp_enabled, is_mhc=is_mhc)
    print(f"[final] val_loss={val_loss:.4f} ppl={ppl:.2f}")

    ckpt_path = os.path.join(run_dir, f"{args.model}_final.pt")
    save_ckpt(ckpt_path, args.model, model, opt, scaler, global_step, cfg=asdict(cfg))
    print(f"[info] Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
