from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer


class PackedLMDataset(Dataset):
    def __init__(self, tokens: np.ndarray, seq_len: int):
        super().__init__()
        self.tokens = tokens.astype(np.int64)
        self.seq_len = seq_len
        self.n = (len(self.tokens) - 1) // seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, i: int):
        start = i * self.seq_len
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + 1 + self.seq_len]
        return torch.from_numpy(x), torch.from_numpy(y)


def build_tokenizer(tokenizer_name: str = "gpt2"):
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def tokenize_texts(tokenizer, texts: list[str]) -> np.ndarray:
    ids = []
    for t in texts:
        out = tokenizer(t, return_tensors=None, add_special_tokens=False)
        ids.extend(out["input_ids"])
        ids.append(tokenizer.eos_token_id)
    return np.array(ids, dtype=np.int64)


def load_wikitext2_tokens(tokenizer_name: str):
    tok = build_tokenizer(tokenizer_name)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_tokens = tokenize_texts(tok, ds["train"]["text"])
    val_tokens = tokenize_texts(tok, ds["validation"]["text"])
    return tok, train_tokens, val_tokens


def make_dataloaders(
    train_tokens: np.ndarray,
    val_tokens: np.ndarray,
    seq_len: int,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    train_ds = PackedLMDataset(train_tokens, seq_len)
    val_ds = PackedLMDataset(val_tokens, seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
