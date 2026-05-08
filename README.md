
# mHC-SSM: Manifold-Constrained Hyper-Connections for State Space Language Models

PyTorch research implementation of **mHC-SSM**, a State Space Language Model architecture that adapts **Manifold-Constrained Hyper-Connections (mHC)** to SSM-based language modeling and extends it with **Stream-Specialized Adapters**.

This project investigates whether stability-constrained multi-stream residual mixing can improve lightweight SSM language models, and whether adding compact stream-specific adapter capacity can further improve validation perplexity without modifying the SSM recurrence itself.

---

## Overview

Standard residual SSM language models use a single residual stream:

```
x_{l+1} = x_l + SSM(x_l)
````

mHC-SSM expands the residual stream into multiple parallel streams:

```
X_l ∈ R^{B × T × n × d}
```

Each mHC-SSM block performs:

1. Multi-stream residual representation
2. Simplex-constrained pre-mixing into a single SSM input
3. Lightweight diagonal SSM computation
4. Simplex-constrained post-mixing back into streams
5. Sinkhorn-projected residual stream mixing
6. Optional stream-specialized adapters before and after the SSM core

The best-performing variant in this repository is:

```
mHC-SSM + Stream-Specialized Adapters
```

---

## Main Contributions

This repository provides:

* A baseline single-stream diagonal SSM language model.
* A static mHC-SSM architecture with Sinkhorn-constrained multi-stream residual mixing.
* A stream-specialized adapter extension using shared bottleneck projections and per-stream scaling.
* Training and benchmarking scripts for WikiText-2.
* Checkpoint-restored benchmarking to fairly measure perplexity after throughput timing.
* Reproducible commands for baseline, static mHC-SSM, and mHC-SSM with adapters.

---

## Architecture

```
tokens
  │
  ▼
Embedding + Positional Encoding
  │
  ▼
Stream Expander
  │
  ▼
Stack of mHC-SSM Blocks
  │
  ├── optional stream-specialized pre-adapter
  ├── simplex pre-mixing
  ├── SSM core
  ├── optional stream-specialized post-adapter
  ├── simplex post-mixing
  └── Sinkhorn-projected residual stream mixing
  │
  ▼
Stream Aggregator
  │
  ▼
Final Norm + Tied LM Head
  │
  ▼
Next-token logits
```

---

## Installation

Create a virtual environment:

```
python -m venv .venv
```

Activate it.

Windows PowerShell:

```
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```
source .venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Requirements

```
torch>=2.1
transformers>=4.41
datasets>=2.19
tqdm>=4.66
einops>=0.8
numpy>=1.26
```

CUDA is recommended. The experiments reported here were run with CUDA AMP enabled.

---

## Dataset

The experiments use **WikiText-2** through the Hugging Face `datasets` library.

Tokenization is performed with the GPT-2 tokenizer from Hugging Face Transformers.

The data loader packs the tokenized corpus into fixed-length autoregressive language modeling sequences:

```
input  = tokens[t : t + seq_len]
target = tokens[t + 1 : t + 1 + seq_len]
```

---

## Training Setup

All reported models use the same core setup:

| Setting             |           Value |
| ------------------- | --------------: |
| Dataset             |      WikiText-2 |
| Tokenizer           | GPT-2 tokenizer |
| Epochs              |              10 |
| Sequence length     |             256 |
| Batch size          |              16 |
| Layers              |               8 |
| Hidden dimension    |             512 |
| Number of streams   |               4 |
| Sinkhorn iterations |               8 |
| Optimizer           |           AdamW |
| Learning rate       |            3e-4 |
| Weight decay        |             0.1 |
| Gradient clipping   |             1.0 |
| AMP                 | Enabled on CUDA |

---

## Training Commands

### 1. Baseline SSM

```
python -m mhc_slm.train `
  --model baseline `
  --epochs 10 `
  --seq_len 256 `
  --batch_size 16 `
  --layers 8 `
  --dim 512 `
  --amp
```

---

### 2. Static mHC-SSM

```
python -m mhc_slm.train `
  --model mhc `
  --epochs 10 `
  --seq_len 256 `
  --batch_size 16 `
  --layers 8 `
  --dim 512 `
  --n_streams 4 `
  --sinkhorn_iters 8 `
  --amp
```

---

### 3. mHC-SSM + Stream-Specialized Adapters

```
python -m mhc_slm.train `
  --model mhc `
  --epochs 10 `
  --seq_len 256 `
  --batch_size 16 `
  --layers 8 `
  --dim 512 `
  --n_streams 4 `
  --sinkhorn_iters 8 `
  --stream_adapters `
  --adapter_rank 64 `
  --adapter_dropout 0.1 `
  --amp
```

---

## Benchmarking

The benchmark script measures:

* Training throughput in tokens/sec
* Peak CUDA memory
* Validation loss
* Perplexity

Important: throughput timing uses synthetic optimizer steps, which normally modify model weights. The benchmark script saves and restores the checkpoint weights before perplexity evaluation, so the reported validation loss and perplexity are computed fairly from the original checkpoint.

---

### Baseline Benchmark

```
python -m mhc_slm.benchmark `
  --model baseline `
  --seq_len 256 `
  --batch_size 8 `
  --steps 50 `
  --warmup 10 `
  --amp `
  --ppl_eval `
  --ppl_batches 62 `
  --ckpt "runs\mhc_slm\baseline_...\baseline_final.pt"
```

---

### Static mHC-SSM Benchmark

```
python -m mhc_slm.benchmark `
  --model mhc `
  --n_streams 4 `
  --sinkhorn_iters 8 `
  --seq_len 256 `
  --batch_size 8 `
  --steps 50 `
  --warmup 10 `
  --amp `
  --ppl_eval `
  --ppl_batches 62 `
  --ckpt "runs\mhc_slm\mhc_static_...\mhc_final.pt"
```

---

### mHC-SSM + Stream Adapters Benchmark

```
python -m mhc_slm.benchmark `
  --model mhc `
  --n_streams 4 `
  --sinkhorn_iters 8 `
  --seq_len 256 `
  --batch_size 8 `
  --steps 50 `
  --warmup 10 `
  --amp `
  --ppl_eval `
  --ppl_batches 62 `
  --stream_adapters `
  --adapter_rank 64 `
  --ckpt "runs\mhc_slm\mhc_static_..._ad64_...\mhc_final.pt"
```

---

## Results

### Final Checkpoint Benchmark

| Model                     | Validation Loss | Perplexity | Tokens/sec | Peak GPU Memory |
| ------------------------- | --------------: | ---------: | ---------: | --------------: |
| Baseline SSM              |          6.3507 |     572.91 |    1025.52 |         2365 MB |
| Static mHC-SSM            |          6.2448 |     515.35 |     964.81 |         2568 MB |
| mHC-SSM + Stream Adapters |      **6.1353** | **461.88** |     938.90 |         3092 MB |

---

## Result Interpretation

Static mHC-SSM improves over the baseline by introducing constrained multi-stream residual mixing.

Compared with the baseline:

```
Validation loss: 6.3507 → 6.2448
Perplexity:      572.91 → 515.35
```

Adding stream-specialized adapters improves the model further:

```
Validation loss: 6.2448 → 6.1353
Perplexity:      515.35 → 461.88
```

Overall, the adapter-augmented model improves perplexity from **572.91** to **461.88**, while reducing throughput from **1025.52** to **938.90 tokens/sec** and increasing peak memory from **2365 MB** to **3092 MB**.

This suggests that stream-specialized adapters provide a useful extension point for mHC-style residual topologies in SSM language models.

---

## License

This project is released under the MIT License.

---

## Acknowledgements

This project builds on ideas from residual learning, state space sequence models, Hyper-Connections, Manifold-Constrained Hyper-Connections, Sinkhorn normalization, and adapter-based parameter-efficient modeling.

