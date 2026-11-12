# Batch scaling of Qwen3-4B inference on a single RTX 3090

**Research question:** how does batch size trade off latency, throughput and memory during LLM
inference, and when does decoding become compute-bound instead of memory-bound? How does context
length move that boundary?

- **Model:** [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) in bf16. Weights, config and
  tokenizer come straight from the Hugging Face Hub.
- **Engine:** a minimal JAX implementation of the Qwen3 forward pass (~150 lines), checked against
  Hugging Face `transformers`.
- **Hardware:** one NVIDIA RTX 3090 (24 GB GDDR6X) on a Linux lab server, JAX on CUDA
  (`jax[cuda12]`).
- **Theory:** the [Scaling Book inference chapter](https://jax-ml.github.io/scaling-book/inference/)
  adapted to Qwen3-4B on a 3090. See the [technical report](report/report.md) for the derivation,
  hypotheses, results and figures.

## Repository layout

| Path | What |
|---|---|
| `notebooks/qwen3_batch_scaling.ipynb` | **the whole experiment**: theory, model, correctness check, sweeps E0–E5, figures |
| `notebooks/qwen3_batch_scaling.py` | the same notebook as a [jupytext](https://jupytext.readthedocs.io/) script (readable diffs). Edit this, then run `jupytext --sync notebooks/qwen3_batch_scaling.py` |
| `requirements-local.txt` | pinned versions (JAX 0.12.0.1 with CUDA 12) |
| `results/full/` | full-mode results (`raw/*.jsonl` + `.csv`, `env.json`, `figures/`, `traces/`) |
| `report/report.md`, `report/report.pdf` | technical report |
| `run_experiments.sh` | smoke test, then the full run, on one GPU (`GPU=<index> bash run_experiments.sh`) |
| `check_split.py` | the E-check in two processes, for servers with less than ~40 GB of free host RAM |

## Setup on the GPU server

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
```

```bash
uv pip install -r requirements-local.txt && uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

`torch` (CPU build) is only needed for the Hugging Face correctness reference. Check that JAX sees
the GPU:

```bash
python -c "import jax; print(jax.devices())"
```

It should print a `CudaDevice`. Also check `nvidia-smi` that nobody else is using the card: the
notebook reserves 95% of its memory at start-up (see "GPU memory" below).

## Running

All commands run from `notebooks/`. Results go to `OUT_DIR`, which defaults to
`../results/<RUN_MODE>`.

**1. Smoke test (Qwen3-0.6B, ~10 min).** Checks every code path.

```bash
cd notebooks && CUDA_VISIBLE_DEVICES=0 RUN_MODE=smoke OUT_DIR=../results/smoke_gpu jupyter nbconvert --to notebook --execute qwen3_batch_scaling.ipynb --output /tmp/smoke.ipynb
```

Check that `env.json` shows the 3090, the correctness cell passes, `bytes_limit` is about
22.5 GiB, and six figures appear in `results/smoke_gpu/figures/`.

**2. Full run (Qwen3-4B, several hours).** Run it inside tmux so it survives a dropped SSH session:

```bash
tmux new -s qwen3
```

```bash
cd notebooks && mkdir -p ../results/full && CUDA_VISIBLE_DEVICES=0 RUN_MODE=full jupyter nbconvert --to notebook --execute qwen3_batch_scaling.ipynb --output-dir ../results/full --output executed_full.ipynb --ExecutePreprocessor.timeout=-1 2>&1 | tee -a ../results/full/run.log
```

Detach with `Ctrl-b d` and re-attach with `tmux attach -t qwen3`. Results land in `results/full/`
(raw data, `env.json`, figures, traces, and the executed notebook with all outputs). `nbconvert`
only writes cell output into the executed notebook at the end, so follow progress through the raw
files, e.g. `tail -n 3 results/full/raw/*.jsonl`, and `nvidia-smi` for memory use.

**If the run dies:** every experiment appends results row by row and skips finished rows on
re-run. Run the same command again and it picks up where it stopped. To redo one experiment,
delete its `raw/<name>.jsonl` first.

**To skip experiments:** set `RUN_E5` (and so on) to `False` in the config cell, or set
`RUN_CHECK=0` to skip the Hugging Face comparison (it needs ~40 GB of host RAM for Qwen3-4B in
fp32).

### GPU memory

By default JAX lets its allocator use only 75% of GPU memory. The notebook sets
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` and keeps preallocation on (JAX
[GPU memory allocation docs](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)), so the
sweep can use about 22.5 GiB and the reported `bytes_limit` is the real cap used for the OOM
predictions and the memory figure. If start-up fails because another process holds GPU memory,
lower the fraction from the shell, e.g. `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`; the predictions
follow the new `bytes_limit`.

### Other local checks

Test the optional 8-device tensor-parallel path (E6, not part of the main experiment) on a CPU
that pretends to be 8 devices:

```bash
cd notebooks && FAKE_CPU_DEVICES=8 RUN_E6=1 RUN_MODE=smoke OUT_DIR=../results/smoke_cpu8 jupyter nbconvert --to notebook --execute qwen3_batch_scaling.ipynb --output /tmp/smoke_cpu8.ipynb
```

To regenerate figures from saved results, open the notebook with `OUT_DIR` pointing at
`results/full` and run only §1–4 and §15. §15 reads only the saved results.

## What the notebook measures

| ID | Experiment | Output file |
|---|---|---|
| E-check | logits vs HF `transformers` (fp32 exact check + bf16 on device) and greedy text | `raw/correctness.json` |
| E0 | achievable bf16 matmul FLOP/s and device-memory bandwidth | `raw/e0_ceilings.csv` |
| E1 | decode step time and memory, batch × context | `raw/e1_decode.jsonl` |
| E2 | prefill latency and throughput | `raw/e2_prefill.jsonl` |
| E3 | end-to-end requests: time to first token, latency, throughput | `raw/e3_e2e.jsonl` |
| E4 | attention share (no-attention ablation), XLA cost analysis, profiler traces | `raw/e4_*.jsonl`, `traces/` |
| E5 | ablations: `repeat_kv` GQA, per-step dispatch, no buffer donation | `raw/e5_*.jsonl` |
| E6 (optional) | 8-device tensor parallel; off by default, skipped with fewer than 8 devices | `raw/e6_tp8.jsonl`, `raw/e6_tp_check.json` |

## Verified so far

- **Correctness** (Qwen3-0.6B): the fp32 JAX model matches HF `transformers` to 6e-5 max
  |Δlogit| with 100% top-1 agreement. Greedy decoding through the KV cache continues "…The capital
  of Japan is" with "Tokyo. The capital of Italy is Rome."
- **Smoke runs** (Qwen3-0.6B) passed earlier on an RTX 3090 and on 8 fake CPU devices.
  - Tensor-parallel and single-device runs agree on the top-1 token.
  - Each device holds exactly 1/8 of the weights and cache.
- **Memory accounting:** XLA's argument bytes equal the theory's `weights + B·S·KV_per_token`.
  Donation aliases the whole cache, so there are no hidden copies.
- **Full run** (Qwen3-4B) completed on an RTX 3090 on 2026-11-12, in about 40 min. The Qwen3-4B
  E-check passed (fp32 max |Δlogit| 5.9e-5) with `check_split.py`. See the
  [technical report](report/report.md).
