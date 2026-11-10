# Batch scaling of Qwen3-4B inference on TPU v5e

**Research question:** how does batch size trade off latency, throughput and memory during LLM
inference, and when does decoding become compute-bound instead of memory-bound? How do context
length and the number of chips move that boundary?

- **Model:** [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) in bf16. Weights, config and
  tokenizer come straight from the Hugging Face Hub.
- **Engine:** a minimal JAX implementation of the Qwen3 forward pass (~150 lines), checked against
  Hugging Face `transformers`.
- **Hardware:** Kaggle TPU v5e-8. The main sweeps use 1 chip; an 8-chip tensor-parallel extension
  uses all of them.
- **Theory:** the [Scaling Book inference chapter](https://jax-ml.github.io/scaling-book/inference/)
  adapted to Qwen3-4B. See [PLAN.md](PLAN.md) for the derivation, hypotheses and experiment design.

## Repository layout

| Path | What |
|---|---|
| `notebooks/qwen3_batch_scaling_tpu.ipynb` | **the whole experiment**: theory, model, correctness check, sweeps E0–E6, figures |
| `notebooks/qwen3_batch_scaling_tpu.py` | the same notebook as a [jupytext](https://jupytext.readthedocs.io/) script (readable diffs). Edit this, then run `jupytext --sync` |
| `PLAN.md` | research plan: theory, hypotheses H1–H7, experiments, TPU-hour budget |
| `results/full/` | raw results downloaded from Kaggle (`raw/*.jsonl` + `.csv`, `env.json`, `figures/`) |
| `report/` | technical report |

## Reproducing on Kaggle (TPU v5e-8)

1. **One-time setup.** Kaggle → Settings → verify your phone number (TPU access needs it).
2. **Import.** Kaggle → *Create → New Notebook → File → Import Notebook* → upload
   `notebooks/qwen3_batch_scaling_tpu.ipynb`.
3. **Settings panel.** Set *Accelerator* to **TPU v5e-8** and turn *Internet* **on**.
4. **Smoke test (~10 min).** Leave `RUN_MODE = "smoke"` in the first code cell and *Run All*.
   Check that:
   - the environment cell shows `device_kind` as a v5e ("TPU v5 lite");
   - the correctness cell passes;
   - six figures appear.
5. **Full run (~3–4 TPU-hours).** Set `RUN_MODE = "full"`, then *Save Version → Save & Run All
   (Commit)*. This runs headless and keeps `/kaggle/working` as versioned output.
6. **Download.** From the version's *Output* tab, download `results_full.zip`. Unzip it into
   `results/full/`.
7. **Stop the interactive session.** An idle TPU session still uses your 20 h/week quota.

**If the session dies:** every experiment appends results row by row and skips finished rows on
re-run. Re-run the notebook and it picks up where it stopped. To redo one experiment, delete its
`raw/<name>.jsonl` first.

**To skip experiments when quota runs low:** set `RUN_E6`, `RUN_E5` (and so on) to `False` in
the config cell.

## Running locally (CPU or NVIDIA GPU)

Use this for development and correctness checks only. It costs no TPU quota.

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
```

```bash
uv pip install -r requirements-local.txt && uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Smoke test on one GPU (Qwen3-0.6B):

```bash
cd notebooks && CUDA_VISIBLE_DEVICES=0 RUN_MODE=smoke OUT_DIR=../results/smoke_gpu jupyter nbconvert --to notebook --execute qwen3_batch_scaling_tpu.ipynb --output /tmp/smoke.ipynb
```

Test the 8-chip tensor-parallel path on a CPU that pretends to be 8 devices:

```bash
cd notebooks && FAKE_CPU_DEVICES=8 RUN_MODE=smoke OUT_DIR=../results/smoke_cpu8 jupyter nbconvert --to notebook --execute qwen3_batch_scaling_tpu.ipynb --output /tmp/smoke_cpu8.ipynb
```

To regenerate figures from downloaded results, open the notebook with `OUT_DIR` pointing at
`results/full` and run only §1–4 and §15. §15 reads only the saved results.

## What the notebook measures

| ID | Experiment | Output file |
|---|---|---|
| E-check | logits vs HF `transformers` (fp32 exact check + bf16 on device) and greedy text | `raw/correctness.json` |
| E0 | achievable bf16 matmul FLOP/s and HBM bandwidth | `raw/e0_ceilings.csv` |
| E1 | decode step time and memory, batch × context, 1 chip | `raw/e1_decode.jsonl` |
| E2 | prefill latency and throughput | `raw/e2_prefill.jsonl` |
| E3 | end-to-end requests: time to first token, latency, throughput | `raw/e3_e2e.jsonl` |
| E4 | attention share (no-attention ablation), XLA cost analysis, profiler traces | `raw/e4_*.jsonl`, `traces/` |
| E5 | ablations: `repeat_kv` GQA, per-step dispatch, no buffer donation | `raw/e5_*.jsonl` |
| E6 | 8-chip tensor parallel | `raw/e6_tp8.jsonl`, `raw/e6_tp_check.json` |

## Verified locally so far

- **Correctness** (Qwen3-0.6B): the fp32 JAX model matches HF `transformers` to 6e-5 max
  |Δlogit| with 100% top-1 agreement. Greedy decoding through the KV cache continues "…The capital
  of Japan is" with "Tokyo. The capital of Italy is Rome."
- **Smoke runs** pass on an RTX 3090 and on 8 fake CPU devices.
  - Tensor-parallel and single-device runs agree on the top-1 token.
  - Each device holds exactly 1/8 of the weights and cache.
- **Memory accounting:** XLA's argument bytes equal the theory's `weights + B·S·KV_per_token`.
  Donation aliases the whole cache, so there are no hidden copies.
- **Not yet run:** anything on the actual TPU v5e.
