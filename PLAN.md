# Project Plan: Batch Scaling of Qwen3-4B Inference on a Single RTX 3090

**Research question:** Batch scaling. How does batch size trade off latency, throughput, and memory, and when does inference become compute-bound vs memory-bound? We extend it with: *how does context length move that boundary?*
**Model:** Qwen3-4B (bf16)
**Hardware:** one NVIDIA **RTX 3090** (24 GB GDDR6X) on a Linux lab server, JAX on CUDA (`jax[cuda12]`, pinned in `requirements-local.txt`).
**Deliverable vehicle:** one self-contained Jupyter notebook, run headless on the lab server.

---

## 0. Why this setup

### Theory source
The theory comes from the [Scaling Book inference chapter](https://jax-ml.github.io/scaling-book/inference/). The book works its examples on TPUs, but the roofline argument (weights read once per step, KV read once per sequence, B_crit = FLOP/s ÷ bandwidth) is hardware-agnostic. We re-evaluate every formula with 3090 numbers. Under `jax.jit` the whole decode step is one compiled XLA program, so Python overhead stays out of the timed region.

### Hardware constants (source: NVIDIA Ampere GA102 whitepaper)

These are **nominal** spec-sheet numbers. Real boost clocks, power limits and achievable bandwidth differ from card to card, so E0 measures the actual ceilings and the model fits use the measured values.

| | RTX 3090 |
|---|---|
| Device memory | 24 GiB GDDR6X (JAX's pool is capped at 95% of it, ~22.5 GiB; see §4) |
| Memory bandwidth | 9.36e11 B/s |
| bf16 tensor FLOP/s (dense, fp32 accumulate) | 7.1e13 |
| PCIe (host ↔ GPU) | PCIe 4.0 x16, ~3.2e10 B/s nominal |
| **B_crit = FLOP/s ÷ BW** | **≈ 76** |

### How we use Hugging Face
Qwen3-4B's **weights, config, and tokenizer all come straight from the HF Hub** (`Qwen/Qwen3-4B`, safetensors). The only question is *which code runs the forward pass*:

| Option | Problem for this research question | Role |
|---|---|---|
| HF `transformers` (PyTorch) | per-op kernel launches and Python overhead mix into step time; GQA may `repeat_kv`; no compiler-counted FLOPs/bytes | CPU **reference** for correctness only |
| A serving engine (e.g. vLLM) | a black box: paged KV is preallocated (memory can't be measured per batch), continuous batching makes B and S drift, and per-step times at a fixed (B,S) aren't exposed | not used; noted as a limitation |
| **Minimal JAX forward pass (~200 lines) on the same HF weights** | must be proven correct (E-check) | **Main measurement instrument** |

**Why the main instrument is our own forward pass:** the research question needs step time and device memory at an exact (B, S). It also needs the theory to map one-to-one onto the code: no `repeat_kv`, a donated static cache. Finally, XLA's `cost_analysis()` and `memory_analysis()` give compiler-counted FLOPs, bytes and buffer sizes to set beside the theory.

---

## 1. Architecture (from `Qwen/Qwen3-4B/config.json`)

L = 36, D = 2560, F = 9728, N = 32 query heads, K = 8 KV heads, H = 128, V = 151,936, tied embeddings, bf16. Note that N·H = 4096 ≠ D.

| Quantity | Value | Scaling Book formula |
|---|---|---|
| Parameters | **4.022 B** (non-embedding 3.634 B; embedding 389 M) | L·D·(3F + 2H(N+K)) + D·V |
| Weight bytes (bf16) | **8.04 GB**, about a third of the 3090's memory | 2·P |
| KV per token | **147,456 B = 144 KiB** | 2·bytes·H·K·L |
| Decode FLOPs per token (dense) | **8.04 GFLOP** | 2·P (the tied unembedding counts) |
| Attention FLOPs per token per cached position | **589,824** | 4·N·H·L |

The report's architecture section walks the lecture's diagram, then states the inference costs:
- **Weights** are read once per decode step regardless of batch, so their arithmetic intensity grows with B.
- **The KV cache** is read once *per sequence*, so batching never amortizes it.
- **GQA** with a 4:1 query-to-KV ratio makes decode attention's intensity about 4 (the book's ST/(S+T) ≈ 1, multiplied by N/K).

---

## 2. Theory and hypotheses (Scaling Book adapted)

| Book section | What we take |
|---|---|
| "Linear operations: what bottlenecks us?" | B_crit = C/W_mem ≈ **76** on the 3090 (spec) |
| "What about attention?" | decode attention intensity ≈ 1·(N/K) = 4, so it is memory-bound at every B |
| "What about memory?" | KV and parameter-count formulas |
| "Theoretical estimates for LLM latency and throughput" | T_step and tokens/s formulas |
| "Modeling throughput and latency for LLaMA 2-13B" | the format of our prediction table, which we rebuild for Qwen3-4B and then measure |
| Appendix A, "nits" | the sharp roofline isn't reached because overlap is imperfect → H6 |

### Two step-time models (we let the data choose between them)
```
(A) Book / per-op roofline (matmul and attention times add):
    T_step = B·S·KV_tok / BW  +  max( 2·B·P / C ,  2P / BW )
(B) Aggregate roofline (perfect overlap inside the fused XLA program):
    T_step = max( B·(2P + 4NHL·S) / C ,  (2P + B·S·KV_tok) / BW )
Throughput = B / T_step   →   ceiling min( C/2P ,  BW / (S·KV_tok) )
```
On a GPU the kernels inside one XLA program still run mostly one after another, so model A is the a-priori favourite and model B the optimistic bound. The two models disagree most at short S with large B.

### Prediction table, RTX 3090 (model A, nominal spec; cells = step ms / tokens per s)

Computed with the notebook's own formulas (§4 of the notebook writes the same table to `raw/theory_spec_predictions.csv`). "Max B" is the largest batch whose weights + KV cache fit, ignoring activations, for a nominal 24 GiB and for the 95% cap the notebook sets. OOM cells are the same under either limit.

| S | max B (24 GiB / 95%) | B=1 | B=16 | B=64 | B=128 | B=256 | B=512 | KV-bandwidth cap |
|---|---|---|---|---|---|---|---|---|
| 128 | ~939 / ~870 | 8.6 / 116 | 8.9 / 1.8k | 9.9 / 6.5k | 17.1 / 7.5k | 34.2 / 7.5k | 68.3 / 7.5k | 50k |
| 256 | ~469 / ~435 | 8.6 / 116 | 9.2 / 1.7k | 11.2 / 5.7k | 19.7 / 6.5k | 39.3 / 6.5k | OOM | 25k |
| 512 | ~234 / ~217 | 8.7 / 115 | 9.9 / 1.6k | 13.8 / 4.7k | 24.8 / 5.2k | OOM | OOM | 12k |
| 1024 | ~117 / ~108 | 8.8 / 114 | 11.2 / 1.4k | 18.9 / 3.4k | OOM | | | 6.2k |
| 2048 | ~58 / ~54 | 8.9 / 112 | 13.8 / 1.2k | OOM | | | | 3.1k |
| 4096 | ~29 / ~27 | 9.2 / 108 | 18.9 / 846 | OOM | | | | 1.5k |

Weight-load floor 2P/BW = 8.6 ms; compute cap C/2P ≈ 8.8k tok/s. The context length at which max B drops to B_crit ≈ 76 is S ≈ 1.5k (1.58k at 24 GiB, 1.47k at the 95% cap).

### Hypotheses
- **H1 (latency):** Decode-step latency stays near the 8.6 ms weight-load floor for B ≲ 32 at short S. Once B > B_crit ≈ 76 it rises about linearly in B, which is reachable for S ≲ 1.5k.
- **H2 (throughput):** Throughput grows about linearly in B, then flattens toward min(compute cap ≈ 8.8k tok/s, KV cap BW/(S·KV_tok)).
- **H3 (memory wall):** On the 24 GiB card the OOM limit arrives *before* B_crit for S ≳ 1.5k, so long-context decode is memory-bound at every batch that fits. For shorter contexts the dense matmuls can become compute-bound before memory runs out. The attention share of step time grows with B·S; we measure it with the no-attention ablation and from cost_analysis.
- **H4 (memory):** Peak device memory ≈ 8.04 GB + B·S·144 KiB, plus temporaries (mostly fp32 logits, B·V·4 bytes ≈ 0.6 MB per sequence). The measured OOM batch matches the formula to within a few percent once temporaries are included.
- **H5 (prefill):** Prefill with B·T ≳ 76 tokens is compute-bound. Batching prefill raises latency about linearly with almost no throughput gain. Time to first token is roughly 2·P·B·T / C.
- **H6 (implementation gap):** Measured times exceed the ideal by a gap we can explain. It comes mainly from three sources: (i) imperfect overlap; (ii) KV cache copies if the cache buffer isn't donated; (iii) 4× KV traffic if GQA uses `repeat_kv` instead of grouped einsums. We test (ii) and (iii) as ablations.

---

## 3. Experiments

**Fixed across all runs:** bf16, greedy argmax, synthetic random token IDs with a fixed seed (content doesn't affect dense compute), EOS ignored, a static per-layer KV cache `[B, K, S, H]` updated in place (donated buffer), layers unrolled (no `lax.scan`; on GPU, scan made XLA copy each layer's cache slice, ~3× KV traffic), and `jax.block_until_ready` around every timed region.

| ID | What | Grid | Output |
|---|---|---|---|
| E-check | Correctness | Qwen3-0.6B (smoke) and Qwen3-4B (full), compared with HF `transformers` on CPU | max |Δlogits|, top-1 agreement, a greedy text sample |
| E0 | Measured ceilings | large bf16 matmul; device-memory copy/reduction | achievable FLOP/s and B/s, which replace the spec numbers in model fits |
| E1 | Decode step | B ∈ {1,2,4,8,16,32,48,64,96,128,160,…,512,640,768,896} × S ∈ {128,256,512,1024,2048,4096}, until OOM (dense around B_crit ≈ 76; the top values reach the S = 128 wall) | median/IQR step time (32 steps × 3 repeats), XLA memory_analysis, allocator peak |
| E2 | Prefill | B ∈ {1..64} × T ∈ {128,512,2048} | latency, tokens/s, memory |
| E3 | End-to-end requests | B sweep, prompt 512 + 256 generated tokens | time to first token, request latency, aggregate throughput |
| E4 | Where the time goes | ~18 (B,S) points + 5 cost points + 2 traces | no-attention ablation, XLA cost_analysis FLOPs and bytes, `jax.profiler` traces |
| E5 | Implementation ablations | donated vs non-donated cache; grouped vs `repeat_kv` GQA; fused loop vs per-step dispatch | tests H6 |

The notebook also contains an optional 8-device tensor-parallel extension (E6). It is off by default, needs ≥ 8 devices and is **not part of this experiment**; it is kept only so the sharded code path can still be tested on 8 fake CPU devices.

### Measuring decode at context S without generating S tokens
Allocate a cache of length S and time the step at position S − 1. Content doesn't affect cost. E3 validates this shortcut against real prefill-then-decode runs.

### Protocol
- **Compile and warm up first.** Compile each shape once, then run warm-up steps before timing; compile time is logged separately.
- **Read memory from XLA and the allocator.** `compiled.memory_analysis()` gives the program's argument, output, alias and temporary bytes for every condition. `device.memory_stats()` gives `bytes_limit` and `peak_bytes_in_use`; the peak is a process-wide high-water mark, so it is used per condition only where that condition raised it (`peak_is_new`).
- **Treat OOM as a result.** A `RESOURCE_EXHAUSTED` error is recorded as a data point; larger batches at the same S are then skipped.
- **Make runs resumable.** Each condition is appended to a results file as soon as it finishes, and a rerun skips completed keys, so a crashed or interrupted run can be restarted with the same command.

---

## 4. The notebook (the main build artifact)

`notebooks/qwen3_batch_scaling.ipynb`. A [jupytext](https://jupytext.readthedocs.io/) `.py` twin keeps diffs in git readable. The same notebook runs on CPU (smoke mode) and on the 3090.

| § | Cell group | Notes |
|---|---|---|
| 0 | Title, research question, how to run | markdown |
| 1 | **Config** | `RUN_MODE ∈ {smoke, full}`, `MODEL_ID`, grids, `OUT_DIR` (default `../results/<mode>`), GPU memory settings |
| 2 | Environment | install `safetensors` / `huggingface_hub` / `pandas` / `matplotlib` only if missing; refuse full mode on CPU; save versions, device kind and memory settings to `env.json` |
| 3 | Hardware specs | the table above; choose the row by `device_kind`; read `bytes_limit` |
| 4 | Theory | read config.json; count params, FLOPs, bytes, KV; models A and B; produce the prediction tables |
| 5 | Model | Qwen3 in pure JAX: RMSNorm, RoPE (θ = 1e6), QK-norm, grouped-einsum GQA, SwiGLU, unrolled layers with per-layer weights and KV buffers. Weights load with `safetensors` (`framework="flax"`) |
| 6 | Correctness | E-check, plus unit asserts: param count equals theory |
| 7 | Measurement harness | timer, OOM catch, resumable writer |
| 8–13 | E0 → E5 | one section each, each idempotent |
| 14 | E6 (optional) | 8-device tensor parallel; off by default |
| 15 | Figures | also runs from saved results |
| 16 | Package outputs | CSV copies + a zip for copying results off the server |

**GPU memory.** JAX's GPU allocator by default uses at most 75% of device memory ([JAX docs](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)). The notebook sets `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` and keeps preallocation on (less fragmentation near the OOM boundary) before JAX is imported, so `bytes_limit` is the real usable cap and the predicted OOM batch is computed from it.

**Smoke mode** (about 10 minutes): Qwen3-0.6B, tiny grids, every code path exercised. Run it on the server before the full run.

### Workflow
1. Develop and smoke-test on CPU or the 3090 with Qwen3-0.6B.
2. Full run on the 3090 inside tmux with `jupyter nbconvert --execute` (exact command in the README). Results land in `results/full/`.
3. Commit `results/full/`, regenerate figures, write the report.

### Run time
Not measured yet. Timed work per E1 condition is small (128 steps of 9–70 ms); the total is dominated by compiling each (B, S) shape of a 36-layer unrolled program. `compile_s` is logged per row. If compilation turns out slow, trim `GRIDS["full"]`. If time runs short, cut in this order: E5, then E4's attention-share points, then the largest-B points of E1.

---

## 5. Figures (6; the deliverable requires 3+)
1. **Decode step latency vs B** (log-log), one line per S, with model-A predictions dashed and OOM points marked.
2. **Throughput vs per-token latency**, the trade-off curve (each point a B, colored by S), with the compute cap.
3. **Empirical roofline**: achieved FLOP/s vs arithmetic intensity, with measured (E0) and spec rooflines.
4. **Device memory vs B**: theory (8.04 GB + B·S·144 KiB) vs XLA's compiled footprint (arguments + temporaries) per condition, measured allocator peaks where valid, the `bytes_limit` line, and OOM markers.
5. **Attention share and model fit**: measured vs model-A attention share; measured vs predicted step time for models A and B.
6. **Implementation ablations**: grouped vs `repeat_kv`, fused loop vs per-step dispatch, donation vs copy. (A second tensor-parallel panel appears only if optional E6 data exists.)

## 6. Repository layout
```
BSE_llm_inference/
├── README.md                       # RQ, server run instructions
├── PLAN.md
├── notebooks/
│   ├── qwen3_batch_scaling.ipynb
│   └── qwen3_batch_scaling.py      # jupytext twin
├── requirements-local.txt          # pinned versions (jax[cuda12])
├── results/full/
│   ├── raw/*.jsonl, *.csv
│   ├── env.json
│   ├── figures/*.png
│   └── traces/                     # small profiler excerpts only
└── report/report.md (→ PDF)
```

## 7. Report outline (maps to the rubric)
1. Research question
2. Hypotheses H1–H6
3. Architecture: Qwen3-4B vs the lecture's 0.6B, GQA 4:1, tied embeddings, and per-component costs
4. Theory: Scaling Book formulas cited by section, models A and B, the 3090 prediction table
5. Setup: RTX 3090, JAX on CUDA, the JAX implementation and its correctness check, the protocol, software versions
6. Raw results (tables, plus links to the CSVs)
7. Figures 1–6
8. Interpretation: each hypothesis accepted or rejected, with numbers; which step model fits; where XLA departs from the ideal
9. Limitations: one GPU model; nominal vs measured ceilings, clocks and thermals not controlled; our implementation instead of a production server (no paged attention or continuous batching); synthetic prompts; bf16 only
10. Reproduction: the README commands
11. AI-use statement

## 8. Work split and timeline (adjust to the deadline)
| Phase | Partner A | Partner B | Done when |
|---|---|---|---|
| 1. Build (done) | §5 JAX forward pass + E-check against HF on CPU | §2–4 env, specs, theory; §7 harness | logits match HF; smoke mode passes |
| 2. Server smoke | set up the venv on the server, smoke run on the 3090 | check `env.json`, `bytes_limit`, E0 ceilings | smoke figures look sane |
| 3. Full run | launch the full run in tmux, monitor | spot-check raw files as they arrive | `results/full/` complete and committed |
| 4. Analysis | figures 1–3 | figures 4–6 | the figures cell regenerates everything |
| 5. Write-up | theory, architecture, setup | results, interpretation, limitations | report, README, 3-minute slides |
| 6. Repro check | partner B reruns smoke mode from a fresh clone | | numbers match within noise |

## 9. Risks
| Risk | Mitigation |
|---|---|
| Another process holds GPU memory, so preallocating 95% fails at start-up | check `nvidia-smi` first; lower `XLA_PYTHON_CLIENT_MEM_FRACTION` from the shell; predictions follow the new `bytes_limit` |
| Fragmentation or temporaries make OOM arrive a little before the formula | that is a finding (H4); OOM is recorded as a result and XLA's temp bytes are logged per row |
| SSH session drops mid-run | run inside tmux; resumable results files |
| Clock / thermal variation on a shared server | E0 measures the ceilings in the same run; IQR bands over repeats; note GPU clocks in the report |
| Bugs in our Qwen3 implementation | E-check on 0.6B and 4B against HF logits; greedy text sanity check |
| Host RAM for the fp32 Qwen3-4B check (~40 GB) | `RUN_CHECK=0` if the server can't spare it |
| Compile time dominates the run | one compile per shape, logged per row; trim `GRIDS["full"]` |
| The report just restates the Scaling Book | the book is the cited theory. Our findings are measured deviations, the model-A vs model-B verdict, and where the memory wall sits relative to B_crit on a 24 GB GPU |
