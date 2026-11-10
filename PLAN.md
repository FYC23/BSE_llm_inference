# Project Plan: Batch Scaling of Qwen3-4B Inference on Kaggle TPU v5e

**Research question:** Batch scaling. How does batch size trade off latency, throughput, and memory, and when does inference become compute-bound vs memory-bound? We extend it with: *how do context length and the number of chips move that boundary?*
**Model:** Qwen3-4B (bf16)
**Hardware:** Kaggle **TPU v5e-8** (8 chips × 16 GiB, one host). Budget: **20 TPU-hours/week**, with a **9 h cap per session**.
**Deliverable vehicle:** one self-contained Jupyter notebook, written and tested on the lab server, then uploaded to Kaggle with *File → Import Notebook*.

---

## 0. Why this setup

### TPU v5e matches the theory source exactly
The theory comes from the [Scaling Book inference chapter](https://jax-ml.github.io/scaling-book/inference/). Its worked numbers *are* TPU v5e numbers, so our measurements directly test the book's predictions. Moving off the 3090 also removes the CUDA/HF Python-overhead confound: under `jax.jit` the whole decode step is one compiled XLA program.

### Hardware constants (source: [Scaling Book, "TPU specs"](https://jax-ml.github.io/scaling-book/tpus/#tpu-specs))

| | TPU v5e (per chip) |
|---|---|
| HBM | 16 GiB |
| HBM bandwidth | 8.2e11 B/s |
| bf16 FLOP/s | 1.97e14 |
| int8 OP/s | 3.94e14 |
| ICI per link (one-way / bidirectional) | 4.5e10 / 9.0e10 B/s |
| PCIe (host ↔ chip) | ~1.6e10 B/s |
| Host topology | 4×2 (8 chips; no wraparound below 16) |
| MXU | 4 × 128×128 |
| **B_crit = FLOP/s ÷ BW** | **240** |

### Kaggle facts
- **Sessions:** a TPU session caps at 9 h (CPU/GPU sessions get 12 h) ([fastero](https://fastero.com/blog/kaggle-notebooks), [Kaggle forum](https://www.kaggle.com/general/286404)).
- **Account and network:** TPU access needs phone verification ([kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab/blob/main/README.md)). Turn Internet on in the notebook settings so the notebook can pull from Hugging Face.
- **Software:** the preinstalled JAX version is unknown. The first cell prints the versions, installs `jax[tpu]` only if needed, and records everything to `env.json`. We pin versions after the first success. The notebook also asserts that `device_kind` is v5e.
- **Sanity anchor:** about 130 tok/s for single-stream Qwen-27B on Kaggle v5e-8 via vllm-tpu ([kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab/blob/main/README.md)).

### How we use Hugging Face
Qwen3-4B's **weights, config, and tokenizer all come straight from the HF Hub** (`Qwen/Qwen3-4B`, safetensors). The only question is *which code runs the forward pass on the TPU*:

| Option | Runs Qwen3-4B on v5e? | Problem for this research question | Role |
|---|---|---|---|
| HF `transformers` Flax/JAX class | **No.** Transformers v5 dropped Flax/TF and is PyTorch-only ([HF blog](https://huggingface.co/blog/transformers-v5)), and there was never a Flax Qwen3 class | – | – |
| HF `transformers` PyTorch on TorchTPU / torch_xla | Yes, via `torch.compile(backend="tpu")` + `StaticCache` (TorchTPU was open-sourced Oct 2026) | Very new; Kaggle install unknown; HF's GQA may `repeat_kv`; no `cost_analysis`-style counts. **Decision: not used on TPU**; we stick with JAX | CPU **reference** for correctness only |
| **vLLM-TPU** (`tpu-inference`, whose models are JAX/Flax NNX), `LLM("Qwen/Qwen3-4B")` | Yes. Qwen3-4B is in its tested-model list, and v5e is a recommended chip ([tpu-inference](https://github.com/vllm-project/tpu-inference), [docs](https://docs.vllm.ai/projects/tpu/en/latest/)) | A black box: paged KV is **preallocated** (so memory can't be measured per batch), continuous batching makes B and S drift, and per-step times at a fixed (B,S) aren't exposed | **E7: production reference** for throughput/latency |
| **Minimal JAX forward pass (~200 lines) on the same HF weights** | Yes, as one jit-compiled XLA program | Must be proven correct (E-check) | **Main measurement instrument** |

**Why the main instrument is our own forward pass:** the research question needs step time and HBM at an exact (B, S). It also needs the theory to map one-to-one onto the code: no `repeat_kv`, a donated static cache, and our own sharding. Finally, XLA's `cost_analysis()` gives compiler-counted FLOPs and bytes to set beside the theory. vLLM then shows how far a production engine is from our clean numbers, which is a finding in itself.

---

## 1. Architecture (from `Qwen/Qwen3-4B/config.json`)

L = 36, D = 2560, F = 9728, N = 32 query heads, K = 8 KV heads, H = 128, V = 151,936, tied embeddings, bf16. Note that N·H = 4096 ≠ D.

| Quantity | Value | Scaling Book formula |
|---|---|---|
| Parameters | **4.022 B** (non-embedding 3.634 B; embedding 389 M) | L·D·(3F + 2H(N+K)) + D·V |
| Weight bytes (bf16) | **8.04 GB**, about half of one v5e chip | 2·P |
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
| "Linear operations: what bottlenecks us?" | B_crit = C/W_hbm = **240** on v5e |
| "What about attention?" | decode attention intensity ≈ 1·(N/K) = 4, so it is memory-bound at every B |
| "What about memory?" | KV and parameter-count formulas |
| "Theoretical estimates for LLM latency and throughput" | T_step and tokens/s formulas |
| "Modeling throughput and latency for LLaMA 2-13B" | the format of our prediction table, which we rebuild for Qwen3-4B and then measure |
| Sharding sections, Appendix C | collective latency (~1 µs/hop; latency-bound below ~360 kB for 8-way) → H7 |
| Appendix A, "nits" | the sharp roofline isn't reached because overlap is imperfect → H6 |

### Two step-time models (we let the data choose between them)
```
(A) Book / per-op roofline (matmul and attention times add):
    T_step = B·S·KV_tok / BW  +  max( 2·B·P / C ,  2P / BW )
(B) Aggregate roofline (perfect overlap inside the fused XLA program):
    T_step = max( B·(2P + 4NHL·S) / C ,  (2P + B·S·KV_tok) / BW )
Throughput = B / T_step   →   ceiling min( C/2P ,  BW / (S·KV_tok) )
```
XLA fuses more aggressively than separate CUDA kernels do, so model B is a credible competitor on a TPU, unlike on the 3090. The two models disagree most at short S with large B.

### Prediction table, 1 v5e chip (model A; ~15 GiB usable; cells = step ms / tokens per s)

| S | max B before OOM | B=1 | B=16 | B=64 | B=128 | B=240 | KV-bandwidth cap |
|---|---|---|---|---|---|---|---|
| 128 | ~439 | 9.8 / 102 | 10.2 / 1.6k | 11.3 / 5.7k | 12.8 / 10.0k | 15.3 / 15.7k | 43k |
| 256 | ~219 | 9.9 / 101 | 10.5 / 1.5k | 12.8 / 5.0k | 15.7 / 8.2k | OOM | 22k |
| 512 | ~110 | 9.9 / 101 | 11.3 / 1.4k | 15.7 / 4.1k | OOM | OOM | 11k |
| 1024 | ~55 | 10.0 / 100 | 12.8 / 1.3k | OOM | | | 5.4k |
| 4096 | ~14 | 10.5 / 95 | OOM | | | | 1.4k |

### Prediction table, 8 chips, tensor-parallel (ideal, excluding communication)
Weights are 1 GB per chip, and each chip holds exactly 1 KV head, so the KV cache shards cleanly. The memory wall moves out by roughly 15×: at S = 1024, max B ≈ 811, so **B_crit = 240 becomes reachable at long contexts**. The ideal step at B = 1 is about 1.2 ms. But the 2 all-reduces per layer × 36 layers = **72 collectives per step**. At roughly 10–20 µs each in the latency-bound regime, that adds an estimated **0.7–1.4 ms**, the same size as the ideal step itself.

### Hypotheses
- **H1 (latency):** On 1 chip, decode-step latency stays near the 9.8 ms weight-load floor for B ≲ 64 at short S. It rises linearly once B > 240 (reachable only at S ≲ 128).
- **H2 (throughput):** Throughput grows about linearly in B, then flattens toward min(compute cap 24.5k tok/s, KV cap BW/(S·KV_tok)).
- **H3 (memory wall):** On a single 16 GiB chip, the OOM limit arrives *before* B_crit for every S ≳ 200 tokens. Single-chip decode is effectively memory-bound everywhere except tiny contexts. The attention share of step time grows with B·S, and we measure it from the profiler and from cost_analysis.
- **H4 (memory):** Peak HBM ≈ 8.04 GB + B·S·144 KiB, plus small activations. The measured OOM batch matches the formula to within a few percent.
- **H5 (prefill):** Prefill with B·T ≳ 240 tokens is compute-bound. Batching prefill raises latency about linearly with almost no throughput gain. Time to first token is roughly 2·P·B·T / C.
- **H6 (implementation gap):** Measured times exceed the ideal by a gap we can explain. It comes mainly from three sources: (i) imperfect overlap; (ii) KV cache copies if the cache buffer isn't donated; (iii) 4× KV traffic if GQA uses `repeat_kv` instead of grouped einsums. We test (ii) and (iii) as ablations.
- **H7 (8-chip tensor parallelism):** TP-8 cuts latency well short of 8× at small B, because collective latency dominates. It recovers toward 8× as B grows, and it makes B_crit reachable at S = 1–2k.

---

## 3. Experiments

**Fixed across all runs:** bf16, greedy argmax, synthetic random token IDs with a fixed seed (content doesn't affect dense compute), EOS ignored, a static per-layer KV cache `[B, K, S, H]` updated in place (donated buffer), layers unrolled (no `lax.scan`; on GPU, scan made XLA copy each layer's cache slice, ~3× KV traffic), and `jax.block_until_ready` around every timed region.

| ID | What | Grid | Output |
|---|---|---|---|
| E-check | Correctness | Qwen3-0.6B locally and Qwen3-4B on Kaggle, compared with HF `transformers` on CPU | max |Δlogits|, top-1 agreement, a greedy text sample |
| E0 | Measured ceilings | large bf16 matmul; HBM copy/reduction | achievable FLOP/s and B/s, which replace the spec numbers in model fits |
| E1 | Decode step, 1 chip | B ∈ {1,2,4,8,16,32,64,96,128,192,256,320,384,448} × S ∈ {128,256,512,1024,2048,4096}, until OOM. Sample densely at S ≤ 256 | median/IQR step time (≥30 steps × 3 repeats), peak HBM, XLA cost_analysis FLOPs and bytes |
| E2 | Prefill | B ∈ {1..64} × T ∈ {128,512,2048} | latency, tokens/s, HBM |
| E3 | End-to-end requests | B sweep, prompt 512 + 256 generated tokens | time to first token, request latency, aggregate throughput |
| E4 | Profile | ~6 (B,S) points | `jax.profiler` trace: matmul vs attention vs collective time |
| E5 | Implementation ablations | donated vs non-donated cache; grouped vs `repeat_kv` GQA | tests H6 |
| E6 | Multi-chip (extension) | TP-8 (and optionally DP-8) on a reduced E1 grid | tests H7 |
| E7 | vLLM-TPU reference | `LLM("Qwen/Qwen3-4B")`, `max_num_seqs` = B, fixed input/output lengths, `ignore_eos`, on 1 chip | time per output token, throughput, and vLLM's reported max concurrency, compared against E1/E3 |

### Measuring decode at context S without generating S tokens
Allocate a cache of length S filled with random values and time the step at position S − 1. Content doesn't affect cost. E3 validates this shortcut against real prefill-then-decode runs.

### Protocol
- **Compile and warm up first.** Compile each shape once, then run warm-up steps before timing; compile time is logged separately.
- **Read memory from XLA, not a profiler UI.** HBM numbers come from `device.memory_stats()` (`peak_bytes_in_use`, `bytes_limit`) and `compiled.memory_analysis()`.
- **Treat OOM as a result.** A `RESOURCE_EXHAUSTED` error is recorded as a data point.
- **Make runs resumable.** Each condition is appended to a CSV as soon as it finishes, and a rerun skips completed keys. This matters because sessions end at 9 h, and anything in `/kaggle/working` is lost unless you commit the notebook version or download it.

---

## 4. The notebook (the main build artifact)

`notebooks/qwen3_batch_scaling_tpu.ipynb`. A [jupytext](https://jupytext.readthedocs.io/) `.py` twin keeps diffs in git readable. The notebook runs unchanged in three places: locally on CPU, locally on GPU 0, and on Kaggle TPU.

| § | Cell group | Notes |
|---|---|---|
| 0 | Title, research question, how to run | markdown |
| 1 | **Config** | `RUN_MODE ∈ {smoke, full}`, `MODEL_ID`, grids, `OUT_DIR` (Kaggle → `/kaggle/working`, else `./results`) |
| 2 | Environment | detect Kaggle vs local; install `jax[tpu]` / `safetensors` / `huggingface_hub` only if missing; print and save versions, device kind, and chip count to `env.json` |
| 3 | Hardware specs | the Scaling Book table above; choose the row by `device_kind` |
| 4 | Theory | read config.json; count params, FLOPs, bytes, KV; models A and B; produce the prediction tables |
| 5 | Model | Qwen3 in pure JAX: RMSNorm, RoPE (θ = 1e6), QK-norm, grouped-einsum GQA, SwiGLU, unrolled layers with per-layer weights and KV buffers. Weights load with `safetensors` (`framework="flax"`) |
| 6 | Correctness | E-check, plus unit asserts: param count equals theory; KV buffer bytes equal 144 KiB·B·S |
| 7 | Measurement harness | timer, OOM catch, resumable CSV writer |
| 8–13 | E0 → E5 | one section each, each idempotent |
| 14 | E6 multi-chip | `Mesh` + `NamedSharding` (shard heads, F, and vocab over 8 devices) |
| 15 | Figures | also runs locally from downloaded CSVs |
| 16 | Package outputs | zip `results/` + `figures/` for download |

**Smoke mode** (about 10 minutes): Qwen3-0.6B, a 3×3 grid, every code path exercised. Run it locally first, then once on Kaggle before committing to a full run.

### Porting workflow
1. Develop and test on the lab server. JAX on CPU and GPU 0 runs the same code with Qwen3-0.6B, and none of this spends TPU quota.
2. On Kaggle, use *File → Import Notebook* to upload the `.ipynb`. In settings, set Accelerator = TPU v5e-8 and turn Internet on.
3. Run smoke mode interactively (about 0.5 TPU-h).
4. Run full mode with *Save Version → Save & Run All*. This runs headless and keeps `/kaggle/working` as versioned output. Then download the outputs.
5. Commit the CSVs to the repo, regenerate the figures locally, and write the report.

### TPU-hour budget (20 h/week)
All debugging happens locally (CPU + GPU 0), so TPU hours go only to measurements.

| Item | Estimated TPU-h |
|---|---|
| Smoke run on Kaggle | 1 |
| E0 + E1 (1 chip, full grid) | 2–3 |
| E2 + E3 + E5 | 1.5 |
| E4 profiling | 0.5 |
| E6 TP-8 | 1.5 |
| E7 vLLM-TPU (install + runs) | 1 |
| **Planned total** | **~8**, leaving ~12 h as buffer for reruns and a second week if needed |

To protect the quota, **stop the session as soon as a run finishes**, because an idle TPU session still burns quota. If the budget runs short, cut in this order: E7, then E6, then E5.

---

## 5. Figures (6; the deliverable requires 3+)
1. **Decode step latency vs B** (log-log), one line per S, with model-A and model-B predictions dashed.
2. **Throughput vs per-token latency**, the Pareto curve (each point a B, colored by S), with the KV-bandwidth caps.
3. **Empirical roofline**: achieved FLOP/s vs arithmetic intensity, using measured ceilings, points for each (B,S), and cost_analysis values.
4. **HBM vs B**: measured vs 8.04 GB + B·S·144 KiB, the 16 GiB line, and OOM markers.
5. **Regime map** over B × S: the B_crit line, the memory wall for 1 chip vs 8 chips, the measured regime, and residuals of models A and B.
6. **Implementation and scaling**: ablations (donation, `repeat_kv`) plus 1-chip vs TP-8 latency and throughput.

## 6. Repository layout
```
llm_inference/
├── README.md                       # RQ, Kaggle run instructions, figure gallery
├── PLAN.md
├── notebooks/
│   ├── qwen3_batch_scaling_tpu.ipynb
│   └── qwen3_batch_scaling_tpu.py  # jupytext twin
├── requirements-local.txt          # pinned versions for local dev / figure regeneration
├── results/
│   ├── raw/*.csv
│   ├── env.json
│   └── traces/                     # small profiler excerpts only
├── figures/*.png
└── report/report.md (→ PDF)
```

## 7. Report outline (maps to the rubric)
1. Research question
2. Hypotheses H1–H7
3. Architecture: Qwen3-4B vs the lecture's 0.6B, GQA 4:1, tied embeddings, and per-component costs
4. Theory: Scaling Book formulas cited by section, models A and B, the 1-chip and 8-chip tables
5. Setup: Kaggle v5e-8, the JAX implementation and its correctness check, the protocol, software versions
6. Raw results (tables, plus links to the CSVs)
7. Figures 1–6
8. Interpretation: each hypothesis accepted or rejected, with numbers; which step model fits; where XLA departs from the ideal
9. Limitations: one TPU generation; Kaggle's shared environment and unknown clocks; our implementation instead of a production server (no paged attention or continuous batching); synthetic prompts; bf16 only; TP-8 has a 4×2 topology with no wraparound
10. Reproduction: the Kaggle steps from §4, plus local figure regeneration
11. AI-use statement

## 8. Work split and timeline (adjust to the deadline)
| Phase | Partner A | Partner B | Done when |
|---|---|---|---|
| 1. Local build (days 1–3) | §5 JAX forward pass + E-check against HF on CPU (0.6B and 4B; the lab server has 125 GB RAM) | §2–4 env, specs, theory; §7 harness | logits match HF; smoke mode passes locally |
| 2. Kaggle smoke (day 4) | phone verification, smoke run | E0 ceilings | `env.json` records v5e; ceilings measured |
| 3. Full runs (days 5–6) | E1, E2, E3 | E4, E5, E6 | all CSVs downloaded and committed |
| 4. Analysis (days 7–8) | figures 1–3 | figures 4–6 | the figures cell regenerates everything |
| 5. Write-up (days 9–10) | theory, architecture, setup | results, interpretation, limitations | report, README, 3-minute slides |
| 6. Repro check | partner B reruns smoke mode from a fresh Kaggle import | | numbers match within noise |

## 9. Risks
| Risk | Mitigation |
|---|---|
| TPU quota exhausted before the full sweep | all debugging local; resumable CSV; cut order E7 → E6 → E5 |
| vLLM-TPU won't install on the Kaggle image | E7 is optional; give it at most 30 min, then drop it and note this as a limitation |
| JAX/libtpu version problems on Kaggle | install cell with a fallback; pin after the first success; smoke run before any long job |
| The 9 h session ends mid-sweep | resumable CSV; full runs via Save & Run All; total sweep ≈ 3 h |
| Bugs in our Qwen3 implementation | E-check on 0.6B and 4B against HF logits; greedy text sanity check |
| Compile time blows the budget | one compile per (B, S) shape, logged per row (~3 s per shape for 0.6B on GPU); grids are trimmed in `GRIDS` if TPU compiles turn out slow |
| The report just restates the Scaling Book | the book is the cited theory. Our findings are measured deviations, the model-A vs model-B verdict, the memory wall, and TP-8 collective overhead |
| *(Stretch)* int8 weights should halve B_crit to ~120 (book, Section 1) | only if time remains; a native int8 matmul on v5e makes this cleaner than it would be on the 3090 |
