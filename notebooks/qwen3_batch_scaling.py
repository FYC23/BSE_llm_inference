# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Batch scaling of Qwen3-4B inference on a single RTX 3090
#
# **Research question.** How does batch size trade off latency, throughput and memory during
# inference, and when does decoding become compute-bound vs memory-bound? How does context length
# move that boundary?
#
# **What this notebook does.** It loads Qwen3-4B weights straight from the Hugging Face Hub and runs
# them through a minimal JAX implementation of the Qwen3 forward pass, so every FLOP and byte
# maps onto the theory from the
# [Scaling Book inference chapter](https://jax-ml.github.io/scaling-book/inference/). It then sweeps
# batch size × context length on one NVIDIA RTX 3090 (24 GB, JAX on CUDA) and records step time,
# memory and XLA's own FLOP/byte counts.
#
# **How to run** (see the README for the exact commands).
# * `RUN_MODE=smoke` (Qwen3-0.6B, tiny grids, ~10 min) on CPU or GPU checks every code path.
# * `RUN_MODE=full` (Qwen3-4B, real grids) on the 3090, headless with
#   `jupyter nbconvert --execute` inside tmux. Results land in `results/full/`.
#
# Each experiment appends rows to a results file as it goes, and a re-run skips finished rows, so a
# run that dies halfway can be resumed.

# %% [markdown]
# ## 1. Configuration

# %%
import os

# "smoke" = Qwen3-0.6B + tiny grids (checks every code path); "full" = Qwen3-4B + real grids.
RUN_MODE = os.environ.get("RUN_MODE", "smoke")

# Which experiments to run (all are resumable / idempotent).
RUN_CHECK = os.environ.get("RUN_CHECK", "1") == "1"      # E-check: logits vs Hugging Face
RUN_E0 = True    # measured FLOP/s and bandwidth ceilings
RUN_E1 = True    # decode step: batch x context sweep, 1 device
RUN_E2 = True    # prefill sweep
RUN_E3 = True    # end-to-end requests (prefill + generation)
RUN_E4 = True    # attention share (no-attention ablation) + XLA cost analysis + profiler trace
RUN_E5 = True    # implementation ablations (repeat_kv, dispatch, buffer donation)
# Optional extension, not part of the single-GPU experiment: 8-device tensor parallel. Only runs
# with >= 8 devices (e.g. FAKE_CPU_DEVICES=8 RUN_E6=1 to test the code path on a CPU).
RUN_E6 = os.environ.get("RUN_E6", "0") == "1"

# Testing only: pretend the CPU has N devices so the tensor-parallel code path can be exercised
# without multiple accelerators. Must be set before JAX is imported.
FAKE_CPU_DEVICES = int(os.environ.get("FAKE_CPU_DEVICES", "0"))
if FAKE_CPU_DEVICES:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + f" --xla_force_host_platform_device_count={FAKE_CPU_DEVICES}")

# GPU memory (https://docs.jax.dev/en/latest/gpu_memory_allocation.html). By default JAX
# preallocates 75% of GPU memory, and the BFC allocator caps usable memory at that fraction even
# with PREALLOCATE=false, so a 24 GB card would report bytes_limit ~ 18 GiB and OOM early. We set
# the fraction explicitly (fixed limit, no version-dependent growth) and keep preallocation on,
# which the docs say fragments less than on-demand allocation: that matters because the sweeps
# run right up to the OOM boundary. The remaining 5% covers the CUDA context and libraries outside
# JAX's pool. bytes_limit (HBM_LIMIT below) then equals the real usable cap. These must be set
# before JAX is imported; override them from the shell if the GPU is shared. CPU ignores them.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

GRIDS = {
    "smoke": dict(
        model="Qwen/Qwen3-0.6B",
        decode_B=[1, 4, 16], decode_S=[128, 512],
        prefill_B=[1, 4], prefill_T=[64, 256],
        e2e_B=[1, 4], e2e_prompt=64, e2e_gen=16,
        attn_points=[(1, 128), (16, 512)],
        cost_points=[(1, 128), (16, 512)],
        ablation_points=[(4, 512)],
        tp_B=[1, 8], tp_S=[256],
        steps=4, repeats=2, profile_points=[(4, 128)],
    ),
    "full": dict(
        model="Qwen/Qwen3-4B",
        # dense around the spec B_crit (~76 on a 3090); 640-896 reach the S=128 OOM wall (~870)
        decode_B=[1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512,
                  640, 768, 896],
        decode_S=[128, 256, 512, 1024, 2048, 4096],
        prefill_B=[1, 2, 4, 8, 16, 32, 64], prefill_T=[128, 512, 2048],
        e2e_B=[1, 4, 16, 32, 48], e2e_prompt=512, e2e_gen=256,
        attn_points=[(B, S) for S in (128, 512, 2048) for B in (1, 8, 32, 64, 128, 256)],
        cost_points=[(1, 128), (64, 128), (256, 128), (16, 1024), (8, 4096)],
        ablation_points=[(1, 1024), (16, 1024), (64, 256)],
        tp_B=[1, 4, 16, 64, 128, 240, 320, 480, 640, 800],
        tp_S=[256, 1024, 2048],
        steps=32, repeats=3, profile_points=[(1, 1024), (64, 256)],
    ),
}
G = GRIDS[RUN_MODE]
MODEL_ID = os.environ.get("MODEL_ID", G["model"])
DTYPE_NAME = "bfloat16"

OUT_DIR = os.path.abspath(os.environ.get("OUT_DIR", f"../results/{RUN_MODE}"))
RAW_DIR = os.path.join(OUT_DIR, "raw")
FIG_DIR = os.path.join(OUT_DIR, "figures")
for d in (RAW_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)
print(f"RUN_MODE={RUN_MODE}  MODEL_ID={MODEL_ID}\nOUT_DIR={OUT_DIR}")

# %% [markdown]
# ## 2. Environment
# Installs the small pure-Python dependencies if they are missing. JAX itself must already be
# installed (`jax[cuda12]` for the GPU, see `requirements-local.txt`).

# %%
import importlib
import subprocess
import sys


def ensure(module, pip_spec):
    try:
        importlib.import_module(module)
    except ImportError:
        print(f"installing {pip_spec} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_spec])


for mod, spec in [("safetensors", "safetensors"), ("huggingface_hub", "huggingface_hub"),
                  ("pandas", "pandas"), ("matplotlib", "matplotlib")]:
    ensure(mod, spec)

# %%
import gc
import glob
import json
import math
import platform
import time
import warnings
from dataclasses import dataclass, asdict
from functools import partial

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

warnings.filterwarnings("ignore", message=".*donated buffers were not usable.*")
warnings.filterwarnings("ignore", message=".*Some donated buffers.*")

DTYPE = jnp.dtype(DTYPE_NAME)
DEVICES = jax.devices()
if RUN_MODE == "full" and DEVICES[0].platform == "cpu":
    raise RuntimeError(f"JAX sees only {DEVICES}. Full mode needs an accelerator: install "
                       "jax[cuda12] (requirements-local.txt) and check `nvidia-smi`.")
DEV0 = DEVICES[0]
CPU = jax.devices("cpu")[0]


def versions():
    out = {"python": platform.python_version(), "jax": jax.__version__}
    for m in ("jaxlib", "jax_cuda12_plugin", "safetensors", "huggingface_hub", "transformers", "torch"):
        try:
            out[m] = importlib.import_module(m).__version__
        except Exception:
            out[m] = None
    return out


ENV = dict(
    run_mode=RUN_MODE, model_id=MODEL_ID, dtype=DTYPE_NAME,
    platform=DEV0.platform, device_kind=DEV0.device_kind, n_devices=len(DEVICES),
    xla_env={k: os.environ.get(k) for k in ("XLA_PYTHON_CLIENT_PREALLOCATE",
                                            "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_FLAGS",
                                            "CUDA_VISIBLE_DEVICES")},
    host=platform.node(), time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    **versions(),
)
with open(os.path.join(OUT_DIR, "env.json"), "w") as f:
    json.dump(ENV, f, indent=2)
print(json.dumps(ENV, indent=2))

# %% [markdown]
# ## 3. Hardware specs
# The RTX 3090 numbers are nominal spec-sheet values (NVIDIA Ampere GA102 whitepaper: 936 GB/s
# GDDR6X, 71 TFLOP/s dense bf16 tensor-core math with fp32 accumulate). Real clocks and achievable
# throughput differ, so E0 measures what the card actually achieves, and the theory below is
# evaluated with both. The TPU v5e row (from the
# [Scaling Book, "TPU specs"](https://jax-ml.github.io/scaling-book/tpus/#tpu-specs)) is kept so
# the notebook stays device-generic. The code says "HBM" for device memory whatever its type.

# %%
HW_SPECS = {
    "tpu_v5e": dict(label="TPU v5e", hbm_bytes=16 * 2**30, hbm_bw=8.2e11, bf16_flops=1.97e14,
                    int8_ops=3.94e14, ici_oneway_per_link=4.5e10,
                    source="Scaling Book, TPU specs"),
    "rtx3090": dict(label="RTX 3090", hbm_bytes=24 * 2**30, hbm_bw=9.36e11, bf16_flops=7.1e13,
                    int8_ops=None, ici_oneway_per_link=None,
                    source="NVIDIA GA102 whitepaper (nominal; bf16 dense, fp32 accumulate)"),
    "other": dict(label=DEV0.device_kind, hbm_bytes=None, hbm_bw=None, bf16_flops=None,
                  int8_ops=None, ici_oneway_per_link=None, source="none: use E0 measurements"),
}


def detect_hw(dev):
    kind = dev.device_kind.lower()
    if "v5 lite" in kind or "v5e" in kind or "v5litepod" in kind:
        return "tpu_v5e"
    if "3090" in kind:
        return "rtx3090"
    return "other"


HW_KEY = detect_hw(DEV0)
SPEC = HW_SPECS[HW_KEY]
B_CRIT_SPEC = SPEC["bf16_flops"] / SPEC["hbm_bw"] if SPEC["hbm_bw"] else float("nan")
print(f"hardware: {HW_KEY} ({DEV0.device_kind}) x{len(DEVICES)}   spec B_crit = {B_CRIT_SPEC:.0f}")


def mem_stats(dev=DEV0):
    try:
        return dev.memory_stats() or {}
    except Exception:
        return {}


HBM_LIMIT = mem_stats().get("bytes_limit") or SPEC["hbm_bytes"]
print("usable device memory (bytes_limit):", HBM_LIMIT,
      f"({HBM_LIMIT / 2**30:.2f} GiB)" if HBM_LIMIT else "")

# %% [markdown]
# ## 4. Model config and theory
# All formulas follow the Scaling Book inference chapter, with the lecture's notation:
# L layers, D width, F MLP width, N query heads, K KV heads, H head dim, V vocab.

# %%
from huggingface_hub import snapshot_download

MODEL_DIR = snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.safetensors", "*.txt"])


@dataclass(frozen=True)
class Cfg:
    L: int
    D: int
    F: int
    N: int
    K: int
    H: int
    V: int
    eps: float
    theta: float
    tied: bool

    @property
    def G(self):  # query heads per KV head
        return self.N // self.K


def load_cfg(model_dir):
    c = json.load(open(os.path.join(model_dir, "config.json")))
    assert c.get("rope_scaling") in (None, {}), "rope scaling not implemented"
    assert not c.get("use_sliding_window", False), "sliding window not implemented"
    return Cfg(L=c["num_hidden_layers"], D=c["hidden_size"], F=c["intermediate_size"],
               N=c["num_attention_heads"], K=c["num_key_value_heads"],
               H=c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
               V=c["vocab_size"], eps=c["rms_norm_eps"], theta=c["rope_theta"],
               tied=c.get("tie_word_embeddings", False))


CFG = load_cfg(MODEL_DIR)
BYTES = DTYPE.itemsize


def param_counts(c: Cfg):
    attn = c.D * c.N * c.H + 2 * c.D * c.K * c.H + c.N * c.H * c.D + 2 * c.H  # q,k,v,o + qk-norm
    mlp = 3 * c.D * c.F
    layer = attn + mlp + 2 * c.D
    emb = c.V * c.D
    head = 0 if c.tied else c.V * c.D
    total = c.L * layer + emb + head + c.D
    matmul = c.L * (attn - 2 * c.H + mlp) + c.V * c.D  # weights used in matmuls per token
    read = c.L * layer + c.V * c.D + c.D              # weights read per decode step
    return dict(total=total, layer=layer, embedding=emb, matmul=matmul, read_per_step=read)


PC = param_counts(CFG)
KV_PER_TOKEN = 2 * CFG.L * CFG.K * CFG.H * BYTES          # Scaling Book: 2 * bytes * H * K * L
ATTN_FLOPS_PER_TOKEN_POS = 4 * CFG.N * CFG.H * CFG.L       # QK^T and PV, per cached position


def decode_theory(B, S, c=CFG):
    """FLOPs and HBM bytes for one decode step: batch B, every sequence attending to S positions."""
    flops_dense = 2 * B * PC["matmul"]
    flops_attn = B * S * ATTN_FLOPS_PER_TOKEN_POS
    bytes_w = BYTES * PC["read_per_step"]
    bytes_kv = B * S * KV_PER_TOKEN
    return dict(flops_dense=flops_dense, flops_attn=flops_attn, flops=flops_dense + flops_attn,
                bytes_w=bytes_w, bytes_kv=bytes_kv, bytes=bytes_w + bytes_kv)


def step_time_models(th, C, BW):
    """(A) per-op roofline: matmul and attention times add (Scaling Book T_step).
       (B) aggregate roofline: perfect overlap across the whole step."""
    tA = max(th["flops_attn"] / C, th["bytes_kv"] / BW) + max(th["flops_dense"] / C, th["bytes_w"] / BW)
    tB = max(th["flops"] / C, th["bytes"] / BW)
    return tA, tB


def prefill_theory(B, T, c=CFG):
    """Lecture formula (dense square attention, only the last position is unembedded)."""
    flops = c.L * (4 * B * T * c.D * c.H * (c.N + c.K) + 6 * B * T * c.D * c.F
                   + 4 * B * c.N * T * T * c.H) + 2 * B * c.D * c.V
    bytes_ = BYTES * PC["read_per_step"] + B * T * KV_PER_TOKEN
    return dict(flops=flops, bytes=bytes_)


def memory_theory(B, S):
    return BYTES * PC["total"] + B * S * KV_PER_TOKEN


def max_batch(S, limit=HBM_LIMIT):
    return (limit - BYTES * PC["total"]) / (S * KV_PER_TOKEN) if limit else float("nan")


print(CFG)
print(f"params total {PC['total']/1e9:.3f} B | weight bytes {BYTES*PC['total']/1e9:.2f} GB | "
      f"KV/token {KV_PER_TOKEN} B ({KV_PER_TOKEN/1024:.0f} KiB) | "
      f"dense FLOPs/token {2*PC['matmul']/1e9:.2f} G")

# %%
def prediction_table(Bs, Ss, C, BW):
    rows = []
    for S in Ss:
        for B in Bs:
            th = decode_theory(B, S)
            tA, tB = step_time_models(th, C, BW)
            rows.append(dict(S=S, B=B, mem_GiB=memory_theory(B, S) / 2**30,
                             fits=memory_theory(B, S) <= (HBM_LIMIT or float("inf")),
                             tA_ms=1e3 * tA, tokps_A=B / tA, tB_ms=1e3 * tB,
                             attn_share_A=max(th["flops_attn"] / C, th["bytes_kv"] / BW) / tA))
    return pd.DataFrame(rows)


if SPEC["hbm_bw"]:
    print(f"Spec-based predictions ({SPEC['label']}): B_crit = {B_CRIT_SPEC:.0f}, "
          f"weight-load floor = {1e3*BYTES*PC['read_per_step']/SPEC['hbm_bw']:.2f} ms")
    for S in G["decode_S"]:
        print(f"  S={S:5d}: max batch before OOM ~ {max_batch(S):7.0f}   "
              f"KV-bandwidth throughput cap = {SPEC['hbm_bw']/(S*KV_PER_TOKEN):9.0f} tok/s")
    PRED_SPEC = prediction_table([1, 16, 64, 128, 256, 512], G["decode_S"], SPEC["bf16_flops"], SPEC["hbm_bw"])
    PRED_SPEC.to_csv(os.path.join(RAW_DIR, "theory_spec_predictions.csv"), index=False)
    try:
        from IPython.display import display
        display(PRED_SPEC.round(2))
    except ImportError:
        print(PRED_SPEC.round(2))

# %% [markdown]
# ## 5. Qwen3 in JAX
# The forward pass mirrors `transformers.models.qwen3.modeling_qwen3`:
# * RMSNorm computed in fp32;
# * per-head RMSNorm on Q and K (QK-norm);
# * RoPE with the rotate-half convention;
# * GQA, with each block of G = N/K consecutive query heads sharing one KV head;
# * SwiGLU MLP and a tied unembedding.
#
# **Design choices that keep the code honest to the theory:**
# * The KV cache is one static `[B, K, S, H]` buffer per layer and per K/V, updated in place with
#   `dynamic_update_slice` (the cache argument is donated). The KV-head-major layout is what the
#   attention matmuls want, so XLA never has to transpose the cache.
# * Layers are an unrolled Python loop over per-layer weights, as in HF. With `lax.scan` over
#   stacked weights and caches, XLA on GPU materialized each layer's cache slice before the
#   matmul, about 3× the KV traffic. Unrolling avoids any slicing, and it also lets XLA's cost
#   analysis count every layer (it counts a loop body only once).
# * Attention is a *grouped* einsum, so each KV head is read once, not copied out to N heads.
# * The decode loop runs on device (`fori_loop`), so Python dispatch is outside the timed region.

# %%
def rms_norm(x, w, eps):
    xf = x.astype(jnp.float32)
    xf = xf * lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return w * xf.astype(x.dtype)


def rope_tables(positions, H, theta):
    inv = 1.0 / (theta ** (jnp.arange(0, H, 2, dtype=jnp.float32) / H))
    f = positions.astype(jnp.float32)[:, None] * inv[None, :]
    emb = jnp.concatenate([f, f], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)  # [T, H] fp32


def apply_rope(x, cos, sin):  # x: [B, T, heads, H]
    c = cos[None, :, None, :].astype(x.dtype)
    s = sin[None, :, None, :].astype(x.dtype)
    h = x.shape[-1] // 2
    rot = jnp.concatenate([-x[..., h:], x[..., :h]], axis=-1)
    return x * c + rot * s


def attention(q, ck, cv, mask, cfg, impl):
    """q: [B,T,N,H]; ck/cv: [B,K,S,H] (this layer's cache); mask: [T,S] bool."""
    B, T, N, H = q.shape
    scale = H ** -0.5
    neg = jnp.finfo(jnp.float32).min
    if impl == "grouped":
        qg = q.reshape(B, T, cfg.K, cfg.G, H)
        s = jnp.einsum("btkgh,bksh->bkgts", qg, ck, preferred_element_type=jnp.float32) * scale
        s = jnp.where(mask[None, None, None], s, neg)
        p = jax.nn.softmax(s, axis=-1).astype(q.dtype)
        o = jnp.einsum("bkgts,bksh->btkgh", p, cv)
        return o.reshape(B, T, N * H)
    if impl == "repeat":  # what HF's eager path does: materialize K/V for every query head
        kr = jnp.repeat(ck, cfg.G, axis=1)
        vr = jnp.repeat(cv, cfg.G, axis=1)
        s = jnp.einsum("btnh,bnsh->bnts", q, kr, preferred_element_type=jnp.float32) * scale
        s = jnp.where(mask[None, None], s, neg)
        p = jax.nn.softmax(s, axis=-1).astype(q.dtype)
        return jnp.einsum("bnts,bnsh->btnh", p, vr).reshape(B, T, N * H)
    raise ValueError(impl)


def forward(params, tokens, cache, pos0, *, cfg, attn_impl="grouped", logits="last",
            cache_sharding=None):
    """tokens [B,T] int32 at positions pos0..pos0+T-1. cache = {'k': [L x [B,K,S,H]], 'v': ...}.
    attn_impl='none' skips reading the cache (used to measure the attention share)."""
    B, T = tokens.shape
    S = cache["k"][0].shape[2]
    x = params["embed"][tokens]
    positions = pos0 + jnp.arange(T, dtype=jnp.int32)
    cos, sin = rope_tables(positions, cfg.H, cfg.theta)
    mask = jnp.arange(S, dtype=jnp.int32)[None, :] <= positions[:, None]
    new_k, new_v = [], []
    for i, lp in enumerate(params["layers"]):  # unrolled: every layer has its own buffers
        h = rms_norm(x, lp["ln1"], cfg.eps)
        q = (h @ lp["wq"]).reshape(B, T, cfg.N, cfg.H)
        k = (h @ lp["wk"]).reshape(B, T, cfg.K, cfg.H)
        v = (h @ lp["wv"]).reshape(B, T, cfg.K, cfg.H)
        q = apply_rope(rms_norm(q, lp["q_norm"], cfg.eps), cos, sin)
        k = apply_rope(rms_norm(k, lp["k_norm"], cfg.eps), cos, sin)
        # in-place write of the T new positions into this layer's [B,K,S,H] cache
        ck = lax.dynamic_update_slice(cache["k"][i], k.transpose(0, 2, 1, 3).astype(cache["k"][i].dtype),
                                      (0, 0, pos0, 0))
        cv = lax.dynamic_update_slice(cache["v"][i], v.transpose(0, 2, 1, 3).astype(cache["v"][i].dtype),
                                      (0, 0, pos0, 0))
        if cache_sharding is not None:
            ck = lax.with_sharding_constraint(ck, cache_sharding)
            cv = lax.with_sharding_constraint(cv, cache_sharding)
        new_k.append(ck)
        new_v.append(cv)
        if attn_impl == "none":
            o = jnp.repeat(v, cfg.G, axis=2).reshape(B, T, cfg.N * cfg.H)
        else:
            o = attention(q, ck, cv, mask, cfg, attn_impl)
        x = x + o @ lp["wo"]
        h = rms_norm(x, lp["ln2"], cfg.eps)
        x = x + (jax.nn.silu(h @ lp["wg"]) * (h @ lp["wu"])) @ lp["wd"]
    ck, cv = new_k, new_v
    x = rms_norm(x, params["final_norm"], cfg.eps)
    if logits == "last":
        x = x[:, -1:]
    W = params["embed"] if cfg.tied else params["lm_head"]  # [V, D]
    out = jnp.einsum("btd,vd->btv", x, W, preferred_element_type=jnp.float32)
    return out, {"k": ck, "v": cv}


def decode_loop(params, tok, cache, pos, n_steps, *, cfg, attn_impl="grouped", advance=False,
                cache_sharding=None):
    """n_steps greedy decode steps entirely on device. advance=False keeps the context length
    fixed at `pos + 1` (used to time a step at an exact context S); advance=True generates."""
    def step(i, carry):
        tok, cache = carry
        p = pos + i if advance else pos
        logits, cache = forward(params, tok, cache, p, cfg=cfg, attn_impl=attn_impl,
                                cache_sharding=cache_sharding)
        return jnp.argmax(logits[:, -1], axis=-1).astype(jnp.int32)[:, None], cache
    return lax.fori_loop(0, n_steps, step, (tok, cache))


# %% [markdown]
# ### Loading weights from Hugging Face safetensors
# Tensors are read and stacked on the host CPU first, then placed on the accelerator. This avoids
# holding two copies of the weights in device memory.

# %%
from safetensors import safe_open

LAYER_KEYS = {  # our name -> (HF suffix, transpose?)
    "ln1": ("input_layernorm.weight", False), "ln2": ("post_attention_layernorm.weight", False),
    "wq": ("self_attn.q_proj.weight", True), "wk": ("self_attn.k_proj.weight", True),
    "wv": ("self_attn.v_proj.weight", True), "wo": ("self_attn.o_proj.weight", True),
    "q_norm": ("self_attn.q_norm.weight", False), "k_norm": ("self_attn.k_norm.weight", False),
    "wg": ("mlp.gate_proj.weight", True), "wu": ("mlp.up_proj.weight", True),
    "wd": ("mlp.down_proj.weight", True),
}


def load_params_host(model_dir, cfg, dtype):
    raw = {}
    with jax.default_device(CPU):
        for fn in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            with safe_open(fn, framework="flax") as f:
                for k in f.keys():
                    raw[k] = f.get_tensor(k)
        layers = []
        for i in range(cfg.L):
            lp = {}
            for name, (suffix, transpose) in LAYER_KEYS.items():
                t = raw.pop(f"model.layers.{i}.{suffix}")
                lp[name] = (t.T if transpose else t).astype(dtype)
            layers.append(lp)
        params = {"embed": raw.pop("model.embed_tokens.weight").astype(dtype),
                  "final_norm": raw.pop("model.norm.weight").astype(dtype), "layers": layers}
        if not cfg.tied:
            params["lm_head"] = raw.pop("lm_head.weight").astype(dtype)
        raw.pop("lm_head.weight", None)
    assert not raw, f"unused tensors: {list(raw)[:5]}"
    return params


CACHE_TP_SPEC = P(None, "tp", None, None)  # [B, K, S, H]: one KV head per device when K = 8


def tp_param_specs(cfg):
    col, row = P(None, "tp"), P("tp", None)  # weights are stored [in, out]
    layer = dict(ln1=P(), ln2=P(), q_norm=P(), k_norm=P(), wq=col, wk=col, wv=col, wo=row,
                 wg=col, wu=col, wd=row)
    specs = {"embed": P("tp", None), "final_norm": P(), "layers": [dict(layer) for _ in range(cfg.L)]}
    if not cfg.tied:
        specs["lm_head"] = P("tp", None)
    return specs


def place(tree, devices=None):
    """devices=None -> first accelerator; a Mesh -> tensor-parallel sharding."""
    if isinstance(devices, Mesh):
        shard = jax.tree.map(lambda s: NamedSharding(devices, s), tp_param_specs(CFG),
                             is_leaf=lambda s: isinstance(s, P))
        return jax.device_put(tree, shard)
    return jax.device_put(tree, devices or DEV0)


def tree_bytes(tree):
    return sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(tree))


t0 = time.perf_counter()
PARAMS_HOST = load_params_host(MODEL_DIR, CFG, DTYPE)
n_loaded = sum(x.size for x in jax.tree.leaves(PARAMS_HOST))
print(f"loaded {n_loaded/1e9:.3f} B params in {time.perf_counter()-t0:.1f}s")
assert n_loaded == PC["total"], (n_loaded, PC["total"])  # theory parameter count is exact
PARAMS = place(PARAMS_HOST)
jax.block_until_ready(PARAMS)
print("on device:", DEV0, "| bytes_in_use:", mem_stats().get("bytes_in_use"))

# %% [markdown]
# ## 6. Correctness check (E-check)
# 1. **Exact check.** Our JAX model and HF `transformers` (PyTorch) both run in fp32 on CPU, on the
#    same prompt. Logits should agree to ~1e-4.
# 2. **Accelerator check.** Our bf16 model on the accelerator against the HF fp32 reference
#    (top-1 agreement and the size of the gap).
# 3. **Sanity check.** A short greedy generation decoded to text.
#
# Needs `torch` and `transformers` on the host. If they are missing, step 1 is skipped. For
# Qwen3-4B the fp32 CPU check needs roughly 40 GB of host RAM and a few CPU minutes; set
# `RUN_CHECK=0` to skip it.

# %%
def new_cache(B, S, cfg=CFG, dtype=DTYPE, sharding=None, device=None):
    """Per-layer KV buffers {'k': [L x [B,K,S,H]], 'v': [...]}."""
    shape = (B, cfg.K, S, cfg.H)
    if sharding is not None:
        make = jax.jit(lambda: jnp.zeros(shape, dtype), out_shardings=sharding)
        return {"k": [make() for _ in range(cfg.L)], "v": [make() for _ in range(cfg.L)]}
    dev = device or DEV0
    return {kv: [jnp.zeros(shape, dtype, device=dev) for _ in range(cfg.L)] for kv in ("k", "v")}


CHECK = {}
if RUN_CHECK:
    PROMPT = "The capital of France is Paris. The capital of Japan is"
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        HAVE_TORCH = True
    except Exception as e:
        HAVE_TORCH = False
        print("torch/transformers unavailable, skipping the exact check:", e)

    if HAVE_TORCH:
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        ids = tok(PROMPT, return_tensors="np")["input_ids"].astype(np.int32)
        try:  # transformers >= 4.56 uses `dtype`; older versions use `torch_dtype`
            hf = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
        except TypeError:
            hf = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
        hf = hf.float()
        hf.eval()
        with torch.no_grad():
            ref = hf(torch.from_numpy(ids.astype(np.int64))).logits[0].float().numpy()
        del hf
        gc.collect()

        # (1) fp32 JAX on CPU vs fp32 HF
        with jax.default_device(CPU):
            p32 = jax.tree.map(lambda x: x.astype(jnp.float32), PARAMS_HOST)
            c32 = new_cache(1, ids.shape[1], dtype=jnp.float32, device=CPU)
            out32, _ = jax.jit(partial(forward, cfg=CFG, logits="all"))(p32, jnp.asarray(ids), c32, 0)
            out32 = np.asarray(out32[0])
            del p32, c32
        CHECK["fp32_max_abs_diff"] = float(np.abs(out32 - ref).max())
        CHECK["fp32_top1_agree"] = float((out32.argmax(-1) == ref.argmax(-1)).mean())

        # (2) bf16 on the accelerator vs fp32 HF
        cache = new_cache(1, ids.shape[1])
        outb, _ = jax.jit(partial(forward, cfg=CFG, logits="all"))(PARAMS, jnp.asarray(ids), cache, 0)
        outb = np.asarray(outb[0], dtype=np.float32)
        CHECK["bf16_max_abs_diff"] = float(np.abs(outb - ref).max())
        CHECK["bf16_top1_agree"] = float((outb.argmax(-1) == ref.argmax(-1)).mean())
        CHECK["ref_logit_std"] = float(ref.std())

        # (3) greedy generation through the cached decode path
        Tp, n_new = ids.shape[1], 12
        cache = new_cache(1, Tp + n_new)
        logits, cache = jax.jit(partial(forward, cfg=CFG))(PARAMS, jnp.asarray(ids), cache, 0)
        first = jnp.argmax(logits[:, -1], -1).astype(jnp.int32)[:, None]
        gen = [int(first[0, 0])]
        step = jax.jit(partial(forward, cfg=CFG))
        t = first
        for i in range(n_new - 1):
            logits, cache = step(PARAMS, t, cache, Tp + i)
            t = jnp.argmax(logits[:, -1], -1).astype(jnp.int32)[:, None]
            gen.append(int(t[0, 0]))
        CHECK["greedy_text"] = PROMPT + tok.decode(gen)
        # the first generated token must match HF's argmax at the last prompt position
        CHECK["first_token_matches_hf"] = bool(gen[0] == int(ref[-1].argmax()))

    print(json.dumps(CHECK, indent=2))
    with open(os.path.join(RAW_DIR, "correctness.json"), "w") as f:
        json.dump(CHECK, f, indent=2)
    if "fp32_max_abs_diff" in CHECK:
        assert CHECK["fp32_max_abs_diff"] < 5e-3, "fp32 logits disagree with Hugging Face"
        assert CHECK["fp32_top1_agree"] == 1.0
        assert CHECK["first_token_matches_hf"]

# %% [markdown]
# ## 7. Measurement harness
# * Each condition is compiled once (ahead of time), warmed up, then timed `repeats` times. A timed
#   call runs `steps` decode steps inside one compiled loop and is closed by `block_until_ready`.
# * Out-of-memory errors are recorded as results. Larger batches at the same context are then
#   skipped.
# * Rows are appended to CSV immediately, and finished keys are skipped on re-run.

# %%
def raw_path(name):  # one JSON object per line: rows may have different fields (ok vs oom)
    return os.path.join(RAW_DIR, f"{name}.jsonl")


def read_raw(name):
    p = raw_path(name)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    return pd.read_json(p, lines=True)


def done_keys(name, cols):
    df = read_raw(name)
    if df is None:
        return set()
    return set(map(tuple, df[cols].astype(str).values.tolist()))


def append_row(name, row):
    with open(raw_path(name), "a") as f:
        f.write(json.dumps(row, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")


def is_oom(e):
    s = str(e)
    return any(m in s for m in ("RESOURCE_EXHAUSTED", "Out of memory", "out of memory",
                                "Resource exhausted", "OOM"))


def cost_numbers(compiled):
    try:
        ca = compiled.cost_analysis()
        ca = ca[0] if isinstance(ca, (list, tuple)) else ca
        return dict(xla_flops=ca.get("flops"), xla_bytes=ca.get("bytes accessed"))
    except Exception:
        return dict(xla_flops=None, xla_bytes=None)


def memory_numbers(compiled):
    try:
        m = compiled.memory_analysis()
        return dict(xla_arg_bytes=m.argument_size_in_bytes, xla_out_bytes=m.output_size_in_bytes,
                    xla_temp_bytes=m.temp_size_in_bytes, xla_alias_bytes=m.alias_size_in_bytes)
    except Exception:
        return {}


def timed_runs(fn, args_fn, repeats, steps):
    """fn(*args) -> (tok, cache); args_fn(prev_out) builds the next args (threading donated state)."""
    times = []
    out = None
    for _ in range(repeats):
        args = args_fn(out)
        t = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t) / steps)
    a = np.array(times)
    return dict(step_ms_median=1e3 * np.median(a), step_ms_p25=1e3 * np.percentile(a, 25),
                step_ms_p75=1e3 * np.percentile(a, 75), step_ms_min=1e3 * a.min()), out


_DECODE_FNS = {}


def decode_fn(attn_impl="grouped", advance=False, cache_sharding=None, donate=True):
    key = (attn_impl, advance, cache_sharding, donate)
    if key not in _DECODE_FNS:
        f = partial(decode_loop, cfg=CFG, attn_impl=attn_impl, advance=advance,
                    cache_sharding=cache_sharding)
        _DECODE_FNS[key] = jax.jit(f, static_argnames=("n_steps",),
                                   donate_argnames=("cache",) if donate else ())
    return _DECODE_FNS[key]


def bench_decode(B, S, steps, repeats, params=None, attn_impl="grouped", mesh=None):
    """Time one decode step at batch B with every sequence attending to exactly S positions."""
    params = params if params is not None else PARAMS
    cache_sh = NamedSharding(mesh, CACHE_TP_SPEC) if mesh is not None else None
    row = dict(B=B, S=S, steps=steps, repeats=repeats, attn_impl=attn_impl,
               n_chips=len(mesh.devices.flat) if mesh is not None else 1)
    th = decode_theory(B, S)
    row.update(theory_flops=th["flops"], theory_bytes=th["bytes"], theory_bytes_kv=th["bytes_kv"],
               theory_mem_bytes=memory_theory(B, S))
    cache = tok = None
    # peak_bytes_in_use is a process-wide high-water mark that never resets. It is this condition's
    # true peak only if the condition raised it (peak_is_new); otherwise it is just an upper bound.
    peak_before = mem_stats().get("peak_bytes_in_use")
    try:
        cache = new_cache(B, S, sharding=cache_sh)
        tok = jnp.zeros((B, 1), jnp.int32)
        if cache_sh is not None:
            tok = jax.device_put(tok, NamedSharding(mesh, P()))
        row["bytes_in_use_after_alloc"] = mem_stats().get("bytes_in_use")
        pos = jnp.int32(S - 1)
        fn = decode_fn(attn_impl, False, cache_sh)
        t = time.perf_counter()
        compiled = fn.lower(params, tok, cache, pos, n_steps=steps).compile()
        row["compile_s"] = time.perf_counter() - t
        row.update(memory_numbers(compiled))
        tok, cache = compiled(params, tok, cache, pos)  # warm-up
        jax.block_until_ready(tok)
        state = {"tok": tok, "cache": cache}

        def args_fn(out):
            if out is not None:
                state["tok"], state["cache"] = out
            return params, state["tok"], state["cache"], pos

        stats, out = timed_runs(compiled, args_fn, repeats, steps)
        peak = mem_stats().get("peak_bytes_in_use")
        row.update(stats, status="ok", peak_bytes_in_use=peak,
                   peak_is_new=bool(peak is not None and peak_before is not None and peak > peak_before))
        del out, state
    except Exception as e:
        if not is_oom(e):
            raise
        row.update(status="oom", error=str(e).splitlines()[0][:200])
    finally:
        del cache, tok
        gc.collect()
    return row


def sweep(name, grid, run_one, key_cols=("B", "S")):
    """grid: list of (B, S) sorted by S then B. After an OOM, larger B at the same S is skipped."""
    done = done_keys(name, list(key_cols))
    oom_at = {}
    prev = read_raw(name)
    if prev is not None:
        for _, r in prev[prev.status != "ok"].iterrows():
            oom_at[r[key_cols[1]]] = min(oom_at.get(r[key_cols[1]], 1 << 30), r[key_cols[0]])
    for B, S in grid:
        if (str(B), str(S)) in done:
            continue
        if B >= oom_at.get(S, 1 << 30):
            append_row(name, dict(B=B, S=S, status="skipped_after_oom"))
            continue
        row = run_one(B, S)
        row["exp"] = name
        append_row(name, row)
        if row["status"] != "ok":
            oom_at[S] = B
        ms = row.get("step_ms_median", row.get("decode_ms_per_step", float("nan")))
        extra = f"  compile {row['compile_s']:.1f}s" if "compile_s" in row else ""
        print(f"{name}: B={B:4d} S={S:5d} -> {row['status']:4s} {ms:9.3f} ms/step{extra}")
    return read_raw(name)


# %% [markdown]
# ## 8. E0: measured ceilings
# Achievable bf16 matmul FLOP/s and device-memory bandwidth on one device. These replace the
# nominal spec numbers when fitting the step-time models.

# %%
def e0_ceilings():
    p = os.path.join(RAW_DIR, "e0_ceilings.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    plat = DEV0.platform
    n = 8192 if plat in ("tpu", "gpu") else 1024
    nbytes = (2 * 2**30) if plat in ("tpu", "gpu") else (256 * 2**20)
    rows = []

    k1, k2 = jax.random.split(jax.random.key(0))
    a = jax.random.normal(k1, (n, n), DTYPE)
    b = jax.random.normal(k2, (n, n), DTYPE)
    mm = jax.jit(lambda a, b: jnp.dot(a, b, preferred_element_type=jnp.float32).astype(DTYPE))
    jax.block_until_ready(mm(a, b))
    ts = []
    for _ in range(10):
        t = time.perf_counter()
        jax.block_until_ready(mm(a, b))
        ts.append(time.perf_counter() - t)
    rows.append(dict(what="matmul_flops", value=2 * n**3 / np.median(ts), size=n))
    del a, b

    x = jnp.ones((nbytes // BYTES,), DTYPE)
    rd = jax.jit(lambda x: jnp.sum(x, dtype=jnp.float32))
    cp = jax.jit(lambda x: x * 2)
    for name, f, moved in (("hbm_read_bw", rd, nbytes), ("hbm_copy_bw", cp, 2 * nbytes)):
        jax.block_until_ready(f(x))
        ts = []
        for _ in range(10):
            t = time.perf_counter()
            jax.block_until_ready(f(x))
            ts.append(time.perf_counter() - t)
        rows.append(dict(what=name, value=moved / np.median(ts), size=nbytes))
    del x
    df = pd.DataFrame(rows)
    df.to_csv(p, index=False)
    return df


if RUN_E0:
    E0 = e0_ceilings()
    print(E0)
    C_MEAS = float(E0.loc[E0.what == "matmul_flops", "value"].iloc[0])
    BW_MEAS = float(E0.loc[E0.what.str.startswith("hbm"), "value"].max())
    print(f"measured: {C_MEAS/1e12:.1f} TFLOP/s, {BW_MEAS/1e9:.0f} GB/s -> B_crit ~ {C_MEAS/BW_MEAS:.0f}"
          + (f"   (spec: {SPEC['bf16_flops']/1e12:.0f} TFLOP/s, {SPEC['hbm_bw']/1e9:.0f} GB/s)"
             if SPEC["hbm_bw"] else ""))

# %% [markdown]
# ## 9. E1: decode step, batch × context, 1 device
# The headline sweep. For each context S we allocate a cache of length S and time steps at position
# S−1, so every sequence attends to exactly S positions. E3 checks this shortcut against real
# generation.

# %%
if RUN_E1:
    grid = [(B, S) for S in G["decode_S"] for B in G["decode_B"]]
    E1 = sweep("e1_decode", grid, lambda B, S: bench_decode(B, S, G["steps"], G["repeats"]))

# %% [markdown]
# ## 10. E2: prefill
# Process T prompt tokens for B sequences in one pass (dense causal attention, last-position
# logits). We expect this to be compute-bound once B·T ≳ B_crit.

# %%
_PREFILL = jax.jit(partial(forward, cfg=CFG), donate_argnames=("cache",))


def bench_prefill(B, T):
    th = prefill_theory(B, T)
    row = dict(B=B, S=T, T=T, theory_flops=th["flops"], theory_bytes=th["bytes"],
               theory_mem_bytes=memory_theory(B, T))
    cache = None
    try:
        cache = new_cache(B, T)
        toks = jax.random.randint(jax.random.key(B * 7919 + T), (B, T), 0, CFG.V, jnp.int32)
        t = time.perf_counter()
        compiled = _PREFILL.lower(PARAMS, toks, cache, jnp.int32(0)).compile()
        row["compile_s"] = time.perf_counter() - t
        row.update(memory_numbers(compiled), **cost_numbers(compiled))
        out = compiled(PARAMS, toks, cache, jnp.int32(0))
        jax.block_until_ready(out)
        state = {"cache": out[1]}

        def args_fn(o):
            if o is not None:
                state["cache"] = o[1]
            return PARAMS, toks, state["cache"], jnp.int32(0)

        stats, out = timed_runs(compiled, args_fn, max(G["repeats"], 3), 1)
        row.update(stats, status="ok", tokens_per_s=B * T / (stats["step_ms_median"] / 1e3))
        del out, state
    except Exception as e:
        if not is_oom(e):
            raise
        row.update(status="oom", error=str(e).splitlines()[0][:200])
    finally:
        del cache
        gc.collect()
    return row


if RUN_E2:
    grid = [(B, T) for T in G["prefill_T"] for B in G["prefill_B"]]
    E2 = sweep("e2_prefill", grid, bench_prefill)

# %% [markdown]
# ## 11. E3: end-to-end requests
# Prefill a prompt of length T for B requests, then generate n tokens with a growing position.
# Reports time to first token, total latency and throughput. The static cache always spans T + n
# positions, as HF's `StaticCache` does.

# %%
def bench_e2e(B, T):
    n = G["e2e_gen"]
    row = dict(B=B, S=T + n, prompt=T, gen=n)
    cache = None
    try:
        cache = new_cache(B, T + n)
        toks = jax.random.randint(jax.random.key(B), (B, T), 0, CFG.V, jnp.int32)
        gen = decode_fn("grouped", advance=True)
        cp = _PREFILL.lower(PARAMS, toks, cache, jnp.int32(0)).compile()
        tok0 = jnp.zeros((B, 1), jnp.int32)
        cg = gen.lower(PARAMS, tok0, cache, jnp.int32(T), n_steps=n).compile()
        ttft, total = [], []
        for r in range(G["repeats"] + 1):  # first repeat is warm-up
            t0 = time.perf_counter()
            logits, cache = cp(PARAMS, toks, cache, jnp.int32(0))
            first = jnp.argmax(logits[:, -1], -1).astype(jnp.int32)[:, None]
            jax.block_until_ready(first)
            t1 = time.perf_counter()
            tok, cache = cg(PARAMS, first, cache, jnp.int32(T))
            jax.block_until_ready(tok)
            t2 = time.perf_counter()
            if r:
                ttft.append(t1 - t0)
                total.append(t2 - t0)
        ttft, total = np.median(ttft), np.median(total)
        row.update(status="ok", ttft_ms=1e3 * ttft, total_ms=1e3 * total,
                   decode_ms_per_step=1e3 * (total - ttft) / n,
                   throughput_tok_s=B * n / (total - ttft), e2e_tok_s=B * n / total)
    except Exception as e:
        if not is_oom(e):
            raise
        row.update(status="oom", error=str(e).splitlines()[0][:200])
    finally:
        del cache
        gc.collect()
    return row


if RUN_E3:
    grid = [(B, G["e2e_prompt"]) for B in G["e2e_B"]]
    E3 = sweep("e3_e2e", grid, bench_e2e)

# %% [markdown]
# ## 12. E4: where the time goes
# * **Attention share.** The same decode step with `attn_impl='none'`, which skips reading the KV
#   cache while keeping every matmul. Attention share = 1 − t_none / t_full.
# * **XLA cost analysis.** XLA's own FLOP and byte counts for one decode step (layers are unrolled,
#   so every layer is counted), compared with the theory.
# * **Profiler trace.** A trace of a few points is saved for TensorBoard/XProf or Perfetto.

# %%
def bench_attn_share(B, S):
    row = bench_decode(B, S, G["steps"], G["repeats"], attn_impl="none")
    return row


def bench_cost(B, S):
    th = decode_theory(B, S)
    row = dict(B=B, S=S, theory_flops=th["flops"], theory_bytes=th["bytes"],
               theory_flops_dense=th["flops_dense"], theory_flops_attn=th["flops_attn"],
               theory_bytes_w=th["bytes_w"], theory_bytes_kv=th["bytes_kv"])
    try:
        cache = jax.eval_shape(lambda: new_cache(B, S))  # shapes only: no device memory needed
        tok = jax.ShapeDtypeStruct((B, 1), jnp.int32)
        f = jax.jit(partial(forward, cfg=CFG))
        t = time.perf_counter()
        compiled = f.lower(PARAMS, tok, cache, jnp.int32(S - 1)).compile()
        row["compile_s"] = time.perf_counter() - t
        row.update(cost_numbers(compiled), status="ok")
    except Exception as e:
        if not is_oom(e):
            raise
        row.update(status="oom", error=str(e).splitlines()[0][:200])
    return row


if RUN_E4:
    E4a = sweep("e4_attn_share", sorted(G["attn_points"], key=lambda p: (p[1], p[0])), bench_attn_share)
    E4b = sweep("e4_cost", sorted(G["cost_points"], key=lambda p: (p[1], p[0])), bench_cost)
    trace_dir = os.path.join(OUT_DIR, "traces")
    if not os.path.isdir(trace_dir):
        for B, S in G["profile_points"]:
            try:
                with jax.profiler.trace(os.path.join(trace_dir, f"B{B}_S{S}")):
                    bench_decode(B, S, steps=4, repeats=1)
            except Exception as e:  # profiling is a nice-to-have; never let it kill the run
                print("profiler trace failed:", e)
    print(E4b[["B", "S", "theory_flops", "xla_flops", "theory_bytes", "xla_bytes"]])

# %% [markdown]
# ## 13. E5: implementation ablations
# * `repeat`: GQA by copying K/V out to all N query heads (HF eager style). In theory this means up
#   to G× the KV traffic, unless XLA fuses the copy away.
# * `dispatch_donated`: one Python dispatch per step, cache buffer donated (the usual HF-style loop).
# * `dispatch_copy`: one dispatch per step **without** donation, so XLA has to copy the whole cache
#   every step.

# %%
def bench_dispatch(B, S, donate):
    row = dict(B=B, S=S, steps=G["steps"], repeats=G["repeats"], attn_impl="grouped")
    cache = None
    try:
        cache = new_cache(B, S)
        tok = jnp.zeros((B, 1), jnp.int32)
        pos = jnp.int32(S - 1)
        fn = decode_fn("grouped", False, None, donate=donate)
        compiled = fn.lower(PARAMS, tok, cache, pos, n_steps=1).compile()
        tok, cache = compiled(PARAMS, tok, cache, pos)
        jax.block_until_ready(tok)
        times = []
        for _ in range(G["repeats"]):
            t = time.perf_counter()
            for _ in range(G["steps"]):
                tok, cache = compiled(PARAMS, tok, cache, pos)
            jax.block_until_ready(tok)
            times.append((time.perf_counter() - t) / G["steps"])
        a = np.array(times)
        row.update(status="ok", step_ms_median=1e3 * np.median(a),
                   step_ms_p25=1e3 * np.percentile(a, 25), step_ms_p75=1e3 * np.percentile(a, 75))
    except Exception as e:
        if not is_oom(e):
            raise
        row.update(status="oom", error=str(e).splitlines()[0][:200])
    finally:
        del cache
        gc.collect()
    return row


if RUN_E5:
    pts = sorted(G["ablation_points"], key=lambda p: (p[1], p[0]))
    variants = {
        "e5_repeat_kv": lambda B, S: bench_decode(B, S, G["steps"], G["repeats"], attn_impl="repeat"),
        "e5_dispatch_donated": lambda B, S: bench_dispatch(B, S, True),
        "e5_dispatch_copy": lambda B, S: bench_dispatch(B, S, False),
    }
    for name, fn in variants.items():
        sweep(name, pts, fn)

# %% [markdown]
# ## 14. Optional extension, E6: tensor parallel over 8 devices
# **Not part of the single-GPU experiment.** It is off by default (`RUN_E6=1` enables it) and only
# runs with at least 8 devices. On one machine it can be exercised on 8 fake CPU devices
# (`FAKE_CPU_DEVICES=8 RUN_E6=1`) to check that the sharded code path still matches the
# single-device model. The weights are split Megatron-style across 8 devices:
# * Q/K/V/gate/up are split by columns and O/down by rows.
# * The vocabulary is split for the embedding and unembedding.
# * The KV cache is split by KV head. With K = 8, each device holds exactly one KV head.
#
# Each layer then needs two all-reduces, so 72 collectives per decode step.

# %%
if RUN_E6 and len(DEVICES) >= 8:
    MESH = Mesh(np.array(DEVICES[:8]), ("tp",))
    assert CFG.K % 8 == 0 and CFG.F % 8 == 0 and CFG.V % 8 == 0
    del PARAMS
    gc.collect()
    PARAMS_TP = place(PARAMS_HOST, MESH)
    jax.block_until_ready(PARAMS_TP)
    grid = [(B, S) for S in G["tp_S"] for B in G["tp_B"]]
    E6 = sweep("e6_tp8", grid,
               lambda B, S: bench_decode(B, S, G["steps"], G["repeats"], params=PARAMS_TP, mesh=MESH))
    # sanity: TP output must equal single-device output on a short prompt
    ids = jnp.arange(1, 17, dtype=jnp.int32)[None]
    tp_logits, _ = jax.jit(partial(forward, cfg=CFG))(PARAMS_TP, ids, new_cache(
        1, 16, sharding=NamedSharding(MESH, CACHE_TP_SPEC)), 0)
    del PARAMS_TP
    gc.collect()
    PARAMS = place(PARAMS_HOST)
    one_logits, _ = jax.jit(partial(forward, cfg=CFG))(PARAMS, ids, new_cache(1, 16), 0)
    tp_np, one_np = np.asarray(tp_logits), np.asarray(one_logits)  # different devices -> host
    tp_diff = float(np.abs(tp_np - one_np).max())
    print("TP-8 vs 1-device max |logit diff|:", tp_diff)
    with open(os.path.join(RAW_DIR, "e6_tp_check.json"), "w") as f:
        json.dump({"max_abs_logit_diff": tp_diff,
                   "top1_equal": bool(tp_np[0, -1].argmax() == one_np[0, -1].argmax())}, f)
elif RUN_E6:
    print(f"E6 skipped: needs 8 devices, have {len(DEVICES)}")
else:
    print("E6 (optional 8-device extension) disabled; set RUN_E6=1 to enable")

# %% [markdown]
# ## 15. Figures
# This section only reads the CSVs, so it can also be re-run locally on downloaded results. Colours:
# context length S uses a single-hue sequential ramp (short = light, long = dark); model and variant
# comparisons use fixed categorical slots.

# %%
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984",
    "axes.labelcolor": "#2b2a27", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 5,
})
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#2b2a27"
MUTED = "#8a8984"


def s_colors(Ss):
    Ss = sorted(Ss)
    cmap = plt.get_cmap("Blues")
    return {S: cmap(0.35 + 0.6 * i / max(1, len(Ss) - 1)) for i, S in enumerate(Ss)}


def load(name):
    return read_raw(name)


def ceilings():
    p = os.path.join(RAW_DIR, "e0_ceilings.csv")
    if os.path.exists(p):
        e0 = pd.read_csv(p)
        return (float(e0.loc[e0.what == "matmul_flops", "value"].iloc[0]),
                float(e0.loc[e0.what.str.startswith("hbm"), "value"].max()))
    return SPEC["bf16_flops"], SPEC["hbm_bw"]


def plain(ax):
    """Plain-number tick labels on log axes (12, not 1.2x10^1)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
        if scale == "log":
            if axis.get_transform().base == 10:
                axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(NullFormatter())


def save(fig, name):
    for ax in fig.axes:
        plain(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), bbox_inches="tight")
    plt.show()
    plt.close(fig)


def add_predictions(df, C, BW):
    df = df.copy()
    pa, pb = [], []
    for _, r in df.iterrows():
        tA, tB = step_time_models(decode_theory(int(r.B), int(r.S)), C, BW)
        pa.append(1e3 * tA)
        pb.append(1e3 * tB)
    df["pred_A_ms"], df["pred_B_ms"] = pa, pb
    return df


def make_figures():
    C, BW = ceilings()
    e1 = load("e1_decode")
    if e1 is None:
        print("no E1 data yet")
        return
    ok = add_predictions(e1[e1.status == "ok"].astype({"B": int, "S": int}), C, BW)
    ok["tok_s"] = ok.B / (ok.step_ms_median / 1e3)
    oom = e1[e1.status == "oom"]
    Ss = sorted(ok.S.unique())
    col = s_colors(Ss)

    # Fig 1: decode step latency vs batch
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for S in Ss:
        d = ok[ok.S == S].sort_values("B")
        ax.fill_between(d.B, d.step_ms_p25, d.step_ms_p75, color=col[S], alpha=0.18, lw=0)
        ax.plot(d.B, d.step_ms_median, "-o", color=col[S], label=f"S = {S}")
        ax.plot(d.B, d.pred_A_ms, "--", color=col[S], lw=1)
    for _, r in oom.iterrows():  # OOM marked at the model-A time it would have taken
        tA = step_time_models(decode_theory(int(r.B), int(r.S)), C, BW)[0]
        ax.plot(r.B, 1e3 * tA, "x", color=col.get(r.S, MUTED), ms=8, mew=2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.axvline(C / BW, color=MUTED, lw=1, ls=":")
    ax.annotate(f"B_crit ≈ {C/BW:.0f}", (C / BW, 1), xycoords=("data", "axes fraction"),
                xytext=(-4, -4), textcoords="offset points", ha="right", va="top", color=MUTED, fontsize=8)
    ax.set_xlabel("batch size B (sequences)")
    ax.set_ylabel("decode step time (ms)")
    ax.set_title("Decode step time vs batch size", pad=16)
    ax.text(0, 1.01, "solid = measured (band = IQR), dashed = model A prediction, × = out of memory",
            transform=ax.transAxes, fontsize=7.5, color=MUTED, va="bottom")
    ax.legend(title="context", ncol=2, fontsize=8)
    save(fig, "fig1_decode_latency_vs_batch.png")

    # Fig 2: throughput vs per-token latency
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for S in Ss:
        d = ok[ok.S == S].sort_values("B")
        ax.plot(d.step_ms_median, d.tok_s, "-o", color=col[S], label=f"S = {S}")
        last = d.iloc[-1]
        ax.annotate(f"B={int(last.B)}", (last.step_ms_median, last.tok_s), fontsize=7, color=INK,
                    xytext=(4, 0), textcoords="offset points", va="center")
    ax.axhline(C / (2 * PC["matmul"]), color=MUTED, lw=1, ls=":")
    ax.annotate("compute cap C / 2P", (0, C / (2 * PC["matmul"])), xycoords=("axes fraction", "data"),
                xytext=(4, 3), textcoords="offset points", color=MUTED, fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("per-token latency = decode step time (ms)")
    ax.set_ylabel("throughput (generated tokens / s, all sequences)")
    ax.set_title("Throughput vs latency trade-off (each point is a batch size)")
    ax.legend(title="context", fontsize=8)
    save(fig, "fig2_throughput_vs_latency.png")

    # Fig 3: empirical roofline
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ok["intensity"] = ok.theory_flops / ok.theory_bytes
    ok["achieved"] = ok.theory_flops / (ok.step_ms_median / 1e3)
    xs = np.logspace(np.log10(max(0.5, ok.intensity.min() / 2)), np.log10(max(1e3, ok.intensity.max() * 2)), 200)
    ax.plot(xs, np.minimum(C, BW * xs) / 1e12, color=INK, lw=1.5, label="measured roofline (E0)")
    if SPEC["hbm_bw"]:
        ax.plot(xs, np.minimum(SPEC["bf16_flops"], SPEC["hbm_bw"] * xs) / 1e12, color=MUTED, lw=1, ls="--",
                label=f"spec roofline ({SPEC['label']})")
    for S in Ss:
        d = ok[ok.S == S]
        ax.plot(d.intensity, d.achieved / 1e12, "o", color=col[S], label=f"S = {S}", mec="white", mew=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (theory FLOPs / theory device-memory bytes)")
    ax.set_ylabel("achieved TFLOP/s (theory FLOPs / measured time)")
    ax.set_title("Empirical roofline of the decode step")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "fig3_roofline.png")

    # Fig 4: memory vs batch. Dots: XLA's compiled footprint of the timed program (arguments incl.
    # weights and cache, + outputs not aliased to donated inputs, + temporaries such as activations
    # and logits). Diamonds: measured allocator peak, only where the condition set a new
    # process-wide high-water mark (otherwise peak_bytes_in_use belongs to an earlier condition).
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for S in Ss:
        d = ok[ok.S == S].sort_values("B")
        if "xla_arg_bytes" in d and d.xla_arg_bytes.notna().any():
            foot = (d.xla_arg_bytes + d.xla_out_bytes.fillna(0) - d.xla_alias_bytes.fillna(0)
                    + d.xla_temp_bytes.fillna(0))
            ax.plot(d.B, foot / 2**30, "o", color=col[S], label=f"S = {S}")
        if "peak_is_new" in d:
            pk = d[d.peak_is_new.fillna(False).astype(bool) & d.peak_bytes_in_use.notna()]
            ax.plot(pk.B, pk.peak_bytes_in_use / 2**30, "D", color=col[S], mfc="none", ms=8, mew=1.2)
        Bs = np.unique(np.concatenate([d.B.values, oom[oom.S == S].B.values]))
        ax.plot(Bs, [memory_theory(B, S) / 2**30 for B in Bs], "-", color=col[S], lw=1)
        for _, r in oom[oom.S == S].iterrows():
            ax.plot(r.B, memory_theory(r.B, S) / 2**30, "x", color=col[S], ms=8, mew=2)
    if HBM_LIMIT:
        ax.axhline(HBM_LIMIT / 2**30, color=CAT[7], lw=1, ls="--")
        ax.annotate("device memory limit", (0, HBM_LIMIT / 2**30), xycoords=("axes fraction", "data"),
                    xytext=(4, -3), textcoords="offset points", va="top", color=INK, fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size B")
    ax.set_ylabel("GiB")
    ax.set_title("Device memory vs batch size", pad=28)
    ax.text(0, 1.01, "lines = weights + B·S·KV_per_token, dots = XLA compiled footprint (args + temp),\n"
            "◇ = measured allocator peak (new high-water marks only), × = out of memory",
            transform=ax.transAxes, fontsize=7.5, color=MUTED, va="bottom")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "fig4_memory_vs_batch.png")

    # Fig 5: attention share + model fit
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    e4 = load("e4_attn_share")
    if e4 is not None:
        e4 = e4[e4.status == "ok"]
        m = e4.merge(ok[["B", "S", "step_ms_median"]], on=["B", "S"], suffixes=("_none", ""))
        m["share"] = 1 - m.step_ms_median_none / m.step_ms_median
        for S in sorted(m.S.unique()):
            d = m[m.S == S].sort_values("B")
            pred = [max(decode_theory(B, S)["flops_attn"] / C, decode_theory(B, S)["bytes_kv"] / BW)
                    / step_time_models(decode_theory(B, S), C, BW)[0] for B in d.B]
            a1.plot(d.B, d.share, "-o", color=col.get(S, MUTED), label=f"S = {S}")
            a1.plot(d.B, pred, "--", color=col.get(S, MUTED), lw=1)
        a1.set_xscale("log", base=2)
        a1.set_ylim(-0.05, 1)
        a1.set_xlabel("batch size B")
        a1.set_ylabel("attention share of step time")
        a1.set_title("(a) Attention share. Solid: measured. Dashed: model A")
        a1.legend(fontsize=8)
    lim = [ok[["pred_A_ms", "pred_B_ms", "step_ms_median"]].min().min() * 0.8,
           ok[["pred_A_ms", "pred_B_ms", "step_ms_median"]].max().max() * 1.2]
    a2.plot(lim, lim, color=MUTED, lw=1)
    a2.plot(ok.pred_A_ms, ok.step_ms_median, "o", color=CAT[0], label="model A (per-op, additive)", ms=6)
    a2.plot(ok.pred_B_ms, ok.step_ms_median, "s", color=CAT[1], label="model B (aggregate overlap)",
            ms=9, mfc="none", mew=1.5)
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("predicted step time (ms, measured ceilings)")
    a2.set_ylabel("measured step time (ms)")
    a2.set_title("(b) Measured vs predicted (line = perfect prediction)")
    a2.legend(fontsize=8)
    save(fig, "fig5_attention_share_and_model_fit.png")

    # Fig 6: ablations (+ the optional tensor-parallel panel only when E6 data exists)
    e6 = load("e6_tp8")
    e6 = e6[e6.status == "ok"] if e6 is not None else None
    has_tp = e6 is not None and len(e6) > 0
    if has_tp:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    else:
        fig, a1 = plt.subplots(figsize=(6.4, 4.2))
    names = [("e1_decode", "grouped GQA, fused loop"), ("e5_repeat_kv", "repeat_kv GQA"),
             ("e5_dispatch_donated", "per-step dispatch"), ("e5_dispatch_copy", "per-step, no donation")]
    pts = G["ablation_points"]
    width = 0.8 / len(names)
    for j, (nm, lab) in enumerate(names):
        d = load(nm)
        if d is None:
            continue
        d = d[d.status == "ok"]
        vals = [d[(d.B == B) & (d.S == S)].step_ms_median.mean() for B, S in pts]
        a1.bar(np.arange(len(pts)) + j * width, vals, width * 0.92, color=CAT[j], label=lab)
    a1.set_xticks(np.arange(len(pts)) + width * (len(names) - 1) / 2)
    a1.set_xticklabels([f"B={B}\nS={S}" for B, S in pts])
    a1.set_ylabel("decode step time (ms)")
    a1.set_title(("(a) " if has_tp else "") + "Implementation ablations (lower is better)")
    a1.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    if not has_tp:
        save(fig, "fig6_ablations.png")
    else:
        for S in sorted(e6.S.unique()):
            c = col.get(S, MUTED)
            d6 = e6[e6.S == S].sort_values("B")
            a2.plot(d6.B, d6.B / (d6.step_ms_median / 1e3), "-o", color=c, label=f"8 devices, S = {S}")
            d1 = ok[ok.S == S].sort_values("B")
            if len(d1):
                a2.plot(d1.B, d1.tok_s, ":", color=c, lw=1.5, label=f"1 device, S = {S}")
        a2.set_xscale("log", base=2)
        a2.set_yscale("log")
        a2.legend(fontsize=7)
        a2.set_xlabel("batch size B")
        a2.set_ylabel("throughput (tokens / s)")
        a2.set_title("(b) 1 device vs 8-device tensor parallel (optional extension)")
        save(fig, "fig6_ablations_and_tp.png")

    # summary table for the report
    summ = ok[["S", "B", "step_ms_median", "pred_A_ms", "pred_B_ms", "tok_s"]].copy()
    summ["ratio_A"] = summ.step_ms_median / summ.pred_A_ms
    summ["ratio_B"] = summ.step_ms_median / summ.pred_B_ms
    summ.to_csv(os.path.join(OUT_DIR, "summary_decode.csv"), index=False)
    print("median measured/predicted: model A %.2f, model B %.2f" % (summ.ratio_A.median(), summ.ratio_B.median()))


make_figures()

# %% [markdown]
# ## 16. Package the outputs
# Everything is already in `OUT_DIR` (`results/full/` for a full run). This adds CSV copies of the
# raw files and a `results_<mode>.zip` next to `OUT_DIR` (git-ignored) for copying the results off
# the server; section 15 can be re-run anywhere from the unzipped files.

# %%
import shutil

for p in glob.glob(os.path.join(RAW_DIR, "*.jsonl")):  # CSV copies for spreadsheets / the report
    pd.read_json(p, lines=True).to_csv(p[:-6] + ".csv", index=False)
zip_path = shutil.make_archive(os.path.join(os.path.dirname(OUT_DIR), f"results_{RUN_MODE}"), "zip", OUT_DIR)
print("wrote", zip_path)
