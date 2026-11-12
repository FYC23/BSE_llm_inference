"""Notebook E-check (§6) split into two processes so torch's fp32 model and JAX's fp32 copy are never
resident together (torch does not hand freed memory back to the OS, so the in-notebook check peaks
near 40 GB for Qwen3-4B).

  python check_split.py ref  <npz>   # process 1: HF fp32 reference logits only (torch, no JAX)
  python check_split.py jax  <npz>   # process 2: notebook §1-§6 setup, then the same three checks
Env: RUN_MODE, OUT_DIR (scratch), CUDA_VISIBLE_DEVICES, NB (path to qwen3_batch_scaling.py).
Steps (1)-(3) and the asserts are copied from the notebook verbatim; only `ref`/`ids` come from the npz.
"""
import os, sys, json
import numpy as np

mode, npz = sys.argv[1], sys.argv[2]
NB = os.environ["NB"]
PROMPT = "The capital of France is Paris. The capital of Japan is"   # same as the notebook

if mode == "ref":
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = {"smoke": "Qwen/Qwen3-0.6B", "full": "Qwen/Qwen3-4B"}[os.environ["RUN_MODE"]]
    model_dir = snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = tok(PROMPT, return_tensors="np")["input_ids"].astype(np.int32)
    hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).float().eval()
    with torch.no_grad():
        ref = hf(torch.from_numpy(ids.astype(np.int64))).logits[0].float().numpy()
    np.savez(npz, ids=ids, ref=ref, model_dir=model_dir)
    print("saved ref", ref.shape, "for", model_id)
    sys.exit(0)

# ---- mode == "jax": run the notebook up to (not including) `if RUN_CHECK:` -------------------
os.environ["RUN_CHECK"] = "0"
src = open(NB).read()
cut = src.index("\nif RUN_CHECK:")
os.chdir(os.path.dirname(os.path.abspath(NB)))
g = {"__name__": "__main__"}
exec(compile(src[:cut], NB, "exec"), g)
globals().update(g)

from transformers import AutoTokenizer
d = np.load(npz)
ids, ref = d["ids"], d["ref"]
assert str(d["model_dir"]) == MODEL_DIR, (str(d["model_dir"]), MODEL_DIR)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
assert (tok(PROMPT, return_tensors="np")["input_ids"].astype(np.int32) == ids).all()
CHECK = {"note": "HF fp32 reference computed in a separate process (check_split.py); else identical to notebook §6"}

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
CHECK["first_token_matches_hf"] = bool(gen[0] == int(ref[-1].argmax()))

print(json.dumps(CHECK, indent=2))
with open(os.path.join(RAW_DIR, "correctness.json"), "w") as f:
    json.dump(CHECK, f, indent=2)
assert CHECK["fp32_max_abs_diff"] < 5e-3, "fp32 logits disagree with Hugging Face"
assert CHECK["fp32_top1_agree"] == 1.0
assert CHECK["first_token_matches_hf"]
print("E-check PASSED")
