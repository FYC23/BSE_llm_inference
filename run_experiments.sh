#!/usr/bin/env bash
# Smoke test, then full run, back to back on one free RTX 3090.
# Usage: GPU=0 bash run_experiments.sh   (run inside tmux; GPU = index of a free card)
set -uo pipefail
GPU=${GPU:?set GPU to the index of a free GPU}
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT/notebooks"
source "$ROOT/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=${FRAC:-0.95}
NB=qwen3_batch_scaling.ipynb
mkdir -p ../results/smoke_gpu ../results/full
FULL_LOG=../results/full/run.log

echo "=== $(date -Is) smoke on GPU $GPU ==="
RUN_MODE=smoke OUT_DIR=../results/smoke_gpu \
  jupyter nbconvert --to notebook --execute $NB \
  --output-dir ../results/smoke_gpu --output executed_smoke.ipynb \
  --ExecutePreprocessor.timeout=-1 2>&1 | tee ../results/smoke_gpu/run.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "SMOKE FAILED, not starting full run"; exit 1; fi
echo "=== $(date -Is) smoke OK ==="

# Full run. The fp32 Qwen3-4B HF check needs ~25-40 GB host RAM and the server may be shared,
# so cap our process at 30 GB: if the check exceeds it, only our process is killed, and we retry
# with RUN_CHECK=0 (nothing is lost: rows append and resume).
echo "=== $(date -Is) full run (RUN_CHECK=1, host RAM cap 30G) ===" | tee -a $FULL_LOG
systemd-run --user --scope -q -p MemoryMax=30G -p MemorySwapMax=0 \
  env RUN_MODE=full RUN_CHECK=1 jupyter nbconvert --to notebook --execute $NB \
  --output-dir ../results/full --output executed_full.ipynb \
  --ExecutePreprocessor.timeout=-1 2>&1 | tee -a $FULL_LOG
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "=== $(date -Is) full run with check FAILED; retrying with RUN_CHECK=0 ===" | tee -a $FULL_LOG
  RUN_MODE=full RUN_CHECK=0 jupyter nbconvert --to notebook --execute $NB \
    --output-dir ../results/full --output executed_full.ipynb \
    --ExecutePreprocessor.timeout=-1 2>&1 | tee -a $FULL_LOG
  st=${PIPESTATUS[0]}
else
  st=0
fi
echo "=== $(date -Is) full run exit $st ===" | tee -a $FULL_LOG
