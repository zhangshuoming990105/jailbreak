#!/usr/bin/env bash
# Serve Base Llama-3-8B-Instruct and Circuit-Breaker (RR) side-by-side with
# vllm + xgrammar. Each on its own port. Tensor-parallel=1 (8B fits on one
# 96GB card with plenty of room; we use CUDA_VISIBLE_DEVICES to pin).
#
# Usage:
#   bash cb_serve.sh base
#   bash cb_serve.sh cb
#   bash cb_serve.sh both         # background both; pipes stdout to cb_base.log / cb_cb.log
set -euo pipefail

BASE_MODEL_PATH=${BASE_MODEL_PATH:-/home/zhangshuoming/models/Meta-Llama-3-8B-Instruct}
CB_MODEL_PATH=${CB_MODEL_PATH:-GraySwanAI/Llama-3-8B-Instruct-RR}

BASE_PORT=${BASE_PORT:-8080}
CB_PORT=${CB_PORT:-8090}

BASE_GPU=${BASE_GPU:-0}
CB_GPU=${CB_GPU:-1}

BACKEND=${BACKEND:-xgrammar}

common_args=(
  --dtype bfloat16
  --structured-outputs-config.backend "${BACKEND}"
  --max-model-len 8192
  --gpu-memory-utilization 0.55
)

serve_base() {
  CUDA_VISIBLE_DEVICES="${BASE_GPU}" vllm serve "${BASE_MODEL_PATH}" \
    --served-model-name base-llama3-8b-instruct \
    --port "${BASE_PORT}" \
    "${common_args[@]}"
}

serve_cb() {
  CUDA_VISIBLE_DEVICES="${CB_GPU}" vllm serve "${CB_MODEL_PATH}" \
    --served-model-name cb-llama3-8b-instruct-rr \
    --port "${CB_PORT}" \
    "${common_args[@]}"
}

case "${1:-both}" in
  base) serve_base ;;
  cb) serve_cb ;;
  both)
    serve_base > cb_base.log 2>&1 &
    BASE_PID=$!
    echo "Base vllm PID=${BASE_PID} (log: cb_base.log)"
    serve_cb > cb_cb.log 2>&1 &
    CB_PID=$!
    echo "CB vllm PID=${CB_PID}   (log: cb_cb.log)"
    echo "tail -f cb_base.log cb_cb.log  to monitor"
    ;;
  *) echo "usage: $0 {base|cb|both}"; exit 1 ;;
esac
