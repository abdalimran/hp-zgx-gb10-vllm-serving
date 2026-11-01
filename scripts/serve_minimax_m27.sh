#!/usr/bin/env bash
# Serve saricles/MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10 (MiniMax-M2.7 pruned 230B -> 172B
# with REAP, NVFP4, 10B active, ~92 GiB) on GB10 with the NGC vLLM image.
# Usage: ./scripts/serve_minimax_m27.sh   (see docs/minimax-m2.7.md)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.env"

MODEL="saricles/MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10"
# The full 196K context needs 23 GiB of FP8 KV cache, which doesn't fit beside 92 GiB
# of weights. -1 lets vLLM use the longest context the remaining memory allows (~110K).
export MAX_MODEL_LEN="${MINIMAX_MAX_MODEL_LEN:--1}"
export FULL_CONTEXT_SEQS=1
export KV_CACHE_DTYPE=fp8
export MAX_NUM_SEQS="${MINIMAX_MAX_NUM_SEQS:-8}"
export STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
# The model card says CUDA graphs deadlock this MoE on GB10; set to 1 to try them anyway.
MINIMAX_CUDA_GRAPHS="${MINIMAX_CUDA_GRAPHS:-0}"
# minimax_m2_append_think keeps <think>...</think> in the reply, which MiniMax recommends
# sending back in chat history; minimax_m2 moves it to a separate reasoning field instead.
MINIMAX_REASONING_PARSER="${MINIMAX_REASONING_PARSER:-minimax_m2_append_think}"

if [[ ! -f "$MODELS_DIR/$MODEL/config.json" ]]; then
  echo "Downloading $MODEL (~92 GiB)..."
  "$SCRIPT_DIR/download_model.sh" "$MODEL"
fi

if [[ "$MINIMAX_CUDA_GRAPHS" == 1 ]]; then
  compilation='{"inductor_compile_config":{"combo_kernels":false,"max_autotune":false,"max_autotune_gemm":false}}'
else
  compilation='{"cudagraph_mode":"none","inductor_compile_config":{"combo_kernels":false,"max_autotune":false,"max_autotune_gemm":false}}'
fi

# FlashInfer autotune is off: it's the full-load spike that preceded a power-off here
# (docs/hp-zgx-power-cut-issue.md), and the model card disables it too.
exec "$SCRIPT_DIR/serve.sh" "$MODEL" \
  --tool-call-parser minimax_m2 --enable-auto-tool-choice \
  --reasoning-parser "$MINIMAX_REASONING_PARSER" \
  --compilation-config "$compilation" \
  --kernel-config '{"enable_flashinfer_autotune": false}' \
  --override-generation-config '{"temperature": 1.0, "top_p": 0.95, "top_k": 40, "min_p": 0.01}'
