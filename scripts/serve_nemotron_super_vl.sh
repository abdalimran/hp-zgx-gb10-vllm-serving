#!/usr/bin/env bash
# Serve nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 (120B hybrid Mamba/MoE
# VLM, 12B active, ~70 GiB) on GB10 with NVIDIA's DGX Spark recipe from the model
# card: upstream vLLM 0.32.0 + the LiLiCorr speculative drafter.
# Usage: ./scripts/serve_nemotron_super_vl.sh   (see docs/nemotron-3.5-super-vl.md)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.env"

MODEL="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
DRAFT_MODEL="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr"
SERVED_NAME="nemotron-3.5-super-vl"
# Upstream vLLM v0.32.0 (CUDA 13.0, matches the host driver); pinned by digest.
IMAGE="${NEMOTRON_IMAGE:-vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34}"
# Kept separate from other images' caches; compiled artefacts aren't compatible.
UPSTREAM_CACHE="$CACHE_DIR/vllm-upstream"
# NVIDIA's DGX Spark values: ~91 GiB for vLLM, one sequence, 7 LiLiCorr draft tokens.
NEMOTRON_GPU_MEMORY_UTILIZATION="${NEMOTRON_GPU_MEMORY_UTILIZATION:-0.75}"
NEMOTRON_MAX_NUM_SEQS="${NEMOTRON_MAX_NUM_SEQS:-1}"
# Empty = the model's full 1M-token context.
NEMOTRON_MAX_MODEL_LEN="${NEMOTRON_MAX_MODEL_LEN-}"
# Draft tokens per step for LiLiCorr; 0 disables speculative decoding.
NEMOTRON_DRAFT_TOKENS="${NEMOTRON_DRAFT_TOKENS:-7}"
# Server-wide reasoning default (on | medium | off) for clients that can't send
# chat_template_kwargs; a request's own chat_template_kwargs still wins.
NEMOTRON_REASONING="${NEMOTRON_REASONING:-on}"

for m in "$MODEL" "$DRAFT_MODEL"; do
  [[ "$m" == "$DRAFT_MODEL" && "$NEMOTRON_DRAFT_TOKENS" == 0 ]] && continue
  if [[ ! -f "$MODELS_DIR/$m/config.json" ]]; then
    echo "Downloading $m..."
    "$SCRIPT_DIR/download_model.sh" "$m"
  fi
done

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
mkdir -p "$UPSTREAM_CACHE"

args=(
  serve "/models/$MODEL"
  --served-model-name "$SERVED_NAME" "$MODEL"
  --host 0.0.0.0 --port 8000
  --trust-remote-code
  --max-num-seqs "$NEMOTRON_MAX_NUM_SEQS"
  --gpu-memory-utilization "$NEMOTRON_GPU_MEMORY_UTILIZATION"
  --max-num-batched-tokens 8192
  --mamba-backend flashinfer
  --mamba-ssm-cache-dtype float16
  --enable-mamba-cache-stochastic-rounding
  --mamba-cache-philox-rounds 5
  --enable-mamba-shared-prefix-checkpoint
  --video-pruning-rate 0.5
  --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}'
  --block-size 64
  --prefix-match-unit 64
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser nemotron_v3
)
[[ -n "$NEMOTRON_MAX_MODEL_LEN" ]] && args+=(--max-model-len "$NEMOTRON_MAX_MODEL_LEN")
case "$NEMOTRON_REASONING" in
  on) ;;
  medium) args+=(--default-chat-template-kwargs '{"enable_thinking": true, "medium_effort": true}') ;;
  off) args+=(--default-chat-template-kwargs '{"enable_thinking": false}') ;;
  *) echo "NEMOTRON_REASONING must be on, medium or off" >&2; exit 1 ;;
esac
if [[ "$NEMOTRON_DRAFT_TOKENS" != 0 ]]; then
  args+=(
    --speculative_config.model "/models/$DRAFT_MODEL"
    --speculative_config.num_speculative_tokens "$NEMOTRON_DRAFT_TOKENS"
    --speculative_config.method dflash
  )
fi
# shellcheck disable=SC2206
[[ -n "$EXTRA_ARGS" ]] && args+=($EXTRA_ARGS)

docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --restart no \
  -p "$PORT:8000" \
  -v "$MODELS_DIR:/models" \
  -v "$UPSTREAM_CACHE:/root/.cache" \
  -e HF_HOME=/models/.hf-home \
  -e HF_HUB_OFFLINE=1 \
  -e TORCHINDUCTOR_COMPILE_THREADS="$COMPILE_THREADS" \
  --entrypoint vllm \
  "$IMAGE" \
  "${args[@]}" >/dev/null

echo "Started $CONTAINER_NAME ($SERVED_NAME) on port $PORT; waiting for readiness (first start can take 15+ min)..."
for _ in $(seq 1 900); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null; then
    echo "Ready: http://localhost:$PORT/v1  (model name: $SERVED_NAME)"
    exit 0
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" != "true" ]]; then
    echo "Container exited during startup. Last logs:" >&2
    docker logs --tail 50 "$CONTAINER_NAME" >&2
    exit 1
  fi
  sleep 2
done
echo "Timed out waiting for server; check: docker logs -f $CONTAINER_NAME" >&2
exit 1
