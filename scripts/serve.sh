#!/usr/bin/env bash
# Start (or restart) the vLLM OpenAI-compatible server in Docker, loading the
# model from $MODELS_DIR on the host, and wait until it is ready.
# Usage: ./scripts/serve.sh [repo_id] [extra vllm serve args...]
#   (see config.env for tunables)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.env"
MODEL="${1:-$MODEL}"
shift || true

if [[ ! -f "$MODELS_DIR/$MODEL/config.json" ]]; then
  echo "Model not found at $MODELS_DIR/$MODEL, downloading..."
  "$SCRIPT_DIR/download_model.sh" "$MODEL"
fi

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
mkdir -p "$CACHE_DIR"

if [[ -z "$KV_CACHE_MEMORY" && "$FULL_CONTEXT_SEQS" != 0 ]]; then
  KV_CACHE_MEMORY="$(python3 "$SCRIPT_DIR/kv_cache_size.py" "$MODELS_DIR/$MODEL" \
    "$FULL_CONTEXT_SEQS" "$KV_CACHE_DTYPE" "$MAX_MODEL_LEN")" || exit 1
fi

args=(
  "/models/$MODEL"
  --served-model-name "$MODEL"
  --host 0.0.0.0 --port 8000
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --enable-prefix-caching
  --enable-chunked-prefill
  --async-scheduling
  --performance-mode "$PERFORMANCE_MODE"
)
if [[ -n "$KV_CACHE_MEMORY" ]]; then
  args+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY")
else
  args+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
fi
[[ -n "$MAX_MODEL_LEN" ]] && args+=(--max-model-len "$MAX_MODEL_LEN")
[[ -n "$QUANTIZATION" ]] && args+=(--quantization "$QUANTIZATION")
# shellcheck disable=SC2206
[[ -n "$EXTRA_ARGS" ]] && args+=($EXTRA_ARGS)
args+=("$@")

docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --restart no \
  -p "$PORT:8000" \
  -v "$MODELS_DIR:/models" \
  -e HF_HOME=/models/.hf-home \
  -e HF_HUB_OFFLINE=1 \
  -v "$CACHE_DIR:/root/.cache" \
  -e TORCHINDUCTOR_COMPILE_THREADS="$COMPILE_THREADS" \
  "$VLLM_IMAGE" \
  vllm serve "${args[@]}" >/dev/null

echo "Started $CONTAINER_NAME ($MODEL) on port $PORT; waiting for readiness..."
# STARTUP_TIMEOUT is in seconds; big models can take 15+ minutes on first start.
for _ in $(seq 1 $(( ${STARTUP_TIMEOUT:-600} / 2 ))); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null; then
    echo "Ready: http://localhost:$PORT/v1"
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
