#!/usr/bin/env bash
# Serve nvidia/Qwen3.8-Flash-Next-FP4-Local (180B MoE, ~96 GiB) on GB10, following
# NVIDIA's DGX Spark recipe from the model card. Needs the community b12x vLLM
# image: upstream vLLM can't load this checkpoint's compressed FP4 scales yet.
# Usage: ./scripts/serve_qwen38_flash_next.sh   (see docs/qwen3.8-flash-next.md)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.env"

MODEL="nvidia/Qwen3.8-Flash-Next-FP4-Local"
IMAGE="${QWEN38_IMAGE:-eugr/spark-vllm-b12x:nightly-20261026}"
# Kept separate from the NGC image's cache; compiled artefacts aren't compatible.
B12X_CACHE="$CACHE_DIR/b12x-image"
# NVIDIA's validated value for DGX Spark (128 GB): ~97 GiB for weights + KV cache.
QWEN38_GPU_MEMORY_UTILIZATION="${QWEN38_GPU_MEMORY_UTILIZATION:-0.80}"
QWEN38_MAX_MODEL_LEN="${QWEN38_MAX_MODEL_LEN:-262144}"
QWEN38_MAX_NUM_SEQS="${QWEN38_MAX_NUM_SEQS:-8}"
# Multi-token prediction: drafts N tokens per step; 0 disables it.
QWEN38_MTP_TOKENS="${QWEN38_MTP_TOKENS:-4}"

if [[ ! -f "$MODELS_DIR/$MODEL/config.json" ]]; then
  echo "Model not found at $MODELS_DIR/$MODEL, downloading (~96 GiB)..."
  "$SCRIPT_DIR/download_model.sh" "$MODEL"
fi

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
mkdir -p "$B12X_CACHE/root-cache" "$B12X_CACHE/triton"

args=(
  serve "/models/$MODEL"
  --served-model-name "$MODEL"
  --host 0.0.0.0 --port 8000
  --tensor-parallel-size 1 --pipeline-parallel-size 1
  --mamba-cache-mode align
  --enable-prefix-caching
  --enable-chunked-prefill
  --dtype bfloat16
  --kv-cache-dtype fp8
  --quantization modelopt_mixed
  --block-size 16
  --load-format b12x
  --max-model-len "$QWEN38_MAX_MODEL_LEN"
  --max-num-seqs "$QWEN38_MAX_NUM_SEQS"
  --max-num-batched-tokens 4096
  --gdn-decode-kernel b12x
  --linear-backend b12x
  --moe-backend b12x
  --no-enable-flashinfer-autotune
  --mm-encoder-tp-mode data
  --reasoning-parser qwen3
  --tool-call-parser qwen3_xml
  --enable-auto-tool-choice
  --compilation-config '{"pass_config":{"fuse_act_quant":true}}'
  --gpu-memory-utilization "$QWEN38_GPU_MEMORY_UTILIZATION"
)
if [[ "$QWEN38_MTP_TOKENS" != 0 ]]; then
  args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$QWEN38_MTP_TOKENS}")
fi
# shellcheck disable=SC2206
[[ -n "$EXTRA_ARGS" ]] && args+=($EXTRA_ARGS)

# seccomp/apparmor unconfined: required by the model card, for io_uring disk reads
# of the PLE n-gram table (Docker's default seccomp profile blocks io_uring).
docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  --restart no \
  -p "$PORT:8000" \
  -v "$MODELS_DIR:/models" \
  -v "$B12X_CACHE/root-cache:/root/.cache" \
  -v "$B12X_CACHE/triton:/root/.triton" \
  -e HF_HOME=/models/.hf-home \
  -e HF_HUB_OFFLINE=1 \
  -e TORCHINDUCTOR_COMPILE_THREADS="$COMPILE_THREADS" \
  -e VLLM_PLE_TABLE_MEMORY=disk \
  -e B12X_DISK_BACKEND=io_uring \
  -e CUTE_DSL_ARCH=sm_121a \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_SSM_CONV_STATE_LAYOUT=DS \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e B12X_POLICY_MODE=auto \
  -e VLLM_MXFP8_LM_HEAD=1 \
  -e VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13 \
  -e VLLM_B12X_MOE_FP4_FORCE_A16=1 \
  -e B12X_W4A16_A4_PREFILL=1 \
  --entrypoint vllm \
  "$IMAGE" \
  "${args[@]}" >/dev/null

echo "Started $CONTAINER_NAME ($MODEL) on port $PORT; waiting for readiness (first start can take 15+ min)..."
for _ in $(seq 1 900); do
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
