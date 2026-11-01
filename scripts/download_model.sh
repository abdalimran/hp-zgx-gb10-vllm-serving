#!/usr/bin/env bash
# Download a Hugging Face model into $MODELS_DIR/<repo id> using the vLLM image's
# own `hf` CLI, so nothing extra is needed on the host.
# Usage: ./scripts/download_model.sh [repo_id] [hf download flags...]
#   e.g. ./scripts/download_model.sh openai/gpt-oss-120b --exclude "original/*"
#   (HF_TOKEN for gated models)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.env"
MODEL="${1:-$MODEL}"
shift || true

mkdir -p "$MODELS_DIR/$MODEL"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HF_HOME=/models/.hf-home \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$MODELS_DIR:/models" \
  --entrypoint hf \
  "$VLLM_IMAGE" \
  download "$MODEL" --local-dir "/models/$MODEL" "$@"

echo "Model saved to $MODELS_DIR/$MODEL"
