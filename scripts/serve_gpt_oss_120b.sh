#!/usr/bin/env bash
# Serve openai/gpt-oss-120b (117B MoE, ~5B active, MXFP4, ~61 GiB) on GB10 with the
# NGC vLLM image, plus NVIDIA's EAGLE3 draft model for speculative decoding.
# Usage: ./scripts/serve_gpt_oss_120b.sh   (see docs/gpt-oss-120b.md)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.env"

MODEL="openai/gpt-oss-120b"
DRAFT_MODEL="nvidia/gpt-oss-120b-Eagle3-v3"
# Draft tokens per step for EAGLE3; 0 disables speculative decoding.
GPT_OSS_EAGLE_TOKENS="${GPT_OSS_EAGLE_TOKENS:-7}"

# The repo also ships original/ and metal/ copies of the weights (~120 GiB extra).
if [[ ! -f "$MODELS_DIR/$MODEL/config.json" ]]; then
  echo "Downloading $MODEL (~61 GiB)..."
  "$SCRIPT_DIR/download_model.sh" "$MODEL" --exclude "original/*" --exclude "metal/*"
fi

spec_args=()
if [[ "$GPT_OSS_EAGLE_TOKENS" != 0 ]]; then
  if [[ ! -f "$MODELS_DIR/$DRAFT_MODEL/config.json" ]]; then
    echo "Downloading $DRAFT_MODEL (~1.5 GiB)..."
    "$SCRIPT_DIR/download_model.sh" "$DRAFT_MODEL"
  fi
  spec_args=(--speculative-config
    "{\"method\":\"eagle3\",\"model\":\"/models/$DRAFT_MODEL\",\"num_speculative_tokens\":$GPT_OSS_EAGLE_TOKENS}")
fi

exec "$SCRIPT_DIR/serve.sh" "$MODEL" \
  --reasoning-parser openai_gptoss \
  --tool-call-parser openai --enable-auto-tool-choice \
  "${spec_args[@]}"
