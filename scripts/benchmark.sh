#!/usr/bin/env bash
# Benchmark the running server with `vllm bench serve` (random prompts) at
# several concurrency levels, saving JSON results to outputs/bench/.
# Usage: ./scripts/benchmark.sh [label]   e.g. ./scripts/benchmark.sh fp8
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.env"
LABEL="${1:-run}"
CONCURRENCIES="${CONCURRENCIES:-1 8 32 128}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
RESULT_DIR="$(dirname "$SCRIPT_DIR")/outputs/bench"
mkdir -p "$RESULT_DIR"

for c in $CONCURRENCIES; do
  prompts=$(( c * 4 > 16 ? c * 4 : 16 ))
  echo "== $LABEL: concurrency=$c prompts=$prompts in=$INPUT_LEN out=$OUTPUT_LEN"
  docker exec "$CONTAINER_NAME" vllm bench serve \
    --model "$MODEL" --tokenizer "/models/$MODEL" \
    --base-url http://localhost:8000 \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$prompts" --max-concurrency "$c" \
    --ignore-eos \
    --save-result --result-dir /tmp/bench \
    --result-filename "${LABEL}_c${c}.json" 2>&1 \
    | grep -E 'Output token throughput|Total Token throughput|Mean TTFT|Mean TPOT|Request throughput'
  docker cp "$CONTAINER_NAME:/tmp/bench/${LABEL}_c${c}.json" "$RESULT_DIR/" >/dev/null
done
