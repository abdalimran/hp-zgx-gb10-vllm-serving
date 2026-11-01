#!/usr/bin/env bash
# Show whether the vLLM server is running, ready, and what it's doing.
# The server runs detached in Docker, so it survives closing the terminal that
# started it; use this (or `docker logs -f vllm`) to check on it from anywhere.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.env"

state="$(docker inspect -f '{{.State.Status}} (started {{.State.StartedAt}}, exit code {{.State.ExitCode}})' "$CONTAINER_NAME" 2>/dev/null)"
if [[ -z "$state" ]]; then
  echo "Container: $CONTAINER_NAME not found (not started, or stopped with stop.sh)"
  exit 1
fi
echo "Container: $CONTAINER_NAME $state"
echo "Image:     $(docker inspect -f '{{.Config.Image}}' "$CONTAINER_NAME")"

if curl -sf "http://localhost:$PORT/health" >/dev/null; then
  echo "Server:    READY at http://localhost:$PORT/v1"
  curl -s "http://localhost:$PORT/v1/models" | python3 -c 'import sys,json; [print("Model:    ", m["id"], "| max_model_len:", m.get("max_model_len")) for m in json.load(sys.stdin)["data"]]' 2>/dev/null
else
  echo "Server:    not ready yet (still starting, or failed; see the log below)"
fi

echo "Memory:    $(free -g | awk '/^Mem:/ {print $3 " GiB used, " $7 " GiB available of " $2}')"
echo "GPU:       $(nvidia-smi --query-gpu=utilization.gpu,power.draw,temperature.gpu --format=csv,noheader 2>/dev/null)"
echo "--- last log lines (follow live with: docker logs -f $CONTAINER_NAME)"
docker logs --tail 3 "$CONTAINER_NAME" 2>&1 | cut -c1-200
