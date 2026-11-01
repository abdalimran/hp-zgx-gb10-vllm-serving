#!/usr/bin/env bash
# Stop and remove the vLLM container.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.env"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 && echo "Stopped $CONTAINER_NAME" || echo "$CONTAINER_NAME is not running"
