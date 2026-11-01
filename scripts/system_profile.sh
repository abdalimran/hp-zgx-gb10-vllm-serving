#!/usr/bin/env bash
# Capture a system profile (OS, CPU, memory, GPU, storage, Docker, network)
# and write it as Markdown to outputs/system_profile_<timestamp>.md.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-$(dirname "$SCRIPT_DIR")/outputs}"
mkdir -p "$OUT_DIR"
OUT_FILE="$OUT_DIR/system_profile_$(date +%Y%m%d_%H%M%S).md"

# Run a command and emit its output in a fenced code block; note if unavailable.
section() {
  local title="$1"; shift
  echo "## $title"
  echo
  if command -v "$1" >/dev/null 2>&1; then
    echo '```'
    "$@" 2>&1 || true
    echo '```'
  else
    echo "_\`$1\` not available._"
  fi
  echo
}

{
  echo "# System Profile: $(hostname)"
  echo
  echo "Generated: $(date -Iseconds)"
  echo
  echo "## Summary"
  echo
  echo "| Item | Value |"
  echo "|---|---|"
  echo "| Hostname | $(hostname) |"
  echo "| OS | $(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-unknown}") |"
  echo "| Kernel | $(uname -r) |"
  echo "| Architecture | $(uname -m) |"
  echo "| CPU | $(lscpu 2>/dev/null | awk -F: '/Model name/ {gsub(/^ +/,"",$2); print $2}' | sort -u | paste -sd ',' | sed 's/,/, /g') |"
  echo "| CPU cores | $(nproc) |"
  echo "| Memory | $(free -h | awk '/^Mem:/ {print $2}') |"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "| GPU | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd ',' | sed 's/,/, /g') |"
    echo "| NVIDIA driver | $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1) |"
    echo "| CUDA version | $(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9.]+') |"
  fi
  command -v nvcc >/dev/null 2>&1 && echo "| nvcc | $(nvcc --version | grep -oP 'release \K[0-9.]+') |"
  command -v docker >/dev/null 2>&1 && echo "| Docker | $(docker --version 2>/dev/null) |"
  command -v python3 >/dev/null 2>&1 && echo "| Python | $(python3 --version 2>&1) |"
  echo "| Uptime | $(uptime -p) |"
  echo

  section "OS Release" cat /etc/os-release
  section "Kernel" uname -a
  section "CPU" lscpu
  section "Memory" free -h
  section "GPU" nvidia-smi
  section "GPU Details" nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,temperature.gpu,power.draw,compute_cap --format=csv
  section "CUDA Compiler" nvcc --version
  section "Block Devices" lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,MODEL
  section "Disk Usage" df -hT -x tmpfs -x devtmpfs -x squashfs -x overlay
  section "Network Interfaces" ip -br addr
  section "Docker Version" docker version
  section "Docker Runtimes" docker info --format '{{json .Runtimes}}'
  section "NVIDIA Container Toolkit" nvidia-ctk --version
  section "PCI Devices" lspci
} > "$OUT_FILE"

echo "System profile written to $OUT_FILE"
