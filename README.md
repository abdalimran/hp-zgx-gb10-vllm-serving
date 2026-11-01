# HP ZGX / GB10 vLLM Serving

Scripts and guides for serving large language models locally with [vLLM](https://github.com/vllm-project/vllm) in Docker on the NVIDIA GB10 (Grace Blackwell) machines: the DGX Spark, the HP ZGX Nano G1n and others built on the same chip.

The result is an OpenAI-compatible API (`http://<host>:8000/v1`) that works with chat apps, coding agents and the OpenAI SDK. Each script is tuned for the GB10's 128 GB of memory shared by the CPU and GPU, and serves models of up to about 180B parameters with their full context window where memory allows.

## Supported models

| Model | Size | Context | Script | Guide | Status |
|---|---|---|---|---|---|
| [Qwen3.8-Flash-Next](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) (NVIDIA FP4-Local) | 180B MoE, 6B active, about 70 GiB loaded | 262K | `serve_qwen38_flash_next.sh` | [qwen3.8-flash-next.md](docs/qwen3.8-flash-next.md) | ✅ Tested |
| [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) + EAGLE3 | 117B MoE, 5B active, about 61 GiB | 131K | `serve_gpt_oss_120b.sh` | [gpt-oss-120b.md](docs/gpt-oss-120b.md) | ✅ Ran, not benchmarked |
| [Nemotron 3.5 Super VL](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4) + LiLiCorr | 120B hybrid Mamba/MoE, 12B active, about 70 GiB | 1M | `serve_nemotron_super_vl.sh` | [nemotron-3.5-super-vl.md](docs/nemotron-3.5-super-vl.md) | ⚠️ Startup triggered a [power-off](docs/hp-zgx-power-cut-issue.md) |
| [MiniMax M2.7 REAP](https://huggingface.co/saricles/MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10) (NVFP4) | 172B MoE (pruned from 230B), 10B active, 92 GiB | about 110K | `serve_minimax_m27.sh` | [minimax-m2.7.md](docs/minimax-m2.7.md) | 🧪 Config checked, not run |
| Any Hugging Face model | — | full by default | `serve.sh <repo id>` | [vllm-docker-gb10.md](docs/vllm-docker-gb10.md) | ✅ Tested with Qwen2.5-1.5B |

The [main guide](docs/vllm-docker-gb10.md) covers setup, configuration, memory sizing, benchmarks and troubleshooting. Each model guide covers why that checkpoint was chosen, its memory plan and its settings.

## Requirements

- An NVIDIA GB10 machine (aarch64, sm_121) running Ubuntu or DGX OS
- Docker with the NVIDIA Container Toolkit (`docker run --rm --gpus all ubuntu nvidia-smi` should work)
- Disk space for the models: 60–100 GiB each

## Quick start

```bash
git clone https://github.com/abdalimran/hp-zgx-gb10-vllm-serving.git
cd hp-zgx-gb10-vllm-serving

# 1. Pull the vLLM image built for GB10 (about 10 GB)
docker pull nvcr.io/nvidia/vllm:26.10-py3

# 2. Start a model; the weights download to ~/Models on first run
./scripts/serve.sh Qwen/Qwen2.5-1.5B-Instruct      # small test model
./scripts/serve_qwen38_flash_next.sh               # or one of the large-model scripts

# 3. Use it
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "messages": [{"role": "user", "content": "Hello!"}]
}'

# 4. Check on it, or stop it
./scripts/status.sh
./scripts/stop.sh
```

The server runs in the background, so closing the terminal doesn't stop it. It isn't started again automatically after a reboot.

## How it works

- **Models stay on the host.** Weights are stored in `~/Models/<repo id>` and mounted at `/models` in the container, which runs offline (`HF_HUB_OFFLINE=1`). Downloads use the image's own `hf` CLI, so nothing needs installing on the host. Set `MODELS_DIR` to use another folder.
- **Memory is sized per model.** On the GB10 the GPU and the OS share one pool of memory, so vLLM's usual approach of reserving most of it starves the desktop. [`kv_cache_size.py`](scripts/kv_cache_size.py) reads the model's `config.json` and works out the KV cache size needed. It takes account of grouped-query and MLA attention, sliding-window layers, Mamba and linear-attention layers, and FP8 caches. vLLM then gets exactly that amount: for example 19.8 GiB for 4 full 131K-token gpt-oss conversations, instead of 70% of RAM.
- **Full context by default.** If the full context doesn't fit, `MAX_MODEL_LEN=-1` gives the KV cache all remaining memory and lets vLLM use the longest context that fits.
- **Compile caches persist** in `~/Models/.cache`, so later starts skip recompiling.
- **Recipes come from the model makers.** The large-model scripts follow NVIDIA's DGX Spark recipes or the GB10 instructions on each model card. They use the image each model needs (NGC vLLM 0.31, upstream vLLM 0.32, or the community `eugr/spark-vllm-b12x` build) and enable speculative decoding (MTP, EAGLE3, LiLiCorr) where available.

## Scripts

| Script | What it does |
|---|---|
| `config.env` | All settings: image, paths, port, context, KV cache, batching. Override any of them per run, for example `PORT=8001 ./scripts/serve.sh`. |
| `serve.sh [repo id] [vllm args...]` | Starts any model with the NGC image and waits until it's ready |
| `serve_*.sh` | Model-specific launchers (see the table above) |
| `download_model.sh <repo id> [hf flags]` | Downloads a model into `~/Models` |
| `status.sh` | Shows container state, readiness, loaded model and context, memory, GPU and recent log lines |
| `stop.sh` | Stops and removes the server container |
| `benchmark.sh [label]` | Runs `vllm bench serve` at several concurrency levels; results go to `outputs/bench/` |
| `kv_cache_size.py` | KV cache calculator used by `serve.sh` |
| `system_profile.sh` | Writes a Markdown hardware and software report to `outputs/` |

## Connecting clients

Any OpenAI-compatible client works. Use the base URL `http://<host-ip>:8000/v1` and the model's repo id as the model name.

- **Chat apps (Chatbox and others):** see [the main guide, section 4.3](docs/vllm-docker-gb10.md#43-connecting-a-chat-app-chatbox).
- **Coding agents:** OpenCode, Cline, Roo Code and Continue work through their "OpenAI compatible" provider. Tool calling is enabled in every model script.
- **Python:** `OpenAI(base_url="http://<host-ip>:8000/v1", api_key="unused")`

> **Security:** the server listens on all network interfaces with no API key, so anyone on your local network can use it. Docker's port publishing bypasses `ufw`. Don't forward port 8000 on your router. If your network isn't trusted, require a key with `EXTRA_ARGS="--api-key <secret>"` (this works with every script).

## Performance

Measured with `Qwen/Qwen2.5-1.5B-Instruct` (BF16, 1,024 tokens in, 256 out):

| Concurrent users | Output tokens/s | Time to first token |
|---:|---:|---:|
| 1 | 99 | 50 ms |
| 32 | 1,798 | 360 ms |
| 128 | 2,674 | 1,013 ms |

Large models are limited by memory bandwidth (273 GB/s). Mixture-of-experts models with few active parameters are the best fit for this machine. For Qwen3.8-Flash-Next, a short reasoning request returned 100 tokens in 3.2 s. See the [benchmark section](docs/vllm-docker-gb10.md#6-benchmark-results) for details.

## Known issues

- **Sudden power-offs under load.** Some GB10 machines switch off abruptly when the GPU goes from idle to full load, for example during kernel autotuning at startup. HP has a BIOS fix, and capping the GPU clock (`sudo nvidia-smi -lgc 300,2200`) is a workaround. See [hp-zgx-power-cut-issue.md](docs/hp-zgx-power-cut-issue.md).
- **The out-of-memory killer can close desktop apps** while a 70–90 GiB model loads. Close large apps first. The server itself survives.
- **First starts are slow** (10–40 minutes for the large models) because of weight loading and kernel compilation or autotuning. Later starts reuse the cache.
- **A power cut can damage the compile cache.** If startup then fails with `EOFError` or `Cubin file ... not found`, delete the image's folder under `~/Models/.cache`.

## Repository layout

```
scripts/   launchers, config and helpers
docs/      main guide, per-model guides, power-off investigation
outputs/   system profiles and benchmark results (generated)
```

## Disclaimer

This is an independent project, not affiliated with NVIDIA, HP or the model publishers. Model weights are downloaded from Hugging Face under their own licenses. Several large-model setups depend on community containers and checkpoints, which are listed in each guide; review them before use.
