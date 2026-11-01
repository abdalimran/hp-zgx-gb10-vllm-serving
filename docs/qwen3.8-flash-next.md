# Serving Qwen3.8-Flash-Next on the HP ZGX Nano (GB10)

This is a step-by-step guide to serving Qwen3.8-Flash-Next locally with vLLM in Docker, using NVIDIA's 4-bit FP4-Local checkpoint and its DGX Spark recipe. It builds on the general guide, [vllm-docker-gb10.md](vllm-docker-gb10.md).

> **Tested on this machine on 31 Oct 2026** with `serve_qwen38_flash_next.sh` and default settings:
> - **First start took about 41 minutes**: weight loading took about 3 minutes, and first-run b12x kernel autotuning the rest. The autotuning results are cached for later starts.
> - **Memory:** the model takes 70.3 GiB, and the KV cache holds 966,574 tokens (3.7 full 262K-token conversations). Once ready, 104 GiB is used and 16 GiB is available.
> - **Out of memory during loading:** VS Code and Chrome were closed by the out-of-memory killer during weight loading, so close large apps first (see [Troubleshooting](#5-troubleshooting)).
> - **Test request:** a short reasoning request returned 100 tokens in 3.2 s. No full benchmark has been run yet.
>
> The machine has also had [sudden power-offs](vllm-docker-gb10.md#sudden-power-off) under load.

## 1. Which checkpoint, and why

Qwen3.8-Flash-Next is a 180B-parameter mixture-of-experts model with about 6B parameters active per token. It has hybrid linear and sparse attention, n-gram embeddings (PLE), multi-token prediction (MTP) and a vision encoder. The native context is 262K tokens.

| Checkpoint | Size | Fits in 121 GiB? |
|---|---:|---|
| `Qwen/Qwen3.8-Flash-Next` (BF16) | 335 GiB | No |
| `Qwen/Qwen3.8-Flash-Next-FP8` | 173 GiB | No |
| `nvidia/Qwen3.8-Flash-Next-NVFP4` | 124 GiB | No (larger than all memory) |
| **`nvidia/Qwen3.8-Flash-Next-FP4-Local`** | **95 GiB** | **Yes**, validated by NVIDIA on DGX Spark |

**FP4-Local** was quantized with quantization-aware distillation and scores 91.5% on GPQA Diamond. Its large n-gram embedding table is read from disk on demand (`VLLM_PLE_TABLE_MEMORY=disk`) instead of sitting in memory. That leaves room for a large FP8 KV cache: NVIDIA reports about 1M tokens on DGX Spark, enough for several full 262K-token conversations.

**Why a different container.** This checkpoint stores its FP4 scales in a new compressed format that upstream vLLM, including the NGC image, can't load yet. The model card requires the community image `eugr/spark-vllm-b12x`, made with the "Local Inference Lab" community organization. `serve_qwen38_flash_next.sh` pins it to `nightly-20261026` (digest `sha256:57c24fc3…`), the current `latest`. Before moving to a newer nightly, test it.

> This is a third-party image. The model card also requires it to run with `seccomp=unconfined` and `apparmor=unconfined`, which turns off two layers of container isolation. The reason given is io_uring disk reads, which Docker's default seccomp profile blocks. Only run it on a machine you're comfortable trusting it with.

## 2. Before you start

1. **Power.** Resolve the [sudden power-off issue](vllm-docker-gb10.md#sudden-power-off) first.
2. **Stop anything else using memory.** Stop other model servers (`./scripts/stop.sh`), notebooks and Ollama. This model needs about 97 GiB of the 121 GiB.
3. **Free the page cache** so vLLM sees the memory as free:

   ```bash
   sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
   free -g          # "available" should be above 105
   ```

4. **Optional:** stop the desktop to free a few more GB. Run `sudo systemctl isolate multi-user.target`, use the machine over SSH, and run `sudo systemctl isolate graphical.target` to bring the desktop back.
5. **Disk:** about 96 GiB for the model plus about 10 GB for the image. There's 3.4 TB free.

## 3. Step by step

### Step 1: Pull the container (about 10 GB)

```bash
docker pull eugr/spark-vllm-b12x:nightly-20261026
```

### Step 2: Download the model to `/home/abdalimran/Models` (about 96 GiB)

```bash
cd ~/Workspace/vllm-serving-docker-gb10
./scripts/download_model.sh nvidia/Qwen3.8-Flash-Next-FP4-Local
```

The files go to `/home/abdalimran/Models/nvidia/Qwen3.8-Flash-Next-FP4-Local/`. The model isn't gated, so no `HF_TOKEN` is needed. If the download is interrupted, run the same command again; it resumes.

### Step 3: Start the server

```bash
./scripts/serve_qwen38_flash_next.sh
```

The script:

- mounts `~/Models` at `/models` and runs offline (`HF_HUB_OFFLINE=1`);
- keeps this image's compile caches in `~/Models/.cache/b12x-image`, separate from the NGC image's;
- uses NVIDIA's DGX Spark flags and environment variables unchanged;
- waits up to 30 minutes for `/health`.

The **first start is slow** (roughly 10–20 minutes): it loads weights (about 2 minutes, 70 GiB in memory) and then autotunes 241 b12x kernels. Later starts reuse the cache. The server keeps running if you close the terminal. To check on it, run `./scripts/status.sh`, or `docker logs -f vllm` to follow the log live.

### Step 4: Test it

```bash
curl -s localhost:8000/v1/models

curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "nvidia/Qwen3.8-Flash-Next-FP4-Local",
  "messages": [{"role": "user", "content": "In two sentences, what is a mixture-of-experts model?"}]
}' | python3 -c 'import sys,json; m=json.load(sys.stdin)["choices"][0]["message"]; print("REASONING:", (m.get("reasoning_content") or m.get("reasoning") or "")[:500]); print("ANSWER:", m["content"])'
```

This is a reasoning model. `--reasoning-parser qwen3` separates its thinking into `reasoning_content` and puts the final answer in `content`. Tool calling is enabled (`--tool-call-parser qwen3_xml --enable-auto-tool-choice`).

### Step 5: Connect Chatbox

Use the same steps as [section 4.3 of the main guide](vllm-docker-gb10.md#43-connecting-a-chat-app-chatbox), with these values:

| Field | Value |
|---|---|
| Model | `nvidia/Qwen3.8-Flash-Next-FP4-Local` |
| Context window | 262,144 |
| Max output | unlimited or empty (reasoning can be long). If a number is required, use 32,768 or more. |
| Temperature / top-p | 1.0 / 0.95 (the model's defaults) |

### Step 6: Stop it

```bash
./scripts/stop.sh
```

## 4. Configuration

| Setting | Value | Why |
|---|---|---|
| `--quantization modelopt_mixed`, `--load-format b12x` | required | Loads the mixed NVFP4/MXFP8 checkpoint with the b12x kernels built for sm_121a. |
| `VLLM_PLE_TABLE_MEMORY=disk` | on | Keeps the n-gram embedding table on the NVMe SSD instead of in memory. This is what lets the model fit with a large KV cache. |
| `--gpu-memory-utilization 0.80` | about 97 GiB | NVIDIA's validated value for DGX Spark. Override with `QWEN38_GPU_MEMORY_UTILIZATION`. Don't go above 0.85 with the desktop running. |
| `--kv-cache-dtype fp8` | on | Halves KV memory; NVIDIA validated about 1M tokens of cache. |
| `--max-model-len 262144` | full native context | Every request can use the whole 262K window. Override with `QWEN38_MAX_MODEL_LEN`. |
| Max output | context − prompt | The model's `generation_config.json` sets no `max_new_tokens`, so a reply can use everything the prompt leaves. |
| `--max-num-seqs 8` | 8 concurrent requests | Sized for local use. Override with `QWEN38_MAX_NUM_SEQS`. |
| `--speculative-config` MTP, 4 tokens | on | Multi-token prediction drafts 4 tokens per step, typically giving a large decode speed-up for a single user. Set `QWEN38_MTP_TOKENS=0` to disable it, for example to compare speed or rule it out when debugging. |
| `--mamba-cache-mode align`, `--block-size 16` | required | KV and state cache layout for the hybrid linear-attention layers. |
| `--max-num-batched-tokens 4096` | NVIDIA default | Prefill chunk size. |

**Expected speed:** about 6B active parameters at 4 bits is roughly 3–4 GB read per token. The bandwidth ceiling is therefore in the tens of tokens/s for one user before MTP, and higher with it. These figures are estimates; nothing has been measured here. To benchmark once it's running:

```bash
MODEL=nvidia/Qwen3.8-Flash-Next-FP4-Local CONCURRENCIES="1 4 8" ./scripts/benchmark.sh qwen38
```

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| CUDA out of memory during load | Something else is using memory. Stop it, drop the page cache (step 2.3), or set `QWEN38_GPU_MEMORY_UTILIZATION=0.75`. |
| VS Code, the terminal or Chrome suddenly close during startup | The Linux out-of-memory killer closed them during weight loading. This happened here on 31 Oct 2026: VS Code and Chrome were killed at the "Loading weights" step, while vLLM kept running. Check with `./scripts/status.sh`; the server survives because it runs in Docker. To avoid it, close Chrome and other large apps before starting, or stop the desktop (step 2.4). |
| `KV cache ... smaller than max_model_len` | Lower `QWEN38_MAX_MODEL_LEN` (for example to 131072) or free memory. |
| `io_uring` / `Operation not permitted` errors | The `seccomp=unconfined` option is missing. Use the script, not a hand-written `docker run`. |
| `unknown quantization modelopt_mixed` or b12x errors | You're on the NGC image. This model needs the `eugr/spark-vllm-b12x` image. |
| Errors mentioning speculative decoding or MTP | Retry with `QWEN38_MTP_TOKENS=0`. |
| Replies are only "thinking" and get cut off | Raise max output in the client. Reasoning can use thousands of tokens. |
| Machine powers off during startup | See [Troubleshooting: sudden power-off](vllm-docker-gb10.md#sudden-power-off). |

Sources:

- [nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local), the model card with the DGX Spark recipe
- [eugr/spark-vllm-b12x](https://hub.docker.com/r/eugr/spark-vllm-b12x)
- [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
