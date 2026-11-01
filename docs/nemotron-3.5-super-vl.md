# Serving Nemotron 3.5 Super VL 120B on the HP ZGX Nano (GB10)

This is a step-by-step guide to serving NVIDIA Nemotron 3.5 Super VL locally with vLLM in Docker. It uses NVIDIA's NVFP4 checkpoint, the DGX Spark recipe from the model card, and the LiLiCorr speculative drafter. It builds on the general guide, [vllm-docker-gb10.md](vllm-docker-gb10.md).

> **Tried on this machine on 1 Nov 2026: the machine powered off during startup.** Weights loaded, the KV cache was sized (1,330,232 tokens, 12.75 GiB) and CUDA graphs were captured. The power cut happened within seconds of the FlashInfer GEMM autotune starting (log ends at `Tuning fp8_gemm: 0/21`, about 08:59 local). The kernel logged no error and left no crash record. The 31 Oct 12:15 power-off also coincided with a FlashInfer autotune starting. This points to the [power-delivery fault](vllm-docker-gb10.md#sudden-power-off): the autotune step jumps from idle to full GPU load. Don't run this model until that is fixed.

## 1. Which checkpoint, and why

Nemotron 3.5 Super VL is a vision-language model with 120B total parameters and about 12B active per token. Its language backbone interleaves 40 Mamba-2 layers, 40 mixture-of-experts layers (512 experts, 22 used per token) and only 8 attention layers. A C-RADIO vision encoder handles images and video. It supports up to **1M tokens** of context, has configurable reasoning, tool calling and a built-in MTP head. The license is OpenMDW-1.1, and the model isn't gated.

| Checkpoint | Size | Fits in 121 GiB? |
|---|---:|---|
| `nvidia/...-BF16` | 232 GiB | No |
| `RedHatAI/...-FP8` | 120 GiB | No (larger than usable memory) |
| **`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`** | **70 GiB** | **Yes**; NVIDIA's DGX Spark recipe targets it |
| `nvidia/...-NVFP4-LiLiCorr` (drafter) | 1.2 GiB | Add-on for faster decoding, recommended by NVIDIA for DGX Spark |

**Why a different container.** NVIDIA's recipe requires upstream **vLLM 0.32.0** (`vllm/vllm-openai:v0.32.0`), one version newer than the NGC image used in the main guide (0.31). Its arm64 build uses **CUDA 13.0.2**, which matches this machine's driver (580, CUDA 13.0). The script pins it by digest (`sha256:befccf0e…`). This is the vLLM project's official image, so no third-party container is needed.

**Why 1M context fits.** Only the 8 attention layers store a per-token KV cache; the Mamba layers keep a fixed-size state. The cost is **8 KiB per token**, so a full 1M-token context needs about 8.8 GiB of KV cache. The same full context on a dense 70B model would take about 320 GiB.

## 2. Memory plan

NVIDIA's DGX Spark recipe sets `--gpu-memory-utilization 0.75`, about 91 GiB:

| Part | Size |
|---|---:|
| Weights (NVFP4 experts + FP8/BF16 elsewhere) | ~70 GiB |
| LiLiCorr drafter | ~1.2 GiB |
| KV cache + Mamba state + activations + CUDA graphs | ~20 GiB (the rest of the 0.75 budget) |
| **Total for vLLM** | **~91 GiB** |
| Left for OS and desktop | ~30 GiB |

The recipe serves **one request at a time** (`--max-num-seqs 1`), which the LiLiCorr drafter is tuned for. That's ideal for one person using a chat app. If several people or agents need it at once, raise `NEMOTRON_MAX_NUM_SEQS`; NVIDIA hasn't published Spark settings for that.

## 3. Before you start

1. **Power.** Resolve the [sudden power-off issue](vllm-docker-gb10.md#sudden-power-off) first.
2. **Stop other model servers** (`./scripts/stop.sh`) and anything else using a lot of memory.
3. **Free the page cache:**

   ```bash
   sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
   free -g          # "available" should be above ~100
   ```

## 4. Step by step

### Step 1: Pull the container (about 10 GB)

```bash
docker pull vllm/vllm-openai:v0.32.0
```

### Step 2: Download the model and drafter to `/home/abdalimran/Models`

The serve script does this automatically if the files are missing. To download ahead of time:

```bash
cd ~/Workspace/vllm-serving-docker-gb10
./scripts/download_model.sh nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4             # ~70 GiB
./scripts/download_model.sh nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr    # ~1.2 GiB
```

### Step 3: Start the server

```bash
./scripts/serve_nemotron_super_vl.sh
```

The script runs NVIDIA's DGX Spark command unchanged, adapted to this setup:

- loads both models from `~/Models` (`/models/...`) and runs offline;
- serves under the short name `nemotron-3.5-super-vl` (the name NVIDIA's examples use); the full repo id also works;
- keeps a separate compile cache in `~/Models/.cache/vllm-upstream`;
- uses `--restart no` and waits up to 30 minutes for `/health`.

The first start takes a while: it loads 70 GiB, compiles kernels and captures CUDA graphs. To watch progress, run `docker logs -f vllm`.

### Step 4: Test it

```bash
curl -s localhost:8000/v1/models

curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "nemotron-3.5-super-vl",
  "messages": [{"role": "user", "content": "Explain why binary search takes logarithmic time."}],
  "temperature": 1.0, "top_p": 0.95, "max_tokens": 16384,
  "chat_template_kwargs": {"enable_thinking": true}
}' | python3 -c 'import sys,json; c=json.load(sys.stdin)["choices"][0]; m=c["message"]; print("FINISH:", c["finish_reason"]); print("REASONING:", (m.get("reasoning_content") or m.get("reasoning") or "")[:500]); print("ANSWER:", m["content"])'
```

**Reasoning levels**, set per request with `chat_template_kwargs`:

| Level | `chat_template_kwargs` |
|---|---|
| On (default) | `{"enable_thinking": true}` |
| Medium (fewer tokens, may lower accuracy) | `{"enable_thinking": true, "medium_effort": true}` |
| Off | `{"enable_thinking": false}` |

Reasoning and the answer share the `max_tokens` budget. If `finish_reason` is `length`, raise `max_tokens`; NVIDIA's examples use 16,384.

**Images:** send them as OpenAI `image_url` content, for example a base64 `data:image/png;base64,...` URL. See the model card's [API client examples](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4#api-client). NVIDIA lists Spark as validated for text and images; video arguments are included in the recipe but aren't listed as validated on Spark.

### Step 5: Connect Chatbox

Use the same steps as [section 4.3 of the main guide](vllm-docker-gb10.md#43-connecting-a-chat-app-chatbox), with these values:

| Field | Value |
|---|---|
| Model | `nemotron-3.5-super-vl` |
| Context window | 1,048,576 (or any smaller value you prefer) |
| Max output | unlimited or empty. If a number is required, use 16,384 or more. |
| Temperature / top-p | 1.0 / 0.95 (the model card's settings) |
| Vision | Turn on image input for this model if Chatbox asks. Images are sent as standard `image_url` content. |

Chatbox can't send `chat_template_kwargs`, so choose the reasoning level when starting the server:

```bash
NEMOTRON_REASONING=medium ./scripts/serve_nemotron_super_vl.sh    # or: on (default) / off
```

This sets `--default-chat-template-kwargs` on the server. Requests that send their own `chat_template_kwargs` still override it.

### Step 6: Stop it

```bash
./scripts/stop.sh
```

## 5. Configuration

All values come from NVIDIA's DGX Spark Quick Start unless marked otherwise.

| Setting | Value | Why |
|---|---|---|
| Image | `vllm/vllm-openai:v0.32.0` (CUDA 13.0.2) | Version required by the recipe; matches the host driver's CUDA 13.0. |
| `--trust-remote-code` | on | The model uses custom code (`NemotronH_Omni_Reasoning_V3`) shipped with the checkpoint. |
| `--gpu-memory-utilization` | 0.75 (`NEMOTRON_GPU_MEMORY_UTILIZATION`) | About 91 GiB for vLLM, about 30 GiB for the OS. |
| `--max-num-seqs` | 1 (`NEMOTRON_MAX_NUM_SEQS`) | One request at a time, for the lowest latency with the drafter. |
| Context | full 1,048,576 (`NEMOTRON_MAX_MODEL_LEN`; empty by default) | Only 8 attention layers, so 1M tokens costs about 8.8 GiB of KV. |
| Max output | context − prompt | `generation_config.json` sets no `max_new_tokens`. Reasoning and answer share it. |
| Speculative decoding | LiLiCorr, `dflash` method, 7 draft tokens (`NEMOTRON_DRAFT_TOKENS`) | NVIDIA recommends LiLiCorr for DGX Spark "in all scenarios". `0` turns it off; NVIDIA suggests comparing speed with it off. |
| `--mamba-backend flashinfer`, `--mamba-ssm-cache-dtype float16`, stochastic rounding (5 Philox rounds), `--enable-mamba-shared-prefix-checkpoint` | on | Mamba-2 state kernels and cache. FP16 state with stochastic rounding saves memory; shared-prefix checkpoints let prefix caching work across Mamba layers. |
| `--block-size 64`, `--prefix-match-unit 64` | on | Cache page and prefix-match granularity for the hybrid layers. |
| `--max-num-batched-tokens` | 8192 | Prefill chunk size. |
| `--video-pruning-rate 0.5`, `--media-io-kwargs` (2 fps, max 256 frames) | on | Video sampling. |
| `--reasoning-parser nemotron_v3`, `--tool-call-parser qwen3_coder --enable-auto-tool-choice` | on | Separates reasoning from the answer; enables function calling. |
| `NEMOTRON_REASONING` | `on` | Added here, not part of NVIDIA's recipe: server-wide reasoning default for clients like Chatbox. |

**Expected speed:** about 12B active parameters at roughly 4–5 bits is roughly 6–7 GB read per token, a ceiling of about 40 tokens/s for one user before speculative decoding. LiLiCorr accepts several draft tokens per step, so real single-user speed can be well above that. These are estimates; nothing has been measured here. To benchmark once it's running:

```bash
MODEL=nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 CONCURRENCIES="1" ./scripts/benchmark.sh nemotron
NEMOTRON_DRAFT_TOKENS=0 ./scripts/serve_nemotron_super_vl.sh && \
  MODEL=nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 CONCURRENCIES="1" ./scripts/benchmark.sh nemotron_no_draft
```

Use the full repo id here, because `benchmark.sh` also loads the tokenizer from `/models/$MODEL`. Concurrency above 1 just queues requests with `--max-num-seqs 1`. The random prompts understate the drafter's real-world gain.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA driver version is insufficient` | You pulled a CUDA 13.x tag newer than the driver supports. Use the pinned `v0.32.0` (CUDA 13.0.2), or `v0.32.0-aarch64-cu129`. |
| `unrecognized arguments` for the `--mamba-*` or `--prefix-match-unit` flags | The image is older than 0.32.0 (for example the NGC 0.31 image). Use the script's image. |
| CUDA out of memory during load | Stop other processes, drop the page cache (step 3.3), or set `NEMOTRON_GPU_MEMORY_UTILIZATION=0.70`. |
| Errors mentioning `dflash`, the speculative config or the drafter | Retry with `NEMOTRON_DRAFT_TOKENS=0`. If that works, the drafter is the problem, not the main model. |
| Requests wait a long time while another is running | Expected with `--max-num-seqs 1`. Raise `NEMOTRON_MAX_NUM_SEQS` (for example to 4); the KV cache has room. |
| `finish_reason: length` with only reasoning | Raise `max_tokens` (16,384+), or use medium or off reasoning. |
| `Model not found` | Use `nemotron-3.5-super-vl` or the full repo id as the model name. |
| After a power-off, startup fails with `EOFError: Ran out of input`, `Cubin file ... not found` or `InductorError: ... has no attribute 'triton_...'` | The power cut left half-written files in the compile cache. Delete it (it's rebuilt automatically, and the weights aren't touched): `docker run --rm -v ~/Models/.cache:/c --entrypoint sh vllm/vllm-openai:v0.32.0 -c 'rm -rf /c/vllm-upstream'`, then start again. |
| Machine powers off during startup | See [Troubleshooting: sudden power-off](vllm-docker-gb10.md#sudden-power-off). |

Sources:

- [nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4), the model card with the DGX Spark recipe
- [nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr)
- [vllm/vllm-openai on Docker Hub](https://hub.docker.com/r/vllm/vllm-openai)
