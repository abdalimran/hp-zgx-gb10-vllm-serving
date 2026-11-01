# Serving a Local LLM with vLLM in Docker on the HP ZGX Nano (GB10)

This guide sets up an OpenAI-compatible LLM server with vLLM in Docker, tuned for this machine. It was tested on 31 Oct 2026 by serving `Qwen/Qwen2.5-1.5B-Instruct`. Results are in [Benchmark results](#benchmark-results).

> **Before you start: known power-off issue.** During testing this machine powered off abruptly four times. Each time, load jumped suddenly from idle to busy: model startup, or the first requests after startup. Long runs at full load did not trigger it. The kernel logged no error, no thermal event and no out-of-memory kill, so this looks like a power-delivery fault, not a vLLM or configuration problem. Read [Troubleshooting: sudden power-off](#sudden-power-off) before running this unattended.

## 1. System this guide is tuned for

| Item | Value |
|---|---|
| Machine | HP ZGX Nano G1n AI Station (BIOS 5.36, 16 Jul 2026) |
| SoC | NVIDIA GB10 (Grace Blackwell), compute capability 12.1 (sm_121) |
| CPU | 20 Arm cores (10× Cortex-X925, 10× Cortex-A725), aarch64 |
| Memory | 128 GB LPDDR5x **unified** (about 121 GiB visible), shared by CPU and GPU, about 273 GB/s |
| OS / kernel | Ubuntu 24.04.5 LTS, 7.0.0-1019-nvidia |
| Driver / CUDA | 580.178.04 / CUDA 13.0 (host) |
| Docker | 29.6.2, NVIDIA Container Toolkit 1.20.1 |

To capture the current profile, run `./scripts/system_profile.sh`; it writes to `outputs/`.

What matters for vLLM on this hardware:

- **Memory is unified.** GPU memory and system RAM are the same pool. Whatever vLLM reserves, the OS and desktop can't use, so `--gpu-memory-utilization` controls system RAM as well as GPU memory.
- **Generating tokens is limited by memory bandwidth.** Each new token reads every model weight once. At 273 GB/s, smaller weights mean faster generation, but only when the quantized compute kernels keep up (see [Results](#benchmark-results)).
- **It's Arm (aarch64) with sm_121.** Most x86 vLLM images won't run. Use NVIDIA's NGC image, which is built for GB10.

## 2. Layout

```
/home/abdalimran/Models/                 ← all model weights (host disk)
├── Qwen/Qwen2.5-1.5B-Instruct/          ← one folder per Hugging Face repo id
├── .hf-home/                            ← HF metadata from downloads
└── .cache/                              ← vLLM torch.compile + CUDA graph cache (owned by root)

vllm-serving-docker-gb10/
├── scripts/config.env        ← all settings (image, paths, tuning)
├── scripts/download_model.sh ← download a model into ~/Models
├── scripts/serve.sh          ← start server, wait until ready
├── scripts/kv_cache_size.py  ← sizes the KV cache from the model's config
├── scripts/stop.sh           ← stop server
├── scripts/benchmark.sh      ← throughput/latency benchmark
└── scripts/system_profile.sh ← system profile → outputs/
```

The container sees `/home/abdalimran/Models` as `/models`. vLLM loads weights from `/models/<repo id>` and runs with `HF_HUB_OFFLINE=1`, so it never downloads anything in the background.

## 3. One-time setup

### 3.1 Check that Docker can see the GPU

```bash
nvidia-smi                                                    # host sees GB10
docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L         # container sees GB10
```

The second command should print `GPU 0: NVIDIA GB10 (...)`. If it fails, run `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.

If `docker` needs `sudo`, add yourself to the group: `sudo usermod -aG docker $USER`, then log out and back in.

### 3.2 Pull the vLLM image

```bash
docker pull nvcr.io/nvidia/vllm:26.10-py3
```

This image contains vLLM 0.31.0, PyTorch 2.15 and CUDA 13.4, built for aarch64 and sm_121. The download is large (tens of GB) and took over 10 minutes here. Pulling from NGC doesn't need a login.

To find newer tags:

```bash
TOKEN=$(curl -s "https://nvcr.io/proxy_auth?scope=repository:nvidia/vllm:pull" | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])')
curl -s -H "Authorization: Bearer $TOKEN" https://nvcr.io/v2/nvidia/vllm/tags/list | python3 -m json.tool | tail -15
```

To upgrade, change `VLLM_IMAGE` in `scripts/config.env`.

### 3.3 Download a model into `/home/abdalimran/Models`

```bash
./scripts/download_model.sh Qwen/Qwen2.5-1.5B-Instruct
```

This runs the `hf` CLI inside the vLLM image as your user, so you don't need to install anything on the host. Files go to `/home/abdalimran/Models/Qwen/Qwen2.5-1.5B-Instruct/`. For gated models (Llama, Gemma and similar), set `HF_TOKEN=hf_xxx` first.

To do the same without the script:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -e HF_HOME=/models/.hf-home -e HF_TOKEN="$HF_TOKEN" \
  -v /home/abdalimran/Models:/models \
  --entrypoint hf nvcr.io/nvidia/vllm:26.10-py3 \
  download Qwen/Qwen2.5-1.5B-Instruct --local-dir /models/Qwen/Qwen2.5-1.5B-Instruct
```

## 4. Every time: start, use, stop

```bash
cd ~/Workspace/vllm-serving-docker-gb10

./scripts/serve.sh                          # start default model, blocks until ready
# or: ./scripts/serve.sh Qwen/Qwen3-8B      # any model in ~/Models (downloads if missing)

curl -s localhost:8000/v1/models            # check it's up
./scripts/status.sh                         # running? ready? memory, GPU, last log lines
docker logs -f vllm                         # follow logs (Ctrl-C to detach)

./scripts/stop.sh                           # stop and remove the container
```

The server runs detached in Docker, so **closing the terminal or VS Code doesn't stop it**. Only `./scripts/stop.sh` or a reboot does. To check on it from any new terminal, run `./scripts/status.sh`.

`serve.sh` downloads the model if it isn't in `~/Models` yet, replaces any running `vllm` container, and polls `/health` until the server is ready. If the container dies during startup, it prints the last 50 log lines.

**Startup time:** about 2 minutes on a cold start (model load, `torch.compile`, CUDA graph capture). The compile cache in `~/Models/.cache` is reused on later starts of the same model and settings.

**Not started at boot.** The container uses `--restart no`. Don't change this to `unless-stopped` until the [power-off issue](#sudden-power-off) is resolved. With auto-restart, every reboot repeats the startup spike that seems to trigger it.

### 4.1 The full `docker run` command

`serve.sh` runs this command with the default settings. It first prints a sizing line like the one below, then passes the computed size as `--kv-cache-memory-bytes`:

```
KV sizing: context=32,768 tokens, 28.0 KiB/token, 4 full-context request(s) -> KV cache 3.8 GiB; weights 2.9 GiB; budget 105.6 GiB
```

No `--max-model-len` is passed, so vLLM uses the model's full native context.

```bash
docker run -d --name vllm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --restart no \
  -p 8000:8000 \
  -v /home/abdalimran/Models:/models \
  -e HF_HOME=/models/.hf-home \
  -e HF_HUB_OFFLINE=1 \
  -v /home/abdalimran/Models/.cache:/root/.cache \
  -e TORCHINDUCTOR_COMPILE_THREADS=8 \
  nvcr.io/nvidia/vllm:26.10-py3 \
  vllm serve /models/Qwen/Qwen2.5-1.5B-Instruct \
    --served-model-name Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --kv-cache-dtype auto \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --async-scheduling \
    --performance-mode balanced \
    --kv-cache-memory-bytes 4133906022
```

### 4.2 Calling the server

The server speaks the OpenAI API at `http://localhost:8000/v1`. The model name is the Hugging Face repo id.

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "messages": [{"role": "user", "content": "What is the capital of France? One sentence."}],
  "max_tokens": 64
}'
```

From Python (`pip install openai`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "Explain unified memory in two sentences."}],
    max_tokens=128,
)
print(resp.choices[0].message.content)
```

The server binds to `0.0.0.0:8000` with no API key, so anyone on your network can reach it. To protect it, add `EXTRA_ARGS="--api-key <secret>"`, or change `-p` to `127.0.0.1:8000:8000`.

### 4.3 Connecting a chat app (Chatbox)

Any chat app that accepts a custom "OpenAI-compatible" provider works the same way. These steps use the Chatbox desktop app.

1. Start the server and wait for `Ready: http://localhost:8000/v1`:

   ```bash
   cd ~/Workspace/vllm-serving-docker-gb10
   ./scripts/serve.sh
   curl -s localhost:8000/v1/models     # should list Qwen/Qwen2.5-1.5B-Instruct
   ```

2. In Chatbox, open **Settings → Model Provider** and choose **Add custom provider**. Label names vary slightly between versions.

3. Fill in:

   | Field | Value |
   |---|---|
   | Name | `vLLM GB10` (anything) |
   | API Mode | **OpenAI API Compatible** |
   | API Host | `http://localhost:8000` if Chatbox runs on the ZGX itself; otherwise `http://<ZGX-IP>:8000` (find the IP with `hostname -I` on the ZGX) |
   | API Path | `/v1/chat/completions` (default) |
   | API Key | any text, e.g. `none` (or your secret if you started the server with `--api-key`) |
   | Model | `Qwen/Qwen2.5-1.5B-Instruct`, exactly as shown, or click **Fetch** to load it from the server |

   If the app asks for one combined Base URL instead of Host plus Path, use `http://localhost:8000/v1`.

4. In the model settings, set the context window to the model's full context (32,768 for Qwen2.5-1.5B). For max output, choose **unlimited** or leave it empty if Chatbox allows it; the server then lets a reply use everything the prompt leaves. If Chatbox needs a number, use the model card's generation limit (8,192 for Qwen2.5). See [section 7](#7-context-window-output-length-and-memory).

5. Save, start a new chat, pick the **vLLM GB10** model, and send a message.

6. When you're done, run `./scripts/stop.sh`.

To require a key, start the server with `EXTRA_ARGS="--api-key mysecret" ./scripts/serve.sh` and enter `mysecret` in Chatbox's **API Key** field.

| Problem | Fix |
|---|---|
| Connection refused | Check the server is running with `docker ps`; if not, read `docker logs vllm`. |
| Works on the ZGX but not from another PC | Use the ZGX's IP, not `localhost`. Open the port if `ufw` is on: `sudo ufw allow 8000/tcp`. |
| "Model not found" | The model name must match `Qwen/Qwen2.5-1.5B-Instruct` exactly. |
| Responses cut off | Increase max output tokens in Chatbox's model settings. |

## 5. Optimized configuration and why

Every setting is in `scripts/config.env`, and any of them can be overridden per run, for example `MAX_MODEL_LEN=8192 ./scripts/serve.sh`.

| Setting | Default | Why |
|---|---|---|
| Image | `nvcr.io/nvidia/vllm:26.10-py3` | NVIDIA build for aarch64 + sm_121 (GB10) with matching CUDA 13 kernels; detects the GPU as capability (12, 1). |
| `--max-model-len` (`MAX_MODEL_LEN`) | not set (model's full native context) | Every request can use the model's whole context window. `-1` shrinks it automatically when the full context doesn't fit in memory. |
| `--kv-cache-memory-bytes` (`FULL_CONTEXT_SEQS`) | room for **4 full-context requests** (3.8 GiB for Qwen2.5-1.5B) | Computed per model from `config.json`. Memory is unified, so a large reservation takes RAM from the whole system. The earlier `--gpu-memory-utilization 0.70` default reserved 85 GiB (80.7 GiB of KV cache, about 3M tokens) for a 1.5B model, about 78% of RAM. Short requests share the cache, so many more than 4 chats still run at once. |
| `KV_CACHE_MEMORY` | not set | Fixed size such as `16G`; overrides `FULL_CONTEXT_SEQS`. |
| `--gpu-memory-utilization` | `0.70` (fallback only) | Used only with `FULL_CONTEXT_SEQS=0`, or if the KV size can't be computed from the config (unusual architectures). |
| Weights dtype (`QUANTIZATION`) | empty (native BF16) | BF16 was **faster than online FP8 at every concurrency level** for 1.5B (up to +17%, see results). For small models the FP8 quantize/scale overhead outweighs the bandwidth it saves. |
| `--kv-cache-dtype` | `auto` (BF16) | Full-precision KV cache. `fp8` halves KV memory, useful for long contexts on big models. |
| `--max-num-seqs` | `256` | Maximum requests decoded in one batch. Throughput kept rising up to 128 concurrent users. |
| `--max-num-batched-tokens` | `8192` | Prefill chunk size per step. Long prompts are split so they don't stall other users' generation. |
| `--enable-chunked-prefill` | on | Interleaves prefill with decode for steady per-token latency under load. |
| `--enable-prefix-caching` | on | Reuses the KV cache for repeated prefixes (system prompts, chat history), so later turns skip that prefill. |
| `--async-scheduling` | on | Overlaps CPU scheduling with GPU execution to close gaps between steps. |
| `--performance-mode` | `balanced` | `interactivity` suits 1–4 users, `throughput` suits many. |
| `--ipc=host`, `--ulimit memlock=-1`, `--ulimit stack=67108864` | on | NVIDIA's recommended flags; the default 64 MB `/dev/shm` is too small for vLLM. |
| Compile cache mount (`CACHE_DIR`) | `~/Models/.cache` | Keeps `torch.compile` and CUDA graph artefacts between container runs. |
| `TORCHINDUCTOR_COMPILE_THREADS` | `8` | Limits parallel compile workers (default is all 20 cores) to soften the startup CPU+GPU power spike. |
| `HF_HUB_OFFLINE=1` | on | Loads only from `~/Models`; never downloads at serve time. |

What vLLM sets automatically, from the startup log: CUDA graphs (`FULL_AND_PIECEWISE`, batch sizes 1–512), `torch.compile` at optimization level `-O2`, the FlashInfer/Cutlass kernels for sm_121, and FlashInfer autotuning.

### 5.1 Throttling

No throttling occurred during the benchmarks. During both full 128-user runs:

- SM clock stayed at 2,470–2,535 MHz (average about 2,509 MHz).
- GPU power peaked at about 77 W and temperature at 65–66 °C. The critical trip point is 104.8 °C.
- `clocks_event_reasons` stayed at `0x0` (no power cap, thermal or HW slowdown).

GB10 doesn't allow power limits to be changed (`nvidia-smi` reports `N/A`). There's nothing to tune for throttling beyond good airflow. To watch it live:

```bash
nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active --format=csv -l 1
```

## 6. Benchmark results

The benchmark used `./scripts/benchmark.sh` (`vllm bench serve`, random prompts, 1,024 input and 256 output tokens, `--ignore-eos`) with `Qwen/Qwen2.5-1.5B-Instruct`, `balanced` mode, `--max-model-len 32768` and `--gpu-memory-utilization 0.70` (the defaults at the time; the KV cache was 80.7 GiB). The current default 3.8 GiB KV cache holds about 144K tokens. That covers the 1- to 32-user runs; 128 users × 1,280 tokens (164K) would queue slightly, so set `KV_CACHE_MEMORY=8G` for high-concurrency benchmarks. Raw JSON is in `outputs/bench/`.

| Concurrent users | BF16 output tok/s | FP8 output tok/s | BF16 TTFT (ms) | BF16 TPOT (ms) | BF16 per-user tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | **98.6** | 91.5 | 50 | 9.98 | 100 |
| 8 | **690** | 669 | 141 | 11.07 | 90 |
| 32 | **1,798** | 1,633 | 360 | 16.43 | 61 |
| 128 | **2,674** | 2,294 | 1,013 | 43.94 | 23 |

*TTFT = time to first token; TPOT = time per output token after the first.*

Takeaways:

- **Single user:** about 100 tokens/s with a 50 ms first-token latency.
- **Best aggregate throughput:** 2,674 output tok/s (13,371 total tok/s including prompt) at 128 users. Throughput keeps scaling while per-user speed drops, so pick the concurrency that matches your latency target.
- **FP8 weights didn't help a model this small.** Try `QUANTIZATION=fp8` (or a pre-quantized FP8 checkpoint) for 8B+ models, where weight bandwidth dominates. That case wasn't tested here.

Not measured: `throughput`/`interactivity` performance modes and the FP8 KV cache. Testing stopped after the machine powered off as the benchmark began. Run them yourself after the power issue is fixed:

```bash
PERFORMANCE_MODE=throughput ./scripts/serve.sh && ./scripts/benchmark.sh bf16_throughput
KV_CACHE_DTYPE=fp8         ./scripts/serve.sh && ./scripts/benchmark.sh bf16_kvfp8
CONCURRENCIES="1 4" INPUT_LEN=512 OUTPUT_LEN=512 ./scripts/benchmark.sh chat_like
```

## 7. Context window, output length and memory

### 7.1 Full context and maximum output

- **Context window:** `MAX_MODEL_LEN` is empty by default, so vLLM uses the model's full native context from `config.json` (`max_position_embeddings`). It extends that value only for linear or dynamic RoPE scaling; YaRN and Llama 3 configs already store the scaled length. For Qwen2.5-1.5B that's 32,768 tokens, and for gpt-oss-120b it's 131,072.
- **Maximum output:** vLLM has no separate output limit. A reply can use everything the prompt leaves (context − prompt tokens). If the client doesn't send `max_tokens`, vLLM allows all of it, unless the model's `generation_config.json` sets `max_new_tokens`; Qwen2.5's doesn't. In practice the limit is the chat app's **max output** setting, so set it there.
- **Model cards vs. config:** some cards list a recommended generation length (Qwen2.5: 8,192). Longer outputs work but quality may drop.

### 7.2 How memory is used

vLLM reserves its whole memory budget at startup, whatever the model's size. On unified memory that comes straight out of system RAM, so `serve.sh` sizes the KV cache to fit the model:

```
KV cache = KV bytes per token × context length × FULL_CONTEXT_SEQS × 1.1
KV bytes per token = 2 (K and V) × layers × KV heads × head dim × 2 bytes (BF16) or 1 (FP8)
```

`scripts/kv_cache_size.py` reads these values from the model's `config.json`. It handles GQA, MLA (DeepSeek) and multimodal configs, counts sliding-window layers (gpt-oss) at their window size, skips linear-attention and Mamba layers (Qwen3-Next, Qwen3.8 and Nemotron-H), and refuses to start if weights + KV cache exceed total RAM minus 16 GiB for the OS. Its output for Qwen2.5-1.5B matches what vLLM measured (28.0 KiB/token).

| Want | Set |
|---|---|
| More simultaneous long chats | `FULL_CONTEXT_SEQS=8 ./scripts/serve.sh` |
| An exact KV size | `KV_CACHE_MEMORY=16G ./scripts/serve.sh` |
| Full context on a model that barely fits | `KV_CACHE_DTYPE=fp8` (halves KV), then `FULL_CONTEXT_SEQS=1` |
| Biggest context that fits | `MAX_MODEL_LEN=-1` (KV gets all remaining memory; vLLM shrinks the context to fit) |
| The old "use 70% of RAM" behaviour (max throughput) | `FULL_CONTEXT_SEQS=0 GPU_MEMORY_UTILIZATION=0.70 ./scripts/serve.sh` |

The startup log line `GPU KV cache size: N tokens, Maximum concurrency for <ctx> tokens per request: X` confirms the result: X should be at least `FULL_CONTEXT_SEQS`.

### 7.3 Bigger models (up to about 200B)

The "200B parameters" figure for GB10 assumes **4-bit weights**. Weight memory is roughly parameters × bytes per parameter. About 100 GB is usable for weights plus KV cache:

| Precision | Bytes/param | Largest practical model |
|---|---:|---|
| BF16 | 2 | ~40–50B |
| FP8 | 1 | ~80–90B |
| FP4 / NVFP4 / MXFP4 | ~0.5 | ~150–200B (200B leaves little room for context) |

Generation speed is limited by memory bandwidth: **max tokens/s ≈ 273 GB/s ÷ GB of active weights read per token**.

| Model | Active weights | Max speed (single user) |
|---|---|---|
| Qwen2.5-1.5B, BF16 (tested) | 3 GB | ~90 (measured 99) |
| Dense 32B, FP8 | 32 GB | ~8 |
| Dense 70B, FP8 | 70 GB | ~4 |
| MoE, e.g. gpt-oss-120b, 4-bit (~5B active) | ~3 GB | tens of tokens/s |
| Dense 200B, FP4 | 100 GB | ~2–3 |

**Mixture-of-experts (MoE) models** suit this machine best: large total size but few weights read per token. Examples are gpt-oss-120b, Qwen3-30B-A3B and Qwen3-Next-80B-A3B. Worked examples: [gpt-oss-120b.md](gpt-oss-120b.md) (117B, NGC image), [nemotron-3.5-super-vl.md](nemotron-3.5-super-vl.md) (120B vision-language, 1M context, upstream vLLM 0.32), [qwen3.8-flash-next.md](qwen3.8-flash-next.md) (180B, community image) and [minimax-m2.7.md](minimax-m2.7.md) (172B pruned MiniMax M2.7, NGC image, about 110K context).

Long contexts on big models need a lot of KV cache. A Llama-70B-class model uses 320 KiB/token, so one full 128K request needs about 44 GiB in BF16 (22 GiB in FP8). The calculation above accounts for this.

Checklist for large models:

1. Fix the [power issue](#sudden-power-off) first; big models mean longer, heavier load.
2. Use a pre-quantized checkpoint (NVFP4, MXFP4, FP8, AWQ), not BF16.
3. Use `KV_CACHE_DTYPE=fp8` and `FULL_CONTEXT_SEQS=1`–`2` for 128K+ contexts.
4. Lower `MAX_NUM_SEQS` (for example to 16) if only a few people will use it.
5. Free memory before loading: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` (the page cache holds the model files). For the largest models, stop the desktop with `sudo systemctl isolate multi-user.target`.
6. Add model-specific flags through `EXTRA_ARGS`: `--trust-remote-code`, `--reasoning-parser`, `--tool-call-parser`.
7. Above about 200B: two GB10 machines linked over ConnectX-7 with `--tensor-parallel-size 2` (not covered here).

Models with custom code need `--trust-remote-code`. Reasoning and tool-calling models need their parsers. Pass extra `vllm serve` arguments after the model name; JSON values with spaces are kept intact:

```bash
./scripts/serve.sh Qwen/Qwen3-8B --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes
```

To skip unneeded files when downloading, pass `hf download` flags after the repo id: `./scripts/download_model.sh openai/gpt-oss-120b --exclude "original/*" --exclude "metal/*"`.

## 8. Troubleshooting

### Sudden power-off

> **Known issue, with a fix:** HP has a BIOS update (1.07) for this, and NVIDIA's forum has a GPU-clock workaround. See [hp-zgx-power-cut-issue.md](hp-zgx-power-cut-issue.md) for the fix and the full record.

**What happened during testing.** On 31 Oct 2026 the machine lost power four times (11:47, 11:52, 12:04, 12:16):

- No kernel panic, thermal event or out-of-memory kill was logged, and `last` shows `crash` with no shutdown.
- Every power-off coincided with a sudden idle-to-busy load step: 21 s into model load and compile, about 2 min into compile and CUDA graph capture, and twice within seconds of the first requests after the server became ready.
- Sustained full load (77 W, 65 °C) ran for minutes without a problem.

**Likely cause.** This points to over-current or voltage protection in the power path, either the external adapter or the board. It does not point to software or heat. Firmware (BIOS and Embedded Controller) is already current according to `fwupdmgr`.

**What to do:**

1. Use the original HP adapter and cable, plugged directly into a wall outlet (not a power strip or a small UPS).
2. Keep the vents clear.
3. If it continues, contact HP support with the timestamps above. A ZGX Nano powering off under a 1.5B-model load is a hardware fault.
4. Until it's fixed, keep `--restart no` and leave the compile cache mounted, which reduces repeated startup spikes.

**Inspecting a crash afterwards:**

```bash
journalctl --list-boots | tail -5                 # boots that ended without shutdown
journalctl -b -1 -k | tail -20                    # kernel log from the previous boot
last -x | grep -E 'crash|shutdown' | head
```

### Other issues

| Symptom | Fix |
|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| `exec format error` | You pulled an x86 image. Use the NGC `nvcr.io/nvidia/vllm` image. |
| `Model not found` / tries to reach huggingface.co | Download first with `./scripts/download_model.sh <repo>`. Serving runs offline. |
| CUDA out of memory at startup | Another process (another container, Ollama, a notebook) holds unified memory. Check `nvidia-smi` and `free -g`, or lower `GPU_MEMORY_UTILIZATION`. |
| Desktop sluggish while serving | Lower `GPU_MEMORY_UTILIZATION` to 0.6. The OS shares the same memory. |
| vLLM sees less free memory than expected | The Linux page cache counts as used. `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` before starting. |
| Can't delete `~/Models/.cache` | It's written by root in the container: `sudo rm -rf ~/Models/.cache`. It's safe to delete; it rebuilds on the next start. |
| Port 8000 in use | `PORT=8001 ./scripts/serve.sh` |
