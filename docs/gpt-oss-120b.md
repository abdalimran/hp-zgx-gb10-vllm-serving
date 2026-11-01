# Serving gpt-oss-120b on the HP ZGX Nano (GB10)

This is a step-by-step guide to serving OpenAI's gpt-oss-120b locally with vLLM in Docker, with NVIDIA's EAGLE3 draft model for faster generation. It builds on the general guide, [vllm-docker-gb10.md](vllm-docker-gb10.md).

> **Not yet tested on this machine.** These steps haven't been run here because of the open [power-off issue](vllm-docker-gb10.md#sudden-power-off). The configuration was checked against the NGC image's vLLM 0.31 source (MXFP4 support, parser names, context-length handling) and the model cards, but nothing has been measured. Loading about 61 GiB and running a 117B model is a much heavier load than the 1.5B test model, so fix the power issue first.

## 1. Which checkpoint, and why

gpt-oss-120b is a 117B-parameter mixture-of-experts model with about 5.1B parameters active per token. It has 128 experts (4 used per token), alternates full and 128-token sliding-window attention, and has a context of 131,072 tokens. It's a reasoning model with adjustable effort and built-in tool calling.

| Item | Value |
|---|---|
| Checkpoint | `openai/gpt-oss-120b` (official; Apache 2.0; not gated) |
| Weight format | MXFP4 for the MoE experts, BF16 for everything else, as published |
| Size on disk | about 61 GiB (the repo's extra `original/` and `metal/` copies are skipped) |
| Fits in 121 GiB? | **Yes**, with room for the KV cache and the OS |
| Container | `nvcr.io/nvidia/vllm:26.10-py3`, the same NGC image as the main guide (already pulled) |
| Draft model (optional) | `nvidia/gpt-oss-120b-Eagle3-v3` (1.5 GiB), NVIDIA's recommended EAGLE3 head |

Unlike Qwen3.8-Flash-Next, this model needs **no community container and no quantization step**. The published checkpoint is already 4-bit, and vLLM 0.31 picks an MXFP4 kernel for sm_121 automatically.

## 2. Memory plan

| Part | Size |
|---|---:|
| Weights (MXFP4 + BF16) | ~61 GiB |
| KV cache: 4 full 131K-token conversations (default) | 19.8 GiB |
| EAGLE3 draft model | ~1.5 GiB |
| Activations and CUDA graphs | ~3–5 GiB |
| **Total for vLLM** | **~85–88 GiB** |
| Left for OS and desktop | ~33–36 GiB |

The KV cache costs **36 KiB per token**. Only the 18 full-attention layers store the whole context; the 18 sliding-window layers keep just their last 128 tokens. `serve.sh` calculates this automatically and prints:

```
KV sizing: context=131,072 tokens, 36.0 KiB/token (avg), 4 full-context request(s) -> KV cache 19.8 GiB; weights ...
```

To allow more simultaneous long conversations, set `FULL_CONTEXT_SEQS=8` (39.6 GiB). This still fits, but leaves the OS only about 15 GiB, so do it with the desktop stopped.

## 3. Before you start

1. **Power.** Resolve the [sudden power-off issue](vllm-docker-gb10.md#sudden-power-off) first.
2. **Stop other model servers** (`./scripts/stop.sh`) and anything else using a lot of memory.
3. **Free the page cache** so the memory freed by reading the model files counts as available:

   ```bash
   sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
   free -g          # "available" should be above ~95
   ```

## 4. Step by step

### Step 1: Image

The NGC image is already on this machine. If not, run:

```bash
docker pull nvcr.io/nvidia/vllm:26.10-py3
```

### Step 2: Download the model and draft model to `/home/abdalimran/Models`

The serve script in step 3 does this automatically if the files are missing. To download ahead of time:

```bash
cd ~/Workspace/vllm-serving-docker-gb10
./scripts/download_model.sh openai/gpt-oss-120b --exclude "original/*" --exclude "metal/*"   # ~61 GiB
./scripts/download_model.sh nvidia/gpt-oss-120b-Eagle3-v3                                   # ~1.5 GiB
```

The `--exclude` flags matter: without them you download two extra copies of the weights (about 120 GiB) that vLLM never uses.

### Step 3: Start the server

```bash
./scripts/serve_gpt_oss_120b.sh
```

This calls the general `serve.sh` with:

- `/models/openai/gpt-oss-120b`, loaded offline from `~/Models`;
- full 131,072-token context, with the KV cache sized for 4 full conversations;
- `--reasoning-parser openai_gptoss`, which puts the model's thinking in `reasoning_content`;
- `--tool-call-parser openai --enable-auto-tool-choice` for function calling;
- `--speculative-config '{"method":"eagle3","model":"/models/nvidia/gpt-oss-120b-Eagle3-v3","num_speculative_tokens":7}'`;
- the main guide's settings: BF16 KV cache, prefix caching, chunked prefill, async scheduling, compile cache, `--restart no`.

The first start takes several minutes: it loads 61 GiB and compiles. Later starts reuse `~/Models/.cache`. To watch progress, run `docker logs -f vllm`.

### Step 4: Test it

```bash
curl -s localhost:8000/v1/models

curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "openai/gpt-oss-120b",
  "messages": [{"role": "user", "content": "In two sentences, what is a mixture-of-experts model?"}],
  "reasoning_effort": "medium"
}' | python3 -c 'import sys,json; m=json.load(sys.stdin)["choices"][0]["message"]; print("REASONING:", (m.get("reasoning_content") or m.get("reasoning") or "")[:500]); print("ANSWER:", m["content"])'
```

`reasoning_effort` can be `low`, `medium` (the default) or `high`. Higher effort means longer thinking, more tokens and slower answers. If a client can't send this field, put `Reasoning: high` in the system prompt instead; that's the model's own way of setting effort.

### Step 5: Connect Chatbox

Use the same steps as [section 4.3 of the main guide](vllm-docker-gb10.md#43-connecting-a-chat-app-chatbox), with these values:

| Field | Value |
|---|---|
| Model | `openai/gpt-oss-120b` |
| Context window | 131,072 |
| Max output | unlimited or empty. If a number is required, use 32,768 or more; reasoning at `high` effort can use thousands of tokens before the answer. |
| Temperature / top-p | 1.0 / 1.0 is the commonly used setting for gpt-oss (the model card doesn't specify one) |

### Step 6: Stop it

```bash
./scripts/stop.sh
```

## 5. Configuration

| Setting | Value | Why |
|---|---|---|
| Image | NGC `26.10-py3` (vLLM 0.31) | Supports `GptOssForCausalLM` and MXFP4 MoE on sm_121 natively. |
| Quantization | none passed | The checkpoint's `quant_method: mxfp4` is detected automatically. Don't set `QUANTIZATION`. |
| Context (`MAX_MODEL_LEN`) | not set, so 131,072 | The model's full context. The config's YaRN factor (32) is already included in `max_position_embeddings`; vLLM doesn't apply it again. |
| Max output | context − prompt | `generation_config.json` sets no `max_new_tokens`. |
| KV cache | `FULL_CONTEXT_SEQS=4` → 19.8 GiB, BF16 | Room for 4 full-length conversations; short chats share it, so many more can run at once. `KV_CACHE_DTYPE=fp8` halves it, at a small accuracy risk. |
| EAGLE3 speculative decoding | 7 draft tokens (`GPT_OSS_EAGLE_TOKENS`) | NVIDIA's recommended draft head. It drafts several tokens that the main model verifies in one pass, so a single user typically gets noticeably faster replies. Set `GPT_OSS_EAGLE_TOKENS=0` to disable it, for example to compare speed or rule it out when debugging. Lower values such as 3 can work better under heavy concurrency. |
| `--max-num-seqs` | 256 (main default) | Fine for this model; the KV cache, not this setting, limits how many long chats run at once. |
| Weights download | `--exclude "original/*" --exclude "metal/*"` | Skips about 120 GiB of duplicate weights. |

**Expected speed:** about 5B active parameters, mostly at 4 bits, is roughly 3–4 GB read per token. With 273 GB/s of bandwidth that's a ceiling of about 70–90 tokens/s for one user before EAGLE3, so real figures will be lower. These are estimates; nothing has been measured here. To benchmark once it's running:

```bash
MODEL=openai/gpt-oss-120b CONCURRENCIES="1 4 16 64" ./scripts/benchmark.sh gpt_oss
GPT_OSS_EAGLE_TOKENS=0 ./scripts/serve_gpt_oss_120b.sh && MODEL=openai/gpt-oss-120b CONCURRENCIES="1 4 16 64" ./scripts/benchmark.sh gpt_oss_no_eagle
```

The random prompts in `benchmark.sh` understate EAGLE3's real-world gain, because draft acceptance is lower on random text.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `KV sizing: does not fit` | Something else is using memory, or `FULL_CONTEXT_SEQS` is too high. Lower it, or set `KV_CACHE_DTYPE=fp8`. |
| CUDA out of memory during load | Stop other processes and drop the page cache (step 3.3). |
| Download is about 180 GiB, not 61 | `--exclude` was left off. Delete `~/Models/openai/gpt-oss-120b/original` and `.../metal`. |
| Errors mentioning EAGLE, speculative decoding or the draft model | Retry with `GPT_OSS_EAGLE_TOKENS=0`. If that works, the draft model or its integration is the problem, not the main model. |
| Answers contain raw `<\|channel\|>` / harmony tags | The reasoning parser is missing. Use the wrapper script, not a bare `serve.sh openai/gpt-oss-120b`. |
| Replies are only "thinking" and get cut off | Raise max output in the client, or use `reasoning_effort: low`. |
| Machine powers off during startup | See [Troubleshooting: sudden power-off](vllm-docker-gb10.md#sudden-power-off). |

Sources:

- [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)
- [nvidia/gpt-oss-120b-Eagle3-v3](https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-v3)
