# Serving MiniMax M2.7 on the HP ZGX Nano (GB10)

This is a step-by-step guide to serving MiniMax M2.7 locally with vLLM in Docker, using a community build that is pruned and quantized to fit one GB10. It builds on the general guide, [vllm-docker-gb10.md](vllm-docker-gb10.md).

> **Not yet tested on this machine.** The settings were checked against the NGC image's vLLM 0.31: the model and its parsers are supported, every flag parses, and the KV sizing was calculated from the real `config.json`. The model hasn't been downloaded or run yet. It is the largest model in this repo (92 GiB of weights), so read [the power-off issue](hp-zgx-power-cut-issue.md) first.

## 1. Which checkpoint, and why

MiniMax M2.7 is a 230B-parameter mixture-of-experts model with about 10B parameters active per token, aimed at coding and agent work. Its full context is 196,608 tokens, and it is a reasoning model that thinks in `<think>` blocks before answering.

| Checkpoint | Size | Fits in 121 GiB? |
|---|---:|---|
| `MiniMaxAI/MiniMax-M2.7` (FP8, official) | about 215 GiB | No |
| `saricles/MiniMax-M2.7-NVFP4-GB10` (all 256 experts) | 130.6 GB | No, needs 2 Sparks |
| **`saricles/MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10`** | **92.1 GiB** | **Yes**, built and tested by its author for one DGX Spark |

**REAP** removes the 25% of experts that matter least (256 → 192 per layer), which brings the model down to 172B parameters. It still uses 8 experts and about 10B parameters per token. The remaining weights are NVFP4 (4-bit), with the expert router kept in full precision. The author hasn't published quality benchmarks for the pruned build yet; the full model scores 56.2 on SWE-bench Pro and 89.8 on GPQA Diamond. Expect the pruned build to be somewhat lower. The model card also notes that SWE-bench-style data wasn't used to choose which experts to keep, so very long multi-step coding sessions may suffer slightly more.

Hugging Face shows "87B params" for this repo. That's because two 4-bit values are packed into each stored number; the model really has 172B parameters.

**Container:** the same NGC image as the main guide (`nvcr.io/nvidia/vllm:26.10-py3`, vLLM 0.31). It supports `MiniMaxM2ForCausalLM`, the NVFP4 (ModelOpt) format and both MiniMax parsers natively, and its transformers 5.17 knows the model. So `--trust-remote-code` isn't needed, and none of the repo's Python files run.

## 2. Memory plan

| Part | Size |
|---|---:|
| Weights (NVFP4) | 92.1 GiB |
| KV cache (FP8), all remaining budget | about 13.5 GiB (about 114K tokens) |
| Activations, CUDA context | about 2–4 GiB |
| **Total for vLLM** | **about 108–110 GiB** |
| Left for OS and desktop | about 11–13 GiB |

All 62 layers use full attention, so the KV cache costs **124 KiB per token** in FP8, about 4 times as much as gpt-oss. One full 196K-token conversation would need 23.25 GiB (25.6 GiB with `kv_cache_size.py`'s 10% margin), and that doesn't fit beside 92 GiB of weights. So the script sets `MAX_MODEL_LEN=-1`, and `kv_cache_size.py` prints:

```
KV sizing: full context doesn't fit; giving KV cache the remaining 13.5 GiB and letting vLLM shrink the context to fit (MAX_MODEL_LEN=-1).
```

vLLM then sets the context to the longest that fits, about **110K tokens**. The startup log line `GPU KV cache size: ... tokens` shows the exact figure, and `./scripts/status.sh` shows the resulting `max_model_len`. The 114K tokens are shared: one conversation can use all of them, or several shorter ones can run at once.

The model card's own command asks for the full 196,608 tokens with `--gpu-memory-utilization 0.85`. By this calculation that can't fit, so this script doesn't copy it.

If the desktop gets slow or apps are killed, trade context for headroom:

```bash
MINIMAX_MAX_MODEL_LEN=65536 ./scripts/serve_minimax_m27.sh    # 8.5 GiB KV, about 5 GiB more for the OS
```

## 3. Before you start

1. **Power.** This is the heaviest model in the repo. Apply the BIOS fix or the GPU clock cap from [hp-zgx-power-cut-issue.md](hp-zgx-power-cut-issue.md) first. The script already turns off FlashInfer autotuning, the step that preceded the Nemotron power-off.
2. **Stop other model servers** (`./scripts/stop.sh`). **Close Chrome, VS Code and other large apps.** With about 12 GiB left for the OS, the out-of-memory killer will close them otherwise, as it did during the Qwen3.8 test. Run the script from a plain terminal, or over SSH.
3. **Free the page cache:**

   ```bash
   sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
   free -g          # "available" should be above ~108
   ```

4. **Optional:** stop the desktop for about 3 GB more headroom (`sudo systemctl isolate multi-user.target`, and `sudo systemctl isolate graphical.target` to bring it back).
5. **Disk:** 92 GiB for the model. There's 3.1 TB free.

## 4. Step by step

### Step 1: Image

The NGC image is already on this machine. If not, run:

```bash
docker pull nvcr.io/nvidia/vllm:26.10-py3
```

### Step 2: Download the model to `/home/abdalimran/Models` (about 92 GiB)

The serve script does this automatically if the files are missing. To download ahead of time:

```bash
cd ~/Workspace/vllm-serving-docker-gb10
./scripts/download_model.sh saricles/MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10
```

The files go to `/home/abdalimran/Models/saricles/MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10/`. The repo isn't gated. If the download is interrupted, run the same command again; it resumes.

### Step 3: Start the server

```bash
./scripts/serve_minimax_m27.sh
```

This calls the general `serve.sh` with:

- `/models/saricles/MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10`, loaded offline from `~/Models`;
- FP8 KV cache with all remaining memory and the longest context that fits (`MAX_MODEL_LEN=-1`);
- `--tool-call-parser minimax_m2 --enable-auto-tool-choice` for function calling;
- `--reasoning-parser minimax_m2_append_think` (see step 4);
- CUDA graphs off and Inductor max-autotune off, as the model card requires;
- `--kernel-config '{"enable_flashinfer_autotune": false}'`;
- the model card's sampling defaults (temperature 1.0, top-p 0.95, top-k 40, min-p 0.01);
- the main guide's settings: prefix caching, chunked prefill, async scheduling, compile cache, `--restart no`.

The script waits up to 30 minutes. The model card expects the first start to take 10–15 minutes; loading 92 GiB is most of that. The server keeps running if you close the terminal. To check on it, run `./scripts/status.sh`, or `docker logs -f vllm` to follow the log live.

### Step 4: Test it

```bash
curl -s localhost:8000/v1/models

curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "saricles/MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10",
  "messages": [{"role": "user", "content": "In two sentences, what is a mixture-of-experts model?"}]
}' | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
```

The reply starts with the model's reasoning inside `<think>...</think>`, followed by the answer. This is deliberate. MiniMax M2 models reason between tool calls ("interleaved thinking"), and MiniMax recommends sending the whole reply, `<think>` block included, back in the chat history. Stripping it lowers quality in multi-step agent work. The `minimax_m2_append_think` parser keeps it in place.

If you prefer the reasoning in a separate `reasoning_content` field, as with the other models, start with:

```bash
MINIMAX_REASONING_PARSER=minimax_m2 ./scripts/serve_minimax_m27.sh
```

### Step 5: Connect Chatbox

Use the same steps as [section 4.3 of the main guide](vllm-docker-gb10.md#43-connecting-a-chat-app-chatbox), with these values:

| Field | Value |
|---|---|
| Model | `saricles/MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10` |
| Context window | the `max_model_len` shown by `./scripts/status.sh` (about 110,000) |
| Max output | unlimited or empty. If a number is required, use 32,768 or more. |
| Temperature / top-p | 1.0 / 0.95, or leave empty to use the server defaults |

Chatbox shows the `<think>` block as collapsible reasoning.

### Step 6: Stop it

```bash
./scripts/stop.sh
```

## 5. Configuration

| Setting | Value | Why |
|---|---|---|
| Image | NGC `26.10-py3` (vLLM 0.31) | Native `MiniMaxM2ForCausalLM`, ModelOpt NVFP4 and the `minimax_m2` parsers. |
| Quantization | none passed | `hf_quant_config.json` (ModelOpt NVFP4) is detected automatically. |
| `--trust-remote-code` | not used | Transformers 5.17 in the image supports `minimax_m2`, so the repo's Python files never run. |
| Context (`MAX_MODEL_LEN`) | `-1`, about 110K | The full 196,608 needs 23 GiB of KV cache and doesn't fit. Override with `MINIMAX_MAX_MODEL_LEN`. |
| Max output | context − prompt | `generation_config.json` sets no `max_new_tokens`. |
| KV cache | FP8, `FULL_CONTEXT_SEQS=1` | FP8 halves KV memory, so the context is twice as long as with BF16. |
| `--max-num-seqs` | 8 | Room for short parallel requests (for example agent sub-tasks), all sharing the ~114K-token cache. The model card uses 32. Override with `MINIMAX_MAX_NUM_SEQS`. |
| CUDA graphs | off (`cudagraph_mode: none`) | The model card says CUDA graphs deadlock this MoE on GB10. This costs some decode speed. `MINIMAX_CUDA_GRAPHS=1` re-enables them if you want to test whether vLLM 0.31 is affected. |
| FlashInfer autotune | off (`--kernel-config`) | Avoids the idle-to-full-load spike linked to the power-offs. Kernels use default settings, which may be slightly slower. |
| Reasoning parser | `minimax_m2_append_think` | Keeps `<think>` in the reply for interleaved thinking. Override with `MINIMAX_REASONING_PARSER=minimax_m2`. |
| Sampling defaults | temperature 1.0, top-p 0.95, top-k 40, min-p 0.01 | From the model card. Requests can still set their own values. |
| Startup timeout | 30 minutes (`STARTUP_TIMEOUT=1800`) | Loading 92 GiB takes longer than `serve.sh`'s 10-minute default. |
| Not copied from the model card | `VLLM_NVFP4_GEMM_BACKEND`, `VLLM_USE_FLASHINFER_MOE_FP4` | vLLM 0.31 no longer reads these variables. It picks the NVFP4 kernels itself. |
| Model has MTP layers | not used | The config lists 3 MTP modules, but this build contains no MTP weights, so there's no speculative decoding. |

**Expected speed:** about 10B active parameters at 4 bits is roughly 6 GB read per token. With 273 GB/s of bandwidth that's a ceiling of about 45 tokens/s for one user; without CUDA graphs, real figures will be lower. The model card projects about 26 tokens/s. Nothing has been measured here. To benchmark once it's running:

```bash
MODEL=saricles/MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10 CONCURRENCIES="1 4 8" ./scripts/benchmark.sh minimax_m27
```

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `KV sizing: does not fit` | Something else is using memory, or `MINIMAX_MAX_MODEL_LEN` is too high. Stop other servers and drop the page cache (step 3.3). |
| CUDA out of memory during load | Close other apps and drop the page cache. If it persists, set `MINIMAX_MAX_MODEL_LEN=65536`. |
| VS Code, the terminal or Chrome close during startup | The out-of-memory killer closed them. The server usually survives; check with `./scripts/status.sh`. Close large apps first, or stop the desktop (step 3.4). |
| Startup hangs at "Capturing CUDA graphs", or requests hang | `MINIMAX_CUDA_GRAPHS=1` was set. Unset it. |
| `--expert-parallel` errors | Don't add `--enable-expert-parallel`; the model card says it breaks under current kernels. |
| Replies contain `<think>...</think>` | Expected with the default parser (step 4). Use `MINIMAX_REASONING_PARSER=minimax_m2` to move it into `reasoning_content`. |
| Replies are only "thinking" and get cut off | Raise max output in the client. |
| After a power-off, startup fails with `EOFError` or `Cubin file ... not found` | The compile cache was damaged. See [hp-zgx-power-cut-issue.md: clear the compile cache](hp-zgx-power-cut-issue.md#after-a-power-off-clear-the-compile-cache); this model uses the NGC cache folders `vllm` and `flashinfer`. |
| Machine powers off during startup | See [hp-zgx-power-cut-issue.md](hp-zgx-power-cut-issue.md). |

Sources:

- [saricles/MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10](https://huggingface.co/saricles/MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10), the model card with the GB10 recipe
- [saricles/MiniMax-M2.7-NVFP4-GB10](https://huggingface.co/saricles/MiniMax-M2.7-NVFP4-GB10), the unpruned build for 2 Sparks
- [MiniMaxAI/MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7)
- [The MiniMax-M2 Series](https://arxiv.org/pdf/2605.26494) (benchmarks)
- [NVIDIA forum: MiniMax M2.7 NVFP4 recipe and benchmarks](https://forums.developer.nvidia.com/t/minimax-m2-7-nfvp4-recipe-benchmarks/366324)
