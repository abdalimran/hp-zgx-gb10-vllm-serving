#!/usr/bin/env python3
"""Size the vLLM KV cache to hold N full-context requests for a model.

Usage: kv_cache_size.py <model_dir> <full_context_seqs> <kv_cache_dtype> [max_model_len]

Prints the KV cache size in bytes on stdout (empty if it can't be estimated)
and a human-readable summary on stderr. Exits 2 if weights + KV cache would not
fit in memory with headroom for the OS.
"""
import json
import os
import sys

GiB = 1024**3
OS_HEADROOM_GiB = 16  # unified memory: keep this much for OS, desktop, activations
MARGIN = 1.10  # vLLM block rounding + scheduler slack


def text_config(cfg):
    for key in ("text_config", "llm_config", "language_config"):
        if isinstance(cfg.get(key), dict):
            return {**cfg, **cfg[key]}
    return cfg


def layer_counts(cfg):
    """(full-attention layers, sliding-window layers); linear-attention layers hold
    a fixed-size state rather than per-token KV, so they're left out."""
    if cfg.get("layers_block_type"):  # Nemotron-H: mamba / moe / mlp / attention blocks
        return cfg["layers_block_type"].count("attention"), 0
    types = cfg.get("layer_types")
    if not types:
        return cfg["num_hidden_layers"], 0
    sliding = sum(t in ("sliding_attention", "local_attention") for t in types)
    linear = sum("linear" in t or "mamba" in t for t in types)
    return len(types) - sliding - linear, sliding


def kv_bytes_per_layer_token(cfg, dtype_bytes):
    if cfg.get("kv_lora_rank"):  # MLA (DeepSeek-style): compressed latent per layer
        return (cfg["kv_lora_rank"] + cfg.get("qk_rope_head_dim", 0)) * dtype_bytes
    heads = cfg["num_attention_heads"]
    kv_heads = cfg.get("num_key_value_heads") or heads
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // heads
    return 2 * kv_heads * head_dim * dtype_bytes


def kv_bytes_per_request(cfg, dtype_bytes, ctx):
    full, sliding = layer_counts(cfg)
    window = min(cfg.get("sliding_window") or ctx, ctx)
    return kv_bytes_per_layer_token(cfg, dtype_bytes) * (full * ctx + sliding * window)


def native_context(cfg):
    ctx = cfg.get("max_position_embeddings") or cfg.get("max_sequence_length")
    # Mirrors vLLM 0.31: YaRN/LongRoPE/Llama3 configs already store the scaled length;
    # other scaled RoPE types (linear, dynamic) extend it by the factor.
    rope = cfg.get("rope_scaling") or cfg.get("rope_parameters") or {}
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    no_extend = ("default", "su", "longrope", "llama3", "yarn", "deepseek_yarn", "deepseek_llama_scaling", "mrope")
    if rope_type not in no_extend and rope.get("factor"):
        ctx = int(ctx * rope["factor"])
    return ctx


def weights_bytes(model_dir):
    return sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(model_dir)
        for f in files
        if f.endswith((".safetensors", ".bin", ".pt"))
    )


def total_memory_bytes():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    return 0


def main():
    model_dir, seqs, kv_dtype = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    max_len_arg = sys.argv[4] if len(sys.argv) > 4 else ""
    auto_fit = max_len_arg == "-1"
    max_len = int(max_len_arg) if max_len_arg and not auto_fit else None
    try:
        cfg = text_config(json.load(open(os.path.join(model_dir, "config.json"))))
        dtype_bytes = 1 if kv_dtype.startswith("fp8") else 2
        ctx = max_len or native_context(cfg)
        per_request = kv_bytes_per_request(cfg, dtype_bytes, ctx)
        per_token = per_request / ctx
    except (KeyError, TypeError, ValueError, OSError) as e:
        print(f"KV sizing: can't estimate from config ({e!r}); using GPU_MEMORY_UTILIZATION", file=sys.stderr)
        return

    kv = int(per_request * seqs * MARGIN)
    weights = weights_bytes(model_dir)
    budget = total_memory_bytes() - OS_HEADROOM_GiB * GiB
    print(
        f"KV sizing: context={ctx:,} tokens, {per_token / 1024:.1f} KiB/token (avg), "
        f"{seqs} full-context request(s) -> KV cache {kv / GiB:.1f} GiB; "
        f"weights {weights / GiB:.1f} GiB; budget {budget / GiB:.1f} GiB",
        file=sys.stderr,
    )
    if weights + kv > budget and auto_fit and budget > weights:
        kv = budget - weights
        print(
            f"KV sizing: full context doesn't fit; giving KV cache the remaining {kv / GiB:.1f} GiB "
            "and letting vLLM shrink the context to fit (MAX_MODEL_LEN=-1).",
            file=sys.stderr,
        )
    elif weights + kv > budget:
        print(
            "KV sizing: does not fit. Lower FULL_CONTEXT_SEQS, set KV_CACHE_DTYPE=fp8, "
            "use a smaller quantized checkpoint, or set MAX_MODEL_LEN=-1 to auto-fit the context.",
            file=sys.stderr,
        )
        sys.exit(2)
    print(kv)


if __name__ == "__main__":
    main()
