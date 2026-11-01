# HP ZGX Nano G1n: sudden power-offs under heavy GPU load

The machine (HP ZGX Nano G1n AI Station, NVIDIA GB10) switches off abruptly when the GPU jumps from idle to full load. HP and NVIDIA both know about the issue. HP's fix is a BIOS update, and NVIDIA's forum workaround is to cap the GPU clock.

**Status on 1 Nov 2026:** not fixed. The installed BIOS may be older than HP's fix (see [step 1](#1-update-the-bios-hps-fix)).

## Symptoms on this machine

| When | What was running | Last thing before the cut |
|---|---|---|
| 31 Oct, 11:47–12:15 (4 power-offs) | Testing the small Qwen2.5-1.5B model | Model startup, or the first requests after startup |
| 31 Oct, 12:15 | Same as above | A FlashInfer kernel autotune starting (its cache folder was created that minute) |
| 1 Nov, about 08:59 | `serve_nemotron_super_vl.sh` (Nemotron 3.5 Super VL 120B) | FlashInfer GEMM autotune starting (`Tuning fp8_gemm: 0/21`). Weights were loaded, the KV cache was sized and CUDA graphs were captured. |

Qwen3.8-Flash-Next and gpt-oss-120b both started without a power cut. Nemotron draws more power at once during startup: about 12B active parameters, against about 6B and 5B for the other two, and its LiLiCorr drafter warms up at the same moment. So it is the most likely of the three to trigger the cut.

## Why it's a power problem, not a software one

- **Nothing was logged.** There's no kernel panic, crash dump (pstore), thermal event, out-of-memory kill, or Xid/NVRM GPU error. The system log just stops. Software can't switch the machine off this way.
- **It happens when load jumps, not under steady load.** Each cut came at a moment when load jumped from idle to full, such as kernel autotuning, where the GPU runs at full clock while every CPU core compiles. Long runs at steady full load didn't trigger it.
- **No thermal cause.** No thermal event was logged before any of the cuts, and `nvidia-smi` shows no thermal slowdown.
- **It matches the known issue.** HP's advisory and the NVIDIA forum report below describe the same pattern. The forum attributes it to the power delivery tripping on fast load spikes.

## Fixes, in order

### 1. Update the BIOS (HP's fix)

HP advisory: *"HP ZGX Nano G1n AI Station PC - Unexpected Shutdowns and Restarts Occur During Heavy Workloads"* (effective 27 Aug 2026). The fix is **BIOS version 1.07 or later**.

This machine reports:

```bash
cat /sys/class/dmi/id/bios_version /sys/class/dmi/id/bios_date
# 5.36_0ACUM028
# 07/16/2026
```

The BIOS is dated 16 Jul 2026, before the advisory. `fwupdmgr get-updates` (run without refreshing first) says "No updates available". Its update list may be out of date, though, and it isn't clear how `0ACUM028` maps to HP's "1.07".

Update with HP's steps:

```bash
sudo apt install fwupd
sudo fwupdmgr refresh            # download the latest update list from LVFS
sudo fwupdmgr get-updates
sudo fwupdmgr update             # reboot when asked
```

Afterwards, `sudo fwupdmgr get-updates` should report "No updates available". If it still offers nothing and the BIOS date hasn't changed, ask HP support whether `5.36_0ACUM028` includes the fix. If it doesn't, ask how to install 1.07.

### 2. Cap the GPU clock (workaround)

The NVIDIA forum workaround is to lock the GPU clock below its 2418 MHz default:

```bash
sudo nvidia-smi -lgc 300,2200    # GPU clock between 300 and 2200 MHz
sudo nvidia-smi -rgc             # undo
```

This keeps power spikes smaller, at a small cost in speed. If 2200 MHz still trips it, try 1500. The setting is lost at reboot, so run it before each start. Use it until the BIOS is updated, or keep it if the update doesn't help.

### 3. Rule out the power supply and airflow

- Use the original HP power adapter, plugged directly into a wall socket with no power strip or UPS in between.
- Place the unit flat on a hard surface with about 10 cm of clearance on every side.

### 4. If it still happens: contact HP support

Tell them:

- The model is an HP ZGX Nano G1n, BIOS `5.36_0ACUM028` (07/16/2026), running Ubuntu with kernel `7.0.0-1024-nvidia`.
- It powers off abruptly with nothing in the logs: no kernel panic, pstore record, thermal event, or Xid error.
- You can reproduce it reliably: starting vLLM with a 120B model cuts power within seconds of FlashInfer's kernel autotune starting.
- Say whether you've applied the BIOS 1.07 advisory and the clock cap.

## After a power-off: clear the compile cache

A power cut during startup can leave half-written files in vLLM's compile cache. The next start then fails with `EOFError: Ran out of input`, `Cubin file ... not found` or `InductorError: ... has no attribute 'triton_...'`.

Delete the cache for the image you were running. It's rebuilt automatically, and the model weights aren't touched:

```bash
# Nemotron (upstream vLLM image)
docker run --rm -v ~/Models/.cache:/c --entrypoint sh vllm/vllm-openai:v0.32.0 -c 'rm -rf /c/vllm-upstream'
```

For Qwen3.8-Flash-Next, the folder is `b12x-image`. For the NGC image (the generic and gpt-oss scripts), it's `vllm` and `flashinfer`.

Don't change the containers to start automatically at boot (`--restart unless-stopped`) until this is fixed. Every reboot would repeat the startup spike.

## Sources

- HP Support: [HP ZGX Nano G1n AI Station — security alerts and advisories](https://support.hp.com/ca-en/product/security-alerts/hp-zgx-nano-g1n-ai-station/model/2103747164), including the unexpected-shutdowns advisory (BIOS 1.07)
- NVIDIA Developer Forums: [GB10 abruptly powers off under heavy GPU load; `nvidia-smi -lgc 300,2200` prevents it](https://forums.developer.nvidia.com/t/gb10-abruptly-powers-off-under-heavy-gpu-load-nvidia-smi-lgc-300-2200-prevents-it/384492)
- HP: [ZGX Nano G1n datasheet](https://h20195.www2.hp.com/v2/GetDocument.aspx?docname=c09212373) and [product page](https://www.hp.com/sg-en/workstations/zgx-nano-ai-station.html)
- Related guides: [vllm-docker-gb10.md: sudden power-off](vllm-docker-gb10.md#sudden-power-off), [nemotron-3.5-super-vl.md](nemotron-3.5-super-vl.md)
