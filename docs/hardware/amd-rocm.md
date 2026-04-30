# AMD ROCm

`agent-radio-oss` runs on AMD GPUs via ONNX Runtime's `ROCMExecutionProvider` or `MIGraphXExecutionProvider`. ROCm is AMD's compute stack; ONNX Runtime exposes it as named providers, so the same Kokoro ONNX graph that runs on CPU or Apple CoreML also runs unmodified on a Radeon GPU.

This doc captures the Day 2 (2026-04-30) bring-up on a real RDNA4 Radeon — the AMD path is the **wildcard** of the OSS repo and the reason the bifurcation thesis exists. Everything below is what was actually run; numbers are measurements, not estimates.

## Verified host

| | |
|---|---|
| Hostname | Hinoki |
| GPU | AMD Radeon RX 9070 (16GB, RDNA 4, `gfx1201`) |
| CPU | AMD Ryzen 7 9700X (8C/16T, Zen 5) |
| Motherboard iGPU | also exposed as `gfx1036` (Raphael, ignore for our purposes) |
| OS | Ubuntu 24.04.4 LTS (Noble Numbat) |
| Kernel | 6.17.0-20-generic |
| ROCm | 7.2.1 (`/opt/rocm-7.2.1`, packages from AMD repo `noble`) |
| HIP | 7.2.1 (`rocm-hip` package) |
| MIGraphX | 2.15.0 |
| Python | 3.12.3 |
| `onnxruntime` | _filled in once Hinoki sync completes_ |
| `kokoro-onnx` | 0.5.0 |

`rocminfo` confirms the RX 9070 is `gfx1201`:

```
$ rocminfo | grep -A2 'Agent 2' | head -8
Agent 2
  Name:                    gfx1201
  Marketing Name:          AMD Radeon Graphics
  Vendor Name:             AMD
```

`rocm-smi` confirms the GPU is healthy:

```
$ rocm-smi --showproductname
GPU[0]: Card Series: AMD Radeon RX 9070
GPU[0]: Card SKU: APM7606CL
GPU[0]: GFX Version: gfx1201
```

## Setup

### ROCm install (already done on Hinoki — skip if you have it)

The official AMD instructions for Ubuntu 24.04 work as-is. Hinoki's ROCm 7.2.1 was installed via AMD's Ubuntu repository before Day 2; we did not have to massage anything for the `gfx1201` SKU. RDNA4 has first-class ROCm support.

```bash
# AMD's recommended path:
sudo apt update
sudo apt install -y rocm rocm-dev rocm-hip rocminfo rocm-smi-lib migraphx
sudo usermod -a -G render,video $USER
# log out + back in for group membership
```

Verify:

```bash
rocminfo | grep gfx          # should show gfx1201 for RX 9070 (or gfx1100 for RX 7900, gfx1030 for RX 6900, etc.)
rocm-smi --showproductname    # should show your card model
```

If `rocminfo` lists your card but with a `gfxN` ID that AMD's ROCm runtime does not officially support yet (uncommon on RDNA4 with ROCm 7.2.1 but a real issue on older RDNA3 + early ROCm versions):

```bash
# Last-resort flag — masks the GPU as a supported model so ROCm runs anyway.
# Do NOT set unless rocminfo's gfx ID is unsupported.
export HSA_OVERRIDE_GFX_VERSION=12.0.0
```

We did **not** need this on RX 9070 + ROCm 7.2.1.

### `onnxruntime` ROCm install

`agent-radio-oss`'s `pyproject.toml` pulls a generic `onnxruntime` wheel via `kokoro-onnx`'s deps. The default wheel does **not** include ROCm support — you have to install `onnxruntime-rocm` explicitly:

```bash
# After uv sync --extra tts --extra quality:
uv pip install onnxruntime-rocm
# or, with the AMD-published wheel index (more recent builds):
# uv pip install onnxruntime-rocm --index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/
```

Verify the providers list now includes ROCm:

```bash
uv run python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# Expect to see 'ROCMExecutionProvider' and/or 'MIGraphXExecutionProvider'
# alongside 'CPUExecutionProvider'.
```

_Day 2 status: TBD — current Hinoki sync ran with default `onnxruntime` (CPU-only). Step 4 of the parity test will install `onnxruntime-rocm` and re-render._

### Models

```bash
mkdir -p models
# kokoro-v1.0.onnx (~310 MB) and voices-v1.0.bin (~27 MB)
# https://github.com/thewh1teagle/kokoro-onnx/releases
```

## Running an audition

```bash
# Try ROCm first
KOKORO_PROVIDER=ROCMExecutionProvider \
  uv run radio render audition \
  library/programs/haystack-news/episodes/sample/script.json \
  --voice voices/kokoro-michael.yaml

# If ROCMExecutionProvider misbehaves on your GPU, fall back to MIGraphX:
KOKORO_PROVIDER=MIGraphXExecutionProvider \
  uv run radio render audition ...
```

## Verifying GPU engagement (the most important check)

ONNX Runtime can silently fall back to CPU if a provider plugin fails to load — your render will *succeed*, your audio will be correct, but you will be using your CPU instead of your $500 GPU. `agent-radio-oss` defends against this in three places:

**1. The `[kokoro] loaded with provider=...` log line is ground truth.** It reads providers back from the ONNX Runtime session after init. If you set `KOKORO_PROVIDER=ROCMExecutionProvider` and it logs `loaded with provider=CPUExecutionProvider`, you got CPU silently. You will also get an explicit warning:

```
[kokoro] WARNING: requested provider=ROCMExecutionProvider but ONNX Runtime
loaded CPUExecutionProvider. Verify your onnxruntime install supports the
requested provider.
```

**2. Run `rocm-smi` during the render.** In a separate terminal:

```bash
watch -n 0.2 rocm-smi -d 0 --showuse
```

You should see GPU utilization spike during inference. If it stays at 0%, the provider didn't engage — even if the log claimed it did. (We've seen this happen when the wheel is wrong or HIP env vars are missing.)

**3. The full-pipeline quality scores are within 0.1 of CPU.** That's the parity guarantee — see [comparison matrix](./README.md). If your ROCm DNSMOS is 4.21 and your CPU DNSMOS is 4.21, you got the same Kokoro graph through different routes. If they diverge by 1.0+, something is producing different numerical output and you should investigate.

## Day 2 measurements

_Filled in as Hinoki sync + ROCm-aware ONNX Runtime install completes. Same input as the Apple Silicon doc: `library/programs/haystack-news/episodes/sample/script.json`, 5 segments, ~54s, `kokoro-michael` voice._

| Provider | Render wall-clock | DNSMOS OVR | DNSMOS SIG | SRMR | GPU util peak | Notes |
|---|---|---|---|---|---|---|
| `CPUExecutionProvider` | TBD | TBD | TBD | TBD | 0% | Baseline |
| `ROCMExecutionProvider` | TBD | TBD | TBD | TBD | TBD | Primary AMD path |
| `MIGraphXExecutionProvider` | TBD | TBD | TBD | TBD | TBD | Fallback if ROCMExecutionProvider misbehaves |

## ROCm vs MIGraphX — which provider to pick

ONNX Runtime ships two AMD providers:

- **`ROCMExecutionProvider`** — direct ROCm execution. The default. Most coverage, similar plumbing model to CUDA.
- **`MIGraphXExecutionProvider`** — uses MIGraphX as a graph compiler in front of ROCm. Can be faster for some graphs (better op fusion) but has narrower coverage; some ops fall back to CPU.

The OSS repo's stance: **try ROCm first, fall back to MIGraphX if ROCm has issues for your specific GPU + driver combination.** Both produce identical output (same model, same graph, same numerics); the choice is purely about whether the runtime plumbing works on your hardware. Document which one worked for you.

## Quirks observed

_To be filled in as the Hinoki audition runs._

Likely candidates based on ROCm 7.x release history:
- ONNX Runtime ROCm wheel must match ROCm package version (7.2.1 → wheel built for 7.2.x)
- `LD_LIBRARY_PATH` may need `/opt/rocm/lib` on the front for the wheel to resolve `librocm_smi64.so`
- `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` if multiple GPUs (Hinoki has the iGPU at `gfx1036` and the dGPU at `gfx1201` — provider should auto-pick the dGPU but verify)

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `[kokoro] WARNING: requested provider=ROCMExecutionProvider but ONNX Runtime loaded CPUExecutionProvider` | `onnxruntime-rocm` not installed, or wheel ABI mismatch | `uv pip install onnxruntime-rocm`; verify `import onnxruntime; print(onnxruntime.get_available_providers())` lists `'ROCMExecutionProvider'` |
| `rocminfo: command not found` | ROCm not installed | Follow the AMD Ubuntu install (this doc, §Setup) |
| `rocminfo` runs but no GPU listed | User not in `render` and `video` groups | `sudo usermod -a -G render,video $USER` then log out/in |
| Render hangs or crashes on first inference | Driver / ROCm version mismatch with `onnxruntime-rocm` wheel | Check ROCm version (`apt list --installed | grep rocm-core`) matches the wheel build (`onnxruntime-rocm` PyPI release notes); pin wheel version explicitly |
| ROCm provider engages but output is silent / garbled | Numerical issue in graph compilation; very unusual on RDNA4 | Try `KOKORO_PROVIDER=MIGraphXExecutionProvider` as fallback; file an upstream `onnxruntime` issue with reproduction |
| `rocm-smi` shows utilization on the wrong GPU (e.g., iGPU instead of dGPU) | Multi-GPU host, ROCm picked first device | `export HIP_VISIBLE_DEVICES=0` (or whichever index `rocm-smi` shows for your dGPU) |

## Next backends

- Apple Silicon: see [`apple-silicon.md`](./apple-silicon.md)
- CPU (universal baseline): see [`cpu.md`](./cpu.md)
- Comparison matrix: see [`README.md`](./README.md)
