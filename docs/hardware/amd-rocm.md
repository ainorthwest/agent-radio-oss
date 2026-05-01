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
| `onnxruntime-migraphx` | 1.23.2 (from `repo.radeon.com`) — replaces stock `onnxruntime` |
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

### `onnxruntime` for AMD: which wheel and where to get it

`agent-radio-oss`'s `pyproject.toml` pulls a generic `onnxruntime` wheel via `kokoro-onnx`'s deps. The default PyPI wheel is **CPU-only** — no AMD providers. You need a wheel with the AMD providers compiled in.

**The trap:** PyPI has `onnxruntime-rocm 1.22.2.post1` — old, no longer current with ROCm 7.x. AMD publishes their own wheel at `repo.radeon.com` matched to each ROCm release, but they package it as **`onnxruntime-migraphx`**, not `onnxruntime-rocm`. The naming is confusing — that single wheel exposes the AMD GPU paths to ONNX Runtime.

For ROCm 7.2.1 + Python 3.12 (Hinoki):

```bash
uv pip install --no-deps "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/onnxruntime_migraphx-1.23.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
```

`--no-deps` is correct — the wheel will replace your existing `onnxruntime` package; we don't want pip to also pull in conflicting `numpy` / etc. Verify:

```bash
uv run python -c "import onnxruntime; print(onnxruntime.__version__); print(onnxruntime.get_available_providers())"
# 1.23.2
# ['MIGraphXExecutionProvider', 'CPUExecutionProvider']
```

**Surprise: this wheel exposes `MIGraphXExecutionProvider` only — not `ROCMExecutionProvider`.** The AMD-published `onnxruntime-migraphx` build ships the MIGraphX path as the GPU provider; the direct `ROCMExecutionProvider` is not included. That's expected — AMD has been steering ONNX users to the MIGraphX path because it does more op fusion and tends to outperform the direct ROCm path on real workloads. So on AMD with ROCm 7.2.1 + this wheel, **`KOKORO_PROVIDER=MIGraphXExecutionProvider` is the GPU path**. There is no separate "ROCM provider" to fall back to.

(If you have an older ROCm install or a custom-built `onnxruntime-rocm` wheel that exposes both, both providers should work — they wrap different bits of the same stack.)

### Models

```bash
mkdir -p models
# kokoro-v1.0.onnx (~310 MB) and voices-v1.0.bin (~27 MB)
# https://github.com/thewh1teagle/kokoro-onnx/releases
```

## Running an audition

With the AMD `onnxruntime-migraphx` wheel installed (see Setup), use the MIGraphX provider:

```bash
KOKORO_PROVIDER=MIGraphXExecutionProvider \
  uv run radio render audition \
  library/programs/haystack-news/episodes/sample/script.json \
  --voice voices/kokoro-michael.yaml
```

If you have a different `onnxruntime` build that *also* exposes `ROCMExecutionProvider` (some custom builds do), `KOKORO_PROVIDER=ROCMExecutionProvider` should also work — both go through the ROCm runtime; MIGraphX adds graph compilation and op fusion on top.

## Verifying GPU engagement (the most important check)

ONNX Runtime can silently fall back to CPU if a provider plugin fails to load — your render will *succeed*, your audio will be correct, but you will be using your CPU instead of your $500 GPU. `agent-radio-oss` defends against this in three places:

**1. The `[kokoro] loaded with provider=...` log line is ground truth.** It reads providers back from the ONNX Runtime session after init. If you set `KOKORO_PROVIDER=MIGraphXExecutionProvider` and it logs `loaded with provider=CPUExecutionProvider`, you got CPU silently. You will also get an explicit warning:

```
[kokoro] WARNING: requested provider=MIGraphXExecutionProvider but ONNX Runtime
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

Same input as the Apple Silicon doc: `library/programs/haystack-news/episodes/sample/script.json`, 5 segments, ~54s, `kokoro-michael` voice.

| Provider | Render wall-clock | DNSMOS OVR | DNSMOS SIG | SRMR | Repetition | GPU peak | Notes |
|---|---|---|---|---|---|---|---|
| `CPUExecutionProvider` | **8.25s** | 4.2133 | 4.3001 | 0.0 ⚠ | 0.934 | 0% | Baseline. SRMR=0 is a known Linux/torchaudio bug, not a parity issue (see commit `356khz`). |
| `MIGraphXExecutionProvider` | **>15 min compile** ⚠ | — | — | — | — | 32% (during partition), then 3% (compile loop) | Engaged, partitioned, model loaded into VRAM (58%), but encoder graph compilation ran > 15 min without producing a WAV. Killed before save. See [Why MIGraphX is slow on Kokoro](#why-migraphx-is-slow-on-kokoro). |

**Cross-host CPU parity:** Hinoki CPU vs Shiro CPU produces audio with all metric deltas under 0.01 (DNSMOS OVR Δ = 0.0047, repetition Δ = 0.0001). Same Kokoro graph, same float32 outputs across operating systems and CPU vendors.

## Why MIGraphX is slow on Kokoro

Day 2 ran the Hinoki MIGraphX path end-to-end and logged exactly what we hit. Useful context for the next AMD operator:

The render begins by loading the Kokoro ONNX graph and asking MIGraphX to plan execution. ONNX Runtime emits the partition-coverage assignment, and MIGraphX warms up. **Within seconds, the GPU shows VRAM allocated to ~58%** — Kokoro's 310 MB model is fully resident on the RX 9070. So far so good.

Then MIGraphX hits one of Kokoro's encoder ops it can't statically shape:

```
[W:onnxruntime:Default, migraphx_execution_provider_utils.h:155 canEvalNodeArgument]
Node:/encoder/Range Input:/encoder/Cast_1_output_0 Can't eval shape
```

This isn't an error — it's a deferment. MIGraphX falls back to dynamic-shape compilation, which means **rebuilding sub-graphs at runtime**. For Kokoro's 2256-node graph with 129 partitions, that recompilation chain pinned a single CPU core at 100% for 15+ minutes before we killed the process.

Two compounding factors make this worse:

1. **No persistent graph cache.** `onnxruntime-migraphx` 1.23.2 does not enable an on-disk MIGraphX `.mxr` cache by default. Every fresh process pays the full compile cost. There is no "second run is fast" path until we configure caching explicitly via `provider_options`.

2. **The graph is built for static-shape inference.** Kokoro's encoder uses `Range` ops that depend on input length — exactly the dynamic-shape pattern MIGraphX struggles with. CoreML and CPU handle this fine because they don't try to compile + fuse aggressively; MIGraphX's whole value-prop is op fusion, which requires shape inference.

**Three v0.1.1 paths to fix this**, in increasing complexity:

1. Set MIGraphX-specific `provider_options` for shape inference and caching (env vars: `ORT_MIGRAPHX_DUMP_MODEL_PATH`, `ORT_MIGRAPHX_LOAD_COMPILED_MODEL`, etc.). May fix the issue without code changes.
2. Find or compile a `onnxruntime-rocm` wheel that exposes `ROCMExecutionProvider` directly — bypasses MIGraphX's compilation step at the cost of less op fusion.
3. Re-export Kokoro to ONNX with static input shapes baked in (operator does this once at install; runtime is then fast).

For v0.1.0-mvp, **the AMD recommendation is `KOKORO_PROVIDER=CPUExecutionProvider`** — the same Hinoki box renders the audition in 8.25s on the Ryzen 7 9700X CPU, well within OSS UX expectations. The GPU path is documented, plumbing is verified, and the perf wall is queued for v0.1.1.

## ROCm vs MIGraphX — which provider to pick

ONNX Runtime defines two AMD providers; **whether you have access to either depends on which `onnxruntime` wheel you installed:**

- **`ROCMExecutionProvider`** — direct ROCm execution. Available in custom-built `onnxruntime` and some older `onnxruntime-rocm` wheels.
- **`MIGraphXExecutionProvider`** — uses MIGraphX as a graph compiler in front of ROCm. Can be faster for some graphs (better op fusion) but has narrower coverage; some ops fall back to CPU.

**With AMD's official `onnxruntime-migraphx` wheel for ROCm 7.2.1, only `MIGraphXExecutionProvider` is exposed.** AMD has standardized on the MIGraphX path because it does the op fusion and tends to outperform direct ROCm on real workloads. So the `KOKORO_PROVIDER` env var on Hinoki gets set to `MIGraphXExecutionProvider`, full stop — there's no separate ROCm path to fall back to with this wheel.

If you compiled `onnxruntime` yourself with both providers enabled, both should work and produce identical output (same model, same graph, same numerics) — pick the one that loads without errors on your hardware.

## Quirks observed

1. **The wheel name is the main trap.** PyPI's `onnxruntime-rocm 1.22.2.post1` is older than current ROCm. AMD's current wheel for ROCm 7.2.1 is published as **`onnxruntime-migraphx` 1.23.2** (note the package-name change) at `repo.radeon.com`. It exposes only `MIGraphXExecutionProvider` — there is no separate `ROCMExecutionProvider` from this wheel.
2. **`onnxruntime-migraphx` and `onnxruntime` collide and uv silently re-resolves.** They both provide the `onnxruntime` Python module. Installing `onnxruntime-migraphx` with `--no-deps` while `onnxruntime` is also present leaves both packages in `pip list` but `onnxruntime` wins on import — so `get_available_providers()` shows only `[AzureExecutionProvider, CPUExecutionProvider]` despite the migraphx wheel being "installed." **Fix: uninstall both, then install migraphx-only.** Day 5's setup-amd.sh will codify this. Manual incantation:
   ```bash
   uv pip uninstall onnxruntime onnxruntime-migraphx
   uv pip install --no-deps "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/onnxruntime_migraphx-1.23.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
   ```
   Verify with `import onnxruntime; print(onnxruntime.get_available_providers())` — should show `['MIGraphXExecutionProvider', 'CPUExecutionProvider']`.
3. **`onnxruntime-migraphx` ships at version 1.23.2; PyPI `onnxruntime` is 1.25.x.** Slight downgrade. Acceptable for v0.1.0 (the API surface kokoro-onnx uses is stable across these versions); will need to track AMD's release cadence over time.
4. **Multi-GPU host disambiguation.** Hinoki has the dGPU at `gfx1201` (RX 9070) and the iGPU at `gfx1036` (Raphael, integrated in the Ryzen 7 9700X). MIGraphX should auto-pick the dGPU; the actual run on Hinoki picked it correctly without `HIP_VISIBLE_DEVICES`. If yours doesn't: `export HIP_VISIBLE_DEVICES=0` (or whichever index `rocm-smi` shows for your dGPU).
5. **MIGraphX graph-compile time on Kokoro is the perf wall.** First-render for the 5-segment audition exceeded 15 minutes of CPU work for graph compilation, never producing a WAV. See [Why MIGraphX is slow on Kokoro](#why-migraphx-is-slow-on-kokoro). For v0.1.0, `CPUExecutionProvider` on AMD hardware is the recommended path (8.25s on Hinoki Ryzen 9700X — same audio output, dramatically better UX).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `[kokoro] WARNING: requested provider=MIGraphXExecutionProvider but ONNX Runtime loaded CPUExecutionProvider` | AMD `onnxruntime-migraphx` wheel not installed, OR `onnxruntime` (CPU PyPI wheel) is co-installed and winning the import race | Uninstall both, then install only `onnxruntime-migraphx` (see [Quirks](#quirks-observed) #2). Verify with `import onnxruntime; print(onnxruntime.get_available_providers())` |
| MIGraphX render hangs at >5 min wall-clock with no progress | First-time graph compile on Kokoro's encoder (`canEvalNodeArgument` warning earlier in the log) | Real perf wall, not a hang. See [Why MIGraphX is slow on Kokoro](#why-migraphx-is-slow-on-kokoro). For v0.1.0, fall back to `KOKORO_PROVIDER=CPUExecutionProvider`. |
| `rocminfo: command not found` | ROCm not installed | Follow the AMD Ubuntu install (this doc, §Setup) |
| `rocminfo` runs but no GPU listed | User not in `render` and `video` groups | `sudo usermod -a -G render,video $USER` then log out/in |
| Render hangs or crashes on first inference | Driver / ROCm version mismatch with `onnxruntime-rocm` wheel | Check ROCm version (`apt list --installed | grep rocm-core`) matches the wheel build (`onnxruntime-rocm` PyPI release notes); pin wheel version explicitly |
| ROCm provider engages but output is silent / garbled | Numerical issue in graph compilation; very unusual on RDNA4 | Try `KOKORO_PROVIDER=MIGraphXExecutionProvider` as fallback; file an upstream `onnxruntime` issue with reproduction |
| `rocm-smi` shows utilization on the wrong GPU (e.g., iGPU instead of dGPU) | Multi-GPU host, ROCm picked first device | `export HIP_VISIBLE_DEVICES=0` (or whichever index `rocm-smi` shows for your dGPU) |

## whisper.cpp on AMD ROCm (Day 3a)

Quality Pillar 3 — speech intelligibility — uses whisper.cpp (MIT,
CMake-based, hardware-portable via compile flags). On Hinoki the HIP
backend works on the first try.

### Hinoki build (RX 9070, gfx1201, ROCm 7.2.1)

```bash
git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build-hip -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release \
  -DAMDGPU_TARGETS=gfx1201
cmake --build build-hip --config Release -j 8
```

Compiles cleanly against `/opt/rocm-7.2.1/lib/llvm/bin/clang++` (Clang
22.0.0). The `whisper-cli` binary lands in `build-hip/bin/`. No special
`HSA_OVERRIDE_GFX_VERSION` flag needed — gfx1201 is supported by ROCm
7.2.1 directly.

Download a model:

```bash
mkdir -p models
bash whisper.cpp/models/download-ggml-model.sh base.en models/
```

Wire it into Agent Radio via env vars (no hard-coded paths in `src/stt.py`):

```bash
export RADIO_WHISPER_BIN=$(pwd)/whisper.cpp/build-hip/bin/whisper-cli
export RADIO_WHISPER_MODEL=$(pwd)/models/ggml-base.en.bin
```

### Verifying GPU engagement

```bash
$ ./whisper.cpp/build-hip/bin/whisper-cli \
    -m models/ggml-base.en.bin -f whisper.cpp/samples/jfk.wav 2>&1 | grep -E 'gpu|backend|Radeon'
whisper_init_with_params_no_state: use gpu    = 1
whisper_init_with_params_no_state: gpu_device = 0
  Device 0: AMD Radeon RX 9070, gfx1201 (0x1201), VMM: no, Wave Size: 32, VRAM: 16304 MiB
  Device 1: AMD Radeon Graphics, gfx1036 (0x1036), VMM: no, Wave Size: 32, VRAM: 13804 MiB
whisper_init_with_params_no_state: backends   = 2
whisper_backend_init_gpu: device 0: ROCm0 (type: 1)
whisper_backend_init_gpu: using ROCm0 backend
```

ROCm0 backend engages on the dGPU automatically; the iGPU is enumerated
but skipped. JFK transcription returns the expected text.

### Pipeline test

The `tests/test_stt.py` integration test is gated on
`RADIO_WHISPER_BIN` / `RADIO_WHISPER_MODEL`; with both set, all 30
stt tests pass on Hinoki:

```
$ uv run pytest tests/test_stt.py -v
30 passed in 0.59s
```

### Vulkan fallback (plumbed, not validated in v0.1.0)

If your AMD setup lacks ROCm or HIP fails, whisper.cpp also supports a
Vulkan backend (`-DGGML_VULKAN=ON`). Hinoki has `vulkaninfo` available;
this path is plumbed but not validated in v0.1.0 since HIP works.
Documented for AMD users on RDNA cards where ROCm support is patchy.

### What this teaches the OSS thesis

Day 2 found that **Kokoro via ONNX/MIGraphX hangs on RX 9070** at first
graph compile (>15 min, never produced WAV). Day 3 found that
**whisper.cpp via HIP works on RX 9070 on the first try**. Same GPU,
two different hardware abstractions, two different outcomes. The gaps
in local-edge inference are real, and they live at the abstraction
layer, not the silicon. Document them, ship them, let operators see
the seam.

## Next backends

- Apple Silicon: see [`apple-silicon.md`](./apple-silicon.md)
- CPU (universal baseline): see [`cpu.md`](./cpu.md)
- Comparison matrix: see [`README.md`](./README.md)
