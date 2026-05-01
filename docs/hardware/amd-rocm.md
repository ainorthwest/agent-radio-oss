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
| `MIGraphXExecutionProvider` | **fails after compile** ⚠ | — | — | — | — | 58% VRAM, partition + compile complete in ~30s | Compile + .mxr cache write succeed, then MIGraphX's runtime API throws a null-pointer error before audio is produced. See [What's actually broken on the AMD GPU path](#whats-actually-broken-on-the-amd-gpu-path). |

**Cross-host CPU parity:** Hinoki CPU vs Shiro CPU produces audio with all metric deltas under 0.01 (DNSMOS OVR Δ = 0.0047, repetition Δ = 0.0001). Same Kokoro graph, same float32 outputs across operating systems and CPU vendors.

## What's actually broken on the AMD GPU path

Day 2 reported a "15-minute hang on first compile" and recommended
CPU on AMD. The CPU recommendation stands, but the diagnosis was
incomplete. A follow-up investigation
([`docs/investigations/kokoro-amd-rocm.md`](../investigations/kokoro-amd-rocm.md))
re-ran the path with explicit `provider_options` and found:

1. **First-inference graph compile completes in 25-41 seconds**, not
   15+ minutes. The Day 2 wait was killed before compile finished and
   was running with default provider options that may have been
   slower.
2. **MIGraphX caching DOES work.** A 32 MB `.mxr` is written under
   `migraphx_model_cache_dir`, and a warm cache reaches `sess.run` in
   ~1 s.
3. **The actual blocker is downstream of compile.** After compile +
   cache write succeed, MIGraphX's runtime parameter-shape API throws:

   ```
   migraphx_program_parameter_shapes_size: Error:
     .../AMDMIGraphX/src/api/api.cpp:1345: operator():
     Bad parameter program_parameter_shapes: Null pointer
   ```

   ONNX Runtime surfaces this as
   `[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code
   returned while running MGXKernel_graph_main_graph_*_1 node ...
   Status Message: Failed to call function`. **No audio is produced.**
   The bug is invariant under fp16/fp32 and cache on/off (4/4
   ablations identical).
4. **MIGraphX itself is healthy.**
   `migraphx-driver perf --test` runs at 46,819 inferences/sec on the
   gfx1201 GPU. The bug is specific to the ONNX Runtime → MIGraphX
   hand-off for this graph.
5. **The `/encoder/Range` warning is benign.** It tells MIGraphX to
   route that node to CPU (correct behavior); the surrounding subgraph
   still compiles.
6. **Direct MIGraphX cannot parse Kokoro at all** —
   `migraphx-driver perf --onnx kokoro-v1.0.onnx` throws
   `PARSE_RANGE: limit arg dynamic shape is not supported`. ONNX
   Runtime correctly avoids this by partitioning Range out before
   handing the graph to MIGraphX. So this parser limitation is real
   but doesn't matter for the ORT path.

GitHub code search returned **0 hits** for the exact null-pointer error
string — this configuration appears to be a novel observation. Related
upstream tickets:

- [ROCm/AMDMIGraphX#4618](https://github.com/ROCm/AMDMIGraphX/issues/4618)
  — open, AMD-assigned. Reporter on gfx1101 + Kokoro v1.0 + ORT
  migraphx 1.23.2 hits the same `/encoder/Range` warning, observes
  per-shape recompiles, and reports kernel 6.17 + Ubuntu 25.10 as the
  proximate cause. Hinoki is on Ubuntu 24.04 + kernel 6.17.
- [ROCm/AMDMIGraphX#4029](https://github.com/ROCm/AMDMIGraphX/issues/4029)
  — closed, gfx1201 + ArtCNN, AMD acknowledged "compile time can be
  15-25 min, expected behavior" but did not ship a fix.

For v0.1.0-mvp, **the AMD recommendation remains
`KOKORO_PROVIDER=CPUExecutionProvider`** — the Hinoki Ryzen 7 9700X
renders the audition in 8.25s. v0.1.1 next-step is filing the
post-compile null-pointer with AMD using our reproducer; until that's
fixed, the GPU path doesn't ship.

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
5. **MIGraphX graph-compile is fast; the runtime hand-off is broken.** First-inference compile completes in 25-41s and writes a working .mxr cache, but the post-compile MIGraphX runtime API then throws a null-pointer error and no audio is produced. See [What's actually broken on the AMD GPU path](#whats-actually-broken-on-the-amd-gpu-path). For v0.1.0, `CPUExecutionProvider` on AMD hardware is the recommended path (8.25s on Hinoki Ryzen 9700X — known-good audio output).
6. **`migraphx_model_cache_dir` is the real cache option.** Older docs and 2025 forum posts mention `migraphx_save_compiled_model` / `migraphx_load_compiled_path` — **these are rejected by the Python bindings as of ORT 1.21+** ([microsoft/onnxruntime#25379](https://github.com/microsoft/onnxruntime/issues/25379)). Use `migraphx_model_cache_dir` (or env `ORT_MIGRAPHX_MODEL_CACHE_PATH`). The cache file is named `{migraphx_version_hex}-{graph_id}-{hash(gcnArchName)}-{hash(input_shapes)}.mxr`, so each `sequence_length` value gets its own cache entry.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `[kokoro] WARNING: requested provider=MIGraphXExecutionProvider but ONNX Runtime loaded CPUExecutionProvider` | AMD `onnxruntime-migraphx` wheel not installed, OR `onnxruntime` (CPU PyPI wheel) is co-installed and winning the import race | Uninstall both, then install only `onnxruntime-migraphx` (see [Quirks](#quirks-observed) #2). Verify with `import onnxruntime; print(onnxruntime.get_available_providers())` |
| MIGraphX render fails with `RUNTIME_EXCEPTION: ... Failed to call function` after ~30s | Post-compile null-pointer in `migraphx_program_parameter_shapes_size` — known novel issue on this stack. See [What's actually broken on the AMD GPU path](#whats-actually-broken-on-the-amd-gpu-path). | For v0.1.0, fall back to `KOKORO_PROVIDER=CPUExecutionProvider`. The GPU path is queued for v0.1.1 pending an AMD upstream fix. |
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
