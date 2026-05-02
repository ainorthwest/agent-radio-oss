# CPU (universal baseline)

CPU is `agent-radio-oss`'s **universal baseline**. Every supported platform must produce the same audio on `KOKORO_PROVIDER=CPUExecutionProvider`. If the GPU path is broken on your hardware, CPU is always available as a fallback. If a contributor's parity numbers come in suspicious, the CPU baseline is the reference everyone else gets compared against.

## One-shot install

```bash
bash scripts/setup-cpu.sh
```

That handles `uv sync`, the whisper.cpp build, model downloads, writes `.env.suggested`, and runs a smoke check. After the script completes, `uv run radio demo` produces a Haystack News episode end-to-end.

> The script installs only the runtime extras (`tts`, `quality`). Contributors who want to run the test suite should run `uv sync --extra dev` once after setup to add ruff/mypy/pytest.

The sections below document what the script does so operators can reproduce it by hand or troubleshoot a partial install.

## When to use

- **Any environment without a GPU.** Pure server installs, Docker without `--gpus`, CI runners, low-end laptops.
- **First validation after install.** Confirm the pipeline works end-to-end on CPU before chasing GPU acceleration.
- **Reproducibility checks.** When two GPU backends disagree on output, run CPU on both hosts and compare against CPU as the reference.
- **Debugging the engine itself.** Removes ONNX Runtime provider plumbing from the stack so issues are isolated to the Kokoro model + DSP chain.

## Setup

CPU support ships in the default `onnxruntime` wheel on every platform. There is no separate package, no compile flag, no env var required beyond setting our `KOKORO_PROVIDER`:

```bash
uv sync --extra tts --extra quality
# add --extra dev as well if you want pytest/ruff/mypy for development

# Verify CPU provider is available (always true):
uv run python -c "import onnxruntime; assert 'CPUExecutionProvider' in onnxruntime.get_available_providers(); print('OK')"
```

Download the Kokoro models into `models/` (~340 MB total):

```bash
mkdir -p models
# kokoro-v1.0.onnx and voices-v1.0.bin
# https://github.com/thewh1teagle/kokoro-onnx/releases
```

## Running an audition

```bash
KOKORO_PROVIDER=CPUExecutionProvider \
  uv run radio render audition \
  library/programs/haystack-news/episodes/sample/script.json \
  --voice voices/kokoro-michael.yaml
```

You should see exactly one log line about the provider — no ONNX Runtime partition warnings (those only fire for accelerator providers):

```
[kokoro] loaded with provider=CPUExecutionProvider
```

If the log says anything other than `CPUExecutionProvider` when you've set it explicitly, your `onnxruntime` install is broken — see [Troubleshooting](#troubleshooting).

## Day 2 measurements

### Shiro (M3 Pro, macOS 26.3)

5-segment Haystack News sample, ~54s of speech, `kokoro-michael` voice.

| Metric | Value |
|---|---|
| Render wall-clock | 9.05s |
| DNSMOS overall | 4.2086 |
| DNSMOS signal | 4.3057 |
| DNSMOS background | 3.7409 |
| DNSMOS P808 | 3.5370 |
| SRMR | 4.9631 |
| Repetition score | 0.9341 |
| SNR | 6.07 dB |
| Output | 53.85s mono float32 @ 24000 Hz |

### Hinoki (AMD Ryzen 7 9700X, Ubuntu 24.04, host CPU path)

| Metric | Value |
|---|---|
| Render wall-clock | 8.25s |
| DNSMOS overall | 4.2133 |
| DNSMOS signal | 4.3001 |
| DNSMOS background | 3.7471 |
| DNSMOS P808 | 3.5460 |
| SRMR | 0.0 ⚠ — known torchaudio-on-Linux bug, not a parity issue (see commit `356khz`) |
| Repetition score | 0.934 |
| SNR | 6.26 dB |
| Output | 53.85s mono float32 @ 24000 Hz |

**Cross-host parity:** all metric deltas vs Shiro CPU under 0.01. DNSMOS OVR Δ = 0.0047. Same Kokoro graph, same float32 outputs across operating systems and CPU vendors.

### Docker (`ubuntu:24.04` clean container)

_Deferred to Day 7 smoke test._ The two host CPU runs above (Mac + Linux) already establish cross-OS portability of the CPU path. A clean-container baseline becomes meaningful when the setup scripts (Day 5) and dogfood pass (Day 7) need to validate "fresh stranger clones the repo and runs `setup-cpu.sh`" reproducibility. Day 2 closes without it.

## Why CPU is the reference

ONNX Runtime guarantees that the same graph, the same input tensors, and the same numerical precision produce **bit-identical or near-bit-identical** outputs across providers. In practice we see:

- DNSMOS / SRMR / repetition-score deltas under 0.001 between CPU and CoreML on Shiro
- Audio waveform RMS difference imperceptible on listening tests

If a non-CPU backend's audio metrics drift by more than ~0.1 from CPU on the same hardware, that's a sign of:

1. The provider didn't actually engage (silent fallback to CPU but with extra overhead)
2. The provider engaged but produced numerically different output (rare; usually FP16 vs FP32)
3. ONNX Runtime version mismatch between host and provider plugin

CPU's role is to be the boring, predictable, always-available reference everyone else gets compared against.

## Quirks observed

1. **Speed varies more by CPU generation than you'd expect.** The 82M-param Kokoro model fits well in modern CPU caches; Zen 5 / M-series perform similarly. Older Intel laptops will be 2–4x slower per segment but still usable for single auditions.
2. **First segment is slower than subsequent segments.** ONNX Runtime warms its operator implementations on first inference — typically ~1s of one-time cost.
3. **No threading config needed.** ONNX Runtime auto-detects core count. If you want to pin (e.g., for server multi-tenancy), use `provider_options` on the session — out of scope for v0.1.0.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `KOKORO_PROVIDER` env unset → log says `CPUExecutionProvider` | Default behavior — CPU is the documented fallback | Working as intended. |
| `[kokoro] WARNING: KOKORO_PROVIDER='X' not recognized` | Typo in provider name | Use one of: `CPUExecutionProvider`, `CUDAExecutionProvider`, `ROCMExecutionProvider`, `MIGraphXExecutionProvider`, `CoreMLExecutionProvider`, `DmlExecutionProvider` |
| `ModuleNotFoundError: kokoro_onnx` | Missing `--extra tts` | `uv sync --extra tts --extra quality --extra dev` |
| Render produces silence / very short WAV | Kokoro model files truncated or wrong version | Re-download `kokoro-v1.0.onnx` (310 MB) and `voices-v1.0.bin` (27 MB); verify SHA |

## whisper.cpp on CPU (Day 3a)

The CPU baseline for Pillar 3. whisper.cpp ships an excellent CPU
backend — it was the original target of the project. No GPU flag, no
provider plumbing.

### Build (any platform)

```bash
git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 4
```

CMake auto-detects platform optimizations: AVX2/AVX512 on x86,
NEON+dotprod on ARM, Accelerate.framework BLAS on macOS. On Linux you
may want `sudo apt install libopenblas-dev` first for OpenBLAS — not
required, but helps.

Download a model:

```bash
mkdir -p models
bash whisper.cpp/models/download-ggml-model.sh base.en models/
```

Wire into Agent Radio:

```bash
export RADIO_WHISPER_BIN=$(pwd)/whisper.cpp/build/bin/whisper-cli
export RADIO_WHISPER_MODEL=$(pwd)/models/ggml-base.en.bin
```

### Performance (rough)

`base.en` on the JFK 11-second sample, single core, no AVX-512:

| Host | CPU | Time |
|---|---|---|
| Hinoki | Ryzen 7 9700X (Zen 5) | sub-second |
| Shiro | Apple M3 Pro (Metal) | sub-second |
| Hinoki | Ryzen 7 9700X (CPU only) | ~2-3s |
| Generic Linux VM | 2 vCPU | ~10-15s |

`base.en` is the v0.1.0 default — small (148 MB), fast on CPU,
acceptable WER for round-trip checks. Upgrade to `medium.en` (1.5 GB)
if you need lower WER and have the time/RAM. `tiny.en` (75 MB) is
faster but its WER on TTS-rendered speech can exceed the gate
threshold; only use it for real-time / streaming.

### What this teaches the OSS thesis

Three different cross-hardware abstractions in one repo:

| Component | Abstraction | What it teaches |
|---|---|---|
| Kokoro (TTS) | ONNX Runtime providers (env var) | Wheel name + provider availability matter |
| whisper.cpp (STT) | CMake compile flag | Pick your backend at build time, not runtime |
| Stable Audio Open (Day 4) | PyTorch device string | Same device API, ROCm aliases as CUDA |

A contributor who runs all three on AMD has learned three different
ways the GPU acceleration story plays out. That's the educational
point of the OSS repo.

## Next backends

- AMD ROCm (RDNA4 + RDNA3): see [`amd-rocm.md`](./amd-rocm.md)
- Apple Silicon (M-series): see [`apple-silicon.md`](./apple-silicon.md)
- Comparison matrix: see [`README.md`](./README.md)
