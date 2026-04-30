# CPU (universal baseline)

CPU is `agent-radio-oss`'s **universal baseline**. Every supported platform must produce the same audio on `KOKORO_PROVIDER=CPUExecutionProvider`. If the GPU path is broken on your hardware, CPU is always available as a fallback. If a contributor's parity numbers come in suspicious, the CPU baseline is the reference everyone else gets compared against.

## When to use

- **Any environment without a GPU.** Pure server installs, Docker without `--gpus`, CI runners, low-end laptops.
- **First validation after install.** Confirm the pipeline works end-to-end on CPU before chasing GPU acceleration.
- **Reproducibility checks.** When two GPU backends disagree on output, run CPU on both hosts and compare against CPU as the reference.
- **Debugging the engine itself.** Removes ONNX Runtime provider plumbing from the stack so issues are isolated to the Kokoro model + DSP chain.

## Setup

CPU support ships in the default `onnxruntime` wheel on every platform. There is no separate package, no compile flag, no env var required beyond setting our `KOKORO_PROVIDER`:

```bash
uv sync --extra tts --extra quality --extra dev

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

_To be filled in once `uv sync --extra tts --extra quality --extra dev` completes on Hinoki and the audition is run with `KOKORO_PROVIDER=CPUExecutionProvider`. Expected: identical audio metrics to Shiro CPU; render wall-clock should be in the same order of magnitude (CPU compute differs but Kokoro is small enough that both modern CPUs handle it comfortably)._

### Docker (`ubuntu:24.04` clean container, host-mounted models)

_To be filled in. The Docker run validates a zero-GPU clean-room install: only the bits available in the bare Ubuntu image plus what `uv sync --extra tts --extra quality` pulls in. This is the closest we get to "fresh stranger clones the repo" reproducibility before Day 5's setup scripts._

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

## Next backends

- AMD ROCm (RDNA4 + RDNA3): see [`amd-rocm.md`](./amd-rocm.md)
- Apple Silicon (M-series): see [`apple-silicon.md`](./apple-silicon.md)
- Comparison matrix: see [`README.md`](./README.md)
