# Hardware support

`agent-radio-oss` is engine-agnostic and hardware-agnostic by design. The Kokoro TTS engine runs on every supported backend by routing through ONNX Runtime's named execution providers — same model, same graph, same audio output, different hardware.

This directory documents what we have actually verified, on actual hardware, with actual measurements. Everything below is a measurement, not a claim.

## Quick pick by hardware

| You have | Read this |
|---|---|
| AMD Radeon (any RDNA generation) | [`amd-rocm.md`](./amd-rocm.md) |
| Mac with M1 / M2 / M3 / M4 | [`apple-silicon.md`](./apple-silicon.md) |
| CPU only (Intel laptop, Linux server, Docker) | [`cpu.md`](./cpu.md) |
| NVIDIA GPU | _Day 5+ — `setup-cuda.sh` ships the install path; verified bring-up deferred to long plan_ |
| Windows + DirectML | _not yet validated; the engine accepts `DmlExecutionProvider`, but no Day 2 evidence_ |

## How we picked the providers we ship

Three different cross-hardware abstractions, taught through three different models:

| Capability | Library | Hardware abstraction |
|---|---|---|
| TTS | Kokoro ONNX | ONNX Runtime **provider strings** (`CPUExecutionProvider`, `ROCMExecutionProvider`, `CoreMLExecutionProvider`, …) |
| STT (Day 3) | whisper.cpp | **Compile flags** (`GGML_HIPBLAS=1`, `GGML_VULKAN=1`, `GGML_METAL=1`) |
| Music (Day 4) | Stable Audio Open | **PyTorch device strings** (`cuda` (HIP-aliased on ROCm), `mps`, `cpu`) |

A contributor who learns this OSS stack learns three different cross-hardware idioms in one repo. That's the educational point.

## Day 2 parity matrix

Same input — `library/programs/haystack-news/episodes/sample/script.json`, 5 segments, 53.85s of speech, `kokoro-michael` voice. Quality scoring via `python -m src.quality`.

| Host | OS | Provider | Render wall-clock | DNSMOS OVR | DNSMOS SIG | SRMR | Repetition | Δ vs CPU baseline |
|---|---|---|---|---|---|---|---|---|
| Shiro (M3 Pro) | macOS 26.3 | `CPUExecutionProvider` | **9.05s** | 4.2086 | 4.3057 | 4.9631 | 0.9341 | _baseline_ |
| Shiro (M3 Pro) | macOS 26.3 | `CoreMLExecutionProvider` | 12.49s | 4.2086 | ≈4.31 | 4.9614 | 0.9342 | < 0.002 ✓ |
| Hinoki (Ryzen 9700X) | Ubuntu 24.04 | `CPUExecutionProvider` | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Hinoki (RX 9070, gfx1201) | Ubuntu 24.04 + ROCm 7.2.1 | `MIGraphXExecutionProvider` | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Docker (`ubuntu:24.04`) | container on Hinoki | `CPUExecutionProvider` | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Parity threshold:** all backends must produce audio with quality-score deltas under 0.1 vs the host's CPU baseline. Same hardware, different provider → same audio. Different hardware, same provider → same audio. This is the ONNX-Runtime portability contract; we measure it because we don't take it on faith.

## Verifying provider engagement on your own machine

`agent-radio-oss` defends against silent CPU fallback in three places:

**1. Engine ground-truth log.** `src/engines/kokoro.py` reads providers back from the ONNX Runtime session after init and logs the *actual* provider, not what was requested:

```
[kokoro] loaded with provider=CoreMLExecutionProvider     # truth
```

If you set `KOKORO_PROVIDER=ROCMExecutionProvider` and the log says `loaded with provider=CPUExecutionProvider`, the runtime silently fell back. You will also get an explicit `WARNING` line.

**2. ONNX Runtime stderr.** Accelerator providers emit a partition-coverage log line on the first inference (Apple's `coreml_execution_provider.cc:113`, AMD's `rocm_execution_provider.cc:...`, etc.). Their absence is a signal that the provider didn't actually attach.

**3. Hardware utilization tools.** `rocm-smi`, `nvidia-smi`, macOS Activity Monitor's GPU pane. If you set `KOKORO_PROVIDER` to a GPU provider and `rocm-smi -d 0 --showuse` stays at 0% during render, you're on CPU regardless of what the log says.

## How we discovered the provider abstraction was broken

This was the central finding of Day 2 and the reason this repo has a regression test for provider plumbing.

When `agent-radio-oss` was first ported from the upstream `agent-radio` repo (Day 1, 2026-04-29), `src/engines/kokoro.py` was written against `kokoro-onnx` 0.4.x, which accepted a `providers=` kwarg in its constructor. Day 2 ran `uv sync --extra tts --extra dev` on a fresh Shiro venv and got `kokoro-onnx` 0.5.0 — which:

- **Removed the `providers=` kwarg.** The engine's `try/except TypeError` fallback caught this silently.
- **Reads `ONNX_PROVIDER` (not `KOKORO_PROVIDER`)** from env to pick its backend.

Net effect: every operator setting `KOKORO_PROVIDER=CoreMLExecutionProvider` on a fresh install got CPU. Silently. The engine still printed `[kokoro] loaded with provider=CoreMLExecutionProvider` because that line logged what was *requested*, not what loaded.

We caught it on Day 2 because:
- Shiro CoreML rendered in 8.52s — suspiciously close to CPU's 8.85s for a graph that should benefit from ANE
- ONNX Runtime did **not** emit its `coreml_execution_provider.cc` partition log
- A `KOKORO_PROVIDER=BogusProvider` test produced no error — should have warned and fallen back to CPU, but the fallback path was already what was running

The fix (commit `138d094`):
- Translate `KOKORO_PROVIDER` → `os.environ["ONNX_PROVIDER"]` before instantiating Kokoro
- Read `sess.get_providers()` after init to log ground truth
- Warn explicitly when ONNX Runtime did not honor the request
- Pin the public contract with `tests/test_engines.py` (11 tests) so future kokoro-onnx version drift can't silently break the cross-hardware story again

**Lesson for the rest of the sprint:** every wrapped library is a candidate for the same class of regression. Day 3 (whisper.cpp via subprocess CLI flags) and Day 4 (Stable Audio Open via `stable-audio-tools` Python API) both have the same risk — log what *actually* ran, not what was requested. Added to `oss-mvp-sprint.md` Risk register.
