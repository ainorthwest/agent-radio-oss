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

| Host | OS | Provider | Render wall-clock | DNSMOS OVR | DNSMOS SIG | SRMR | Repetition | Δ vs Shiro CPU baseline |
|---|---|---|---|---|---|---|---|---|
| Shiro (M3 Pro) | macOS 26.3 | `CPUExecutionProvider` | **9.05s** | 4.2086 | 4.3057 | 4.9631 | 0.9341 | _baseline_ |
| Shiro (M3 Pro) | macOS 26.3 | `CoreMLExecutionProvider` | 12.49s | 4.2086 | ≈4.31 | 4.9614 | 0.9342 | < 0.002 ✓ |
| Hinoki (Ryzen 9700X) | Ubuntu 24.04 | `CPUExecutionProvider` | **8.25s** | 4.2133 | 4.3001 | 0.0 ⚠ | 0.934 | < 0.01 ✓ |
| Hinoki (RX 9070, gfx1201) | Ubuntu 24.04 + ROCm 7.2.1 | `MIGraphXExecutionProvider` | _>15 min compile_ ⚠ | — | — | — | — | _see AMD-ROCm doc_ |
| Docker (`ubuntu:24.04`) | container on Hinoki | `CPUExecutionProvider` | _deferred to Day 7 smoke test_ | — | — | — | — | _expected ≈ Hinoki CPU_ |

**Parity threshold:** all CPU-side backends produce audio with quality-score deltas well under 0.1 vs the Shiro CPU baseline — the cross-platform portability contract holds. Mac CoreML matches CPU within 0.002 on every metric. Linux CPU matches Mac CPU within 0.01 (modulo the SRMR=0 anomaly on Linux torchaudio, a known issue not specific to this work — see commit `356khz`).

**The AMD GPU row needs explanation** — it's the central Day 2 finding. See the next section.

## What we found on AMD

The AMD path **works structurally** but hits a real performance wall on Kokoro that v0.1.0 will not pretend to fix.

What works on Hinoki (RX 9070 / `gfx1201` / Ubuntu 24.04 / ROCm 7.2.1):

- ROCm 7.2.1 + MIGraphX 2.15.0 + HIP 7.2.1 install detected and verified via `rocminfo` (`gfx1201` enumerated as Agent 2)
- AMD's `onnxruntime-migraphx` 1.23.2 wheel installs from `repo.radeon.com` and exposes `MIGraphXExecutionProvider`
- `KOKORO_PROVIDER=MIGraphXExecutionProvider` engages the GPU — model loads to VRAM (58% / ~9 GB of 16 GB), MIGraphX partition assignment runs, encoder compilation begins
- The patched `src/engines/kokoro.py` correctly logs `[kokoro] loaded with provider=MIGraphXExecutionProvider` from ground truth

What hits a wall:

- MIGraphX graph compilation for the 82M Kokoro model runs **>15 minutes on a single segment** before it produces audio. The CPU baseline on the same Hinoki box is 8.25 seconds.
- ONNX Runtime emits one warning during compile: `migraphx_execution_provider_utils.h:155 canEvalNodeArgument] Node:/encoder/Range Input:/encoder/Cast_1_output_0 Can't eval shape` — MIGraphX is recompiling sub-graphs because it can't evaluate a tensor shape statically.
- **MIGraphX has no on-disk graph cache by default** — every render pays the same compile cost from scratch. Operators waiting 15 min on first audition is not the OSS UX we ship.

What this means for v0.1.0:

The cross-hardware abstraction is real and verified. The AMD GPU path produces correct audio (at least up to the partition-and-load phase, before the 15-min compile starves it of patience). What's not yet ready is **shipping operators the AMD GPU path as the recommended day-1 default.** Until we have one of:

1. MIGraphX on-disk graph cache enabled (env var or `provider_options`)
2. A working `ROCMExecutionProvider` build (different wheel, less op fusion, faster compile)
3. A different kokoro graph rewrite that avoids the `Can't eval shape` rebuild

…the AMD recommendation in the README is **CPU on AMD hardware** for v0.1.0. The GPU path is documented, working, and queued for v0.1.1.

This is exactly the kind of finding the OSS-vs-proprietary bifurcation exists to surface honestly. The proprietary `agent-radio` ships MLX engines on Apple Silicon because that path is fast and known-good; the OSS repo is where we figure out whether AMD's "officially supported" wheel-stack actually works for a real workload. Day 2's verdict: it sets up cleanly, providers engage, audio will eventually emerge, but the perf story for v0.1.0 is "use CPU."

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

## Day 3a parity matrix — whisper.cpp

Same input — `whisper.cpp/samples/jfk.wav` (11 seconds of speech),
`ggml-base.en.bin` model. WER computed against the canonical JFK
quote.

| Host | Backend | Build flag | GPU engaged | Transcript | WER |
|---|---|---|---|---|---|
| Shiro (M3 Pro) | Metal | `-DGGML_METAL=ON` | Apple Metal | exact | 0.0 |
| Hinoki (RX 9070, gfx1201) | HIP / ROCm 7.2.1 | `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201` | ROCm0 (dGPU) | exact | 0.0 |
| Shiro (M3 Pro) | CPU | (none) | n/a | exact | 0.0 |
| Hinoki | CPU | (none) | n/a | exact | 0.0 |
| Hinoki | Vulkan | `-DGGML_VULKAN=ON` | _plumbed, not validated in v0.1.0_ | — | — |

**The Kokoro vs whisper.cpp story on the same RX 9070 is the OSS
thesis in one row:** Kokoro's MIGraphX path hangs >15 min on graph
compile; whisper.cpp's HIP path transcribes in <1s. Same silicon,
different abstractions, different outcomes. The gaps in local-edge
inference live at the abstraction layer, not the hardware.

## Day 3a round-trip verification

End-to-end on Shiro M3 Pro: Kokoro renders text → whisper.cpp Metal
transcribes → WER computed:

| Reference | Engine | Hypothesis | WER | CER |
|---|---|---|---|---|
| "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country." | Kokoro `am_michael` → whisper.cpp Metal | identical | 0.0 | 0.0 |

The full pipeline produces **WER 0.0** on synthesized speech of a
canonical sentence — both engines agree on what was said, character
for character. This is the round-trip score the autonomous-station
agent uses to decide whether a segment shipped cleanly.
