# agent-radio-oss

[![CI](https://github.com/ainorthwest/agent-radio-oss/actions/workflows/ci.yml/badge.svg)](https://github.com/ainorthwest/agent-radio-oss/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

> **Open-source autonomous AI radio.** Apache 2.0.
> Core code Apache-2.0; optional DSP extras include a GPL-v3 package — full audit in [LICENSES.md](./LICENSES.md).

> **Status:** `v0.1.0-mvp` pre-release. Day 6 of a 7-day sprint. Days 1–5 shipped; ships Day 7.

## What this is

`agent-radio-oss` is a hardware-agnostic, commercial-deployable autonomous AI radio platform. One operator, one machine, one setup script: gather → render → quality-check → broadcast, with the agent driving the loop. The OSS distribution is the engine-agnostic, license-clean subset of [Lightcone Studios' production stack](https://github.com/ainorthwest/agent-radio); this repo is what a stranger can clone and run their own commercial station with.

The bifurcation rule is **commercial deployability and AINW-specificity**: anything generic and permissive ships here; anything proprietary, MLX-bound, or AINW-specific stays upstream.

## Will it run on my machine? (the 60-second answer)

| Hardware | OS | Setup script | Status | Bring-up doc |
|---|---|---|---|---|
| Apple Silicon (M1+) | macOS 14+ | [`scripts/setup-mac.sh`](./scripts/setup-mac.sh) | ✓ Verified (M3 Pro) | [`docs/hardware/apple-silicon.md`](./docs/hardware/apple-silicon.md) |
| AMD ROCm (RDNA3+) | Ubuntu 22.04+ | [`scripts/setup-amd.sh`](./scripts/setup-amd.sh) | ✓ Plumbed; CPU-recommended\* | [`docs/hardware/amd-rocm.md`](./docs/hardware/amd-rocm.md) |
| NVIDIA CUDA | Linux | [`scripts/setup-cuda.sh`](./scripts/setup-cuda.sh) | ⚠ Untested in v0.1.0 | [`docs/hardware/nvidia-cuda.md`](./docs/hardware/nvidia-cuda.md) |
| CPU only | Linux / macOS / WSL | [`scripts/setup-cpu.sh`](./scripts/setup-cpu.sh) | ✓ Verified (Mac + Linux) | [`docs/hardware/cpu.md`](./docs/hardware/cpu.md) |

\* **AMD asterisk:** [AMDMIGraphX#4618](https://github.com/ROCm/AMDMIGraphX/issues/4618) currently blocks the GPU Kokoro path on `gfx1201` with a post-compile null-pointer in MIGraphX runtime. The Ryzen 7 9700X CPU renders the audition in ~8s and matches Mac CPU within Δ 0.005 on every quality metric — `setup-amd.sh` defaults to CPU and gates `--enable-migraphx` behind a warning. Full root-cause analysis with command-by-command evidence in [`docs/investigations/kokoro-amd-rocm.md`](./docs/investigations/kokoro-amd-rocm.md). `v0.1.1` will revisit when upstream lands a fix.

## Quick start

Pick your platform and run one setup script:

**Apple Silicon:**
```bash
git clone https://github.com/ainorthwest/agent-radio-oss && cd agent-radio-oss
bash scripts/setup-mac.sh
uv run radio demo
```

**AMD ROCm:**
```bash
git clone https://github.com/ainorthwest/agent-radio-oss && cd agent-radio-oss
bash scripts/setup-amd.sh
uv run radio demo
```

**CPU (any platform):**
```bash
git clone https://github.com/ainorthwest/agent-radio-oss && cd agent-radio-oss
bash scripts/setup-cpu.sh
uv run radio demo
```

**NVIDIA CUDA:** [`scripts/setup-cuda.sh`](./scripts/setup-cuda.sh) ships untested in `v0.1.0` — the script is identical-shape to `setup-amd.sh` and is gated behind a clear UNTESTED banner. First contributor with NVIDIA hardware is invited to run it and open a validation issue.

The setup script handles `uv sync`, `whisper.cpp` build, model downloads (SHA-pinned), and platform detection. `radio demo` writes a complete output dir under `library/programs/haystack-news/episodes/{timestamp}/` containing `episode.mp3`, `transcript.txt`, `transcript.srt`, `quality.json`, `episode.md`, `chapters.json`, and a `DEMO_README.md` that walks you through what each artifact means.

## What's in the box

```
curate → script-quality → render → anomaly-detect → quality (3-pillar) → publisher → distribute → stream
```

- **TTS:** Kokoro ONNX — 82M params, dispatched across CPU / CUDA / ROCm / CoreML via the `KOKORO_PROVIDER` env var. The renderer reads back the *actual* loaded provider from the ONNX session and warns on silent CPU fallback.
- **STT:** `whisper.cpp` via subprocess (Pillar 3 WER, transcripts, SRT). Hardware backend is baked at compile time via CMake flags (`GGML_METAL` / `GGML_HIP` / `GGML_CUDA` / vanilla CPU) — `whisper.cpp` HIP works on the same RX 9070 silicon where Kokoro MIGraphX hangs. That contrast is the educational point of this repo.
- **Quality:** 23-metric eval stack across three pillars — librosa (spectral), torchmetrics (DNSMOS / SRMR / PESQ / STOI), and round-trip WER via `whisper.cpp`. Each pillar is independent; failures bubble up with named verdicts (`ship` / `review` / `reject`).
- **Editor + segment cache + anomaly detector:** content-addressed WAV cache keyed on `sha256(text + speaker + register + voice_profile + engine)`. Re-render one segment without re-rendering the episode. Post-render checks (silence-ratio, WER outlier, duration anomaly) carry action suggestions so the agent acts without interpreting raw scores.
- **Publisher:** Podcasting 2.0 (`<podcast:locked>`, `<podcast:person>`, `<podcast:transcript>`, `<podcast:chapters>`) plus `episode.md`, `chapters.json`, `episode.txt` (agent payload), and `episode.jsonld`. Per-show `llms.txt` indexes per [llmstxt.org](https://llmstxt.org).
- **Music:** `v0.1.0` overlays pre-rendered music assets. Stable Audio Open generation is deferred to `v0.1.1` ([GH#9](https://github.com/ainorthwest/agent-radio-oss/issues/9)) — needs a PyTorch-on-ROCm validation pass that did not fit the sprint.

## How honest is `v0.1.0`?

The OSS station aspires to be excellent at eight things. `v0.1.0` is honest about which ones it nails:

- **Strong:** **(4)** render + anomaly + surgical re-render — the segment cache + editor + anomaly loop closes the autonomous-station correction loop. **(5)** audio engineering — 23-metric quality stack with named verdicts, real round-trip WER.
- **Partial:** **(6)** broadcast — AzuraCast HTTP client wired, no scheduler. **(8)** Agent Experience — skills bundle scaffolded (`edit-script`, `publish-episode`) plus `radio demo` for the new-operator path; depth coming.
- **Thin:** **(1)** Wire Desk story selection, **(2)** editorial tracking across days, **(3)** show-aware script writing, **(7)** autoresearch feedback into voice tuning. These are the AINW-flavored production newsroom and stay upstream by the bifurcation rule. The OSS station-runner agent (Hermes-driven reference profile) is written fresh in `v0.1.1+`.

`v0.1.1+` targets the editorial pillars; `v0.1.0` nails the rendering and audio-science halves of the loop.

## Two front doors: CLI and Skills

Both surfaces are first-class and call the same `src/*` primitives.

- **`radio` CLI** (Typer, `src/cli/`): operator and debugging surface. `radio demo`, `radio render audition`, `radio run pipeline`, `radio edit script`, `radio publish episode`, etc. Run `uv run radio --help`.
- **Skills bundle** (`skills/`): the agent's operating manual. Hermes / Claude Code / other agent harnesses load `skills/*` natively. Decompose into both major steps (`render-episode`, `check-quality`) and minor steps (`render-segment`, `decide-ship-review-reject`) so the agent recovers from partial failures without re-running the whole episode.

See [`CLAUDE.md`](./CLAUDE.md) for the full architecture, and [`oss-mvp-sprint.md`](./oss-mvp-sprint.md) for the day-by-day sprint plan and "what we are NOT doing" list.

## License

- **Code:** Apache 2.0 — see [LICENSE](./LICENSE).
- **Bundled model weights:** Apache-2.0 (Kokoro), MIT (`whisper.cpp` + Whisper `base.en`).
- **Per-component audit** including all 108 transitive Python dependencies and the GPL-v3 disposition for the `[tts]` extras: [LICENSES.md](./LICENSES.md).
- **Attribution:** [NOTICE](./NOTICE).

SaaS / internal use is unaffected by the GPL-v3 caveat (no AGPL network clause). Distribution of an installed binary inherits GPL-v3 §4 source-availability obligations from the `pedalboard` and `phonemizer-fork` packages — `v0.1.1` queues both for replacement with permissive equivalents. Read [LICENSES.md](./LICENSES.md#the-gpl-v3-question) before redistributing a build.

## Where this came from

This repo is the open distribution of [`agent-radio`](https://github.com/ainorthwest/agent-radio) — Lightcone Studios' production AINW Radio stack. The split is by deployability: AINW's specific feed list, MLX engines, Steward / Bard agents, and CC-BY-NC music weights stay proprietary; the generic primitives (RSS, web fetch, Discourse API, render pipeline, quality stack, broadcast client) ship here under Apache 2.0. The full sprint plan, the bifurcation rule, and the daily log are in [`oss-mvp-sprint.md`](./oss-mvp-sprint.md).

## Contributing

Sprint in progress; `v0.1.0` ships Day 7 (target: 2026-05-03). Issues welcome now; PRs welcome after release.

## Acknowledgements

- [hexgrad](https://github.com/hexgrad) and the Kokoro contributors — Apache-2.0 TTS weights that make this whole thing work.
- [Georgi Gerganov](https://github.com/ggerganov) and the `whisper.cpp` contributors — MIT, hardware-portable, an existence proof of cross-vendor inference.
- [OpenAI](https://github.com/openai/whisper) — original Whisper STT under MIT.
- The [ONNX Runtime](https://github.com/microsoft/onnxruntime) team — the provider abstraction that makes "one Kokoro on every backend" possible.
