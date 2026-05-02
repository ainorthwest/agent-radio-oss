# agent-radio-oss

[![CI](https://github.com/ainorthwest/agent-radio-oss/actions/workflows/ci.yml/badge.svg)](https://github.com/ainorthwest/agent-radio-oss/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

> **Open-source autonomous AI radio.** Apache 2.0.
> Core code Apache-2.0; optional DSP extras include a GPL-v3 package — full audit in [LICENSES.md](./LICENSES.md).

> **Status:** `v0.1.0` released 2026-05-02 (git tag `v0.1.0-mvp`). **Current Phase:** post-MVP — see [CHANGELOG.md](./CHANGELOG.md).

## Why this exists

**Audio AI is the most underserved frontier in open-source machine learning.** Image and text models have rich, vibrant open ecosystems; speech, music, and broadcast pipelines do not. Production text-to-speech still routes through cloud APIs (ElevenLabs, Google, Azure) or proprietary weights that foreclose commercial use. State-of-the-art open-weights TTS is a fraction of the page count of state-of-the-art open-weights LLMs, and most of what does exist is research-grade — interesting on a benchmark, painful in production. **AI Northwest is correcting this.**

`agent-radio-oss` is one slice of that correction. Apache 2.0 throughout. Engine-agnostic primitives — TTS adapter, render/re-render loop, anomaly detection, 23-metric quality stack, broadcast plumbing — so a stranger on commodity hardware can stand up a real, commercial autonomous radio station with zero cloud dependencies. Kokoro ONNX runs on Apple Silicon, AMD ROCm, NVIDIA CUDA, or pure CPU. `whisper.cpp` transcribes on every backend. The educational point is not "look how good our TTS is" — it's **"look how much open audio AI you can already do, and why this ecosystem deserves more attention than it's getting."**

If you build something on top of this, tell us. The whole reason the OSS distribution exists is to seed an ecosystem; it's hard to seed an ecosystem alone.

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

## How honest is `v0.1.0`? — Excellences scorecard

The OSS station aspires to be excellent at eight things. `v0.1.0` is honest about which it nails:

| # | Excellence | v0.1.0 | What ships | v0.1.1 plan |
|---|---|---|---|---|
| 1 | Wire Desk (story selection) | thin | `gather-news` skill wraps generic curator (RSS / Discourse API) | Story scoring, dedup, multi-source coverage logic |
| 2 | Editorial tracking | thin | `episode_history.py` skeleton, library catalog | Multi-day arc state, beat tracking, freshness gates |
| 3 | Script writing | thin | Curator + script-quality LLM gate | Show-aware `ScriptWriter` Protocol, voice register selection |
| 4 | Render + anomaly + surgical re-render | **strong** | Segment cache (sha256-keyed, atomic) + `editor.py` + `anomaly.py` + the full correction loop | Refinement, additional anomaly classes |
| 5 | Audio engineering | **strong** | 23-metric three-pillar quality (librosa + torchmetrics + whisper.cpp WER), mixer, DSP, named verdicts | Refinement, optional EQ presets per voice |
| 6 | Broadcast management | partial | AzuraCast HTTP client wired (`radio stream …`), R2, Discourse, Podcasting 2.0 RSS | True scheduler, playlist intelligence |
| 7 | Autoresearch | thin | `results.tsv` write-only logging | Closed feedback loop into voice/voice-profile tuning |
| 8 | Agent Experience (AX) | partial | 7 skills (`gather-news`, `render-episode`, `check-quality`, `edit-script`, `publish-episode`, `broadcast`, `run-station`) + [`AGENT_HARNESS.md`](./docs/AGENT_HARNESS.md) + auto-bootstrapping `radio demo` + four setup scripts with strict-smoke fail-fast | Harness-tested reference profiles, structured `--json` everywhere |

Excellences 1–3 and 7 stay thin in `v0.1.0` by the **bifurcation rule** — AI Northwest's editorial taste, Bard's prompt corpus, and the Steward newsroom logic all stay in the proprietary upstream [`agent-radio`](https://github.com/ainorthwest/agent-radio) repo. The OSS station-runner agent is **written fresh**; it does not port AINW IP. Operators bring their own taste and let their agent harness express it.

`v0.1.1+` targets the editorial pillars; `v0.1.0` nails the rendering, audio-science, and AX halves of the loop.

## Two front doors: CLI and Skills

Both surfaces are first-class and call the same `src/*` primitives.

- **`radio` CLI** (Typer, `src/cli/`): operator and debugging surface. `radio demo`, `radio render audition`, `radio run pipeline`, `radio edit script`, `radio publish episode`, etc. Run `uv run radio --help`.
- **Skills bundle** (`skills/`): the agent's operating manual. Each skill is a `SKILL.md` prose contract plus a thin Python wrapper around the CLI — harness-agnostic by design. The v0.1.0 bundle:
  - [`gather-news`](./skills/gather-news/SKILL.md) — start of the loop; generic RSS / Discourse fetch
  - [`render-episode`](./skills/render-episode/SKILL.md) — Kokoro TTS + segment cache + manifest
  - [`check-quality`](./skills/check-quality/SKILL.md) — three-pillar verdict (`ship` / `review` / `reject`)
  - [`edit-script`](./skills/edit-script/SKILL.md) — surgical mutations + targeted re-render
  - [`publish-episode`](./skills/publish-episode/SKILL.md) — Podcasting 2.0, llms.txt, agent payload
  - [`broadcast`](./skills/broadcast/SKILL.md) — R2 + Discourse + RSS + AzuraCast (best-effort branches)
  - [`run-station`](./skills/run-station/SKILL.md) — meta-skill; the autonomous-station playbook

See [`docs/AGENT_HARNESS.md`](./docs/AGENT_HARNESS.md) for how to wire the bundle into Claude Code / Hermes / Gaia / your own harness — including the state contract and the recovery-from-partial-failure procedure. See [`CLAUDE.md`](./CLAUDE.md) for the full architecture and the bifurcation rule, and [`oss-mvp-sprint.md`](./oss-mvp-sprint.md) for the sprint plan and "what we are NOT doing" list.

## License

- **Code:** Apache 2.0 — see [LICENSE](./LICENSE).
- **Bundled model weights:** Apache-2.0 (Kokoro), MIT (`whisper.cpp` + Whisper `base.en`).
- **Per-component audit** including all 108 transitive Python dependencies and the GPL-v3 disposition for the `[tts]` extras: [LICENSES.md](./LICENSES.md).
- **Attribution:** [NOTICE](./NOTICE).

SaaS / internal use is unaffected by the GPL-v3 caveat (no AGPL network clause). Distribution of an installed binary inherits GPL-v3 §4 source-availability obligations from the `pedalboard` and `phonemizer-fork` packages — `v0.1.1` queues both for replacement with permissive equivalents. Read [LICENSES.md](./LICENSES.md#the-gpl-v3-question) before redistributing a build.

## Where this came from

This repo is the open distribution of [`agent-radio`](https://github.com/ainorthwest/agent-radio) — Lightcone Studios' production AINW Radio stack. The split is by deployability: AINW's specific feed list, MLX engines, Steward / Bard agents, and CC-BY-NC music weights stay proprietary; the generic primitives (RSS, web fetch, Discourse API, render pipeline, quality stack, broadcast client) ship here under Apache 2.0. The full sprint plan, the bifurcation rule, and the daily log are in [`oss-mvp-sprint.md`](./oss-mvp-sprint.md).

## Contributing

`v0.1.0-mvp` shipped 2026-05-02. Issues and PRs welcome.

If you build something on top of this — a different show, a different harness, a different engine, anything — open an issue and tell us. The OSS distribution exists to seed an ecosystem that the audio AI space deserves.

## Acknowledgements

- [hexgrad](https://github.com/hexgrad) and the Kokoro contributors — Apache-2.0 TTS weights that make this whole thing work.
- [Georgi Gerganov](https://github.com/ggerganov) and the `whisper.cpp` contributors — MIT, hardware-portable, an existence proof of cross-vendor inference.
- [OpenAI](https://github.com/openai/whisper) — original Whisper STT under MIT.
- The [ONNX Runtime](https://github.com/microsoft/onnxruntime) team — the provider abstraction that makes "one Kokoro on every backend" possible.
