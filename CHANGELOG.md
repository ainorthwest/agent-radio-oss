# Changelog

All notable changes to `agent-radio-oss` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-02 (`v0.1.0-mvp`)

The MVP release. A stranger on commodity hardware can clone this repo, run one
setup script, and produce a real autonomous AI radio episode — gather → render →
quality-check → broadcast — with transcripts, anomaly detection, and per-segment
surgical re-render. Apache 2.0 throughout.

### Added

**Pipeline & engines.**
- Cross-platform Kokoro ONNX TTS adapter with `KOKORO_PROVIDER` provider
  selection (CPU / CUDA / ROCm / CoreML / MIGraphX). The renderer reads back the
  *actual* loaded provider from the ONNX session and warns on silent CPU
  fallback.
- `whisper.cpp` STT pillar (`src/stt.py`) — five primitives: `transcribe`,
  `transcribe_with_timing` + SRT export, pure-Python `wer`/`cer`,
  `round_trip_score`, `transcribe_for_corpus`. Hardware backend baked at compile
  time (Metal / HIP / CUDA / vanilla CPU).
- 23-metric three-pillar quality stack (`src/quality.py`) — librosa spectral,
  torchmetrics (DNSMOS / SRMR / PESQ / STOI), and round-trip WER. Named
  verdicts (`ship` / `review` / `reject`) with `verdict_reason` field.
- Segment cache (`src/segment_cache.py`) — content-addressed WAV cache keyed on
  `sha256(text + speaker + register + voice_profile + engine)`, atomic
  `.tmp`+rename writes. Re-render one segment without re-rendering the episode.
- Script editor (`src/editor.py`) — pure-function ops (`delete_segment`,
  `replace_text`, `reorder_segments`, `insert_segment`, `change_voice`)
  returning `(new_script, ScriptDiff)`. Drives the agent's correction loop.
- Anomaly detector (`src/anomaly.py`) — silence-ratio, WER outlier, duration
  anomaly. Each anomaly carries a `regenerate` / `replace_text` / `manual_review`
  action suggestion so the agent acts without interpreting raw scores.
- Publisher (`src/publisher.py`) — deterministic, byte-stable derivative content
  (`episode.md` with YAML frontmatter, `chapters.json` per Podcasting 2.0,
  `episode.txt` agent payload, `episode.jsonld` schema.org). Per-show
  `llms.txt` indexes per [llmstxt.org](https://llmstxt.org).
- Podcasting 2.0 RSS (`src/podcast.py`) — `<podcast:locked>`, `<podcast:person>`,
  `<podcast:transcript>`, `<podcast:chapters>`.
- AzuraCast HTTP API client (`src/stream.py`) — Apache 2.0, OSS-clean.

**Operator surface.**
- `radio` Typer CLI (`src/cli/`): `config`, `render`, `run`, `library`,
  `distribute`, `stream`, `soundbooth`, `edit`, `publish`, `demo`. All commands
  expose `--help`, `--json`, `--dry-run` where applicable.
- `radio demo` — one-command end-to-end run with auto-bootstrap config
  (copies `config/radio.example.yaml` → `config/radio.yaml` on first run) and
  canned-sample fallback when no `OPENROUTER_API_KEY` is set. Writes a
  `DEMO_README.md` next to artifacts explaining each output file.
- Four setup scripts: `setup-mac.sh` (Apple Silicon, CoreML), `setup-amd.sh`
  (Linux + ROCm, CPU-default with `--enable-migraphx` opt-in), `setup-cpu.sh`
  (universal CPU baseline with auto-installed ffmpeg on Linux),
  `setup-cuda.sh` (UNTESTED banner; ships blind for first NVIDIA contributor).
- `scripts/oss-smoke.sh` — three modes (`--quick`, `--audition`, `--full`) with
  `RADIO_SMOKE_TIMEOUT`, silent-CPU-fallback detection, quality-verdict gate.
- `scripts/download-models.sh` — SHA-pinned, idempotent Kokoro + Whisper
  downloader.
- `RADIO_STRICT_SMOKE` env switch — fail-fast on smoke-test failure during
  setup; default off preserves operator-friendly warn-and-continue.

**Agent surface.**
- Skills bundle (`skills/`) — seven skills with prose SKILL.md contracts and
  thin Python wrappers around `radio` CLI: `edit-script`, `publish-episode`,
  `gather-news`, `render-episode`, `check-quality`, `broadcast`, `run-station`.
  Each skill names what stays upstream so operators understand the bifurcation.
- `docs/AGENT_HARNESS.md` — how Hermes / Claude Code / Gaia / custom harnesses
  load the skills, where state lives, recovery from partial failures.

**Documentation.**
- Hardware bring-up docs: `docs/hardware/{apple-silicon,amd-rocm,cpu,nvidia-cuda}.md`.
- AMD investigation: `docs/investigations/kokoro-amd-rocm.md` (564 lines,
  command-by-command evidence for the AMDMIGraphX#4618 null-pointer).
- License audit: `LICENSES.md` with the GPL-v3 disposition for `[tts]` extras.
- Honest excellences scorecard in `README.md`.

### Hardware verified

- ✓ **Apple Silicon (M3 Pro):** CPU + CoreML provider, end-to-end Haystack News
  demo to verdict=`ship`, quality 0.72.
- ✓ **AMD (Ryzen 7 9700X + RX 9070, gfx1201, ROCm 7.2.1):** CPU path,
  end-to-end demo to verdict=`ship`, quality 0.7165 (within 0.005 of Mac).
- ✓ **Linux CPU (Ubuntu 22.04+):** universal baseline.
- ✓ **Docker `ubuntu:24.04` (CPU):** ffmpeg auto-installed by `setup-cpu.sh`,
  smoke test green from a clean container.
- ⚠ **NVIDIA CUDA:** untested in v0.1.0; `setup-cuda.sh` ships blind, identical
  shape to `setup-amd.sh`, gated behind a clear UNTESTED banner.

### Known thin / deferred to v0.1.1

The OSS station aspires to be excellent at eight things. v0.1.0 is honest about
which it nails: render+anomaly+re-render and audio engineering are **strong**;
broadcast management and Agent Experience are **partial**; **Wire Desk story
selection, editorial tracking, script writing, and autoresearch are thin** by
the bifurcation rule — AINW's editorial taste, Bard's prompt corpus, and the
Steward newsroom logic stay in the proprietary upstream
[`agent-radio`](https://github.com/ainorthwest/agent-radio).

- **Stable Audio Open music generation** — deferred to v0.1.1 ([GH#9](https://github.com/ainorthwest/agent-radio-oss/issues/9)).
  The mixer overlays pre-rendered music assets in v0.1.0; `src/music.py` is an
  honest stub raising `NotImplementedError` with a remediation message.
- **AzuraCast scheduler** — HTTP client wired, no scheduler.
- **Autoresearch feedback loop** — `results.tsv` write-only logging exists; no
  voice-tuning closed loop.
- **AMD GPU rendering on `gfx1201`** — gated by upstream
  [AMDMIGraphX#4618](https://github.com/ROCm/AMDMIGraphX/issues/4618). CPU on
  AMD is recommended for v0.1.0; will revisit when upstream lands a fix.
- **GPL-v3 transitive deps** (`pedalboard`, `phonemizer-fork` via the `[tts]`
  extras) — SaaS / internal use unaffected; binary distribution inherits
  GPL-v3 §4 obligations. Both queued for replacement in v0.1.1. See
  [`LICENSES.md`](./LICENSES.md#the-gpl-v3-question).

### Bifurcation rule

The OSS test is **commercial deployability and AINW-specificity**. Generic
primitives ship here; AINW-specific content (curated feed lists, Bard's prompt
corpus, Steward lore, MLX engines, voice cloning, CC-BY-NC weights, AINW
editorial taste) stays in the proprietary upstream `agent-radio` repo. The
station-runner agent (Hermes-driven reference profile) is **written fresh** in
this repo, not ported.

[0.1.0]: https://github.com/ainorthwest/agent-radio-oss/releases/tag/v0.1.0-mvp
