# agent-radio-oss

**Open-source autonomous AI radio platform.** Hardware-agnostic. Commercial-deployable. Apache 2.0.

> **Status: WIP.** This repo is in active sprint toward v0.1.0-mvp.
>
> Day 2: Kokoro cross-hardware bring-up. Apple CoreML verified end-to-end; cross-host CPU parity verified (Mac M3 Pro + Linux Ryzen 7); AMD ROCm setup plumbed all the way through provider engagement on RX 9070 / `gfx1201`, but MIGraphX graph compilation on Kokoro hits a >15-min perf wall — v0.1.0 recommends CPU on AMD hardware while v0.1.1 tunes the compile/cache. Engine fix shipped that translates `KOKORO_PROVIDER` → `ONNX_PROVIDER` and verifies actual loaded provider against requested.
>
> Day 7 target: a stranger with an AMD GPU + Ubuntu can `git clone`, run a setup script, and produce a Haystack News episode + transcript in under 20 minutes.

## What this is

`agent-radio-oss` is the open distribution of [Agent Radio](https://github.com/ainorthwest/agent-radio) — Lightcone Studios' autonomous AI radio platform. The OSS repo contains the hardware-portable, license-clean subset that any operator can clone and deploy commercially.

The bifurcation is intentional:

| Repo | Purpose | Engines | License story |
|---|---|---|---|
| [`agent-radio`](https://github.com/ainorthwest/agent-radio) | Lightcone production stack (AINW Radio) | Kokoro, Chatterbox, Chatterbox-MLX, CSM-MLX, Dia-MLX, Orpheus 3B MLX, Qwen3-TTS (0.6B + 1.7B), MusicGen MLX | Mixed — MusicGen weights are CC-BY-NC, so the production stack is **not commercially deployable as-is** without swapping the music engine |
| [`agent-radio-oss`](https://github.com/ainorthwest/agent-radio-oss) | Open-source distribution | Kokoro ONNX (TTS) | Apache 2.0 throughout v0.1.0-mvp; Stability Community License once Stable Audio Open lands (planned for v0.1.1, see [#9](https://github.com/ainorthwest/agent-radio-oss/issues/9)) |

## Quick start

```bash
# Install the MVP extras
uv sync --extra tts --extra quality --extra dev

# Run the demo — no curator key required
uv run radio demo
```

The demo runs the full pipeline (render → quality → publisher) against the canned Haystack News sample script and writes a complete output dir under `library/programs/haystack-news/episodes/{timestamp}/` with `episode.mp3`, `transcript.txt`, `transcript.srt`, `quality.json`, `episode.md`, `chapters.json`, and a `DEMO_README.md` that walks you through what each artifact means.

If you have an OpenRouter API key (`OPENROUTER_API_KEY` env var), the demo runs the live curator against your configured Discourse instance instead of the canned script.

## Stack

**Shipped today (Day 1 of MVP sprint):**

- **TTS:** Kokoro ONNX — 82M params, CPU / CUDA / ROCm / CoreML via ONNX Runtime providers
- **Quality:** librosa (spectral) + torchmetrics (DNSMOS / SRMR / PESQ / STOI). WER pillar stubbed pending Day 3.
- **Pipeline:** `curate → render → mix → quality → distribute`
- **CLI:** `radio demo`, `radio config`, `radio distribute`, `radio edit`, `radio library`, `radio publish`, `radio render`, `radio run pipeline`, `radio soundbooth`, `radio stream`
- **Streaming:** AzuraCast HTTP API client (Apache 2.0)
- **Demo show:** Haystack News with three Kokoro hosts (am_michael, af_bella, am_adam). Note: same name as the production show but a different cast — the production version uses Orpheus voices.

**Roadmap (sprint days 2–7):**

- **STT:** whisper.cpp via subprocess (Day 3) — replaces the production stack's `mlx-whisper` dependency
- **Music:** Stable Audio Open via stable-audio-tools (Day 4) — replaces MusicGen for commercial deployability
- **Per-platform setup scripts** (Day 5): `setup-amd.sh`, `setup-cpu.sh`, `setup-mac.sh`, `setup-cuda.sh`
- **Hardware bring-up + smoke tests + dogfood pass** (Days 2, 6, 7)

## Hardware

Verified by Day 2 of the MVP sprint. See [`docs/hardware/`](./docs/hardware/) for per-platform install steps, observed quirks, and parity measurements.

| Backend | Provider string | Status | Doc |
|---|---|---|---|
| Apple Silicon (M-series, CoreML) | `CoreMLExecutionProvider` | ✓ Verified on M3 Pro / macOS 26.3 | [`docs/hardware/apple-silicon.md`](./docs/hardware/apple-silicon.md) |
| CPU (any Linux/Mac/WSL) | `CPUExecutionProvider` | ✓ Verified on Shiro (M3 Pro / macOS 26.3) and Hinoki (Ryzen 7 9700X / Ubuntu 24.04); cross-host parity Δ < 0.01 | [`docs/hardware/cpu.md`](./docs/hardware/cpu.md) |
| AMD ROCm (RDNA4 RX 9070 / `gfx1201`) | `MIGraphXExecutionProvider` | ⚠ Plumbing verified (ROCm 7.2.1 + MIGraphX 2.15.0 + `onnxruntime-migraphx` 1.23.2 wheel), provider engages, model loads to VRAM. Graph compilation hits a perf wall on Kokoro's encoder (>15 min). For v0.1.0 use `KOKORO_PROVIDER=CPUExecutionProvider` on AMD hardware. v0.1.1 will tune compile / cache. | [`docs/hardware/amd-rocm.md`](./docs/hardware/amd-rocm.md) |
| NVIDIA CUDA | `CUDAExecutionProvider` | ⚠️ Scripts ship Day 5; validation deferred to long plan | _not yet_ |

## License

Code: Apache 2.0. See [`LICENSE`](./LICENSE).

Model weights are governed by their upstream licenses. Per-component license audit in `LICENSES.md` (Day 6 deliverable).

## Contributing

Sprint in progress; contributing guidelines arrive with v0.1.0-mvp release.
