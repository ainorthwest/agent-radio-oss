# agent-radio-oss

**Open-source autonomous AI radio platform.** Hardware-agnostic. Commercial-deployable. Apache 2.0.

> **Status: WIP.** This repo is in active sprint toward v0.1.0-mvp.
>
> Day 1 (current): repo scaffolding + curated source port from `ainorthwest/agent-radio`.
>
> Day 7 target: a stranger with an AMD GPU + Ubuntu can `git clone`, run a setup script, and produce a Haystack News episode + transcript in under 20 minutes.

## What this is

`agent-radio-oss` is the open distribution of [Agent Radio](https://github.com/ainorthwest/agent-radio) — Lightcone Studios' autonomous AI radio platform. The OSS repo contains the hardware-portable, license-clean subset that any operator can clone and deploy commercially.

The bifurcation is intentional:

| Repo | Purpose | Engines | License story |
|---|---|---|---|
| [`agent-radio`](https://github.com/ainorthwest/agent-radio) | Lightcone production stack (AINW Radio) | Kokoro, Chatterbox, Chatterbox-MLX, CSM-MLX, Dia-MLX, Orpheus 3B MLX, Qwen3-TTS (0.6B + 1.7B), MusicGen MLX | Mixed — MusicGen weights are CC-BY-NC, so the production stack is **not commercially deployable as-is** without swapping the music engine |
| [`agent-radio-oss`](https://github.com/ainorthwest/agent-radio-oss) | Open-source distribution | Kokoro ONNX (TTS) | Apache 2.0 throughout v0.1.0-mvp; Stability Community License once Stable Audio Open lands (Day 4) |

## Stack

**Shipped today (Day 1 of MVP sprint):**

- **TTS:** Kokoro ONNX — 82M params, CPU / CUDA / ROCm / CoreML via ONNX Runtime providers
- **Quality:** librosa (spectral) + torchmetrics (DNSMOS / SRMR / PESQ / STOI). WER pillar stubbed pending Day 3.
- **Pipeline:** `curate → render → mix → quality → distribute`
- **CLI:** `radio config`, `radio distribute`, `radio library`, `radio render`, `radio run pipeline`, `radio soundbooth`, `radio stream`
- **Streaming:** AzuraCast HTTP API client (Apache 2.0)
- **Demo show:** Haystack News with three Kokoro hosts (am_michael, af_bella, am_adam). Note: same name as the production show but a different cast — the production version uses Orpheus voices.

**Roadmap (sprint days 2–7):**

- **STT:** whisper.cpp via subprocess (Day 3) — replaces the production stack's `mlx-whisper` dependency
- **Music:** Stable Audio Open via stable-audio-tools (Day 4) — replaces MusicGen for commercial deployability
- **Per-platform setup scripts** (Day 5): `setup-amd.sh`, `setup-cpu.sh`, `setup-mac.sh`, `setup-cuda.sh`
- **Hardware bring-up + smoke tests + dogfood pass** (Days 2, 6, 7)

## Hardware (planned for v0.1.0-mvp)

| Backend | Status |
|---|---|
| AMD ROCm (RDNA4) | 🚧 Day 2 bring-up |
| Apple Silicon (CoreML + Metal) | 🚧 Day 2 bring-up |
| CPU (any Linux/Mac/WSL) | 🚧 Day 2 bring-up |
| NVIDIA CUDA | ⚠️ scripts ship Day 5; validation deferred to long plan |

## License

Code: Apache 2.0. See [`LICENSE`](./LICENSE).

Model weights are governed by their upstream licenses. Per-component license audit in `LICENSES.md` (Day 6 deliverable).

## Contributing

Sprint in progress; contributing guidelines arrive with v0.1.0-mvp release.
