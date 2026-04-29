# agent-radio-oss

**Open-source autonomous AI radio platform.** Hardware-agnostic. Commercial-deployable. Apache 2.0.

> **Status: WIP.** This repo is in active sprint (see [`oss-mvp-sprint.md`](./oss-mvp-sprint.md) for the 7-day plan to v0.1.0-mvp).
>
> Day 1 (current): repo scaffolding + curated source port from `ainorthwest/agent-radio`.
>
> Day 7 target: a stranger with an AMD GPU + Ubuntu can `git clone`, run a setup script, and produce a Haystack News episode + transcript in under 20 minutes.

## What this is

`agent-radio-oss` is the open distribution of [Agent Radio](https://github.com/ainorthwest/agent-radio) — Lightcone Studios' autonomous AI radio platform. The OSS repo contains the hardware-portable, license-clean subset that any operator can clone and deploy commercially.

The bifurcation is intentional:

| Repo | Purpose | Engines | License story |
|---|---|---|---|
| [`agent-radio`](https://github.com/ainorthwest/agent-radio) | Lightcone production stack (AINW Radio) | Orpheus / Dia / CSM / Qwen3 / Chatterbox-MLX, MusicGen MLX | Mixed (CC-BY-NC weights present) |
| [`agent-radio-oss`](https://github.com/ainorthwest/agent-radio-oss) | Open-source distribution | Kokoro ONNX (TTS), whisper.cpp (STT), Stable Audio Open (music) | Fully Apache 2.0 / MIT / Stability Community |

## Stack (locked for v0.1.0-mvp)

- **TTS:** Kokoro ONNX — 82M params, CPU / CUDA / ROCm / CoreML via ONNX Runtime providers
- **STT:** whisper.cpp — most hardware-portable Whisper, MIT (Day 3)
- **Music:** Stable Audio Open — text-to-music, commercial-permitted (Day 4)
- **Pipeline:** `curate → render → mix → quality → distribute`, librosa + torchmetrics quality stack
- **CLI:** `radio render`, `radio run pipeline`, `radio library`, `radio distribute`, `radio config`

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

Sprint in progress; contributing guidelines arrive with v0.1.0-mvp release. The working spec lives in [`oss-mvp-sprint.md`](./oss-mvp-sprint.md).
