# NVIDIA CUDA

> **⚠ Unverified in v0.1.0.** No NVIDIA hardware was available during the Day 5 sprint. This doc is structured for the first NVIDIA contributor to fill in measurements. The `scripts/setup-cuda.sh` script ships blind — best-effort scaffolding modeled on `setup-amd.sh`. If you're the first contributor on this path, please open an issue with your hardware + driver + outcome — your bug report becomes the v0.1.1 validation foundation.

`agent-radio-oss` runs on NVIDIA GPUs via ONNX Runtime's `CUDAExecutionProvider`. CUDA is NVIDIA's compute stack; ONNX Runtime exposes it via the `onnxruntime-gpu` PyPI wheel, which ships builds for CUDA 12.

The same Kokoro ONNX graph that runs on AMD ROCm or Apple CoreML or CPU also runs unmodified on an NVIDIA GPU — that's the educational point of the OSS repo.

## One-shot install

```bash
bash scripts/setup-cuda.sh
```

The script verifies `nvidia-smi` is on PATH, runs `uv sync`, installs `onnxruntime-gpu`, builds `whisper.cpp` with `-DGGML_CUDA=ON`, downloads models, and writes `.env.suggested` with `KOKORO_PROVIDER=CUDAExecutionProvider`. It prints an UNTESTED banner at start and end — that's intentional, not a bug.

## Verified host (TBD)

This table will be filled in by the first NVIDIA contributor. Please add:

| | |
|---|---|
| Hardware | _e.g._ NVIDIA RTX 4090 (24GB, Ada Lovelace) |
| OS | _e.g._ Ubuntu 24.04.4 LTS |
| Driver | _e.g._ 550.xx |
| CUDA | _e.g._ 12.4 |
| Python | _e.g._ 3.12 |
| `onnxruntime-gpu` | _e.g._ 1.20.x |
| `kokoro-onnx` | _e.g._ 0.5.0 |

## Setup (manual reproduction)

If `setup-cuda.sh` doesn't work for you, the manual steps are:

```bash
# 1. NVIDIA driver + CUDA toolkit
# https://developer.nvidia.com/cuda-downloads

# 2. Verify the GPU is detected
nvidia-smi

# 3. Python deps
uv sync --extra tts --extra quality

# 4. ONNX Runtime CUDA wheel
uv pip uninstall onnxruntime onnxruntime-gpu  # may fail on first run; that's fine
uv pip install --no-deps onnxruntime-gpu

# 5. Verify the CUDA provider is exposed
uv run python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# Expected: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

# 6. whisper.cpp with CUDA
git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git
cmake -B whisper.cpp/build-cuda -S whisper.cpp -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build whisper.cpp/build-cuda --config Release -j 8

# 7. Models
bash scripts/download-models.sh

# 8. Env vars
export KOKORO_PROVIDER=CUDAExecutionProvider
export RADIO_WHISPER_BIN="$(pwd)/whisper.cpp/build-cuda/bin/whisper-cli"
export RADIO_WHISPER_MODEL="$(pwd)/models/ggml-base.en.bin"
```

## Running an audition

```bash
KOKORO_PROVIDER=CUDAExecutionProvider \
  uv run radio render audition \
  library/programs/haystack-news/episodes/sample/script.json \
  --voice voices/kokoro-michael.yaml
```

The kokoro stderr should show `[kokoro] loaded with provider=CUDAExecutionProvider` (NOT a CPU fallback — `[kokoro] WARNING:` means the GPU path didn't engage). Verify GPU engagement via `nvidia-smi -l 1` in another terminal during the render.

## Measurements (TBD)

Same input as the AMD and Apple Silicon docs: `library/programs/haystack-news/episodes/sample/script.json`, 5 segments, ~54s, `kokoro-michael` voice. The first NVIDIA contributor should fill in:

| Provider | Render wall-clock | DNSMOS OVR | DNSMOS SIG | SRMR | Repetition | GPU peak | Notes |
|---|---|---|---|---|---|---|---|
| `CPUExecutionProvider` | ? | ? | ? | ? | ? | 0% | Baseline |
| `CUDAExecutionProvider` | ? | ? | ? | ? | ? | ? | First-CUDA bring-up |
| `TensorrtExecutionProvider` | ? | ? | ? | ? | ? | ? | Optional, if the wheel includes it |

## Quirks (TBD)

To be filled in by the first NVIDIA contributor. Common axes to investigate:
- Driver version compatibility with `onnxruntime-gpu`'s CUDA build
- Multi-GPU host disambiguation (`CUDA_VISIBLE_DEVICES`)
- TensorRT provider enable/disable
- whisper.cpp `-DCMAKE_CUDA_ARCHITECTURES=...` for non-default GPU generations

## See also

- [`amd-rocm.md`](./amd-rocm.md) — AMD ROCm path (validated)
- [`apple-silicon.md`](./apple-silicon.md) — Apple Silicon path (validated)
- [`cpu.md`](./cpu.md) — universal baseline (validated)
- [`README.md`](./README.md) — comparison matrix
