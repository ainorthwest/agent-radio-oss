#!/usr/bin/env bash
# scripts/setup-cuda.sh — NVIDIA CUDA setup for agent-radio-oss.
#
# ⚠ UNTESTED IN v0.1.0 ⚠
#
# This script ships blind. The Day 5 sprint had no NVIDIA hardware to
# validate against, so the contents are best-effort scaffolding modeled
# on setup-amd.sh (which IS validated). Aaron has NVIDIA hardware coming
# online soon — when it does, real-mode validation flips on and the
# UNTESTED banner gets removed.
#
# What this script attempts:
#   1. Verify nvidia-smi is on PATH and reports a CUDA-capable GPU
#   2. uv sync --extra tts --extra quality
#   3. Install onnxruntime-gpu (ships CUDA + TensorRT providers via
#      the ONNX Runtime PyPI build)
#   4. Build whisper.cpp with -DGGML_CUDA=ON
#   5. Write .env.suggested with KOKORO_PROVIDER=CUDAExecutionProvider
#
# If you run this and hit issues, please open an issue with the output —
# you're the first contributor on this path and your bug report becomes
# the v0.1.1 validation foundation.

set -euo pipefail

_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$_SELF_DIR/lib/common.sh"

_print_untested_banner() {
  local label="$1" # "start" or "end"
  printf '\n'
  printf '⚠ ════════════════════════════════════════════════════════════ ⚠\n'
  printf '⚠  setup-cuda.sh — UNTESTED IN v0.1.0  (%s)\n' "$label"
  printf '⚠\n'
  printf '⚠  This script ships blind. No NVIDIA hardware was available\n'
  printf '⚠  during the Day 5 sprint. Contents are best-effort scaffolding\n'
  printf '⚠  modeled on setup-amd.sh. If you hit issues, please open an\n'
  printf '⚠  issue at https://github.com/ainorthwest/agent-radio-oss/issues\n'
  printf '⚠  — your bug report becomes the v0.1.1 validation foundation.\n'
  printf '⚠ ════════════════════════════════════════════════════════════ ⚠\n'
  printf '\n'
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --dry-run               Print steps without invoking them
  --skip-models           Don't run download-models.sh
  --skip-whisper-build    Don't build whisper.cpp
  --skip-self-test        Don't run the smoke check at the end
  -h, --help              Show this help

⚠ This script is UNTESTED in v0.1.0 — see banner above.
EOF
}

SKIP_MODELS=0
SKIP_WHISPER_BUILD=0
SKIP_SELF_TEST=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run)
      export RADIO_DRY_RUN=1
      shift
      ;;
    --skip-models)
      SKIP_MODELS=1
      shift
      ;;
    --skip-whisper-build)
      SKIP_WHISPER_BUILD=1
      shift
      ;;
    --skip-self-test)
      SKIP_SELF_TEST=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      radio::log_err "unknown option: $1"
      usage >&2
      exit 2
      ;;
  esac
done

_print_untested_banner "start"

radio::log_info "agent-radio-oss CUDA setup"

PLATFORM="$(radio::detect_platform)"
radio::log_info "platform: $PLATFORM"

case "$PLATFORM" in
  linux-cuda) ;;
  linux-cpu)
    radio::status_fail "NVIDIA GPU not detected — nvidia-smi not on PATH" \
      --remedy "install NVIDIA drivers + CUDA: https://developer.nvidia.com/cuda-downloads. If your GPU is AMD, use 'bash scripts/setup-amd.sh' instead." || exit 1
    ;;
  linux-amd)
    radio::status_fail "this is the NVIDIA setup script — detected AMD hardware (rocminfo present)" \
      --remedy "use 'bash scripts/setup-amd.sh' instead" || exit 1
    ;;
  *)
    radio::status_fail "setup-cuda.sh requires Linux + NVIDIA — detected: $PLATFORM" \
      --remedy "macOS: 'bash scripts/setup-mac.sh'; Linux without GPU: 'bash scripts/setup-cpu.sh'" || exit 1
    ;;
esac

# 1. Safety guards.
radio::guard_not_root || exit 1
radio::guard_no_venv || exit 1

# 2. Pre-checks.
if ! radio::python_version_ok; then
  radio::status_fail "Python 3.11+ not found" \
    --remedy "Ubuntu: 'apt install python3.12 python3.12-venv'" || exit 1
fi
radio::status_ok "python3 (3.11+)"

radio::require_cmd ffmpeg \
  --remedy "Ubuntu: 'sudo apt install ffmpeg'" || exit 1
radio::status_ok "ffmpeg"

radio::require_cmd cmake \
  --remedy "Ubuntu: 'sudo apt install cmake'" || exit 1
radio::status_ok "cmake"

radio::require_cmd git \
  --remedy "Ubuntu: 'sudo apt install git'" || exit 1
radio::status_ok "git"

radio::require_cmd uv \
  --remedy "install uv: 'curl -LsSf https://astral.sh/uv/install.sh | sh'" || exit 1
radio::status_ok "uv"

radio::require_cmd nvidia-smi \
  --remedy "install NVIDIA drivers and CUDA: https://developer.nvidia.com/cuda-downloads" || exit 1
radio::status_ok "nvidia-smi"

# 3. Python deps.
radio::log_info "syncing Python deps (tts + quality extras)"
radio::dry_run_or_exec uv sync --extra tts --extra quality
radio::status_ok "uv sync complete"

# 4. onnxruntime-gpu wheel — provides CUDA + TensorRT providers.
# UNTESTED: the PyPI `onnxruntime-gpu` ships CUDA 12 builds for Linux,
# but we haven't verified compatibility with our `kokoro-onnx` pin.
radio::log_info "installing onnxruntime-gpu (CUDA wheel)"
radio::dry_run_or_exec uv pip uninstall onnxruntime onnxruntime-gpu || true
radio::dry_run_or_exec uv pip install --no-deps onnxruntime-gpu
radio::status_ok "onnxruntime-gpu installed"

# 5. whisper.cpp build (CUDA).
if [ "$SKIP_WHISPER_BUILD" != "1" ]; then
  if [ ! -d "whisper.cpp" ]; then
    radio::log_info "cloning whisper.cpp"
    radio::dry_run_or_exec git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git
  else
    radio::log_info "whisper.cpp already cloned — skipping git clone"
  fi
  if [ -f "whisper.cpp/build-cuda/bin/whisper-cli" ]; then
    radio::status_ok "whisper.cpp CUDA build present — skipping cmake"
  else
    radio::log_info "building whisper.cpp (CUDA)"
    radio::dry_run_or_exec cmake -B whisper.cpp/build-cuda -S whisper.cpp \
      -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
    radio::dry_run_or_exec cmake --build whisper.cpp/build-cuda --config Release -j 8
    radio::status_ok "whisper.cpp built (CUDA)"
  fi
else
  radio::status_warn "skipping whisper.cpp build (--skip-whisper-build)"
fi

# 6. Models.
if [ "$SKIP_MODELS" != "1" ]; then
  radio::log_info "downloading models"
  if [ "${RADIO_DRY_RUN:-}" = "1" ]; then
    radio::dry_run_or_exec bash "$_SELF_DIR/download-models.sh" --dry-run
  else
    bash "$_SELF_DIR/download-models.sh"
  fi
  radio::status_ok "models ready"
else
  radio::status_warn "skipping model download (--skip-models)"
fi

# 7. .env.suggested
ENV_FILE=".env.suggested"
if [ "${RADIO_DRY_RUN:-}" = "1" ]; then
  radio::log_info "[dry-run] would write $ENV_FILE with KOKORO_PROVIDER=CUDAExecutionProvider"
else
  cat >"$ENV_FILE" <<'ENVEOF'
# Suggested env vars for agent-radio-oss on NVIDIA CUDA.
# Source this file or copy into .env to use.
#
# UNTESTED IN v0.1.0 — if you're the first contributor on this path,
# please open an issue with your hardware + driver versions + outcome.

# ONNX Runtime provider for Kokoro TTS.
export KOKORO_PROVIDER=CUDAExecutionProvider

# whisper.cpp paths (Pillar 3 quality / transcripts; CUDA build).
export RADIO_WHISPER_BIN="$(pwd)/whisper.cpp/build-cuda/bin/whisper-cli"
export RADIO_WHISPER_MODEL="$(pwd)/models/ggml-base.en.bin"
ENVEOF
  radio::status_ok "$ENV_FILE written"
fi

# 8. Self-test.
if [ "$SKIP_SELF_TEST" != "1" ] && [ "${RADIO_DRY_RUN:-}" != "1" ]; then
  radio::log_info "running smoke test (--quick)"
  if bash "$_SELF_DIR/oss-smoke.sh" --quick; then
    radio::status_ok "smoke test passed"
  else
    radio::status_warn "smoke test failed — install completed but verification did not"
  fi
elif [ "$SKIP_SELF_TEST" = "1" ]; then
  radio::status_warn "skipping self-test (--skip-self-test)"
fi

radio::final_status "NVIDIA CUDA (UNTESTED)"
radio::log_info "next: 'source .env.suggested && uv run radio demo'"
_print_untested_banner "end"
