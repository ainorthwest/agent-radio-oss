#!/usr/bin/env bash
# scripts/setup-mac.sh — Apple Silicon setup for agent-radio-oss.
#
# Apple Silicon (M-series) only — refuses on Intel Mac. Uses CoreML for
# Kokoro inference and Metal for whisper.cpp transcription. The default
# `onnxruntime` PyPI wheel includes CoreMLExecutionProvider on darwin-arm64,
# so no wheel surgery (unlike AMD).
#
# Pre-checks: macOS, arm64, brew, ffmpeg, cmake, git, uv. Installs missing
# brew packages on demand.
#
# Honors --dry-run, --skip-models, --skip-whisper-build, --skip-self-test.

set -euo pipefail

_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$_SELF_DIR/lib/common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --dry-run               Print steps without invoking them
  --skip-models           Don't run download-models.sh
  --skip-whisper-build    Don't build whisper.cpp
  --skip-self-test        Don't run the smoke check at the end
  -h, --help              Show this help
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

radio::log_info "agent-radio-oss Apple Silicon setup"

PLATFORM="$(radio::detect_platform)"
radio::log_info "platform: $PLATFORM"

# Refuse on anything but Apple Silicon.
case "$PLATFORM" in
  mac-arm) ;;
  mac-intel)
    radio::status_fail "setup-mac.sh is for Apple Silicon (M-series) only — detected Intel Mac" \
      --remedy "Intel Macs should use 'bash scripts/setup-cpu.sh' instead" || exit 1
    ;;
  *)
    radio::status_fail "setup-mac.sh requires macOS — detected: $PLATFORM" \
      --remedy "Linux: 'bash scripts/setup-cpu.sh' or 'bash scripts/setup-amd.sh'" || exit 1
    ;;
esac

# 1. Safety guards.
radio::guard_not_root || exit 1
radio::guard_no_venv || exit 1

# 2. Pre-checks.
if ! radio::python_version_ok; then
  radio::status_fail "Python 3.11+ not found" \
    --remedy "brew install python@3.12" || exit 1
fi
radio::status_ok "python3 (3.11+)"

# brew is required so we can install ffmpeg / cmake on demand.
radio::require_cmd brew \
  --remedy "install Homebrew: https://brew.sh" || exit 1
radio::status_ok "brew"

if ! command -v ffmpeg >/dev/null 2>&1; then
  radio::log_info "installing ffmpeg via brew"
  radio::dry_run_or_exec brew install ffmpeg
fi
radio::status_ok "ffmpeg"

if ! command -v cmake >/dev/null 2>&1; then
  radio::log_info "installing cmake via brew"
  radio::dry_run_or_exec brew install cmake
fi
radio::status_ok "cmake"

radio::require_cmd git \
  --remedy "install Xcode Command Line Tools: 'xcode-select --install'" || exit 1
radio::status_ok "git"

radio::require_cmd uv \
  --remedy "install uv: 'curl -LsSf https://astral.sh/uv/install.sh | sh'" || exit 1
radio::status_ok "uv"

# 3. Python deps. CoreML is in the default onnxruntime wheel on darwin-arm64.
radio::log_info "syncing Python deps (tts + quality extras)"
radio::dry_run_or_exec uv sync --extra tts --extra quality
radio::status_ok "uv sync complete"

# 4. whisper.cpp build (Metal).
if [ "$SKIP_WHISPER_BUILD" != "1" ]; then
  if [ ! -d "whisper.cpp" ]; then
    radio::log_info "cloning whisper.cpp"
    radio::dry_run_or_exec git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git
  else
    radio::log_info "whisper.cpp already cloned — skipping git clone"
  fi
  if [ -f "whisper.cpp/build/bin/whisper-cli" ]; then
    radio::status_ok "whisper.cpp already built — skipping cmake"
  else
    radio::log_info "building whisper.cpp (Metal)"
    radio::dry_run_or_exec cmake -B whisper.cpp/build -S whisper.cpp \
      -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
    radio::dry_run_or_exec cmake --build whisper.cpp/build --config Release -j 8
    radio::status_ok "whisper.cpp built (Metal)"
  fi
else
  radio::status_warn "skipping whisper.cpp build (--skip-whisper-build)"
fi

# 5. Models.
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

# 6. .env.suggested
ENV_FILE=".env.suggested"
if [ "${RADIO_DRY_RUN:-}" = "1" ]; then
  radio::log_info "[dry-run] would write $ENV_FILE with KOKORO_PROVIDER=CoreMLExecutionProvider"
else
  cat >"$ENV_FILE" <<'ENVEOF'
# Suggested env vars for agent-radio-oss on Apple Silicon.
# Source this file or copy into .env to use.

# ONNX Runtime provider for Kokoro TTS. CoreML is the Apple Silicon GPU/ANE path.
export KOKORO_PROVIDER=CoreMLExecutionProvider

# whisper.cpp paths (Pillar 3 quality / transcripts; built with Metal).
export RADIO_WHISPER_BIN="$(pwd)/whisper.cpp/build/bin/whisper-cli"
export RADIO_WHISPER_MODEL="$(pwd)/models/ggml-base.en.bin"
ENVEOF
  radio::status_ok "$ENV_FILE written"
fi

# 7. Self-test.
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

radio::final_status "Apple Silicon (CoreML + Metal)"
radio::log_info "next: 'source .env.suggested && uv run radio demo'"
