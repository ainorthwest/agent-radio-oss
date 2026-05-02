#!/usr/bin/env bash
# scripts/setup-cpu.sh — universal CPU baseline setup for agent-radio-oss.
#
# Works on Linux, macOS (intentionally; the Mac path is just CPU + a default
# whisper.cpp build), and WSL. The CPU path is also the recommended v0.1.0
# AMD path because of the MIGraphX runtime null-pointer (see
# docs/investigations/kokoro-amd-rocm.md and AMDMIGraphX#4618).
#
# What it does, in order:
#   1. Refuse to run as root or with an active virtualenv
#   2. Verify Python 3.11+, ffmpeg, cmake, git on PATH
#   3. uv sync --extra tts --extra quality
#   4. (Optional) build whisper.cpp from source — vanilla CPU build
#   5. (Optional) bash scripts/download-models.sh — Kokoro + Whisper base.en
#   6. Write .env.suggested with KOKORO_PROVIDER=CPUExecutionProvider
#   7. (Optional) bash scripts/oss-smoke.sh --quick as self-test
#
# Honors --dry-run, --skip-models, --skip-whisper-build, --skip-self-test.
# Idempotent: rerunning is safe (sha-pinned models, build dir reuse).

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

radio::log_info "agent-radio-oss CPU setup"
radio::log_info "platform: $(radio::detect_platform)"

# 1. Safety guards.
radio::guard_not_root || exit 1
radio::guard_no_venv || exit 1

# 2. Pre-checks.
if ! radio::python_version_ok; then
  radio::status_fail "Python 3.11+ not found" \
    --remedy "install Python 3.11 or newer (Linux: 'apt install python3.12'; macOS: 'brew install python@3.12')" || exit 1
fi
radio::status_ok "python3 (3.11+)"

radio::require_cmd ffmpeg \
  --remedy "Linux: 'apt install ffmpeg'; macOS: 'brew install ffmpeg'" || exit 1
radio::status_ok "ffmpeg"

radio::require_cmd cmake \
  --remedy "Linux: 'apt install cmake'; macOS: 'brew install cmake'" || exit 1
radio::status_ok "cmake"

radio::require_cmd git \
  --remedy "Linux: 'apt install git'; macOS: included with Xcode CLT" || exit 1
radio::status_ok "git"

radio::require_cmd uv \
  --remedy "install uv: 'curl -LsSf https://astral.sh/uv/install.sh | sh'" || exit 1
radio::status_ok "uv"

# 3. Python deps.
radio::log_info "syncing Python deps (tts + quality extras)"
radio::dry_run_or_exec uv sync --extra tts --extra quality
radio::status_ok "uv sync complete"

# 4. whisper.cpp build (CPU/vanilla).
if [ "$SKIP_WHISPER_BUILD" != "1" ]; then
  if [ ! -d "whisper.cpp" ]; then
    radio::log_info "cloning whisper.cpp"
    radio::dry_run_or_exec git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git
  else
    radio::log_info "whisper.cpp already cloned — skipping git clone"
  fi
  if [ -f "whisper.cpp/build/bin/whisper-cli" ] && [ "${RADIO_DRY_RUN:-}" != "1" ]; then
    radio::status_ok "whisper.cpp already built — skipping cmake"
  else
    radio::log_info "building whisper.cpp (CPU)"
    radio::dry_run_or_exec cmake -B whisper.cpp/build -S whisper.cpp -DCMAKE_BUILD_TYPE=Release
    radio::dry_run_or_exec cmake --build whisper.cpp/build --config Release -j 4
    radio::status_ok "whisper.cpp built"
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
  radio::log_info "[dry-run] would write $ENV_FILE with KOKORO_PROVIDER=CPUExecutionProvider"
else
  cat >"$ENV_FILE" <<'ENVEOF'
# Suggested env vars for agent-radio-oss on CPU.
# Source this file or copy into .env to use.

# ONNX Runtime provider for Kokoro TTS. CPU is the universal baseline.
export KOKORO_PROVIDER=CPUExecutionProvider

# whisper.cpp paths (Pillar 3 quality / transcripts).
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

radio::final_status "CPU"
radio::log_info "next: 'source .env.suggested && uv run radio demo'"
