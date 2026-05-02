#!/usr/bin/env bash
# scripts/setup-amd.sh — AMD ROCm setup for agent-radio-oss.
#
# Linux + ROCm only. Refuses on macOS, Windows, or a Linux box without
# rocminfo on PATH.
#
# v0.1.0 default: KOKORO_PROVIDER=CPUExecutionProvider on AMD. The MIGraphX
# runtime null-pointer (AMDMIGraphX#4618) currently blocks GPU rendering on
# gfx1201, so CPU on AMD is the recommended path until upstream lands a fix.
# Pass --enable-migraphx to opt in to the GPU path at your own risk.
# whisper.cpp DOES build with HIP successfully (Day 3a confirmed on RX 9070),
# so the GPU path is engaged for whisper.cpp regardless of the Kokoro choice.
#
# Honors --dry-run, --skip-models, --skip-whisper-build, --skip-self-test,
# --enable-migraphx, --gfx-target <name>.

set -euo pipefail

_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$_SELF_DIR/lib/common.sh"

# AMD onnxruntime-migraphx wheel for ROCm 7.2.1, Python 3.12. Pinned URL.
# When operators update ROCm, they're on their own wheel until we re-pin
# (and probably re-test).
#
# v0.1.0 supply-chain note: this URL is versioned (rocm-rel-7.2.1) so
# AMD shouldn't be rotating contents under it, but we do not sha256-pin
# this wheel the way `download-models.sh` pins Kokoro/Whisper. Tracked
# for v0.1.1.
ONNXRUNTIME_MIGRAPHX_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/onnxruntime_migraphx-1.23.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"

DEFAULT_GFX_TARGET="gfx1201" # RX 9070; operators on other AMD GPUs override

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --dry-run               Print steps without invoking them
  --skip-models           Don't run download-models.sh
  --skip-whisper-build    Don't build whisper.cpp
  --skip-self-test        Don't run the smoke check at the end
  --enable-migraphx       Default provider becomes MIGraphXExecutionProvider
                          (the GPU path; currently blocked on gfx1201 by
                           AMDMIGraphX#4618 — opt in at your own risk)
  --gfx-target <name>     AMDGPU target for whisper.cpp HIP build
                          (default: gfx1201)
  -h, --help              Show this help
EOF
}

SKIP_MODELS=0
SKIP_WHISPER_BUILD=0
SKIP_SELF_TEST=0
ENABLE_MIGRAPHX=0
GFX_TARGET="$DEFAULT_GFX_TARGET"

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
    --enable-migraphx)
      ENABLE_MIGRAPHX=1
      shift
      ;;
    --gfx-target)
      GFX_TARGET="${2:?--gfx-target requires a name like gfx1201}"
      shift 2
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

radio::log_info "agent-radio-oss AMD ROCm setup"

PLATFORM="$(radio::detect_platform)"
radio::log_info "platform: $PLATFORM"

# Refuse if not linux-amd. The detect_platform autodetect treats missing
# rocminfo as not-AMD; an explicit override (RADIO_PLATFORM_OVERRIDE=linux-amd)
# is allowed for tests / contributors who know what they're doing.
case "$PLATFORM" in
  linux-amd) ;;
  linux-cpu)
    radio::status_fail "ROCm not detected — rocminfo not on PATH" \
      --remedy "install ROCm: 'sudo apt install rocm rocm-dev rocm-hip rocminfo' (Ubuntu 24.04 + AMD repo). Add yourself to render+video groups and log out/in. See docs/hardware/amd-rocm.md." || exit 1
    ;;
  linux-cuda)
    radio::status_fail "this is the AMD setup script — detected NVIDIA hardware (nvidia-smi present)" \
      --remedy "use 'bash scripts/setup-cuda.sh' instead" || exit 1
    ;;
  *)
    radio::status_fail "setup-amd.sh requires Linux + ROCm — detected: $PLATFORM" \
      --remedy "macOS: 'bash scripts/setup-mac.sh'; Linux without GPU: 'bash scripts/setup-cpu.sh'" || exit 1
    ;;
esac

# 1. Safety guards.
radio::guard_not_root || exit 1
radio::guard_no_venv || exit 1

# 2. Pre-checks.
if ! radio::python_version_ok; then
  radio::status_fail "Python 3.11+ not found" \
    --remedy "Ubuntu 24.04: 'apt install python3.12 python3.12-venv'" || exit 1
fi
radio::status_ok "python3 (3.11+)"

# AMD wheel is built specifically for cp312 — refuse other Python versions
# up front so operators see an actionable error instead of a cryptic pip
# wheel-format mismatch later.
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)' 2>/dev/null; then
  radio::status_fail "AMD onnxruntime-migraphx wheel requires Python 3.12 (detected: $(python3 --version 2>&1))" \
    --remedy "install Python 3.12: 'sudo apt install python3.12 python3.12-venv'; then 'uv python pin 3.12'" || exit 1
fi
radio::status_ok "python3.12 (AMD wheel constraint)"

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

radio::require_cmd rocminfo \
  --remedy "install ROCm and add yourself to render+video groups" || exit 1
# Verify the GPU is detectable. Capture rocminfo output and grep for gfx1xxx.
ROCM_INFO_OUT="$(rocminfo 2>/dev/null || true)"
if ! echo "$ROCM_INFO_OUT" | grep -qE 'gfx1[0-9]{3}'; then
  radio::status_fail "rocminfo runs but no gfx1xxx GPU detected" \
    --remedy "your ROCm install isn't seeing your AMD GPU — check 'rocm-smi' and group memberships (render, video). See docs/hardware/amd-rocm.md." || exit 1
fi
radio::status_ok "rocminfo (GPU detected)"

# 3. Python deps.
radio::log_info "syncing Python deps (tts + quality extras)"
radio::dry_run_or_exec uv sync --extra tts --extra quality
radio::status_ok "uv sync complete"

# 4. onnxruntime-migraphx wheel surgery.
# Per docs/hardware/amd-rocm.md quirk #2: when both `onnxruntime` and
# `onnxruntime-migraphx` are installed, the stock one wins on import and
# the MIGraphX provider is silently invisible. Fix: uninstall both, then
# install only migraphx.
#
# `|| true` on the uninstall: the package(s) may not be present yet on
# first run, in which case `uv pip uninstall` exits nonzero and `set -e`
# would abort the script before we got to the install step. The
# uninstall is best-effort — the install line below is the load-bearing
# one.
radio::log_info "installing AMD onnxruntime-migraphx wheel"
radio::dry_run_or_exec uv pip uninstall onnxruntime onnxruntime-migraphx || true
radio::dry_run_or_exec uv pip install --no-deps "$ONNXRUNTIME_MIGRAPHX_URL"
radio::status_ok "onnxruntime-migraphx 1.23.2 installed"

# 5. whisper.cpp build (HIP). Note: HIP works on RX 9070 even though
# Kokoro/MIGraphX hangs on the same GPU — different abstractions over
# the same silicon. That's the educational point of the OSS repo.
if [ "$SKIP_WHISPER_BUILD" != "1" ]; then
  if [ ! -d "whisper.cpp" ]; then
    radio::log_info "cloning whisper.cpp"
    radio::dry_run_or_exec git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git
  else
    radio::log_info "whisper.cpp already cloned — skipping git clone"
  fi
  # The HIP build lands in build-hip/ rather than build/ so a CPU build
  # done previously doesn't get clobbered.
  if [ -f "whisper.cpp/build-hip/bin/whisper-cli" ]; then
    radio::status_ok "whisper.cpp HIP build present — skipping cmake"
  else
    radio::log_info "building whisper.cpp (HIP, target=$GFX_TARGET)"
    radio::dry_run_or_exec cmake -B whisper.cpp/build-hip -S whisper.cpp \
      -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS="$GFX_TARGET"
    radio::dry_run_or_exec cmake --build whisper.cpp/build-hip --config Release -j 8
    radio::status_ok "whisper.cpp built (HIP, $GFX_TARGET)"
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
if [ "$ENABLE_MIGRAPHX" = "1" ]; then
  KOKORO_PROVIDER_VALUE="MIGraphXExecutionProvider"
  PROVIDER_NOTE="# WARNING: MIGraphX GPU rendering for Kokoro is currently unreliable.
# Confirmed broken on gfx1201 (AMDMIGraphX#4618); gfx1101 reports the
# same null-pointer with the identical stack. Other RDNA3+ cards may
# also be affected. You opted in via --enable-migraphx — render may
# hang or fail. See docs/investigations/kokoro-amd-rocm.md for the
# full diagnosis."
else
  KOKORO_PROVIDER_VALUE="CPUExecutionProvider"
  PROVIDER_NOTE="# v0.1.0 recommendation: CPU on AMD. The MIGraphX runtime null-pointer
# (AMDMIGraphX#4618) blocks GPU rendering on gfx1201, and gfx1101
# reports the same issue. CPU on Ryzen 7 9700X renders the audition
# in 8.25s — a known-good path. To opt in to the GPU path on hardware
# where you've confirmed MIGraphX works, rerun with --enable-migraphx."
fi

if [ "${RADIO_DRY_RUN:-}" = "1" ]; then
  radio::log_info "[dry-run] would write $ENV_FILE with KOKORO_PROVIDER=$KOKORO_PROVIDER_VALUE"
else
  cat >"$ENV_FILE" <<ENVEOF
# Suggested env vars for agent-radio-oss on AMD ROCm.
# Source this file from the repo root (where you ran setup-amd.sh).
# The \$(pwd) expressions below resolve at source-time, so sourcing
# from a different directory will produce wrong paths.

# ONNX Runtime provider for Kokoro TTS.
$PROVIDER_NOTE
export KOKORO_PROVIDER=$KOKORO_PROVIDER_VALUE

# whisper.cpp paths (Pillar 3 quality / transcripts; HIP build engages GPU).
export RADIO_WHISPER_BIN="\$(pwd)/whisper.cpp/build-hip/bin/whisper-cli"
export RADIO_WHISPER_MODEL="\$(pwd)/models/ggml-base.en.bin"
ENVEOF
  radio::status_ok "$ENV_FILE written (provider=$KOKORO_PROVIDER_VALUE)"
fi

# 8. Self-test.
# Default: warn-and-continue on smoke-test failure (operator-friendly).
# RADIO_STRICT_SMOKE=1 promotes the failure to an exit-2 — used by CI.
if [ "$SKIP_SELF_TEST" != "1" ] && [ "${RADIO_DRY_RUN:-}" != "1" ]; then
  radio::log_info "running smoke test (--quick)"
  if bash "$_SELF_DIR/oss-smoke.sh" --quick; then
    radio::status_ok "smoke test passed"
  else
    if [ "${RADIO_STRICT_SMOKE:-0}" = "1" ]; then
      radio::status_fail "smoke test failed (RADIO_STRICT_SMOKE=1 — failing setup)" \
        --remedy "run 'bash scripts/oss-smoke.sh --quick' manually to see the underlying error"
      exit 2
    fi
    radio::status_warn \
      "smoke test failed — install completed but verification did not (set RADIO_STRICT_SMOKE=1 to fail-fast)"
  fi
elif [ "$SKIP_SELF_TEST" = "1" ]; then
  radio::status_warn "skipping self-test (--skip-self-test)"
fi

radio::final_status "AMD ROCm ($GFX_TARGET, kokoro=$KOKORO_PROVIDER_VALUE)"
# v0.1.0 ships CPU-default on AMD because the MIGraphX GPU path is gated on
# upstream AMDMIGraphX#4618. Surface that decision visibly so an operator
# doesn't have to read the README to learn why their RX 9070 isn't GPU-accelerated.
if [ "$ENABLE_MIGRAPHX" != "1" ]; then
  radio::log_info \
    "AMD GPU (MIGraphX) Kokoro path gated on AMDMIGraphX#4618 — running CPU. See docs/investigations/kokoro-amd-rocm.md."
  radio::log_info "to override: rerun with --enable-migraphx (not recommended for v0.1.0)"
fi
radio::log_info "next: 'source .env.suggested && uv run radio demo'"
