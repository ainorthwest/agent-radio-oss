#!/usr/bin/env bash
# scripts/download-models.sh — download Kokoro + Whisper model files.
#
# Idempotent: skips files that already exist in --models-dir with the
# expected sha256. Honors --dry-run, --force, --no-network, and
# RADIO_DRY_RUN / RADIO_NO_NETWORK env vars.
#
# Referenced by src/engines/kokoro.py:78 — fresh-clone installs that hit
# "Kokoro model files missing" point at this script.

set -euo pipefail

# Locate common.sh relative to this script (works whether invoked by
# absolute path, relative path, or a symlink).
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$_SELF_DIR/lib/common.sh"

# ----- pinned URLs + sha256 -------------------------------------------------
#
# Kokoro v1.0 — Apache 2.0 weights from the kokoro-onnx project's
# "model-files-v1.0" GitHub release (Jan 28, 2025).
KOKORO_ONNX_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_ONNX_SHA="7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5"
KOKORO_VOICES_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
KOKORO_VOICES_SHA="bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"

# Whisper GGML models — MIT, hosted on huggingface.co/ggerganov/whisper.cpp.
# Three sizes ship: base.en (148 MB) is the v0.1.0 default; small.en
# (488 MB) and medium.en (1.5 GB) for users who want lower WER.
WHISPER_BASE_EN_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
WHISPER_BASE_EN_SHA="a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002"
WHISPER_SMALL_EN_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin"
WHISPER_SMALL_EN_SHA="" # not pinned — set when first contributor verifies on a fresh download
WHISPER_MEDIUM_EN_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin"
WHISPER_MEDIUM_EN_SHA=""

# ----- args -----------------------------------------------------------------

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --models-dir <path>   Directory to write models into (default: ./models)
  --model <size>        Whisper model size: base.en (default), small.en, medium.en
  --dry-run             Print what would happen, do not download
  --force               Re-download even if file present
  --no-network          Refuse to download (idempotent verify only)
  --allow-unpinned      Accept downloads with no sha256 pin (small.en, medium.en)
  --skip-kokoro         Skip Kokoro models (Whisper only)
  --skip-whisper        Skip Whisper models (Kokoro only)
  -h, --help            Show this help
EOF
}

MODELS_DIR="./models"
WHISPER_MODEL="base.en"
FORCE=0
SKIP_KOKORO=0
SKIP_WHISPER=0
ALLOW_UNPINNED=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --models-dir)
      MODELS_DIR="${2:?--models-dir requires a path}"
      shift 2
      ;;
    --model)
      WHISPER_MODEL="${2:?--model requires a size}"
      shift 2
      ;;
    --dry-run)
      export RADIO_DRY_RUN=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --no-network)
      export RADIO_NO_NETWORK=1
      shift
      ;;
    --skip-kokoro)
      SKIP_KOKORO=1
      shift
      ;;
    --skip-whisper)
      SKIP_WHISPER=1
      shift
      ;;
    --allow-unpinned)
      ALLOW_UNPINNED=1
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

case "$WHISPER_MODEL" in
  base.en | small.en | medium.en) ;;
  *)
    radio::status_fail "unknown whisper model size: $WHISPER_MODEL" \
      --remedy "use one of: base.en, small.en, medium.en" || exit 1
    ;;
esac

# Resolve which Whisper URL/sha to use.
case "$WHISPER_MODEL" in
  base.en)
    WHISPER_URL="$WHISPER_BASE_EN_URL"
    WHISPER_SHA="$WHISPER_BASE_EN_SHA"
    ;;
  small.en)
    WHISPER_URL="$WHISPER_SMALL_EN_URL"
    WHISPER_SHA="$WHISPER_SMALL_EN_SHA"
    ;;
  medium.en)
    WHISPER_URL="$WHISPER_MEDIUM_EN_URL"
    WHISPER_SHA="$WHISPER_MEDIUM_EN_SHA"
    ;;
esac
WHISPER_FILE="ggml-${WHISPER_MODEL}.bin"

# Fail fast on unpinned Whisper sizes — even in dry-run mode. The Hugging
# Face URLs use ``resolve/main`` which can change at any time, so an
# unpinned download is a supply-chain gap. The operator must explicitly
# opt in via --allow-unpinned to acknowledge the risk.
if [ "$SKIP_WHISPER" != "1" ] && [ -z "$WHISPER_SHA" ] && [ "$ALLOW_UNPINNED" != "1" ]; then
  radio::status_fail "no sha256 pin for $WHISPER_FILE" \
    --remedy "this Whisper size has not been verified yet; pass --allow-unpinned to override at your own risk, or use --model base.en (sha-pinned)" || exit 1
fi

# ----- helpers --------------------------------------------------------------

# _sha256 <file> — print just the hash (no filename suffix).
# Exits the script on failure (no checksum tool available) — bash's
# `set -e` does NOT trigger on `radio::status_fail` returning nonzero
# inside an if/elif/else block, so an explicit `exit 1` is required.
_sha256() {
  local f="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$f" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$f" | awk '{print $1}'
  else
    radio::status_fail "neither sha256sum nor shasum found" \
      --remedy "install coreutils (Linux: 'apt install coreutils', macOS: built in)" || exit 1
  fi
}

# _verify_sha <file> <expected_sha>
#   Returns 0 if file's sha matches expected_sha, 1 otherwise.
#   Empty expected_sha is reachable only via --allow-unpinned (the
#   up-front policy check at top-level rejects unpinned sizes by default);
#   in that case we warn and pass.
_verify_sha() {
  local f="$1"
  local expected="$2"
  if [ -z "$expected" ]; then
    radio::log_warn "no sha256 pin for $(basename "$f"); skipping integrity check (--allow-unpinned)"
    return 0
  fi
  local actual
  actual="$(_sha256 "$f")"
  if [ "$actual" = "$expected" ]; then
    return 0
  fi
  return 1
}

# _download <url> <dest> <expected_sha>
#   Idempotent download with sha256 verify. Prints status lines.
_download() {
  local url="$1"
  local dest="$2"
  local expected_sha="$3"
  local name
  name="$(basename "$dest")"

  # Idempotent skip if dest exists and (sha matches or no sha pinned and not --force).
  if [ "$FORCE" != "1" ] && [ -f "$dest" ]; then
    if _verify_sha "$dest" "$expected_sha"; then
      radio::status_ok "$name (already present, sha verified)"
      return 0
    fi
    radio::log_warn "$name present but sha mismatch — redownloading"
  fi

  if ! radio::guard_network_allowed; then
    if [ -f "$dest" ]; then
      radio::status_warn "$name present but unable to verify (no network); using as-is"
      return 0
    fi
    radio::status_fail "$name not present and RADIO_NO_NETWORK=1" \
      --remedy "either unset RADIO_NO_NETWORK or copy $name into $(dirname "$dest")/ from a trusted source"
  fi

  if [ "${RADIO_DRY_RUN:-}" = "1" ]; then
    radio::log_info "[dry-run] would download $url → $dest"
    return 0
  fi

  mkdir -p "$(dirname "$dest")"
  local tmp="${dest}.tmp"
  rm -f "$tmp"

  if command -v curl >/dev/null 2>&1; then
    radio::log_info "downloading $name from $url"
    curl -fL --progress-bar --output "$tmp" "$url"
  elif command -v wget >/dev/null 2>&1; then
    radio::log_info "downloading $name from $url"
    wget -q --show-progress -O "$tmp" "$url"
  else
    rm -f "$tmp"
    # `set -e` does not fire on a function returning nonzero from inside
    # an else branch — explicit `exit 1` so we don't fall through into
    # _verify_sha on a tmp that no longer exists (which would surface a
    # confusing sha-mismatch error instead of "install curl or wget").
    radio::status_fail "neither curl nor wget found" \
      --remedy "install curl or wget" || exit 1
  fi

  if ! _verify_sha "$tmp" "$expected_sha"; then
    local actual
    actual="$(_sha256 "$tmp")"
    rm -f "$tmp"
    radio::status_fail "$name sha256 mismatch (got $actual, expected $expected_sha)" \
      --remedy "delete $tmp and rerun; if persistent, the upstream URL may have changed"
  fi

  mv "$tmp" "$dest"
  radio::status_ok "$name (downloaded, sha verified)"
}

# ----- main -----------------------------------------------------------------

mkdir -p "$MODELS_DIR"

if [ "$SKIP_KOKORO" != "1" ]; then
  radio::log_info "Kokoro v1.0 models → $MODELS_DIR"
  _download "$KOKORO_ONNX_URL" "$MODELS_DIR/kokoro-v1.0.onnx" "$KOKORO_ONNX_SHA"
  _download "$KOKORO_VOICES_URL" "$MODELS_DIR/voices-v1.0.bin" "$KOKORO_VOICES_SHA"
fi

if [ "$SKIP_WHISPER" != "1" ]; then
  radio::log_info "Whisper $WHISPER_MODEL → $MODELS_DIR/$WHISPER_FILE"
  _download "$WHISPER_URL" "$MODELS_DIR/$WHISPER_FILE" "$WHISPER_SHA"
fi

if [ "${RADIO_DRY_RUN:-}" = "1" ]; then
  radio::log_info "dry-run complete — no files written"
else
  radio::log_info "models ready in $MODELS_DIR"
fi
