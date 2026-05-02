#!/usr/bin/env bash
# scripts/oss-smoke.sh — universal smoke test for agent-radio-oss.
#
# Three modes:
#   --quick     sub-second sanity check: `radio --help` parses, no real render
#   --audition  default: render a short audition from the canned sample script
#               and verify the artifact exists
#   --full      run `radio demo` end-to-end and assert quality.json.verdict
#               is in {ship, review} (i.e. overall_score >= 0.5)
#
# Honors RADIO_SMOKE_TIMEOUT (default 300s) on long runs.
#
# Detects silent CPU fallback by grepping kokoro stderr for the
# "[kokoro] WARNING:" line that fires when a requested provider could
# not be loaded.

set -euo pipefail

_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$_SELF_DIR/lib/common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--quick | --audition | --full]

Modes:
  --quick      Sub-second sanity test ('radio --help' parses)
  --audition   Render a short audition from the canned sample (default)
  --full       Run 'radio demo' and assert quality verdict is ship or review

Environment:
  RADIO_SMOKE_TIMEOUT  Soft deadline in seconds (default: 300)
EOF
}

MODE="audition"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --quick)
      MODE="quick"
      shift
      ;;
    --audition)
      MODE="audition"
      shift
      ;;
    --full)
      MODE="full"
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      radio::log_err "unknown smoke mode: $1"
      usage >&2
      exit 2
      ;;
  esac
done

SMOKE_TIMEOUT="${RADIO_SMOKE_TIMEOUT:-300}"
SAMPLE_SCRIPT="library/programs/haystack-news/episodes/sample/script.json"
SAMPLE_VOICE="voices/kokoro-michael.yaml"

# Resolve the timeout command. GNU `timeout` is on Linux by default;
# macOS doesn't ship one but `brew install coreutils` provides
# `gtimeout`. If neither is present, run without a timeout (the
# operator can still kill the process with Ctrl-C).
_TIMEOUT_CMD=""
if command -v timeout >/dev/null 2>&1; then
  _TIMEOUT_CMD="timeout"
elif command -v gtimeout >/dev/null 2>&1; then
  _TIMEOUT_CMD="gtimeout"
fi

# _run_with_timeout <seconds> <cmd...>
#   Runs cmd under a wall-clock limit if a timeout binary is available;
#   otherwise runs it directly with a warning.
_run_with_timeout() {
  local secs="$1"
  shift
  if [ -n "$_TIMEOUT_CMD" ]; then
    "$_TIMEOUT_CMD" "$secs" "$@"
  else
    radio::log_debug "no timeout binary available — running without wall-clock limit"
    "$@"
  fi
}

# _grep_kokoro_warning <stderr-file>
#   Returns 0 if no silent-fallback warning, 1 otherwise.
_grep_kokoro_warning() {
  local f="$1"
  if grep -q "\[kokoro\] WARNING:" "$f"; then
    radio::log_warn "kokoro emitted a provider-fallback WARNING — your hardware backend"
    radio::log_warn "did not load. See $f."
    return 1
  fi
  return 0
}

case "$MODE" in
  quick)
    radio::log_info "smoke: --quick (radio --help)"
    if uv run radio --help >/dev/null 2>&1; then
      radio::status_ok "radio CLI parses"
    else
      radio::status_fail "radio --help failed" \
        --remedy "run 'uv sync --extra tts --extra quality' and retry" || exit 1
    fi
    ;;
  audition)
    radio::log_info "smoke: --audition (renders 1 segment from the canned sample)"
    if [ ! -f "$SAMPLE_SCRIPT" ]; then
      radio::status_fail "canned sample script not found at $SAMPLE_SCRIPT" \
        --remedy "are you running this from the repo root? this script expects the sample to live where it ships in the repo" || exit 1
    fi
    if [ ! -f "$SAMPLE_VOICE" ]; then
      radio::status_fail "voice profile not found at $SAMPLE_VOICE" \
        --remedy "are you running this from the repo root?" || exit 1
    fi
    stderr_log="$(mktemp -t radio-smoke-stderr.XXXXXX)"
    trap 'rm -f "$stderr_log"' EXIT
    if _run_with_timeout "$SMOKE_TIMEOUT" uv run radio render audition \
      "$SAMPLE_SCRIPT" --voice "$SAMPLE_VOICE" 2>"$stderr_log"; then
      radio::status_ok "audition rendered successfully"
    else
      cat "$stderr_log" >&2
      radio::status_fail "audition render failed (or timed out after ${SMOKE_TIMEOUT}s)" \
        --remedy "see stderr above; common causes: missing models, missing extras, missing whisper.cpp" || exit 1
    fi
    if ! _grep_kokoro_warning "$stderr_log"; then
      radio::status_warn "render succeeded but provider was a silent fallback"
    fi
    ;;
  full)
    radio::log_info "smoke: --full ('radio demo' end-to-end)"
    stderr_log="$(mktemp -t radio-smoke-stderr.XXXXXX)"
    trap 'rm -f "$stderr_log"' EXIT
    if ! _run_with_timeout "$SMOKE_TIMEOUT" uv run radio demo 2>"$stderr_log"; then
      cat "$stderr_log" >&2
      radio::status_fail "radio demo failed (or timed out after ${SMOKE_TIMEOUT}s)" \
        --remedy "see stderr above" || exit 1
    fi
    # Find the most recently written quality.json under the demo dir.
    quality_json="$(find library/programs/haystack-news/episodes -name quality.json -print 2>/dev/null | sort -r | head -1)"
    if [ -z "$quality_json" ]; then
      radio::status_fail "demo completed but no quality.json was written" \
        --remedy "the pipeline returned 0 but the quality stage didn't produce output — likely a missing dep (see stderr above)" || exit 1
    fi
    verdict="$(python3 -c "import json,sys; print(json.load(open('$quality_json')).get('verdict','unknown'))")"
    case "$verdict" in
      ship | review)
        radio::status_ok "demo verdict: $verdict (overall_score >= 0.5)"
        ;;
      *)
        radio::status_fail "demo verdict: $verdict (expected ship or review)" \
          --remedy "see $quality_json; the demo ran but quality is below threshold" || exit 1
        ;;
    esac
    if ! _grep_kokoro_warning "$stderr_log"; then
      radio::status_warn "demo succeeded but provider was a silent fallback"
    fi
    ;;
esac

radio::log_info "smoke: $MODE complete"
