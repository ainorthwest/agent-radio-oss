# shellcheck shell=bash
# scripts/lib/common.sh — sourced helpers for agent-radio-oss setup scripts.
#
# Sourced (not run as a top-level program) by setup-cpu.sh, setup-mac.sh,
# setup-amd.sh, setup-cuda.sh, oss-smoke.sh, and download-models.sh.
# Provides logging, status reporting (✓/⚠/✗), platform detection,
# command-presence checks, safety guards (no-root, no-virtualenv), and a
# dry-run wrapper.
#
# Honors three env vars:
#   RADIO_DRY_RUN=1     skip destructive actions (download/install/build)
#   RADIO_VERBOSE=1     emit debug-level log lines
#   RADIO_NO_NETWORK=1  refuse to make any network call
#
# Test contract: tests/test_setup_scripts.py — every helper called below
# has a corresponding test that fails red until the helper is added.

set -euo pipefail

# ----- logging --------------------------------------------------------------

# Color codes are wrapped so unsupported terminals don't see escape garbage.
if [ -t 2 ] && [ -z "${NO_COLOR:-}" ]; then
  _RADIO_C_RESET=$'\033[0m'
  _RADIO_C_DIM=$'\033[2m'
  _RADIO_C_BLUE=$'\033[34m'
  _RADIO_C_YELLOW=$'\033[33m'
  _RADIO_C_RED=$'\033[31m'
  _RADIO_C_GREEN=$'\033[32m'
else
  _RADIO_C_RESET=""
  _RADIO_C_DIM=""
  _RADIO_C_BLUE=""
  _RADIO_C_YELLOW=""
  _RADIO_C_RED=""
  _RADIO_C_GREEN=""
fi

radio::log_info() {
  printf '%s[radio]%s %s\n' "${_RADIO_C_BLUE}" "${_RADIO_C_RESET}" "$*" >&2
}

radio::log_warn() {
  printf '%s[radio]%s %swarn:%s %s\n' \
    "${_RADIO_C_YELLOW}" "${_RADIO_C_RESET}" \
    "${_RADIO_C_YELLOW}" "${_RADIO_C_RESET}" "$*" >&2
}

radio::log_err() {
  printf '%s[radio]%s %serror:%s %s\n' \
    "${_RADIO_C_RED}" "${_RADIO_C_RESET}" \
    "${_RADIO_C_RED}" "${_RADIO_C_RESET}" "$*" >&2
}

radio::log_debug() {
  if [ "${RADIO_VERBOSE:-}" = "1" ]; then
    printf '%s[radio:debug]%s %s\n' \
      "${_RADIO_C_DIM}" "${_RADIO_C_RESET}" "$*" >&2
  fi
}

# ----- status reporting (✓ / ⚠ / ✗) -----------------------------------------

# radio::status_ok "<msg>"
#   Emit a green ✓ line. Returns 0.
radio::status_ok() {
  printf '%s✓%s %s\n' "${_RADIO_C_GREEN}" "${_RADIO_C_RESET}" "$*"
}

# radio::status_warn "<msg>"
#   Emit a yellow ⚠ line. Returns 0 — warnings are non-blocking.
radio::status_warn() {
  printf '%s⚠%s %s\n' "${_RADIO_C_YELLOW}" "${_RADIO_C_RESET}" "$*"
}

# radio::status_fail "<msg>" [--remedy "<actionable hint>"]
#   Emit a red ✗ line plus an indented "to fix:" line. Returns 1.
#   No "wall of stderr" failures: every failure surfaces a remedy.
radio::status_fail() {
  local msg="$1"
  shift
  local remedy=""
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --remedy)
        remedy="${2:-}"
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done
  printf '%s✗%s %s\n' "${_RADIO_C_RED}" "${_RADIO_C_RESET}" "$msg" >&2
  if [ -n "$remedy" ]; then
    printf '  %sto fix:%s %s\n' "${_RADIO_C_DIM}" "${_RADIO_C_RESET}" "$remedy" >&2
  fi
  return 1
}

# ----- platform detection ---------------------------------------------------

# radio::detect_platform → emits one of:
#   linux-amd | linux-cuda | linux-cpu | mac-arm | mac-intel | wsl-cpu
#
# Reads `uname -s` / `uname -m` and probes for `rocminfo` / `nvidia-smi`
# on PATH. WSL is detected via /proc/version containing "microsoft".
radio::detect_platform() {
  local kernel
  local arch
  kernel="$(uname -s)"
  arch="$(uname -m)"

  case "$kernel" in
    Darwin)
      if [ "$arch" = "arm64" ]; then
        echo "mac-arm"
      else
        echo "mac-intel"
      fi
      return 0
      ;;
    Linux)
      if [ -r /proc/version ] && grep -qi microsoft /proc/version 2>/dev/null; then
        echo "wsl-cpu"
        return 0
      fi
      if command -v nvidia-smi >/dev/null 2>&1; then
        echo "linux-cuda"
        return 0
      fi
      if command -v rocminfo >/dev/null 2>&1; then
        echo "linux-amd"
        return 0
      fi
      echo "linux-cpu"
      return 0
      ;;
    *)
      radio::log_warn "unknown kernel: $kernel — defaulting to linux-cpu"
      echo "linux-cpu"
      return 0
      ;;
  esac
}

# ----- command presence -----------------------------------------------------

# radio::require_cmd <name> [--remedy "<hint>"]
#   Pass: command exists, return 0.
#   Fail: emit a ✗ line + remedy, return 1.
radio::require_cmd() {
  local name="$1"
  shift
  local remedy="install '$name' and try again"
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --remedy)
        remedy="${2:-}"
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done
  if command -v "$name" >/dev/null 2>&1; then
    return 0
  fi
  radio::status_fail "required command not found: $name" --remedy "$remedy"
}

# radio::python_version_ok
#   Returns 0 if `python3` is >=3.11, 1 otherwise.
radio::python_version_ok() {
  if ! command -v python3 >/dev/null 2>&1; then
    return 1
  fi
  python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'
}

# ----- safety guards --------------------------------------------------------

# radio::guard_not_root
#   Aborts with ✗ if running as root. Tests can override via EUID_OVERRIDE
#   since real EUID can't be spoofed without privilege escalation.
radio::guard_not_root() {
  local euid="${EUID_OVERRIDE:-${EUID:-1000}}"
  if [ "$euid" = "0" ]; then
    radio::status_fail "this script must not be run as root" \
      --remedy "rerun as your normal user; the script will sudo for the steps that need it"
    return 1
  fi
  return 0
}

# radio::guard_no_venv
#   Aborts if VIRTUAL_ENV is set. uv manages its own venv; activating one
#   in your shell silently misroutes the install.
radio::guard_no_venv() {
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    radio::status_fail "active Python virtualenv detected at $VIRTUAL_ENV" \
      --remedy "deactivate first ('deactivate' or 'unset VIRTUAL_ENV') and rerun. uv manages its own .venv."
    return 1
  fi
  return 0
}

# ----- dry-run wrapper ------------------------------------------------------

# radio::dry_run_or_exec <cmd> [args...]
#   In dry-run mode (RADIO_DRY_RUN=1), prints what would be run and
#   returns 0 without invoking. Otherwise invokes the command verbatim.
radio::dry_run_or_exec() {
  if [ "${RADIO_DRY_RUN:-}" = "1" ]; then
    printf '%s[dry-run]%s would run: %s\n' \
      "${_RADIO_C_DIM}" "${_RADIO_C_RESET}" "$*" >&2
    return 0
  fi
  "$@"
}

# ----- network policy -------------------------------------------------------

# radio::guard_network_allowed
#   Returns 1 if RADIO_NO_NETWORK=1, 0 otherwise. Setup scripts call this
#   before any download/install step that hits the internet.
radio::guard_network_allowed() {
  if [ "${RADIO_NO_NETWORK:-}" = "1" ]; then
    radio::log_warn "RADIO_NO_NETWORK=1 — skipping network-dependent step"
    return 1
  fi
  return 0
}

# ----- final summary --------------------------------------------------------

# radio::final_status <hardware-backend>
#   Print a summary block at the end of a setup script. Captures the
#   reproducibility surface in one place.
radio::final_status() {
  local backend="${1:-unknown}"
  printf '\n%s── Setup complete ──%s\n' "${_RADIO_C_GREEN}" "${_RADIO_C_RESET}"
  printf '  Backend:    %s\n' "$backend"
  if command -v python3 >/dev/null 2>&1; then
    printf '  Python:     %s\n' "$(python3 --version 2>&1)"
  fi
  if command -v uv >/dev/null 2>&1; then
    printf '  uv:         %s\n' "$(uv --version 2>&1 | head -1)"
  fi
  printf '\n'
}
