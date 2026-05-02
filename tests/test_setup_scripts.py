"""Tests for `scripts/*.sh` — the Day 5 shell-script foundation.

This file establishes the three-tier test pattern for shell scripts in the
OSS repo:

- **Tier 1 (static):** every script under ``scripts/`` has a bash shebang,
  ``set -euo pipefail`` near the top, and passes shellcheck. The discovery
  test auto-enrolls every new script — no allowlist.
- **Tier 2 (mock):** the ``shell_runner`` fixture from ``conftest.py``
  builds a stub-PATH so the script runs *for real* with bash but external
  commands (``uv``, ``cmake``, ``apt-get``, ``brew``, ``rocminfo``,
  ``nvidia-smi``, ``curl``, ``sha256sum`` …) are intercepted and logged.
  Tests assert the right commands fired in the right order. CI runs Tier
  2.
- **Tier 3 (real, env-gated):** ``@pytest.mark.skipif`` on
  ``RADIO_RUN_REAL_SETUP=1``. Operator runs manually on each target host.
  CI never runs Tier 3.

PR 1 establishes only Tier 1 and the Tier 2 fixture. Later PRs add per-script
Tier 2 + Tier 3 tests as their scripts land.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.shell_harness import ShellRunner

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _discover_scripts() -> list[Path]:
    """Return every ``scripts/*.sh`` (excluding ``scripts/lib/*``).

    Empty list when ``scripts/`` doesn't exist yet — that's the bootstrap
    case in PR 1 where no setup scripts have landed.
    """
    if not SCRIPTS_DIR.exists():
        return []
    return sorted(p for p in SCRIPTS_DIR.glob("*.sh") if p.is_file())


def _discover_lib_scripts() -> list[Path]:
    """Return every ``scripts/lib/*.sh`` (sourced helpers, not executables)."""
    lib_dir = SCRIPTS_DIR / "lib"
    if not lib_dir.exists():
        return []
    return sorted(p for p in lib_dir.glob("*.sh") if p.is_file())


SCRIPTS = _discover_scripts()
LIB_SCRIPTS = _discover_lib_scripts()
ALL_SHELL = SCRIPTS + LIB_SCRIPTS

SHELLCHECK = shutil.which("shellcheck")


# ---------------------------------------------------------------------------
# Tier 1 — Static checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_executable_script_has_bash_shebang(script: Path) -> None:
    """Every executable script under ``scripts/`` starts with bash shebang."""
    first_line = script.read_text().splitlines()[0]
    assert first_line == "#!/usr/bin/env bash", (
        f"{script.name} must start with '#!/usr/bin/env bash', got {first_line!r}"
    )


@pytest.mark.parametrize("script", ALL_SHELL, ids=lambda p: p.name)
def test_script_has_strict_mode(script: Path) -> None:
    """Every shell file (executable + lib) sets ``-euo pipefail`` near the top.

    Lib files are sourced, so they don't need their own shebang — but they
    must still set strict mode if they manipulate state, otherwise an error
    in a sourced helper silently fails.

    Window is 30 lines so library files with substantial doc headers can
    keep the explanation up top before the ``set`` line.
    """
    text = script.read_text()
    head = "\n".join(text.splitlines()[:30])
    assert "set -euo pipefail" in head, (
        f"{script.relative_to(REPO_ROOT)} must contain 'set -euo pipefail' in the first 30 lines"
    )


@pytest.mark.skipif(SHELLCHECK is None, reason="shellcheck not installed")
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_executable_script_passes_shellcheck(script: Path) -> None:
    """Every script under ``scripts/`` passes ``shellcheck -x``."""
    result = subprocess.run(
        [SHELLCHECK, "-x", "-S", "warning", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"shellcheck failed for {script.name}:\n{result.stdout}\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Tier 2 — shell_runner fixture sanity
# ---------------------------------------------------------------------------
#
# These tests double as living documentation for how Tier-2 mock tests in
# later PRs should use the harness. Note that ``shell_runner`` and
# ``tmp_path`` are the *same* tmp_path under the hood — pytest passes the
# same instance to all fixtures and tests in a single invocation, so the
# probe scripts written into ``tmp_path`` sit alongside the fixture's
# ``_stubs_bin/`` and ``_stubs_calls.log``.


def test_shell_runner_stubs_intercept_external_commands(
    shell_runner: ShellRunner, tmp_path: Path
) -> None:
    """A stubbed command logs invocations and returns the configured exit code."""
    shell_runner.stub("fakecmd", returncode=0, stdout="hello\n")

    script = tmp_path / "probe.sh"
    script.write_text("#!/usr/bin/env bash\nset -euo pipefail\nfakecmd one two\n")
    script.chmod(0o755)

    result = shell_runner.run(str(script))

    assert result.returncode == 0
    assert any(call[0].endswith("fakecmd") and call[1:] == ["one", "two"] for call in result.calls)


def test_shell_runner_handles_args_with_pipes_and_slashes(
    shell_runner: ShellRunner, tmp_path: Path
) -> None:
    """Arguments containing ``|``, ``/``, spaces survive the call log round-trip.

    The harness uses NUL-separated records inside each call log line, so
    arguments with shell-meta characters or path-like content are preserved
    intact. This is critical for stubbing curl / cmake / uv etc. with realistic
    URL and path arguments in later PRs.
    """
    shell_runner.stub("downloader", returncode=0)

    script = tmp_path / "probe.sh"
    script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'downloader "https://example.com/path?a=1|b=2" "/tmp/with space"\n'
    )
    script.chmod(0o755)

    result = shell_runner.run(str(script))

    assert result.returncode == 0
    assert any(
        call[0].endswith("downloader")
        and call[1:] == ["https://example.com/path?a=1|b=2", "/tmp/with space"]
        for call in result.calls
    )


def test_shell_runner_propagates_nonzero_exit(shell_runner: ShellRunner, tmp_path: Path) -> None:
    """A stub that returns nonzero should propagate up through the script."""
    shell_runner.stub("breakme", returncode=2)

    script = tmp_path / "probe.sh"
    script.write_text("#!/usr/bin/env bash\nset -euo pipefail\nbreakme\n")
    script.chmod(0o755)

    result = shell_runner.run(str(script))

    assert result.returncode != 0


def test_shell_runner_restores_path_after_run(shell_runner: ShellRunner, tmp_path: Path) -> None:
    """The fixture must not leak ``$PATH`` modifications across tests."""
    original_path = os.environ.get("PATH", "")
    shell_runner.stub("ephemeral", returncode=0)

    script = tmp_path / "probe.sh"
    script.write_text("#!/usr/bin/env bash\nset -euo pipefail\nephemeral\n")
    script.chmod(0o755)
    shell_runner.run(str(script))

    # After the run, the parent process's PATH is unchanged.
    assert os.environ.get("PATH", "") == original_path


def test_shell_runner_passes_env_through(shell_runner: ShellRunner, tmp_path: Path) -> None:
    """Caller-supplied env variables reach the script."""
    script = tmp_path / "probe.sh"
    script.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\necho "GOT=${RADIO_TEST_VAR:-unset}"\n'
    )
    script.chmod(0o755)

    result = shell_runner.run(str(script), env={"RADIO_TEST_VAR": "hello-world"})

    assert "GOT=hello-world" in result.stdout


# ---------------------------------------------------------------------------
# scripts/lib/common.sh — sourced helper library shared by all setup scripts
# ---------------------------------------------------------------------------

COMMON_SH = SCRIPTS_DIR / "lib" / "common.sh"


def _common_sh_probe(body: str) -> str:
    """Return a one-shot probe script that sources common.sh and runs ``body``.

    The probe lives in the test's ``tmp_path``; sourcing common.sh from a
    relative path makes the test independent of where the runner CWDs.
    """
    return f'#!/usr/bin/env bash\nset -euo pipefail\nsource "{COMMON_SH}"\n{body}\n'


@pytest.mark.skipif(not COMMON_SH.exists(), reason="scripts/lib/common.sh not yet created")
def test_common_sh_sources_cleanly(shell_runner: ShellRunner, tmp_path: Path) -> None:
    """Sourcing common.sh in strict mode does not abort."""
    probe = tmp_path / "probe.sh"
    probe.write_text(_common_sh_probe('echo "sourced ok"'))
    probe.chmod(0o755)

    result = shell_runner.run(str(probe))

    assert result.returncode == 0, (
        f"sourcing failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "sourced ok" in result.stdout


@pytest.mark.skipif(not COMMON_SH.exists(), reason="scripts/lib/common.sh not yet created")
def test_common_sh_status_glyphs_emitted(shell_runner: ShellRunner, tmp_path: Path) -> None:
    """``radio::status_ok / status_warn / status_fail`` emit ✓ / ⚠ / ✗ glyphs."""
    probe = tmp_path / "probe.sh"
    probe.write_text(
        _common_sh_probe(
            'radio::status_ok "step one done" || true\n'
            'radio::status_warn "step two skipped" || true\n'
            "# status_fail uses --remedy and returns nonzero on purpose; capture and continue.\n"
            'radio::status_fail "step three blew up" --remedy "do X" || true\n'
        )
    )
    probe.chmod(0o755)

    result = shell_runner.run(str(probe))

    combined = result.stdout + result.stderr
    assert "✓" in combined, "status_ok must emit ✓"
    assert "⚠" in combined, "status_warn must emit ⚠"
    assert "✗" in combined, "status_fail must emit ✗"
    assert "do X" in combined, "status_fail --remedy must surface the remedy text"


@pytest.mark.skipif(not COMMON_SH.exists(), reason="scripts/lib/common.sh not yet created")
@pytest.mark.parametrize(
    ("uname_s", "uname_m", "extra_env", "expected"),
    [
        ("Darwin", "arm64", {}, "mac-arm"),
        ("Darwin", "x86_64", {}, "mac-intel"),
        ("Linux", "x86_64", {"_RADIO_FAKE_ROCMINFO": "1"}, "linux-amd"),
        ("Linux", "x86_64", {"_RADIO_FAKE_NVIDIA_SMI": "1"}, "linux-cuda"),
        ("Linux", "x86_64", {}, "linux-cpu"),
    ],
)
def test_common_sh_detect_platform(
    shell_runner: ShellRunner,
    tmp_path: Path,
    uname_s: str,
    uname_m: str,
    extra_env: dict[str, str],
    expected: str,
) -> None:
    """``radio::detect_platform`` emits one of the known platform tokens.

    We stub uname so the test is platform-agnostic. rocminfo / nvidia-smi
    presence on PATH (controlled via stub) is what distinguishes
    linux-amd / linux-cuda / linux-cpu.
    """
    shell_runner.stub("uname", returncode=0, stdout="")
    # Re-stub uname with branching on its arg via a richer custom script.
    uname_path = tmp_path / "_stubs_bin" / "uname"
    uname_path.write_text(
        "#!/usr/bin/env bash\n"
        f'if [ "${{1:-}}" = "-s" ]; then printf "%s\\n" {uname_s!r}; '
        f'elif [ "${{1:-}}" = "-m" ]; then printf "%s\\n" {uname_m!r}; '
        f'else printf "%s %s\\n" {uname_s!r} {uname_m!r}; fi\n'
    )
    uname_path.chmod(0o755)

    if extra_env.get("_RADIO_FAKE_ROCMINFO") == "1":
        shell_runner.stub("rocminfo", returncode=0, stdout="gfx1201\n")
    if extra_env.get("_RADIO_FAKE_NVIDIA_SMI") == "1":
        shell_runner.stub("nvidia-smi", returncode=0, stdout="NVIDIA-SMI ...\n")

    probe = tmp_path / "probe.sh"
    probe.write_text(
        _common_sh_probe('platform="$(radio::detect_platform)"\necho "PLATFORM=$platform"')
    )
    probe.chmod(0o755)

    result = shell_runner.run(str(probe))

    assert result.returncode == 0, f"detect_platform failed:\nstderr: {result.stderr}"
    assert f"PLATFORM={expected}" in result.stdout


@pytest.mark.skipif(not COMMON_SH.exists(), reason="scripts/lib/common.sh not yet created")
def test_common_sh_platform_override_short_circuits(
    shell_runner: ShellRunner, tmp_path: Path
) -> None:
    """``RADIO_PLATFORM_OVERRIDE`` short-circuits autodetect.

    Operators on dual-GPU hosts (NVIDIA + AMD) need a way to declare
    intent because the autodetect order is fixed. Any value goes — we
    don't validate against the canonical token list, callers are
    expected to know what they're doing.
    """
    probe = tmp_path / "probe.sh"
    probe.write_text(_common_sh_probe('echo "PLATFORM=$(radio::detect_platform)"'))
    probe.chmod(0o755)

    result = shell_runner.run(str(probe), env={"RADIO_PLATFORM_OVERRIDE": "linux-amd"})

    assert "PLATFORM=linux-amd" in result.stdout


@pytest.mark.skipif(not COMMON_SH.exists(), reason="scripts/lib/common.sh not yet created")
def test_common_sh_require_cmd_passes_when_present(
    shell_runner: ShellRunner, tmp_path: Path
) -> None:
    """``radio::require_cmd`` returns 0 when the command is on PATH."""
    shell_runner.stub("phantom", returncode=0)
    probe = tmp_path / "probe.sh"
    probe.write_text(_common_sh_probe('radio::require_cmd phantom\necho "ok"'))
    probe.chmod(0o755)

    result = shell_runner.run(str(probe))

    assert result.returncode == 0
    assert "ok" in result.stdout


@pytest.mark.skipif(not COMMON_SH.exists(), reason="scripts/lib/common.sh not yet created")
def test_common_sh_require_cmd_fails_with_remedy_when_absent(
    shell_runner: ShellRunner, tmp_path: Path
) -> None:
    """``radio::require_cmd`` exits nonzero with an actionable hint."""
    probe = tmp_path / "probe.sh"
    # No stub for `definitely-missing-cmd-xyz`. With a clean stub bin
    # prepended to PATH it will not resolve there; but the system PATH
    # may still contain unrelated binaries — pick a name unlikely to
    # exist anywhere.
    probe.write_text(
        _common_sh_probe('radio::require_cmd definitely-missing-cmd-xyz\necho "should-not-print"')
    )
    probe.chmod(0o755)

    result = shell_runner.run(str(probe))

    assert result.returncode != 0
    assert "should-not-print" not in result.stdout
    combined = result.stdout + result.stderr
    assert "definitely-missing-cmd-xyz" in combined


@pytest.mark.skipif(not COMMON_SH.exists(), reason="scripts/lib/common.sh not yet created")
def test_common_sh_refuses_root(shell_runner: ShellRunner, tmp_path: Path) -> None:
    """``radio::guard_not_root`` exits nonzero when ``$EUID == 0``.

    We can't actually run as root in tests, so we override EUID via the
    function's lookup. The function reads ``$EUID`` from bash, which we
    can override with ``EUID_OVERRIDE`` if the function honors it.
    Convention: function reads ``${EUID_OVERRIDE:-$EUID}``.
    """
    probe = tmp_path / "probe.sh"
    probe.write_text(_common_sh_probe("radio::guard_not_root || echo CAUGHT_ROOT"))
    probe.chmod(0o755)

    result = shell_runner.run(str(probe), env={"EUID_OVERRIDE": "0"})

    assert "CAUGHT_ROOT" in result.stdout, (
        f"guard_not_root must catch fake-root via EUID_OVERRIDE\nstdout={result.stdout}\nstderr={result.stderr}"
    )


@pytest.mark.skipif(not COMMON_SH.exists(), reason="scripts/lib/common.sh not yet created")
def test_common_sh_refuses_active_virtualenv(shell_runner: ShellRunner, tmp_path: Path) -> None:
    """``radio::guard_no_venv`` aborts when ``$VIRTUAL_ENV`` is set."""
    probe = tmp_path / "probe.sh"
    probe.write_text(_common_sh_probe("radio::guard_no_venv || echo CAUGHT_VENV"))
    probe.chmod(0o755)

    result = shell_runner.run(str(probe), env={"VIRTUAL_ENV": "/tmp/some-venv"})

    assert "CAUGHT_VENV" in result.stdout


@pytest.mark.skipif(not COMMON_SH.exists(), reason="scripts/lib/common.sh not yet created")
def test_common_sh_dry_run_skips_destructive_actions(
    shell_runner: ShellRunner, tmp_path: Path
) -> None:
    """``radio::dry_run_or_exec`` prints the command in dry-run, runs it otherwise."""
    shell_runner.stub("victim", returncode=0)

    # Dry-run path: command is not actually invoked.
    dry_probe = tmp_path / "dry.sh"
    dry_probe.write_text(_common_sh_probe("radio::dry_run_or_exec victim arg1 arg2"))
    dry_probe.chmod(0o755)
    dry_result = shell_runner.run(str(dry_probe), env={"RADIO_DRY_RUN": "1"})
    assert dry_result.returncode == 0
    assert all("victim" not in c[0] for c in dry_result.calls), (
        f"dry-run must not actually invoke the command; calls={dry_result.calls}"
    )

    # Non-dry path: command is invoked.
    real_probe = tmp_path / "real.sh"
    real_probe.write_text(_common_sh_probe("radio::dry_run_or_exec victim arg1 arg2"))
    real_probe.chmod(0o755)
    real_result = shell_runner.run(str(real_probe))
    assert real_result.returncode == 0
    assert any(c[0].endswith("victim") and c[1:] == ["arg1", "arg2"] for c in real_result.calls), (
        f"non-dry path must invoke the command; calls={real_result.calls}"
    )


# ---------------------------------------------------------------------------
# scripts/download-models.sh — model downloader, idempotent + checksummed
# ---------------------------------------------------------------------------

DOWNLOAD_MODELS_SH = SCRIPTS_DIR / "download-models.sh"


@pytest.mark.skipif(
    not DOWNLOAD_MODELS_SH.exists(), reason="scripts/download-models.sh not yet created"
)
def test_download_models_dry_run_does_not_invoke_curl(
    shell_runner: ShellRunner, tmp_path: Path
) -> None:
    """``--dry-run`` must not call curl/wget — only print what it would do."""
    shell_runner.stub("curl", returncode=0)
    shell_runner.stub("wget", returncode=0)
    shell_runner.stub("sha256sum", returncode=0, stdout="")
    shell_runner.stub("shasum", returncode=0, stdout="")

    result = shell_runner.run(
        str(DOWNLOAD_MODELS_SH),
        args=["--dry-run", "--models-dir", str(tmp_path / "models")],
    )

    assert result.returncode == 0, f"dry-run failed:\nstderr: {result.stderr}"
    invoked = [c[0] for c in result.calls]
    assert not any(c.endswith("/curl") or c.endswith("/wget") for c in invoked), (
        f"dry-run must not invoke curl/wget; calls={result.calls}"
    )


@pytest.mark.skipif(
    not DOWNLOAD_MODELS_SH.exists(), reason="scripts/download-models.sh not yet created"
)
def test_download_models_rejects_unknown_model_size(
    shell_runner: ShellRunner, tmp_path: Path
) -> None:
    """``--model giant.de`` (invalid size) exits nonzero with an error."""
    result = shell_runner.run(
        str(DOWNLOAD_MODELS_SH),
        args=["--dry-run", "--model", "giant.de", "--models-dir", str(tmp_path / "models")],
    )

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "giant.de" in combined or "unknown" in combined.lower() or "invalid" in combined.lower()


def _read_pinned_shas() -> dict[str, str]:
    """Parse pinned sha256 constants out of download-models.sh.

    Returns a mapping ``filename -> sha`` so tests can stub sha256sum to
    return the expected value per file. We pair the URL constants
    (KOKORO_ONNX_URL → kokoro-v1.0.onnx) with their adjacent _SHA constants
    so the test stays in sync if a future PR rotates a sha.
    """
    if not DOWNLOAD_MODELS_SH.exists():
        return {}
    text = DOWNLOAD_MODELS_SH.read_text()
    shas: dict[str, str] = {}
    import re

    pairs = [
        ("KOKORO_ONNX", "kokoro-v1.0.onnx"),
        ("KOKORO_VOICES", "voices-v1.0.bin"),
        ("WHISPER_BASE_EN", "ggml-base.en.bin"),
        ("WHISPER_SMALL_EN", "ggml-small.en.bin"),
        ("WHISPER_MEDIUM_EN", "ggml-medium.en.bin"),
    ]
    for prefix, fname in pairs:
        m = re.search(rf'^{prefix}_SHA="([^"]*)"', text, re.MULTILINE)
        if m:
            shas[fname] = m.group(1)
    return shas


@pytest.mark.skipif(
    not DOWNLOAD_MODELS_SH.exists(), reason="scripts/download-models.sh not yet created"
)
def test_download_models_unpinned_size_fails_without_allow_flag(
    shell_runner: ShellRunner, tmp_path: Path
) -> None:
    """``--model small.en`` without ``--allow-unpinned`` must hard-fail.

    The Hugging Face URLs for whisper.cpp models point at ``resolve/main``,
    which can change at any time. Until small.en/medium.en have pinned
    shas, an unverified download is a supply-chain gap — silent acceptance
    is unsafe.
    """
    shell_runner.stub("curl", returncode=0)
    shell_runner.stub("wget", returncode=0)
    shell_runner.stub("sha256sum", returncode=0, stdout="aaaaaaaa  fake\n")
    shell_runner.stub("shasum", returncode=0, stdout="aaaaaaaa  fake\n")

    result = shell_runner.run(
        str(DOWNLOAD_MODELS_SH),
        args=[
            "--dry-run",
            "--model",
            "small.en",
            "--skip-kokoro",
            "--models-dir",
            str(tmp_path / "models"),
        ],
    )

    assert result.returncode != 0, "unpinned download must hard-fail without --allow-unpinned"
    combined = result.stdout + result.stderr
    assert "allow-unpinned" in combined or "no sha256 pin" in combined.lower()


@pytest.mark.skipif(
    not DOWNLOAD_MODELS_SH.exists(), reason="scripts/download-models.sh not yet created"
)
def test_download_models_no_network_with_missing_files_fails_clean(
    shell_runner: ShellRunner, tmp_path: Path
) -> None:
    """``--no-network`` with no cached files exits nonzero with an actionable remedy.

    Regression for the code-review finding around the if/elif/else +
    set -e exemption. The previous flow could fall through into
    ``_verify_sha`` on a deleted .tmp file, surfacing a misleading
    sha-mismatch message. With the fix, the failure is clean: a single
    ✗ line plus a "to fix:" remedy pointing at how to bypass.
    """
    result = shell_runner.run(
        str(DOWNLOAD_MODELS_SH),
        args=[
            "--no-network",
            "--skip-whisper",
            "--models-dir",
            str(tmp_path / "empty-models"),
        ],
    )

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "RADIO_NO_NETWORK" in combined
    assert "to fix:" in combined
    # Critical regression check: the fix must NOT surface a sha mismatch
    # error here — the failure is "no network", not "sha mismatch".
    assert "sha256 mismatch" not in combined.lower(), (
        "fix from code review: must not fall through to sha-verify when network is unavailable"
    )


@pytest.mark.skipif(
    not DOWNLOAD_MODELS_SH.exists(), reason="scripts/download-models.sh not yet created"
)
def test_download_models_idempotent_when_files_present_with_correct_sha(
    shell_runner: ShellRunner, tmp_path: Path
) -> None:
    """If the models exist on disk with matching sha256, skip the download.

    We pre-create the three model files in the fake models dir and write
    a sha256sum stub that returns the *correct* pinned SHA for each
    filename it's asked about (parsed from download-models.sh). The
    downloader must detect the cache hit and avoid invoking curl/wget.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "kokoro-v1.0.onnx").write_text("fake")
    (models_dir / "voices-v1.0.bin").write_text("fake")
    (models_dir / "ggml-base.en.bin").write_text("fake")

    pinned = _read_pinned_shas()
    # Build a per-filename lookup table inside the stub. The stub reads
    # the basename of its last argument and prints the matching sha.
    lookup_entries = "\n".join(
        f'  if [ "$base" = "{fname}" ]; then echo "{sha}  $1"; exit 0; fi'
        for fname, sha in pinned.items()
        if sha
    )
    sha_stub_body = (
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?RADIO_TEST_CALL_LOG not set}"\n'
        '{ printf "%s" "$0";'
        ' for arg in "$@"; do printf "\\x00%s" "$arg"; done;'
        ' printf "\\n"; }'
        ' >> "$log"\n'
        "# Locate the file argument (last positional).\n"
        'for f in "$@"; do :; done\n'
        'base="$(basename "$f")"\n'
        f"{lookup_entries}\n"
        "# Unknown filename — print a deadbeef sha to force mismatch.\n"
        'echo "deadbeefdeadbeef  $f"\n'
    )
    sha_stub_path = tmp_path / "_stubs_bin" / "sha256sum"
    sha_stub_path.write_text(sha_stub_body)
    sha_stub_path.chmod(0o755)
    shasum_stub_path = tmp_path / "_stubs_bin" / "shasum"
    shasum_stub_path.write_text(sha_stub_body)
    shasum_stub_path.chmod(0o755)

    shell_runner.stub("curl", returncode=99, stdout="curl was called!\n")
    shell_runner.stub("wget", returncode=99, stdout="wget was called!\n")

    result = shell_runner.run(
        str(DOWNLOAD_MODELS_SH),
        args=["--models-dir", str(models_dir)],
    )

    assert result.returncode == 0, (
        f"idempotent run should succeed; stdout={result.stdout}\nstderr={result.stderr}"
    )
    invoked = [c[0] for c in result.calls]
    curl_called = any(c.endswith("/curl") for c in invoked)
    wget_called = any(c.endswith("/wget") for c in invoked)
    assert not (curl_called or wget_called), (
        f"idempotent path must not redownload when files are present; calls={result.calls}"
    )


# ---------------------------------------------------------------------------
# config/radio.example.yaml — referenced by src/config.py:90 but missing
# ---------------------------------------------------------------------------

EXAMPLE_CONFIG = REPO_ROOT / "config" / "radio.example.yaml"


def test_radio_example_yaml_exists() -> None:
    """The example config referenced by src/config.py:90 is present in the repo."""
    assert EXAMPLE_CONFIG.exists(), (
        "config/radio.example.yaml is referenced by src/config.py:90 but missing. "
        "PR2 of Day 5 ships it."
    )


@pytest.mark.skipif(not EXAMPLE_CONFIG.exists(), reason="config/radio.example.yaml not yet created")
def test_radio_example_yaml_loads_into_radioconfig(monkeypatch: pytest.MonkeyPatch) -> None:
    """``load_config(config/radio.example.yaml)`` populates a full RadioConfig.

    Uses monkeypatched env vars so secrets resolve to harmless test values.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("AGENT_RADIO_DISCOURSE_API_KEY", "discourse-test")
    monkeypatch.setenv("AGENT_RADIO_R2_BUCKET", "test-bucket")
    monkeypatch.setenv("AGENT_RADIO_R2_ENDPOINT", "https://r2.test")
    monkeypatch.setenv("AGENT_RADIO_R2_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AGENT_RADIO_R2_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setenv("AGENT_RADIO_R2_PUBLIC_URL_BASE", "https://cdn.test")
    monkeypatch.setenv("AGENT_RADIO_AZURACAST_API_KEY", "azuracast-test")

    from src.config import RadioConfig, load_config

    config = load_config(EXAMPLE_CONFIG)

    assert isinstance(config, RadioConfig)
    assert config.discourse.base_url
    assert config.curator.model
    assert config.renderer.engine
    # Voices block must include at least one named profile.
    assert len(config.voices) > 0
