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


# ---------------------------------------------------------------------------
# scripts/setup-cpu.sh — universal CPU baseline setup
# ---------------------------------------------------------------------------

SETUP_CPU_SH = SCRIPTS_DIR / "setup-cpu.sh"


def _clean_setup_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return a base env for setup-script tests.

    The tests run under ``uv run pytest``, which sets ``VIRTUAL_ENV``;
    setup-cpu.sh's ``guard_no_venv`` would correctly refuse to proceed.
    Tests need to clear it. We can't easily *unset* via the shell_runner's
    env-merge (it copies os.environ first), so we set ``VIRTUAL_ENV=""``
    which the bash check ``[ -n "$VIRTUAL_ENV" ]`` treats as unset.
    """
    base = {"RADIO_PLATFORM_OVERRIDE": "linux-cpu", "VIRTUAL_ENV": ""}
    if extra:
        base.update(extra)
    return base


@pytest.mark.skipif(not SETUP_CPU_SH.exists(), reason="scripts/setup-cpu.sh not yet created")
def test_setup_cpu_dry_run_does_not_invoke_destructive_commands(
    shell_runner: ShellRunner, tmp_path: Path
) -> None:
    """``--dry-run`` must not actually call uv/cmake/curl/etc."""
    for cmd in ("uv", "cmake", "make", "curl", "git", "sha256sum", "shasum", "ffmpeg"):
        shell_runner.stub(cmd, returncode=0)

    # python3 stub claiming 3.11+ for python_version_ok.
    py_stub = tmp_path / "_stubs_bin" / "python3"
    py_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        "exit 0\n"
    )
    py_stub.chmod(0o755)

    result = shell_runner.run(
        str(SETUP_CPU_SH),
        args=["--dry-run", "--skip-self-test"],
        env=_clean_setup_env(),
    )

    assert result.returncode == 0, (
        f"dry-run should succeed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    # No 'uv sync', no 'cmake -B ...', no 'curl --output', no 'git clone' may fire.
    # Read-only probes like 'uv --version' (called by radio::final_status) are fine.
    destructive_calls = []
    for call in result.calls:
        cmd = call[0]
        args = call[1:]
        if cmd.endswith("/uv") and "sync" in args:
            destructive_calls.append(call)
        elif cmd.endswith("/cmake") and "-B" in args:
            destructive_calls.append(call)
        elif cmd.endswith("/make"):
            destructive_calls.append(call)
        elif cmd.endswith("/curl") and any(a.startswith("http") for a in args):
            destructive_calls.append(call)
        elif cmd.endswith("/git") and "clone" in args:
            destructive_calls.append(call)

    assert not destructive_calls, f"dry-run invoked destructive commands: {destructive_calls}"


@pytest.mark.skipif(not SETUP_CPU_SH.exists(), reason="scripts/setup-cpu.sh not yet created")
def test_setup_cpu_calls_uv_sync_with_required_extras(
    shell_runner: ShellRunner, tmp_path: Path
) -> None:
    """Real run must invoke ``uv sync --extra tts --extra quality``.

    The CPU path doesn't need ``--extra distribute`` since R2 is optional,
    but tts + quality are mandatory for `radio demo` to work end-to-end.
    """
    for cmd in (
        "uv",
        "cmake",
        "make",
        "curl",
        "git",
        "sha256sum",
        "shasum",
        "ffmpeg",
        "python3",
    ):
        shell_runner.stub(cmd, returncode=0)

    # python3 stub needs to fake a 3.11+ version when called for python_version_ok.
    py_stub = tmp_path / "_stubs_bin" / "python3"
    py_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        '# python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)"\n'
        "exit 0\n"
    )
    py_stub.chmod(0o755)

    result = shell_runner.run(
        str(SETUP_CPU_SH),
        args=["--skip-models", "--skip-whisper-build", "--skip-self-test"],
        env=_clean_setup_env(),
    )

    assert result.returncode == 0, (
        f"setup-cpu should succeed with all skips:\nstdout={result.stdout}\nstderr={result.stderr}"
    )

    # Find the uv sync invocation and verify extras.
    uv_calls = [c for c in result.calls if c[0].endswith("/uv")]
    assert any(
        "sync" in c and "--extra" in c and "tts" in c and "quality" in c for c in uv_calls
    ), f"setup-cpu must run 'uv sync --extra tts --extra quality'; uv_calls={uv_calls}"


@pytest.mark.skipif(not SETUP_CPU_SH.exists(), reason="scripts/setup-cpu.sh not yet created")
def test_setup_cpu_skip_whisper_build_skips_cmake(
    shell_runner: ShellRunner, tmp_path: Path
) -> None:
    """``--skip-whisper-build`` must not invoke cmake."""
    for cmd in ("uv", "cmake", "make", "curl", "git", "sha256sum", "shasum", "ffmpeg"):
        shell_runner.stub(cmd, returncode=0)
    py_stub = tmp_path / "_stubs_bin" / "python3"
    py_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        "exit 0\n"
    )
    py_stub.chmod(0o755)

    result = shell_runner.run(
        str(SETUP_CPU_SH),
        args=["--skip-whisper-build", "--skip-models", "--skip-self-test"],
        env=_clean_setup_env(),
    )

    assert result.returncode == 0
    invoked = [c[0] for c in result.calls]
    assert not any(c.endswith("/cmake") for c in invoked), (
        f"--skip-whisper-build must not invoke cmake; calls={result.calls}"
    )


@pytest.mark.skipif(not SETUP_CPU_SH.exists(), reason="scripts/setup-cpu.sh not yet created")
def test_setup_cpu_writes_env_suggested(shell_runner: ShellRunner, tmp_path: Path) -> None:
    """Real run leaves a ``.env.suggested`` with KOKORO_PROVIDER=CPUExecutionProvider."""
    for cmd in ("uv", "cmake", "make", "curl", "git", "sha256sum", "shasum", "ffmpeg"):
        shell_runner.stub(cmd, returncode=0)
    py_stub = tmp_path / "_stubs_bin" / "python3"
    py_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        "exit 0\n"
    )
    py_stub.chmod(0o755)

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    result = shell_runner.run(
        str(SETUP_CPU_SH),
        args=["--skip-models", "--skip-whisper-build", "--skip-self-test"],
        env=_clean_setup_env(),
        cwd=workdir,
    )

    assert result.returncode == 0
    env_suggested = workdir / ".env.suggested"
    assert env_suggested.exists(), (
        f".env.suggested must be written; workdir contents: {list(workdir.iterdir())}"
    )
    content = env_suggested.read_text()
    assert "KOKORO_PROVIDER=CPUExecutionProvider" in content


# ---------------------------------------------------------------------------
# scripts/setup-mac.sh — Apple Silicon path
# ---------------------------------------------------------------------------

SETUP_MAC_SH = SCRIPTS_DIR / "setup-mac.sh"


@pytest.mark.skipif(not SETUP_MAC_SH.exists(), reason="scripts/setup-mac.sh not yet created")
def test_setup_mac_refuses_on_intel(shell_runner: ShellRunner, tmp_path: Path) -> None:
    """``setup-mac.sh`` is Apple Silicon only — refuses on Intel Mac."""
    for cmd in ("uv", "cmake", "make", "curl", "git", "brew", "ffmpeg"):
        shell_runner.stub(cmd, returncode=0)
    py_stub = tmp_path / "_stubs_bin" / "python3"
    py_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        "exit 0\n"
    )
    py_stub.chmod(0o755)

    result = shell_runner.run(
        str(SETUP_MAC_SH),
        args=["--dry-run", "--skip-self-test"],
        env=_clean_setup_env({"RADIO_PLATFORM_OVERRIDE": "mac-intel"}),
    )

    assert result.returncode != 0, "setup-mac.sh must refuse on Intel Mac"
    combined = result.stdout + result.stderr
    assert "Apple Silicon" in combined or "arm64" in combined or "intel" in combined.lower()


@pytest.mark.skipif(not SETUP_MAC_SH.exists(), reason="scripts/setup-mac.sh not yet created")
def test_setup_mac_writes_coreml_provider_in_env(shell_runner: ShellRunner, tmp_path: Path) -> None:
    """The .env.suggested file must set KOKORO_PROVIDER=CoreMLExecutionProvider."""
    for cmd in ("uv", "cmake", "make", "curl", "git", "brew", "ffmpeg"):
        shell_runner.stub(cmd, returncode=0)
    py_stub = tmp_path / "_stubs_bin" / "python3"
    py_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        "exit 0\n"
    )
    py_stub.chmod(0o755)

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    result = shell_runner.run(
        str(SETUP_MAC_SH),
        args=["--skip-models", "--skip-whisper-build", "--skip-self-test"],
        env=_clean_setup_env({"RADIO_PLATFORM_OVERRIDE": "mac-arm"}),
        cwd=workdir,
    )

    assert result.returncode == 0, (
        f"setup-mac should succeed on mac-arm with all skips:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    env_suggested = workdir / ".env.suggested"
    assert env_suggested.exists()
    assert "KOKORO_PROVIDER=CoreMLExecutionProvider" in env_suggested.read_text()


@pytest.mark.skipif(not SETUP_MAC_SH.exists(), reason="scripts/setup-mac.sh not yet created")
def test_setup_mac_whisper_build_uses_metal(shell_runner: ShellRunner, tmp_path: Path) -> None:
    """The whisper.cpp cmake invocation must include -DGGML_METAL=ON."""
    for cmd in ("uv", "cmake", "make", "curl", "brew", "ffmpeg"):
        shell_runner.stub(cmd, returncode=0)
    # git clone needs to *appear* to succeed. Stub git.
    shell_runner.stub("git", returncode=0)
    py_stub = tmp_path / "_stubs_bin" / "python3"
    py_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        "exit 0\n"
    )
    py_stub.chmod(0o755)

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    result = shell_runner.run(
        str(SETUP_MAC_SH),
        args=["--skip-models", "--skip-self-test"],
        env=_clean_setup_env({"RADIO_PLATFORM_OVERRIDE": "mac-arm"}),
        cwd=workdir,
    )

    cmake_calls = [c for c in result.calls if c[0].endswith("/cmake")]
    metal_call = next(
        (c for c in cmake_calls if any("GGML_METAL=ON" in arg for arg in c)),
        None,
    )
    assert metal_call is not None, (
        f"setup-mac must invoke cmake with -DGGML_METAL=ON; cmake_calls={cmake_calls}"
    )


# ---------------------------------------------------------------------------
# scripts/setup-amd.sh — AMD ROCm path
# ---------------------------------------------------------------------------

SETUP_AMD_SH = SCRIPTS_DIR / "setup-amd.sh"


@pytest.mark.skipif(not SETUP_AMD_SH.exists(), reason="scripts/setup-amd.sh not yet created")
def test_setup_amd_refuses_without_rocminfo(shell_runner: ShellRunner, tmp_path: Path) -> None:
    """``setup-amd.sh`` requires ROCm — refuses if rocminfo is absent."""
    for cmd in ("uv", "cmake", "make", "curl", "git", "ffmpeg"):
        shell_runner.stub(cmd, returncode=0)
    py_stub = tmp_path / "_stubs_bin" / "python3"
    py_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        "exit 0\n"
    )
    py_stub.chmod(0o755)
    # NB: deliberately no `rocminfo` stub.

    result = shell_runner.run(
        str(SETUP_AMD_SH),
        args=["--dry-run", "--skip-self-test"],
        env=_clean_setup_env({"RADIO_PLATFORM_OVERRIDE": "linux-cpu"}),
    )

    assert result.returncode != 0, "setup-amd.sh must refuse without rocminfo on PATH"
    combined = result.stdout + result.stderr
    assert "rocminfo" in combined.lower() or "rocm" in combined.lower()


@pytest.mark.skipif(not SETUP_AMD_SH.exists(), reason="scripts/setup-amd.sh not yet created")
def test_setup_amd_default_provider_is_cpu(shell_runner: ShellRunner, tmp_path: Path) -> None:
    """Default KOKORO_PROVIDER on AMD is CPUExecutionProvider per Day 2 follow-up.

    The MIGraphX runtime null-pointer (AMDMIGraphX#4618) blocks GPU rendering
    on gfx1201; until that lands, CPU on AMD is the v0.1.0 recommendation.
    """
    for cmd in ("uv", "cmake", "make", "curl", "git", "ffmpeg"):
        shell_runner.stub(cmd, returncode=0)
    # rocminfo needs to claim a gfx1xxx agent.
    rocminfo_stub = tmp_path / "_stubs_bin" / "rocminfo"
    rocminfo_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        'echo "  Name:                    gfx1201"\n'
        "exit 0\n"
    )
    rocminfo_stub.chmod(0o755)
    py_stub = tmp_path / "_stubs_bin" / "python3"
    py_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        "exit 0\n"
    )
    py_stub.chmod(0o755)

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    result = shell_runner.run(
        str(SETUP_AMD_SH),
        args=["--skip-models", "--skip-whisper-build", "--skip-self-test"],
        env=_clean_setup_env({"RADIO_PLATFORM_OVERRIDE": "linux-amd"}),
        cwd=workdir,
    )

    assert result.returncode == 0, (
        f"setup-amd should succeed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    env_suggested = workdir / ".env.suggested"
    assert env_suggested.exists()
    content = env_suggested.read_text()
    assert "KOKORO_PROVIDER=CPUExecutionProvider" in content, (
        f".env.suggested must default to CPUExecutionProvider on AMD; content:\n{content}"
    )


@pytest.mark.skipif(not SETUP_AMD_SH.exists(), reason="scripts/setup-amd.sh not yet created")
def test_setup_amd_enable_migraphx_flag_changes_provider(
    shell_runner: ShellRunner, tmp_path: Path
) -> None:
    """``--enable-migraphx`` opts in to the GPU path (writes MIGraphXExecutionProvider)."""
    for cmd in ("uv", "cmake", "make", "curl", "git", "ffmpeg"):
        shell_runner.stub(cmd, returncode=0)
    rocminfo_stub = tmp_path / "_stubs_bin" / "rocminfo"
    rocminfo_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        'echo "  Name:                    gfx1201"\n'
        "exit 0\n"
    )
    rocminfo_stub.chmod(0o755)
    py_stub = tmp_path / "_stubs_bin" / "python3"
    py_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        "exit 0\n"
    )
    py_stub.chmod(0o755)

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    result = shell_runner.run(
        str(SETUP_AMD_SH),
        args=[
            "--enable-migraphx",
            "--skip-models",
            "--skip-whisper-build",
            "--skip-self-test",
        ],
        env=_clean_setup_env({"RADIO_PLATFORM_OVERRIDE": "linux-amd"}),
        cwd=workdir,
    )

    assert result.returncode == 0
    env_suggested = workdir / ".env.suggested"
    assert "KOKORO_PROVIDER=MIGraphXExecutionProvider" in env_suggested.read_text()


@pytest.mark.skipif(not SETUP_AMD_SH.exists(), reason="scripts/setup-amd.sh not yet created")
def test_setup_amd_survives_first_run_when_onnxruntime_not_installed(
    shell_runner: ShellRunner, tmp_path: Path
) -> None:
    """First-run regression: `uv pip uninstall onnxruntime onnxruntime-migraphx`
    fails with nonzero when neither package is installed yet. With `set -e`
    active, that would abort before the install step. The script must
    swallow the uninstall failure (`|| true`) so the install line still
    runs.

    We simulate "first run" by stubbing uv to fail on uninstall and
    succeed on install. Without the `|| true` fix, the script would exit
    1 here.
    """
    rocminfo_stub = tmp_path / "_stubs_bin" / "rocminfo"
    rocminfo_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        'echo "  Name:                    gfx1201"\n'
        "exit 0\n"
    )
    rocminfo_stub.chmod(0o755)
    # uv stub: fail on `pip uninstall ...`, succeed on everything else.
    uv_stub = tmp_path / "_stubs_bin" / "uv"
    uv_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        '# Mimic first-run state: uninstall fails with "package not installed"\n'
        'if [ "${1:-}" = "pip" ] && [ "${2:-}" = "uninstall" ]; then\n'
        '  echo "ERROR: package not installed" >&2\n'
        "  exit 1\n"
        "fi\n"
        "exit 0\n"
    )
    uv_stub.chmod(0o755)
    py_stub = tmp_path / "_stubs_bin" / "python3"
    py_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        "exit 0\n"
    )
    py_stub.chmod(0o755)
    for cmd in ("cmake", "make", "curl", "git", "ffmpeg"):
        shell_runner.stub(cmd, returncode=0)

    result = shell_runner.run(
        str(SETUP_AMD_SH),
        args=["--skip-models", "--skip-whisper-build", "--skip-self-test"],
        env=_clean_setup_env({"RADIO_PLATFORM_OVERRIDE": "linux-amd"}),
    )

    assert result.returncode == 0, (
        f"setup-amd must survive first-run state where uninstall fails:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    # Verify install was still called even though uninstall failed.
    uv_calls = [c for c in result.calls if c[0].endswith("/uv")]
    install_seen = any(
        "install" in c and any("onnxruntime_migraphx" in a for a in c) for c in uv_calls
    )
    assert install_seen, (
        f"install step must run even when uninstall fails on first run; uv_calls={uv_calls}"
    )


@pytest.mark.skipif(not SETUP_AMD_SH.exists(), reason="scripts/setup-amd.sh not yet created")
def test_setup_amd_refuses_python_other_than_312(shell_runner: ShellRunner, tmp_path: Path) -> None:
    """The AMD wheel is cp312-only — refuse Python 3.11 / 3.13 / etc up front.

    Operators on Python 3.11 hitting the unguarded install would see a
    cryptic pip wheel-format mismatch error. Catching this in the
    pre-check turns it into an actionable remedy.
    """
    rocminfo_stub = tmp_path / "_stubs_bin" / "rocminfo"
    rocminfo_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        'echo "  Name:                    gfx1201"\n'
        "exit 0\n"
    )
    rocminfo_stub.chmod(0o755)
    # python3 stub that claims 3.11 (passes >= 3.11 check but fails == 3.12 check)
    py_stub = tmp_path / "_stubs_bin" / "python3"
    py_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        "# Claim Python 3.11 — passes radio::python_version_ok (>=3.11)"
        " but fails the cp312-specific check.\n"
        'if [ "${1:-}" = "-c" ]; then\n'
        '  if echo "${2:-}" | grep -q "version_info\\[:2\\] == (3, 12)"; then\n'
        "    exit 1  # Not 3.12\n"
        "  fi\n"
        '  if echo "${2:-}" | grep -q "version_info >= (3, 11)"; then\n'
        "    exit 0  # Is 3.11+\n"
        "  fi\n"
        "fi\n"
        "exit 0\n"
    )
    py_stub.chmod(0o755)
    for cmd in ("uv", "cmake", "make", "curl", "git", "ffmpeg"):
        shell_runner.stub(cmd, returncode=0)

    result = shell_runner.run(
        str(SETUP_AMD_SH),
        args=["--dry-run"],
        env=_clean_setup_env({"RADIO_PLATFORM_OVERRIDE": "linux-amd"}),
    )

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "3.12" in combined or "Python 3.12" in combined


@pytest.mark.skipif(not SETUP_AMD_SH.exists(), reason="scripts/setup-amd.sh not yet created")
def test_setup_amd_migraphx_banner_not_only_gfx1201(
    shell_runner: ShellRunner, tmp_path: Path
) -> None:
    """The --enable-migraphx warning banner must not imply only gfx1201 is affected.

    Per code-review feedback: gfx1101 reports the same null-pointer with
    the identical stack. Operators on other RDNA3+ cards reading "blocked
    on gfx1201" might enable the broken path with false confidence.
    """
    text = SETUP_AMD_SH.read_text()
    # The banner should mention multiple gfx targets or use generic language.
    # Cheapest assertion: it does NOT say only "blocked on gfx1201" without
    # also acknowledging gfx1101 / RDNA3+.
    if "MIGraphX" in text and "gfx1201" in text:
        assert "gfx1101" in text or "RDNA3" in text or "other" in text.lower(), (
            "the MIGraphX warning must broaden beyond gfx1201; "
            "gfx1101 and other RDNA3+ cards are likely affected too"
        )


@pytest.mark.skipif(not SETUP_AMD_SH.exists(), reason="scripts/setup-amd.sh not yet created")
def test_setup_amd_uninstalls_onnxruntime_before_install(
    shell_runner: ShellRunner, tmp_path: Path
) -> None:
    """The onnxruntime-migraphx wheel install must follow uninstall of stock onnxruntime.

    Per docs/hardware/amd-rocm.md quirk #2: when both `onnxruntime` and
    `onnxruntime-migraphx` are installed, the stock one wins on import
    and the migraphx provider is silently invisible. Setup must
    uninstall both, then install only migraphx.
    """
    for cmd in ("cmake", "make", "curl", "git", "ffmpeg"):
        shell_runner.stub(cmd, returncode=0)
    rocminfo_stub = tmp_path / "_stubs_bin" / "rocminfo"
    rocminfo_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        'echo "  Name:                    gfx1201"\n'
        "exit 0\n"
    )
    rocminfo_stub.chmod(0o755)
    # uv stub must succeed for both `pip uninstall` and `pip install`.
    shell_runner.stub("uv", returncode=0)
    py_stub = tmp_path / "_stubs_bin" / "python3"
    py_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        "exit 0\n"
    )
    py_stub.chmod(0o755)

    result = shell_runner.run(
        str(SETUP_AMD_SH),
        args=["--skip-models", "--skip-whisper-build", "--skip-self-test"],
        env=_clean_setup_env({"RADIO_PLATFORM_OVERRIDE": "linux-amd"}),
    )

    assert result.returncode == 0, f"setup-amd must succeed:\nstderr={result.stderr}"
    uv_calls = [c for c in result.calls if c[0].endswith("/uv")]
    # Find the pip uninstall and pip install steps.
    uninstall_idx = next(
        (i for i, c in enumerate(uv_calls) if "uninstall" in c and "onnxruntime-migraphx" in c),
        None,
    )
    install_idx = next(
        (
            i
            for i, c in enumerate(uv_calls)
            if "install" in c and any("onnxruntime_migraphx" in a for a in c)
        ),
        None,
    )
    assert uninstall_idx is not None, (
        f"setup-amd must call 'uv pip uninstall onnxruntime-migraphx' to clear the import collision; uv_calls={uv_calls}"
    )
    assert install_idx is not None, (
        f"setup-amd must call 'uv pip install onnxruntime_migraphx-*.whl'; uv_calls={uv_calls}"
    )
    assert uninstall_idx < install_idx, (
        f"uninstall must precede install (uninstall_idx={uninstall_idx}, install_idx={install_idx})"
    )


# ---------------------------------------------------------------------------
# scripts/setup-cuda.sh — NVIDIA path (ships blind in v0.1.0)
# ---------------------------------------------------------------------------

SETUP_CUDA_SH = SCRIPTS_DIR / "setup-cuda.sh"


@pytest.mark.skipif(not SETUP_CUDA_SH.exists(), reason="scripts/setup-cuda.sh not yet created")
def test_setup_cuda_refuses_without_nvidia_smi(
    shell_runner: ShellRunner, tmp_path: Path
) -> None:
    """``setup-cuda.sh`` requires NVIDIA — refuses if nvidia-smi is absent."""
    for cmd in ("uv", "cmake", "make", "curl", "git", "ffmpeg"):
        shell_runner.stub(cmd, returncode=0)
    py_stub = tmp_path / "_stubs_bin" / "python3"
    py_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        "exit 0\n"
    )
    py_stub.chmod(0o755)
    # NB: deliberately no `nvidia-smi` stub.

    result = shell_runner.run(
        str(SETUP_CUDA_SH),
        args=["--dry-run", "--skip-self-test"],
        env=_clean_setup_env({"RADIO_PLATFORM_OVERRIDE": "linux-cpu"}),
    )

    assert result.returncode != 0, "setup-cuda.sh must refuse without nvidia-smi"
    combined = result.stdout + result.stderr
    assert "nvidia" in combined.lower() or "cuda" in combined.lower()


@pytest.mark.skipif(not SETUP_CUDA_SH.exists(), reason="scripts/setup-cuda.sh not yet created")
def test_setup_cuda_emits_untested_banner(shell_runner: ShellRunner, tmp_path: Path) -> None:
    """The script must print a prominent UNTESTED warning at start AND end.

    Per the sprint plan, setup-cuda.sh ships blind in v0.1.0 (no NVIDIA
    hardware to validate against this sprint). The banner is the
    operator-facing acknowledgement that the script is best-effort
    scaffolding rather than a verified install path. When NVIDIA hardware
    becomes available, only real-mode validation flips on.
    """
    for cmd in ("uv", "cmake", "make", "curl", "git", "ffmpeg"):
        shell_runner.stub(cmd, returncode=0)
    nvidia_stub = tmp_path / "_stubs_bin" / "nvidia-smi"
    nvidia_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        'echo "NVIDIA-SMI 550.00.00"\n'
        "exit 0\n"
    )
    nvidia_stub.chmod(0o755)
    py_stub = tmp_path / "_stubs_bin" / "python3"
    py_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        "exit 0\n"
    )
    py_stub.chmod(0o755)

    result = shell_runner.run(
        str(SETUP_CUDA_SH),
        args=["--dry-run", "--skip-self-test"],
        env=_clean_setup_env({"RADIO_PLATFORM_OVERRIDE": "linux-cuda"}),
    )

    combined = result.stdout + result.stderr
    untested_count = combined.lower().count("untested")
    assert untested_count >= 2, (
        f"banner must appear at start AND end of run; saw {untested_count} occurrences\n"
        f"output: {combined}"
    )


@pytest.mark.skipif(not SETUP_CUDA_SH.exists(), reason="scripts/setup-cuda.sh not yet created")
def test_setup_cuda_writes_cuda_provider_in_env(
    shell_runner: ShellRunner, tmp_path: Path
) -> None:
    """The .env.suggested file sets KOKORO_PROVIDER=CUDAExecutionProvider."""
    for cmd in ("uv", "cmake", "make", "curl", "git", "ffmpeg"):
        shell_runner.stub(cmd, returncode=0)
    nvidia_stub = tmp_path / "_stubs_bin" / "nvidia-smi"
    nvidia_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        'echo "NVIDIA-SMI 550.00.00"\n'
        "exit 0\n"
    )
    nvidia_stub.chmod(0o755)
    py_stub = tmp_path / "_stubs_bin" / "python3"
    py_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        "exit 0\n"
    )
    py_stub.chmod(0o755)

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    result = shell_runner.run(
        str(SETUP_CUDA_SH),
        args=["--skip-models", "--skip-whisper-build", "--skip-self-test"],
        env=_clean_setup_env({"RADIO_PLATFORM_OVERRIDE": "linux-cuda"}),
        cwd=workdir,
    )

    assert result.returncode == 0, (
        f"setup-cuda mock run should succeed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    env_suggested = workdir / ".env.suggested"
    assert env_suggested.exists()
    assert "KOKORO_PROVIDER=CUDAExecutionProvider" in env_suggested.read_text()


@pytest.mark.skipif(not SETUP_CUDA_SH.exists(), reason="scripts/setup-cuda.sh not yet created")
def test_setup_cuda_whisper_build_uses_cuda(shell_runner: ShellRunner, tmp_path: Path) -> None:
    """The whisper.cpp cmake invocation must include -DGGML_CUDA=ON."""
    for cmd in ("uv", "cmake", "make", "curl", "ffmpeg"):
        shell_runner.stub(cmd, returncode=0)
    shell_runner.stub("git", returncode=0)
    nvidia_stub = tmp_path / "_stubs_bin" / "nvidia-smi"
    nvidia_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        'echo "NVIDIA-SMI 550.00.00"\n'
        "exit 0\n"
    )
    nvidia_stub.chmod(0o755)
    py_stub = tmp_path / "_stubs_bin" / "python3"
    py_stub.write_text(
        "#!/usr/bin/env bash\n"
        'log="${RADIO_TEST_CALL_LOG:?}"\n'
        '{ printf "%s" "$0"; for arg in "$@"; do printf "\\x00%s" "$arg"; done; printf "\\n"; } >> "$log"\n'
        "exit 0\n"
    )
    py_stub.chmod(0o755)

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    result = shell_runner.run(
        str(SETUP_CUDA_SH),
        args=["--skip-models", "--skip-self-test"],
        env=_clean_setup_env({"RADIO_PLATFORM_OVERRIDE": "linux-cuda"}),
        cwd=workdir,
    )

    cmake_calls = [c for c in result.calls if c[0].endswith("/cmake")]
    cuda_call = next(
        (c for c in cmake_calls if any("GGML_CUDA=ON" in arg for arg in c)),
        None,
    )
    assert cuda_call is not None, (
        f"setup-cuda must invoke cmake with -DGGML_CUDA=ON; cmake_calls={cmake_calls}"
    )


# ---------------------------------------------------------------------------
# scripts/oss-smoke.sh — universal smoke test
# ---------------------------------------------------------------------------

OSS_SMOKE_SH = SCRIPTS_DIR / "oss-smoke.sh"


@pytest.mark.skipif(not OSS_SMOKE_SH.exists(), reason="scripts/oss-smoke.sh not yet created")
def test_oss_smoke_quick_returns_zero(shell_runner: ShellRunner, tmp_path: Path) -> None:
    """``--quick`` is a sub-second sanity test — just check radio --help works."""
    # Stub uv to succeed so `uv run radio --help` returns 0.
    shell_runner.stub("uv", returncode=0, stdout="Usage: radio ...\n")

    result = shell_runner.run(str(OSS_SMOKE_SH), args=["--quick"])

    assert result.returncode == 0, (
        f"smoke --quick should succeed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    invoked = [c[0] for c in result.calls]
    assert any(c.endswith("/uv") for c in invoked), "smoke --quick must invoke uv at minimum"


@pytest.mark.skipif(not OSS_SMOKE_SH.exists(), reason="scripts/oss-smoke.sh not yet created")
def test_oss_smoke_audition_invokes_radio_render(shell_runner: ShellRunner, tmp_path: Path) -> None:
    """``--audition`` must invoke the renderer against the canned sample.

    Run with cwd=REPO_ROOT so the canned sample and voice profile checks
    resolve. We're verifying the script's *invocation* of uv, not its
    file-existence checks (those would fail in tmp_path even though the
    files exist in the repo).
    """
    shell_runner.stub("uv", returncode=0)
    shell_runner.stub("timeout", returncode=0)  # the script wraps uv in `timeout`

    result = shell_runner.run(str(OSS_SMOKE_SH), args=["--audition"], cwd=REPO_ROOT)

    invoked_cmds = [c for c in result.calls if c[0].endswith("/uv") or c[0].endswith("/timeout")]
    flat = [arg for c in invoked_cmds for arg in c]
    assert "render" in flat or "demo" in flat, (
        f"--audition must invoke uv run radio render or demo; calls={result.calls}"
    )


@pytest.mark.skipif(not OSS_SMOKE_SH.exists(), reason="scripts/oss-smoke.sh not yet created")
def test_oss_smoke_quality_path_not_interpolated_into_python(
    shell_runner: ShellRunner, tmp_path: Path
) -> None:
    """Regression for shell-injection finding from the PR3 review.

    The verdict-extraction step in --full mode parses ``quality.json``
    via Python. The path comes from ``find`` output and is operator-
    controlled (episode dir names). It must be passed as ``sys.argv[1]``,
    NEVER interpolated into a Python source literal — otherwise a
    directory named like ``foo'); __import__('os').system('id') #`` would
    execute arbitrary code.

    This test reads the script source and asserts the extraction reads
    the path from sys.argv (heredoc + positional arg), not from a
    formatted string literal.
    """
    text = OSS_SMOKE_SH.read_text()
    # Find the verdict-extraction block.
    assert "quality_json" in text
    # The extraction must use sys.argv (heredoc + positional), not the
    # vulnerable pattern open('$quality_json').
    assert "sys.argv" in text, (
        "verdict extraction must read the path via sys.argv, not interpolation"
    )
    assert "open('$quality_json'" not in text, (
        "shell injection regression: do not interpolate $quality_json into a Python literal"
    )


@pytest.mark.skipif(not OSS_SMOKE_SH.exists(), reason="scripts/oss-smoke.sh not yet created")
def test_oss_smoke_unknown_mode_rejected(shell_runner: ShellRunner, tmp_path: Path) -> None:
    """An unrecognized mode must exit nonzero with a usage message."""
    shell_runner.stub("uv", returncode=0)

    result = shell_runner.run(str(OSS_SMOKE_SH), args=["--bogus-mode"])

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "bogus-mode" in combined or "unknown" in combined.lower() or "usage" in combined.lower()


# ---------------------------------------------------------------------------
# config/radio.example.yaml load round-trip
# ---------------------------------------------------------------------------


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
