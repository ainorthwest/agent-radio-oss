"""Configuration management commands.

This is the *reference implementation* — all other command groups
follow the same pattern established here:

1. Create a ``typer.Typer()`` named ``app``.
2. Each command receives ``ctx: typer.Context``, pulls ``state = ctx.obj``.
3. Delegates to existing module functions (thin wrapper, no logic).
4. Outputs via ``_output.output(state, data, human_fmt=...)``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import typer

from src.cli._output import err, output

app = typer.Typer(name="config", help="Configuration management.")


# ---------------------------------------------------------------------------
# radio config show
# ---------------------------------------------------------------------------


@app.command()
def show(ctx: typer.Context) -> None:
    """Dump active configuration (secrets redacted)."""
    state = ctx.obj
    try:
        cfg = state.config
    except FileNotFoundError as exc:
        err(str(exc))

    raw = dataclasses.asdict(cfg)

    # Redact anything that looks like a secret
    _redact(raw)

    if state.json_output:
        output(state, raw)
    else:
        import yaml

        print(yaml.dump(raw, default_flow_style=False, sort_keys=False))


# ---------------------------------------------------------------------------
# radio config validate
# ---------------------------------------------------------------------------


@app.command()
def validate(ctx: typer.Context) -> None:
    """Check config file, secrets, and probe optional dependencies."""
    state = ctx.obj
    issues: list[str] = []
    ok: list[str] = []

    # 1. Config file
    config_path = Path(state.config_path)
    if config_path.exists():
        ok.append(f"config file: {config_path}")
    else:
        issues.append(f"config file missing: {config_path}")

    # 2. Try loading config (validates YAML + secrets)
    try:
        state.config  # trigger load to validate
        ok.append("config loads successfully")
    except Exception as exc:  # noqa: BLE001
        issues.append(f"config load failed: {exc}")
        pass

    # 3. Probe optional dependencies (OSS extras only)
    extras = {
        "tts": "kokoro_onnx",
        "quality": "librosa",
        "distribute": "boto3",
    }
    for extra_name, mod in extras.items():
        try:
            __import__(mod)
            ok.append(f"extra [{extra_name}]: installed")
        except ImportError:
            ok.append(f"extra [{extra_name}]: not installed")

    result = {"ok": ok, "issues": issues}

    if state.json_output:
        output(state, result)
    else:
        for item in ok:
            print(f"  ok  {item}")
        for item in issues:
            print(f"  !!  {item}")
        if issues:
            print(f"\n{len(issues)} issue(s) found.")
        else:
            print("\nAll checks passed.")


# ---------------------------------------------------------------------------
# radio config engines
# ---------------------------------------------------------------------------


@app.command()
def engines(ctx: typer.Context) -> None:
    """List registered TTS engines.

    OSS ships with Kokoro ONNX only. Use ``radio soundbooth engines``
    for the same listing plus the active ONNX execution provider.
    """
    from src.engines import available_engines

    state = ctx.obj
    found = [{"id": name} for name in available_engines()]

    if state.json_output:
        output(state, found)
    else:
        if not found:
            print("No TTS engines registered. Install extras: uv sync --extra tts")
            return
        for eng in found:
            print(f"  {eng['id']}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _redact(d: dict, _secret_keys: set[str] | None = None) -> None:
    """Recursively redact values whose keys suggest secrets."""
    if _secret_keys is None:
        _secret_keys = {
            "api_key",
            "secret",
            "password",
            "token",
            "r2_access_key_id",
            "r2_secret_access_key",
        }
    for k, v in d.items():
        if isinstance(v, dict):
            _redact(v, _secret_keys)
        elif isinstance(v, str) and any(s in k.lower() for s in _secret_keys):
            d[k] = "***" if v else ""
