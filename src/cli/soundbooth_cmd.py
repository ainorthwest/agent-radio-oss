"""Soundbooth commands — voice profile + engine inspection.

OSS scope: list voices and registered engines. The voice-authoring
web UI (``radio soundbooth serve``) ships in the proprietary
agent-radio repo and depends on FastAPI + the soundbooth/ package.

``radio soundbooth engines`` — list registered TTS engines
``radio soundbooth voices``  — list voice profiles in voices/
"""

from __future__ import annotations

from pathlib import Path

import typer

from src.cli._output import output

app = typer.Typer(name="soundbooth", help="Voice profile & engine inspection.")


@app.command()
def engines(ctx: typer.Context) -> None:
    """List TTS engines registered in src.engines."""
    from src.engines import available_engines
    from src.engines.kokoro import active_provider

    state = ctx.obj
    names = available_engines()
    payload = [
        {
            "id": name,
            "active_provider": active_provider() if name == "kokoro" else None,
        }
        for name in names
    ]

    if state.json_output:
        output(state, payload)
    else:
        if not names:
            print("No TTS engines registered. (Did you install --extra tts?)")
            return
        for entry in payload:
            extra = f"  provider={entry['active_provider']}" if entry["active_provider"] else ""
            print(f"  {entry['id']}{extra}")


@app.command()
def voices(ctx: typer.Context) -> None:
    """List voice profiles in the voices/ directory."""
    import yaml

    state = ctx.obj
    voices_dir = Path("voices")
    if not voices_dir.exists():
        print("No voices/ directory found.")
        return

    profiles = sorted(voices_dir.glob("*.yaml"))
    results = []
    for p in profiles:
        with p.open() as f:
            data = yaml.safe_load(f) or {}
        results.append(
            {
                "file": p.name,
                "engine": data.get("engine", "?"),
                "name": p.stem,
            }
        )

    if state.json_output:
        output(state, results)
    else:
        if not results:
            print("No voice profiles found in voices/")
            return
        for v in results:
            print(f"  {v['name']:25s} engine={v['engine']}")
