"""Rendering commands — TTS, mixing, audition, and reference generation.

``radio render episode``    — full episode render from script JSON
``radio render segment``    — re-render specific segment(s) by index
``radio render audition``   — voice audition (single voice, test script)
``radio render remix``      — re-mix existing segments without re-rendering
``radio render reference``  — generate Kokoro reference clip for voice cloning
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from src.cli._output import err, output

app = typer.Typer(name="render", help="TTS rendering and audio production.")


# ---------------------------------------------------------------------------
# radio render episode
# ---------------------------------------------------------------------------


@app.command()
def episode(
    ctx: typer.Context,
    script_path: Path = typer.Argument(..., help="Script JSON to render"),
    segments_only: bool = typer.Option(
        False, "--segments-only", help="WAVs + manifest only, no mix"
    ),
) -> None:
    """Render a full episode from script JSON → MP3."""
    from src.renderer import render

    state = ctx.obj
    if not script_path.exists():
        err(f"Script not found: {script_path}")

    result = render(
        config=state.config,
        script_path=script_path,
        segments_only=segments_only,
        program_slug=state.program,
        no_music=state.no_music,
    )
    if state.json_output:
        output(state, {"output": str(result), "segments_only": segments_only})
    else:
        print(f"Output: {result}")


# ---------------------------------------------------------------------------
# radio render segment
# ---------------------------------------------------------------------------


@app.command()
def segment(
    ctx: typer.Context,
    manifest_path: Path = typer.Argument(..., help="Manifest JSON from previous render"),
    indices: str = typer.Option(
        ..., "--indices", "-i", help="Segment indices to re-render (comma-separated)"
    ),
) -> None:
    """Re-render specific segment(s) by index, then re-mix."""
    from src.mixer import mix
    from src.renderer import render_segments

    state = ctx.obj
    if not manifest_path.exists():
        err(f"Manifest not found: {manifest_path}")

    parsed_indices = set(int(x.strip()) for x in indices.split(","))

    # Find the script
    script_path = manifest_path.parent / "script.json"
    if not script_path.exists():
        script_path = manifest_path.parent / "bard-draft.json"
    if not script_path.exists():
        err("Cannot find script.json alongside manifest")

    script_data = json.loads(script_path.read_text())
    all_segments = script_data.get("segments", [])

    # Validate indices
    for idx in sorted(parsed_indices):
        if idx >= len(all_segments):
            err(f"Segment index {idx} out of range (script has {len(all_segments)} segments)")

    # Surgical re-render: only the specified segments are re-rendered,
    # existing WAVs and manifest entries for other segments are preserved.
    render_segments(
        config=state.config,
        script_path=script_path,
        episode_dir=manifest_path.parent,
        program_slug=state.program,
        indices=parsed_indices,
    )
    result = mix(manifest_path, no_music=state.no_music)

    if state.json_output:
        output(state, {"output": str(result), "re_rendered": sorted(parsed_indices)})
    else:
        print(f"Output: {result}")


# ---------------------------------------------------------------------------
# radio render audition
# ---------------------------------------------------------------------------


@app.command()
def audition(
    ctx: typer.Context,
    script_path: Path = typer.Argument(..., help="Audition script JSON"),
    voice: str = typer.Option(..., "--voice", help="Voice profile YAML path"),
    experiment: Path | None = typer.Option(None, "--experiment", help="Experiment overrides YAML"),
) -> None:
    """Voice audition — render single voice through test script."""
    from src.renderer import render_voice_audition

    state = ctx.obj
    if not script_path.exists():
        err(f"Script not found: {script_path}")

    result = render_voice_audition(
        voice_profile_path=voice,
        script_path=script_path,
        experiment_path=experiment,
    )
    if state.json_output:
        output(state, {"output": str(result), "voice": voice})
    else:
        print(f"Audition output: {result}")


# ---------------------------------------------------------------------------
# radio render remix
# ---------------------------------------------------------------------------


@app.command()
def remix(
    ctx: typer.Context,
    manifest_path: Path = typer.Argument(..., help="Manifest JSON to re-mix"),
) -> None:
    """Re-mix existing segment WAVs without re-rendering."""
    from src.mixer import mix

    state = ctx.obj
    if not manifest_path.exists():
        err(f"Manifest not found: {manifest_path}")

    result = mix(manifest_path, no_music=state.no_music)
    if state.json_output:
        output(state, {"output": str(result)})
    else:
        print(f"Output: {result}")


# ---------------------------------------------------------------------------
# radio render reference
# ---------------------------------------------------------------------------


@app.command()
def reference(
    ctx: typer.Context,
    script_path: Path = typer.Argument(..., help="Script for reference text"),
    voice: str = typer.Option(..., "--voice", help="Kokoro voice profile YAML"),
    output_path: Path | None = typer.Option(None, "-o", "--output", help="Output WAV path"),
) -> None:
    """Generate Kokoro reference clip for voice cloning engines."""
    from src.renderer import generate_reference_clip

    state = ctx.obj
    if not script_path.exists():
        err(f"Script not found: {script_path}")

    result = generate_reference_clip(
        kokoro_profile_path=voice,
        script_path=script_path,
        output_path=output_path,
    )
    if state.json_output:
        output(state, {"output": str(result), "voice": voice})
    else:
        print(f"Reference clip: {result}")
