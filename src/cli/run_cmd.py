"""Daily operational entry points.

``radio run pipeline`` — full curate → render → evaluate → distribute

The production repo also exposes ``radio run newsroom`` (Steward / Bard /
Wire Desk chain) and ``radio run music`` (MusicGen MLX music sets);
those are intentionally not part of the OSS distribution.
"""

from __future__ import annotations

import typer

from src.cli._output import output

app = typer.Typer(name="run", help="Run production pipelines.")


@app.command()
def pipeline(ctx: typer.Context) -> None:
    """Full pipeline: curate → render → evaluate → distribute."""
    from src.pipeline import run as pipeline_run

    state = ctx.obj
    exit_code = pipeline_run(
        config_path=state.config_path,
        dry_run=state.dry_run,
        program_slug=state.program,
        no_music=state.no_music,
    )
    if state.json_output:
        output(state, {"exit_code": exit_code, "success": exit_code == 0})
    if exit_code != 0:
        raise SystemExit(exit_code)
