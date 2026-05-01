"""``radio publish`` — derivative content fan-out.

Two surfaces:

``radio publish episode <dir>``
    Run the publisher over a single rendered episode directory.
    Writes episode.md, chapters.json, episode.txt, episode.jsonld.

``radio publish llms-index <program-dir> --show-name X --description Y``
    Generate the per-show llms.txt index from existing episode.md files.

Honors ``--dry-run`` (no writes; reports what would change).
"""

from __future__ import annotations

from pathlib import Path

import typer

from src.cli._output import err, output

app = typer.Typer(name="publish", help="Generate derivative content from rendered episodes.")


@app.command()
def episode(
    ctx: typer.Context,
    episode_dir: Path = typer.Argument(
        ..., help="Episode directory containing script.json + manifest.json"
    ),
    llm: bool = typer.Option(
        False, "--llm", help="Enable LLM-derived outputs (description, social copy)"
    ),
) -> None:
    """Run the publisher over a single episode."""
    from src.publisher import publish

    state = ctx.obj
    if not episode_dir.exists() or not episode_dir.is_dir():
        err(f"Episode directory not found: {episode_dir}")

    if state.dry_run:
        output(
            state,
            {"episode_dir": str(episode_dir), "wrote": False},
            human_fmt=f"[dry-run] would publish derivatives in {episode_dir}",
        )
        return

    try:
        result = publish(episode_dir, llm_enabled=llm)
    except FileNotFoundError as exc:
        err(str(exc))

    output(
        state,
        {"episode_dir": str(episode_dir), **result, "wrote": True},
        human_fmt=f"published {len(result['written'])} artifacts in {episode_dir}",
    )


@app.command(name="llms-index")
def llms_index(
    ctx: typer.Context,
    program_dir: Path = typer.Argument(..., help="Program directory (library/programs/<slug>)"),
    show_name: str = typer.Option(..., "--show-name", help="Human-readable show name"),
    description: str = typer.Option("", "--description", help="One-line show description"),
) -> None:
    """Generate llms.txt for a program."""
    from src.publisher import build_llms_txt

    state = ctx.obj
    if not program_dir.exists() or not program_dir.is_dir():
        err(f"Program directory not found: {program_dir}")

    body = build_llms_txt(program_dir, show_name=show_name, description=description)
    out_path = program_dir / "llms.txt"

    if state.dry_run:
        output(
            state,
            {"path": str(out_path), "preview": body[:200], "wrote": False},
            human_fmt=f"[dry-run] would write {out_path}",
        )
        return

    out_path.write_text(body)
    output(
        state,
        {"path": str(out_path), "wrote": True},
        human_fmt=f"wrote {out_path}",
    )
