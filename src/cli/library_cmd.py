"""Library & catalog commands — programs, episodes, stories, status.

``radio library programs``  — list registered programs
``radio library episodes``  — query episodes (by program, status)
``radio library tracks``    — list music tracks
``radio library status``    — catalog stats + recent activity
``radio library approve``   — change episode lifecycle status
"""

from __future__ import annotations

import dataclasses

import typer

from src.cli._output import err, output

app = typer.Typer(name="library", help="Station library & catalog.")


def _open_catalog(ctx: typer.Context):  # noqa: ANN202
    """Open a Catalog connection using global config."""
    from pathlib import Path

    from src.library import Catalog
    from src.paths import LibraryPaths

    state = ctx.obj
    config = state.config
    paths = LibraryPaths(Path(config.library.root))
    return Catalog(paths.db)


@app.command()
def programs(ctx: typer.Context) -> None:
    """List registered programs."""
    state = ctx.obj
    with _open_catalog(ctx) as catalog:
        progs = catalog.list_programs()

    if state.json_output:
        output(state, [dataclasses.asdict(p) for p in progs])
    else:
        if not progs:
            print("No programs registered.")
            return
        for p in progs:
            print(f"  {p.slug:25s} {p.name:30s} [{p.program_type}] ({p.status})")


@app.command()
def episodes(
    ctx: typer.Context,
    limit: int = typer.Option(20, "--limit", help="Max episodes to show"),
) -> None:
    """Query episodes — filterable by --program."""
    state = ctx.obj
    with _open_catalog(ctx) as catalog:
        eps = catalog.list_episodes(program_slug=state.program, limit=limit)

    if state.json_output:
        output(state, [dataclasses.asdict(e) for e in eps])
    else:
        if not eps:
            print("No episodes found.")
            return
        for e in eps:
            score = f"{e.quality_score:.2f}" if e.quality_score else "-.--"
            print(f"  #{e.id:<4d} {e.program_slug:20s} {e.date}  score={score}  [{e.status}]")


@app.command()
def tracks(
    ctx: typer.Context,
    limit: int = typer.Option(20, "--limit", help="Max tracks to show"),
) -> None:
    """List music tracks in the catalog."""
    state = ctx.obj
    with _open_catalog(ctx) as catalog:
        trks = catalog.list_tracks(program_slug=state.program, limit=limit)

    if state.json_output:
        output(state, [dataclasses.asdict(t) for t in trks])
    else:
        if not trks:
            print("No tracks found.")
            return
        for t in trks:
            score = f"{t.quality_score:.2f}" if t.quality_score else "-.--"
            dur = f"{t.duration_seconds:.0f}s" if t.duration_seconds else "?s"
            print(f"  #{t.id:<4d} {t.program_slug:20s} {t.title:30s} {dur:>6s}  score={score}")


@app.command()
def status(ctx: typer.Context) -> None:
    """Catalog stats and recent activity."""
    state = ctx.obj
    with _open_catalog(ctx) as catalog:
        progs = catalog.list_programs()
        eps = catalog.list_episodes(limit=5)
        trks = catalog.list_tracks(limit=5)

    stats = {
        "programs": len(progs),
        "recent_episodes": [
            {"id": e.id, "program": e.program_slug, "date": e.date, "status": e.status} for e in eps
        ],
        "recent_tracks": [{"id": t.id, "program": t.program_slug, "title": t.title} for t in trks],
    }

    if state.json_output:
        output(state, stats)
    else:
        print(f"Programs: {len(progs)}")
        print("\nRecent episodes:")
        for e in eps:
            print(f"  {e.program_slug:20s} {e.date} [{e.status}]")
        print("\nRecent tracks:")
        for t in trks:
            print(f"  {t.program_slug:20s} {t.title}")


@app.command()
def approve(
    ctx: typer.Context,
    episode_id: int = typer.Argument(..., help="Episode ID to update"),
    new_status: str = typer.Argument(
        ..., help="New status (reviewed, approved, scheduled, aired, archived)"
    ),
) -> None:
    """Change episode lifecycle status."""
    state = ctx.obj
    valid = {"reviewed", "approved", "scheduled", "aired", "archived"}
    if new_status not in valid:
        err(f"Invalid status '{new_status}'. Must be one of: {', '.join(sorted(valid))}")

    with _open_catalog(ctx) as catalog:
        catalog.set_episode_status(episode_id, new_status)

    if state.json_output:
        output(state, {"episode_id": episode_id, "status": new_status})
    else:
        print(f"Episode #{episode_id} → {new_status}")
