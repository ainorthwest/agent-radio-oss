"""Distribution commands — R2 upload, podcast RSS, distribution status.

``radio distribute episode``  — upload episode → R2 + Discourse + AzuraCast
``radio distribute feed``     — regenerate + upload podcast RSS feed
``radio distribute status``   — show distribution history
"""

from __future__ import annotations

from pathlib import Path

import typer

from src.cli._output import err, output, require_extra

app = typer.Typer(name="distribute", help="Content distribution (R2, Discourse, podcast).")


@app.command()
def episode(
    ctx: typer.Context,
    mp3_path: Path = typer.Argument(..., help="Episode MP3 to distribute"),
    script_path: Path = typer.Argument(..., help="Script JSON for show notes"),
    r2_key: str | None = typer.Option(None, "--r2-key", help="Override R2 object key"),
) -> None:
    """Upload episode to R2 and post show notes to Discourse."""
    require_extra("distribute", "boto3")
    from src.distributor import distribute

    state = ctx.obj
    if not mp3_path.exists():
        err(f"MP3 not found: {mp3_path}")
    if not script_path.exists():
        err(f"Script not found: {script_path}")

    library_root = Path(state.config.library.root)
    url = distribute(
        config=state.config,
        mp3_path=mp3_path,
        script_path=script_path,
        r2_key_override=r2_key,
        library_root=library_root,
    )
    if state.json_output:
        output(state, {"url": url, "mp3": str(mp3_path)})
    else:
        print(f"Distributed: {url}")


@app.command()
def feed(
    ctx: typer.Context,
    podcast_config: Path = typer.Option(
        Path("config/podcast.yaml"), "--podcast-config", help="Podcast metadata YAML"
    ),
    output_path: Path = typer.Option(
        Path("output/feed.xml"), "-o", "--output", help="Output feed.xml path"
    ),
    url_base: str = typer.Option("", "--url-base", help="Public URL base for episode links"),
) -> None:
    """Regenerate and upload podcast RSS feed."""
    from src.podcast import generate_feed

    state = ctx.obj
    library_root = Path(state.config.library.root)
    result = generate_feed(
        config_path=podcast_config,
        output_path=output_path,
        public_url_base=url_base,
        library_root=library_root,
    )
    if state.json_output:
        output(state, {"feed": str(result)})
    else:
        print(f"Feed written to: {result}")


@app.command()
def status(ctx: typer.Context) -> None:
    """Show distribution history from the catalog."""
    from src.library import Catalog
    from src.paths import LibraryPaths

    state = ctx.obj
    config = state.config
    paths = LibraryPaths(Path(config.library.root))
    db_path = paths.db

    with Catalog(db_path) as catalog:
        programs = catalog.list_programs()
        records = []
        for prog in programs:
            eps = catalog.list_episodes(program_slug=prog.slug, limit=50)
            for ep in eps:
                dists = catalog.get_distributions("episode", ep.id)
                for d in dists:
                    records.append(
                        {
                            "program": prog.slug,
                            "episode_id": ep.id,
                            "date": ep.date,
                            "destination": getattr(d, "destination", "?"),
                            "url": getattr(d, "url", "?"),
                        }
                    )

    if state.json_output:
        output(state, records)
    else:
        if not records:
            print("No distributions recorded yet.")
        for r in records:
            print(f"  {r['program']:20s} {r['date']}  {r['destination']:10s} {r['url']}")
