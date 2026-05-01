"""agent-radio-oss unified CLI.

Single entry point for all station operations:
    radio <group> <command> [options]

Global options (available on every command):
    --config PATH      Path to radio.yaml (default: config/radio.yaml)
    --program SLUG     Target program slug (e.g., haystack-news)
    --dry-run          Plan without side effects
    -v / --verbose     Detailed output
    --json             Machine-readable JSON output
    --no-music         Skip music overlay
"""

from __future__ import annotations

from dataclasses import dataclass, field

import typer

# ---------------------------------------------------------------------------
# Shared state — populated by the root callback, read by every command
# ---------------------------------------------------------------------------


@dataclass
class State:
    """Global CLI state passed through typer.Context.obj."""

    config_path: str = "config/radio.yaml"
    program: str | None = None
    dry_run: bool = False
    verbose: bool = False
    json_output: bool = False
    no_music: bool = False

    _config: object | None = field(default=None, repr=False)

    @property
    def config(self):  # noqa: ANN201
        """Load RadioConfig on first access (lazy)."""
        if self._config is None:
            from src.config import load_config

            self._config = load_config(self.config_path)
        return self._config


# ---------------------------------------------------------------------------
# Root application
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="radio",
    help="agent-radio-oss — station control CLI.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


@app.callback()
def main(
    ctx: typer.Context,
    config: str = typer.Option("config/radio.yaml", help="Path to radio.yaml"),
    program: str | None = typer.Option(None, help="Target program slug"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan without side effects"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Detailed output"),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
    no_music: bool = typer.Option(False, "--no-music", help="Skip music overlay"),
) -> None:
    """agent-radio-oss — station control CLI."""
    ctx.obj = State(
        config_path=config,
        program=program,
        dry_run=dry_run,
        verbose=verbose,
        json_output=json_output,
        no_music=no_music,
    )


# ---------------------------------------------------------------------------
# Register OSS-safe command groups.
#
# Kept from the production CLI: config, distribute, library, render, run,
# soundbooth, stream. Stream stays because a station without an AzuraCast
# transmitter is just a podcast; AzuraCast is Apache 2.0.
#
# Dropped from the production CLI: edit, eval, music, viz, wire, write
# (all newsroom / Steward / Bard / MLX-music / matplotlib viz dependent).
# ---------------------------------------------------------------------------

from src.cli.config_cmd import app as config_app  # noqa: E402
from src.cli.distribute_cmd import app as distribute_app  # noqa: E402
from src.cli.edit_cmd import app as edit_app  # noqa: E402
from src.cli.library_cmd import app as library_app  # noqa: E402
from src.cli.render_cmd import app as render_app  # noqa: E402
from src.cli.run_cmd import app as run_app  # noqa: E402
from src.cli.soundbooth_cmd import app as soundbooth_app  # noqa: E402
from src.cli.stream_cmd import app as stream_app  # noqa: E402

app.add_typer(config_app, name="config")
app.add_typer(distribute_app, name="distribute")
app.add_typer(edit_app, name="edit")
app.add_typer(library_app, name="library")
app.add_typer(render_app, name="render")
app.add_typer(run_app, name="run")
app.add_typer(soundbooth_app, name="soundbooth")
app.add_typer(stream_app, name="stream")
