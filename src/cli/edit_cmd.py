"""``radio edit`` — script-segment editor + anomaly inspector.

Two surfaces:

``radio edit script <path>``
    Mutate a script.json in place. One operation per invocation:
    --delete N, --replace N --text "..."  , --reorder "0,2,1,3", or
    --change-voice N --speaker host_b. Writes the modified script back
    to the same path. Honors ``--dry-run`` (no write).

``radio edit anomalies <manifest>``
    Run the post-render anomaly detector over a rendered episode's
    manifest.json. Writes ``anomalies.json`` alongside the manifest
    and prints a summary.

The agent-facing skill (``skills/edit-script/``) is a thin wrapper
around these CLI subcommands.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from src.cli._output import err, output

app = typer.Typer(name="edit", help="Edit a script or inspect render anomalies.")


# ---------------------------------------------------------------------------
# radio edit script
# ---------------------------------------------------------------------------


@app.command()
def script(
    ctx: typer.Context,
    script_path: Path = typer.Argument(..., help="Path to script.json"),
    delete: int | None = typer.Option(
        None, "--delete", help="Delete the segment at this 0-based index"
    ),
    replace: int | None = typer.Option(
        None, "--replace", help="Replace text of segment at this index (use --text)"
    ),
    text: str | None = typer.Option(None, "--text", help="New text for --replace"),
    reorder: str | None = typer.Option(
        None,
        "--reorder",
        help="Comma-separated new index order, e.g. '2,0,1' for a 3-segment script",
    ),
    change_voice: int | None = typer.Option(
        None, "--change-voice", help="Reassign segment to a different speaker (use --speaker)"
    ),
    speaker: str | None = typer.Option(
        None, "--speaker", help="New speaker key for --change-voice"
    ),
) -> None:
    """Apply one editorial operation to a script.json. Writes back in place."""
    from src import editor

    state = ctx.obj
    if not script_path.exists():
        err(f"Script not found: {script_path}")
    script_data = json.loads(script_path.read_text())

    ops_chosen = sum(1 for x in (delete, replace, reorder, change_voice) if x is not None)
    if ops_chosen != 1:
        err(
            "Exactly one of --delete, --replace, --reorder, or --change-voice "
            "must be supplied per invocation."
        )

    new_script: dict
    diff: editor.ScriptDiff
    try:
        if delete is not None:
            new_script, diff = editor.delete_segment(script_data, delete)
        elif replace is not None:
            if text is None:
                err("--replace requires --text")
            new_script, diff = editor.replace_text(script_data, replace, text)
        elif reorder is not None:
            try:
                order = [int(x) for x in reorder.split(",")]
            except ValueError:
                err(f"--reorder must be a comma-separated list of integers, got {reorder!r}")
            new_script, diff = editor.reorder_segments(script_data, order)
        elif change_voice is not None:
            if speaker is None:
                err("--change-voice requires --speaker")
            new_script, diff = editor.change_voice(script_data, change_voice, speaker)
        else:
            # Unreachable: ops_chosen guard above ensures exactly one option.
            err("internal error: no edit operation selected")
    except (IndexError, ValueError) as exc:
        err(str(exc))

    if state.dry_run:
        output(
            state,
            {"path": str(script_path), "diff": diff.to_dict(), "wrote": False},
            human_fmt=f"[dry-run] would update {script_path}: {diff.to_dict()}",
        )
        return

    script_path.write_text(json.dumps(new_script, indent=2))
    output(
        state,
        {"path": str(script_path), "diff": diff.to_dict(), "wrote": True},
        human_fmt=f"updated {script_path}: {diff.to_dict()}",
    )


# ---------------------------------------------------------------------------
# radio edit anomalies
# ---------------------------------------------------------------------------


@app.command()
def anomalies(
    ctx: typer.Context,
    manifest_path: Path = typer.Argument(..., help="Path to manifest.json"),
) -> None:
    """Run anomaly detection over a rendered episode's manifest."""
    from src.anomaly import detect_anomalies

    state = ctx.obj
    if not manifest_path.exists():
        err(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())

    # Optional WER input — quality.json may sit next to manifest.
    per_segment_wer: list[dict] = []
    quality_path = manifest_path.parent / "quality.json"
    if quality_path.exists():
        quality = json.loads(quality_path.read_text())
        per_segment_wer = quality.get("per_segment_wer", []) or []

    report = detect_anomalies(manifest, per_segment_wer=per_segment_wer)
    out_path = manifest_path.parent / "anomalies.json"
    out_path.write_text(json.dumps(report.to_dict(), indent=2))

    summary = (
        f"no anomalies found ({0} anomalies)"
        if not report.anomalies
        else f"{len(report.anomalies)} anomalies flagged — see {out_path}"
    )
    output(
        state,
        {"path": str(out_path), "count": len(report.anomalies), "anomalies": report.anomalies},
        human_fmt=summary,
    )
