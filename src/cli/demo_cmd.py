"""``radio demo`` — one-command guided tour of the station pipeline.

Goal: a stranger who just cloned the repo runs ``uv run radio demo``
and gets a complete, defensible Haystack News episode + transcripts +
quality report + publisher artifacts in a single timestamped directory,
without learning config patching, voice mapping, or which extras to
install.

Behavior:
  - Detects whether OpenRouter / curator credentials are configured.
    Present → runs the full pipeline (curator → render → quality →
    publisher).
    Absent → uses the canned sample script that ships in the repo
    (``library/programs/haystack-news/episodes/sample/script.json``)
    so the renderer + STT + mixer + publisher chain can be exercised
    without any LLM key.
  - Always uses ``--no-distribute`` so a demo run never accidentally
    posts to R2, Discourse, or AzuraCast.
  - Writes a small ``DEMO_README.md`` next to the artifacts naming
    which path was taken, which engine + voices ran, what the quality
    verdict means, and which artifact is what.

This is the AX-roadmap "Phase 4: the demo path is a first-class path"
goal, applied to OSS.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import typer

from src.cli._output import output

# Repo-relative path to the canned sample script. Used by the no-creds path.
SAMPLE_SCRIPT_PATH = Path("library/programs/haystack-news/episodes/sample/script.json")

DEMO_PROGRAM_SLUG = "haystack-news"

# First-run bootstrap: when the operator has run `bash scripts/download-models.sh`
# but not copied the example config, the demo command auto-copies it. Forcing
# a manual `cp config/radio.example.yaml config/radio.yaml` would be the kind
# of friction excellence #8 (AX) explicitly forbids.
CONFIG_PATH = Path("config/radio.yaml")
EXAMPLE_CONFIG_PATH = Path("config/radio.example.yaml")


def _ensure_config_present() -> None:
    """Auto-copy ``config/radio.example.yaml`` → ``config/radio.yaml`` if missing."""
    if CONFIG_PATH.exists():
        return
    if not EXAMPLE_CONFIG_PATH.exists():
        return  # Nothing we can do — caller will hit a clearer error from load_config.
    import shutil

    print(
        f"ⓘ {CONFIG_PATH} not found — bootstrapping from {EXAMPLE_CONFIG_PATH}. "
        "Edit it later to add credentials for the full pipeline."
    )
    shutil.copy2(EXAMPLE_CONFIG_PATH, CONFIG_PATH)


def _curator_credentials_present(config_path: str) -> bool:
    """Return True if the curator has a usable LLM key."""
    try:
        from src.config import load_config

        config = load_config(config_path)
        return bool(config.curator.api_key)
    except Exception:
        return False


def _write_readme(demo_dir: Path, *, used_curator: bool, exit_code: int, date_str: str) -> None:
    """Drop a small README into the demo dir naming what was rendered."""
    quality_path = demo_dir / "quality.json"
    quality_summary = "(quality.json not produced — render likely failed)"
    if quality_path.exists():
        try:
            import json as _json

            qr = _json.loads(quality_path.read_text())
            quality_summary = (
                f"verdict={qr.get('verdict', 'unknown')} "
                f"(overall_score={qr.get('overall_score', 0):.4f}) — "
                f"{qr.get('verdict_reason', '')}"
            )
        except Exception:
            pass

    source = (
        "curator (live LLM call against OpenRouter)"
        if used_curator
        else (f"canned sample script ({SAMPLE_SCRIPT_PATH}) — no OPENROUTER_API_KEY found")
    )

    body = f"""# agent-radio-oss demo run

Date: {date_str}
Exit code: {exit_code}
Script source: {source}

## What's in this directory

- `script.json` — the multi-voice script the renderer consumed
- `manifest.json` — per-segment render metadata (engine, voice, hash, cache hit)
- `episode.mp3` — the rendered episode (mixer output, with pre-rendered music beds)
- `quality.json` — three-pillar quality report (signal + perceived + intelligibility)
- `transcript.txt` / `transcript.srt` — whisper.cpp Pillar 3 outputs (when available)
- `episode.md`, `chapters.json`, `episode.txt`, `episode.jsonld` — publisher artifacts
- `anomalies.json` — post-render anomaly detector output
- `script-quality.json` — pre-render LLM-judged structural quality

## Quality verdict

{quality_summary}

The verdict is one of:

- `ship` — overall_score >= 0.7. Pipeline would distribute.
- `review` — overall_score in [0.5, 0.7). Pipeline holds for human review.
- `hold` — overall_score < 0.5. Pipeline holds and recommends investigation.

For thresholds and the score breakdown, see `src/quality.py`.

## Re-running

```
uv run radio demo
```

Each demo run writes to a fresh timestamped directory under
`library/programs/{DEMO_PROGRAM_SLUG}/episodes/`.
"""
    (demo_dir / "DEMO_README.md").write_text(body)


def demo(ctx: typer.Context) -> None:
    """Run the demo pipeline end-to-end against the haystack-news show.

    Always runs with --no-distribute on. Detects whether curator
    credentials are configured; uses the canned sample script if not.
    Auto-bootstraps ``config/radio.yaml`` from the example when missing.
    """
    state = ctx.obj
    from src.paths import LibraryPaths
    from src.pipeline import run as pipeline_run

    _ensure_config_present()

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d-demo-%H%M%S")
    used_curator = _curator_credentials_present(state.config_path)

    if used_curator:
        print("✓ Curator credentials detected — running full pipeline (live LLM).")
        script_override = None
    else:
        if not SAMPLE_SCRIPT_PATH.exists():
            print(
                f"ERROR: No curator key found and sample script missing at "
                f"{SAMPLE_SCRIPT_PATH}. Cannot run demo."
            )
            raise SystemExit(1)
        print(
            f"ⓘ No curator credentials found — using canned sample script ({SAMPLE_SCRIPT_PATH})."
        )
        script_override = SAMPLE_SCRIPT_PATH

    print(f"  Demo episode directory: {timestamp}")
    print()

    exit_code = pipeline_run(
        config_path=state.config_path,
        dry_run=False,
        program_slug=DEMO_PROGRAM_SLUG,
        no_music=state.no_music,
        no_distribute=True,  # demos never distribute
        script_override=script_override,
        date_override=timestamp,
    )

    # Resolve where the pipeline wrote its artifacts so we can drop a README.
    try:
        from src.config import load_config

        cfg = load_config(state.config_path)
        lib_paths = LibraryPaths(Path(cfg.library.root))
        demo_dir = lib_paths.episode_dir(DEMO_PROGRAM_SLUG, timestamp)
    except Exception:
        demo_dir = Path("library/programs") / DEMO_PROGRAM_SLUG / "episodes" / timestamp

    if demo_dir.exists():
        _write_readme(demo_dir, used_curator=used_curator, exit_code=exit_code, date_str=timestamp)
        episode_mp3 = demo_dir / "episode.mp3"
        quality_json = demo_dir / "quality.json"
        if exit_code == 0 and episode_mp3.exists():
            print()
            print("=" * 50)
            print("  Demo complete.")
            print(f"  Listen:  {episode_mp3}")
            print(f"  Scores:  {quality_json}")
            print(f"  README:  {demo_dir / 'DEMO_README.md'}")
            print("=" * 50)

    if state.json_output:
        output(
            state,
            {
                "exit_code": exit_code,
                "demo_dir": str(demo_dir),
                "used_curator": used_curator,
            },
        )

    if exit_code != 0:
        raise SystemExit(exit_code)
