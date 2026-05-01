"""Pipeline orchestrator: curate -> render -> evaluate -> distribute.

Entry point for the daily systemd timer on Hinoki. Runs all stages
in sequence with an optional quality gate between render and distribute.

Supports two modes:
  - Legacy: artifacts in output/episodes/{date}/ (default, backward-compatible)
  - Library: artifacts in library/programs/{slug}/episodes/{date}/ (--program flag)

Usage:
    uv run python -m src.pipeline
    uv run python -m src.pipeline --program haystack-news   # library-aware
    uv run python -m src.pipeline --config config/radio.yaml
    uv run python -m src.pipeline --dry-run   # curate + render + eval only, skip distribute
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from src.config import load_config

# Quality gate thresholds (from steward/SOUL.md)
QUALITY_SHIP = 0.7  # score >= this: ship it
QUALITY_REVIEW = 0.5  # score >= this but < SHIP: flag for human review


def run(
    config_path: str = "config/radio.yaml",
    dry_run: bool = False,
    program_slug: str | None = None,
    no_music: bool = False,
) -> int:
    """Run the full pipeline. Returns exit code (0 = success, 1 = failure).

    If program_slug is set, uses the library path resolver for output and
    records the episode in the catalog. Otherwise uses legacy output/ paths.
    If no_music is True, skip all music overlays (voice-only output).
    """
    out = Path("output")
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")

    print(f"\n{'=' * 50}")
    print(f"  Agent Radio — {date_str}")
    print(f"{'=' * 50}\n")

    try:
        config = load_config(config_path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    # Library-aware path resolution
    ep_dir: Path | None = None
    r2_key: str | None = None
    lib_root: Path | None = None
    catalog = None
    if program_slug:
        from src.library import Catalog
        from src.paths import LibraryPaths

        lib_paths = LibraryPaths(Path(config.library.root))
        lib_root = lib_paths.root
        lib_paths.ensure_structure()
        lib_paths.ensure_program(program_slug, "talk")
        ep_dir = lib_paths.episode_dir(program_slug, date_str)
        r2_key = lib_paths.r2_episode_key(program_slug, date_str)
        catalog = Catalog(lib_paths.db)
        # Ensure program exists in catalog (idempotent)
        if not catalog.get_program(program_slug):
            catalog.register_program(program_slug, program_slug, "talk")
        print(f"  Library: {lib_paths.root} / {program_slug}")

    try:
        return _run_stages(
            config,
            out,
            date_str,
            ep_dir,
            r2_key,
            lib_root,
            catalog,
            program_slug,
            dry_run,
            no_music=no_music,
        )
    finally:
        if catalog:
            catalog.close()


def _run_stages(
    config,  # noqa: ANN001
    out: Path,
    date_str: str,
    ep_dir: Path | None,
    r2_key: str | None,
    lib_root: Path | None,
    catalog,  # noqa: ANN001
    program_slug: str | None,
    dry_run: bool,
    no_music: bool = False,
) -> int:
    """Run pipeline stages. Catalog cleanup handled by caller."""
    from src.curator import curate
    from src.renderer import render

    # Stage 1: Curate
    print("[1/4] Curating episode script...")
    try:
        script_path = curate(config, output_dir=out, episode_dir=ep_dir)
    except Exception as exc:
        print(f"ERROR in curator: {exc}")
        return 1

    # Stage 1.5: Script quality evaluation (pre-render gate)
    import json as _json

    from src.script_quality import evaluate_script

    print("\n[1.5/4] Evaluating script structure...")
    script_data: dict = {}
    try:
        script_data = _json.loads(script_path.read_text())
        script_report = evaluate_script(script_data)
        print(f"  Script score: {script_report.overall_score:.2f}")
        for note in script_report.notes:
            print(f"  - {note}")

        episode_dir = script_path.parent
        script_report_path = episode_dir / "script-quality.json"
        script_report_path.write_text(script_report.to_json())
        print(f"  Report saved: {script_report_path}")

        # Script anatomy visualization
        try:
            from src.visualize import render_script_anatomy

            anatomy_path = render_script_anatomy(
                script_data, episode_dir, score=script_report.overall_score
            )
            print(f"  Anatomy:  {anatomy_path}")
        except Exception as viz_exc:
            print(f"  WARNING: Script anatomy visualization failed: {viz_exc}")

        if script_report.overall_score < QUALITY_REVIEW:
            print(
                f"\n  HOLD — script score {script_report.overall_score:.2f} "
                f"below threshold ({QUALITY_REVIEW})"
            )
            print("  Script held for revision. Not rendering.")
            return 1
        elif script_report.overall_score < QUALITY_SHIP:
            print(
                f"\n  REVIEW — script score {script_report.overall_score:.2f} "
                f"below ship threshold ({QUALITY_SHIP})"
            )
            print("  Flagging for review but proceeding with render.")
    except Exception as exc:
        print(f"  WARNING: Script evaluation failed: {exc}")
        print("  Proceeding without script quality gate.")

    # Stage 2: Render
    print("\n[2/4] Rendering audio...")
    try:
        audio_path = render(
            config,
            script_path,
            output_dir=out,
            episode_dir=ep_dir,
            program_slug=program_slug,
            no_music=no_music,
        )
    except Exception as exc:
        print(f"ERROR in renderer: {exc}")
        return 1

    # Stage 2.5: Anomaly detection (post-render, pre-quality, non-blocking)
    try:
        from src.anomaly import detect_anomalies

        episode_dir = audio_path.parent
        manifest_path = episode_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            anomaly_report = detect_anomalies(manifest, per_segment_wer=[])
            anomalies_path = episode_dir / "anomalies.json"
            anomalies_path.write_text(json.dumps(anomaly_report.to_dict(), indent=2))
            count = len(anomaly_report.anomalies)
            if count == 0:
                print("  Anomalies: 0 — clean render.")
            else:
                print(f"  Anomalies: {count} flagged — see {anomalies_path}")
    except Exception as exc:  # noqa: BLE001 — anomaly stage is non-blocking
        print(f"  WARNING: Anomaly detection failed: {exc}")

    # Stage 3: Quality evaluation
    report = None
    quality_score = 0.0
    print("\n[3/4] Evaluating audio quality...")
    try:
        from src.quality import evaluate

        # Build flat script text for WER evaluation (Pillar 3)
        _wer_script_text: str | None = None
        if script_data.get("segments"):
            _wer_script_text = (
                " ".join(seg.get("text", "") for seg in script_data["segments"]) or None
            )

        report = evaluate(audio_path, script_text=_wer_script_text)
        quality_score = report.overall_score
        print(f"  Overall score: {quality_score:.2f}")
        for note in report.notes:
            print(f"  - {note}")

        # Save report in the episode bundle directory
        episode_dir = audio_path.parent
        report_path = episode_dir / "quality.json"
        report_path.write_text(report.to_json())
        print(f"  Report saved: {report_path}")
    except ImportError:
        print("  Quality evaluation unavailable (install with: uv sync --extra quality)")
        print("  Skipping quality gate.")
        quality_score = 1.0  # pass through if eval not installed
    except Exception as exc:
        print(f"  WARNING: Quality evaluation failed: {exc}")
        print("  Proceeding without quality gate.")
        quality_score = 1.0  # don't block on eval failure

    # Quality gate
    if quality_score < QUALITY_REVIEW:
        print(
            f"\n  HOLD — quality score {quality_score:.2f} below review threshold ({QUALITY_REVIEW})"
        )
        print("  Episode held for human review. Not distributing.")
        dry_run = True
    elif quality_score < QUALITY_SHIP:
        print(
            f"\n  REVIEW — quality score {quality_score:.2f} below ship threshold ({QUALITY_SHIP})"
        )
        print("  Episode flagged for human review before distribution.")
        dry_run = True

    # Episode history (cross-episode learning — non-blocking)
    try:
        from src.episode_history import EpisodeSummary, append_episode

        _script_score: float | None = None
        _script_structure: dict = {}
        try:
            _script_score = script_report.overall_score  # type: ignore[possibly-undefined]
            _script_structure = getattr(script_report, "dimension_scores", {}) or {}  # type: ignore[possibly-undefined]
        except NameError:
            pass

        ep_summary = EpisodeSummary(
            date=date_str,
            overall_score=quality_score,
            speaker_scores=getattr(report, "speaker_scores", {}),
            chemistry_score=getattr(report, "chemistry_score", 0.0),
            segment_count=len(script_data.get("segments", [])),
            total_duration_s=getattr(report, "duration_seconds", 0.0),
            mean_dnsmos=(report.dnsmos_ovr if report.dnsmos_ovr > 0 else None)
            if report is not None
            else None,
            mean_wer=(report.wer if report.wer >= 0 else None) if report is not None else None,
            script_score=_script_score,
            script_structure=_script_structure,
        )
        history_path = Path("data/episode-history.jsonl")
        append_episode(ep_summary, history_path)
        print(f"  Episode logged: {history_path}")
    except Exception as exc:
        print(f"  WARNING: Episode history logging failed: {exc}")

    # Record episode in catalog (before distribution, so we track even dry-runs)
    episode_id: int | None = None
    if catalog and program_slug:
        try:
            episode_id = catalog.add_episode(
                program_slug=program_slug,
                date=date_str,
                file_path=str(audio_path),
                duration_seconds=getattr(report, "duration_seconds", None),
                quality_score=quality_score,
                segment_count=len(script_data.get("segments", [])),
                script_path=str(script_path),
                manifest_path=str(script_path.parent / "manifest.json"),
            )
            print(f"  Cataloged: episode #{episode_id}")
        except Exception as exc:
            print(f"  WARNING: Catalog recording failed: {exc}")

    # Stage 4: Distribute
    if dry_run:
        print("\n[4/4] Distribute skipped (--dry-run or quality gate)")
        post_url = f"[not distributed] {audio_path}"
    else:
        from src.distributor import distribute

        print("\n[4/4] Distributing episode...")
        try:
            post_url = distribute(
                config,
                audio_path,
                script_path,
                r2_key_override=r2_key,
                library_root=lib_root,
            )
        except Exception as exc:
            print(f"ERROR in distributor: {exc}")
            return 1

        # Record distribution in catalog — walk through lifecycle states
        if catalog and episode_id:
            try:
                catalog.record_distribution("episode", episode_id, "r2", url=post_url)
                # Auto-approve pipeline-distributed episodes
                catalog.set_episode_status(episode_id, "reviewed")
                catalog.set_episode_status(episode_id, "approved")
                catalog.set_episode_status(episode_id, "distributed")
            except Exception as exc:
                print(f"  WARNING: Distribution tracking failed: {exc}")

    # Stage 4b: Stream to AzuraCast (non-blocking)
    stream_url = ""
    has_azuracast = (
        config.stream.enabled
        and bool(config.stream.base_url)
        and bool(config.stream.api_key)
        and bool(config.stream.playlist_name)
    )
    if has_azuracast and not dry_run:
        from src.stream import AzuraCastConfig, update_episode

        print("\n[4b] Streaming to AzuraCast...")
        try:
            az_config = AzuraCastConfig(
                base_url=config.stream.base_url,
                api_key=config.stream.api_key,
                station_id=config.stream.station_id,
            )
            metadata = {"title": script_data.get("title", ""), "artist": "Agent Radio"}
            media_path = update_episode(
                az_config, audio_path, config.stream.playlist_name, metadata
            )
            stream_url = f"{config.stream.base_url}/public/{config.stream.station_id}"
            print(f"  Uploaded to station: {stream_url}")
            print(f"  Media path: {media_path}")
        except Exception as exc:
            print(f"  WARNING: AzuraCast streaming failed: {exc}")
            print("  Episode distributed to R2. Streaming is non-blocking.")
    elif has_azuracast and dry_run:
        print("\n[4b] AzuraCast streaming skipped (--dry-run)")

    print(f"\n{'=' * 50}")
    print("  Done")
    print(f"  Script:  {script_path}")
    print(f"  Audio:   {audio_path}")
    print(f"  Quality: {quality_score:.2f}")
    print(f"  Post:    {post_url}")
    if stream_url:
        print(f"  Stream:  {stream_url}")
    print(f"{'=' * 50}\n")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent Radio — daily digest pipeline")
    parser.add_argument("--config", default="config/radio.yaml", help="Path to radio.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run curator + renderer + eval only, skip distribution",
    )
    parser.add_argument(
        "--program",
        default=None,
        help="Program slug (e.g. haystack-news) — uses library paths + catalog",
    )
    parser.add_argument(
        "--no-music",
        action="store_true",
        help="Skip all music overlays — voice-only output",
    )
    args = parser.parse_args()
    sys.exit(
        run(
            config_path=args.config,
            dry_run=args.dry_run,
            program_slug=args.program,
            no_music=args.no_music,
        )
    )


if __name__ == "__main__":
    main()
