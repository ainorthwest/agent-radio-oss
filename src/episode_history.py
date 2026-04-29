"""Cross-episode learning: quality tracking, voice drift, and pattern analysis.

Persistent JSONL storage for episode summaries. Enables the Steward to
detect quality trends, voice drift, and correlate script patterns with
production quality across episodes.

Usage:
    from src.episode_history import append_episode, load_history, detect_voice_drift

    summary = extract_summary(episode_report, script_report, production_report)
    append_episode(summary, Path("data/episode-history.jsonl"))
    history = load_history(Path("data/episode-history.jsonl"))

    # CLI:
    python -m src.episode_history data/episode-history.jsonl
    python -m src.episode_history data/episode-history.jsonl --drift host_a
    python -m src.episode_history data/episode-history.jsonl --json
    python -m src.episode_history data/episode-history.jsonl --viz output/viz/
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EpisodeSummary:
    """Compact summary of one episode for cross-episode tracking."""

    date: str
    overall_score: float
    speaker_scores: dict[str, float]
    chemistry_score: float
    production_score: float | None = None
    script_score: float | None = None
    mean_dnsmos: float | None = None
    mean_wer: float | None = None
    segment_count: int = 0
    total_duration_s: float = 0.0
    voice_fingerprints: dict[str, dict[str, float]] = field(default_factory=dict)
    script_structure: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ── Storage ──────────────────────────────────────────────────────────────────


def append_episode(summary: EpisodeSummary, path: Path) -> None:
    """Append one episode summary as a JSONL record. Creates file if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(summary.to_json() + "\n")


def load_history(path: Path) -> list[EpisodeSummary]:
    """Load all episode summaries from a JSONL file, sorted by date.

    Skips corrupt lines with a warning rather than crashing.
    """
    if not path.exists():
        return []

    summaries: list[EpisodeSummary] = []
    for line_num, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            summaries.append(EpisodeSummary(**data))
        except (json.JSONDecodeError, TypeError) as exc:
            print(f"  WARNING: Skipping corrupt line {line_num}: {exc}", file=sys.stderr)

    summaries.sort(key=lambda s: s.date)
    return summaries


# ── Extraction ───────────────────────────────────────────────────────────────


def extract_summary(
    episode_report: Any | None = None,
    script_report: Any | None = None,
    production_report: Any | None = None,
) -> EpisodeSummary:
    """Build an EpisodeSummary from evaluation reports.

    Accepts any combination of reports — missing reports contribute None
    for their fields. Designed to work with:
      - episode_report: src.quality.EpisodeReport
      - script_report: src.script_quality.ScriptReport
      - production_report: src.production_dashboard.ProductionReport
    """
    date = "unknown"
    overall = 0.0
    speaker_scores: dict[str, float] = {}
    chemistry = 0.0
    segment_count = 0
    total_duration = 0.0
    voice_fingerprints: dict[str, dict[str, float]] = {}
    mean_dnsmos: float | None = None
    mean_wer: float | None = None

    if episode_report is not None:
        date = getattr(episode_report, "episode_date", "unknown")
        overall = getattr(episode_report, "overall_score", 0.0)

        sr = getattr(episode_report, "speaker_reports", {})
        for speaker, report in sr.items():
            speaker_scores[speaker] = report.score
            # Voice fingerprint: key spectral features for drift detection
            mf = getattr(report, "mean_features", {})
            voice_fingerprints[speaker] = {
                k: mf[k]
                for k in [
                    "spectral_centroid_mean",
                    "pitch_variance",
                    "pitch_range_normalized",
                    "lufs_approx",
                    "speech_rate_variation",
                ]
                if k in mf
            }

        chem = getattr(episode_report, "chemistry", None)
        if chem is not None:
            chemistry = getattr(chem, "overall_chemistry", 0.0)

        segs = getattr(episode_report, "segment_reports", [])
        segment_count = len(segs)
        total_duration = sum(s.duration_seconds for s in segs)

        # Extract DNSMOS and WER from segment features
        dnsmos_vals = [
            s.features.get("dnsmos_ovr", 0.0) for s in segs if s.features.get("dnsmos_ovr", 0.0) > 0
        ]
        if dnsmos_vals:
            mean_dnsmos = sum(dnsmos_vals) / len(dnsmos_vals)

        wer_vals = [s.features.get("wer", -1.0) for s in segs if s.features.get("wer", -1.0) >= 0]
        if wer_vals:
            mean_wer = sum(wer_vals) / len(wer_vals)

    production_score: float | None = None
    if production_report is not None:
        production_score = getattr(production_report, "overall_score", None)

    script_score: float | None = None
    script_structure: dict[str, Any] = {}
    if script_report is not None:
        script_score = getattr(script_report, "overall_score", None)
        # Extract structure features for pattern analysis
        scores = getattr(script_report, "dimension_scores", {})
        if isinstance(scores, dict):
            script_structure = {k: v for k, v in scores.items() if isinstance(v, (int, float))}

    return EpisodeSummary(
        date=date,
        overall_score=overall,
        speaker_scores=speaker_scores,
        chemistry_score=chemistry,
        production_score=production_score,
        script_score=script_score,
        mean_dnsmos=mean_dnsmos,
        mean_wer=mean_wer,
        segment_count=segment_count,
        total_duration_s=round(total_duration, 2),
        voice_fingerprints=voice_fingerprints,
        script_structure=script_structure,
    )


# ── Analysis ─────────────────────────────────────────────────────────────────


def detect_voice_drift(
    history: list[EpisodeSummary],
    speaker: str,
    window: int = 5,
) -> dict[str, float]:
    """Detect voice characteristic drift for a speaker across recent episodes.

    Returns coefficient of variation (CV) per feature across the last `window`
    episodes. High CV (>0.15) suggests the voice is drifting.
    """
    # Collect fingerprints for this speaker
    fingerprints: list[dict[str, float]] = []
    for ep in history[-window:]:
        if speaker in ep.voice_fingerprints:
            fingerprints.append(ep.voice_fingerprints[speaker])

    if len(fingerprints) < 2:
        return {}

    # Compute CV per feature
    drift: dict[str, float] = {}
    all_keys: set[str] = set()
    for fp in fingerprints:
        all_keys.update(fp.keys())

    for key in sorted(all_keys):
        values = [fp[key] for fp in fingerprints if key in fp]
        if len(values) < 2:
            continue
        mean = sum(values) / len(values)
        if abs(mean) < 1e-10:
            drift[key] = 0.0
            continue
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance**0.5
        drift[key] = round(std / abs(mean), 4)

    return drift


def score_trend(
    history: list[EpisodeSummary],
    metric: str = "overall_score",
    window: int = 10,
) -> dict[str, Any]:
    """Analyze score trends across recent episodes.

    Returns:
        mean, std, slope, latest, best, worst for the metric.
    """
    values: list[float] = []
    for ep in history[-window:]:
        val = getattr(ep, metric, None)
        if val is not None and isinstance(val, (int, float)):
            values.append(float(val))

    if not values:
        return {"mean": 0.0, "std": 0.0, "slope": 0.0, "latest": 0.0, "best": 0.0, "worst": 0.0}

    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = variance**0.5

    # Simple linear regression slope
    if n >= 2:
        x_mean = (n - 1) / 2.0
        numerator = sum((i - x_mean) * (v - mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator > 0 else 0.0
    else:
        slope = 0.0

    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "slope": round(slope, 4),
        "latest": round(values[-1], 4),
        "best": round(max(values), 4),
        "worst": round(min(values), 4),
    }


def find_effective_patterns(
    history: list[EpisodeSummary],
    min_episodes: int = 5,
) -> dict[str, Any]:
    """Correlate script structure features with episode quality scores.

    Returns Pearson correlation per script structure dimension vs overall_score.
    Positive correlation = this structure feature predicts higher quality.
    """
    if len(history) < min_episodes:
        return {
            "status": "insufficient_data",
            "episodes": len(history),
            "min_required": min_episodes,
        }

    # Collect paired data: (structure features, overall_score)
    all_structure_keys: set[str] = set()
    for ep in history:
        all_structure_keys.update(ep.script_structure.keys())

    correlations: dict[str, float] = {}
    for key in sorted(all_structure_keys):
        pairs: list[tuple[float, float]] = []
        for ep in history:
            if key in ep.script_structure:
                pairs.append((float(ep.script_structure[key]), ep.overall_score))

        if len(pairs) < min_episodes:
            continue

        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        n = len(pairs)
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n

        cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / n
        x_std = (sum((x - x_mean) ** 2 for x in xs) / n) ** 0.5
        y_std = (sum((y - y_mean) ** 2 for y in ys) / n) ** 0.5

        if x_std > 1e-10 and y_std > 1e-10:
            correlations[key] = round(cov / (x_std * y_std), 4)

    return {"correlations": correlations, "episodes_analyzed": len(history)}


# ── Visualization ────────────────────────────────────────────────────────────


def render_quality_trend(
    history: list[EpisodeSummary],
    output_dir: Path,
) -> Path | None:
    """Render quality score timeline with rolling average and per-speaker lines.

    Returns path to PNG, or None if insufficient data.
    """
    if len(history) < 2:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_overall, ax_speakers) = plt.subplots(2, 1, figsize=(14, 8), facecolor="#1a1a1a")

    dates = [ep.date for ep in history]
    scores = [ep.overall_score for ep in history]
    x = range(len(dates))

    # Panel 1: Overall score timeline with rolling average
    ax_overall.set_facecolor("#2a2a2a")
    ax_overall.plot(x, scores, "o-", color="#4CAF50", linewidth=1.5, markersize=4, label="Overall")

    # Rolling average (window=3 or all if fewer)
    win = min(3, len(scores))
    if win >= 2:
        rolling = []
        for i in range(len(scores)):
            start = max(0, i - win + 1)
            rolling.append(sum(scores[start : i + 1]) / (i - start + 1))
        ax_overall.plot(
            x, rolling, "--", color="#FFC107", linewidth=2, label=f"Rolling avg ({win})"
        )

    ax_overall.set_ylabel("Score", color="#e0e0e0", fontsize=11)
    ax_overall.set_title("Episode Quality Trend", color="#e0e0e0", fontsize=13, fontweight="bold")
    ax_overall.legend(facecolor="#333333", edgecolor="#555555", labelcolor="#e0e0e0")
    ax_overall.tick_params(colors="#e0e0e0")
    ax_overall.set_ylim(0, 1.05)
    ax_overall.set_xticks(list(x))
    ax_overall.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
    for spine in ax_overall.spines.values():
        spine.set_color("#555555")

    # Panel 2: Per-speaker score lines
    speaker_colors = {
        "host_a": "#2196F3",
        "host_b": "#FF5722",
        "host_c": "#4CAF50",
        "host_d": "#FFC107",
    }
    ax_speakers.set_facecolor("#2a2a2a")

    all_speakers: set[str] = set()
    for ep in history:
        all_speakers.update(ep.speaker_scores.keys())

    for speaker in sorted(all_speakers):
        sp_scores = [ep.speaker_scores.get(speaker) for ep in history]
        valid_x = [i for i, s in enumerate(sp_scores) if s is not None]
        valid_y = [s for s in sp_scores if s is not None]
        if valid_y:
            color = speaker_colors.get(speaker, "#888888")
            ax_speakers.plot(
                valid_x, valid_y, "o-", color=color, linewidth=1.5, markersize=3, label=speaker
            )

    ax_speakers.set_ylabel("Score", color="#e0e0e0", fontsize=11)
    ax_speakers.set_xlabel("Episode", color="#e0e0e0", fontsize=11)
    ax_speakers.set_title("Per-Speaker Scores", color="#e0e0e0", fontsize=13, fontweight="bold")
    ax_speakers.legend(facecolor="#333333", edgecolor="#555555", labelcolor="#e0e0e0")
    ax_speakers.tick_params(colors="#e0e0e0")
    ax_speakers.set_ylim(0, 1.05)
    ax_speakers.set_xticks(list(x))
    ax_speakers.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
    for spine in ax_speakers.spines.values():
        spine.set_color("#555555")

    fig.tight_layout()
    out_path = output_dir / "quality-trend.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close(fig)
    return out_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def _format_report(history: list[EpisodeSummary]) -> str:
    """Format a human-readable summary of episode history."""
    lines = [f"Episode History: {len(history)} episodes\n"]

    if not history:
        lines.append("No episodes recorded.")
        return "\n".join(lines)

    # Overall trend
    trend = score_trend(history, "overall_score")
    lines.append("Overall Score Trend:")
    lines.append(
        f"  Latest: {trend['latest']:.2f}  Best: {trend['best']:.2f}  Worst: {trend['worst']:.2f}"
    )
    lines.append(
        f"  Mean: {trend['mean']:.2f}  Std: {trend['std']:.2f}  Slope: {trend['slope']:+.4f}"
    )

    direction = (
        "improving"
        if trend["slope"] > 0.005
        else "declining"
        if trend["slope"] < -0.005
        else "stable"
    )
    lines.append(f"  Trend: {direction}\n")

    # Per-speaker summary
    all_speakers: set[str] = set()
    for ep in history:
        all_speakers.update(ep.speaker_scores.keys())

    if all_speakers:
        lines.append("Speaker Averages:")
        for speaker in sorted(all_speakers):
            scores = [ep.speaker_scores[speaker] for ep in history if speaker in ep.speaker_scores]
            if scores:
                avg = sum(scores) / len(scores)
                lines.append(f"  {speaker}: {avg:.2f} (over {len(scores)} episodes)")
        lines.append("")

    # Latest episode
    latest = history[-1]
    lines.append(f"Latest Episode: {latest.date}")
    lines.append(f"  Score: {latest.overall_score:.2f}  Chemistry: {latest.chemistry_score:.2f}")
    lines.append(f"  Segments: {latest.segment_count}  Duration: {latest.total_duration_s:.1f}s")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent Radio episode history — cross-episode quality analysis"
    )
    parser.add_argument("history_file", help="Path to episode-history.jsonl")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--viz", metavar="DIR", help="Generate quality trend visualization")
    parser.add_argument("--drift", metavar="SPEAKER", help="Check voice drift for a speaker")
    parser.add_argument("--window", type=int, default=10, help="Analysis window size (default: 10)")
    args = parser.parse_args()

    path = Path(args.history_file)
    history = load_history(path)

    if not history:
        print(f"No episodes found in {path}")
        sys.exit(0)

    if args.json:
        output: dict[str, Any] = {
            "episodes": len(history),
            "trend": score_trend(history, "overall_score", args.window),
            "patterns": find_effective_patterns(history),
        }
        if args.drift:
            output["drift"] = detect_voice_drift(history, args.drift, args.window)
        print(json.dumps(output, indent=2))
    elif args.drift:
        drift = detect_voice_drift(history, args.drift, args.window)
        if not drift:
            print(f"No voice data for {args.drift} in last {args.window} episodes")
        else:
            print(f"Voice drift for {args.drift} (last {args.window} episodes):")
            for feature, cv in sorted(drift.items()):
                flag = " *** DRIFTING" if cv > 0.15 else ""
                print(f"  {feature}: CV={cv:.4f}{flag}")
    else:
        print(_format_report(history))

    if args.viz:
        viz_path = render_quality_trend(history, Path(args.viz))
        if viz_path:
            print(f"\nVisualization: {viz_path}")
        else:
            print("\nVisualization skipped (need 2+ episodes or matplotlib)")


if __name__ == "__main__":
    main()
