"""Render-anomaly detector — flags suspect segments after render.

Three classes of anomaly, ordered by how cheaply they detect:

1. **Silence** — segment is mostly silence, suggesting a TTS dropout
   (`check_silence`). Cheap: librosa RMS over frames.
2. **Duration** — rendered duration is far from what the text length
   predicts (`check_duration`). Cheap: text word count × avg word
   duration vs. WAV length.
3. **WER outlier** — round-trip WER is well above the episode median
   (`check_wer_outliers`). Requires whisper transcripts (Day 3a).

The orchestrator (`detect_anomalies`) runs all three over a manifest +
per-segment WER list and returns an :class:`AnomalyReport`. Each anomaly
carries a ``suggestion`` (regenerate / replace_text / manual_review) so
the autonomous-station agent can act without interpreting raw scores.

Outputs are advisory; the editor / decide-ship-review-reject skill
makes the final call. The detector never auto-fixes.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

# Anchor heuristics for duration check. Real speech sits around
# 130-180 wpm; we use 0.4-0.5s/word as the "expected" band and flag
# segments whose duration is <0.5x or >2x the expected range.
_AVG_WORD_S = 0.45
_DURATION_LOW_RATIO = 0.5
_DURATION_HIGH_RATIO = 2.0
_SILENCE_DEFAULT_THRESHOLD = 0.4
# Frames whose RMS is below this fraction of the segment's peak count
# as silent. Most VADs use 5-10% of peak; we deliberately use a lower
# floor (1%) because Kokoro's natural prosody includes very-quiet
# unvoiced fricatives we don't want to misclassify as silence. Combined
# with the 40% segment-level threshold above, false positives on clean
# Kokoro output stay rare. Tune per-engine if needed.
_SILENCE_FRAME_THRESHOLD = 0.01
_WER_FLOOR = 0.05
_WER_OUTLIER_RATIO = 2.0


@dataclass
class AnomalyReport:
    """List of flagged segments with suggested actions.

    Each entry is a dict: ``{index, check, severity, suggestion, **detail}``
    where ``detail`` is check-specific (e.g. ``ratio`` for silence,
    ``wer`` for WER outliers, ``actual``/``expected`` for duration).
    """

    anomalies: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"anomalies": list(self.anomalies)}


def check_silence(
    wav_path: Path, threshold: float = _SILENCE_DEFAULT_THRESHOLD
) -> dict[str, Any] | None:
    """Flag a segment whose silence ratio exceeds ``threshold``.

    Uses librosa RMS over short frames. A frame is "silent" if its RMS
    is below ~1% of the segment's peak. ``threshold`` is the maximum
    fraction of silent frames before the segment is flagged.
    """
    try:
        import librosa
        import numpy as np
    except ImportError:
        return None

    try:
        y, sr = librosa.load(str(wav_path), sr=None)
    except Exception:
        return None

    if len(y) == 0:
        return {
            "check": "silence",
            "severity": "high",
            "suggestion": "regenerate",
            "ratio": 1.0,
        }

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    peak = float(rms.max()) if rms.size else 0.0
    if peak == 0:
        return {
            "check": "silence",
            "severity": "high",
            "suggestion": "regenerate",
            "ratio": 1.0,
        }

    silent_frames = float(np.sum(rms < peak * _SILENCE_FRAME_THRESHOLD))
    ratio = silent_frames / len(rms)
    if ratio < threshold:
        return None
    return {
        "check": "silence",
        "severity": "high" if ratio >= 0.7 else "medium",
        "suggestion": "regenerate",
        "ratio": round(ratio, 3),
    }


def check_duration(text: str, actual_duration: float) -> dict[str, Any] | None:
    """Flag a segment whose duration is far from what the text predicts.

    Empty text returns None (nothing to compare). Otherwise compute
    ``expected = word_count * 0.45s`` and flag if actual is outside
    ``[0.5 * expected, 2.0 * expected]``.
    """
    text = text.strip()
    if not text:
        return None
    word_count = len(text.split())
    if word_count == 0:
        return None
    expected = word_count * _AVG_WORD_S
    low = expected * _DURATION_LOW_RATIO
    high = expected * _DURATION_HIGH_RATIO
    if actual_duration < low:
        return {
            "check": "duration",
            "severity": "short",
            "suggestion": "regenerate",
            "expected_seconds": round(expected, 2),
            "actual_seconds": round(actual_duration, 2),
        }
    if actual_duration > high:
        return {
            "check": "duration",
            "severity": "long",
            "suggestion": "regenerate",
            "expected_seconds": round(expected, 2),
            "actual_seconds": round(actual_duration, 2),
        }
    return None


def check_wer_outliers(per_segment_wer: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag segments whose WER is >2× the episode median (and > 0.05 absolute).

    Sentinel WER values (-1.0 = not computed) are excluded from the
    median. Returns a list of anomaly dicts (possibly empty). Caller
    appends them to the AnomalyReport.
    """
    valid = [s for s in per_segment_wer if s.get("wer", -1.0) >= 0.0]
    if len(valid) < 2:
        return []
    wers = [s["wer"] for s in valid]
    med = median(wers)
    cutoff = max(_WER_FLOOR, _WER_OUTLIER_RATIO * med)
    flagged: list[dict[str, Any]] = []
    for s in valid:
        if s["wer"] > cutoff:
            flagged.append(
                {
                    "index": s["index"],
                    "check": "wer_outlier",
                    "severity": "high" if s["wer"] >= 0.3 else "medium",
                    "suggestion": "replace_text",
                    "wer": s["wer"],
                    "episode_median_wer": round(med, 4),
                }
            )
    return flagged


def detect_anomalies(
    manifest: dict[str, Any],
    per_segment_wer: list[dict[str, Any]] | None = None,
) -> AnomalyReport:
    """Run all three checks across an episode manifest. Returns an AnomalyReport.

    ``manifest`` matches the renderer's manifest.json shape (must
    include ``segments`` and ``segments_dir``). ``per_segment_wer``
    is the list emitted by :func:`src.stt.round_trip_score` (or its
    ``per_segment`` field). Pass ``[]`` if WER hasn't been computed.
    """
    report = AnomalyReport()
    segments = manifest.get("segments", [])
    seg_dir = Path(manifest.get("segments_dir", ""))

    for seg in segments:
        idx = seg.get("index")
        text = str(seg.get("text", ""))
        duration = float(seg.get("duration_seconds", 0.0))
        wav_name = seg.get("file", "")
        wav_path = seg_dir / wav_name

        # Silence — only if WAV is present.
        if wav_path.exists():
            sil = check_silence(wav_path)
            if sil is not None:
                report.anomalies.append({"index": idx, **sil})
        else:
            print(
                f"[anomaly] segment {idx}: WAV missing at {wav_path}, skipping silence check",
                file=sys.stderr,
            )

        # Duration — text-based.
        dur = check_duration(text, duration)
        if dur is not None:
            report.anomalies.append({"index": idx, **dur})

    # WER outliers — across the whole episode.
    for entry in check_wer_outliers(per_segment_wer or []):
        report.anomalies.append(entry)

    return report
