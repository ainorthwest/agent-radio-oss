"""Editorial manifest support for post-production control.

The editorial manifest extends the renderer's segment manifest (v1) with
an `editorial` block that gives the Steward fine-grained control over the
final mix — segment-level volume, skip, gap overrides, music cues, and
global pacing adjustments.

This module provides parsing and application helpers. The mixer imports
these to honor editorial annotations when assembling the final episode.

Usage:
    from src.editorial import load_editorial, apply_segment_override

    editorial = load_editorial(manifest)
    for seg in manifest["segments"]:
        seg = apply_segment_override(seg, editorial)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SegmentOverride:
    """Per-segment editorial override."""

    volume_db: float | None = None
    skip: bool = False
    gap_after_seconds: float | None = None
    note: str = ""


@dataclass
class MusicCue:
    """A music cue to place in the mix."""

    type: str  # "transition" | "bed" | "sting"
    after_segment: int
    asset: str
    fade_in_s: float = 1.0
    fade_out_s: float = 1.5
    volume_db: float = 0.0


@dataclass
class PacingConfig:
    """Global pacing adjustments."""

    global_gap_multiplier: float = 1.0


@dataclass
class EditorialManifest:
    """Parsed editorial block from a v2 manifest."""

    segment_overrides: dict[int, SegmentOverride] = field(default_factory=dict)
    music_cues: list[MusicCue] = field(default_factory=list)
    pacing: PacingConfig = field(default_factory=PacingConfig)

    def has_overrides(self) -> bool:
        return bool(
            self.segment_overrides or self.music_cues or self.pacing.global_gap_multiplier != 1.0
        )


def load_editorial(manifest: dict[str, Any]) -> EditorialManifest:
    """Parse the editorial block from a manifest dict.

    Returns an EditorialManifest with defaults if no editorial block exists.
    Works with both v1 (no editorial) and v2 (with editorial) manifests.
    """
    editorial_data = manifest.get("editorial")
    if not editorial_data:
        return EditorialManifest()

    # Parse segment overrides
    overrides: dict[int, SegmentOverride] = {}
    for seg_idx_str, override_data in editorial_data.get("segment_overrides", {}).items():
        seg_idx = int(seg_idx_str)
        overrides[seg_idx] = SegmentOverride(
            volume_db=override_data.get("volume_db"),
            skip=bool(override_data.get("skip", False)),
            gap_after_seconds=override_data.get("gap_after_seconds"),
            note=str(override_data.get("note", "")),
        )

    # Parse music cues
    cues: list[MusicCue] = []
    for cue_data in editorial_data.get("music_cues", []):
        cues.append(
            MusicCue(
                type=str(cue_data.get("type", "transition")),
                after_segment=int(cue_data.get("after_segment", 0)),
                asset=str(cue_data.get("asset", "")),
                fade_in_s=float(cue_data.get("fade_in_s", 1.0)),
                fade_out_s=float(cue_data.get("fade_out_s", 1.5)),
                volume_db=float(cue_data.get("volume_db", 0.0)),
            )
        )

    # Parse pacing
    pacing_data = editorial_data.get("pacing", {})
    pacing = PacingConfig(
        global_gap_multiplier=float(pacing_data.get("global_gap_multiplier", 1.0)),
    )

    return EditorialManifest(
        segment_overrides=overrides,
        music_cues=cues,
        pacing=pacing,
    )


def should_skip_segment(index: int, editorial: EditorialManifest) -> bool:
    """Check if a segment should be skipped per editorial annotations."""
    override = editorial.segment_overrides.get(index)
    return override.skip if override else False


def get_volume_adjustment(index: int, editorial: EditorialManifest) -> float:
    """Get volume adjustment in dB for a segment. Returns 0.0 if no override."""
    override = editorial.segment_overrides.get(index)
    if override and override.volume_db is not None:
        return override.volume_db
    return 0.0


def get_gap_override(index: int, editorial: EditorialManifest) -> float | None:
    """Get gap duration override for after a segment. Returns None if no override."""
    override = editorial.segment_overrides.get(index)
    if override and override.gap_after_seconds is not None:
        return override.gap_after_seconds
    return None


def get_music_cues_after(segment_index: int, editorial: EditorialManifest) -> list[MusicCue]:
    """Get music cues that should play after a given segment index."""
    return [cue for cue in editorial.music_cues if cue.after_segment == segment_index]


def write_editorial_manifest(
    manifest_path: Path,
    editorial: EditorialManifest,
) -> Path:
    """Write editorial annotations into an existing manifest file.

    Reads the manifest, adds/replaces the editorial block, writes back.
    Sets version to 2 to indicate editorial annotations are present.
    """
    manifest = json.loads(manifest_path.read_text())
    manifest["version"] = 2

    editorial_dict: dict[str, Any] = {}

    if editorial.segment_overrides:
        overrides_dict: dict[str, Any] = {}
        for idx, override in editorial.segment_overrides.items():
            entry: dict[str, Any] = {}
            if override.volume_db is not None:
                entry["volume_db"] = override.volume_db
            if override.skip:
                entry["skip"] = True
            if override.gap_after_seconds is not None:
                entry["gap_after_seconds"] = override.gap_after_seconds
            if override.note:
                entry["note"] = override.note
            overrides_dict[str(idx)] = entry
        editorial_dict["segment_overrides"] = overrides_dict

    if editorial.music_cues:
        editorial_dict["music_cues"] = [
            {
                "type": cue.type,
                "after_segment": cue.after_segment,
                "asset": cue.asset,
                "fade_in_s": cue.fade_in_s,
                "fade_out_s": cue.fade_out_s,
                "volume_db": cue.volume_db,
            }
            for cue in editorial.music_cues
        ]

    if editorial.pacing.global_gap_multiplier != 1.0:
        editorial_dict["pacing"] = {
            "global_gap_multiplier": editorial.pacing.global_gap_multiplier,
        }

    manifest["editorial"] = editorial_dict

    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path
