"""Show format sonic palette — music identity for radio shows.

Each show format defines its sonic palette: pre-rendered assets for
production use, MusicGen prompts for dynamic generation, and sampling
params. The Steward uses this to develop and refine show identity
through iterative generation and taste refinement.

Palette files: shows/{name}.yaml

Usage:
    from src.show_palette import load_palette, resolve_cue

    palette = load_palette("shows/haystack-news.yaml")
    intro_path = resolve_cue(palette, "intro")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ShowPalette:
    """Sonic identity definition for a radio show format."""

    name: str
    description: str = ""
    aesthetic: str = ""
    assets: dict[str, str] = field(default_factory=dict)
    prompts: dict[str, str] = field(default_factory=dict)
    musicgen: dict[str, Any] = field(default_factory=dict)
    durations: dict[str, float] = field(default_factory=dict)


def load_palette(palette_path: str | Path) -> ShowPalette:
    """Load a show palette from YAML.

    Args:
        palette_path: Path to the show palette YAML file.

    Returns:
        ShowPalette with all fields populated.

    Raises:
        FileNotFoundError: If the palette file doesn't exist.
        ValueError: If required 'name' field is missing.
    """
    path = Path(palette_path)
    if not path.exists():
        raise FileNotFoundError(f"Show palette not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    name = data.get("name")
    if not name:
        raise ValueError(f"Show palette missing required 'name' field: {path}")

    return ShowPalette(
        name=name,
        description=data.get("description", ""),
        aesthetic=data.get("aesthetic", ""),
        assets=data.get("assets", {}),
        prompts=data.get("prompts", {}),
        musicgen=data.get("musicgen", {}),
        durations=data.get("durations", {}),
    )


def resolve_cue(
    palette: ShowPalette,
    cue_type: str,
    generate_if_missing: bool = False,
    output_dir: Path = Path("assets/music/generated"),
) -> str | None:
    """Resolve a music cue from the palette.

    Priority: pre-rendered asset file > generate via MusicGen > None.

    Args:
        palette: The show palette to resolve from.
        cue_type: Type of cue — "intro", "outro", "sting", "bumper", "bed".
        generate_if_missing: If True and no asset file exists, generate via MusicGen.
        output_dir: Directory for generated assets.

    Returns:
        Path to the resolved WAV file, or None if unavailable.
    """
    # Check pre-rendered asset
    asset_path = palette.assets.get(cue_type)
    if asset_path and Path(asset_path).exists():
        return asset_path

    if not generate_if_missing:
        return None

    # Generate via MusicGen
    prompt = palette.prompts.get(cue_type)
    if not prompt:
        return None

    try:
        from src.musicgen_engine import MusicGenConfig, generate_music

        duration = palette.durations.get(cue_type, 5.0)
        mg_config = palette.musicgen

        config = MusicGenConfig(
            prompt=prompt,
            duration=duration,
            model=mg_config.get("model", "facebook/musicgen-stereo-medium"),
            temperature=mg_config.get("temperature", 1.0),
            top_k=mg_config.get("top_k", 250),
            cfg_coef=mg_config.get("cfg_coef", 3.0),
        )
        asset = generate_music(config, output_dir=output_dir)
        return str(asset.path)
    except Exception as exc:
        print(f"  WARNING: MusicGen generation failed for {cue_type}: {exc}")
        return None
