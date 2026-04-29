"""Tests for show palette loading and cue resolution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.show_palette import ShowPalette, load_palette, resolve_cue

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def palette_yaml(tmp_path: Path) -> Path:
    """Create a minimal show palette YAML for testing."""
    palette = {
        "name": "Test Show",
        "description": "A test show palette",
        "aesthetic": "Electronic meets acoustic",
        "assets": {
            "intro": str(tmp_path / "intro.wav"),
            "outro": str(tmp_path / "outro.wav"),
            "sting": str(tmp_path / "sting.wav"),
        },
        "prompts": {
            "intro": "electronic intro theme",
            "outro": "gentle electronic outro",
            "sting": "quick transition sting",
            "bed": "ambient bed 20 seconds",
        },
        "musicgen": {
            "model": "facebook/musicgen-stereo-medium",
            "temperature": 1.0,
            "top_k": 250,
            "cfg_coef": 3.0,
        },
        "durations": {
            "intro": 5.0,
            "outro": 5.0,
            "sting": 3.0,
            "bed": 20.0,
        },
    }
    path = tmp_path / "test-show.yaml"
    with open(path, "w") as f:
        yaml.dump(palette, f)

    # Create the asset files so resolve_cue can find them
    for asset_path in palette["assets"].values():
        Path(asset_path).touch()

    return path


@pytest.fixture
def palette(palette_yaml: Path) -> ShowPalette:
    """Load the test palette."""
    return load_palette(palette_yaml)


# ── load_palette tests ───────────────────────────────────────────────────────


class TestLoadPalette:
    def test_load_valid_palette(self, palette: ShowPalette) -> None:
        assert palette.name == "Test Show"
        assert palette.description == "A test show palette"
        assert "intro" in palette.assets
        assert "intro" in palette.prompts
        assert palette.musicgen["model"] == "facebook/musicgen-stereo-medium"
        assert palette.durations["sting"] == 3.0

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_palette("nonexistent/palette.yaml")

    def test_missing_name_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        with open(path, "w") as f:
            yaml.dump({"description": "no name"}, f)

        with pytest.raises(ValueError, match="missing required 'name'"):
            load_palette(path)

    def test_empty_yaml_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("")

        with pytest.raises(ValueError, match="missing required 'name'"):
            load_palette(path)

    def test_minimal_palette(self, tmp_path: Path) -> None:
        path = tmp_path / "minimal.yaml"
        with open(path, "w") as f:
            yaml.dump({"name": "Minimal"}, f)

        palette = load_palette(path)
        assert palette.name == "Minimal"
        assert palette.assets == {}
        assert palette.prompts == {}
        assert palette.musicgen == {}
        assert palette.durations == {}


# ── resolve_cue tests ────────────────────────────────────────────────────────


class TestResolveCue:
    def test_prerendered_asset_found(self, palette: ShowPalette) -> None:
        result = resolve_cue(palette, "intro")
        assert result is not None
        assert "intro.wav" in result

    def test_unknown_cue_type_returns_none(self, palette: ShowPalette) -> None:
        result = resolve_cue(palette, "nonexistent")
        assert result is None

    def test_missing_asset_no_generate(self, palette: ShowPalette) -> None:
        # bed has a prompt but no pre-rendered asset
        result = resolve_cue(palette, "bed", generate_if_missing=False)
        assert result is None

    @pytest.mark.skip(
        reason="src.musicgen_engine is not in OSS — Stable Audio Open replacement lands Day 4"
    )
    @patch("src.musicgen_engine.generate_music")
    def test_generate_when_missing(
        self,
        mock_generate: MagicMock,
        palette: ShowPalette,
        tmp_path: Path,
    ) -> None:
        # Mock generate_music to return a fake asset
        mock_asset = MagicMock()
        mock_asset.path = tmp_path / "generated-bed.wav"
        mock_generate.return_value = mock_asset

        result = resolve_cue(palette, "bed", generate_if_missing=True, output_dir=tmp_path)

        assert result == str(mock_asset.path)
        mock_generate.assert_called_once()

    def test_asset_file_missing_on_disk(self, tmp_path: Path) -> None:
        palette = ShowPalette(
            name="Test",
            assets={"intro": str(tmp_path / "does-not-exist.wav")},
        )
        result = resolve_cue(palette, "intro")
        assert result is None

    def test_no_prompt_for_generation(self, tmp_path: Path) -> None:
        palette = ShowPalette(
            name="Test",
            prompts={},  # No prompts defined
        )
        result = resolve_cue(palette, "intro", generate_if_missing=True)
        assert result is None
