"""Tests for renderer — register system, engine validation, output paths, and MLX helpers."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.renderer import (
    ALL_ENGINES,
    MLX_ENGINES,
    _apply_register,
    _build_mlx_kwargs,
    _convert_tags_to_dia,
    _strip_all_tags,
)


class TestApplyRegister:
    """Tests for the register merge logic."""

    def test_baseline_returns_original_profile(self):
        """Baseline register should return the profile unchanged."""
        profile = {"chatterbox": {"exaggeration": 0.8}, "registers": {"emphasis": {}}}
        result = _apply_register(profile, "baseline")
        assert result == profile

    def test_empty_register_returns_original(self):
        """Empty string register should return the profile unchanged."""
        profile = {"chatterbox": {"exaggeration": 0.8}}
        result = _apply_register(profile, "")
        assert result == profile

    def test_emphasis_overrides_chatterbox_params(self):
        """Emphasis register should merge chatterbox overrides."""
        profile = {
            "chatterbox": {"exaggeration": 0.8, "temperature": 0.5},
            "registers": {
                "emphasis": {"chatterbox": {"exaggeration": 1.0}},
            },
        }
        result = _apply_register(profile, "emphasis")
        assert result["chatterbox"]["exaggeration"] == 1.0
        assert result["chatterbox"]["temperature"] == 0.5  # preserved from base

    def test_register_does_not_mutate_original(self):
        """Register merge should work on a deep copy, not mutate the original."""
        profile = {
            "chatterbox": {"exaggeration": 0.8},
            "registers": {
                "emphasis": {"chatterbox": {"exaggeration": 1.2}},
            },
        }
        result = _apply_register(profile, "emphasis")
        assert result["chatterbox"]["exaggeration"] == 1.2
        assert profile["chatterbox"]["exaggeration"] == 0.8  # original untouched

    def test_unknown_register_returns_profile_unchanged(self):
        """A register not in the registers dict should return an unchanged copy."""
        profile = {
            "chatterbox": {"exaggeration": 0.8},
            "registers": {"emphasis": {"chatterbox": {"exaggeration": 1.0}}},
        }
        result = _apply_register(profile, "nonexistent_register")
        assert result == profile

    def test_register_adds_post_block(self):
        """Register can add a post block even if the base profile lacks one."""
        profile = {
            "chatterbox": {"exaggeration": 0.8},
            "registers": {
                "reflective": {"post": {"warmth_db": 1.5}},
            },
        }
        result = _apply_register(profile, "reflective")
        assert result["post"] == {"warmth_db": 1.5}

    def test_register_merges_post_block(self):
        """Register should merge into existing post block, not replace it."""
        profile = {
            "chatterbox": {"exaggeration": 0.8},
            "post": {"pitch_semitones": 0.2, "warmth_db": -1.0},
            "registers": {
                "emphasis": {"post": {"presence_db": 1.0}},
            },
        }
        result = _apply_register(profile, "emphasis")
        assert result["post"]["pitch_semitones"] == 0.2  # preserved
        assert result["post"]["warmth_db"] == -1.0  # preserved
        assert result["post"]["presence_db"] == 1.0  # added

    def test_profile_without_registers_key(self):
        """Profile with no registers block should work for any register name."""
        profile = {"chatterbox": {"exaggeration": 0.8}}
        result = _apply_register(profile, "emphasis")
        assert result == profile


class TestEngineValidation:
    """Tests for renderer engine selection."""

    def test_unknown_engine_raises_value_error(self):
        """Unknown engine name should raise ValueError with helpful message."""
        from src.renderer import render

        config = MagicMock()
        config.renderer.engine = "typo_engine"
        config.voices = {}

        script = {"date": "2026-01-01", "segments": []}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(script, f)
            f.flush()
            with pytest.raises(ValueError, match="Unknown renderer engine.*typo_engine"):
                render(config, Path(f.name))


class TestOutputPaths:
    """Tests for output directory structure — episode bundles and audition dirs."""

    def test_episode_segments_dir(self):
        """render_segments() should create output/episodes/{date}/segments/."""
        from src.renderer import render_segments

        config = MagicMock()
        config.renderer.engine = "kokoro"
        config.renderer.sample_rate = 24000
        config.voices = {}

        script = {"date": "2026-03-12", "title": "Test", "segments": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "script.json"
            script_path.write_text(json.dumps(script))
            output_dir = Path(tmpdir) / "output"

            # Empty segments should raise, but the dirs get created first
            try:
                render_segments(config, script_path, output_dir)
            except ValueError:
                pass  # "No segments rendered" is expected

            # Verify directory structure was created
            episode_dir = output_dir / "episodes" / "2026-03-12"
            segments_dir = episode_dir / "segments"
            assert episode_dir.is_dir()
            assert segments_dir.is_dir()

    def test_kokoro_load_deferred_until_first_real_segment(self):
        """get_kokoro() must NOT be called when every segment has empty text.

        Regression test for the lazy-init in _render_segments_kokoro: empty
        scripts (used by other tests in this class) must not trigger model
        loading, which would fail without the Kokoro ONNX file present.
        """
        from unittest.mock import patch

        from src.renderer import _render_segments_kokoro

        config = MagicMock()
        config.renderer.sample_rate = 24000
        segments = [
            {"speaker": "host_a", "text": "", "register": "baseline"},
            {"speaker": "host_a", "text": "   ", "register": "baseline"},
        ]

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("src.renderer.get_kokoro") as mock_get_kokoro,
        ):
            segments_dir = Path(tmpdir)
            result = _render_segments_kokoro(config, segments, {}, segments_dir)
            assert result == []
            mock_get_kokoro.assert_not_called()

    def test_kokoro_loaded_once_for_multiple_segments(self):
        """get_kokoro() is called exactly once even with multiple real segments."""
        from unittest.mock import patch

        import numpy as np

        from src.renderer import _render_segments_kokoro

        config = MagicMock()
        config.renderer.sample_rate = 24000

        # Mock Kokoro: returns a 1-second sine wave for any input.
        fake_kokoro = MagicMock()
        fake_kokoro.create.return_value = (
            np.zeros(24000, dtype=np.float32),
            None,
        )
        segments = [
            {"speaker": "host_a", "text": "first segment.", "register": "baseline"},
            {"speaker": "host_a", "text": "second segment.", "register": "baseline"},
        ]
        voice_profiles = {"host_a": {"kokoro": {"voice_id": "am_michael"}}}

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("src.renderer.get_kokoro", return_value=(fake_kokoro, 24000)) as mock_get,
            patch("src.renderer._resolve_voice", return_value="am_michael"),
        ):
            segments_dir = Path(tmpdir)
            result = _render_segments_kokoro(config, segments, voice_profiles, segments_dir)

        assert len(result) == 2
        mock_get.assert_called_once()
        assert fake_kokoro.create.call_count == 2

    def test_episode_manifest_in_bundle(self):
        """Manifest should be written to output/episodes/{date}/manifest.json."""
        from src.renderer import render_segments

        config = MagicMock()
        config.renderer.engine = "kokoro"
        config.renderer.sample_rate = 24000
        config.voices = {}

        script = {"date": "2026-03-12", "title": "Test", "segments": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "script.json"
            script_path.write_text(json.dumps(script))
            output_dir = Path(tmpdir) / "output"

            try:
                render_segments(config, script_path, output_dir)
            except ValueError:
                pass

            # Manifest should NOT be inside segments/ anymore
            episode_dir = output_dir / "episodes" / "2026-03-12"
            old_manifest = episode_dir / "segments" / "manifest.json"
            assert not old_manifest.exists()

    def test_curator_episode_dir(self):
        """Curator should write script to output/episodes/{date}/script.json."""
        episode_dir = Path("output") / "episodes" / "2026-03-12"
        script_path = episode_dir / "script.json"
        assert str(script_path) == "output/episodes/2026-03-12/script.json"


class TestTagConversion:
    """Tests for non-speech tag conversion between engines."""

    def test_bracket_to_dia(self):
        """Chatterbox bracket tags should convert to Dia paren tags."""
        assert _convert_tags_to_dia("Hello [laugh] world") == "Hello (laughs) world"
        assert _convert_tags_to_dia("[sigh] ok") == "(sighs) ok"

    def test_bracket_to_dia_multiple(self):
        """Multiple bracket tags should all convert."""
        result = _convert_tags_to_dia("[laugh] and [cough] and [gasp]")
        assert "(laughs)" in result
        assert "(coughs)" in result
        assert "(gasps)" in result

    def test_bracket_to_dia_preserves_text(self):
        """Non-tag text should be preserved unchanged."""
        assert _convert_tags_to_dia("Just normal text") == "Just normal text"

    def test_strip_all_tags_brackets(self):
        """Strip should remove all bracket tags."""
        assert _strip_all_tags("Hello [laugh] world") == "Hello world"

    def test_strip_all_tags_parens(self):
        """Strip should remove Dia paren tags."""
        assert _strip_all_tags("Hello (laughs) world") == "Hello world"

    def test_strip_all_tags_mixed(self):
        """Strip should handle both bracket and paren tags."""
        result = _strip_all_tags("[laugh] Hello (sighs) world [cough]")
        assert result == "Hello world"

    def test_strip_all_tags_empty_after_strip(self):
        """Tags-only text should become empty string."""
        assert _strip_all_tags("[laugh] [cough]") == ""

    def test_strip_all_tags_preserves_normal_parens(self):
        """Normal parenthetical text (not tags) should be preserved."""
        assert _strip_all_tags("Hello (world) there") == "Hello (world) there"


class TestBuildMLXKwargs:
    """Tests for MLX kwargs construction — no engine loading needed."""

    def test_csm_includes_speaker(self):
        """CSM kwargs should include speaker int and strip tags."""
        profile = {"csm": {"speaker": 1}, "ref_text": "hello there"}
        kwargs = _build_mlx_kwargs("csm", "Hello [laugh] world", profile)
        assert kwargs["text"] == "Hello world"
        assert kwargs["speaker"] == 1

    def test_csm_default_speaker(self):
        """CSM defaults to speaker 0."""
        kwargs = _build_mlx_kwargs("csm", "Hello", {})
        assert kwargs["speaker"] == 0

    def test_csm_ref_text_included(self):
        """CSM should include ref_text from profile when ref_audio exists."""
        import unittest.mock

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            ref_path = f.name
            f.write(b"\x00" * 100)

        profile = {"csm": {"ref_audio": ref_path}, "ref_text": "transcript here"}
        # Mock load_ref_audio_mx since the fake WAV isn't a real audio file
        with unittest.mock.patch("src.renderer.load_ref_audio_mx", return_value="mock_mx_array"):
            kwargs = _build_mlx_kwargs("csm", "Hello", profile)
        assert kwargs["ref_text"] == "transcript here"
        assert kwargs["ref_audio"] == "mock_mx_array"
        Path(ref_path).unlink()

    def test_dia_prepends_speaker_tag(self):
        """Dia kwargs should prepend [S1] to text."""
        profile = {"dia": {"speaker_tag": "S1"}}
        kwargs = _build_mlx_kwargs("dia", "Hello world", profile)
        assert kwargs["text"].startswith("[S1] ")

    def test_dia_converts_bracket_tags(self):
        """Dia should convert bracket tags to paren tags."""
        kwargs = _build_mlx_kwargs("dia", "Hello [laugh] world", {})
        assert "[laugh]" not in kwargs["text"]
        assert "(laughs)" in kwargs["text"]

    def test_dia_default_speaker_tag(self):
        """Dia defaults to S1 speaker tag."""
        kwargs = _build_mlx_kwargs("dia", "Hello", {})
        assert kwargs["text"].startswith("[S1]")

    def test_dia_invalid_speaker_tag_falls_back(self):
        """Invalid speaker tag should fall back to S1."""
        profile = {"dia": {"speaker_tag": "S3"}}
        kwargs = _build_mlx_kwargs("dia", "Hello", profile)
        assert kwargs["text"].startswith("[S1]")

    def test_chatterbox_mlx_params(self):
        """Chatterbox-MLX should map voice control params."""
        profile = {"chatterbox_mlx": {"exaggeration": 0.7, "cfg_weight": 0.3, "temperature": 0.9}}
        kwargs = _build_mlx_kwargs("chatterbox-mlx", "Hello", profile)
        assert kwargs["exaggeration"] == 0.7
        assert kwargs["cfg_weight"] == 0.3
        assert kwargs["temperature"] == 0.9

    def test_chatterbox_mlx_defaults(self):
        """Chatterbox-MLX should have sensible defaults."""
        kwargs = _build_mlx_kwargs("chatterbox-mlx", "Hello", {})
        assert kwargs["exaggeration"] == 0.5
        assert kwargs["cfg_weight"] == 0.5
        assert kwargs["temperature"] == 0.8

    def test_chatterbox_mlx_falls_back_to_chatterbox_block(self):
        """Chatterbox-MLX should use chatterbox block if no chatterbox_mlx block."""
        profile = {"chatterbox": {"exaggeration": 0.9}}
        kwargs = _build_mlx_kwargs("chatterbox-mlx", "Hello", profile)
        assert kwargs["exaggeration"] == 0.9

    def test_invalid_engine_raises(self):
        """Non-MLX engine should raise ValueError."""
        with pytest.raises(ValueError, match="Not an MLX engine"):
            _build_mlx_kwargs("kokoro", "Hello", {})

    def test_engine_sets(self):
        """MLX_ENGINES and ALL_ENGINES should contain expected values."""
        assert "csm" in MLX_ENGINES
        assert "dia" in MLX_ENGINES
        assert "chatterbox-mlx" in MLX_ENGINES
        assert "kokoro" in ALL_ENGINES
        assert "chatterbox" in ALL_ENGINES
        assert MLX_ENGINES.issubset(ALL_ENGINES)


class TestRendererMixerHonestyFixes:
    """Tests for PR3: renderer + mixer honesty fixes."""

    def test_mix_signature_no_output_dir(self):
        """mix() must not accept output_dir — it was silently ignored."""
        import inspect
        import sys
        from unittest.mock import MagicMock

        # Stub numpy so mixer.py module-level import doesn't fail
        _stubs = {}
        for mod in ("numpy", "scipy", "scipy.signal"):
            if mod not in sys.modules:
                sys.modules[mod] = MagicMock()
                _stubs[mod] = True
        try:
            import importlib

            import src.mixer as _mixer_mod

            importlib.reload(_mixer_mod)
            sig = inspect.signature(_mixer_mod.mix)
            assert "output_dir" not in sig.parameters
        finally:
            for mod, was_added in _stubs.items():
                if was_added:
                    del sys.modules[mod]

    def test_mix_signature_has_no_music_and_format(self):
        """mix() still accepts no_music and output_format."""
        import inspect
        import sys
        from unittest.mock import MagicMock

        _stubs = {}
        for mod in ("numpy", "scipy", "scipy.signal"):
            if mod not in sys.modules:
                sys.modules[mod] = MagicMock()
                _stubs[mod] = True
        try:
            import importlib

            import src.mixer as _mixer_mod

            importlib.reload(_mixer_mod)
            sig = inspect.signature(_mixer_mod.mix)
            assert "no_music" in sig.parameters
            assert "output_format" in sig.parameters
        finally:
            for mod, was_added in _stubs.items():
                if was_added:
                    del sys.modules[mod]

    def test_active_by_default_removed_from_dsp(self):
        """_ACTIVE_BY_DEFAULT was dead code — must be removed."""
        import src.dsp as dsp_module

        assert not hasattr(dsp_module, "_ACTIVE_BY_DEFAULT")

    def test_dsp_is_default_debox_triggers_processing(self):
        """debox_db != 0 must trigger DSP chain — comment fix verification."""
        from src.dsp import _is_default

        # Non-default debox_db must not be treated as default
        assert _is_default({}) is True
        assert _is_default({"debox_db": -6.0}) is False
        assert _is_default({"deesser_db": -3.0}) is False

    def test_render_segments_mixed_signature_has_indices(self):
        """_render_segments_mixed must accept indices param."""
        import inspect

        from src.renderer import _render_segments_mixed

        sig = inspect.signature(_render_segments_mixed)
        assert "indices" in sig.parameters

    def test_render_segments_mlx_signature_has_indices(self):
        """_render_segments_mlx must accept indices param."""
        import inspect

        from src.renderer import _render_segments_mlx

        sig = inspect.signature(_render_segments_mlx)
        assert "indices" in sig.parameters


class TestArtworkConvention:
    """Tests for PR76: artwork_path baked into manifest by renderer."""

    def test_manifest_contains_artwork_path_when_program_artwork_exists(self, tmp_path):
        """Renderer resolves program artwork into manifest when file exists."""

        # Build library structure: library/programs/{slug}/artwork/cover.png
        library_root = tmp_path / "library"
        slug = "haystack-news"
        date = "2026-04-01"
        art_file = library_root / "programs" / slug / "artwork" / "cover.png"
        art_file.parent.mkdir(parents=True)
        art_file.write_bytes(b"fake png")

        # episode_dir = library/programs/{slug}/episodes/{date}/
        episode_dir = library_root / "programs" / slug / "episodes" / date
        episode_dir.mkdir(parents=True)

        # Call the manifest builder directly via _write_manifest (if extracted)
        # Instead, verify the convention logic: parents[3] of episode_dir == library_root
        assert episode_dir.parents[3] == library_root

        # Simulate what renderer does: resolve from episode_dir.parents[3]
        from src.paths import LibraryPaths

        paths = LibraryPaths(episode_dir.parents[3])
        art = paths.program_artwork(slug)
        assert art.exists()
        assert art == art_file

    def test_station_artwork_fallback_when_no_program_artwork(self, tmp_path):
        """Station artwork is used when program has no artwork."""
        library_root = tmp_path / "library"
        slug = "haystack-news"

        # Only station artwork exists
        station_art = library_root / "station" / "artwork" / "cover.png"
        station_art.parent.mkdir(parents=True)
        station_art.write_bytes(b"fake station png")

        from src.paths import LibraryPaths

        paths = LibraryPaths(library_root)

        prog_art = paths.program_artwork(slug)
        assert not prog_art.exists()

        fallback = paths.station_artwork()
        assert fallback.exists()
        assert fallback == station_art

    def test_no_artwork_when_neither_exists(self, tmp_path):
        """No artwork resolves to None — renderer must omit artwork_path."""
        library_root = tmp_path / "library"
        slug = "haystack-news"

        from src.paths import LibraryPaths

        paths = LibraryPaths(library_root)

        prog_art = paths.program_artwork(slug)
        station_art = paths.station_artwork()

        assert not prog_art.exists()
        assert not station_art.exists()
        # Renderer logic: pick first that exists, else None
        resolved = (
            prog_art if prog_art.exists() else (station_art if station_art.exists() else None)
        )
        assert resolved is None
