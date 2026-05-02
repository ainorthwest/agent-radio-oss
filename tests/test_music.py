"""Tests for the v0.1.0 music stub.

The stub exists so the mixer's import at ``src/mixer.py:658-680``
succeeds even though music generation is deferred to v0.1.1. These
tests pin the contract the mixer relies on and verify the
agent-experience message in the deferral exception.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.music import MusicAsset, MusicParams, generate


class TestMusicParams:
    """Mixer call sites depend on these field names + defaults."""

    def test_musicgen_call_shape_used_by_mixer(self) -> None:
        params = MusicParams(engine="musicgen", prompt="ambient piano")
        assert params.engine == "musicgen"
        assert params.prompt == "ambient piano"

    def test_midi_call_shape_used_by_mixer(self) -> None:
        params = MusicParams(type="sting", key="Am")
        assert params.type == "sting"
        assert params.key == "Am"

    def test_defaults_are_conservative(self) -> None:
        """Bare MusicParams() is still a defensible request shape."""
        params = MusicParams()
        assert params.engine == "none"
        assert params.prompt == ""
        assert params.type == "sting"
        assert params.key == "C"
        assert params.duration_s == 5.0
        assert params.seed is None
        assert params.extras == {}


class TestMusicAsset:
    """The mixer reads .path off the returned asset."""

    def test_construction(self, tmp_path: Path) -> None:
        params = MusicParams()
        asset = MusicAsset(path=tmp_path / "out.wav", type="sting", duration_s=2.0, params=params)
        assert asset.path == tmp_path / "out.wav"
        assert asset.type == "sting"
        assert asset.duration_s == 2.0
        assert asset.params is params


class TestGenerateDeferred:
    """v0.1.0 contract: generate() raises NotImplementedError with a
    message that points the operator at the v0.1.1 tracking issue and
    the workaround. Mixer catches this exception type specifically."""

    def test_raises_not_implemented(self, tmp_path: Path) -> None:
        with pytest.raises(NotImplementedError):
            generate(MusicParams(), output_dir=tmp_path)

    def test_message_names_tracking_issue(self, tmp_path: Path) -> None:
        with pytest.raises(NotImplementedError) as exc_info:
            generate(MusicParams(), output_dir=tmp_path)
        assert "issues/9" in str(exc_info.value)

    def test_message_names_workaround(self, tmp_path: Path) -> None:
        """Without 'use a pre-rendered asset' guidance the operator is
        stuck — they need to know what to do, not just what failed."""
        with pytest.raises(NotImplementedError) as exc_info:
            generate(MusicParams(), output_dir=tmp_path)
        msg = str(exc_info.value).lower()
        assert "pre-rendered" in msg
        assert "program.yaml" in msg

    def test_accepts_extra_kwargs_from_mixer(self, tmp_path: Path) -> None:
        """Mixer passes midi_only=False on the MIDI path. The stub must
        accept arbitrary kwargs so the call site doesn't have to know
        which keyword the engine version cares about."""
        with pytest.raises(NotImplementedError):
            generate(MusicParams(), output_dir=tmp_path, midi_only=False)


class TestMixerHandlesDeferral:
    """End-to-end integration with the mixer's exception handler.

    The mixer at src/mixer.py:679-682 catches NotImplementedError
    specifically and surfaces an AX-friendly warning. This test
    verifies the exception type the stub raises matches what the
    mixer's handler expects.
    """

    def test_generate_raises_exception_type_mixer_catches(self, tmp_path: Path) -> None:
        """If this fails, the mixer's targeted handler will fall
        through to the generic Exception branch and the operator will
        get a less-helpful message."""
        try:
            generate(MusicParams(), output_dir=tmp_path)
        except NotImplementedError:
            pass
        else:
            pytest.fail("generate() did not raise NotImplementedError")
