"""Tests for pipeline library integration (Phase 2).

Tests that the pipeline correctly resolves paths through the library
when --program is specified, and falls back to legacy paths otherwise.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.config import LibraryConfig, RadioConfig


@pytest.fixture
def mock_config():
    """Minimal RadioConfig for testing pipeline wiring."""
    return MagicMock(
        spec=RadioConfig,
        library=LibraryConfig(root="library", db_name="radio.db"),
        stream=MagicMock(enabled=False),
    )


class TestLibraryConfig:
    def test_library_config_defaults(self):
        cfg = LibraryConfig()
        assert cfg.root == "library"
        assert cfg.db_name == "radio.db"

    def test_library_config_custom(self):
        cfg = LibraryConfig(root="/custom/path", db_name="custom.db")
        assert cfg.root == "/custom/path"
        assert cfg.db_name == "custom.db"


class TestRadioConfigLibrary:
    def test_radio_config_has_library(self):
        """RadioConfig should include library field with defaults."""
        from src.config import (
            CuratorConfig,
            DiscourseConfig,
            DistributorConfig,
            RendererConfig,
            StreamConfig,
        )

        config = RadioConfig(
            discourse=DiscourseConfig(base_url="", api_key="", api_username=""),
            curator=CuratorConfig(),
            renderer=RendererConfig(),
            distributor=DistributorConfig(),
            stream=StreamConfig(),
            voices={},
        )
        assert isinstance(config.library, LibraryConfig)
        assert config.library.root == "library"


class TestCuratorEpisodeDirOverride:
    def test_curate_signature_accepts_episode_dir(self):
        """Curator curate() should accept episode_dir parameter."""
        import inspect

        from src.curator import curate

        sig = inspect.signature(curate)
        assert "episode_dir" in sig.parameters
        assert sig.parameters["episode_dir"].default is None


class TestRendererEpisodeDirOverride:
    def test_render_segments_signature_accepts_episode_dir(self):
        """render_segments should accept episode_dir parameter."""
        import inspect

        from src.renderer import render, render_segments

        sig_segments = inspect.signature(render_segments)
        assert "episode_dir" in sig_segments.parameters

        sig_render = inspect.signature(render)
        assert "episode_dir" in sig_render.parameters


class TestDistributorR2KeyOverride:
    def test_distribute_signature_accepts_r2_key(self):
        """distribute should accept r2_key_override parameter."""
        import inspect

        from src.distributor import distribute

        sig = inspect.signature(distribute)
        assert "r2_key_override" in sig.parameters


class TestPodcastLibraryMode:
    def test_collect_episodes_legacy_mode(self, tmp_path):
        """Default behavior: scan output/episodes/ directory."""
        import json

        from src.podcast import collect_episodes

        # Create a legacy episode
        ep_dir = tmp_path / "episodes" / "2026-03-19"
        ep_dir.mkdir(parents=True)
        (ep_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "date": "2026-03-19",
                    "title": "Test Episode",
                    "segments": [{"duration_seconds": 100}],
                }
            )
        )
        (ep_dir / "episode.mp3").write_bytes(b"\x00" * 16000)

        entries = collect_episodes(episodes_dir=tmp_path / "episodes")
        assert len(entries) == 1
        assert entries[0].title == "Test Episode"
        assert "episodes/2026-03-19.mp3" in entries[0].guid

    def test_collect_episodes_library_mode(self, tmp_path):
        """Library mode: scan library/programs/*/episodes/ directories."""
        import json

        from src.podcast import collect_episodes

        # Create a library episode
        ep_dir = tmp_path / "programs" / "haystack-news" / "episodes" / "2026-03-19"
        ep_dir.mkdir(parents=True)
        (ep_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "date": "2026-03-19",
                    "title": "Haystack Episode",
                    "segments": [{"duration_seconds": 200}],
                }
            )
        )
        (ep_dir / "episode.mp3").write_bytes(b"\x00" * 32000)

        entries = collect_episodes(library_root=tmp_path)
        assert len(entries) == 1
        assert entries[0].title == "Haystack Episode"
        assert "programs/haystack-news/2026-03-19.mp3" in entries[0].guid

    def test_collect_episodes_library_multi_program(self, tmp_path):
        """Library mode with multiple programs."""
        import json

        from src.podcast import collect_episodes

        for prog in ["alpha-show", "beta-show"]:
            ep_dir = tmp_path / "programs" / prog / "episodes" / "2026-03-19"
            ep_dir.mkdir(parents=True)
            (ep_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "date": "2026-03-19",
                        "title": f"{prog} episode",
                        "segments": [{"duration_seconds": 60}],
                    }
                )
            )
            (ep_dir / "episode.mp3").write_bytes(b"\x00" * 8000)

        entries = collect_episodes(library_root=tmp_path)
        assert len(entries) == 2

    def test_collect_episodes_library_empty(self, tmp_path):
        """Library mode with no programs dir returns empty."""
        from src.podcast import collect_episodes

        entries = collect_episodes(library_root=tmp_path)
        assert entries == []


class TestPipelineRunSignature:
    def test_run_accepts_program_slug(self):
        """pipeline.run() should accept program_slug parameter."""
        import inspect

        from src.pipeline import run

        sig = inspect.signature(run)
        assert "program_slug" in sig.parameters
        # Default should be None (legacy mode)
        assert sig.parameters["program_slug"].default is None

    def test_run_accepts_no_distribute(self):
        """pipeline.run() should accept no_distribute (default False).

        Sprint Day 4 surfaced that the documented `radio run pipeline
        --no-distribute` command was missing — `--dry-run` was the only
        way to skip Stage 4, but that flag also affects earlier stages.
        no_distribute is the targeted skip.
        """
        import inspect

        from src.pipeline import run

        sig = inspect.signature(run)
        assert "no_distribute" in sig.parameters
        assert sig.parameters["no_distribute"].default is False


class TestPipelineDataQualityFixes:
    """Tests for PR2: pipeline data quality fixes."""

    def test_evaluate_receives_script_text(self):
        """evaluate() must be called with script_text for WER to fire."""
        import inspect

        from src.quality import evaluate

        sig = inspect.signature(evaluate)
        assert "script_text" in sig.parameters

    def test_script_report_has_dimension_scores(self):
        """ScriptReport.dimension_scores must be populated after evaluate_script()."""
        from src.script_quality import evaluate_script

        script = {
            "program": "test",
            "segments": [
                {"speaker": "host_a", "text": "Welcome to the show.", "register": "baseline"},
                {
                    "speaker": "host_b",
                    "text": "Thanks for having me today.",
                    "register": "baseline",
                },
                {"speaker": "host_a", "text": "Let's talk about AI.", "register": "emphasis"},
                {
                    "speaker": "host_b",
                    "text": "Absolutely, it's been quite a year.",
                    "register": "baseline",
                },
            ],
        }
        report = evaluate_script(script)
        assert isinstance(report.dimension_scores, dict)
        assert len(report.dimension_scores) > 0
        # All dimension scores must be floats in [0, 1]
        for k, v in report.dimension_scores.items():
            assert 0.0 <= v <= 1.0, f"dimension_scores[{k!r}] = {v} out of range"

    def test_script_report_dimension_scores_in_json(self):
        """dimension_scores must survive to_json() / from-JSON round-trip."""
        import json

        from src.script_quality import evaluate_script

        script = {
            "program": "test",
            "segments": [
                {"speaker": "host_a", "text": "Hello world.", "register": "baseline"},
            ],
        }
        report = evaluate_script(script)
        data = json.loads(report.to_json())
        assert "dimension_scores" in data
        assert isinstance(data["dimension_scores"], dict)

    def test_add_episode_duplicate_returns_existing_id(self):
        """add_episode on duplicate (program_slug, date) returns existing ID, no raise."""
        import tempfile
        from pathlib import Path

        from src.library import Catalog

        with tempfile.TemporaryDirectory() as tmp:
            cat = Catalog(Path(tmp) / "radio.db")
            cat.register_program("haystack-news", "Haystack News")
            first_id = cat.add_episode("haystack-news", "2026-04-01", "ep1.mp3")
            second_id = cat.add_episode("haystack-news", "2026-04-01", "ep2.mp3")
            assert first_id == second_id
            assert first_id > 0

    def test_add_episode_unregistered_program_raises(self):
        """add_episode with unregistered program_slug raises IntegrityError (FK violation)."""
        import sqlite3
        import tempfile
        from pathlib import Path

        import pytest

        from src.library import Catalog

        with tempfile.TemporaryDirectory() as tmp:
            cat = Catalog(Path(tmp) / "radio.db")
            # No register_program call — FK violation expected
            with pytest.raises(sqlite3.IntegrityError):
                cat.add_episode("ghost-program", "2026-04-01", "ep1.mp3")

    def test_episode_history_report_none_guard(self):
        """EpisodeSummary construction must not raise when report is None."""
        from src.episode_history import EpisodeSummary

        # Simulate what pipeline does when report is None
        report = None
        mean_dnsmos = (
            (report.dnsmos_ovr if report.dnsmos_ovr > 0 else None) if report is not None else None
        )
        mean_wer = (report.wer if report.wer >= 0 else None) if report is not None else None
        speaker_scores = getattr(report, "speaker_scores", {})
        chemistry_score = getattr(report, "chemistry_score", 0.0)

        ep = EpisodeSummary(
            date="2026-04-01",
            overall_score=0.0,
            speaker_scores=speaker_scores,
            chemistry_score=chemistry_score,
            mean_dnsmos=mean_dnsmos,
            mean_wer=mean_wer,
        )
        assert ep.mean_dnsmos is None
        assert ep.mean_wer is None
        assert ep.speaker_scores == {}
        assert ep.chemistry_score == 0.0


class TestPipelineErrorMessages:
    """Tests for Day 7 AX polish: friendlier failures on novel-situation paths."""

    def test_corrupt_script_json_returns_named_error(self, tmp_path, capsys):
        """A malformed script.json must surface a JSONDecodeError remedy + non-zero rc.

        Before Day 7 the JSON load was nested inside the script-quality try, so a
        corrupt file masqueraded as a "Script evaluation failed" warning and the
        pipeline tried to render anyway. The remedy must name the file path and
        the line/column so an operator (or agent harness) can act without
        reading a stack trace.
        """
        from src.pipeline import run

        # Minimal config + corrupt override script. We use script_override to
        # bypass the curator entirely and isolate the JSON-load failure path.
        config_path = tmp_path / "radio.yaml"
        config_path.write_text("library:\n  root: lib\nsecrets:\n  openrouter_api_key_env: NOPE\n")
        bad_script = tmp_path / "broken-script.json"
        bad_script.write_text('{"segments": [unclosed')

        rc = run(
            config_path=str(config_path),
            dry_run=True,
            script_override=bad_script,
            date_override="2026-05-02",
        )
        assert rc == 1
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "invalid JSON" in combined
        assert "Re-run the curator stage" in combined
        # Path of the corrupt file must be named so the operator can find it.
        assert "script.json" in combined

    def test_curator_exception_includes_class_name(self, tmp_path, capsys, monkeypatch):
        """When the curator raises, the error line must include the exception class name.

        Agent harnesses pattern-match on `[ConnectionError]` / `[TimeoutError]` /
        etc. without parsing prose. Before Day 7 the format was
        `ERROR in curator: {exc}` with no class, which was opaque.
        """
        from src.pipeline import run

        config_path = tmp_path / "radio.yaml"
        config_path.write_text("library:\n  root: lib\nsecrets:\n  openrouter_api_key_env: NOPE\n")

        def fake_curate(*_args, **_kwargs):
            raise ConnectionError("openrouter.ai resolved to nothing")

        monkeypatch.setattr("src.curator.curate", fake_curate)

        rc = run(config_path=str(config_path), dry_run=True, date_override="2026-05-02")
        assert rc == 1
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "[ConnectionError]" in combined
        assert "openrouter.ai resolved to nothing" in combined
