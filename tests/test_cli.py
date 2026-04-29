"""CLI test suite — verifies the OSS command groups wire correctly to backends.

Uses Typer's CliRunner with mocked backends. Each test verifies:
1. Correct backend function is called
2. Arguments are passed correctly
3. Exit code is correct
4. --json output mode works where applicable

NOTE: CLI commands use lazy imports (inside function bodies). Some backend
modules require optional extras (librosa, etc.). We inject fake modules via
sys.modules so the lazy imports succeed without the actual dependencies.

OSS scope: tests cover config, run, render, distribute, stream, library,
soundbooth, and the output helpers. The production-only commands
(eval, music, viz, wire, edit, write, run-newsroom, soundbooth-serve)
do not exist in this distribution and have no tests here.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.cli import State, app
from src.config import (
    CuratorConfig,
    DiscourseConfig,
    DistributorConfig,
    LibraryConfig,
    RadioConfig,
    RendererConfig,
    StreamConfig,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path) -> RadioConfig:
    """Build a minimal RadioConfig pointing at tmp_path for library root."""
    return RadioConfig(
        discourse=DiscourseConfig(base_url="https://example.com", api_key="k", api_username="u"),
        curator=CuratorConfig(api_key="k"),
        renderer=RendererConfig(),
        distributor=DistributorConfig(),
        stream=StreamConfig(base_url="https://radio.example.com", api_key="az-key", station_id=1),
        voices={},
        library=LibraryConfig(root=str(tmp_path / "library")),
    )


@pytest.fixture()
def tmp_config(tmp_path: Path) -> RadioConfig:
    cfg = _make_config(tmp_path)
    (tmp_path / "library").mkdir()
    return cfg


def _invoke(args: list[str], config: RadioConfig) -> Any:
    """Invoke CLI, injecting a fake config so no YAML/secrets are loaded."""
    with patch("src.cli.State.config", new_callable=lambda: property(lambda self: config)):
        return runner.invoke(app, args, catch_exceptions=False)


def _invoke_safe(args: list[str], config: RadioConfig) -> Any:
    """Invoke CLI, catching SystemExit for error-path tests."""
    with patch("src.cli.State.config", new_callable=lambda: property(lambda self: config)):
        return runner.invoke(app, args)


def _fake_module(name: str, **attrs: Any) -> ModuleType:
    """Create a fake module with the given attributes."""
    mod = ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# ---------------------------------------------------------------------------
# Root app
# ---------------------------------------------------------------------------


class TestRootApp:
    def test_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "agent-radio-oss" in result.output

    def test_no_args_shows_help(self) -> None:
        result = runner.invoke(app, [])
        assert "agent-radio-oss" in result.output

    def test_dropped_commands_absent(self) -> None:
        """Production-only command groups must not register in the OSS root app."""
        groups = {cmd.name for cmd in app.registered_groups}
        for missing in ("edit", "eval", "music", "viz", "wire", "write"):
            assert missing not in groups, f"command group {missing!r} leaked into OSS CLI"


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


class TestConfigCmd:
    def test_show_json(self, tmp_config: RadioConfig) -> None:
        result = _invoke(["--json", "config", "show"], tmp_config)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "discourse" in data

    def test_show_redacts_secrets(self, tmp_config: RadioConfig) -> None:
        result = _invoke(["--json", "config", "show"], tmp_config)
        data = json.loads(result.output)
        assert data["discourse"]["api_key"] == "***"

    def test_show_human(self, tmp_config: RadioConfig) -> None:
        result = _invoke(["config", "show"], tmp_config)
        assert result.exit_code == 0
        assert "discourse" in result.output

    def test_validate(self, tmp_config: RadioConfig) -> None:
        result = _invoke(["config", "validate"], tmp_config)
        assert result.exit_code == 0
        assert "config loads successfully" in result.output

    def test_engines(self, tmp_config: RadioConfig) -> None:
        with patch("src.engines.available_engines", return_value=["kokoro"]):
            result = _invoke(["config", "engines"], tmp_config)
        assert result.exit_code == 0
        assert "kokoro" in result.output

    def test_engines_json(self, tmp_config: RadioConfig) -> None:
        with patch("src.engines.available_engines", return_value=["kokoro"]):
            result = _invoke(["--json", "config", "engines"], tmp_config)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0]["id"] == "kokoro"

    def test_engines_none_found(self, tmp_config: RadioConfig) -> None:
        with patch("src.engines.available_engines", return_value=[]):
            result = _invoke(["config", "engines"], tmp_config)
        assert result.exit_code == 0
        assert "No TTS engines" in result.output


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


class TestRunCmd:
    def test_pipeline(self, tmp_config: RadioConfig) -> None:
        with patch("src.pipeline.run", return_value=0) as mock:
            result = _invoke(["run", "pipeline"], tmp_config)
        assert result.exit_code == 0
        mock.assert_called_once_with(
            config_path="config/radio.yaml",
            dry_run=False,
            program_slug=None,
            no_music=False,
        )

    def test_pipeline_with_flags(self, tmp_config: RadioConfig) -> None:
        with patch("src.pipeline.run", return_value=0) as mock:
            result = _invoke(
                ["--program", "haystack-news", "--dry-run", "--no-music", "run", "pipeline"],
                tmp_config,
            )
        assert result.exit_code == 0
        mock.assert_called_once_with(
            config_path="config/radio.yaml",
            dry_run=True,
            program_slug="haystack-news",
            no_music=True,
        )

    def test_pipeline_nonzero_exit(self, tmp_config: RadioConfig) -> None:
        with patch("src.pipeline.run", return_value=1):
            result = _invoke_safe(["run", "pipeline"], tmp_config)
        assert result.exit_code == 1

    def test_pipeline_json(self, tmp_config: RadioConfig) -> None:
        with patch("src.pipeline.run", return_value=0):
            result = _invoke(["--json", "run", "pipeline"], tmp_config)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True

    def test_newsroom_command_dropped(self, tmp_config: RadioConfig) -> None:
        """`radio run newsroom` is a production-only command and must not exist."""
        result = _invoke_safe(["run", "newsroom"], tmp_config)
        assert result.exit_code != 0

    def test_music_command_dropped(self, tmp_config: RadioConfig) -> None:
        """`radio run music` is a production-only command and must not exist."""
        result = _invoke_safe(["run", "music"], tmp_config)
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


class TestRenderCmd:
    def test_episode(self, tmp_path: Path, tmp_config: RadioConfig) -> None:
        script = tmp_path / "script.json"
        script.write_text('{"segments": []}')
        mock_renderer = _fake_module(
            "src.renderer", render=MagicMock(return_value=Path("/tmp/out.mp3"))
        )
        with patch.dict(sys.modules, {"src.renderer": mock_renderer}):
            result = _invoke(["render", "episode", str(script)], tmp_config)
        assert result.exit_code == 0
        mock_renderer.render.assert_called_once()
        assert mock_renderer.render.call_args.kwargs["script_path"] == script

    def test_episode_file_not_found(self, tmp_config: RadioConfig) -> None:
        result = _invoke_safe(["render", "episode", "/no/such/script.json"], tmp_config)
        assert result.exit_code == 1

    def test_audition(self, tmp_path: Path, tmp_config: RadioConfig) -> None:
        script = tmp_path / "audition.json"
        script.write_text('{"segments": []}')
        mock_renderer = _fake_module(
            "src.renderer",
            render_voice_audition=MagicMock(return_value=Path("/tmp/audition.wav")),
        )
        with patch.dict(sys.modules, {"src.renderer": mock_renderer}):
            result = _invoke(
                ["render", "audition", str(script), "--voice", "voices/test.yaml"],
                tmp_config,
            )
        assert result.exit_code == 0
        mock_renderer.render_voice_audition.assert_called_once_with(
            voice_profile_path="voices/test.yaml",
            script_path=script,
            experiment_path=None,
        )

    def test_remix(self, tmp_path: Path, tmp_config: RadioConfig) -> None:
        manifest = tmp_path / "manifest.json"
        manifest.write_text("{}")
        mock_mixer = _fake_module("src.mixer", mix=MagicMock(return_value=Path("/tmp/out.mp3")))
        with patch.dict(sys.modules, {"src.mixer": mock_mixer}):
            result = _invoke(["render", "remix", str(manifest)], tmp_config)
        assert result.exit_code == 0
        mock_mixer.mix.assert_called_once()

    def test_reference(self, tmp_path: Path, tmp_config: RadioConfig) -> None:
        script = tmp_path / "script.json"
        script.write_text("{}")
        mock_renderer = _fake_module(
            "src.renderer",
            generate_reference_clip=MagicMock(return_value=Path("/tmp/ref.wav")),
        )
        with patch.dict(sys.modules, {"src.renderer": mock_renderer}):
            result = _invoke(
                ["render", "reference", str(script), "--voice", "voices/kokoro.yaml"],
                tmp_config,
            )
        assert result.exit_code == 0
        mock_renderer.generate_reference_clip.assert_called_once_with(
            kokoro_profile_path="voices/kokoro.yaml",
            script_path=script,
            output_path=None,
        )

    def test_segment(self, tmp_path: Path, tmp_config: RadioConfig) -> None:
        """Verify render segment passes indices for surgical re-rendering."""
        manifest = tmp_path / "manifest.json"
        manifest.write_text("{}")
        script = tmp_path / "script.json"
        script.write_text('{"segments": [{"text": "Hello"}, {"text": "World"}]}')
        mock_renderer = _fake_module("src.renderer", render_segments=MagicMock())
        mock_mixer = _fake_module("src.mixer", mix=MagicMock(return_value=Path("/tmp/out.mp3")))
        with patch.dict(sys.modules, {"src.renderer": mock_renderer, "src.mixer": mock_mixer}):
            result = _invoke(
                ["render", "segment", str(manifest), "--indices", "0,1"],
                tmp_config,
            )
        assert result.exit_code == 0
        mock_renderer.render_segments.assert_called_once()
        # Verify indices are passed as a set for surgical re-rendering
        call_kwargs = mock_renderer.render_segments.call_args.kwargs
        assert call_kwargs["indices"] == {0, 1}


# ---------------------------------------------------------------------------
# distribute
# ---------------------------------------------------------------------------


class TestDistributeCmd:
    def test_episode(self, tmp_path: Path, tmp_config: RadioConfig) -> None:
        mp3 = tmp_path / "episode.mp3"
        mp3.write_bytes(b"audio")
        script = tmp_path / "script.json"
        script.write_text("{}")
        mock_dist = _fake_module(
            "src.distributor", distribute=MagicMock(return_value="https://example.com/post")
        )
        with (
            patch("src.cli.distribute_cmd.require_extra"),
            patch.dict(sys.modules, {"src.distributor": mock_dist}),
        ):
            result = _invoke(["distribute", "episode", str(mp3), str(script)], tmp_config)
        assert result.exit_code == 0
        mock_dist.distribute.assert_called_once()
        assert "example.com" in result.output

    def test_feed(self, tmp_config: RadioConfig) -> None:
        mock_podcast = _fake_module(
            "src.podcast", generate_feed=MagicMock(return_value=Path("/tmp/feed.xml"))
        )
        with patch.dict(sys.modules, {"src.podcast": mock_podcast}):
            result = _invoke(["distribute", "feed"], tmp_config)
        assert result.exit_code == 0
        mock_podcast.generate_feed.assert_called_once()

    def test_status(self, tmp_path: Path, tmp_config: RadioConfig) -> None:
        """Verifies distribute status queries episodes then distributions (bug #1 fix)."""
        mock_prog = MagicMock(slug="haystack-news")
        mock_ep = MagicMock(id=1, date="2026-03-20")
        mock_dist = MagicMock(destination="r2", url="https://r2.example.com/ep.mp3")

        mock_catalog = MagicMock()
        mock_catalog.list_programs.return_value = [mock_prog]
        mock_catalog.list_episodes.return_value = [mock_ep]
        mock_catalog.get_distributions.return_value = [mock_dist]
        mock_catalog.__enter__ = MagicMock(return_value=mock_catalog)
        mock_catalog.__exit__ = MagicMock(return_value=False)

        with (
            patch("src.library.Catalog", return_value=mock_catalog),
            patch("src.paths.LibraryPaths"),
        ):
            result = _invoke(["distribute", "status"], tmp_config)

        assert result.exit_code == 0
        # Verify get_distributions was called with int episode ID, not string slug
        mock_catalog.get_distributions.assert_called_once_with("episode", 1)

    def test_status_json(self, tmp_config: RadioConfig) -> None:
        mock_catalog = MagicMock()
        mock_catalog.list_programs.return_value = []
        mock_catalog.__enter__ = MagicMock(return_value=mock_catalog)
        mock_catalog.__exit__ = MagicMock(return_value=False)

        with (
            patch("src.library.Catalog", return_value=mock_catalog),
            patch("src.paths.LibraryPaths"),
        ):
            result = _invoke(["--json", "distribute", "status"], tmp_config)

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)


# ---------------------------------------------------------------------------
# stream
# ---------------------------------------------------------------------------


class TestStreamCmd:
    def test_status(self, tmp_config: RadioConfig) -> None:
        data = {
            "now_playing": {"song": {"title": "Test", "artist": "Bot"}},
            "listeners": {"total": 5},
        }
        with patch("src.stream.get_now_playing", return_value=data):
            result = _invoke(["stream", "status"], tmp_config)
        assert result.exit_code == 0
        assert "Test" in result.output

    def test_health(self, tmp_config: RadioConfig) -> None:
        data = {"backendRunning": True, "frontendRunning": True}
        with patch("src.stream.get_service_health", return_value=data):
            result = _invoke(["stream", "health"], tmp_config)
        assert result.exit_code == 0
        assert "running" in result.output

    def test_upload(self, tmp_path: Path, tmp_config: RadioConfig) -> None:
        audio = tmp_path / "track.mp3"
        audio.write_bytes(b"audio")
        with patch("src.stream.upload_media", return_value="/media/track.mp3") as mock:
            result = _invoke(["stream", "upload", str(audio), "--title", "My Track"], tmp_config)
        assert result.exit_code == 0
        mock.assert_called_once()

    def test_playlist(self, tmp_config: RadioConfig) -> None:
        playlists = [{"id": 1, "name": "Default", "is_enabled": True, "type": "default"}]
        with patch("src.stream.list_playlists", return_value=playlists):
            result = _invoke(["stream", "playlist"], tmp_config)
        assert result.exit_code == 0
        assert "Default" in result.output

    def test_schedule(self, tmp_config: RadioConfig) -> None:
        entries = [{"start": "20:00", "end": "06:00", "playlist": {"name": "Late Night"}}]
        with patch("src.stream.get_schedule", return_value=entries):
            result = _invoke(["stream", "schedule"], tmp_config)
        assert result.exit_code == 0
        assert "Late Night" in result.output

    def test_listeners(self, tmp_config: RadioConfig) -> None:
        with patch("src.stream.get_listeners", return_value=[]):
            result = _invoke(["stream", "listeners"], tmp_config)
        assert result.exit_code == 0
        assert "0" in result.output

    def test_history(self, tmp_config: RadioConfig) -> None:
        hist = [{"played_at": "2026-03-20 10:00", "song": {"title": "Intro", "artist": "Bot"}}]
        with patch("src.stream.get_history", return_value=hist):
            result = _invoke(["stream", "history"], tmp_config)
        assert result.exit_code == 0
        assert "Intro" in result.output

    def test_queue(self, tmp_config: RadioConfig) -> None:
        with patch("src.stream.get_queue", return_value=[]):
            result = _invoke(["stream", "queue"], tmp_config)
        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_update(self, tmp_path: Path, tmp_config: RadioConfig) -> None:
        audio = tmp_path / "episode.mp3"
        audio.write_bytes(b"audio")
        with patch("src.stream.update_episode", return_value="/media/ep.mp3") as mock:
            result = _invoke(
                ["stream", "update", str(audio), "--playlist", "Haystack News"],
                tmp_config,
            )
        assert result.exit_code == 0
        mock.assert_called_once()

    def test_status_json(self, tmp_config: RadioConfig) -> None:
        data = {"now_playing": None, "listeners": {"total": 0}}
        with patch("src.stream.get_now_playing", return_value=data):
            result = _invoke(["--json", "stream", "status"], tmp_config)
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "listeners" in parsed


# ---------------------------------------------------------------------------
# library
# ---------------------------------------------------------------------------


class TestLibraryCmd:
    def _mock_catalog(self, **method_returns: Any) -> MagicMock:
        catalog = MagicMock()
        for method, retval in method_returns.items():
            getattr(catalog, method).return_value = retval
        catalog.__enter__ = MagicMock(return_value=catalog)
        catalog.__exit__ = MagicMock(return_value=False)
        return catalog

    def test_programs(self, tmp_config: RadioConfig) -> None:
        @dataclass
        class FakeProg:
            slug: str = "haystack-news"
            name: str = "Haystack News"
            program_type: str = "talk"
            status: str = "active"

        catalog = self._mock_catalog(list_programs=[FakeProg()])
        with (
            patch("src.library.Catalog", return_value=catalog),
            patch("src.paths.LibraryPaths"),
        ):
            result = _invoke(["library", "programs"], tmp_config)
        assert result.exit_code == 0
        assert "haystack-news" in result.output

    def test_programs_empty(self, tmp_config: RadioConfig) -> None:
        catalog = self._mock_catalog(list_programs=[])
        with (
            patch("src.library.Catalog", return_value=catalog),
            patch("src.paths.LibraryPaths"),
        ):
            result = _invoke(["library", "programs"], tmp_config)
        assert result.exit_code == 0
        assert "No programs" in result.output

    def test_episodes(self, tmp_config: RadioConfig) -> None:
        @dataclass
        class FakeEp:
            id: int = 1
            program_slug: str = "haystack-news"
            date: str = "2026-03-20"
            quality_score: float = 0.75
            status: str = "approved"

        catalog = self._mock_catalog(list_episodes=[FakeEp()])
        with (
            patch("src.library.Catalog", return_value=catalog),
            patch("src.paths.LibraryPaths"),
        ):
            result = _invoke(["library", "episodes"], tmp_config)
        assert result.exit_code == 0
        assert "haystack-news" in result.output

    def test_tracks(self, tmp_config: RadioConfig) -> None:
        @dataclass
        class FakeTrack:
            id: int = 1
            program_slug: str = "late-night-lofi"
            title: str = "Ambient Dreams"
            duration_seconds: float = 30.0
            quality_score: float = 0.8

        catalog = self._mock_catalog(list_tracks=[FakeTrack()])
        with (
            patch("src.library.Catalog", return_value=catalog),
            patch("src.paths.LibraryPaths"),
        ):
            result = _invoke(["library", "tracks"], tmp_config)
        assert result.exit_code == 0
        assert "Ambient Dreams" in result.output

    def test_status(self, tmp_config: RadioConfig) -> None:
        prog = MagicMock(slug="haystack-news")
        catalog = self._mock_catalog(list_programs=[prog], list_episodes=[], list_tracks=[])
        with (
            patch("src.library.Catalog", return_value=catalog),
            patch("src.paths.LibraryPaths"),
        ):
            result = _invoke(["library", "status"], tmp_config)
        assert result.exit_code == 0
        assert "Programs: 1" in result.output

    def test_approve(self, tmp_config: RadioConfig) -> None:
        catalog = self._mock_catalog()
        with (
            patch("src.library.Catalog", return_value=catalog),
            patch("src.paths.LibraryPaths"),
        ):
            result = _invoke(["library", "approve", "42", "approved"], tmp_config)
        assert result.exit_code == 0
        catalog.set_episode_status.assert_called_once_with(42, "approved")

    def test_approve_invalid_status(self, tmp_config: RadioConfig) -> None:
        result = _invoke_safe(["library", "approve", "1", "invalid"], tmp_config)
        assert result.exit_code == 1

    def test_programs_json(self, tmp_config: RadioConfig) -> None:
        @dataclass
        class FakeProg:
            slug: str = "test"
            name: str = "Test"
            program_type: str = "talk"
            status: str = "active"

        catalog = self._mock_catalog(list_programs=[FakeProg()])
        with (
            patch("src.library.Catalog", return_value=catalog),
            patch("src.paths.LibraryPaths"),
        ):
            result = _invoke(["--json", "library", "programs"], tmp_config)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0]["slug"] == "test"


# ---------------------------------------------------------------------------
# soundbooth — OSS exposes only `engines` and `voices` (no `serve`)
# ---------------------------------------------------------------------------


class TestSoundboothCmd:
    def test_engines(self, tmp_config: RadioConfig) -> None:
        result = _invoke(["soundbooth", "engines"], tmp_config)
        assert result.exit_code == 0
        # OSS registers Kokoro by default
        assert "kokoro" in result.output

    def test_voices_no_dir(self, tmp_config: RadioConfig) -> None:
        result = _invoke(["soundbooth", "voices"], tmp_config)
        assert result.exit_code == 0

    def test_serve_command_dropped(self, tmp_config: RadioConfig) -> None:
        """`radio soundbooth serve` is production-only (FastAPI UI)."""
        result = _invoke_safe(["soundbooth", "serve"], tmp_config)
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


class TestOutputHelpers:
    def test_output_json_dict(self, capsys: pytest.CaptureFixture[str]) -> None:
        from src.cli._output import output

        state = State(json_output=True)
        output(state, {"key": "value"})
        data = json.loads(capsys.readouterr().out)
        assert data["key"] == "value"

    def test_output_json_dataclass(self, capsys: pytest.CaptureFixture[str]) -> None:
        @dataclass
        class Dummy:
            score: float = 0.5
            name: str = "test"

        from src.cli._output import output

        state = State(json_output=True)
        output(state, Dummy())
        data = json.loads(capsys.readouterr().out)
        assert data["score"] == 0.5

    def test_output_human_fmt(self, capsys: pytest.CaptureFixture[str]) -> None:
        from src.cli._output import output

        state = State(json_output=False)
        output(state, {"ignored": True}, human_fmt="hello world")
        assert "hello world" in capsys.readouterr().out

    def test_output_json_string(self, capsys: pytest.CaptureFixture[str]) -> None:
        from src.cli._output import output

        state = State(json_output=True)
        output(state, "plain string")
        data = json.loads(capsys.readouterr().out)
        assert data["result"] == "plain string"

    def test_err_exits(self) -> None:
        from src.cli._output import err

        with pytest.raises(SystemExit, match="1"):
            err("something broke")

    def test_require_extra_installed(self) -> None:
        from src.cli._output import require_extra

        require_extra("test", "json")

    def test_require_extra_missing(self) -> None:
        from src.cli._output import require_extra

        with pytest.raises(SystemExit):
            require_extra("nonexistent", "totally_fake_module_xyz")
