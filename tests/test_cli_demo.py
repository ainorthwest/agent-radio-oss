"""Tests for ``radio demo`` — the one-command guided-tour command.

The demo command is the AX-Phase-4 deliverable: a stranger runs one
command and gets a defensible end-to-end run, no setup required.
These tests pin the contract:

  - it always runs with no_distribute=True so demos can't accidentally
    push to R2 / Discourse / AzuraCast
  - it routes through the haystack-news program so the show bible's
    music + voices apply
  - it detects curator credentials and uses the canned sample script
    when none are present
  - it embeds a timestamp in the date string so two demos on the same
    day don't collide
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from src.cli import app
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


def _make_config(tmp_path: Path, *, curator_key: str = "") -> RadioConfig:
    return RadioConfig(
        discourse=DiscourseConfig(base_url="https://example.com", api_key="k", api_username="u"),
        curator=CuratorConfig(api_key=curator_key),
        renderer=RendererConfig(),
        distributor=DistributorConfig(),
        stream=StreamConfig(base_url="https://radio.example.com", api_key="az", station_id=1),
        voices={},
        library=LibraryConfig(root=str(tmp_path / "library")),
    )


@pytest.fixture()
def tmp_config_no_creds(tmp_path: Path) -> RadioConfig:
    cfg = _make_config(tmp_path, curator_key="")
    (tmp_path / "library").mkdir()
    return cfg


@pytest.fixture()
def tmp_config_with_creds(tmp_path: Path) -> RadioConfig:
    cfg = _make_config(tmp_path, curator_key="sk-fake-openrouter-key")
    (tmp_path / "library").mkdir()
    return cfg


def _invoke(args: list[str], config: RadioConfig) -> Any:
    """Invoke CLI with a fake config injected."""
    # Patch both State.config (for in-CLI lookups) and load_config (for
    # the demo command's direct credential check) so the entire run
    # uses the fixture config, not config/radio.yaml from disk.
    with (
        patch("src.cli.State.config", new_callable=lambda: property(lambda self: config)),
        patch("src.config.load_config", return_value=config),
    ):
        return runner.invoke(app, args, catch_exceptions=False)


class TestDemoCommand:
    """The demo command always runs no-distribute, always against haystack-news."""

    def test_demo_calls_pipeline_with_no_distribute(
        self, tmp_config_with_creds: RadioConfig
    ) -> None:
        with patch("src.pipeline.run", return_value=0) as mock:
            result = _invoke(["demo"], tmp_config_with_creds)
        assert result.exit_code == 0
        mock.assert_called_once()
        kwargs = mock.call_args.kwargs
        assert kwargs["no_distribute"] is True
        assert kwargs["dry_run"] is False
        assert kwargs["program_slug"] == "haystack-news"

    def test_demo_uses_sample_script_when_no_creds(self, tmp_config_no_creds: RadioConfig) -> None:
        with patch("src.pipeline.run", return_value=0) as mock:
            result = _invoke(["demo"], tmp_config_no_creds)
        assert result.exit_code == 0
        kwargs = mock.call_args.kwargs
        assert kwargs["script_override"] is not None
        # The sample script is the canned haystack-news script in the repo.
        assert "haystack-news" in str(kwargs["script_override"])
        assert "sample" in str(kwargs["script_override"])

    def test_demo_uses_curator_when_creds_present(self, tmp_config_with_creds: RadioConfig) -> None:
        with patch("src.pipeline.run", return_value=0) as mock:
            result = _invoke(["demo"], tmp_config_with_creds)
        assert result.exit_code == 0
        kwargs = mock.call_args.kwargs
        assert kwargs["script_override"] is None  # let curator run

    def test_demo_embeds_timestamp_in_date(self, tmp_config_with_creds: RadioConfig) -> None:
        """Two same-day demos must not collide. date_override carries
        a -demo-HHMMSS suffix to keep them separate."""
        with patch("src.pipeline.run", return_value=0) as mock:
            _invoke(["demo"], tmp_config_with_creds)
        kwargs = mock.call_args.kwargs
        date_override = kwargs["date_override"]
        assert date_override is not None
        assert "-demo-" in date_override

    def test_demo_propagates_pipeline_failure_exit_code(
        self, tmp_config_with_creds: RadioConfig
    ) -> None:
        with patch("src.pipeline.run", return_value=1):
            result = runner.invoke(
                app,
                ["demo"],
                # don't patch State.config, just verify SystemExit propagates
                catch_exceptions=True,
            )
        assert result.exit_code == 1

    def test_demo_errors_when_sample_missing_and_no_creds(
        self, tmp_config_no_creds: RadioConfig
    ) -> None:
        """If the sample script file is gone (e.g. someone deleted it
        from a fork), the demo can't recover. Exit cleanly with a
        message, not a stack trace."""
        with (
            patch("src.cli.demo_cmd.SAMPLE_SCRIPT_PATH", Path("/nonexistent/script.json")),
            patch("src.pipeline.run") as mock,
        ):
            result = runner.invoke(
                app,
                ["demo"],
                catch_exceptions=True,
            )
        # Should fail before pipeline runs
        assert mock.call_count == 0
        assert result.exit_code == 1


class TestPipelineScriptOverride:
    """pipeline.run() must accept script_override + date_override now."""

    def test_run_accepts_script_override(self) -> None:
        import inspect

        from src.pipeline import run

        sig = inspect.signature(run)
        assert "script_override" in sig.parameters
        assert sig.parameters["script_override"].default is None

    def test_run_accepts_date_override(self) -> None:
        import inspect

        from src.pipeline import run

        sig = inspect.signature(run)
        assert "date_override" in sig.parameters
        assert sig.parameters["date_override"].default is None


class TestDemoSampleScriptShape:
    """The sample script must satisfy the renderer's contract."""

    def test_sample_script_exists(self) -> None:
        from src.cli.demo_cmd import SAMPLE_SCRIPT_PATH

        assert SAMPLE_SCRIPT_PATH.exists(), (
            "Demo command depends on this canned script — if it's gone, "
            "the no-creds path can't run."
        )

    def test_sample_script_has_segments(self) -> None:
        import json

        from src.cli.demo_cmd import SAMPLE_SCRIPT_PATH

        data = json.loads(SAMPLE_SCRIPT_PATH.read_text())
        assert data["program"] == "haystack-news"
        assert len(data["segments"]) > 0
        for seg in data["segments"]:
            assert "speaker" in seg
            assert "text" in seg
