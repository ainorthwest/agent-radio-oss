"""Tests for `radio publish` CLI surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

SAMPLE_SCRIPT = {
    "title": "Demo",
    "date": "2026-05-01",
    "program": "haystack-news",
    "summary": "Test summary.",
    "segments": [
        {"speaker": "host_a", "text": "First.", "topic": "intro", "register": "baseline"},
    ],
}

SAMPLE_MANIFEST = {
    "version": 2,
    "date": "2026-05-01",
    "title": "Demo",
    "segments": [
        {"index": 0, "file": "seg-000.wav", "speaker": "host_a", "duration_seconds": 4.0},
    ],
    "cast": {"host_a": {"character_name": "Michael", "engine": "kokoro", "profile": "x"}},
    "program": "haystack-news",
}


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def episode_dir(tmp_path: Path) -> Path:
    """Build a tmp episode dir with script.json + manifest.json."""
    (tmp_path / "script.json").write_text(json.dumps(SAMPLE_SCRIPT, indent=2))
    (tmp_path / "manifest.json").write_text(json.dumps(SAMPLE_MANIFEST, indent=2))
    return tmp_path


# ── radio publish episode ───────────────────────────────────────────────────


class TestPublishEpisode:
    def test_writes_artifacts(self, runner, episode_dir: Path):
        from src.cli import app

        result = runner.invoke(app, ["publish", "episode", str(episode_dir)])
        assert result.exit_code == 0, result.output
        assert (episode_dir / "episode.md").exists()
        assert (episode_dir / "chapters.json").exists()
        assert (episode_dir / "episode.txt").exists()
        assert (episode_dir / "episode.jsonld").exists()

    def test_missing_episode_dir(self, runner, tmp_path: Path):
        from src.cli import app

        missing = tmp_path / "does-not-exist"
        result = runner.invoke(app, ["publish", "episode", str(missing)])
        assert result.exit_code != 0

    def test_dry_run_does_not_write(self, runner, episode_dir: Path):
        from src.cli import app

        result = runner.invoke(app, ["--dry-run", "publish", "episode", str(episode_dir)])
        assert result.exit_code == 0
        assert not (episode_dir / "episode.md").exists()


# ── radio publish llms-index ────────────────────────────────────────────────


class TestPublishLlmsIndex:
    def test_writes_llms_txt(self, runner, tmp_path: Path):
        from src.cli import app

        program_dir = tmp_path / "haystack-news"
        ep_dir = program_dir / "episodes" / "2026-05-01"
        ep_dir.mkdir(parents=True)
        (ep_dir / "episode.md").write_text("---\ntitle: Hello\ndate: '2026-05-01'\n---\n\nx\n")

        result = runner.invoke(
            app,
            [
                "publish",
                "llms-index",
                str(program_dir),
                "--show-name",
                "Haystack News",
                "--description",
                "Demo show.",
            ],
        )
        assert result.exit_code == 0, result.output
        llms = program_dir / "llms.txt"
        assert llms.exists()
        body = llms.read_text()
        assert "# Haystack News" in body
        assert "> Demo show." in body
