"""Tests for `radio edit` CLI surface.

Uses Typer's CliRunner so we can drive the commands without spawning
subprocesses. Tests cover the JSON-op subcommands (delete, replace,
reorder), the anomalies-report path, and dry-run / error handling.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

SAMPLE_SCRIPT = {
    "title": "Demo",
    "date": "sample",
    "program": "haystack-news",
    "segments": [
        {"speaker": "host_a", "text": "First.", "topic": "intro", "register": "baseline"},
        {"speaker": "host_b", "text": "Second.", "topic": "intro", "register": "baseline"},
        {"speaker": "host_c", "text": "Third.", "topic": "main", "register": "emphasis"},
    ],
}


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def script_file(tmp_path: Path) -> Path:
    """Write SAMPLE_SCRIPT to a temp file and return the path."""
    p = tmp_path / "script.json"
    p.write_text(json.dumps(SAMPLE_SCRIPT, indent=2))
    return p


# ── radio edit script delete ────────────────────────────────────────────────


class TestEditScriptDelete:
    def test_deletes_segment_writes_back(self, runner, script_file):
        from src.cli import app

        result = runner.invoke(app, ["edit", "script", str(script_file), "--delete", "1"])
        assert result.exit_code == 0, result.output

        updated = json.loads(script_file.read_text())
        assert len(updated["segments"]) == 2
        assert updated["segments"][0]["text"] == "First."
        assert updated["segments"][1]["text"] == "Third."

    def test_dry_run_does_not_write(self, runner, script_file):
        from src.cli import app

        result = runner.invoke(
            app, ["--dry-run", "edit", "script", str(script_file), "--delete", "0"]
        )
        assert result.exit_code == 0
        # Original file unchanged
        original = json.loads(script_file.read_text())
        assert len(original["segments"]) == 3

    def test_out_of_range_returns_error(self, runner, script_file):
        from src.cli import app

        result = runner.invoke(app, ["edit", "script", str(script_file), "--delete", "99"])
        assert result.exit_code != 0


# ── radio edit script replace ───────────────────────────────────────────────


class TestEditScriptReplace:
    def test_replaces_text(self, runner, script_file):
        from src.cli import app

        result = runner.invoke(
            app,
            ["edit", "script", str(script_file), "--replace", "0", "--text", "Replaced!"],
        )
        assert result.exit_code == 0, result.output
        updated = json.loads(script_file.read_text())
        assert updated["segments"][0]["text"] == "Replaced!"

    def test_empty_text_rejected(self, runner, script_file):
        from src.cli import app

        result = runner.invoke(
            app,
            ["edit", "script", str(script_file), "--replace", "0", "--text", ""],
        )
        assert result.exit_code != 0


# ── radio edit script reorder ───────────────────────────────────────────────


class TestEditScriptReorder:
    def test_reorders(self, runner, script_file):
        from src.cli import app

        result = runner.invoke(app, ["edit", "script", str(script_file), "--reorder", "2,0,1"])
        assert result.exit_code == 0, result.output
        updated = json.loads(script_file.read_text())
        assert updated["segments"][0]["text"] == "Third."
        assert updated["segments"][1]["text"] == "First."

    def test_invalid_permutation_rejected(self, runner, script_file):
        from src.cli import app

        result = runner.invoke(app, ["edit", "script", str(script_file), "--reorder", "0,0,1"])
        assert result.exit_code != 0


# ── radio edit anomalies ────────────────────────────────────────────────────


class TestEditAnomalies:
    def test_reports_no_anomalies_for_empty_episode(self, runner, tmp_path):
        from src.cli import app

        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"segments": [], "segments_dir": str(tmp_path)}))
        result = runner.invoke(app, ["edit", "anomalies", str(manifest)])
        assert result.exit_code == 0
        assert "0 anomalies" in result.output or "no anomalies" in result.output.lower()

    def test_writes_anomalies_json_alongside_manifest(self, runner, tmp_path):
        import numpy as np
        import soundfile as sf

        from src.cli import app

        # Build a one-segment manifest with a silent WAV (will flag silence)
        wav = tmp_path / "seg-000-host_a.wav"
        sf.write(str(wav), np.zeros(24000 * 2, dtype=np.float32), 24000)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "index": 0,
                            "file": "seg-000-host_a.wav",
                            "speaker": "host_a",
                            "text": "one two three four five",
                            "duration_seconds": 2.0,
                        }
                    ],
                    "segments_dir": str(tmp_path),
                }
            )
        )

        result = runner.invoke(app, ["edit", "anomalies", str(manifest)])
        assert result.exit_code == 0, result.output
        anomalies_path = tmp_path / "anomalies.json"
        assert anomalies_path.exists()
        data = json.loads(anomalies_path.read_text())
        assert "anomalies" in data
        assert len(data["anomalies"]) >= 1
