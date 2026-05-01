"""Tests for src/publisher.py — deterministic content fan-out.

Pure-function outputs (chapters.json, episode.txt, episode.jsonld,
episode.md frontmatter) regenerate every call and are byte-stable for
unchanged input. LLM-derived outputs are tested in test_publisher_llm.py.

Tests use sample script + manifest dicts; no actual TTS or whisper
required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SAMPLE_SCRIPT = {
    "title": "Hello from agent-radio-oss",
    "date": "2026-05-01",
    "program": "haystack-news",
    "summary": "A short canned script for testing.",
    "segments": [
        {"speaker": "host_a", "text": "First segment.", "topic": "intro", "register": "baseline"},
        {"speaker": "host_b", "text": "Second segment.", "topic": "intro", "register": "baseline"},
        {"speaker": "host_c", "text": "Third segment.", "topic": "main", "register": "emphasis"},
    ],
}

SAMPLE_MANIFEST = {
    "version": 2,
    "date": "2026-05-01",
    "title": "Hello from agent-radio-oss",
    "engine": "kokoro",
    "sample_rate": 24000,
    "segments": [
        {
            "index": 0,
            "file": "seg-000-host_a.wav",
            "speaker": "host_a",
            "duration_seconds": 4.5,
            "register": "baseline",
        },
        {
            "index": 1,
            "file": "seg-001-host_b.wav",
            "speaker": "host_b",
            "duration_seconds": 5.2,
            "register": "baseline",
        },
        {
            "index": 2,
            "file": "seg-002-host_c.wav",
            "speaker": "host_c",
            "duration_seconds": 3.8,
            "register": "emphasis",
        },
    ],
    "cast": {
        "host_a": {
            "character_name": "Michael",
            "profile": "voices/kokoro-michael.yaml",
            "engine": "kokoro",
        },
        "host_b": {
            "character_name": "Bella",
            "profile": "voices/kokoro-bella.yaml",
            "engine": "kokoro",
        },
        "host_c": {
            "character_name": "Adam",
            "profile": "voices/kokoro-adam.yaml",
            "engine": "kokoro",
        },
    },
    "program": "haystack-news",
}


@pytest.fixture
def episode_dir(tmp_path: Path) -> Path:
    """Build a tmp episode dir with script.json + manifest.json."""
    (tmp_path / "script.json").write_text(json.dumps(SAMPLE_SCRIPT, indent=2))
    (tmp_path / "manifest.json").write_text(json.dumps(SAMPLE_MANIFEST, indent=2))
    return tmp_path


# ── chapters.json (Podcasting 2.0 cloud chapters spec) ──────────────────────


class TestBuildChapters:
    def test_one_chapter_per_segment(self):
        from src.publisher import build_chapters

        chapters = build_chapters(SAMPLE_MANIFEST)
        assert chapters["version"] == "1.2.0"
        assert len(chapters["chapters"]) == 3
        # First chapter starts at 0
        assert chapters["chapters"][0]["startTime"] == 0.0

    def test_chapters_have_cumulative_start_times(self):
        from src.publisher import build_chapters

        chapters = build_chapters(SAMPLE_MANIFEST)
        # 0.0, 4.5, 9.7 — cumulative durations
        assert chapters["chapters"][0]["startTime"] == 0.0
        assert chapters["chapters"][1]["startTime"] == 4.5
        assert abs(chapters["chapters"][2]["startTime"] - 9.7) < 1e-6

    def test_chapter_titles_use_speaker_and_register(self):
        from src.publisher import build_chapters

        chapters = build_chapters(SAMPLE_MANIFEST)
        # Each chapter title should mention the speaker (or character name)
        assert any("Michael" in c["title"] or "host_a" in c["title"] for c in chapters["chapters"])

    def test_deterministic(self):
        from src.publisher import build_chapters

        a = build_chapters(SAMPLE_MANIFEST)
        b = build_chapters(SAMPLE_MANIFEST)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ── episode.txt (script flattened, agent-readable payload) ──────────────────


class TestBuildEpisodeText:
    def test_includes_all_segment_text(self):
        from src.publisher import build_episode_text

        txt = build_episode_text(SAMPLE_SCRIPT, SAMPLE_MANIFEST)
        assert "First segment." in txt
        assert "Second segment." in txt
        assert "Third segment." in txt

    def test_includes_speaker_attribution(self):
        from src.publisher import build_episode_text

        txt = build_episode_text(SAMPLE_SCRIPT, SAMPLE_MANIFEST)
        # Speaker labels should appear (character name preferred, fallback to slot)
        assert "Michael" in txt or "host_a" in txt

    def test_includes_episode_title(self):
        from src.publisher import build_episode_text

        txt = build_episode_text(SAMPLE_SCRIPT, SAMPLE_MANIFEST)
        assert "Hello from agent-radio-oss" in txt

    def test_deterministic(self):
        from src.publisher import build_episode_text

        a = build_episode_text(SAMPLE_SCRIPT, SAMPLE_MANIFEST)
        b = build_episode_text(SAMPLE_SCRIPT, SAMPLE_MANIFEST)
        assert a == b


# ── episode.jsonld (schema.org PodcastEpisode) ──────────────────────────────


class TestBuildJsonLd:
    def test_minimal_required_fields(self):
        from src.publisher import build_jsonld

        ld = build_jsonld(SAMPLE_SCRIPT, SAMPLE_MANIFEST)
        assert ld["@context"] == "https://schema.org"
        assert ld["@type"] == "PodcastEpisode"
        assert ld["name"] == "Hello from agent-radio-oss"
        assert ld["datePublished"] == "2026-05-01"

    def test_duration_iso8601(self):
        from src.publisher import build_jsonld

        ld = build_jsonld(SAMPLE_SCRIPT, SAMPLE_MANIFEST)
        # Total: 4.5 + 5.2 + 3.8 = 13.5s -> "PT13.5S" or "PT14S" rounded
        assert "duration" in ld
        assert ld["duration"].startswith("PT")

    def test_deterministic(self):
        from src.publisher import build_jsonld

        a = build_jsonld(SAMPLE_SCRIPT, SAMPLE_MANIFEST)
        b = build_jsonld(SAMPLE_SCRIPT, SAMPLE_MANIFEST)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ── episode.md (markdown with YAML frontmatter) ─────────────────────────────


class TestBuildEpisodeMarkdown:
    def test_starts_with_frontmatter(self):
        from src.publisher import build_episode_markdown

        md = build_episode_markdown(SAMPLE_SCRIPT, SAMPLE_MANIFEST)
        assert md.startswith("---\n")
        assert md.split("---\n", 2)[1].strip()  # frontmatter has content

    def test_frontmatter_has_required_fields(self):
        import yaml

        from src.publisher import build_episode_markdown

        md = build_episode_markdown(SAMPLE_SCRIPT, SAMPLE_MANIFEST)
        _, frontmatter, body = md.split("---\n", 2)
        meta = yaml.safe_load(frontmatter)
        assert meta["title"] == "Hello from agent-radio-oss"
        assert meta["date"] == "2026-05-01"
        assert meta["program"] == "haystack-news"
        assert meta["duration_seconds"] == 13.5
        assert "host_a" in meta["hosts"] or "Michael" in meta["hosts"]

    def test_body_contains_full_transcript(self):
        from src.publisher import build_episode_markdown

        md = build_episode_markdown(SAMPLE_SCRIPT, SAMPLE_MANIFEST)
        # Body should contain segment text (we want script content browsable)
        assert "First segment." in md
        assert "Second segment." in md

    def test_deterministic(self):
        from src.publisher import build_episode_markdown

        a = build_episode_markdown(SAMPLE_SCRIPT, SAMPLE_MANIFEST)
        b = build_episode_markdown(SAMPLE_SCRIPT, SAMPLE_MANIFEST)
        assert a == b


# ── publish() — orchestrator ────────────────────────────────────────────────


class TestPublish:
    def test_writes_all_deterministic_artifacts(self, episode_dir):
        from src.publisher import publish

        result = publish(episode_dir, llm_enabled=False)
        assert (episode_dir / "episode.md").exists()
        assert (episode_dir / "chapters.json").exists()
        assert (episode_dir / "episode.txt").exists()
        assert (episode_dir / "episode.jsonld").exists()
        assert "episode.md" in result["written"]

    def test_idempotent(self, episode_dir):
        """Re-running publish() with no input changes should produce identical files."""
        from src.publisher import publish

        publish(episode_dir, llm_enabled=False)
        first = (episode_dir / "episode.md").read_text()
        publish(episode_dir, llm_enabled=False)
        second = (episode_dir / "episode.md").read_text()
        assert first == second

    def test_missing_script_raises(self, tmp_path: Path):
        from src.publisher import publish

        with pytest.raises(FileNotFoundError, match="script"):
            publish(tmp_path, llm_enabled=False)

    def test_missing_manifest_raises(self, tmp_path: Path):
        from src.publisher import publish

        (tmp_path / "script.json").write_text(json.dumps(SAMPLE_SCRIPT))
        with pytest.raises(FileNotFoundError, match="manifest"):
            publish(tmp_path, llm_enabled=False)


# ── llms.txt generator (per-show index) ─────────────────────────────────────


class TestLlmsTxt:
    def test_writes_index_for_show_with_one_episode(self, tmp_path: Path):
        from src.publisher import build_llms_txt

        # Build a tmp program tree with one episode that has episode.md
        program_dir = tmp_path / "haystack-news"
        ep_dir = program_dir / "episodes" / "2026-05-01"
        ep_dir.mkdir(parents=True)
        (ep_dir / "episode.md").write_text("---\ntitle: Hello\ndate: '2026-05-01'\n---\n\nbody\n")

        body = build_llms_txt(
            program_dir, show_name="Haystack News", description="A short canned demo."
        )
        assert body.startswith("# Haystack News")
        assert "> A short canned demo." in body
        assert "## Episodes" in body
        # Link should point at the episode.md relative to program_dir
        assert "episodes/2026-05-01/episode.md" in body

    def test_handles_missing_episodes_dir(self, tmp_path: Path):
        from src.publisher import build_llms_txt

        program_dir = tmp_path / "empty"
        program_dir.mkdir()
        body = build_llms_txt(program_dir, show_name="Empty", description="")
        assert "# Empty" in body
        # Empty episodes section is OK
        assert "## Episodes" in body

    def test_lists_episodes_in_reverse_date_order(self, tmp_path: Path):
        from src.publisher import build_llms_txt

        program_dir = tmp_path / "show"
        for date in ["2026-04-01", "2026-05-01", "2026-04-15"]:
            ep_dir = program_dir / "episodes" / date
            ep_dir.mkdir(parents=True)
            (ep_dir / "episode.md").write_text(f"---\ntitle: '{date}'\ndate: '{date}'\n---\n\nx\n")

        body = build_llms_txt(program_dir, show_name="Show", description="")
        # Newest first: May 1 should appear before April 15 should appear before April 1
        idx_may = body.index("2026-05-01")
        idx_apr15 = body.index("2026-04-15")
        idx_apr1 = body.index("2026-04-01")
        assert idx_may < idx_apr15 < idx_apr1

    def test_skips_directories_without_episode_md(self, tmp_path: Path):
        """Episodes that haven't been published yet (no episode.md) shouldn't appear."""
        from src.publisher import build_llms_txt

        program_dir = tmp_path / "show"
        published = program_dir / "episodes" / "2026-05-01"
        published.mkdir(parents=True)
        (published / "episode.md").write_text("---\ntitle: x\ndate: '2026-05-01'\n---\nx\n")
        unpublished = program_dir / "episodes" / "2026-05-02"
        unpublished.mkdir(parents=True)
        # No episode.md

        body = build_llms_txt(program_dir, show_name="Show", description="")
        assert "2026-05-01" in body
        assert "2026-05-02" not in body
