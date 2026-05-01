"""Tests for podcast RSS feed generation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from xml.etree.ElementTree import fromstring

import pytest

from src.podcast import (
    EpisodeEntry,
    PodcastMetadata,
    _format_duration,
    build_feed,
    collect_episodes,
    generate_feed,
    load_podcast_config,
)


class TestPodcastMetadata:
    def test_defaults(self) -> None:
        m = PodcastMetadata()
        assert m.title == "Haystack News by AI Northwest"
        assert m.language == "en-us"
        assert m.explicit is False

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        config = tmp_path / "podcast.yaml"
        config.write_text("title: Test Show\ndescription: A test\ncategory: Science\n")
        m = load_podcast_config(config)
        assert m.title == "Test Show"
        assert m.description == "A test"
        assert m.category == "Science"

    def test_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        m = load_podcast_config(tmp_path / "nonexistent.yaml")
        assert m.title == "Haystack News by AI Northwest"


class TestFormatDuration:
    def test_seconds_only(self) -> None:
        assert _format_duration(45) == "0:45"

    def test_minutes_and_seconds(self) -> None:
        assert _format_duration(185) == "3:05"

    def test_hours(self) -> None:
        assert _format_duration(3661) == "1:01:01"

    def test_zero(self) -> None:
        assert _format_duration(0) == "0:00"


class TestBuildFeed:
    def test_valid_xml(self) -> None:
        metadata = PodcastMetadata(title="Test", description="Desc")
        xml = build_feed(metadata, [])
        root = fromstring(xml)
        assert root.tag == "rss"
        assert root.attrib["version"] == "2.0"

    def test_channel_elements(self) -> None:
        metadata = PodcastMetadata(title="My Show", description="About stuff", language="en-gb")
        xml = build_feed(metadata, [])
        root = fromstring(xml)
        channel = root.find("channel")
        assert channel is not None
        assert channel.findtext("title") == "My Show"
        assert channel.findtext("description") == "About stuff"
        assert channel.findtext("language") == "en-gb"

    def test_itunes_namespace(self) -> None:
        metadata = PodcastMetadata(
            title="Test",
            author="Test Author",
            category="Technology",
            subcategory="Tech News",
        )
        xml = build_feed(metadata, [])
        # Verify iTunes tags are present in the XML
        assert "itunes" in xml
        assert "Test Author" in xml
        assert "Technology" in xml

    def test_episode_items(self) -> None:
        metadata = PodcastMetadata(title="Test")
        episodes = [
            EpisodeEntry(
                title="Episode 1",
                description="First ep",
                audio_url="https://example.com/ep1.mp3",
                pub_date=datetime(2026, 3, 16, tzinfo=UTC),
                duration_seconds=180,
                guid="episodes/2026-03-16.mp3",
                file_size_bytes=2880000,
            ),
        ]
        xml = build_feed(metadata, episodes)
        root = fromstring(xml)
        items = root.findall("channel/item")
        assert len(items) == 1
        assert items[0].findtext("title") == "Episode 1"
        assert items[0].findtext("guid") == "episodes/2026-03-16.mp3"

    def test_enclosure_attributes(self) -> None:
        metadata = PodcastMetadata(title="Test")
        episodes = [
            EpisodeEntry(
                title="Ep",
                description="",
                audio_url="https://r2.example.com/ep.mp3",
                pub_date=datetime(2026, 3, 16, tzinfo=UTC),
                duration_seconds=60,
                guid="ep1",
                file_size_bytes=960000,
            ),
        ]
        xml = build_feed(metadata, episodes)
        root = fromstring(xml)
        enclosure = root.find("channel/item/enclosure")
        assert enclosure is not None
        assert enclosure.attrib["url"] == "https://r2.example.com/ep.mp3"
        assert enclosure.attrib["type"] == "audio/mpeg"
        assert enclosure.attrib["length"] == "960000"

    def test_episodes_sorted_newest_first(self) -> None:
        metadata = PodcastMetadata(title="Test")
        episodes = [
            EpisodeEntry(
                title="Older",
                description="",
                audio_url="",
                pub_date=datetime(2026, 3, 14, tzinfo=UTC),
                duration_seconds=60,
                guid="old",
            ),
            EpisodeEntry(
                title="Newer",
                description="",
                audio_url="",
                pub_date=datetime(2026, 3, 16, tzinfo=UTC),
                duration_seconds=60,
                guid="new",
            ),
        ]
        xml = build_feed(metadata, episodes)
        root = fromstring(xml)
        items = root.findall("channel/item")
        assert items[0].findtext("title") == "Newer"
        assert items[1].findtext("title") == "Older"

    def test_empty_episodes_valid_xml(self) -> None:
        xml = build_feed(PodcastMetadata(), [])
        root = fromstring(xml)
        items = root.findall("channel/item")
        assert len(items) == 0

    def test_pubdate_rfc2822(self) -> None:
        episodes = [
            EpisodeEntry(
                title="Ep",
                description="",
                audio_url="",
                pub_date=datetime(2026, 3, 16, 12, 0, 0, tzinfo=UTC),
                duration_seconds=60,
                guid="ep",
            ),
        ]
        xml = build_feed(PodcastMetadata(), episodes)
        # RFC 2822 dates contain day names and timezone
        assert "Mon, 16 Mar 2026" in xml


PODCAST_NS = "https://podcastindex.org/namespace/1.0"


class TestPodcastingTwoNamespace:
    """Verify Podcasting 2.0 namespace tags emit correctly."""

    def test_namespace_declared(self) -> None:
        metadata = PodcastMetadata(title="Test")
        xml = build_feed(metadata, [])
        # The podcast namespace must appear when registered
        assert "podcastindex.org/namespace/1.0" in xml or "podcast:" in xml

    def test_per_episode_transcript_tag(self) -> None:
        metadata = PodcastMetadata(title="Test")
        ep = EpisodeEntry(
            title="Episode 1",
            description="First ep",
            audio_url="https://example.com/ep1.mp3",
            pub_date=datetime(2026, 5, 1, tzinfo=UTC),
            duration_seconds=180,
            guid="ep1",
            transcript_url="https://example.com/ep1.srt",
        )
        xml = build_feed(metadata, [ep])
        root = fromstring(xml)
        # Transcript tag should be on the item with type=application/x-subrip
        item = root.find("channel/item")
        assert item is not None
        transcript = item.find(f"{{{PODCAST_NS}}}transcript")
        assert transcript is not None
        assert transcript.attrib["url"] == "https://example.com/ep1.srt"
        assert "x-subrip" in transcript.attrib["type"]

    def test_per_episode_chapters_tag(self) -> None:
        metadata = PodcastMetadata(title="Test")
        ep = EpisodeEntry(
            title="Episode 1",
            description="First ep",
            audio_url="https://example.com/ep1.mp3",
            pub_date=datetime(2026, 5, 1, tzinfo=UTC),
            duration_seconds=180,
            guid="ep1",
            chapters_url="https://example.com/ep1-chapters.json",
        )
        xml = build_feed(metadata, [ep])
        root = fromstring(xml)
        item = root.find("channel/item")
        assert item is not None
        chapters = item.find(f"{{{PODCAST_NS}}}chapters")
        assert chapters is not None
        assert chapters.attrib["url"] == "https://example.com/ep1-chapters.json"
        assert "json+chapters" in chapters.attrib["type"]

    def test_channel_person_tags_from_persons(self) -> None:
        metadata = PodcastMetadata(
            title="Test",
            persons=[
                {"name": "Michael", "role": "host", "img": "https://example.com/m.jpg"},
                {"name": "Bella", "role": "host"},
            ],
        )
        xml = build_feed(metadata, [])
        root = fromstring(xml)
        channel = root.find("channel")
        assert channel is not None
        persons = channel.findall(f"{{{PODCAST_NS}}}person")
        assert len(persons) == 2
        names = {p.text for p in persons}
        assert names == {"Michael", "Bella"}

    def test_optional_tags_omitted_when_unset(self) -> None:
        """Episodes without transcript/chapters URLs should not emit empty tags."""
        metadata = PodcastMetadata(title="Test")
        ep = EpisodeEntry(
            title="Episode 1",
            description="First ep",
            audio_url="https://example.com/ep1.mp3",
            pub_date=datetime(2026, 5, 1, tzinfo=UTC),
            duration_seconds=180,
            guid="ep1",
        )
        xml = build_feed(metadata, [ep])
        root = fromstring(xml)
        item = root.find("channel/item")
        assert item is not None
        assert item.find(f"{{{PODCAST_NS}}}transcript") is None
        assert item.find(f"{{{PODCAST_NS}}}chapters") is None


class TestCollectEpisodes:
    def test_collects_from_manifests(self, tmp_path: Path) -> None:
        ep_dir = tmp_path / "episodes" / "2026-03-16"
        ep_dir.mkdir(parents=True)

        manifest = {
            "date": "2026-03-16",
            "title": "Test Episode",
            "segments": [
                {"duration_seconds": 10.0},
                {"duration_seconds": 20.0},
            ],
        }
        (ep_dir / "manifest.json").write_text(json.dumps(manifest))
        (ep_dir / "episode_000.mp3").write_bytes(b"\x00" * 1000)

        entries = collect_episodes(tmp_path / "episodes", public_url_base="https://r2.example.com")
        assert len(entries) == 1
        assert entries[0].title == "Test Episode"
        assert entries[0].duration_seconds == 30
        assert entries[0].audio_url == "https://r2.example.com/episodes/2026-03-16.mp3"
        assert entries[0].guid == "episodes/2026-03-16.mp3"

    def test_empty_dir(self, tmp_path: Path) -> None:
        entries = collect_episodes(tmp_path / "nope")
        assert entries == []

    def test_skips_dirs_without_manifest(self, tmp_path: Path) -> None:
        ep_dir = tmp_path / "episodes" / "2026-03-16"
        ep_dir.mkdir(parents=True)
        (ep_dir / "episode_000.mp3").write_bytes(b"\x00" * 100)
        entries = collect_episodes(tmp_path / "episodes")
        assert entries == []

    def test_skips_dirs_without_mp3(self, tmp_path: Path) -> None:
        ep_dir = tmp_path / "episodes" / "2026-03-16"
        ep_dir.mkdir(parents=True)
        (ep_dir / "manifest.json").write_text('{"date":"2026-03-16","segments":[]}')
        entries = collect_episodes(tmp_path / "episodes")
        assert entries == []

    def test_guid_stability(self, tmp_path: Path) -> None:
        ep_dir = tmp_path / "episodes" / "2026-03-16"
        ep_dir.mkdir(parents=True)
        (ep_dir / "manifest.json").write_text('{"date":"2026-03-16","segments":[]}')
        (ep_dir / "episode_000.mp3").write_bytes(b"\x00" * 500)

        entries1 = collect_episodes(tmp_path / "episodes")
        entries2 = collect_episodes(tmp_path / "episodes")
        assert entries1[0].guid == entries2[0].guid


class TestGenerateFeed:
    def test_writes_file(self, tmp_path: Path) -> None:
        output = tmp_path / "feed.xml"
        result = generate_feed(
            config_path=Path("config/podcast.yaml"),
            output_path=output,
            episodes_dir=tmp_path / "episodes",
        )
        assert result == output
        assert output.exists()
        content = output.read_text()
        assert "<?xml" in content
        assert "rss" in content


ITUNES_NS = "http://www.itunes.com/dtds/podcast-1.0.dtd"


class TestArtworkConvention:
    """Tests for PR76: convention-based artwork in RSS feed."""

    def _make_episode_dir(
        self, base: Path, slug: str, date: str, artwork_path: str | None = None
    ) -> Path:
        ep_dir = base / "programs" / slug / "episodes" / date
        ep_dir.mkdir(parents=True)
        manifest: dict = {
            "version": 2,
            "date": date,
            "title": f"Episode {date}",
            "engine": "kokoro",
            "sample_rate": 24000,
            "segments_dir": str(ep_dir / "segments"),
            "cast": {},
            "segments": [{"index": 0, "duration_seconds": 60.0}],
            "music": {},
            "music_config": {},
        }
        if artwork_path:
            manifest["artwork_path"] = artwork_path
        (ep_dir / "manifest.json").write_text(json.dumps(manifest))
        (ep_dir / "episode.mp3").write_bytes(b"fake mp3")
        return ep_dir

    def test_episode_entry_has_artwork_url_field(self):
        ep = EpisodeEntry(
            title="Test",
            description="desc",
            audio_url="https://cdn/ep.mp3",
            pub_date=datetime.now(UTC),
            duration_seconds=60,
            guid="ep.mp3",
        )
        assert ep.artwork_url is None  # default

    def test_build_feed_includes_itunes_image_when_artwork_url_set(self):
        ep = EpisodeEntry(
            title="Test",
            description="desc",
            audio_url="https://cdn/ep.mp3",
            pub_date=datetime.now(UTC),
            duration_seconds=60,
            guid="ep.mp3",
            artwork_url="https://cdn/programs/haystack-news/artwork/cover.png",
        )
        xml_str = build_feed(PodcastMetadata(), [ep])
        root = fromstring(xml_str.split("\n", 1)[1])  # strip xml declaration
        item = root.find("channel/item")
        assert item is not None
        img = item.find(f"{{{ITUNES_NS}}}image")
        assert img is not None
        assert img.attrib["href"] == "https://cdn/programs/haystack-news/artwork/cover.png"

    def test_build_feed_omits_itunes_image_when_artwork_url_none(self):
        ep = EpisodeEntry(
            title="Test",
            description="desc",
            audio_url="https://cdn/ep.mp3",
            pub_date=datetime.now(UTC),
            duration_seconds=60,
            guid="ep.mp3",
        )
        xml_str = build_feed(PodcastMetadata(), [ep])
        root = fromstring(xml_str.split("\n", 1)[1])
        item = root.find("channel/item")
        assert item is not None
        assert item.find(f"{{{ITUNES_NS}}}image") is None

    def test_collect_episodes_resolves_artwork_url_from_manifest(self, tmp_path):
        library_root = tmp_path / "library"
        # Create artwork file
        art_file = library_root / "programs" / "haystack-news" / "artwork" / "cover.png"
        art_file.parent.mkdir(parents=True)
        art_file.write_bytes(b"fake png")

        self._make_episode_dir(
            library_root,
            "haystack-news",
            "2026-04-01",
            artwork_path=str(art_file),
        )

        entries = collect_episodes(
            library_root=library_root,
            public_url_base="https://radio.ainorthwest.org",
        )
        assert len(entries) == 1
        assert (
            entries[0].artwork_url
            == "https://radio.ainorthwest.org/programs/haystack-news/artwork/cover.png"
        )

    def test_collect_episodes_no_artwork_url_when_path_missing(self, tmp_path):
        library_root = tmp_path / "library"
        # manifest has artwork_path pointing to nonexistent file
        self._make_episode_dir(
            library_root,
            "haystack-news",
            "2026-04-01",
            artwork_path="/nonexistent/cover.png",
        )

        entries = collect_episodes(
            library_root=library_root,
            public_url_base="https://radio.ainorthwest.org",
        )
        assert len(entries) == 1
        # artwork_path not under library_root → ValueError → None
        assert entries[0].artwork_url is None

    def test_collect_episodes_no_artwork_url_when_manifest_has_none(self, tmp_path):
        library_root = tmp_path / "library"
        self._make_episode_dir(library_root, "haystack-news", "2026-04-01")

        entries = collect_episodes(
            library_root=library_root,
            public_url_base="https://radio.ainorthwest.org",
        )
        assert len(entries) == 1
        assert entries[0].artwork_url is None


class TestSilentDegradationFixes:
    """Tests for PR4: honest reporting when deps are missing."""

    def test_generate_feed_library_root_uses_absolute_path(self, tmp_path):
        """generate_feed with library_root writes feed.xml inside library_root, not cwd."""
        library_root = tmp_path / "library"
        library_root.mkdir()
        # No episodes needed — empty feed still writes
        result = generate_feed(
            config_path=Path("config/podcast.yaml"),
            library_root=library_root,
        )
        assert result == library_root / "feed.xml"
        assert result.is_absolute()
        assert result.exists()

    def test_generate_feed_explicit_output_path_respected(self, tmp_path):
        """generate_feed with explicit output_path must not override with library_root."""
        library_root = tmp_path / "library"
        library_root.mkdir()
        custom_output = tmp_path / "custom" / "feed.xml"
        result = generate_feed(
            config_path=Path("config/podcast.yaml"),
            output_path=custom_output,
            library_root=library_root,
        )
        assert result == custom_output

    @pytest.mark.skipif(
        not Path("src/quality.py").exists(),
        reason="src/quality.py not yet ported (lands in Commit 3 of Day 1 sprint)",
    )
    def test_quality_dnsmos_warning_emitted_when_torch_missing(self, capsys):
        """DNSMOS warning must actually reach stderr when torch is absent."""
        import sys
        import unittest.mock as mock

        # Stub numpy so the function body can be entered (numpy used before torch check)
        np_stub = mock.MagicMock()
        np_stub.zeros.return_value = np_stub
        with mock.patch.dict(sys.modules, {"torch": None, "numpy": np_stub}):
            if "src.quality" in sys.modules:
                del sys.modules["src.quality"]
            import src.quality as q_module

            q_module._compute_perceived_quality(np_stub, 16000)

        captured = capsys.readouterr()
        assert "torch" in captured.err.lower() or "dnsmos" in captured.err.lower()

        # Restore clean module state
        if "src.quality" in sys.modules:
            del sys.modules["src.quality"]

    def test_discourse_skip_reason_logged_when_r2_missing(self, tmp_path, capsys):
        """When post_to_discourse=True but R2 creds missing, reason must be logged."""
        import json
        from unittest.mock import MagicMock

        script = {"title": "Test", "date": "2026-04-01", "program": "test", "segments": []}
        script_path = tmp_path / "script.json"
        script_path.write_text(json.dumps(script))
        mp3_path = tmp_path / "ep.mp3"
        mp3_path.write_bytes(b"\x00" * 100)

        config = MagicMock()
        config.distributor.post_to_discourse = True
        config.distributor.r2_bucket = ""
        config.distributor.r2_access_key_id = ""
        config.distributor.public_url_base = "https://example.com"

        from src.distributor import distribute

        distribute(config, mp3_path, script_path)
        captured = capsys.readouterr()
        assert "R2" in captured.out or "credentials" in captured.out.lower()
