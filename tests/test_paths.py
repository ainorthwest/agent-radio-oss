"""Tests for the library path resolver."""

from __future__ import annotations

import pytest

from src.paths import LibraryPaths


@pytest.fixture
def paths(tmp_path):
    return LibraryPaths(tmp_path / "library")


class TestProgramPaths:
    def test_program_dir(self, paths):
        assert paths.program_dir("haystack-news") == paths.root / "programs" / "haystack-news"

    def test_program_config(self, paths):
        assert paths.program_config("haystack-news").name == "program.yaml"

    def test_program_assets(self, paths):
        assert (
            paths.program_assets("haystack-news")
            == paths.root / "programs" / "haystack-news" / "assets"
        )


class TestEpisodePaths:
    def test_episode_dir(self, paths):
        result = paths.episode_dir("haystack-news", "2026-03-19")
        assert result == paths.root / "programs" / "haystack-news" / "episodes" / "2026-03-19"

    def test_episode_audio(self, paths):
        result = paths.episode_audio("haystack-news", "2026-03-19")
        assert result.name == "episode.mp3"

    def test_episode_script(self, paths):
        result = paths.episode_script("haystack-news", "2026-03-19")
        assert result.name == "script.json"

    def test_episode_manifest(self, paths):
        result = paths.episode_manifest("haystack-news", "2026-03-19")
        assert result.name == "manifest.json"

    def test_episode_segments(self, paths):
        result = paths.episode_segments("haystack-news", "2026-03-19")
        assert result.name == "segments"


class TestTrackPaths:
    def test_track_dir(self, paths):
        result = paths.track_dir("late-night-lofi", "2026-03-20")
        assert result == paths.root / "programs" / "late-night-lofi" / "tracks" / "2026-03-20"

    def test_track_audio(self, paths):
        result = paths.track_audio("late-night-lofi", "2026-03-20", 1)
        assert result.name == "track-001.wav"

    def test_track_audio_zero_padded(self, paths):
        result = paths.track_audio("late-night-lofi", "2026-03-20", 42)
        assert result.name == "track-042.wav"


class TestSetPaths:
    def test_set_dir(self, paths):
        result = paths.set_dir("late-night-lofi", "2026-03-20")
        assert result == paths.root / "programs" / "late-night-lofi" / "sets" / "2026-03-20"

    def test_set_audio(self, paths):
        assert paths.set_audio("late-night-lofi", "2026-03-20").name == "set.mp3"

    def test_set_manifest(self, paths):
        assert paths.set_manifest("late-night-lofi", "2026-03-20").name == "manifest.json"

    def test_set_dj_segments(self, paths):
        assert paths.set_dj_segments("late-night-lofi", "2026-03-20").name == "dj-segments"


class TestOtherContentPaths:
    def test_special_dir(self, paths):
        assert paths.special_dir("launch-special") == paths.root / "specials" / "launch-special"

    def test_external_dir(self, paths):
        assert paths.external_dir("guest-interview") == paths.root / "external" / "guest-interview"

    def test_spot_dir(self, paths):
        assert paths.spot_dir("acme-spring") == paths.root / "spots" / "acme-spring"


class TestStationPaths:
    def test_station_ids(self, paths):
        assert paths.station_ids() == paths.root / "station" / "ids"

    def test_station_promos(self, paths):
        assert paths.station_promos() == paths.root / "station" / "promos"

    def test_station_bumpers(self, paths):
        assert paths.station_bumpers() == paths.root / "station" / "bumpers"


class TestSharedPaths:
    def test_shared_music(self, paths):
        assert paths.shared_music() == paths.root / "shared" / "music"

    def test_shared_voices(self, paths):
        assert paths.shared_voices() == paths.root / "shared" / "voices"

    def test_shared_sfx(self, paths):
        assert paths.shared_sfx() == paths.root / "shared" / "sfx"


class TestR2Keys:
    def test_r2_episode_key(self, paths):
        assert (
            paths.r2_episode_key("haystack-news", "2026-03-19")
            == "programs/haystack-news/2026-03-19.mp3"
        )

    def test_r2_track_key(self, paths):
        result = paths.r2_track_key("late-night-lofi", "2026-03-20", 3)
        assert result == "programs/late-night-lofi/tracks/2026-03-20/track-003.wav"

    def test_r2_set_key(self, paths):
        assert (
            paths.r2_set_key("late-night-lofi", "2026-03-20")
            == "programs/late-night-lofi/sets/2026-03-20.mp3"
        )

    def test_r2_feed_key_program(self, paths):
        assert paths.r2_feed_key("haystack-news") == "programs/haystack-news/feed.xml"

    def test_r2_feed_key_station(self, paths):
        assert paths.r2_feed_key() == "feed.xml"


class TestScaffolding:
    def test_ensure_structure(self, paths):
        paths.ensure_structure()
        assert (paths.root / "programs").is_dir()
        assert (paths.root / "specials").is_dir()
        assert (paths.root / "external").is_dir()
        assert (paths.root / "spots").is_dir()
        assert paths.station_ids().is_dir()
        assert paths.station_promos().is_dir()
        assert paths.station_bumpers().is_dir()
        assert paths.shared_music().is_dir()
        assert paths.shared_voices().is_dir()
        assert paths.shared_sfx().is_dir()

    def test_ensure_structure_idempotent(self, paths):
        paths.ensure_structure()
        paths.ensure_structure()  # should not raise

    def test_ensure_program_talk(self, paths):
        paths.ensure_program("haystack-news", "talk")
        assert paths.program_dir("haystack-news").is_dir()
        assert paths.program_assets("haystack-news").is_dir()
        assert (paths.program_dir("haystack-news") / "episodes").is_dir()

    def test_ensure_program_music(self, paths):
        paths.ensure_program("late-night-lofi", "music")
        assert paths.program_dir("late-night-lofi").is_dir()
        assert paths.program_assets("late-night-lofi").is_dir()
        assert (paths.program_dir("late-night-lofi") / "tracks").is_dir()
        assert (paths.program_dir("late-night-lofi") / "sets").is_dir()


class TestArtworkPaths:
    def test_program_artwork_path(self, paths):
        result = paths.program_artwork("haystack-news")
        assert result == paths.root / "programs" / "haystack-news" / "artwork" / "cover.png"

    def test_station_artwork_path(self, paths):
        result = paths.station_artwork()
        assert result == paths.root / "station" / "artwork" / "cover.png"

    def test_program_artwork_exists_when_file_present(self, paths, tmp_path):
        art = paths.program_artwork("haystack-news")
        art.parent.mkdir(parents=True)
        art.write_bytes(b"fake png")
        assert art.exists()

    def test_station_artwork_exists_when_file_present(self, paths):
        art = paths.station_artwork()
        art.parent.mkdir(parents=True)
        art.write_bytes(b"fake png")
        assert art.exists()

    def test_program_artwork_absent_when_not_created(self, paths):
        assert not paths.program_artwork("no-such-program").exists()

    def test_station_artwork_absent_when_not_created(self, paths):
        assert not paths.station_artwork().exists()


class TestDatabasePath:
    def test_db_path(self, paths):
        assert paths.db == paths.root / "radio.db"
