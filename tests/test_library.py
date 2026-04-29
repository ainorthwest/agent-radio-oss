"""Tests for the station library catalog (SQLite-backed)."""

from __future__ import annotations

import sqlite3

import pytest

from src.library import (
    Asset,
    Catalog,
    Distribution,
    Episode,
    ExternalContent,
    Feedback,
    Program,
    Spot,
    Track,
)


@pytest.fixture
def catalog(tmp_path):
    """Fresh catalog in a temp directory."""
    db_path = tmp_path / "radio.db"
    cat = Catalog(db_path)
    yield cat
    cat.close()


@pytest.fixture
def catalog_with_program(catalog):
    """Catalog with a registered talk program."""
    catalog.register_program("haystack-news", "Haystack News", "talk")
    return catalog


@pytest.fixture
def catalog_with_music_program(catalog):
    """Catalog with a registered music program."""
    catalog.register_program("late-night-lofi", "Late Night Lofi", "music")
    return catalog


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_creates_all_tables(self, catalog):
        tables = catalog._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = sorted(t["name"] for t in tables)
        assert "programs" in table_names
        assert "episodes" in table_names
        assert "tracks" in table_names
        assert "assets" in table_names
        assert "distributions" in table_names
        assert "feedback" in table_names
        assert "spots" in table_names
        assert "external_content" in table_names

    def test_schema_idempotent(self, tmp_path):
        db_path = tmp_path / "radio.db"
        cat1 = Catalog(db_path)
        cat1.register_program("test", "Test", "talk")
        cat1.close()
        # Reopening should not fail or lose data
        cat2 = Catalog(db_path)
        assert cat2.get_program("test") is not None
        cat2.close()

    def test_context_manager(self, tmp_path):
        db_path = tmp_path / "ctx.db"
        with Catalog(db_path) as cat:
            cat.register_program("test", "Test", "talk")
            prog = cat.get_program("test")
            assert prog is not None
        # Connection should be closed after exiting context
        with pytest.raises(Exception):
            cat._conn.execute("SELECT 1")

    def test_foreign_keys_enabled(self, catalog):
        result = catalog._conn.execute("PRAGMA foreign_keys").fetchone()
        assert result[0] == 1

    def test_wal_mode(self, catalog):
        result = catalog._conn.execute("PRAGMA journal_mode").fetchone()
        assert result[0] == "wal"


# ---------------------------------------------------------------------------
# Programs
# ---------------------------------------------------------------------------


class TestPrograms:
    def test_register_program(self, catalog):
        pid = catalog.register_program("haystack-news", "Haystack News", "talk")
        assert pid >= 1

    def test_get_program(self, catalog_with_program):
        prog = catalog_with_program.get_program("haystack-news")
        assert prog is not None
        assert isinstance(prog, Program)
        assert prog.slug == "haystack-news"
        assert prog.name == "Haystack News"
        assert prog.program_type == "talk"
        assert prog.status == "active"

    def test_get_program_not_found(self, catalog):
        assert catalog.get_program("nonexistent") is None

    def test_list_programs(self, catalog):
        catalog.register_program("aaa", "AAA", "talk")
        catalog.register_program("zzz", "ZZZ", "music")
        programs = catalog.list_programs()
        assert len(programs) == 2
        assert programs[0].name == "AAA"  # sorted by name

    def test_list_programs_by_status(self, catalog):
        catalog.register_program("active-one", "Active", "talk", status="active")
        catalog.register_program("dev-one", "Dev", "music", status="dev")
        active = catalog.list_programs(status="active")
        assert len(active) == 1
        assert active[0].slug == "active-one"

    def test_duplicate_slug_raises(self, catalog_with_program):
        with pytest.raises(sqlite3.IntegrityError):
            catalog_with_program.register_program("haystack-news", "Duplicate", "talk")

    def test_update_program(self, catalog_with_program):
        catalog_with_program.update_program("haystack-news", status="retired")
        prog = catalog_with_program.get_program("haystack-news")
        assert prog is not None
        assert prog.status == "retired"

    def test_update_program_cast_config(self, catalog_with_program):
        cast = {"host_a": {"profile": "voices/leo.yaml"}}
        catalog_with_program.update_program("haystack-news", cast_config=cast)
        prog = catalog_with_program.get_program("haystack-news")
        assert prog is not None
        assert prog.cast_config == cast

    def test_register_with_all_fields(self, catalog):
        catalog.register_program(
            slug="full-test",
            name="Full Test",
            program_type="music",
            status="dev",
            schedule_cron="0 22 * * *",
            palette_path="shows/test.yaml",
            cast_config={"dj": {"profile": "voices/dj.yaml"}},
        )
        prog = catalog.get_program("full-test")
        assert prog is not None
        assert prog.program_type == "music"
        assert prog.status == "dev"
        assert prog.schedule_cron == "0 22 * * *"
        assert prog.palette_path == "shows/test.yaml"
        assert prog.cast_config == {"dj": {"profile": "voices/dj.yaml"}}


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


class TestEpisodes:
    def test_add_episode(self, catalog_with_program):
        eid = catalog_with_program.add_episode(
            "haystack-news", "2026-03-19", "programs/haystack-news/episodes/2026-03-19/episode.mp3"
        )
        assert eid >= 1

    def test_get_episode(self, catalog_with_program):
        catalog_with_program.add_episode(
            "haystack-news",
            "2026-03-19",
            "ep.mp3",
            duration_seconds=320.0,
            quality_score=0.75,
            segment_count=18,
        )
        ep = catalog_with_program.get_episode("haystack-news", "2026-03-19")
        assert ep is not None
        assert isinstance(ep, Episode)
        assert ep.duration_seconds == 320.0
        assert ep.quality_score == 0.75
        assert ep.segment_count == 18
        assert ep.status == "generated"

    def test_get_episode_not_found(self, catalog_with_program):
        assert catalog_with_program.get_episode("haystack-news", "2099-01-01") is None

    def test_duplicate_program_date_returns_existing_id(self, catalog_with_program):
        """Duplicate (program_slug, date) should return the existing episode ID, not raise."""
        first_id = catalog_with_program.add_episode("haystack-news", "2026-03-19", "ep1.mp3")
        second_id = catalog_with_program.add_episode("haystack-news", "2026-03-19", "ep2.mp3")
        assert first_id == second_id
        # Only one record should exist
        episodes = catalog_with_program.list_episodes("haystack-news")
        assert len([e for e in episodes if e.date == "2026-03-19"]) == 1

    def test_list_episodes(self, catalog_with_program):
        catalog_with_program.add_episode("haystack-news", "2026-03-17", "a.mp3")
        catalog_with_program.add_episode("haystack-news", "2026-03-18", "b.mp3")
        catalog_with_program.add_episode("haystack-news", "2026-03-19", "c.mp3")
        episodes = catalog_with_program.list_episodes("haystack-news")
        assert len(episodes) == 3
        assert episodes[0].date == "2026-03-19"  # most recent first

    def test_list_episodes_with_limit(self, catalog_with_program):
        for i in range(5):
            catalog_with_program.add_episode("haystack-news", f"2026-03-{15 + i:02d}", f"ep{i}.mp3")
        episodes = catalog_with_program.list_episodes("haystack-news", limit=2)
        assert len(episodes) == 2

    def test_latest_episode(self, catalog_with_program):
        catalog_with_program.add_episode("haystack-news", "2026-03-17", "a.mp3")
        catalog_with_program.add_episode("haystack-news", "2026-03-19", "c.mp3")
        catalog_with_program.add_episode("haystack-news", "2026-03-18", "b.mp3")
        latest = catalog_with_program.latest_episode("haystack-news")
        assert latest is not None
        assert latest.date == "2026-03-19"

    def test_latest_episode_empty(self, catalog_with_program):
        assert catalog_with_program.latest_episode("haystack-news") is None

    def test_update_episode(self, catalog_with_program):
        eid = catalog_with_program.add_episode("haystack-news", "2026-03-19", "ep.mp3")
        catalog_with_program.update_episode(eid, status="reviewed", quality_score=0.82)
        ep = catalog_with_program.get_episode("haystack-news", "2026-03-19")
        assert ep is not None
        assert ep.status == "reviewed"
        assert ep.quality_score == 0.82


# ---------------------------------------------------------------------------
# Lifecycle transitions
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_track_default_status(self, catalog_with_music_program):
        tid = catalog_with_music_program.add_track("late-night-lofi", "Beat", "t.wav", "2026-03-20")
        track = catalog_with_music_program.get_track(tid)
        assert track.status == "generated"

    def test_track_valid_transition(self, catalog_with_music_program):
        tid = catalog_with_music_program.add_track("late-night-lofi", "Beat", "t.wav", "2026-03-20")
        catalog_with_music_program.set_track_status(tid, "reviewed")
        track = catalog_with_music_program.get_track(tid)
        assert track.status == "reviewed"
        assert track.reviewed_at is not None

    def test_track_full_lifecycle(self, catalog_with_music_program):
        tid = catalog_with_music_program.add_track("late-night-lofi", "Beat", "t.wav", "2026-03-20")
        for status in ["reviewed", "approved", "scheduled", "aired", "archived"]:
            catalog_with_music_program.set_track_status(tid, status)
        track = catalog_with_music_program.get_track(tid)
        assert track.status == "archived"
        assert track.reviewed_at is not None
        assert track.approved_at is not None
        assert track.scheduled_at is not None
        assert track.aired_at is not None

    def test_track_invalid_transition_raises(self, catalog_with_music_program):
        tid = catalog_with_music_program.add_track("late-night-lofi", "Beat", "t.wav", "2026-03-20")
        with pytest.raises(ValueError, match="Invalid transition"):
            catalog_with_music_program.set_track_status(tid, "aired")  # can't skip to aired

    def test_track_rejection(self, catalog_with_music_program):
        tid = catalog_with_music_program.add_track("late-night-lofi", "Beat", "t.wav", "2026-03-20")
        catalog_with_music_program.set_track_status(tid, "rejected")
        track = catalog_with_music_program.get_track(tid)
        assert track.status == "rejected"

    def test_rejected_can_regenerate(self, catalog_with_music_program):
        tid = catalog_with_music_program.add_track("late-night-lofi", "Beat", "t.wav", "2026-03-20")
        catalog_with_music_program.set_track_status(tid, "rejected")
        catalog_with_music_program.set_track_status(tid, "generated")
        track = catalog_with_music_program.get_track(tid)
        assert track.status == "generated"

    def test_track_not_found_raises(self, catalog_with_music_program):
        with pytest.raises(ValueError, match="not found"):
            catalog_with_music_program.set_track_status(9999, "reviewed")

    def test_list_tracks_by_status(self, catalog_with_music_program):
        t1 = catalog_with_music_program.add_track("late-night-lofi", "A", "a.wav", "2026-03-20")
        catalog_with_music_program.add_track("late-night-lofi", "B", "b.wav", "2026-03-20")
        catalog_with_music_program.set_track_status(t1, "reviewed")
        catalog_with_music_program.set_track_status(t1, "approved")
        generated = catalog_with_music_program.list_tracks_by_status("generated")
        approved = catalog_with_music_program.list_tracks_by_status("approved")
        assert len(generated) == 1
        assert generated[0].title == "B"
        assert len(approved) == 1
        assert approved[0].title == "A"

    def test_episode_lifecycle(self, catalog_with_program):
        eid = catalog_with_program.add_episode("haystack-news", "2026-03-19", "ep.mp3")
        assert catalog_with_program.get_episode("haystack-news", "2026-03-19").status == "generated"
        catalog_with_program.set_episode_status(eid, "reviewed")
        catalog_with_program.set_episode_status(eid, "approved")
        ep = catalog_with_program.get_episode("haystack-news", "2026-03-19")
        assert ep.status == "approved"
        assert ep.reviewed_at is not None
        assert ep.approved_at is not None

    def test_episode_invalid_transition_raises(self, catalog_with_program):
        eid = catalog_with_program.add_episode("haystack-news", "2026-03-19", "ep.mp3")
        with pytest.raises(ValueError, match="Invalid transition"):
            catalog_with_program.set_episode_status(eid, "approved")  # must review first

    def test_episode_cannot_schedule(self, catalog_with_program):
        """Episodes use 'distributed' not 'scheduled' — different lifecycle than tracks."""
        eid = catalog_with_program.add_episode("haystack-news", "2026-03-19", "ep.mp3")
        catalog_with_program.set_episode_status(eid, "reviewed")
        catalog_with_program.set_episode_status(eid, "approved")
        catalog_with_program.set_episode_status(eid, "distributed")
        ep = catalog_with_program.get_episode("haystack-news", "2026-03-19")
        assert ep.status == "distributed"

    def test_rejected_track_timestamps_cleared_on_regenerate(self, catalog_with_music_program):
        """Re-entering 'generated' clears all lifecycle timestamps."""
        tid = catalog_with_music_program.add_track("late-night-lofi", "Beat", "t.wav", "2026-03-20")
        catalog_with_music_program.set_track_status(tid, "reviewed")
        catalog_with_music_program.set_track_status(tid, "rejected")
        # Timestamps should exist from the review
        track = catalog_with_music_program.get_track(tid)
        assert track.reviewed_at is not None
        assert track.rejected_at is not None
        # Re-generate should clear them
        catalog_with_music_program.set_track_status(tid, "generated")
        track = catalog_with_music_program.get_track(tid)
        assert track.status == "generated"
        assert track.reviewed_at is None
        assert track.approved_at is None
        assert track.rejected_at is None

    def test_rejected_and_archived_have_timestamps(self, catalog_with_music_program):
        """Terminal states should record when they happened."""
        tid = catalog_with_music_program.add_track("late-night-lofi", "Beat", "t.wav", "2026-03-20")
        catalog_with_music_program.set_track_status(tid, "rejected")
        track = catalog_with_music_program.get_track(tid)
        assert track.rejected_at is not None

    def test_archived_has_timestamp(self, catalog_with_music_program):
        tid = catalog_with_music_program.add_track("late-night-lofi", "Beat", "t.wav", "2026-03-20")
        for status in ["reviewed", "approved", "scheduled", "aired", "archived"]:
            catalog_with_music_program.set_track_status(tid, status)
        track = catalog_with_music_program.get_track(tid)
        assert track.archived_at is not None


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------


class TestTracks:
    def test_add_track(self, catalog_with_music_program):
        tid = catalog_with_music_program.add_track(
            "late-night-lofi",
            "Chill Beat #1",
            "tracks/2026-03-20/track-001.wav",
            "2026-03-20",
            duration_seconds=240.0,
            prompt="lofi hip hop beat with warm vinyl crackle",
        )
        assert tid >= 1

    def test_get_track(self, catalog_with_music_program):
        tid = catalog_with_music_program.add_track(
            "late-night-lofi",
            "Beat",
            "t.wav",
            "2026-03-20",
            duration_seconds=180.0,
            prompt="jazzy lofi",
            quality_score=0.8,
        )
        track = catalog_with_music_program.get_track(tid)
        assert track is not None
        assert isinstance(track, Track)
        assert track.title == "Beat"
        assert track.prompt == "jazzy lofi"
        assert track.play_count == 0
        assert track.heart_count == 0

    def test_list_tracks_by_program(self, catalog_with_music_program):
        catalog_with_music_program.add_track("late-night-lofi", "A", "a.wav", "2026-03-20")
        catalog_with_music_program.add_track("late-night-lofi", "B", "b.wav", "2026-03-21")
        tracks = catalog_with_music_program.list_tracks("late-night-lofi")
        assert len(tracks) == 2
        assert tracks[0].date == "2026-03-21"  # most recent first

    def test_increment_play_count(self, catalog_with_music_program):
        tid = catalog_with_music_program.add_track("late-night-lofi", "A", "a.wav", "2026-03-20")
        catalog_with_music_program.increment_play_count(tid)
        catalog_with_music_program.increment_play_count(tid)
        track = catalog_with_music_program.get_track(tid)
        assert track is not None
        assert track.play_count == 2

    def test_get_top_rated(self, catalog_with_music_program):
        t1 = catalog_with_music_program.add_track("late-night-lofi", "Loved", "a.wav", "2026-03-20")
        t2 = catalog_with_music_program.add_track("late-night-lofi", "Okay", "b.wav", "2026-03-20")
        # Add hearts to t1
        for _ in range(5):
            catalog_with_music_program.record_feedback("track", t1, "heart")
        catalog_with_music_program.record_feedback("track", t2, "heart")
        top = catalog_with_music_program.get_top_rated("late-night-lofi", limit=2)
        assert len(top) == 2
        assert top[0].id == t1
        assert top[0].heart_count == 5


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------


class TestAssets:
    def test_register_asset(self, catalog_with_program):
        aid = catalog_with_program.register_asset(
            "theme.wav",
            "music",
            "program",
            "programs/haystack-news/assets/theme.wav",
            program_slug="haystack-news",
            duration_seconds=5.0,
        )
        assert aid >= 1

    def test_find_assets_by_type(self, catalog_with_program):
        catalog_with_program.register_asset(
            "theme.wav", "music", "program", "a.wav", program_slug="haystack-news"
        )
        catalog_with_program.register_asset("ref.wav", "voice", "shared", "b.wav")
        music = catalog_with_program.find_assets(asset_type="music")
        assert len(music) == 1
        assert isinstance(music[0], Asset)
        assert music[0].filename == "theme.wav"

    def test_find_assets_by_scope(self, catalog_with_program):
        catalog_with_program.register_asset("a.wav", "music", "shared", "a.wav")
        catalog_with_program.register_asset(
            "b.wav", "music", "program", "b.wav", program_slug="haystack-news"
        )
        shared = catalog_with_program.find_assets(scope="shared")
        assert len(shared) == 1

    def test_find_assets_combined_filters(self, catalog_with_program):
        catalog_with_program.register_asset(
            "a.wav", "music", "program", "a.wav", program_slug="haystack-news"
        )
        catalog_with_program.register_asset(
            "b.wav", "voice", "program", "b.wav", program_slug="haystack-news"
        )
        result = catalog_with_program.find_assets(asset_type="music", program_slug="haystack-news")
        assert len(result) == 1

    def test_asset_metadata(self, catalog):
        catalog.register_program("test", "Test", "talk")
        catalog.register_asset(
            "sting.wav",
            "music",
            "station",
            "s.wav",
            metadata={"genre": "electronic", "bpm": 120},
        )
        assets = catalog.find_assets(scope="station")
        assert len(assets) == 1
        assert assets[0].metadata == {"genre": "electronic", "bpm": 120}


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


class TestDistributions:
    def test_record_distribution(self, catalog_with_program):
        eid = catalog_with_program.add_episode("haystack-news", "2026-03-19", "ep.mp3")
        did = catalog_with_program.record_distribution(
            "episode",
            eid,
            "r2",
            url="https://radio.ainorthwest.org/programs/haystack-news/2026-03-19.mp3",
        )
        assert did >= 1

    def test_get_distributions(self, catalog_with_program):
        eid = catalog_with_program.add_episode("haystack-news", "2026-03-19", "ep.mp3")
        catalog_with_program.record_distribution(
            "episode", eid, "r2", url="https://r2.example/ep.mp3"
        )
        catalog_with_program.record_distribution("episode", eid, "azuracast", success=True)
        catalog_with_program.record_distribution(
            "episode", eid, "discourse", success=False, error_message="timeout"
        )
        dists = catalog_with_program.get_distributions("episode", eid)
        assert len(dists) == 3
        assert isinstance(dists[0], Distribution)
        # Most recent first
        failed = [d for d in dists if not d.success]
        assert len(failed) == 1
        assert failed[0].error_message == "timeout"

    def test_distribution_for_tracks(self, catalog_with_music_program):
        tid = catalog_with_music_program.add_track("late-night-lofi", "Beat", "t.wav", "2026-03-20")
        catalog_with_music_program.record_distribution("track", tid, "azuracast")
        dists = catalog_with_music_program.get_distributions("track", tid)
        assert len(dists) == 1


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


class TestFeedback:
    def test_record_heart(self, catalog_with_music_program):
        tid = catalog_with_music_program.add_track("late-night-lofi", "Beat", "t.wav", "2026-03-20")
        fid = catalog_with_music_program.record_feedback("track", tid, "heart", "web")
        assert fid >= 1

    def test_heart_increments_track_count(self, catalog_with_music_program):
        tid = catalog_with_music_program.add_track("late-night-lofi", "Beat", "t.wav", "2026-03-20")
        catalog_with_music_program.record_feedback("track", tid, "heart")
        catalog_with_music_program.record_feedback("track", tid, "heart")
        track = catalog_with_music_program.get_track(tid)
        assert track is not None
        assert track.heart_count == 2

    def test_skip_does_not_increment_hearts(self, catalog_with_music_program):
        tid = catalog_with_music_program.add_track("late-night-lofi", "Beat", "t.wav", "2026-03-20")
        catalog_with_music_program.record_feedback("track", tid, "skip")
        track = catalog_with_music_program.get_track(tid)
        assert track is not None
        assert track.heart_count == 0

    def test_get_feedback(self, catalog_with_music_program):
        tid = catalog_with_music_program.add_track("late-night-lofi", "Beat", "t.wav", "2026-03-20")
        catalog_with_music_program.record_feedback("track", tid, "heart", "web")
        catalog_with_music_program.record_feedback("track", tid, "skip", "discourse")
        fb = catalog_with_music_program.get_feedback("track", tid)
        assert len(fb) == 2
        assert isinstance(fb[0], Feedback)

    def test_count_feedback(self, catalog_with_music_program):
        tid = catalog_with_music_program.add_track("late-night-lofi", "Beat", "t.wav", "2026-03-20")
        for _ in range(3):
            catalog_with_music_program.record_feedback("track", tid, "heart")
        catalog_with_music_program.record_feedback("track", tid, "skip")
        assert catalog_with_music_program.count_feedback("track", tid, "heart") == 3
        assert catalog_with_music_program.count_feedback("track", tid, "skip") == 1

    def test_aggregate_feedback(self, catalog_with_music_program):
        t1 = catalog_with_music_program.add_track("late-night-lofi", "A", "a.wav", "2026-03-20")
        t2 = catalog_with_music_program.add_track("late-night-lofi", "B", "b.wav", "2026-03-20")
        t3 = catalog_with_music_program.add_track("late-night-lofi", "C", "c.wav", "2026-03-20")
        for _ in range(5):
            catalog_with_music_program.record_feedback("track", t1, "heart")
        for _ in range(3):
            catalog_with_music_program.record_feedback("track", t2, "heart")
        catalog_with_music_program.record_feedback("track", t3, "heart")
        agg = catalog_with_music_program.aggregate_feedback("track", "heart", limit=2)
        assert len(agg) == 2
        assert agg[0] == (t1, 5)
        assert agg[1] == (t2, 3)

    def test_feedback_on_episodes(self, catalog_with_program):
        eid = catalog_with_program.add_episode("haystack-news", "2026-03-19", "ep.mp3")
        catalog_with_program.record_feedback("episode", eid, "heart")
        assert catalog_with_program.count_feedback("episode", eid, "heart") == 1

    def test_recount_hearts(self, catalog_with_music_program):
        tid = catalog_with_music_program.add_track("late-night-lofi", "Beat", "t.wav", "2026-03-20")
        # Add 3 hearts via record_feedback
        for _ in range(3):
            catalog_with_music_program.record_feedback("track", tid, "heart")
        assert catalog_with_music_program.get_track(tid).heart_count == 3
        # Manually corrupt the cache
        catalog_with_music_program._conn.execute(
            "UPDATE tracks SET heart_count = 999 WHERE id = ?", (tid,)
        )
        catalog_with_music_program._conn.commit()
        assert catalog_with_music_program.get_track(tid).heart_count == 999
        # Repair via recount
        count = catalog_with_music_program.recount_hearts(tid)
        assert count == 3
        assert catalog_with_music_program.get_track(tid).heart_count == 3


# ---------------------------------------------------------------------------
# Spots
# ---------------------------------------------------------------------------


class TestSpots:
    def test_add_spot(self, catalog):
        sid = catalog.add_spot(
            "Spring Promo", "spots/spring.mp3", sponsor="Acme Corp", duration_seconds=30.0
        )
        assert sid >= 1

    def test_list_active_spots(self, catalog):
        catalog.add_spot("Active", "a.mp3")
        s2 = catalog.add_spot("Inactive", "b.mp3")
        catalog.toggle_spot(s2, False)
        active = catalog.list_active_spots()
        assert len(active) == 1
        assert isinstance(active[0], Spot)
        assert active[0].name == "Active"

    def test_spot_with_dates(self, catalog):
        catalog.add_spot(
            "Seasonal",
            "s.mp3",
            start_date="2026-03-01",
            end_date="2026-06-01",
            rotation_weight=2.0,
        )
        spots = catalog.list_active_spots()
        assert spots[0].start_date == "2026-03-01"
        assert spots[0].rotation_weight == 2.0

    def test_expired_spot_excluded(self, catalog):
        catalog.add_spot("Expired", "e.mp3", start_date="2020-01-01", end_date="2020-12-31")
        catalog.add_spot("Current", "c.mp3", start_date="2020-01-01", end_date="2099-12-31")
        catalog.add_spot("No Dates", "n.mp3")  # no date bounds = always active
        active = catalog.list_active_spots()
        names = [s.name for s in active]
        assert "Expired" not in names
        assert "Current" in names
        assert "No Dates" in names

    def test_future_spot_excluded(self, catalog):
        catalog.add_spot("Future", "f.mp3", start_date="2099-01-01", end_date="2099-12-31")
        active = catalog.list_active_spots()
        assert len(active) == 0

    def test_toggle_spot(self, catalog):
        sid = catalog.add_spot("Test", "t.mp3")
        catalog.toggle_spot(sid, False)
        assert len(catalog.list_active_spots()) == 0
        catalog.toggle_spot(sid, True)
        assert len(catalog.list_active_spots()) == 1


# ---------------------------------------------------------------------------
# External Content
# ---------------------------------------------------------------------------


class TestExternalContent:
    def test_add_external(self, catalog):
        eid = catalog.add_external(
            "Guest Interview",
            "external/guest.mp3",
            submitter="community_member",
            license="CC-BY-4.0",
        )
        assert eid >= 1

    def test_list_external_all(self, catalog):
        catalog.add_external("A", "a.mp3")
        catalog.add_external("B", "b.mp3")
        assert len(catalog.list_external()) == 2

    def test_list_external_approved_only(self, catalog):
        catalog.add_external("Pending", "p.mp3")
        e2 = catalog.add_external("Approved", "a.mp3")
        catalog.approve_external(e2)
        approved = catalog.list_external(approved_only=True)
        assert len(approved) == 1
        assert approved[0].name == "Approved"
        assert isinstance(approved[0], ExternalContent)

    def test_external_metadata(self, catalog):
        catalog.add_external(
            "With Meta",
            "m.mp3",
            metadata={"source": "podcast", "original_url": "https://example.com"},
        )
        items = catalog.list_external()
        assert items[0].metadata == {"source": "podcast", "original_url": "https://example.com"}

    def test_approve_external(self, catalog):
        eid = catalog.add_external("Test", "t.mp3")
        catalog.approve_external(eid)
        items = catalog.list_external(approved_only=True)
        assert len(items) == 1
        assert items[0].approved is True
