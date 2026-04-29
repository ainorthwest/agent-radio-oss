"""Station library catalog — SQLite-backed content registry.

Tracks programs, episodes, music tracks, assets, distributions, feedback,
sponsor spots, and external content. Single-file database, no server needed.

Usage:
    from src.library import Catalog
    catalog = Catalog(Path("library/radio.db"))
    catalog.register_program("haystack-news", "Haystack News", "talk")
    catalog.add_episode("haystack-news", "2026-03-19", "programs/haystack-news/episodes/2026-03-19/episode.mp3", 320.0)
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Program:
    id: int
    slug: str
    name: str
    program_type: str  # "talk" | "music"
    status: str  # "active" | "dev" | "retired"
    schedule_cron: str | None
    palette_path: str | None
    cast_config: dict[str, Any] | None
    created_at: str
    updated_at: str


@dataclass
class Episode:
    id: int
    program_slug: str
    date: str
    file_path: str
    script_path: str | None
    manifest_path: str | None
    duration_seconds: float | None
    quality_score: float | None
    script_score: float | None
    dnsmos_score: float | None
    segment_count: int | None
    status: str
    created_at: str
    reviewed_at: str | None
    approved_at: str | None


@dataclass
class Track:
    id: int
    program_slug: str
    title: str
    file_path: str
    date: str
    duration_seconds: float | None
    prompt: str | None
    quality_score: float | None
    status: str
    play_count: int
    heart_count: int
    created_at: str
    reviewed_at: str | None
    approved_at: str | None
    scheduled_at: str | None
    aired_at: str | None
    rejected_at: str | None
    archived_at: str | None


@dataclass
class Asset:
    id: int
    filename: str
    asset_type: str
    scope: str
    program_slug: str | None
    file_path: str
    duration_seconds: float | None
    metadata: dict[str, Any] | None
    created_at: str


@dataclass
class Distribution:
    id: int
    content_type: str
    content_id: int
    destination: str
    url: str | None
    success: bool
    error_message: str | None
    created_at: str


@dataclass
class Feedback:
    id: int
    content_type: str
    content_id: int
    feedback_type: str
    source: str
    created_at: str


@dataclass
class Spot:
    id: int
    name: str
    sponsor: str | None
    file_path: str
    duration_seconds: float | None
    active: bool
    rotation_weight: float
    start_date: str | None
    end_date: str | None
    created_at: str


@dataclass
class ExternalContent:
    id: int
    name: str
    submitter: str | None
    file_path: str
    duration_seconds: float | None
    approved: bool
    license: str | None
    metadata: dict[str, Any] | None
    created_at: str


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS programs (
    id INTEGER PRIMARY KEY,
    slug TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    program_type TEXT NOT NULL DEFAULT 'talk',
    status TEXT NOT NULL DEFAULT 'active',
    schedule_cron TEXT,
    palette_path TEXT,
    cast_config TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY,
    program_slug TEXT NOT NULL REFERENCES programs(slug),
    date TEXT NOT NULL,
    file_path TEXT NOT NULL,
    script_path TEXT,
    manifest_path TEXT,
    duration_seconds REAL,
    quality_score REAL,
    script_score REAL,
    dnsmos_score REAL,
    segment_count INTEGER,
    status TEXT NOT NULL DEFAULT 'generated',
    created_at TEXT NOT NULL,
    reviewed_at TEXT,
    approved_at TEXT,
    UNIQUE(program_slug, date)
);

CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY,
    program_slug TEXT NOT NULL REFERENCES programs(slug),
    title TEXT NOT NULL,
    file_path TEXT NOT NULL,
    date TEXT NOT NULL,
    duration_seconds REAL,
    prompt TEXT,
    quality_score REAL,
    status TEXT NOT NULL DEFAULT 'generated',
    play_count INTEGER NOT NULL DEFAULT 0,
    heart_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    reviewed_at TEXT,
    approved_at TEXT,
    scheduled_at TEXT,
    aired_at TEXT,
    rejected_at TEXT,
    archived_at TEXT
);

CREATE TABLE IF NOT EXISTS assets (
    id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    scope TEXT NOT NULL,
    program_slug TEXT REFERENCES programs(slug),
    file_path TEXT NOT NULL,
    duration_seconds REAL,
    metadata TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS distributions (
    id INTEGER PRIMARY KEY,
    content_type TEXT NOT NULL,
    content_id INTEGER NOT NULL,
    destination TEXT NOT NULL,
    url TEXT,
    success INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY,
    content_type TEXT NOT NULL,
    content_id INTEGER NOT NULL,
    feedback_type TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'web',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS spots (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    sponsor TEXT,
    file_path TEXT NOT NULL,
    duration_seconds REAL,
    active INTEGER NOT NULL DEFAULT 1,
    rotation_weight REAL DEFAULT 1.0,
    start_date TEXT,
    end_date TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS external_content (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    submitter TEXT,
    file_path TEXT NOT NULL,
    duration_seconds REAL,
    approved INTEGER NOT NULL DEFAULT 0,
    license TEXT,
    metadata TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_episodes_program ON episodes(program_slug);
CREATE INDEX IF NOT EXISTS idx_episodes_date ON episodes(date);
CREATE INDEX IF NOT EXISTS idx_tracks_program ON tracks(program_slug);
CREATE INDEX IF NOT EXISTS idx_tracks_date ON tracks(date);
CREATE INDEX IF NOT EXISTS idx_distributions_content ON distributions(content_type, content_id);
CREATE INDEX IF NOT EXISTS idx_feedback_content ON feedback(content_type, content_id);
CREATE INDEX IF NOT EXISTS idx_assets_type ON assets(asset_type);
CREATE INDEX IF NOT EXISTS idx_assets_scope ON assets(scope);
"""


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


class Catalog:
    """SQLite-backed station content catalog."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Catalog:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- Programs -----------------------------------------------------------

    def register_program(
        self,
        slug: str,
        name: str,
        program_type: str = "talk",
        status: str = "active",
        schedule_cron: str | None = None,
        palette_path: str | None = None,
        cast_config: dict[str, Any] | None = None,
    ) -> int:
        now = _now()
        cast_json = json.dumps(cast_config) if cast_config else None
        cur = self._conn.execute(
            """INSERT INTO programs
               (slug, name, program_type, status, schedule_cron, palette_path, cast_config, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (slug, name, program_type, status, schedule_cron, palette_path, cast_json, now, now),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_program(self, slug: str) -> Program | None:
        row = self._conn.execute("SELECT * FROM programs WHERE slug = ?", (slug,)).fetchone()
        if not row:
            return None
        return self._row_to_program(row)

    def list_programs(self, status: str | None = None) -> list[Program]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM programs WHERE status = ? ORDER BY name", (status,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM programs ORDER BY name").fetchall()
        return [self._row_to_program(r) for r in rows]

    def update_program(self, slug: str, **kwargs: Any) -> None:
        # Safety: `allowed` is the security boundary. Column names from this set
        # are interpolated into SQL. Only add real column names here — never user input.
        allowed = {"name", "program_type", "status", "schedule_cron", "palette_path", "cast_config"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        if "cast_config" in updates and isinstance(updates["cast_config"], dict):
            updates["cast_config"] = json.dumps(updates["cast_config"])
        updates["updated_at"] = _now()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [slug]
        self._conn.execute(f"UPDATE programs SET {set_clause} WHERE slug = ?", values)
        self._conn.commit()

    @staticmethod
    def _row_to_program(row: sqlite3.Row) -> Program:
        cast = json.loads(row["cast_config"]) if row["cast_config"] else None
        return Program(
            id=row["id"],
            slug=row["slug"],
            name=row["name"],
            program_type=row["program_type"],
            status=row["status"],
            schedule_cron=row["schedule_cron"],
            palette_path=row["palette_path"],
            cast_config=cast,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # -- Episodes -----------------------------------------------------------

    def add_episode(
        self,
        program_slug: str,
        date: str,
        file_path: str,
        duration_seconds: float | None = None,
        quality_score: float | None = None,
        script_score: float | None = None,
        dnsmos_score: float | None = None,
        segment_count: int | None = None,
        script_path: str | None = None,
        manifest_path: str | None = None,
    ) -> int:
        import sqlite3

        try:
            cur = self._conn.execute(
                """INSERT INTO episodes
                   (program_slug, date, file_path, script_path, manifest_path,
                    duration_seconds, quality_score, script_score, dnsmos_score,
                    segment_count, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    program_slug,
                    date,
                    file_path,
                    script_path,
                    manifest_path,
                    duration_seconds,
                    quality_score,
                    script_score,
                    dnsmos_score,
                    segment_count,
                    _now(),
                ),
            )
            self._conn.commit()
            return cur.lastrowid  # type: ignore[return-value]
        except sqlite3.IntegrityError:
            # If a duplicate (program_slug, date) row exists, return its ID.
            # Otherwise re-raise — this is a FK violation or other real error.
            row = self._conn.execute(
                "SELECT id FROM episodes WHERE program_slug = ? AND date = ?",
                (program_slug, date),
            ).fetchone()
            if row:
                return row["id"]
            raise

    def get_episode(self, program_slug: str, date: str) -> Episode | None:
        row = self._conn.execute(
            "SELECT * FROM episodes WHERE program_slug = ? AND date = ?",
            (program_slug, date),
        ).fetchone()
        if not row:
            return None
        return self._row_to_episode(row)

    def list_episodes(self, program_slug: str | None = None, limit: int = 50) -> list[Episode]:
        if program_slug:
            rows = self._conn.execute(
                "SELECT * FROM episodes WHERE program_slug = ? ORDER BY date DESC LIMIT ?",
                (program_slug, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM episodes ORDER BY date DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def latest_episode(self, program_slug: str) -> Episode | None:
        row = self._conn.execute(
            "SELECT * FROM episodes WHERE program_slug = ? ORDER BY date DESC LIMIT 1",
            (program_slug,),
        ).fetchone()
        if not row:
            return None
        return self._row_to_episode(row)

    def update_episode(self, episode_id: int, **kwargs: Any) -> None:
        # Safety: `allowed` is the security boundary. Column names from this set
        # are interpolated into SQL. Only add real column names here — never user input.
        allowed = {
            "file_path",
            "script_path",
            "manifest_path",
            "duration_seconds",
            "quality_score",
            "script_score",
            "dnsmos_score",
            "segment_count",
            "status",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [episode_id]
        self._conn.execute(f"UPDATE episodes SET {set_clause} WHERE id = ?", values)
        self._conn.commit()

    @staticmethod
    def _row_to_episode(row: sqlite3.Row) -> Episode:
        return Episode(
            id=row["id"],
            program_slug=row["program_slug"],
            date=row["date"],
            file_path=row["file_path"],
            script_path=row["script_path"],
            manifest_path=row["manifest_path"],
            duration_seconds=row["duration_seconds"],
            quality_score=row["quality_score"],
            script_score=row["script_score"],
            dnsmos_score=row["dnsmos_score"],
            segment_count=row["segment_count"],
            status=row["status"],
            created_at=row["created_at"],
            reviewed_at=row["reviewed_at"],
            approved_at=row["approved_at"],
        )

    # -- Tracks -------------------------------------------------------------

    def add_track(
        self,
        program_slug: str,
        title: str,
        file_path: str,
        date: str,
        duration_seconds: float | None = None,
        prompt: str | None = None,
        quality_score: float | None = None,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO tracks
               (program_slug, title, file_path, date, duration_seconds, prompt, quality_score, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (program_slug, title, file_path, date, duration_seconds, prompt, quality_score, _now()),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_track(self, track_id: int) -> Track | None:
        row = self._conn.execute("SELECT * FROM tracks WHERE id = ?", (track_id,)).fetchone()
        if not row:
            return None
        return self._row_to_track(row)

    def list_tracks(self, program_slug: str | None = None, limit: int = 50) -> list[Track]:
        if program_slug:
            rows = self._conn.execute(
                "SELECT * FROM tracks WHERE program_slug = ? ORDER BY date DESC, id DESC LIMIT ?",
                (program_slug, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM tracks ORDER BY date DESC, id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_track(r) for r in rows]

    def get_top_rated(
        self,
        program_slug: str | None = None,
        limit: int = 20,
    ) -> list[Track]:
        if program_slug:
            rows = self._conn.execute(
                "SELECT * FROM tracks WHERE program_slug = ? ORDER BY heart_count DESC LIMIT ?",
                (program_slug, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM tracks ORDER BY heart_count DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_track(r) for r in rows]

    def increment_play_count(self, track_id: int) -> None:
        self._conn.execute(
            "UPDATE tracks SET play_count = play_count + 1 WHERE id = ?", (track_id,)
        )
        self._conn.commit()

    @staticmethod
    def _row_to_track(row: sqlite3.Row) -> Track:
        return Track(
            id=row["id"],
            program_slug=row["program_slug"],
            title=row["title"],
            file_path=row["file_path"],
            date=row["date"],
            duration_seconds=row["duration_seconds"],
            prompt=row["prompt"],
            quality_score=row["quality_score"],
            status=row["status"],
            play_count=row["play_count"],
            heart_count=row["heart_count"],
            created_at=row["created_at"],
            reviewed_at=row["reviewed_at"],
            approved_at=row["approved_at"],
            scheduled_at=row["scheduled_at"],
            aired_at=row["aired_at"],
            rejected_at=row["rejected_at"],
            archived_at=row["archived_at"],
        )

    # -- Lifecycle transitions ----------------------------------------------

    # Tracks: full production lifecycle through broadcast
    TRACK_TRANSITIONS: dict[str, set[str]] = {
        "generated": {"reviewed", "rejected"},
        "reviewed": {"approved", "rejected"},
        "approved": {"scheduled", "rejected"},
        "scheduled": {"aired"},
        "aired": {"archived"},
        "rejected": {"generated"},  # allow re-generation
    }

    # Episodes: shorter lifecycle (no scheduling/airing — distribution is the endpoint)
    EPISODE_TRANSITIONS: dict[str, set[str]] = {
        "generated": {"reviewed", "rejected"},
        "reviewed": {"approved", "rejected"},
        "approved": {"distributed", "rejected"},
        "distributed": {"archived"},
        "rejected": {"generated"},
    }

    def set_track_status(self, track_id: int, new_status: str) -> None:
        """Transition a track to a new lifecycle state.

        Clears stale timestamps when re-entering 'generated' from 'rejected'.
        """
        track = self.get_track(track_id)
        if track is None:
            raise ValueError(f"Track {track_id} not found")
        allowed = self.TRACK_TRANSITIONS.get(track.status, set())
        if new_status not in allowed:
            raise ValueError(
                f"Invalid transition: {track.status} → {new_status}. Allowed: {allowed}"
            )
        now = _now()

        # Re-entering generated: clear all lifecycle timestamps
        if new_status == "generated":
            self._conn.execute(
                """UPDATE tracks SET status = 'generated',
                   reviewed_at = NULL, approved_at = NULL,
                   scheduled_at = NULL, aired_at = NULL,
                   rejected_at = NULL, archived_at = NULL
                   WHERE id = ?""",
                (track_id,),
            )
            self._conn.commit()
            return

        # Map status to its timestamp column
        timestamp_col = {
            "reviewed": "reviewed_at",
            "approved": "approved_at",
            "scheduled": "scheduled_at",
            "aired": "aired_at",
            "rejected": "rejected_at",
            "archived": "archived_at",
        }.get(new_status)
        if timestamp_col:
            self._conn.execute(
                f"UPDATE tracks SET status = ?, {timestamp_col} = ? WHERE id = ?",
                (new_status, now, track_id),
            )
        else:
            self._conn.execute(
                "UPDATE tracks SET status = ? WHERE id = ?",
                (new_status, track_id),
            )
        self._conn.commit()

    def set_episode_status(self, episode_id: int, new_status: str) -> None:
        """Transition an episode to a new lifecycle state."""
        row = self._conn.execute(
            "SELECT status FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Episode {episode_id} not found")
        current = row["status"]
        allowed = self.EPISODE_TRANSITIONS.get(current, set())
        if new_status not in allowed:
            raise ValueError(f"Invalid transition: {current} → {new_status}. Allowed: {allowed}")
        now = _now()
        timestamp_col = {
            "reviewed": "reviewed_at",
            "approved": "approved_at",
        }.get(new_status)
        if timestamp_col:
            self._conn.execute(
                f"UPDATE episodes SET status = ?, {timestamp_col} = ? WHERE id = ?",
                (new_status, now, episode_id),
            )
        else:
            self._conn.execute(
                "UPDATE episodes SET status = ? WHERE id = ?",
                (new_status, episode_id),
            )
        self._conn.commit()

    def list_tracks_by_status(
        self, status: str, program_slug: str | None = None, limit: int = 50
    ) -> list[Track]:
        """List tracks filtered by lifecycle status."""
        if program_slug:
            rows = self._conn.execute(
                "SELECT * FROM tracks WHERE status = ? AND program_slug = ? ORDER BY date DESC LIMIT ?",
                (status, program_slug, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM tracks WHERE status = ? ORDER BY date DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        return [self._row_to_track(r) for r in rows]

    # -- Assets -------------------------------------------------------------

    def register_asset(
        self,
        filename: str,
        asset_type: str,
        scope: str,
        file_path: str,
        program_slug: str | None = None,
        duration_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        meta_json = json.dumps(metadata) if metadata else None
        cur = self._conn.execute(
            """INSERT INTO assets
               (filename, asset_type, scope, program_slug, file_path, duration_seconds, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                filename,
                asset_type,
                scope,
                program_slug,
                file_path,
                duration_seconds,
                meta_json,
                _now(),
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def find_assets(
        self,
        asset_type: str | None = None,
        scope: str | None = None,
        program_slug: str | None = None,
    ) -> list[Asset]:
        conditions: list[str] = []
        params: list[Any] = []
        if asset_type:
            conditions.append("asset_type = ?")
            params.append(asset_type)
        if scope:
            conditions.append("scope = ?")
            params.append(scope)
        if program_slug:
            conditions.append("program_slug = ?")
            params.append(program_slug)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self._conn.execute(
            f"SELECT * FROM assets{where} ORDER BY filename", params
        ).fetchall()
        return [self._row_to_asset(r) for r in rows]

    @staticmethod
    def _row_to_asset(row: sqlite3.Row) -> Asset:
        meta = json.loads(row["metadata"]) if row["metadata"] else None
        return Asset(
            id=row["id"],
            filename=row["filename"],
            asset_type=row["asset_type"],
            scope=row["scope"],
            program_slug=row["program_slug"],
            file_path=row["file_path"],
            duration_seconds=row["duration_seconds"],
            metadata=meta,
            created_at=row["created_at"],
        )

    # -- Distributions ------------------------------------------------------

    def record_distribution(
        self,
        content_type: str,
        content_id: int,
        destination: str,
        url: str | None = None,
        success: bool = True,
        error_message: str | None = None,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO distributions
               (content_type, content_id, destination, url, success, error_message, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (content_type, content_id, destination, url, int(success), error_message, _now()),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_distributions(self, content_type: str, content_id: int) -> list[Distribution]:
        rows = self._conn.execute(
            "SELECT * FROM distributions WHERE content_type = ? AND content_id = ? ORDER BY created_at DESC",
            (content_type, content_id),
        ).fetchall()
        return [self._row_to_distribution(r) for r in rows]

    @staticmethod
    def _row_to_distribution(row: sqlite3.Row) -> Distribution:
        return Distribution(
            id=row["id"],
            content_type=row["content_type"],
            content_id=row["content_id"],
            destination=row["destination"],
            url=row["url"],
            success=bool(row["success"]),
            error_message=row["error_message"],
            created_at=row["created_at"],
        )

    # -- Feedback -----------------------------------------------------------

    def record_feedback(
        self,
        content_type: str,
        content_id: int,
        feedback_type: str = "heart",
        source: str = "web",
    ) -> int:
        """Record a feedback event (heart, skip, etc.).

        Note: tracks.heart_count is a denormalized cache updated here for fast
        queries. If feedback rows are ever deleted directly, call
        recount_hearts() to resync.
        """
        cur = self._conn.execute(
            """INSERT INTO feedback (content_type, content_id, feedback_type, source, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (content_type, content_id, feedback_type, source, _now()),
        )
        # Update heart_count on tracks table for fast queries
        if content_type == "track" and feedback_type == "heart":
            self._conn.execute(
                "UPDATE tracks SET heart_count = heart_count + 1 WHERE id = ?",
                (content_id,),
            )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def recount_hearts(self, track_id: int) -> int:
        """Resync tracks.heart_count from the feedback table (repair method)."""
        row = self._conn.execute(
            """SELECT COUNT(*) as cnt FROM feedback
               WHERE content_type = 'track' AND content_id = ? AND feedback_type = 'heart'""",
            (track_id,),
        ).fetchone()
        count = row["cnt"] if row else 0
        self._conn.execute("UPDATE tracks SET heart_count = ? WHERE id = ?", (count, track_id))
        self._conn.commit()
        return count

    def get_feedback(self, content_type: str, content_id: int) -> list[Feedback]:
        rows = self._conn.execute(
            "SELECT * FROM feedback WHERE content_type = ? AND content_id = ? ORDER BY created_at DESC",
            (content_type, content_id),
        ).fetchall()
        return [self._row_to_feedback(r) for r in rows]

    def count_feedback(
        self,
        content_type: str,
        content_id: int,
        feedback_type: str = "heart",
    ) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM feedback WHERE content_type = ? AND content_id = ? AND feedback_type = ?",
            (content_type, content_id, feedback_type),
        ).fetchone()
        return row["cnt"] if row else 0

    def aggregate_feedback(
        self,
        content_type: str,
        feedback_type: str = "heart",
        limit: int = 20,
    ) -> list[tuple[int, int]]:
        """Return (content_id, count) pairs ordered by most feedback."""
        rows = self._conn.execute(
            """SELECT content_id, COUNT(*) as cnt
               FROM feedback
               WHERE content_type = ? AND feedback_type = ?
               GROUP BY content_id
               ORDER BY cnt DESC
               LIMIT ?""",
            (content_type, feedback_type, limit),
        ).fetchall()
        return [(row["content_id"], row["cnt"]) for row in rows]

    @staticmethod
    def _row_to_feedback(row: sqlite3.Row) -> Feedback:
        return Feedback(
            id=row["id"],
            content_type=row["content_type"],
            content_id=row["content_id"],
            feedback_type=row["feedback_type"],
            source=row["source"],
            created_at=row["created_at"],
        )

    # -- Spots --------------------------------------------------------------

    def add_spot(
        self,
        name: str,
        file_path: str,
        sponsor: str | None = None,
        duration_seconds: float | None = None,
        rotation_weight: float = 1.0,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO spots
               (name, sponsor, file_path, duration_seconds, rotation_weight, start_date, end_date, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                name,
                sponsor,
                file_path,
                duration_seconds,
                rotation_weight,
                start_date,
                end_date,
                _now(),
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def list_active_spots(self) -> list[Spot]:
        today = datetime.now(UTC).date().isoformat()
        rows = self._conn.execute(
            """SELECT * FROM spots
               WHERE active = 1
                 AND (start_date IS NULL OR start_date <= ?)
                 AND (end_date IS NULL OR end_date >= ?)
               ORDER BY name""",
            (today, today),
        ).fetchall()
        return [self._row_to_spot(r) for r in rows]

    def toggle_spot(self, spot_id: int, active: bool) -> None:
        self._conn.execute("UPDATE spots SET active = ? WHERE id = ?", (int(active), spot_id))
        self._conn.commit()

    @staticmethod
    def _row_to_spot(row: sqlite3.Row) -> Spot:
        return Spot(
            id=row["id"],
            name=row["name"],
            sponsor=row["sponsor"],
            file_path=row["file_path"],
            duration_seconds=row["duration_seconds"],
            active=bool(row["active"]),
            rotation_weight=row["rotation_weight"],
            start_date=row["start_date"],
            end_date=row["end_date"],
            created_at=row["created_at"],
        )

    # -- External Content ---------------------------------------------------

    def add_external(
        self,
        name: str,
        file_path: str,
        submitter: str | None = None,
        duration_seconds: float | None = None,
        license: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        meta_json = json.dumps(metadata) if metadata else None
        cur = self._conn.execute(
            """INSERT INTO external_content
               (name, submitter, file_path, duration_seconds, license, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (name, submitter, file_path, duration_seconds, license, meta_json, _now()),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def list_external(self, approved_only: bool = False) -> list[ExternalContent]:
        if approved_only:
            rows = self._conn.execute(
                "SELECT * FROM external_content WHERE approved = 1 ORDER BY name"
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM external_content ORDER BY name").fetchall()
        return [self._row_to_external(r) for r in rows]

    def approve_external(self, external_id: int) -> None:
        self._conn.execute("UPDATE external_content SET approved = 1 WHERE id = ?", (external_id,))
        self._conn.commit()

    @staticmethod
    def _row_to_external(row: sqlite3.Row) -> ExternalContent:
        meta = json.loads(row["metadata"]) if row["metadata"] else None
        return ExternalContent(
            id=row["id"],
            name=row["name"],
            submitter=row["submitter"],
            file_path=row["file_path"],
            duration_seconds=row["duration_seconds"],
            approved=bool(row["approved"]),
            license=row["license"],
            metadata=meta,
            created_at=row["created_at"],
        )
