"""Library path resolver — deterministic content location within the station library.

All modules use this instead of building paths manually. Every content type
has a predictable home, and R2 keys follow the same structure.

Usage:
    from src.paths import LibraryPaths
    paths = LibraryPaths(Path("library"))
    episode_dir = paths.episode_dir("haystack-news", "2026-03-19")
    # => library/programs/haystack-news/episodes/2026-03-19/
"""

from __future__ import annotations

from pathlib import Path


class LibraryPaths:
    """Resolves content paths within the station library structure."""

    def __init__(self, root: Path = Path("library")) -> None:
        self.root = root

    # -- Database -----------------------------------------------------------

    @property
    def db(self) -> Path:
        return self.root / "radio.db"

    # -- Programs -----------------------------------------------------------

    def program_dir(self, slug: str) -> Path:
        return self.root / "programs" / slug

    def program_config(self, slug: str) -> Path:
        return self.program_dir(slug) / "program.yaml"

    def program_assets(self, slug: str) -> Path:
        return self.program_dir(slug) / "assets"

    def program_artwork(self, slug: str) -> Path:
        """Convention: library/programs/{slug}/artwork/cover.png"""
        return self.program_dir(slug) / "artwork" / "cover.png"

    def station_artwork(self) -> Path:
        """Convention: library/station/artwork/cover.png — fallback for all programs."""
        return self.root / "station" / "artwork" / "cover.png"

    # -- Episodes (talk programs) -------------------------------------------

    def episode_dir(self, program_slug: str, date: str) -> Path:
        return self.program_dir(program_slug) / "episodes" / date

    def episode_audio(self, program_slug: str, date: str) -> Path:
        return self.episode_dir(program_slug, date) / "episode.mp3"

    def episode_script(self, program_slug: str, date: str) -> Path:
        return self.episode_dir(program_slug, date) / "script.json"

    def episode_manifest(self, program_slug: str, date: str) -> Path:
        return self.episode_dir(program_slug, date) / "manifest.json"

    def episode_segments(self, program_slug: str, date: str) -> Path:
        return self.episode_dir(program_slug, date) / "segments"

    # -- Tracks (music programs) --------------------------------------------

    def track_dir(self, program_slug: str, date: str) -> Path:
        return self.program_dir(program_slug) / "tracks" / date

    def track_audio(self, program_slug: str, date: str, track_number: int) -> Path:
        return self.track_dir(program_slug, date) / f"track-{track_number:03d}.wav"

    # -- Sets (assembled music playlists) -----------------------------------

    def set_dir(self, program_slug: str, date: str) -> Path:
        return self.program_dir(program_slug) / "sets" / date

    def set_audio(self, program_slug: str, date: str) -> Path:
        return self.set_dir(program_slug, date) / "set.mp3"

    def set_manifest(self, program_slug: str, date: str) -> Path:
        return self.set_dir(program_slug, date) / "manifest.json"

    def set_dj_segments(self, program_slug: str, date: str) -> Path:
        return self.set_dir(program_slug, date) / "dj-segments"

    # -- Specials -----------------------------------------------------------

    def special_dir(self, slug: str) -> Path:
        return self.root / "specials" / slug

    # -- External content ---------------------------------------------------

    def external_dir(self, slug: str) -> Path:
        return self.root / "external" / slug

    # -- Spots --------------------------------------------------------------

    def spot_dir(self, slug: str) -> Path:
        return self.root / "spots" / slug

    # -- Station identity ---------------------------------------------------

    def station_dir(self) -> Path:
        return self.root / "station"

    def station_ids(self) -> Path:
        return self.station_dir() / "ids"

    def station_promos(self) -> Path:
        return self.station_dir() / "promos"

    def station_bumpers(self) -> Path:
        return self.station_dir() / "bumpers"

    # -- Shared assets ------------------------------------------------------

    def shared(self) -> Path:
        return self.root / "shared"

    def shared_music(self) -> Path:
        return self.shared() / "music"

    def shared_voices(self) -> Path:
        return self.shared() / "voices"

    def shared_sfx(self) -> Path:
        return self.shared() / "sfx"

    # -- R2 keys ------------------------------------------------------------

    def r2_episode_key(self, program_slug: str, date: str) -> str:
        return f"programs/{program_slug}/{date}.mp3"

    def r2_track_key(self, program_slug: str, date: str, track_number: int) -> str:
        return f"programs/{program_slug}/tracks/{date}/track-{track_number:03d}.wav"

    def r2_set_key(self, program_slug: str, date: str) -> str:
        return f"programs/{program_slug}/sets/{date}.mp3"

    def r2_feed_key(self, program_slug: str | None = None) -> str:
        if program_slug:
            return f"programs/{program_slug}/feed.xml"
        return "feed.xml"

    # -- Newsroom -----------------------------------------------------------

    def brief_path(self, program_slug: str, date: str) -> Path:
        return self.episode_dir(program_slug, date) / "brief.json"

    def wire_desk_notes(self, program_slug: str, date: str) -> Path:
        return self.episode_dir(program_slug, date) / "wire-desk-notes.md"

    def editor_notes(self, program_slug: str, date: str) -> Path:
        return self.episode_dir(program_slug, date) / "editor-notes.md"

    def bard_notes(self, program_slug: str, date: str) -> Path:
        return self.episode_dir(program_slug, date) / "bard-notes.md"

    # -- Scaffold -----------------------------------------------------------

    def ensure_structure(self) -> None:
        """Create the full library directory tree."""
        dirs = [
            self.root / "programs",
            self.root / "specials",
            self.root / "external",
            self.root / "spots",
            self.station_ids(),
            self.station_promos(),
            self.station_bumpers(),
            self.shared_music(),
            self.shared_voices(),
            self.shared_sfx(),
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def ensure_program(self, slug: str, program_type: str = "talk") -> None:
        """Create directory structure for a program."""
        self.program_dir(slug).mkdir(parents=True, exist_ok=True)
        self.program_assets(slug).mkdir(exist_ok=True)
        if program_type == "talk":
            (self.program_dir(slug) / "episodes").mkdir(exist_ok=True)
        elif program_type == "music":
            (self.program_dir(slug) / "tracks").mkdir(exist_ok=True)
            (self.program_dir(slug) / "sets").mkdir(exist_ok=True)
