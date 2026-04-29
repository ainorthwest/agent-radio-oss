"""Podcast RSS feed generator for Agent Radio.

Generates a valid RSS 2.0 feed with iTunes namespace extensions for
distribution to Apple Podcasts, Spotify, and all podcast apps.

Uses xml.etree.ElementTree from stdlib — no new dependencies.

Usage:
    from src.podcast import generate_feed

    feed_path = generate_feed()  # writes output/feed.xml

CLI:
    uv run python -m src.podcast                   # generate feed
    uv run python -m src.podcast --output feed.xml  # custom output
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import format_datetime
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element, SubElement, register_namespace, tostring

import yaml

# iTunes namespace — register so ElementTree uses "itunes:" prefix (not "ns0:")
ITUNES_NS = "http://www.itunes.com/dtds/podcast-1.0.dtd"
CONTENT_NS = "http://purl.org/rss/1.0/modules/content/"
register_namespace("itunes", ITUNES_NS)
register_namespace("content", CONTENT_NS)


@dataclass
class PodcastMetadata:
    """Show-level metadata for the podcast feed."""

    title: str = "Haystack News by AI Northwest"
    description: str = ""
    link: str = "https://ainorthwest.org/radio"
    language: str = "en-us"
    author: str = "AI Northwest"
    email: str = ""
    image_url: str = ""
    category: str = "Technology"
    subcategory: str = "Tech News"
    explicit: bool = False


@dataclass
class EpisodeEntry:
    """Per-episode metadata for an RSS item."""

    title: str
    description: str
    audio_url: str
    pub_date: datetime
    duration_seconds: int
    guid: str
    file_size_bytes: int = 0
    artwork_url: str | None = None


def load_podcast_config(
    config_path: Path = Path("config/podcast.yaml"),
) -> PodcastMetadata:
    """Load show metadata from podcast.yaml."""
    if not config_path.exists():
        return PodcastMetadata()

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    return PodcastMetadata(
        title=str(data.get("title", "Haystack News by AI Northwest")),
        description=str(data.get("description", "")),
        link=str(data.get("link", "https://ainorthwest.org/radio")),
        language=str(data.get("language", "en-us")),
        author=str(data.get("author", "AI Northwest")),
        email=str(data.get("email", "")),
        image_url=str(data.get("image_url", "")),
        category=str(data.get("category", "Technology")),
        subcategory=str(data.get("subcategory", "")),
        explicit=bool(data.get("explicit", False)),
    )


def _format_duration(seconds: int) -> str:
    """Format seconds as HH:MM:SS for iTunes duration tag."""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def build_feed(metadata: PodcastMetadata, episodes: list[EpisodeEntry]) -> str:
    """Generate podcast RSS XML string.

    Returns well-formed RSS 2.0 XML with iTunes namespace extensions.
    """
    rss = Element("rss", version="2.0")
    # Namespace declarations handled by register_namespace() at module level

    channel = SubElement(rss, "channel")

    # Required channel elements
    SubElement(channel, "title").text = metadata.title
    SubElement(channel, "link").text = metadata.link
    SubElement(channel, "language").text = metadata.language
    SubElement(channel, "description").text = metadata.description
    SubElement(channel, "lastBuildDate").text = format_datetime(datetime.now(UTC))

    # iTunes channel elements
    SubElement(channel, f"{{{ITUNES_NS}}}author").text = metadata.author
    SubElement(channel, f"{{{ITUNES_NS}}}summary").text = metadata.description
    SubElement(channel, f"{{{ITUNES_NS}}}explicit").text = "yes" if metadata.explicit else "no"

    if metadata.email:
        owner = SubElement(channel, f"{{{ITUNES_NS}}}owner")
        SubElement(owner, f"{{{ITUNES_NS}}}name").text = metadata.author
        SubElement(owner, f"{{{ITUNES_NS}}}email").text = metadata.email

    if metadata.image_url:
        SubElement(channel, f"{{{ITUNES_NS}}}image", href=metadata.image_url)

    if metadata.category:
        cat = SubElement(channel, f"{{{ITUNES_NS}}}category", text=metadata.category)
        if metadata.subcategory:
            SubElement(cat, f"{{{ITUNES_NS}}}category", text=metadata.subcategory)

    # Episode items (newest first)
    for ep in sorted(episodes, key=lambda e: e.pub_date, reverse=True):
        item = SubElement(channel, "item")
        SubElement(item, "title").text = ep.title
        SubElement(item, "description").text = ep.description
        SubElement(item, "pubDate").text = format_datetime(ep.pub_date)
        SubElement(item, "guid", isPermaLink="false").text = ep.guid

        SubElement(
            item,
            "enclosure",
            url=ep.audio_url,
            length=str(ep.file_size_bytes),
            type="audio/mpeg",
        )

        SubElement(item, f"{{{ITUNES_NS}}}duration").text = _format_duration(ep.duration_seconds)
        SubElement(item, f"{{{ITUNES_NS}}}explicit").text = "yes" if metadata.explicit else "no"
        if ep.artwork_url:
            SubElement(item, f"{{{ITUNES_NS}}}image", href=ep.artwork_url)

    # Serialize with XML declaration
    xml_bytes = tostring(rss, encoding="unicode", xml_declaration=False)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_bytes


def collect_episodes(
    episodes_dir: Path = Path("output/episodes"),
    public_url_base: str = "",
    library_root: Path | None = None,
) -> list[EpisodeEntry]:
    """Scan episode directories for manifests and build episode entries.

    If library_root is set, scans library/programs/*/episodes/ instead of
    the legacy output/episodes/ path. This enables multi-program feeds.

    Looks for manifest.json in each date-named subdirectory.
    The audio URL is constructed from public_url_base + R2 key pattern.
    """
    entries: list[EpisodeEntry] = []

    # Collect all (date_dir, program_slug) pairs to scan
    scan_dirs: list[tuple[Path, str | None]] = []
    if library_root is not None:
        programs_dir = library_root / "programs"
        if programs_dir.exists():
            for prog_dir in sorted(programs_dir.iterdir()):
                if not prog_dir.is_dir():
                    continue
                ep_dir = prog_dir / "episodes"
                if ep_dir.exists():
                    for date_dir in sorted(ep_dir.iterdir()):
                        if date_dir.is_dir():
                            scan_dirs.append((date_dir, prog_dir.name))
    else:
        if episodes_dir.exists():
            for date_dir in sorted(episodes_dir.iterdir()):
                if date_dir.is_dir():
                    scan_dirs.append((date_dir, None))

    for date_dir, program_slug in scan_dirs:
        manifest_path = date_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        with open(manifest_path) as f:
            manifest: dict[str, Any] = json.load(f)

        date_str = manifest.get("date", date_dir.name)
        title = str(manifest.get("title", f"Episode — {date_str}"))

        # Find the latest episode MP3
        mp3_files = sorted(date_dir.glob("episode_*.mp3"))
        if not mp3_files:
            mp3_files = sorted(date_dir.glob("episode.mp3"))
        if not mp3_files:
            continue

        mp3_path = mp3_files[-1]  # latest numbered episode
        file_size = mp3_path.stat().st_size

        # Estimate duration from manifest segments
        segments = manifest.get("segments", [])
        total_duration = sum(float(s.get("duration_seconds", 0)) for s in segments)
        if total_duration == 0:
            # Fallback: estimate from file size (128kbps MP3 ≈ 16KB/s)
            total_duration = file_size / 16000

        # Build audio URL — library uses program-scoped R2 keys
        if program_slug:
            r2_key = f"programs/{program_slug}/{date_str}.mp3"
        else:
            r2_key = f"episodes/{date_str}.mp3"
        audio_url = f"{public_url_base.rstrip('/')}/{r2_key}" if public_url_base else r2_key

        # Parse date
        try:
            pub_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError:
            pub_date = datetime.now(UTC)

        # Resolve per-episode artwork URL from manifest artwork_path
        artwork_url: str | None = None
        artwork_path_str = manifest.get("artwork_path")
        if artwork_path_str and library_root and public_url_base:
            try:
                rel = Path(artwork_path_str).relative_to(library_root)
                artwork_url = f"{public_url_base.rstrip('/')}/{rel.as_posix()}"
            except ValueError:
                print(
                    f"  WARNING: artwork_path {artwork_path_str!r} not under "
                    f"library_root {library_root} — skipping per-item artwork",
                    file=sys.stderr,
                )

        entries.append(
            EpisodeEntry(
                title=title,
                description=f"{title} — {len(segments)} segments",
                audio_url=audio_url,
                pub_date=pub_date,
                duration_seconds=int(total_duration),
                guid=r2_key,
                file_size_bytes=file_size,
                artwork_url=artwork_url,
            )
        )

    return entries


def generate_feed(
    config_path: Path = Path("config/podcast.yaml"),
    output_path: Path | None = None,
    episodes_dir: Path = Path("output/episodes"),
    public_url_base: str = "",
    library_root: Path | None = None,
) -> Path:
    """Generate podcast RSS feed from episode manifests.

    Args:
        config_path: Path to podcast.yaml with show metadata.
        output_path: Where to write the feed XML. Defaults to
                     library_root/feed.xml when library_root is set,
                     otherwise output/feed.xml (legacy).
        episodes_dir: Directory containing date-named episode subdirs (legacy mode).
        public_url_base: Base URL for audio file links (e.g. R2 public URL).
        library_root: If set, scans library/programs/*/episodes/ instead of episodes_dir.

    Returns:
        Path to the generated feed.xml file.
    """
    metadata = load_podcast_config(config_path)
    episodes = collect_episodes(episodes_dir, public_url_base, library_root=library_root)

    xml = build_feed(metadata, episodes)

    # Resolve output path: library_root-relative when available, legacy fallback otherwise
    if output_path is None:
        output_path = (
            library_root / "feed.xml" if library_root is not None else Path("output/feed.xml")
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(xml)

    print(f"Podcast feed: {output_path} ({len(episodes)} episodes)")
    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate podcast RSS feed from Agent Radio episodes."
    )
    parser.add_argument("--config", default="config/podcast.yaml", help="Podcast config YAML")
    parser.add_argument("--output", default="output/feed.xml", help="Output feed XML path")
    parser.add_argument("--episodes-dir", default="output/episodes", help="Episodes directory")
    parser.add_argument("--url-base", default="", help="Public URL base for audio links")

    args = parser.parse_args()
    generate_feed(
        config_path=Path(args.config),
        output_path=Path(args.output),
        episodes_dir=Path(args.episodes_dir),
        public_url_base=args.url_base,
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
