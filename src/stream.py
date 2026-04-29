"""AzuraCast live streaming API client for Agent Radio.

Complete programmatic control over an AzuraCast radio station:
media management, playlist scheduling, station control, analytics.

AzuraCast REST API docs: https://www.azuracast.com/docs/developers/apis/

Usage:
    from src.stream import AzuraCastConfig, upload_media, update_episode

    config = AzuraCastConfig(
        base_url="https://radio.ainorthwest.org",
        api_key=get_secret("AGENT_RADIO_AZURACAST_API_KEY"),
    )

    # Rolling playlist update (recommended for daily episodes):
    update_episode(config, Path("episode.mp3"), "Haystack News", {"title": "..."})

    # Schedule a show for weekdays 8-8:30am:
    pid = create_playlist(config, "Haystack News", playlist_type="default")
    set_playlist_schedule(config, pid, [
        {"start_time": "0800", "end_time": "0830", "days": [1,2,3,4,5]}
    ])

CLI:
    uv run python -m src.stream status
    uv run python -m src.stream health
    uv run python -m src.stream schedule
    uv run python -m src.stream listeners
    uv run python -m src.stream history
    uv run python -m src.stream queue
    uv run python -m src.stream upload episode.mp3
    uv run python -m src.stream update episode.mp3 --playlist "Haystack News"
    uv run python -m src.stream playlists
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass
class AzuraCastConfig:
    """AzuraCast server connection configuration."""

    base_url: str = ""  # e.g. https://radio.ainorthwest.org
    api_key: str = ""  # from AGENT_RADIO_AZURACAST_API_KEY env var
    station_id: int = 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _headers(config: AzuraCastConfig) -> dict[str, str]:
    """Build request headers with API key auth."""
    return {"X-API-Key": config.api_key, "Accept": "application/json"}


def _station_url(config: AzuraCastConfig, path: str) -> str:
    """Build a station-scoped API URL."""
    base = config.base_url.rstrip("/")
    return f"{base}/api/station/{config.station_id}/{path}"


def _get(config: AzuraCastConfig, path: str, timeout: int = 30) -> Any:
    """GET a station-scoped endpoint, return parsed JSON."""
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(_station_url(config, path), headers=_headers(config))
        resp.raise_for_status()
    return resp.json()


def _post(
    config: AzuraCastConfig, path: str, payload: dict[str, Any] | None = None, timeout: int = 30
) -> Any:
    """POST JSON to a station-scoped endpoint, return parsed JSON."""
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(_station_url(config, path), headers=_headers(config), json=payload or {})
        resp.raise_for_status()
    return resp.json()


def _put(
    config: AzuraCastConfig, path: str, payload: dict[str, Any] | None = None, timeout: int = 30
) -> Any:
    """PUT JSON to a station-scoped endpoint, return parsed JSON."""
    with httpx.Client(timeout=timeout) as client:
        resp = client.put(_station_url(config, path), headers=_headers(config), json=payload or {})
        resp.raise_for_status()
    return resp.json()


def _delete(config: AzuraCastConfig, path: str, timeout: int = 30) -> Any:
    """DELETE a station-scoped endpoint, return parsed JSON (or None if empty body)."""
    with httpx.Client(timeout=timeout) as client:
        resp = client.delete(_station_url(config, path), headers=_headers(config))
        resp.raise_for_status()
    if resp.content:
        return resp.json()
    return None


# ===========================================================================
# Media Management
# ===========================================================================


def upload_media(
    config: AzuraCastConfig,
    audio_path: Path,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Upload a media file to the AzuraCast station library.

    Uses: POST /api/station/{id}/files/upload (multipart/form-data)

    The base64 JSON endpoint (/files) fails on large payloads through
    Cloudflare tunnels. The multipart upload endpoint streams the file
    and works reliably at any size.

    Args:
        config: AzuraCast connection config.
        audio_path: Path to the audio file (MP3/WAV).
        metadata: Optional metadata dict (title, artist, etc.).

    Returns:
        Media file path string (used as identifier for playlist operations).
    """
    url = _station_url(config, "files/upload")

    with httpx.Client(timeout=120) as client:
        with audio_path.open("rb") as f:
            files = {"file": (audio_path.name, f, "audio/mpeg")}
            data: dict[str, str] = {"path": audio_path.name}
            if metadata:
                for k, v in metadata.items():
                    data[str(k)] = str(v)

            resp = client.post(
                url,
                headers=_headers(config),
                files=files,
                data=data,
            )
            resp.raise_for_status()

    return audio_path.name


def list_media(config: AzuraCastConfig) -> list[dict[str, Any]]:
    """List all media files in the station library.

    Uses: GET /api/station/{id}/files
    """
    result: list[dict[str, Any]] = _get(config, "files")
    return result


def delete_media(config: AzuraCastConfig, file_id: str) -> None:
    """Delete a media file from the station library.

    Uses: DELETE /api/station/{id}/file/{file_id}
    """
    _delete(config, f"file/{file_id}")


# ===========================================================================
# Playlist Management
# ===========================================================================


def create_playlist(
    config: AzuraCastConfig,
    name: str,
    playlist_type: str = "default",
    schedule_entries: list[dict[str, Any]] | None = None,
) -> str:
    """Create a playlist on the AzuraCast station.

    Uses: POST /api/station/{id}/playlists

    Args:
        config: AzuraCast connection config.
        name: Playlist name (e.g. "Haystack News", "Late Night DJ").
        playlist_type: Playlist type — "default", "once_per_x_songs", etc.
        schedule_entries: Optional list of schedule dicts, each with:
            - start_time: "HHMM" format (e.g. "0800")
            - end_time: "HHMM" format (e.g. "0830")
            - days: list of ints 1-7 (1=Mon, 7=Sun)
            - start_date/end_date: optional date range

    Returns:
        Playlist ID string.
    """
    payload: dict[str, Any] = {
        "name": name,
        "type": playlist_type,
        "source": "songs",
        "is_enabled": True,
    }
    if schedule_entries:
        payload["schedule_items"] = schedule_entries

    return str(_post(config, "playlists", payload)["id"])


def get_playlist(config: AzuraCastConfig, playlist_id: str) -> dict[str, Any]:
    """Get full details for a playlist including schedule entries.

    Uses: GET /api/station/{id}/playlist/{pid}
    """
    result: dict[str, Any] = _get(config, f"playlist/{playlist_id}")
    return result


def update_playlist(
    config: AzuraCastConfig,
    playlist_id: str,
    updates: dict[str, Any],
) -> dict[str, Any]:
    """Update a playlist's properties (name, type, schedule, etc.).

    Uses: PUT /api/station/{id}/playlist/{pid}

    Args:
        config: AzuraCast connection config.
        playlist_id: Playlist to update.
        updates: Dict of fields to update. Common fields:
            - name: new name
            - type: playlist type
            - is_enabled: bool
            - schedule_items: list of schedule entry dicts

    Returns:
        Updated playlist dict.
    """
    result: dict[str, Any] = _put(config, f"playlist/{playlist_id}", updates)
    return result


def set_playlist_schedule(
    config: AzuraCastConfig,
    playlist_id: str,
    schedule_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Set the schedule for a playlist (convenience wrapper).

    Args:
        config: AzuraCast connection config.
        playlist_id: Playlist to schedule.
        schedule_entries: List of schedule dicts, each with:
            - start_time: "HHMM" format (e.g. "0800" for 8:00 AM)
            - end_time: "HHMM" format (e.g. "0830" for 8:30 AM)
            - days: list of ints 1-7 (1=Monday, 7=Sunday)

    Returns:
        Updated playlist dict.
    """
    return update_playlist(config, playlist_id, {"schedule_items": schedule_entries})


def list_playlists(config: AzuraCastConfig) -> list[dict[str, Any]]:
    """List all playlists on the station.

    Uses: GET /api/station/{id}/playlists
    """
    result: list[dict[str, Any]] = _get(config, "playlists")
    return result


def get_or_create_playlist(config: AzuraCastConfig, name: str) -> str:
    """Find a playlist by name, or create it if it doesn't exist.

    Returns:
        Playlist ID string.
    """
    playlists = list_playlists(config)
    for pl in playlists:
        if pl.get("name") == name:
            return str(pl["id"])
    return create_playlist(config, name)


def toggle_playlist(config: AzuraCastConfig, playlist_id: str) -> dict[str, Any]:
    """Toggle a playlist on/off (enable/disable without deleting).

    Uses: PUT /api/station/{id}/playlist/{pid}/toggle
    """
    result: dict[str, Any] = _put(config, f"playlist/{playlist_id}/toggle")
    return result


def delete_playlist(config: AzuraCastConfig, playlist_id: str) -> None:
    """Delete a playlist.

    Uses: DELETE /api/station/{id}/playlist/{pid}
    """
    _delete(config, f"playlist/{playlist_id}")


def empty_playlist(config: AzuraCastConfig, playlist_id: str) -> None:
    """Remove all media from a playlist (for rolling content updates).

    Uses: DELETE /api/station/{id}/playlist/{pid}/empty
    """
    _delete(config, f"playlist/{playlist_id}/empty")


def import_to_playlist(
    config: AzuraCastConfig,
    playlist_id: str,
    media_paths: list[str],
) -> dict[str, Any]:
    """Import media files into a playlist by path.

    Uses: POST /api/station/{id}/playlist/{pid}/import (multipart/form-data)

    Args:
        config: AzuraCast connection config.
        playlist_id: Target playlist ID.
        media_paths: List of media file paths (relative to station media dir).

    Returns:
        Import result dict with matched/unmatched files.
    """
    url = _station_url(config, f"playlist/{playlist_id}/import")
    content = "\n".join(media_paths).encode("utf-8")
    files = {"playlist_file": ("playlist.m3u", content, "text/plain")}

    with httpx.Client(timeout=30) as client:
        resp = client.post(url, headers=_headers(config), files=files)
        resp.raise_for_status()

    result: dict[str, Any] = resp.json()
    return result


def schedule_episode(
    config: AzuraCastConfig,
    media_path: str,
    playlist_id: str,
    schedule_time: str = "",
) -> None:
    """Assign a media file to a playlist by importing a file list.

    Uses import_to_playlist under the hood.

    Args:
        config: AzuraCast connection config.
        media_path: Media file path from upload_media() (e.g. "episode.mp3").
        playlist_id: Target playlist ID.
        schedule_time: Reserved for future scheduled playout support.
    """
    import_to_playlist(config, playlist_id, [media_path])


def get_playlist_order(config: AzuraCastConfig, playlist_id: str) -> list[dict[str, Any]]:
    """Get the track order for a playlist.

    Uses: GET /api/station/{id}/playlist/{pid}/order
    """
    result: list[dict[str, Any]] = _get(config, f"playlist/{playlist_id}/order")
    return result


def set_playlist_order(
    config: AzuraCastConfig,
    playlist_id: str,
    order: list[dict[str, Any]],
) -> None:
    """Set the track order for a playlist.

    Uses: PUT /api/station/{id}/playlist/{pid}/order
    """
    _put(config, f"playlist/{playlist_id}/order", {"order": order})


# ===========================================================================
# Schedule
# ===========================================================================


def get_schedule(config: AzuraCastConfig) -> list[dict[str, Any]]:
    """Get upcoming and currently active schedule entries.

    Uses: GET /api/station/{id}/schedule

    Returns:
        List of schedule entry dicts with playlist info and time ranges.
    """
    result: list[dict[str, Any]] = _get(config, "schedule")
    return result


# ===========================================================================
# Station Control & Status
# ===========================================================================


def get_now_playing(config: AzuraCastConfig) -> dict[str, Any]:
    """Get now-playing data — current track, listeners, station info.

    Uses: GET /api/nowplaying/{station_id} (public endpoint)

    Returns:
        Dict with keys: station, listeners, now_playing, song_history, etc.
    """
    base = config.base_url.rstrip("/")
    url = f"{base}/api/nowplaying/{config.station_id}"

    with httpx.Client(timeout=30) as client:
        resp = client.get(url, headers=_headers(config))
        resp.raise_for_status()

    result: dict[str, Any] = resp.json()
    return result


# Keep old name as alias for backwards compatibility
get_station_status = get_now_playing


def get_service_health(config: AzuraCastConfig) -> dict[str, Any]:
    """Get station service health — backend and frontend running status.

    Uses: GET /api/station/{id}/status

    Returns:
        Dict with keys: backendRunning, frontendRunning.
    """
    result: dict[str, Any] = _get(config, "status")
    return result


def restart_station(config: AzuraCastConfig) -> dict[str, Any]:
    """Restart all station services (Liquidsoap + Icecast).

    Uses: POST /api/station/{id}/restart
    """
    result: dict[str, Any] = _post(config, "restart")
    return result


# ===========================================================================
# Queue
# ===========================================================================


def get_queue(config: AzuraCastConfig) -> list[dict[str, Any]]:
    """Get the upcoming playback queue.

    Uses: GET /api/station/{id}/queue
    """
    result: list[dict[str, Any]] = _get(config, "queue")
    return result


def remove_from_queue(config: AzuraCastConfig, queue_id: str) -> None:
    """Remove a specific item from the playback queue.

    Uses: DELETE /api/station/{id}/queue/{queue_id}
    """
    _delete(config, f"queue/{queue_id}")


# ===========================================================================
# Analytics & History
# ===========================================================================


def get_history(config: AzuraCastConfig) -> list[dict[str, Any]]:
    """Get song playback history.

    Uses: GET /api/station/{id}/history
    """
    result: list[dict[str, Any]] = _get(config, "history")
    return result


def get_listeners(config: AzuraCastConfig) -> list[dict[str, Any]]:
    """Get current listeners with details.

    Uses: GET /api/station/{id}/listeners
    """
    result: list[dict[str, Any]] = _get(config, "listeners")
    return result


def get_listener_charts(config: AzuraCastConfig) -> dict[str, Any]:
    """Get listener count charts over time.

    Uses: GET /api/station/{id}/reports/overview/charts
    """
    result: dict[str, Any] = _get(config, "reports/overview/charts")
    return result


def get_best_and_worst(config: AzuraCastConfig) -> dict[str, Any]:
    """Get best and worst performing tracks.

    Uses: GET /api/station/{id}/reports/overview/best-and-worst
    """
    result: dict[str, Any] = _get(config, "reports/overview/best-and-worst")
    return result


def get_listeners_by_country(config: AzuraCastConfig) -> list[dict[str, Any]]:
    """Get listener demographics by country.

    Uses: GET /api/station/{id}/reports/overview/by-country
    """
    result: list[dict[str, Any]] = _get(config, "reports/overview/by-country")
    return result


def get_listeners_by_client(config: AzuraCastConfig) -> list[dict[str, Any]]:
    """Get listener demographics by client/app.

    Uses: GET /api/station/{id}/reports/overview/by-client
    """
    result: list[dict[str, Any]] = _get(config, "reports/overview/by-client")
    return result


# ===========================================================================
# High-level convenience methods
# ===========================================================================


def update_episode(
    config: AzuraCastConfig,
    audio_path: Path,
    playlist_name: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Rolling playlist update: upload new episode, swap into playlist.

    This is the recommended way to publish an episode:
    1. Upload the audio file to the station library
    2. Find (or create) the named playlist
    3. Empty the playlist (remove previous episode)
    4. Assign the new file to the playlist

    Args:
        config: AzuraCast connection config.
        audio_path: Path to the episode MP3.
        playlist_name: Name of the rolling playlist (created if absent).
        metadata: Optional metadata (title, artist).

    Returns:
        Media file path string for the uploaded file.
    """
    media_path = upload_media(config, audio_path, metadata)
    playlist_id = get_or_create_playlist(config, playlist_name)
    empty_playlist(config, playlist_id)
    schedule_episode(config, media_path, playlist_id)
    return media_path


# ===========================================================================
# CLI
# ===========================================================================


def _load_cli_config() -> AzuraCastConfig:
    """Load config from env vars for CLI usage."""
    from src.secrets import get_secret

    config = AzuraCastConfig(
        base_url=get_secret("AGENT_RADIO_AZURACAST_URL"),
        api_key=get_secret("AGENT_RADIO_AZURACAST_API_KEY"),
        station_id=int(get_secret("AGENT_RADIO_AZURACAST_STATION_ID") or "1"),
    )

    if not config.base_url or not config.api_key:
        print("ERROR: AGENT_RADIO_AZURACAST_URL and AGENT_RADIO_AZURACAST_API_KEY must be set.")
        sys.exit(1)

    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="AzuraCast streaming client for Agent Radio")
    sub = parser.add_subparsers(dest="command")

    # Now playing
    sub.add_parser("status", help="Now playing + listener count")

    # Service health
    sub.add_parser("health", help="Backend/frontend service health")

    # Schedule
    sub.add_parser("schedule", help="View upcoming schedule")

    # Playlists
    sub.add_parser("playlists", help="List all playlists")

    # Queue
    sub.add_parser("queue", help="View upcoming playback queue")

    # Listeners
    sub.add_parser("listeners", help="Current listeners")

    # History
    sub.add_parser("history", help="Recent playback history")

    # Upload
    upload_p = sub.add_parser("upload", help="Upload a media file")
    upload_p.add_argument("file", type=Path, help="Audio file to upload")
    upload_p.add_argument("--title", default="", help="Track title metadata")
    upload_p.add_argument("--artist", default="Agent Radio", help="Artist metadata")

    # Update (rolling playlist)
    update_p = sub.add_parser("update", help="Rolling playlist update")
    update_p.add_argument("file", type=Path, help="Audio file to upload")
    update_p.add_argument("--playlist", required=True, help="Playlist name")
    update_p.add_argument("--title", default="", help="Track title metadata")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    config = _load_cli_config()

    if args.command == "status":
        data = get_now_playing(config)
        np = data.get("now_playing", {})
        song = np.get("song", {}) if np else {}
        listeners = data.get("listeners", {})
        print(f"Now playing: {song.get('title', 'N/A')} — {song.get('artist', 'N/A')}")
        print(f"Listeners:   {listeners.get('total', 0)}")

    elif args.command == "health":
        health = get_service_health(config)
        print(f"Backend:  {'running' if health.get('backendRunning') else 'STOPPED'}")
        print(f"Frontend: {'running' if health.get('frontendRunning') else 'STOPPED'}")

    elif args.command == "schedule":
        entries = get_schedule(config)
        if not entries:
            print("No scheduled entries.")
        for entry in entries:
            pl = entry.get("playlist", {})
            print(
                f"  {entry.get('start', '?')} — {entry.get('end', '?')}  "
                f"{pl.get('name', 'Unknown playlist')}"
            )

    elif args.command == "playlists":
        for pl in list_playlists(config):
            enabled = "on" if pl.get("is_enabled") else "off"
            print(f"  [{enabled}] #{pl['id']} {pl['name']} ({pl.get('type', '?')})")

    elif args.command == "queue":
        queue = get_queue(config)
        if not queue:
            print("Queue is empty.")
        for item in queue[:10]:
            song = item.get("song", {})
            print(
                f"  {item.get('cued_at', '?')}  {song.get('title', '?')} — {song.get('artist', '?')}"
            )

    elif args.command == "listeners":
        listeners = get_listeners(config)
        print(f"Current listeners: {len(listeners)}")
        for li in listeners[:10]:
            print(f"  {li.get('ip', '?')} — {li.get('user_agent', '?')}")

    elif args.command == "history":
        history = get_history(config)
        for item in history[:10]:
            song = item.get("song", {})
            print(
                f"  {item.get('played_at', '?')}  {song.get('title', '?')} — {song.get('artist', '?')}"
            )

    elif args.command == "upload":
        metadata = {"title": args.title, "artist": args.artist}
        media_path = upload_media(config, args.file, metadata)
        print(f"Uploaded: {args.file.name} → {media_path}")

    elif args.command == "update":
        metadata = {"title": args.title, "artist": "Agent Radio"}
        media_path = update_episode(config, args.file, args.playlist, metadata)
        print(f"Updated playlist '{args.playlist}' with {args.file.name} → {media_path}")


if __name__ == "__main__":
    main()
