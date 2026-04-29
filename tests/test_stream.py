"""Tests for AzuraCast streaming API client."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.stream import (
    AzuraCastConfig,
    create_playlist,
    delete_media,
    empty_playlist,
    get_best_and_worst,
    get_history,
    get_listener_charts,
    get_listeners,
    get_listeners_by_client,
    get_listeners_by_country,
    get_now_playing,
    get_or_create_playlist,
    get_playlist,
    get_playlist_order,
    get_queue,
    get_schedule,
    get_service_health,
    import_to_playlist,
    list_media,
    list_playlists,
    remove_from_queue,
    restart_station,
    schedule_episode,
    set_playlist_order,
    set_playlist_schedule,
    toggle_playlist,
    update_episode,
    update_playlist,
    upload_media,
)

# Shared test config
TEST_CONFIG = AzuraCastConfig(
    base_url="https://radio.example.com",
    api_key="test-api-key",
    station_id=1,
)


def _mock_client(response_json: dict | list | None = None, status_code: int = 200):
    """Create a mock httpx.Client context manager with a preset response."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = status_code
    mock_resp.json.return_value = response_json if response_json is not None else {}
    mock_resp.raise_for_status = MagicMock()

    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=mock_resp,
        )

    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp
    mock_client.get.return_value = mock_resp
    mock_client.put.return_value = mock_resp
    mock_client.delete.return_value = mock_resp

    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_client)
    mock_ctx.__exit__ = MagicMock(return_value=False)

    return mock_ctx, mock_client, mock_resp


# ===========================================================================
# Config
# ===========================================================================


class TestAzuraCastConfig:
    def test_defaults(self) -> None:
        config = AzuraCastConfig()
        assert config.base_url == ""
        assert config.api_key == ""
        assert config.station_id == 1

    def test_custom_values(self) -> None:
        config = AzuraCastConfig(
            base_url="https://radio.example.com",
            api_key="test-key",
            station_id=2,
        )
        assert config.base_url == "https://radio.example.com"
        assert config.station_id == 2


# ===========================================================================
# Media
# ===========================================================================


class TestUploadMedia:
    def test_upload_returns_file_path(self, tmp_path: Path) -> None:
        audio = tmp_path / "episode.mp3"
        audio.write_bytes(b"fake audio data")

        mock_ctx, mock_client, _ = _mock_client({"path": "episode.mp3"})
        with patch("httpx.Client", return_value=mock_ctx):
            result = upload_media(TEST_CONFIG, audio, {"title": "Test Episode"})

        assert result == "episode.mp3"
        mock_client.post.assert_called_once()
        args, kwargs = mock_client.post.call_args
        assert args[0] == "https://radio.example.com/api/station/1/files/upload"
        assert kwargs["headers"]["X-API-Key"] == "test-api-key"

    def test_upload_sends_multipart(self, tmp_path: Path) -> None:
        audio = tmp_path / "episode.mp3"
        audio.write_bytes(b"fake audio data")

        mock_ctx, mock_client, _ = _mock_client({"success": True})
        with patch("httpx.Client", return_value=mock_ctx):
            upload_media(TEST_CONFIG, audio)

        _, kwargs = mock_client.post.call_args
        assert "files" in kwargs
        assert kwargs["data"]["path"] == "episode.mp3"

    def test_upload_raises_on_auth_error(self, tmp_path: Path) -> None:
        audio = tmp_path / "episode.mp3"
        audio.write_bytes(b"fake audio data")

        mock_ctx, _, _ = _mock_client(status_code=401)
        with patch("httpx.Client", return_value=mock_ctx):
            with pytest.raises(httpx.HTTPStatusError):
                upload_media(TEST_CONFIG, audio)

    def test_upload_sends_metadata(self, tmp_path: Path) -> None:
        audio = tmp_path / "episode.mp3"
        audio.write_bytes(b"fake audio data")

        mock_ctx, mock_client, _ = _mock_client({"success": True})
        with patch("httpx.Client", return_value=mock_ctx):
            upload_media(TEST_CONFIG, audio, {"title": "My Show", "artist": "Agent Radio"})

        _, kwargs = mock_client.post.call_args
        assert kwargs["data"]["title"] == "My Show"
        assert kwargs["data"]["artist"] == "Agent Radio"


class TestListMedia:
    def test_returns_list(self) -> None:
        files = [{"id": 1, "path": "a.mp3"}, {"id": 2, "path": "b.mp3"}]
        mock_ctx, _, _ = _mock_client(files)
        with patch("httpx.Client", return_value=mock_ctx):
            result = list_media(TEST_CONFIG)
        assert len(result) == 2


class TestDeleteMedia:
    def test_calls_delete(self) -> None:
        mock_ctx, mock_client, _ = _mock_client()
        with patch("httpx.Client", return_value=mock_ctx):
            delete_media(TEST_CONFIG, "42")
        mock_client.delete.assert_called_once()
        args, _ = mock_client.delete.call_args
        assert "file/42" in args[0]


# ===========================================================================
# Playlists
# ===========================================================================


class TestCreatePlaylist:
    def test_create_returns_id(self) -> None:
        mock_ctx, mock_client, _ = _mock_client({"id": 7, "name": "Test"})
        with patch("httpx.Client", return_value=mock_ctx):
            result = create_playlist(TEST_CONFIG, "Haystack News")

        assert result == "7"
        _, kwargs = mock_client.post.call_args
        assert kwargs["json"]["name"] == "Haystack News"

    def test_create_with_schedule(self) -> None:
        schedule = [{"start_time": "0800", "end_time": "0830", "days": [1, 2, 3, 4, 5]}]
        mock_ctx, mock_client, _ = _mock_client({"id": 3})
        with patch("httpx.Client", return_value=mock_ctx):
            create_playlist(TEST_CONFIG, "Morning Show", schedule_entries=schedule)

        _, kwargs = mock_client.post.call_args
        assert kwargs["json"]["schedule_items"] == schedule


class TestGetPlaylist:
    def test_returns_details(self) -> None:
        pl = {"id": 5, "name": "Show", "schedule_items": []}
        mock_ctx, _, _ = _mock_client(pl)
        with patch("httpx.Client", return_value=mock_ctx):
            result = get_playlist(TEST_CONFIG, "5")
        assert result["name"] == "Show"


class TestUpdatePlaylist:
    def test_sends_updates(self) -> None:
        mock_ctx, mock_client, _ = _mock_client({"id": 5, "name": "Renamed"})
        with patch("httpx.Client", return_value=mock_ctx):
            result = update_playlist(TEST_CONFIG, "5", {"name": "Renamed"})
        assert result["name"] == "Renamed"
        mock_client.put.assert_called_once()


class TestSetPlaylistSchedule:
    def test_sets_schedule_items(self) -> None:
        schedule = [{"start_time": "0800", "end_time": "0830", "days": [1, 2, 3]}]
        with patch("src.stream.update_playlist", return_value={"id": 5}) as mock_update:
            set_playlist_schedule(TEST_CONFIG, "5", schedule)
        mock_update.assert_called_once_with(TEST_CONFIG, "5", {"schedule_items": schedule})


class TestListPlaylists:
    def test_returns_list(self) -> None:
        playlists = [{"id": 1, "name": "Show A"}, {"id": 2, "name": "Show B"}]
        mock_ctx, _, _ = _mock_client(playlists)
        with patch("httpx.Client", return_value=mock_ctx):
            result = list_playlists(TEST_CONFIG)
        assert len(result) == 2

    def test_returns_empty_list(self) -> None:
        mock_ctx, _, _ = _mock_client([])
        with patch("httpx.Client", return_value=mock_ctx):
            result = list_playlists(TEST_CONFIG)
        assert result == []


class TestGetOrCreatePlaylist:
    def test_returns_existing(self) -> None:
        playlists = [{"id": 5, "name": "Haystack News"}, {"id": 6, "name": "Other"}]
        mock_ctx, _, _ = _mock_client(playlists)
        with patch("httpx.Client", return_value=mock_ctx):
            result = get_or_create_playlist(TEST_CONFIG, "Haystack News")
        assert result == "5"

    def test_creates_if_not_found(self) -> None:
        with patch("src.stream.list_playlists", return_value=[]):
            with patch("src.stream.create_playlist", return_value="9") as mock_create:
                result = get_or_create_playlist(TEST_CONFIG, "New Show")
        assert result == "9"
        mock_create.assert_called_once_with(TEST_CONFIG, "New Show")


class TestTogglePlaylist:
    def test_calls_toggle(self) -> None:
        mock_ctx, mock_client, _ = _mock_client({"id": 5, "is_enabled": False})
        with patch("httpx.Client", return_value=mock_ctx):
            toggle_playlist(TEST_CONFIG, "5")
        mock_client.put.assert_called_once()
        args, _ = mock_client.put.call_args
        assert "playlist/5/toggle" in args[0]


class TestEmptyPlaylist:
    def test_calls_delete_endpoint(self) -> None:
        mock_ctx, mock_client, _ = _mock_client()
        with patch("httpx.Client", return_value=mock_ctx):
            empty_playlist(TEST_CONFIG, "7")
        mock_client.delete.assert_called_once()
        args, _ = mock_client.delete.call_args
        assert "playlist/7/empty" in args[0]


class TestGetPlaylistOrder:
    def test_returns_order(self) -> None:
        order = [{"id": 1, "title": "Track A"}, {"id": 2, "title": "Track B"}]
        mock_ctx, mock_client, _ = _mock_client(order)
        with patch("httpx.Client", return_value=mock_ctx):
            result = get_playlist_order(TEST_CONFIG, "5")
        assert len(result) == 2
        args, _ = mock_client.get.call_args
        assert "playlist/5/order" in args[0]


class TestSetPlaylistOrder:
    def test_sends_order(self) -> None:
        order = [{"id": 2}, {"id": 1}]
        mock_ctx, mock_client, _ = _mock_client({})
        with patch("httpx.Client", return_value=mock_ctx):
            set_playlist_order(TEST_CONFIG, "5", order)
        mock_client.put.assert_called_once()
        _, kwargs = mock_client.put.call_args
        assert kwargs["json"]["order"] == order


class TestImportToPlaylist:
    def test_sends_multipart(self) -> None:
        mock_ctx, mock_client, _ = _mock_client({"success": True})
        with patch("httpx.Client", return_value=mock_ctx):
            import_to_playlist(TEST_CONFIG, "7", ["ep1.mp3", "ep2.mp3"])
        mock_client.post.assert_called_once()
        args, kwargs = mock_client.post.call_args
        assert "playlist/7/import" in args[0]
        assert "files" in kwargs


class TestScheduleEpisode:
    def test_calls_import(self) -> None:
        with patch("src.stream.import_to_playlist") as mock_import:
            schedule_episode(TEST_CONFIG, "episode.mp3", "7")
        mock_import.assert_called_once_with(TEST_CONFIG, "7", ["episode.mp3"])


# ===========================================================================
# Schedule
# ===========================================================================


class TestGetSchedule:
    def test_returns_entries(self) -> None:
        entries = [{"start": "08:00", "end": "08:30", "playlist": {"name": "News"}}]
        mock_ctx, _, _ = _mock_client(entries)
        with patch("httpx.Client", return_value=mock_ctx):
            result = get_schedule(TEST_CONFIG)
        assert len(result) == 1
        assert result[0]["playlist"]["name"] == "News"


# ===========================================================================
# Station Control
# ===========================================================================


class TestGetNowPlaying:
    def test_returns_data(self) -> None:
        status = {"now_playing": {"song": {"title": "Test"}}, "listeners": {"total": 5}}
        mock_ctx, mock_client, _ = _mock_client(status)
        with patch("httpx.Client", return_value=mock_ctx):
            result = get_now_playing(TEST_CONFIG)
        assert result["listeners"]["total"] == 5
        args, _ = mock_client.get.call_args
        assert args[0] == "https://radio.example.com/api/nowplaying/1"

    def test_raises_on_server_error(self) -> None:
        mock_ctx, _, _ = _mock_client(status_code=500)
        with patch("httpx.Client", return_value=mock_ctx):
            with pytest.raises(httpx.HTTPStatusError):
                get_now_playing(TEST_CONFIG)


class TestGetServiceHealth:
    def test_returns_health(self) -> None:
        health = {"backendRunning": True, "frontendRunning": True}
        mock_ctx, mock_client, _ = _mock_client(health)
        with patch("httpx.Client", return_value=mock_ctx):
            result = get_service_health(TEST_CONFIG)
        assert result["backendRunning"] is True
        args, _ = mock_client.get.call_args
        assert "status" in args[0]


class TestRestartStation:
    def test_calls_restart(self) -> None:
        mock_ctx, mock_client, _ = _mock_client({"success": True})
        with patch("httpx.Client", return_value=mock_ctx):
            restart_station(TEST_CONFIG)
        mock_client.post.assert_called_once()
        args, _ = mock_client.post.call_args
        assert "restart" in args[0]


# ===========================================================================
# Queue
# ===========================================================================


class TestGetQueue:
    def test_returns_queue(self) -> None:
        queue = [{"id": 1, "song": {"title": "Next Up"}}]
        mock_ctx, _, _ = _mock_client(queue)
        with patch("httpx.Client", return_value=mock_ctx):
            result = get_queue(TEST_CONFIG)
        assert len(result) == 1


class TestRemoveFromQueue:
    def test_calls_delete(self) -> None:
        mock_ctx, mock_client, _ = _mock_client()
        with patch("httpx.Client", return_value=mock_ctx):
            remove_from_queue(TEST_CONFIG, "99")
        mock_client.delete.assert_called_once()
        args, _ = mock_client.delete.call_args
        assert "queue/99" in args[0]


# ===========================================================================
# Analytics
# ===========================================================================


class TestGetHistory:
    def test_returns_history(self) -> None:
        history = [{"played_at": "2026-03-19T08:00", "song": {"title": "Morning News"}}]
        mock_ctx, _, _ = _mock_client(history)
        with patch("httpx.Client", return_value=mock_ctx):
            result = get_history(TEST_CONFIG)
        assert len(result) == 1


class TestGetListeners:
    def test_returns_listeners(self) -> None:
        listeners = [{"ip": "1.2.3.4", "user_agent": "VLC"}]
        mock_ctx, _, _ = _mock_client(listeners)
        with patch("httpx.Client", return_value=mock_ctx):
            result = get_listeners(TEST_CONFIG)
        assert len(result) == 1


class TestGetListenerCharts:
    def test_returns_charts(self) -> None:
        charts = {"labels": ["Mon", "Tue"], "datasets": []}
        mock_ctx, _, _ = _mock_client(charts)
        with patch("httpx.Client", return_value=mock_ctx):
            result = get_listener_charts(TEST_CONFIG)
        assert "labels" in result


class TestGetBestAndWorst:
    def test_returns_data(self) -> None:
        data = {"best": [], "worst": []}
        mock_ctx, _, _ = _mock_client(data)
        with patch("httpx.Client", return_value=mock_ctx):
            result = get_best_and_worst(TEST_CONFIG)
        assert "best" in result


class TestGetListenersByCountry:
    def test_returns_data(self) -> None:
        data = [{"country": "US", "listeners": 42}]
        mock_ctx, _, _ = _mock_client(data)
        with patch("httpx.Client", return_value=mock_ctx):
            result = get_listeners_by_country(TEST_CONFIG)
        assert result[0]["country"] == "US"


class TestGetListenersByClient:
    def test_returns_data(self) -> None:
        data = [{"client": "VLC", "listeners": 10}]
        mock_ctx, _, _ = _mock_client(data)
        with patch("httpx.Client", return_value=mock_ctx):
            result = get_listeners_by_client(TEST_CONFIG)
        assert result[0]["client"] == "VLC"


# ===========================================================================
# Convenience
# ===========================================================================


class TestUpdateEpisode:
    def test_full_rolling_sequence(self, tmp_path: Path) -> None:
        audio = tmp_path / "episode.mp3"
        audio.write_bytes(b"fake audio")

        with (
            patch("src.stream.upload_media", return_value="episode.mp3") as mock_upload,
            patch("src.stream.get_or_create_playlist", return_value="7") as mock_playlist,
            patch("src.stream.empty_playlist") as mock_empty,
            patch("src.stream.schedule_episode") as mock_schedule,
        ):
            result = update_episode(TEST_CONFIG, audio, "Haystack News", {"title": "Ep 1"})

        assert result == "episode.mp3"
        mock_upload.assert_called_once_with(TEST_CONFIG, audio, {"title": "Ep 1"})
        mock_playlist.assert_called_once_with(TEST_CONFIG, "Haystack News")
        mock_empty.assert_called_once_with(TEST_CONFIG, "7")
        mock_schedule.assert_called_once_with(TEST_CONFIG, "episode.mp3", "7")
