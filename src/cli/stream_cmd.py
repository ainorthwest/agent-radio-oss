"""AzuraCast streaming commands — station control, playlists, analytics.

``radio stream status``     — now playing + listener count
``radio stream health``     — backend/frontend service health
``radio stream upload``     — upload media file to station library
``radio stream playlist``   — list playlists
``radio stream schedule``   — view broadcast schedule
``radio stream listeners``  — current listeners
``radio stream history``    — recent playback history
``radio stream queue``      — upcoming playback queue
``radio stream update``     — rolling playlist update (upload + swap)
"""

from __future__ import annotations

from pathlib import Path

import typer

from src.cli._output import err, output

app = typer.Typer(name="stream", help="AzuraCast streaming — station control.")


def _az_config(ctx: typer.Context):  # noqa: ANN202
    """Build AzuraCastConfig from the global config."""
    from src.stream import AzuraCastConfig

    state = ctx.obj
    sc = state.config.stream
    if not sc.base_url or not sc.api_key:
        err(
            "AzuraCast not configured. Set stream.base_url in radio.yaml "
            "and AGENT_RADIO_AZURACAST_API_KEY in environment."
        )
    return AzuraCastConfig(base_url=sc.base_url, api_key=sc.api_key, station_id=sc.station_id)


@app.command()
def status(ctx: typer.Context) -> None:
    """Now playing + listener count."""
    from src.stream import get_now_playing

    config = _az_config(ctx)
    data = get_now_playing(config)
    state = ctx.obj

    if state.json_output:
        output(state, data)
    else:
        np = data.get("now_playing", {})
        song = np.get("song", {}) if np else {}
        listeners = data.get("listeners", {})
        print(f"Now playing: {song.get('title', 'N/A')} — {song.get('artist', 'N/A')}")
        print(f"Listeners:   {listeners.get('total', 0)}")


@app.command()
def health(ctx: typer.Context) -> None:
    """Backend/frontend service health."""
    from src.stream import get_service_health

    config = _az_config(ctx)
    data = get_service_health(config)
    state = ctx.obj

    if state.json_output:
        output(state, data)
    else:
        print(f"Backend:  {'running' if data.get('backendRunning') else 'STOPPED'}")
        print(f"Frontend: {'running' if data.get('frontendRunning') else 'STOPPED'}")


@app.command()
def upload(
    ctx: typer.Context,
    file: Path = typer.Argument(..., help="Audio file to upload"),
    title: str = typer.Option("", "--title", help="Track title metadata"),
    artist: str = typer.Option("Agent Radio", "--artist", help="Artist metadata"),
) -> None:
    """Upload a media file to the station library."""
    from src.stream import upload_media

    config = _az_config(ctx)
    state = ctx.obj
    if not file.exists():
        err(f"File not found: {file}")

    metadata = {"title": title, "artist": artist}
    media_path = upload_media(config, file, metadata)

    if state.json_output:
        output(state, {"media_path": media_path, "file": str(file)})
    else:
        print(f"Uploaded: {file.name} → {media_path}")


@app.command()
def playlist(ctx: typer.Context) -> None:
    """List all playlists on the station."""
    from src.stream import list_playlists

    config = _az_config(ctx)
    state = ctx.obj
    playlists = list_playlists(config)

    if state.json_output:
        output(state, playlists)
    else:
        for pl in playlists:
            enabled = "on" if pl.get("is_enabled") else "off"
            print(f"  [{enabled}] #{pl['id']} {pl['name']} ({pl.get('type', '?')})")


@app.command()
def schedule(ctx: typer.Context) -> None:
    """View broadcast schedule."""
    from src.stream import get_schedule

    config = _az_config(ctx)
    state = ctx.obj
    entries = get_schedule(config)

    if state.json_output:
        output(state, entries)
    else:
        if not entries:
            print("No scheduled entries.")
        for entry in entries:
            pl = entry.get("playlist", {})
            print(
                f"  {entry.get('start', '?')} — {entry.get('end', '?')}  "
                f"{pl.get('name', 'Unknown playlist')}"
            )


@app.command()
def listeners(ctx: typer.Context) -> None:
    """Current listeners."""
    from src.stream import get_listeners

    config = _az_config(ctx)
    state = ctx.obj
    listener_list = get_listeners(config)

    if state.json_output:
        output(state, listener_list)
    else:
        print(f"Current listeners: {len(listener_list)}")
        for li in listener_list[:10]:
            print(f"  {li.get('ip', '?')} — {li.get('user_agent', '?')}")


@app.command()
def history(ctx: typer.Context) -> None:
    """Recent playback history."""
    from src.stream import get_history

    config = _az_config(ctx)
    state = ctx.obj
    hist = get_history(config)

    if state.json_output:
        output(state, hist)
    else:
        for item in hist[:10]:
            song = item.get("song", {})
            print(
                f"  {item.get('played_at', '?')}  {song.get('title', '?')} — {song.get('artist', '?')}"
            )


@app.command()
def queue(ctx: typer.Context) -> None:
    """Upcoming playback queue."""
    from src.stream import get_queue

    config = _az_config(ctx)
    state = ctx.obj
    q = get_queue(config)

    if state.json_output:
        output(state, q)
    else:
        if not q:
            print("Queue is empty.")
        for item in q[:10]:
            song = item.get("song", {})
            print(
                f"  {item.get('cued_at', '?')}  {song.get('title', '?')} — {song.get('artist', '?')}"
            )


@app.command()
def update(
    ctx: typer.Context,
    file: Path = typer.Argument(..., help="Audio file to upload"),
    playlist_name: str = typer.Option(..., "--playlist", help="Playlist name (created if absent)"),
    title: str = typer.Option("", "--title", help="Track title metadata"),
) -> None:
    """Rolling playlist update — upload episode + swap into playlist."""
    from src.stream import update_episode

    config = _az_config(ctx)
    state = ctx.obj
    if not file.exists():
        err(f"File not found: {file}")

    metadata = {"title": title, "artist": "Agent Radio"}
    media_path = update_episode(config, file, playlist_name, metadata)

    if state.json_output:
        output(state, {"media_path": media_path, "playlist": playlist_name, "file": str(file)})
    else:
        print(f"Updated playlist '{playlist_name}' with {file.name} → {media_path}")
