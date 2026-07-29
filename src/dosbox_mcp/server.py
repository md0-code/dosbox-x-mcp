"""An MCP server for driving a DOSBox-X guest without touching the desktop.

Every tool here works through the emulator process itself -- its memory and its
window -- so nothing steals focus, moves the pointer, or types into whatever the
user happens to be doing.  Keys go into the guest's BIOS keyboard ring, clicks
go into the game's own post-INT-33 mouse words, and frames are photographed with
PrintWindow rather than off the screen.

Configuration comes from the environment:

    DOSBOX_MCP_PROFILE_DIR   where profiles live (default: ../profiles)
    DOSBOX_MCP_PROFILE       default profile name (default: the only one there)
    DOSBOX_MCP_EXECUTABLE    dosbox-x.exe to launch
    DOSBOX_MCP_OUTPUT_DIR    default directory for captures
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP, Image
from mcp.types import ToolAnnotations

from .profile import Profile, ProfileError, load_profile
from .screen import CaptureError, capture_window, downscale_png, write_outputs
from .session import Condition, Registry, Session, SessionError
from .win32 import ProcessError, find_windows

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_DIR = Path(
    os.environ.get("DOSBOX_MCP_PROFILE_DIR", PACKAGE_ROOT / "profiles")
).resolve()
DEFAULT_PROFILE_NAME = os.environ.get("DOSBOX_MCP_PROFILE")
DEFAULT_EXECUTABLE = os.environ.get("DOSBOX_MCP_EXECUTABLE")
DEFAULT_OUTPUT_DIR = Path(os.environ.get("DOSBOX_MCP_OUTPUT_DIR", Path.cwd())).resolve()

mcp = FastMCP(
    "dosbox",
    instructions=(
        "Drive a DOSBox-X guest headlessly: launch it, inject keys and clicks "
        "into the guest's own input state, read and patch the game's data "
        "segment, wait on memory conditions, and capture exact frames.  Start "
        "with dosbox_profiles to see what is known about the game, then "
        "dosbox_launch or dosbox_attach.  Nothing here touches the host "
        "desktop, so it is safe to use while the user is working."
    ),
)

registry = Registry()
_profiles: dict[str, Profile] = {}


# ── helpers ────────────────────────────────────────────────────────────────


def _profile_files() -> list[Path]:
    return sorted(DEFAULT_PROFILE_DIR.glob("*.json")) if DEFAULT_PROFILE_DIR.is_dir() else []


def _resolve_profile(name: str | None) -> Profile:
    """Load a profile by name, by path, or fall back to the configured default."""
    wanted = name or DEFAULT_PROFILE_NAME
    if wanted is None:
        files = _profile_files()
        if len(files) == 1:
            wanted = files[0].stem
        elif not files:
            raise ProfileError(
                f"no profiles in {DEFAULT_PROFILE_DIR}.  Point DOSBOX_MCP_PROFILE_DIR "
                f"at a directory of profile JSON files, or pass an explicit path."
            )
        else:
            names = ", ".join(path.stem for path in files)
            raise ProfileError(
                f"several profiles are available ({names}); pass one, or set "
                f"DOSBOX_MCP_PROFILE."
            )
    if wanted in _profiles:
        return _profiles[wanted]
    candidate = Path(wanted)
    if not candidate.is_file():
        candidate = DEFAULT_PROFILE_DIR / f"{wanted}.json"
    profile = load_profile(candidate)
    _profiles[wanted] = profile
    _profiles[profile.name] = profile
    return profile


def _session(pid: int | None) -> Session:
    return registry.get(pid)


def _condition(
    session: Session,
    offset: int | str,
    size: int,
    equals: int | None,
    not_equals: int | None,
    at_least: int | None,
    at_most: int | None,
    any_of: list[int] | None,
) -> Condition:
    resolved, symbol_size = session.profile.resolve(offset)
    if all(test is None for test in (equals, not_equals, at_least, at_most, any_of)):
        raise SessionError(
            "give at least one test: equals, not_equals, at_least, at_most or any_of"
        )
    return Condition(
        offset=resolved,
        size=size or symbol_size,
        equals=equals,
        not_equals=not_equals,
        at_least=at_least,
        at_most=at_most,
        any_of=tuple(any_of) if any_of else None,
    )


def _window_of(session: Session):
    window = session.window()
    if window is None:
        raise SessionError(
            f"process {session.pid} has no visible window; it may have exited, or be "
            "running with output disabled"
        )
    return window


class MemoryWrite(BaseModel):
    """One patch into the guest's data segment."""

    offset: int | str = Field(
        description="Byte offset in the data segment, or a profile symbol name. "
        "Accepts 0x-prefixed strings."
    )
    value: int | None = Field(
        default=None, description="Value to write, little-endian, `size` bytes wide."
    )
    hex: str | None = Field(
        default=None, description="Raw bytes as hex, used instead of `value`."
    )
    size: int = Field(default=1, ge=1, le=4, description="Width in bytes when using `value`.")


# ── discovery and lifecycle ────────────────────────────────────────────────


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    description=(
        "List the game profiles this server knows and, for one of them, every "
        "named memory offset it exposes.  A profile says how to recognise the "
        "game's data segment, where the game keeps its mouse state, and what "
        "its screen looks like.  Call this first: the symbol names it returns "
        "can be used anywhere an offset is accepted."
    ),
)
def dosbox_profiles(profile: str | None = None) -> dict[str, Any]:
    available = [path.stem for path in _profile_files()]
    result: dict[str, Any] = {
        "profile_directory": str(DEFAULT_PROFILE_DIR),
        "available": available,
        "default": DEFAULT_PROFILE_NAME,
    }
    try:
        loaded = _resolve_profile(profile)
    except ProfileError as error:
        result["detail"] = str(error)
        return result
    result["loaded"] = {
        "name": loaded.name,
        "description": loaded.description,
        "window_title": loaded.window_title,
        "screen": f"{loaded.screen_width}x{loaded.screen_height}",
        "palette_entries": len(loaded.palette),
        "mouse_buttons_offset": loaded.mouse_buttons,
        "mouse_position_offset": loaded.mouse_position,
        "symbols": {
            name: {
                "offset": f"{symbol.offset:#06x}",
                "size": symbol.size,
                "description": symbol.description,
            }
            for name, symbol in sorted(loaded.symbols.items())
        },
    }
    return result


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True),
    description=(
        "Start DOSBox-X with a config file and attach to it.  Returns the process "
        "id every other tool takes.  The config's [autoexec] section decides what "
        "runs; an AUTOTYPE line there is the reliable way past intro screens.  "
        "Waits until the game's data segment is findable, so a successful call "
        "means the guest is really up and drivable."
    ),
)
async def dosbox_launch(
    config: Annotated[str, Field(description="Path to the DOSBox-X .conf file.")],
    executable: Annotated[
        str | None, Field(description="dosbox-x.exe to run; defaults to DOSBOX_MCP_EXECUTABLE.")
    ] = None,
    profile: Annotated[str | None, Field(description="Profile name or path.")] = None,
    wait_seconds: Annotated[
        float, Field(ge=0, le=300, description="How long to wait for the data segment.")
    ] = 60.0,
    working_directory: str | None = None,
) -> dict[str, Any]:
    resolved = _resolve_profile(profile)
    binary = executable or DEFAULT_EXECUTABLE
    if not binary:
        raise SessionError(
            "no emulator given: pass executable, or set DOSBOX_MCP_EXECUTABLE"
        )
    binary_path = Path(binary).resolve()
    if not binary_path.is_file():
        raise SessionError(f"no emulator at {binary_path}")
    config_path = Path(config).resolve()
    if not config_path.is_file():
        raise SessionError(f"no config file at {config_path}")

    process = subprocess.Popen(
        [str(binary_path), "-conf", str(config_path)],
        cwd=working_directory or str(binary_path.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    session = registry.attach(process.pid, resolved)
    registry.mark_launched(process.pid)

    def settle() -> str:
        deadline = time.time() + wait_seconds
        last = "not attempted"
        while time.time() < deadline:
            if process.poll() is not None:
                return f"the emulator exited with code {process.returncode}"
            try:
                session.locate(rescan=True)
                return "attached"
            except (SessionError, ProcessError) as error:
                last = str(error)
            time.sleep(2.0)
        return last

    outcome = await asyncio.to_thread(settle)
    window = session.window()
    return {
        "pid": process.pid,
        "profile": resolved.name,
        "attached": outcome == "attached",
        "detail": outcome,
        "window_title": window.title if window else None,
        "data_segment_host_address": f"0x{session.ds_base:X}" if session.ds_base else None,
        "config": str(config_path),
    }


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True),
    description=(
        "Attach to a DOSBox-X process that is already running, by process id, and "
        "locate the game's data segment inside it.  Use dosbox_sessions to find "
        "candidate process ids."
    ),
)
async def dosbox_attach(
    pid: Annotated[int, Field(description="DOSBox-X process id.")],
    profile: str | None = None,
) -> dict[str, Any]:
    resolved = _resolve_profile(profile)
    session = registry.attach(pid, resolved)
    await asyncio.to_thread(session.locate, True)
    window = session.window()
    return {
        "pid": pid,
        "profile": resolved.name,
        "attached": True,
        "data_segment_host_address": f"0x{session.ds_base:X}",
        "guest_ram_host_address": (
            f"0x{session.ram_base:X}" if session.ram_base is not None else None
        ),
        "keyboard_injection": session.ram_base is not None,
        "window_title": window.title if window else None,
    }


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description=(
        "List attached sessions and any other visible DOSBox-X windows on the "
        "machine, with their process ids."
    ),
)
def dosbox_sessions(window_title: str | None = None) -> dict[str, Any]:
    title = window_title or "DOSBox-X"
    windows = [
        {"pid": window.pid, "title": window.title} for window in find_windows(title)
    ]
    sessions = []
    for session in registry.all():
        try:
            alive = session.process().alive()
        except ProcessError:
            alive = False
        sessions.append(
            {
                "pid": session.pid,
                "profile": session.profile.name,
                "alive": alive,
                "launched_by_server": registry.launched(session.pid),
                "data_segment_host_address": (
                    f"0x{session.ds_base:X}" if session.ds_base else None
                ),
                "segment_scans": session.scans,
            }
        )
    return {"attached": sessions, "windows": windows}


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=True),
    description=(
        "Close a DOSBox-X session.  By default only sessions this server "
        "launched may be killed; pass force to end one it merely attached to.  "
        "The guest is not asked to save anything first."
    ),
)
async def dosbox_quit(
    pid: Annotated[int | None, Field(description="Process id; omit if only one is attached.")] = None,
    force: Annotated[bool, Field(description="Allow killing a process this server did not start.")] = False,
) -> dict[str, Any]:
    session = _session(pid)
    if not registry.launched(session.pid) and not force:
        raise SessionError(
            f"process {session.pid} was not launched by this server; pass force=true "
            f"to end it anyway"
        )
    target = session.pid
    session.close()

    def kill() -> int:
        completed = subprocess.run(
            ["taskkill", "/PID", str(target), "/T", "/F"],
            capture_output=True,
            text=True,
        )
        return completed.returncode

    code = await asyncio.to_thread(kill)
    registry.forget(target)
    return {"pid": target, "terminated": code == 0, "taskkill_exit_code": code}


# ── input ──────────────────────────────────────────────────────────────────


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True),
    description=(
        "Type into the guest by appending to its BIOS keyboard ring buffer -- no "
        "window focus, no host keystrokes.  Each item is a single character or a "
        "key name (space, enter, esc, tab, backspace, up, down, left, right, "
        "home, end, pgup, pgdn, plus, minus, f1..f10).  Keys are paced apart so "
        "the game consumes them one screen at a time; the ring holds 15, so send "
        "small batches and let the game catch up.  Fails cleanly if the ring is "
        "full, which means the game is waiting on the mouse, not the keyboard."
    ),
)
async def dosbox_send_keys(
    keys: Annotated[list[str], Field(min_length=1, max_length=64, description="Keys to send, in order.")],
    pid: int | None = None,
    pace_seconds: Annotated[float, Field(ge=0, le=10, description="Delay between keys.")] = 0.75,
) -> dict[str, Any]:
    session = _session(pid)
    sent = await asyncio.to_thread(session.press, keys, pace_seconds)
    return {"pid": session.pid, "sent": sent, "count": len(sent)}


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True),
    description=(
        "Click at a guest screen coordinate by writing the game's own mouse "
        "state, the words its INT 33h handler normally fills in.  The host "
        "pointer never moves.  The state is rewritten continuously for `duration` "
        "because the game's handler keeps overwriting it; a single write usually "
        "loses that race.  Beware: on 'click to continue' pages one click can "
        "advance two pages -- send a key instead when paging through lists."
    ),
)
async def dosbox_click(
    x: Annotated[int, Field(ge=0, description="Guest X, 0 to screen width - 1.")],
    y: Annotated[int, Field(ge=0, description="Guest Y, 0 to screen height - 1.")],
    pid: int | None = None,
    duration_seconds: Annotated[float, Field(ge=0.01, le=5, description="How long to hold the click.")] = 0.5,
    button: Annotated[int, Field(ge=1, le=3, description="1 left, 2 right, 3 both.")] = 1,
) -> dict[str, Any]:
    session = _session(pid)
    if not (0 <= x < session.profile.screen_width and 0 <= y < session.profile.screen_height):
        raise SessionError(
            f"({x},{y}) is outside the {session.profile.screen_width}x"
            f"{session.profile.screen_height} guest screen"
        )
    writes = await asyncio.to_thread(session.click, x, y, duration_seconds, button)
    return {"pid": session.pid, "x": x, "y": y, "button": button, "writes": writes}


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True),
    description=(
        "Hold mouse buttons down for a stretch of time by writing the guest's "
        "button word continuously -- the way to drive anything that reads a held "
        "button, such as an action sequence that only advances while a button is "
        "down.  Sparse writes lose to the game's own interrupt handler, so this "
        "busy-writes.  Give stop_when to release as soon as a memory field says "
        "the sequence is over, instead of holding for the full time."
    ),
)
async def dosbox_hold_buttons(
    seconds: Annotated[float, Field(gt=0, le=120, description="Maximum time to hold.")],
    pid: int | None = None,
    buttons: Annotated[int, Field(ge=1, le=3, description="1 left, 2 right, 3 both.")] = 3,
    stop_when_offset: Annotated[
        int | str | None, Field(description="Offset or symbol to watch while holding.")
    ] = None,
    stop_when_size: Annotated[int, Field(ge=1, le=4)] = 1,
    stop_when_equals: int | None = None,
    stop_when_any_of: list[int] | None = None,
    stop_when_at_least: int | None = None,
    poll_seconds: Annotated[float, Field(ge=0.02, le=5)] = 0.25,
) -> dict[str, Any]:
    session = _session(pid)
    condition = None
    if stop_when_offset is not None:
        condition = _condition(
            session,
            stop_when_offset,
            stop_when_size,
            stop_when_equals,
            None,
            stop_when_at_least,
            None,
            stop_when_any_of,
        )
    result = await asyncio.to_thread(session.hold, buttons, seconds, condition, poll_seconds)
    return {
        "pid": session.pid,
        "buttons": buttons,
        "watched": condition.describe() if condition else None,
        **result,
    }


# ── memory ─────────────────────────────────────────────────────────────────


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description=(
        "Read bytes from the game's data segment.  `offset` is a number or a "
        "profile symbol name.  Choose a format: hex (raw bytes), u8/u16/u32 "
        "(little-endian numbers), or text (NUL-terminated string).  This is how "
        "to read game state -- money, a unit's health, which screen is up -- "
        "without any of it being visible on screen."
    ),
)
async def dosbox_read_memory(
    offset: Annotated[int | str, Field(description="Data-segment offset or symbol name.")],
    length: Annotated[int, Field(ge=1, le=4096, description="Bytes to read.")] = 1,
    pid: int | None = None,
    format: Literal["hex", "u8", "u16", "u32", "text"] = "hex",
) -> dict[str, Any]:
    session = _session(pid)
    resolved, symbol_size = session.profile.resolve(offset)
    width = length if length != 1 else symbol_size
    data = await asyncio.to_thread(session.read, resolved, width)
    result: dict[str, Any] = {
        "pid": session.pid,
        "offset": f"{resolved:#06x}",
        "length": width,
    }
    if format == "hex":
        result["hex"] = data.hex()
    elif format == "text":
        result["text"] = data.split(b"\0")[0].decode("latin-1")
    else:
        stride = {"u8": 1, "u16": 2, "u32": 4}[format]
        result["values"] = [
            int.from_bytes(data[i : i + stride], "little")
            for i in range(0, len(data) - len(data) % stride, stride)
        ]
    return result


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=True),
    description=(
        "Patch the game's data segment.  This is how to set up a state that "
        "would otherwise take hours of play -- a treasury, a rank, a flag -- so "
        "a particular screen or code path can be reached.  Writes land in the "
        "running game and are not undoable; the game may recompute a field "
        "shortly after it is written, so patch at a moment where it will be read "
        "before it is recomputed."
    ),
)
async def dosbox_write_memory(
    writes: Annotated[list[MemoryWrite], Field(min_length=1, max_length=64)],
    pid: int | None = None,
) -> dict[str, Any]:
    session = _session(pid)
    prepared: list[tuple[int, bytes]] = []
    for item in writes:
        resolved, symbol_size = session.profile.resolve(item.offset)
        if item.hex is not None:
            payload = bytes.fromhex(item.hex)
        elif item.value is not None:
            width = item.size if item.size != 1 else symbol_size
            try:
                payload = int(item.value).to_bytes(width, "little")
            except OverflowError:
                raise SessionError(
                    f"{item.value} does not fit in {width} byte(s) at {item.offset}"
                ) from None
        else:
            raise SessionError(f"write to {item.offset} gives neither value nor hex")
        prepared.append((resolved, payload))

    def apply() -> list[str]:
        done = []
        for resolved, payload in prepared:
            session.write(resolved, payload)
            done.append(f"DS:{resolved:#06x} = {payload.hex()}")
        return done

    applied = await asyncio.to_thread(apply)
    return {"pid": session.pid, "applied": applied, "count": len(applied)}


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description=(
        "Write the whole 64 KiB data segment to a file -- the state of the game "
        "at one instant, for later analysis or as a test fixture."
    ),
)
async def dosbox_dump_segment(
    path: Annotated[str, Field(description="Output file path.")],
    pid: int | None = None,
) -> dict[str, Any]:
    session = _session(pid)
    segment = await asyncio.to_thread(session.segment)
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(segment)
    return {
        "pid": session.pid,
        "path": str(target),
        "bytes": len(segment),
        "host_address": f"0x{session.ds_base:X}",
    }


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description=(
        "Block until a memory field satisfies a test, or the timeout runs out.  "
        "Use it to wait for a screen transition, an animation to finish, or a "
        "state byte to settle, instead of guessing at sleeps.  Returns whether "
        "the test was satisfied and the last value seen -- it does not raise on "
        "timeout, so a caller can decide what a timeout means."
    ),
)
async def dosbox_wait_for(
    offset: Annotated[int | str, Field(description="Data-segment offset or symbol name.")],
    pid: int | None = None,
    size: Annotated[int, Field(ge=1, le=4)] = 1,
    equals: int | None = None,
    not_equals: int | None = None,
    at_least: int | None = None,
    at_most: int | None = None,
    any_of: list[int] | None = None,
    timeout_seconds: Annotated[float, Field(gt=0, le=600)] = 30.0,
    poll_seconds: Annotated[float, Field(ge=0.05, le=10)] = 0.5,
) -> dict[str, Any]:
    session = _session(pid)
    condition = _condition(session, offset, size, equals, not_equals, at_least, at_most, any_of)
    result = await asyncio.to_thread(session.wait_for, condition, timeout_seconds, poll_seconds)
    return {"pid": session.pid, "condition": condition.describe(), **result}


# ── screen ─────────────────────────────────────────────────────────────────


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description=(
        "Look at the emulator's screen.  Returns the current frame as an image, "
        "photographed from the window without activating it.  Use this to see "
        "where the game actually is between steps; use dosbox_capture_screen "
        "when the frame has to be exact and saved."
    ),
)
async def dosbox_view_screen(
    pid: int | None = None,
    max_width: Annotated[int, Field(ge=160, le=1280, description="Downscale wider images to this.")] = 640,
) -> Image:
    session = _session(pid)
    window = _window_of(session)

    def shoot() -> bytes:
        capture = capture_window(window.handle, session.profile, scale=None, exact=False)
        return downscale_png(capture.png_bytes, max_width)

    return Image(data=await asyncio.to_thread(shoot), format="png")


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True),
    description=(
        "Capture the current frame exactly and save it.  The window is resized "
        "to an integer multiple of the guest resolution and sampled one pixel "
        "per guest pixel, so the palette indices come back exactly as the guest "
        "drew them; writes <stem>.rgb (one byte per pixel), <stem>.png, and with "
        "with_segment also <stem>_ds.bin, the data segment from the same instant "
        "-- which is what makes a captured frame replayable as a fixture.  Fails "
        "rather than writing an approximate frame; pass exact=false for a plain "
        "screenshot instead."
    ),
)
async def dosbox_capture_screen(
    stem: Annotated[str, Field(description="Output path without a suffix; relative to DOSBOX_MCP_OUTPUT_DIR.")],
    pid: int | None = None,
    scale: Annotated[int, Field(ge=1, le=6, description="Window scale to capture at.")] = 2,
    exact: Annotated[bool, Field(description="Require every pixel to be a palette colour.")] = True,
    with_segment: Annotated[bool, Field(description="Also dump the data segment from the same instant.")] = True,
    settle_seconds: Annotated[float, Field(ge=0, le=30, description="Wait for the window to finish resizing.")] = 4.0,
) -> dict[str, Any]:
    session = _session(pid)
    window = _window_of(session)
    target = Path(stem)
    if not target.is_absolute():
        target = DEFAULT_OUTPUT_DIR / target

    def shoot() -> dict[str, Any]:
        segment = session.segment() if with_segment else None
        capture = capture_window(
            window.handle, session.profile, scale=scale, settle=settle_seconds, exact=exact
        )
        written = write_outputs(target, capture, segment)
        return {
            "files": written,
            "scale": capture.scale,
            "distinct_indices": capture.distinct,
            "exact": capture.indices is not None,
        }

    result = await asyncio.to_thread(shoot)
    return {"pid": session.pid, **result}


def main() -> None:
    if sys.platform != "win32":
        raise SystemExit("dosbox-mcp drives a Windows DOSBox-X process; it needs Windows")
    mcp.run()


if __name__ == "__main__":
    main()
