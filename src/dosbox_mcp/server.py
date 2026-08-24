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

from . import emulator, guest as dosguest, platform as host, trace as tracing, vga
from .profile import Profile, ProfileError, load_profile
from .screen import CaptureError, capture_window, downscale_png, write_outputs
from .session import Condition, Registry, Session, SessionError
from .platform import ProcessError, find_windows

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
        "into the guest's own input state, read and patch any guest segment, "
        "wait on and sample memory, read the screen out of video memory, and "
        "watch code as it runs.  Nothing here touches the host desktop, so it "
        "is safe to use while the user is working.  "
        "With a profile for the game, start at dosbox_profiles, then "
        "dosbox_launch or dosbox_attach.  With no profile -- a game nobody has "
        "described yet -- attach anyway and call dosbox_find_guest, which "
        "reports the DOS memory chain and every loaded program; dosbox_search_"
        "memory then turns a known string into a segment, and that is the "
        "marker a profile needs.  dosbox_capabilities says what this host can "
        "do before anything is tried."
    ),
)

registry = Registry()
_profiles: dict[str, Profile] = {}


# ── helpers ────────────────────────────────────────────────────────────────


def _profile_files() -> list[Path]:
    return sorted(DEFAULT_PROFILE_DIR.glob("*.json")) if DEFAULT_PROFILE_DIR.is_dir() else []


def _resolve_profile(name: str | None) -> Profile:
    """Load a profile by name, by path, or fall back to the configured default.

    ``"none"`` asks for no profile at all, which is what a game that has not
    been described yet needs: the guest is still found and addressed, there is
    simply no named data segment.
    """
    if name is not None and name.strip().lower() in ("none", "-", ""):
        return Profile.anonymous()
    wanted = name or DEFAULT_PROFILE_NAME
    if wanted is None:
        files = _profile_files()
        if len(files) == 1:
            wanted = files[0].stem
        elif not files:
            return Profile.anonymous()
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


def _segment_number(value: int | str | None) -> int | None:
    """A guest segment, as a number or a 0x-string.  None means the profile's."""
    if value is None:
        return None
    number = int(value, 0) if isinstance(value, str) else int(value)
    if not 0 <= number <= 0xFFFF:
        raise SessionError(f"{number:#x} is not a guest segment; segments are 16-bit")
    return number


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
        "Waits until the guest is findable, so a successful call means it is "
        "really up and drivable.  On Linux, isolated=true puts the emulator on "
        "its own Xvfb display -- entirely off the user's desktop, and the only "
        "way the emulator's own keyboard shortcuts can be used without taking "
        "a keystroke from whatever the user is doing."
    ),
)
async def dosbox_launch(
    config: Annotated[str, Field(description="Path to the DOSBox-X .conf file.")],
    executable: Annotated[
        str | None, Field(description="dosbox-x.exe to run; defaults to DOSBOX_MCP_EXECUTABLE.")
    ] = None,
    profile: Annotated[str | None, Field(description="Profile name or path.")] = None,
    wait_seconds: Annotated[
        float, Field(ge=0, le=300, description="How long to wait for the guest.")
    ] = 60.0,
    working_directory: str | None = None,
    isolated: Annotated[
        bool, Field(description="Linux only: run on a private Xvfb display.")
    ] = False,
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

    process, display = emulator.launch(
        binary_path, config_path, working_directory, isolated=isolated
    )
    session = registry.attach(process.pid, resolved)
    session.display = display
    session.config = config_path
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
        "guest_ram_host_address": (
            f"0x{session.ram_base:X}" if session.ram_base is not None else None
        ),
        "display": display.name if display is not None else None,
        "config": str(config_path),
        "captures_directory": str(emulator.captures_directory(config_path)),
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

    display = session.display
    code = await asyncio.to_thread(host.terminate, target)
    if display is not None:
        display.stop()
    registry.forget(target)
    return {
        "pid": target,
        "terminated": code == 0,
        "exit_code": code,
        "display_stopped": display.name if display is not None else None,
    }


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
        "Read bytes from the guest.  With no segment this is the game's own "
        "data segment and `offset` may be a profile symbol name; with "
        "segment=0x1234 it is any segment in the guest -- another program in a "
        "launcher chain, a TSR, an overlay, or the interrupt vector table at "
        "segment 0.  Choose a format: hex (raw bytes), u8/u16/u32 "
        "(little-endian numbers), or text (NUL-terminated string).  This is how "
        "to read game state -- money, a unit's health, which screen is up -- "
        "without any of it being visible on screen."
    ),
)
async def dosbox_read_memory(
    offset: Annotated[int | str, Field(description="Offset, or a symbol name when no segment is given.")],
    length: Annotated[int, Field(ge=1, le=4096, description="Bytes to read.")] = 1,
    pid: int | None = None,
    format: Literal["hex", "u8", "u16", "u32", "text"] = "hex",
    segment: Annotated[
        int | str | None,
        Field(description="Guest segment; omit for the profile's data segment."),
    ] = None,
) -> dict[str, Any]:
    session = _session(pid)
    where = _segment_number(segment)
    if where is None:
        resolved, symbol_size = session.profile.resolve(offset)
        width = length if length != 1 else symbol_size
        data = await asyncio.to_thread(session.read, resolved, width)
    else:
        resolved = int(offset, 0) if isinstance(offset, str) else int(offset)
        width = length
        data = await asyncio.to_thread(
            lambda: session.guest().read(where, resolved, width)
        )
    result: dict[str, Any] = {
        "pid": session.pid,
        "segment": f"{where:04X}" if where is not None else session.profile.name,
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
    segment: Annotated[
        int | str | None,
        Field(description="Guest segment for every write; omit for the profile's."),
    ] = None,
) -> dict[str, Any]:
    session = _session(pid)
    where = _segment_number(segment)
    prepared: list[tuple[int, bytes]] = []
    for item in writes:
        if where is None:
            resolved, symbol_size = session.profile.resolve(item.offset)
        else:
            resolved = int(item.offset, 0) if isinstance(item.offset, str) else int(item.offset)
            symbol_size = 1
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
        target = session.guest() if where is not None else None
        for resolved, payload in prepared:
            if target is None:
                session.write(resolved, payload)
                done.append(f"DS:{resolved:#06x} = {payload.hex()}")
            else:
                target.write(where, resolved, payload)
                done.append(f"{where:04X}:{resolved:04X} = {payload.hex()}")
        return done

    applied = await asyncio.to_thread(apply)
    return {"pid": session.pid, "applied": applied, "count": len(applied)}


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description=(
        "Write a whole 64 KiB guest segment to a file -- the state at one "
        "instant, for later analysis or as a test fixture.  With no segment "
        "this is the profile's data segment; with segment=0x1234 it is any "
        "segment in the guest, which is how the marker for a new profile is "
        "captured."
    ),
)
async def dosbox_dump_segment(
    path: Annotated[str, Field(description="Output file path.")],
    pid: int | None = None,
    segment: Annotated[
        int | str | None,
        Field(description="Guest segment; omit for the profile's data segment."),
    ] = None,
) -> dict[str, Any]:
    session = _session(pid)
    where = _segment_number(segment)
    if where is None:
        data = await asyncio.to_thread(session.segment)
        source = f"0x{session.ds_base:X}"
    else:
        data = await asyncio.to_thread(lambda: session.guest().read(where, 0, 0x10000))
        source = f"{where:04X}:0000"
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {
        "pid": session.pid,
        "path": str(target),
        "bytes": len(data),
        "source": source,
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
        "Capture the current frame exactly and save it: <stem>.rgb (one byte "
        "per pixel), <stem>.png, and with with_segment also <stem>_ds.bin, the "
        "data segment from the same instant -- which is what makes a captured "
        "frame replayable as a fixture.\n\n"
        "source='vram' reads the frame out of the emulator's video memory, "
        "which is exact, cheap, cannot catch a half-drawn frame and does not "
        "slow the guest.  source='window' photographs the window instead, "
        "resizing it to an integer multiple of the guest resolution; that is "
        "the one visible effect this server has on the desktop, and it costs "
        "the guest real time.  'auto' takes video memory when the page can be "
        "found and falls back to the window."
    ),
)
async def dosbox_capture_screen(
    stem: Annotated[str, Field(description="Output path without a suffix; relative to DOSBOX_MCP_OUTPUT_DIR.")],
    pid: int | None = None,
    scale: Annotated[int, Field(ge=1, le=6, description="Window scale to capture at, for source='window'.")] = 2,
    exact: Annotated[bool, Field(description="Require every pixel to be a palette colour.")] = True,
    with_segment: Annotated[bool, Field(description="Also dump the data segment from the same instant.")] = True,
    settle_seconds: Annotated[float, Field(ge=0, le=30, description="Wait for the window to finish resizing.")] = 4.0,
    source: Literal["auto", "vram", "window"] = "auto",
) -> dict[str, Any]:
    session = _session(pid)
    target = Path(stem)
    if not target.is_absolute():
        target = DEFAULT_OUTPUT_DIR / target

    def from_vram() -> dict[str, Any]:
        found = _locate_framebuffer(session)
        page = session.framebuffer().read()
        segment = session.segment() if with_segment else None
        target.parent.mkdir(parents=True, exist_ok=True)
        written = [str(target.with_suffix(".rgb"))]
        target.with_suffix(".rgb").write_bytes(page)
        palette = vga.palette_bytes(session.profile.palette) if session.profile.palette else None
        target.with_suffix(".png").write_bytes(
            vga.png(page, palette, session.profile.screen_width, session.profile.screen_height)
        )
        written.append(str(target.with_suffix(".png")))
        if segment is not None:
            dump = target.with_name(f"{target.name}_ds.bin")
            dump.write_bytes(segment)
            written.append(str(dump))
        return {
            "source": "vram",
            "files": written,
            "distinct_indices": len(set(page)),
            "exact": True,
            **found.as_dict(),
        }

    def from_window() -> dict[str, Any]:
        window = _window_of(session)
        segment = session.segment() if with_segment else None
        capture = capture_window(
            window.handle, session.profile, scale=scale, settle=settle_seconds, exact=exact
        )
        written = write_outputs(target, capture, segment)
        return {
            "source": "window",
            "files": written,
            "scale": capture.scale,
            "distinct_indices": capture.distinct,
            "exact": capture.indices is not None,
        }

    def shoot() -> dict[str, Any]:
        if source == "window":
            return from_window()
        try:
            return from_vram()
        except (vga.VgaError, SessionError, ProcessError) as failure:
            if source == "vram":
                raise
            fell_back = from_window()
            fell_back["fell_back_because"] = str(failure)
            return fell_back

    result = await asyncio.to_thread(shoot)
    return {"pid": session.pid, **result}



# -- the guest, with or without a profile ----------------------------------


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    description=(
        "What this host can do, and what it cannot.  Guest memory works "
        "everywhere there is a backend; window capture and the emulator's own "
        "menu commands do not.  Call this when a tool refuses, or before "
        "planning a session that depends on frames or recordings."
    ),
)
def dosbox_capabilities() -> dict[str, Any]:
    return host.capabilities().as_dict()


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description=(
        "Find the DOS guest inside an emulator process and report what is "
        "loaded in it -- with no profile and no prior knowledge.  This is the "
        "way in for a game nobody has described yet: it locates guest physical "
        "zero from the BIOS data area, then walks the DOS memory chain and "
        "names every loaded program, its load segment and the path it was "
        "started from.  Those segments are what dosbox_read_memory(segment=...) "
        "and dosbox_search_memory take, and what a new profile is written from."
    ),
)
async def dosbox_find_guest(pid: int | None = None) -> dict[str, Any]:
    session = _session(pid)

    def look() -> dict[str, Any]:
        process = session.process()
        bases = dosguest.ram_bases(process)
        if not bases:
            raise SessionError(
                f"no DOS guest in process {session.pid}: nothing in its memory looks "
                f"like a BIOS data area.  The emulator may not have booted DOS yet."
            )
        out: dict[str, Any] = {
            "pid": session.pid,
            "guests": [f"0x{value:X}" for value in bases],
        }
        target = dosguest.Guest(process=process, ram_base=bases[0])
        out.update(dosguest.describe(target))
        return out

    return await asyncio.to_thread(look)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description=(
        "Search the guest's 640 KiB of conventional memory for a byte pattern "
        "and report every hit as a guest segment:offset.  This is how a known "
        "string -- a file name the game opens, a message it prints -- becomes "
        "the segment that holds it, which is the first step in writing a "
        "profile.  Give text, or hex for raw bytes."
    ),
)
async def dosbox_search_memory(
    text: Annotated[str | None, Field(description="Literal text to find (latin-1).")] = None,
    hex: Annotated[str | None, Field(description="Raw bytes as hex, instead of text.")] = None,
    pid: int | None = None,
    limit: Annotated[int, Field(ge=1, le=512)] = 64,
) -> dict[str, Any]:
    if (text is None) == (hex is None):
        raise SessionError("give exactly one of text or hex")
    needle = bytes.fromhex(hex) if hex is not None else text.encode("latin-1")
    if not needle:
        raise SessionError("the search pattern is empty")
    session = _session(pid)

    def look() -> list[str]:
        found = session.guest().search(needle, limit)
        return [f"{segment:04X}:{offset:04X}" for segment, offset in found]

    hits = await asyncio.to_thread(look)
    return {
        "pid": session.pid,
        "pattern": needle.hex(),
        "hits": hits,
        "count": len(hits),
        "truncated": len(hits) >= limit,
    }


# -- the emulator itself ---------------------------------------------------


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True),
    description=(
        "Fire one of DOSBox-X's own menu items without focus and without a "
        "keystroke: record_opl, record_midi, record_wave, record_video, "
        "screenshot, save_state, load_state.  Firing a record action again "
        "stops that recording.  Windows only -- DOSBox-X draws its own menu "
        "elsewhere; see dosbox_capabilities."
    ),
)
async def dosbox_emulator_command(
    action: Annotated[
        str,
        Field(
            description="record_opl, record_midi, record_wave, record_video, "
            "screenshot, save_state, load_state."
        ),
    ],
    pid: int | None = None,
) -> dict[str, Any]:
    session = _session(pid)
    result = await asyncio.to_thread(emulator.command, session.pid, action)
    return {"pid": session.pid, **result}


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True),
    description=(
        "Record the guest's sound for a stretch of time and return the file "
        "that appeared: an OPL register log (.dro), the MIDI stream (.mid) or "
        "the mixer output (.wav).  This is the only independent record of what "
        "a game's sound driver actually writes.  Arm it EARLY -- DOSBox-X "
        "starts an OPL capture at the guest's first register write, so arming "
        "late does not truncate the capture, it starts it somewhere else "
        "entirely."
    ),
)
async def dosbox_record(
    seconds: Annotated[float, Field(gt=0, le=600, description="How long to record.")],
    pid: int | None = None,
    kind: Literal["record_opl", "record_midi", "record_wave", "record_video"] = "record_opl",
    config: Annotated[
        str | None, Field(description="Config file, to find its captures= directory.")
    ] = None,
    captures_directory: str | None = None,
    settle_seconds: Annotated[float, Field(ge=0, le=60)] = 0.0,
) -> dict[str, Any]:
    session = _session(pid)
    where = Path(config) if config else session.config
    if where is None:
        raise SessionError(
            "which config this emulator was started with is not known, so its "
            "captures directory cannot be found; pass config or captures_directory"
        )
    result = await asyncio.to_thread(
        emulator.record,
        session.pid,
        Path(where),
        kind,
        seconds,
        settle_seconds,
        Path(captures_directory) if captures_directory else None,
    )
    return {"pid": session.pid, **result}


# -- the framebuffer -------------------------------------------------------


def _reference_from_window(session: Session) -> bytes | None:
    """One frame off the window, as indices, to anchor the video-memory search."""
    window = session.window()
    if window is None or not session.profile.palette:
        return None
    try:
        capture = capture_window(window.handle, session.profile, scale=2, settle=2.0, exact=True)
    except (CaptureError, ProcessError):
        return None
    return capture.indices


def _locate_framebuffer(session: Session) -> vga.Located:
    buffer = session.framebuffer()
    if buffer.found is not None:
        return buffer.found
    reference = _reference_from_window(session) if host.capabilities().window_capture else None
    try:
        return buffer.locate(reference)
    except vga.VgaError:
        if reference is None:
            raise
        return buffer.locate(None)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description=(
        "Read the guest's screen straight out of the emulator's video memory: "
        "the palette indices the guest wrote, at the instant of the read, with "
        "no window, no scaling and no palette round trip.  Prefer this over "
        "dosbox_capture_screen for anything quantitative -- photographing the "
        "window costs the guest real time and can catch a half-drawn frame.  "
        "Writes <stem>.bin (one byte per pixel) and <stem>.png when stem is "
        "given.  A page-flipping game has more than one page; pass page_offset "
        "from the pages list to read the other."
    ),
)
async def dosbox_read_framebuffer(
    pid: int | None = None,
    stem: Annotated[
        str | None, Field(description="Output path without a suffix; omit to only report.")
    ] = None,
    page_offset: Annotated[
        int | None,
        Field(description="Byte offset of the page, from a previous call's pages list."),
    ] = None,
    relocate: Annotated[
        bool, Field(description="Search for the page again rather than reusing the cached one.")
    ] = False,
) -> dict[str, Any]:
    session = _session(pid)

    def read() -> dict[str, Any]:
        if relocate:
            session.framebuffer().found = None
        found = _locate_framebuffer(session)
        buffer = session.framebuffer()
        page = buffer.read(page_offset)
        result: dict[str, Any] = {
            "pid": session.pid,
            **found.as_dict(),
            "pages": buffer.pages(),
            "distinct_indices": len(set(page)),
            "bytes": len(page),
        }
        if stem:
            target = Path(stem)
            if not target.is_absolute():
                target = DEFAULT_OUTPUT_DIR / target
            target.parent.mkdir(parents=True, exist_ok=True)
            target.with_suffix(".bin").write_bytes(page)
            palette = (
                vga.palette_bytes(session.profile.palette) if session.profile.palette else None
            )
            target.with_suffix(".png").write_bytes(
                vga.png(
                    page,
                    palette,
                    session.profile.screen_width,
                    session.profile.screen_height,
                )
            )
            result["files"] = [
                str(target.with_suffix(".bin")),
                str(target.with_suffix(".png")),
            ]
        return result

    return await asyncio.to_thread(read)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True),
    description=(
        "Watch the guest's video memory and return every distinct frame it put "
        "up, with the moment each appeared.  This is how to measure an "
        "animation, a fade or a transition -- reading video memory does not "
        "slow the guest the way filming the window does, so the timings are "
        "the guest's own.  Frames are written as <stem>_NNN.png."
    ),
)
async def dosbox_watch_frames(
    seconds: Annotated[float, Field(gt=0, le=300, description="How long to watch.")],
    stem: Annotated[
        str | None, Field(description="Where to write the frames; omit to only count them.")
    ] = None,
    pid: int | None = None,
    interval_seconds: Annotated[float, Field(ge=0.005, le=1.0)] = 0.02,
    max_frames: Annotated[int, Field(ge=1, le=2000)] = 200,
) -> dict[str, Any]:
    session = _session(pid)

    def watch() -> dict[str, Any]:
        _locate_framebuffer(session)
        frames = session.framebuffer().watch(seconds, interval_seconds)
        kept = frames[:max_frames]
        result: dict[str, Any] = {
            "pid": session.pid,
            "distinct_frames": len(frames),
            "at_seconds": [when for when, _ in kept],
            "dropped": max(0, len(frames) - len(kept)),
        }
        if stem:
            target = Path(stem)
            if not target.is_absolute():
                target = DEFAULT_OUTPUT_DIR / target
            target.parent.mkdir(parents=True, exist_ok=True)
            palette = (
                vga.palette_bytes(session.profile.palette) if session.profile.palette else None
            )
            written = []
            for index, (_, page) in enumerate(kept):
                path = target.with_name(f"{target.name}_{index:03d}.png")
                path.write_bytes(
                    vga.png(
                        page,
                        palette,
                        session.profile.screen_width,
                        session.profile.screen_height,
                    )
                )
                written.append(str(path))
            result["files"] = written
        return result

    return await asyncio.to_thread(watch)


# -- sampling --------------------------------------------------------------


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True),
    description=(
        "Sample a set of memory fields repeatedly and write the series as a "
        "TSV.  dosbox_read_memory answers what a value is and dosbox_wait_for "
        "answers when it becomes something; this answers how it MOVES, which "
        "is what most questions about a running program actually are.  Every "
        "field in a row comes from one snapshot, so the columns do not drift "
        "against each other.  watch takes symbol names, offsets, or the name "
        "of a watch set the profile defines.  The whole segment at the entry "
        "instant is dumped beside the TSV, so a question nobody thought to put "
        "in the watch list can still be answered offline."
    ),
)
async def dosbox_sample(
    watch: Annotated[
        list[str],
        Field(min_length=1, max_length=64, description="Symbols, offsets, or a watch-set name."),
    ],
    seconds: Annotated[float, Field(gt=0, le=600, description="How long to sample.")] = 6.0,
    pid: int | None = None,
    hz: Annotated[float, Field(gt=0, le=200, description="Samples per second.")] = 20.0,
    out: Annotated[
        str | None, Field(description="Write <out>.tsv and <out>_entry_ds.bin.")
    ] = None,
    pokes: Annotated[
        list[MemoryWrite] | None,
        Field(description="Written once after the entry sample, before the series."),
    ] = None,
    settle_seconds: Annotated[
        float,
        Field(ge=0, le=60, description="Wait between the entry sample and the pokes."),
    ] = 0.0,
) -> dict[str, Any]:
    session = _session(pid)
    fields = session.resolve_watch(watch)
    prepared: list[tuple[int, int, int]] = []
    for item in pokes or ():
        offset, symbol_size = session.profile.resolve(item.offset)
        if item.value is None:
            raise SessionError(f"poke at {item.offset} gives no value")
        prepared.append(
            (offset, int(item.value), item.size if item.size != 1 else symbol_size)
        )

    def run() -> dict[str, Any]:
        entry = session.segment()
        rows = session.sample(fields, seconds, hz, prepared, settle_seconds)
        result: dict[str, Any] = {
            "pid": session.pid,
            "samples": len(rows),
            "columns": ["t_ms"] + [name for name, _, _ in fields],
            "entry": rows[0] if rows else {},
            "last": rows[-1] if rows else {},
        }
        moved = {name for name, _, _ in fields if len({row[name] for row in rows}) > 1}
        result["changed"] = sorted(moved)
        result["constant"] = sorted({name for name, _, _ in fields} - moved)
        if out:
            target = Path(out)
            if not target.is_absolute():
                target = DEFAULT_OUTPUT_DIR / target
            target.parent.mkdir(parents=True, exist_ok=True)
            columns = result["columns"]
            with target.with_suffix(".tsv").open("w", encoding="ascii") as handle:
                handle.write("\t".join(columns) + "\n")
                for row in rows:
                    handle.write("\t".join(str(row.get(name, -1)) for name in columns) + "\n")
            dump = target.with_name(f"{target.name}_entry_ds.bin")
            dump.write_bytes(entry)
            result["files"] = [str(target.with_suffix(".tsv")), str(dump)]
        return result

    return await asyncio.to_thread(run)


# -- watching code run -----------------------------------------------------


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description=(
        "Find runs of zero in a guest code segment long enough to hold a trace "
        "cave.  Padding between routines is the usual supply.  Longest first; "
        "pass one of these offsets to dosbox_install_trace."
    ),
)
async def dosbox_find_cave(
    segment: Annotated[int | str, Field(description="Guest code segment, e.g. 0x1000.")],
    bytes_needed: Annotated[
        int, Field(ge=8, le=512, description="How much room the cave needs.")
    ] = 64,
    pid: int | None = None,
) -> dict[str, Any]:
    session = _session(pid)
    where = int(segment, 0) if isinstance(segment, str) else segment

    def look() -> list[dict[str, Any]]:
        code = session.guest().read(where, 0, 0x10000)
        return [
            {"offset": f"{offset:#06x}", "length": length}
            for offset, length in tracing.find_caves(code, bytes_needed)
        ]

    caves = await asyncio.to_thread(look)
    return {"pid": session.pid, "segment": f"{where:04X}", "caves": caves}


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=True),
    description=(
        "Watch a running guest's code without a debugger.  Redirects one near "
        "CALL in the live code segment through a zero-filled cave that calls "
        "the displaced target, preserves its flags, and records the registers "
        "and memory words you name -- then read them with dosbox_read_trace "
        "while the guest runs at full speed.  Nothing on disk changes; the "
        "patch dies with the emulator.  "
        "captures take: a register (ax bx cx dx si di bp sp), a segment "
        "register (ds es ss cs), flags, or memory -- word:0x9D10, "
        "byte:DS:0x7A0E, word:ES:0x0004.  Memory is read through whatever DS "
        "holds when the cave runs; capture ds too if the routine changes it.  "
        "Give expect_hex (the three bytes the call site should hold) so the "
        "trace refuses rather than landing on a moved target."
    ),
)
async def dosbox_install_trace(
    segment: Annotated[int | str, Field(description="Guest code segment.")],
    call_at: Annotated[int | str, Field(description="Offset of the near CALL to displace.")],
    captures: Annotated[
        list[str], Field(min_length=1, max_length=24, description="What to record at each hit.")
    ],
    cave_at: Annotated[
        int | str, Field(description="Offset of a zero-filled run, from dosbox_find_cave.")
    ],
    pid: int | None = None,
    expect_hex: Annotated[
        str | None, Field(description="The three bytes the call site must hold.")
    ] = None,
    handle: Annotated[
        str | None, Field(description="Name for this trace; defaults to segment_offset.")
    ] = None,
) -> dict[str, Any]:
    session = _session(pid)
    where = int(segment, 0) if isinstance(segment, str) else segment
    site = int(call_at, 0) if isinstance(call_at, str) else call_at
    cave = int(cave_at, 0) if isinstance(cave_at, str) else cave_at

    def put() -> dict[str, Any]:
        installed = tracing.install(
            session.guest(),
            where,
            site,
            list(captures),
            cave,
            bytes.fromhex(expect_hex) if expect_hex else None,
            handle,
        )
        session.traces[installed.handle] = installed
        return installed.describe()

    return {"pid": session.pid, **await asyncio.to_thread(put)}


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description=(
        "Read what an installed trace has recorded: the values from its most "
        "recent hit, and how many times it has fired.  A hit count of zero "
        "means the traced call has not run -- which is itself an answer."
    ),
)
async def dosbox_read_trace(
    handle: Annotated[
        str | None, Field(description="Trace handle; omit if only one is installed.")
    ] = None,
    pid: int | None = None,
) -> dict[str, Any]:
    session = _session(pid)
    installed = _trace_of(session, handle)
    result = await asyncio.to_thread(tracing.read, session.guest(), installed)
    return {"pid": session.pid, **result}


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=True),
    description=(
        "Remove an installed trace: the call site is restored first so nothing "
        "new can enter the cave, then the cave is blanked."
    ),
)
async def dosbox_remove_trace(
    handle: Annotated[
        str | None, Field(description="Trace handle; omit if only one is installed.")
    ] = None,
    pid: int | None = None,
) -> dict[str, Any]:
    session = _session(pid)
    installed = _trace_of(session, handle)
    await asyncio.to_thread(tracing.remove, session.guest(), installed)
    session.traces.pop(installed.handle, None)
    return {
        "pid": session.pid,
        "removed": installed.handle,
        "remaining": sorted(session.traces),
    }


def _trace_of(session: Session, handle: str | None) -> tracing.Trace:
    if handle is not None:
        found = session.traces.get(handle)
        if found is None:
            known = ", ".join(sorted(session.traces)) or "(none installed)"
            raise SessionError(f"no trace named {handle!r}.  Installed: {known}")
        return found
    if len(session.traces) == 1:
        return next(iter(session.traces.values()))
    if not session.traces:
        raise SessionError("no traces are installed; call dosbox_install_trace first")
    raise SessionError(
        f"several traces are installed ({', '.join(sorted(session.traces))}); pass handle"
    )


def main() -> None:
    have = host.capabilities()
    if not have.memory:
        raise SystemExit(
            "dosbox-x-mcp cannot reach guest memory on this host.  "
            + "  ".join(have.notes)
        )
    mcp.run()


if __name__ == "__main__":
    main()
