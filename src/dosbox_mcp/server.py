"""An MCP server for driving a DOSBox-X guest without touching the desktop.

Every tool here works through the emulator process itself -- its memory and its
window -- so nothing steals focus, moves the pointer, or types into whatever the
user happens to be doing.  Menu keys go into the BIOS ring; authentic held keys
use a versioned DOSBox-X hook; clicks enter the game's post-INT-33 mouse words.

Configuration comes from the environment:

    DOSBOX_MCP_PROFILE_DIR   where profiles live (default: ../profiles)
    DOSBOX_MCP_PROFILE       default profile name (default: the only one there)
    DOSBOX_MCP_EXECUTABLE    dosbox-x.exe to launch
    DOSBOX_MCP_OUTPUT_DIR    default directory for captures
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP, Image
from mcp.types import ToolAnnotations

from . import emulator, guest as dosguest, perception, platform as host, trace as tracing, vga
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
        "marker a profile needs.  dosbox_compare_frames, dosbox_act_and_observe, "
        "dosbox_scan_changes and dosbox_correlate_memory add structured visual/"
        "memory experiments without assigning game semantics to the evidence.  "
        "dosbox_key_state and dosbox_hold_keys provide real make/break input "
        "when DOSBox-X has the companion keyboard hook.  "
        "dosbox_capabilities says what this host can do before anything is tried."
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


def _output(path: str) -> Path:
    """Where a tool writes.  One rule, so two tools never disagree about it.

    A relative path is relative to ``DOSBOX_MCP_OUTPUT_DIR``, not to whatever
    the server's working directory happens to be -- which is not knowable from
    the caller's side.  Absolute paths are taken as given.  Every tool returns
    the resolved path, so there is never any doubt where a file went.
    """
    target = Path(path)
    if not target.is_absolute():
        target = DEFAULT_OUTPUT_DIR / target
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _input(path: str) -> Path:
    """Resolve an existing input by the same rule used for tool outputs."""
    target = Path(path)
    if not target.is_absolute():
        target = DEFAULT_OUTPUT_DIR / target
    target = target.resolve()
    if not target.is_file():
        raise SessionError(f"no input file at {target}")
    return target


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


class MemoryProbe(BaseModel):
    """One numeric field sampled during an action experiment."""

    label: str | None = Field(
        default=None,
        min_length=1,
        max_length=80,
        description="Neutral result label; defaults to the supplied offset/address.",
    )
    offset: int | str = Field(
        description="Offset, or a profile symbol when segment is omitted."
    )
    size: int = Field(default=1, ge=1, le=4, description="Little-endian field width.")
    segment: int | str | None = Field(
        default=None,
        description="Guest segment; omit for the profile's named data segment.",
    )


class MemoryRangeProbe(BaseModel):
    """One bounded range snapshotted before and after an experiment."""

    offset: int | str = Field(description="First byte offset in the segment.")
    length: int = Field(ge=1, le=65536, description="Number of bytes to compare.")
    segment: int | str | None = Field(
        default=None,
        description="Guest segment; omit for the profile's named data segment.",
    )
    value_size: Literal[1, 2, 4] = Field(
        default=1, description="Candidate field width for the returned change scan."
    )
    aligned: bool = Field(
        default=True,
        description="Only test offsets aligned to value_size, avoiding overlapping duplicates.",
    )
    relation: Literal["changed", "unchanged", "increased", "decreased", "delta"] = Field(
        default="changed", description="Numeric comparison applied to candidate fields."
    )
    expected_delta: int | None = Field(
        default=None, description="Required when relation=delta."
    )
    candidate_limit: int = Field(
        default=100, ge=1, le=2000, description="Maximum ranked candidates returned."
    )


class ExperimentAction(BaseModel):
    """One technically valid bounded input or control action."""

    kind: Literal["idle", "keys", "hold_keys", "click", "hold_buttons"] = Field(
        default="idle",
        description=(
            "Idle control, BIOS key events, authentic simultaneous held keys, "
            "one click, or held game mouse buttons."
        ),
    )
    keys: list[str] = Field(
        default_factory=list,
        max_length=15,
        description="Key names/characters for kind=keys; these are BIOS queue events, not held keys.",
    )
    pace_seconds: float = Field(
        default=0.0, ge=0, le=5, description="Delay between BIOS key events."
    )
    x: int | None = Field(default=None, ge=0, description="Guest X for kind=click.")
    y: int | None = Field(default=None, ge=0, description="Guest Y for kind=click.")
    buttons: int = Field(
        default=1, ge=1, le=3, description="Mouse buttons: 1 left, 2 right, 3 both."
    )
    duration_seconds: float = Field(
        default=0.25,
        ge=0.01,
        le=30,
        description="Click/hold duration; must fit inside the observation interval.",
    )


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
        "raw_keyboard": session.raw_keyboard_supported(),
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
        "raw_keyboard": session.raw_keyboard_supported(),
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
                "raw_keyboard": session.raw_keyboard_supported(),
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
        "Set one authentic DOSBox-X keyboard make/break state without activating "
        "its window.  This requires the versioned companion keyboard hook.  A "
        "pressed key is automatically released when its lease expires, when the "
        "session closes, or when pressed=false is sent, so interrupted clients do "
        "not leave keys stuck."
    ),
)
async def dosbox_key_state(
    key: Annotated[str, Field(description="Physical key name, such as right, space, or leftctrl.")],
    pressed: bool,
    pid: int | None = None,
    lease_seconds: Annotated[
        float, Field(gt=0, le=30, description="Automatic release deadline for pressed=true.")
    ] = 5.0,
) -> dict[str, Any]:
    session = _session(pid)
    result = await asyncio.to_thread(session.key_state, key, pressed, lease_seconds)
    return {"pid": session.pid, "transport": "dosbox_x_keyboard_hook_v1", **result}


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True),
    description=(
        "Hold one or more physical keys simultaneously through DOSBox-X's own "
        "KEYBOARD_AddKey path, then release every key in reverse order even if "
        "the hold is interrupted.  This supplies authentic make/break input to "
        "IRQ1/action games without focus or host keystrokes."
    ),
)
async def dosbox_hold_keys(
    keys: Annotated[list[str], Field(min_length=1, max_length=16)],
    duration_seconds: Annotated[float, Field(gt=0, le=30)],
    pid: int | None = None,
) -> dict[str, Any]:
    session = _session(pid)
    result = await asyncio.to_thread(session.hold_keys, keys, duration_seconds)
    return {"pid": session.pid, "transport": "dosbox_x_keyboard_hook_v1", **result}


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
    path: Annotated[
        str,
        Field(description="Output file; a relative path goes under DOSBOX_MCP_OUTPUT_DIR."),
    ],
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
    target = _output(path)
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
        "the guest real time.  'auto' takes video memory when the page has been "
        "CONFIRMED against a reference and falls back to the window otherwise, "
        "so it never quietly saves the wrong 64,000 bytes."
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
    reference: Annotated[
        str | None,
        Field(description="Path to a 64,000-byte frame on screen now; confirms a vram capture."),
    ] = None,
) -> dict[str, Any]:
    session = _session(pid)
    target = _output(stem)

    def from_vram() -> dict[str, Any]:
        found = _locate_framebuffer(session, reference)
        page = session.framebuffer().read()
        segment = session.segment() if with_segment else None
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


def _load_reference(session: Session, reference: str | None) -> bytes | None:
    """A reference frame: from a file, or photographed off the window."""
    if reference:
        path = Path(reference)
        if not path.is_absolute():
            path = DEFAULT_OUTPUT_DIR / path
        if not path.is_file():
            raise SessionError(f"no reference frame at {path}")
        raw = path.read_bytes()
        wanted = session.profile.screen_width * session.profile.screen_height
        if len(raw) != wanted:
            raise SessionError(
                f"a reference is {wanted} bytes, one palette index per pixel; "
                f"{path} is {len(raw)}"
            )
        return raw
    if host.capabilities().window_capture:
        return _reference_from_window(session)
    return None


def _locate_framebuffer(
    session: Session, reference: str | None = None, allow_unconfirmed: bool = False
) -> vga.Located:
    """Find the page, preferring a reference and saying so when there is none."""
    buffer = session.framebuffer()
    if buffer.found is not None and (buffer.found.confirmed or allow_unconfirmed):
        return buffer.found
    anchor = _load_reference(session, reference)
    if anchor is not None:
        try:
            return buffer.locate(anchor)
        except vga.VgaError:
            if not allow_unconfirmed:
                raise
    if not allow_unconfirmed:
        raise vga.VgaError(
            "the page could not be confirmed.  A blind scan finds the most "
            "picture-like page in the emulator's memory, and on a real guest "
            "that is often a sprite bank or an offscreen buffer rather than the "
            "screen -- one measured against a live game scored 0.999 and was "
            "wrong.  Give reference= a 64,000-byte frame of what is on screen "
            "(a port's own render of it is ideal), or pass "
            "allow_unconfirmed=true to take the guess and check it yourself."
        )
    return buffer.locate(None)


def _indexed_frame(path: str, width: int, height: int) -> tuple[Path, bytes]:
    source = _input(path)
    raw = source.read_bytes()
    wanted = width * height
    if len(raw) != wanted:
        raise perception.PerceptionError(
            f"indexed frame {source} must be exactly {wanted} bytes for "
            f"{width}x{height}; it has {len(raw)}"
        )
    return source, raw


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    description=(
        "Compare two or more saved indexed framebuffers and return compact, "
        "structured visual evidence: exact changed-pixel counts, connected "
        "changed regions, and neutral region tracks across a sequence.  Frame "
        "files contain one palette index per pixel, as written by "
        "dosbox_read_framebuffer or dosbox_act_and_observe.  bridge_radius can "
        "join the separated old/new footprints of a moving shape.  Track IDs "
        "are visual associations only, never claims that a region is a player, "
        "enemy, wall, or other game object."
    ),
)
async def dosbox_compare_frames(
    frames: Annotated[
        list[str],
        Field(min_length=2, max_length=100, description="Indexed .bin frames in time order."),
    ],
    width: Annotated[int, Field(ge=1, le=4096)] = 320,
    height: Annotated[int, Field(ge=1, le=4096)] = 200,
    bridge_radius: Annotated[
        int, Field(ge=0, le=32, description="Difference-mask dilation radius in pixels.")
    ] = 0,
    min_changed_pixels: Annotated[
        int, Field(ge=1, description="Discard connected regions with fewer exact changes.")
    ] = 1,
    connectivity: Literal[4, 8] = 8,
    track_max_distance: Annotated[float, Field(gt=0, le=1024)] = 48.0,
    track_max_missed: Annotated[int, Field(ge=0, le=20)] = 1,
    global_change_fraction: Annotated[float, Field(gt=0, le=1)] = 0.35,
) -> dict[str, Any]:
    def compare() -> dict[str, Any]:
        loaded = [_indexed_frame(path, width, height) for path in frames]
        comparisons = [
            perception.compare_frames(
                before[1],
                after[1],
                width,
                height,
                bridge_radius=bridge_radius,
                min_changed_pixels=min_changed_pixels,
                connectivity=connectivity,
            )
            for before, after in zip(loaded, loaded[1:])
        ]
        transitions = []
        for index, evidence in enumerate(comparisons):
            transitions.append(
                {
                    "from": str(loaded[index][0]),
                    "to": str(loaded[index + 1][0]),
                    **evidence.as_dict(),
                }
            )
        return {
            "frame_count": len(loaded),
            "transitions": transitions,
            "tracking": perception.track_regions(
                comparisons,
                max_distance=track_max_distance,
                max_missed=track_max_missed,
                global_change_fraction=global_change_fraction,
            ),
            "epistemic_status": "observed visual evidence; no game semantics inferred",
        }

    return await asyncio.to_thread(compare)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    description=(
        "Compare bounded binary memory snapshots and rank numeric offsets by "
        "observed change.  Supports 1-, 2-, and 4-byte little-endian fields, "
        "exact deltas, an idle/control snapshot, and an opposite-action "
        "snapshot.  Use aligned=true to avoid floods of overlapping shifted "
        "interpretations.  Results are candidate offsets with evidence, not "
        "semantic declarations about what the fields mean."
    ),
)
async def dosbox_scan_changes(
    baseline: Annotated[str, Field(description="Baseline binary snapshot path.")],
    experiment: Annotated[str, Field(description="Post-experiment binary snapshot path.")],
    value_size: Literal[1, 2, 4] = 1,
    relation: Literal["changed", "unchanged", "increased", "decreased", "delta"] = "changed",
    expected_delta: int | None = None,
    control: Annotated[
        str | None, Field(description="Optional post-idle/control snapshot from the same baseline.")
    ] = None,
    opposite: Annotated[
        str | None, Field(description="Optional post-opposite-action snapshot from the same baseline.")
    ] = None,
    require_control_stable: bool = True,
    require_opposite_direction: bool = True,
    aligned: bool = True,
    offset_base: Annotated[int, Field(ge=0, le=0xFFFF)] = 0,
    limit: Annotated[int, Field(ge=1, le=2000)] = 100,
) -> dict[str, Any]:
    def scan() -> dict[str, Any]:
        paths = {
            "baseline": _input(baseline),
            "experiment": _input(experiment),
            "control": _input(control) if control else None,
            "opposite": _input(opposite) if opposite else None,
        }
        raw = {
            name: path.read_bytes() if path is not None else None
            for name, path in paths.items()
        }
        if len(raw["baseline"]) > 1024 * 1024:
            raise SessionError("memory scans are bounded to 1 MiB per snapshot")
        candidates = perception.scan_memory_changes(
            raw["baseline"],
            raw["experiment"],
            value_size=value_size,
            relation=relation,
            expected_delta=expected_delta,
            control=raw["control"],
            opposite=raw["opposite"],
            require_control_stable=require_control_stable,
            require_opposite_direction=require_opposite_direction,
            aligned=aligned,
            offset_base=offset_base,
        )
        return {
            "snapshots": {
                name: str(path) if path is not None else None for name, path in paths.items()
            },
            "bytes": len(raw["baseline"]),
            "value_size": value_size,
            "relation": relation,
            "total_candidates": len(candidates),
            "count": min(limit, len(candidates)),
            "truncated": len(candidates) > limit,
            "candidates": candidates[:limit],
            "epistemic_status": "ranked memory-change evidence; meanings remain unconfirmed",
        }

    return await asyncio.to_thread(scan)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    description=(
        "Rank equally sampled memory series against one numeric visual series, "
        "such as a changed-region track centroid.  Reports Pearson correlation "
        "of raw values and first differences, same/inverse direction, and "
        "simple fixed scales (1, 2, 4, 8, 16, 256) where they fit.  Correlation "
        "is evidence only; it does not establish semantics or causality."
    ),
)
def dosbox_correlate_memory(
    visual_series: Annotated[list[float], Field(min_length=3, max_length=10000)],
    memory_series: Annotated[
        dict[str, list[float]],
        Field(min_length=1, max_length=1000, description="Field label to paired numeric samples."),
    ],
    limit: Annotated[int, Field(ge=1, le=1000)] = 100,
) -> dict[str, Any]:
    ranked = perception.rank_correlations(visual_series, memory_series)
    return {
        "visual_samples": len(visual_series),
        "total_fields": len(ranked),
        "count": min(limit, len(ranked)),
        "truncated": len(ranked) > limit,
        "correlations": ranked[:limit],
        "epistemic_status": "numeric association only; no game semantics or causality inferred",
    }


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description=(
        "Read the guest's screen straight out of the emulator's video memory: "
        "the palette indices the guest wrote, at the instant of the read, with "
        "no window, no scaling and no palette round trip.  Prefer this over "
        "dosbox_capture_screen for anything quantitative -- photographing the "
        "window costs the guest real time and can catch a half-drawn frame.\n\n"
        "FINDING THE PAGE NEEDS A REFERENCE.  Give reference= a 64,000-byte "
        "frame of what is on screen and the page is confirmed byte for byte; "
        "without one this refuses, because a blind scan finds the most "
        "picture-like page in memory and on a real guest that is often a "
        "sprite bank or an offscreen buffer.  Once confirmed the location is "
        "cached and every later read is free.  allow_unconfirmed=true takes "
        "the guess anyway.\n\n"
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
    reference: Annotated[
        str | None,
        Field(description="Path to a 64,000-byte frame of what is on screen, to confirm the page."),
    ] = None,
    allow_unconfirmed: Annotated[
        bool, Field(description="Accept an unconfirmed blind scan; check the result yourself.")
    ] = False,
) -> dict[str, Any]:
    session = _session(pid)

    def read() -> dict[str, Any]:
        if relocate:
            session.framebuffer().found = None
        found = _locate_framebuffer(session, reference, allow_unconfirmed)
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
            target = _output(stem)
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


def _prepare_probes(
    session: Session, probes: list[MemoryProbe]
) -> list[tuple[str, int | None, int, int]]:
    prepared: list[tuple[str, int | None, int, int]] = []
    labels: set[str] = set()
    for probe in probes:
        segment = _segment_number(probe.segment)
        if segment is None:
            offset, symbol_size = session.profile.resolve(probe.offset)
            size = probe.size if probe.size != 1 else symbol_size
            address = f"DS:{offset:04X}"
        else:
            offset = int(probe.offset, 0) if isinstance(probe.offset, str) else int(probe.offset)
            size = probe.size
            address = f"{segment:04X}:{offset:04X}"
        if offset < 0 or offset + size > 0x10000:
            raise SessionError(f"memory probe {address}/{size} runs outside its 64 KiB segment")
        label = probe.label or (str(probe.offset) if segment is None else address)
        if label in labels:
            raise SessionError(f"memory probe label {label!r} is repeated")
        labels.add(label)
        prepared.append((label, segment, offset, size))
    return prepared


def _read_probes(
    session: Session, probes: list[tuple[str, int | None, int, int]]
) -> dict[str, int]:
    values: dict[str, int] = {}
    named = [probe for probe in probes if probe[1] is None]
    if named:
        snapshot = session.segment()
        for label, _, offset, size in named:
            values[label] = int.from_bytes(snapshot[offset : offset + size], "little")

    grouped: dict[int, list[tuple[str, int | None, int, int]]] = {}
    for probe in probes:
        if probe[1] is not None:
            grouped.setdefault(probe[1], []).append(probe)
    target = session.guest() if grouped else None
    for segment, fields in grouped.items():
        start = min(field[2] for field in fields)
        end = max(field[2] + field[3] for field in fields)
        snapshot = target.read(segment, start, end - start)
        for label, _, offset, size in fields:
            relative = offset - start
            values[label] = int.from_bytes(snapshot[relative : relative + size], "little")
    return values


def _prepare_range(
    session: Session, probe: MemoryRangeProbe | None
) -> tuple[int | None, int, int] | None:
    if probe is None:
        return None
    segment = _segment_number(probe.segment)
    if segment is None:
        offset, _ = session.profile.resolve(probe.offset)
    else:
        offset = int(probe.offset, 0) if isinstance(probe.offset, str) else int(probe.offset)
    if offset < 0 or offset + probe.length > 0x10000:
        address = "DS" if segment is None else f"{segment:04X}"
        raise SessionError(
            f"memory range {address}:{offset:04X}+{probe.length} runs outside its 64 KiB segment"
        )
    return segment, offset, probe.length


def _read_range(session: Session, prepared: tuple[int | None, int, int]) -> bytes:
    segment, offset, length = prepared
    if segment is None:
        return session.read(offset, length)
    return session.guest().read(segment, offset, length)


def _perform_action(session: Session, action: ExperimentAction, observation_seconds: float):
    if action.kind == "idle":
        if action.keys:
            raise SessionError("kind=idle cannot also contain keys")
        return {
            "kind": "idle",
            "transport": "none",
            "result": "control observation; no input was applied",
        }
    if action.kind == "keys":
        if not action.keys:
            raise SessionError("kind=keys requires at least one key")
        action_time = max(0, len(action.keys) - 1) * action.pace_seconds
        if action_time > observation_seconds:
            raise SessionError(
                f"paced key action takes {action_time:.3f}s, longer than the "
                f"{observation_seconds:.3f}s observation"
            )
        sent = session.press(action.keys, action.pace_seconds)
        return {
            "kind": "keys",
            "transport": "bios_keyboard_ring",
            "sent": sent,
            "limitation": (
                "BIOS queue events are not held/raw keyboard state and may not be "
                "consumed by action games that read IRQ1, port 0x60, or a key-state table."
            ),
        }
    if action.kind == "hold_keys":
        if not action.keys:
            raise SessionError("kind=hold_keys requires at least one key")
        if action.duration_seconds > observation_seconds:
            raise SessionError(
                f"action duration {action.duration_seconds:.3f}s is longer than the "
                f"{observation_seconds:.3f}s observation"
            )
        result = session.hold_keys(action.keys, action.duration_seconds)
        return {
            "kind": "hold_keys",
            "transport": "dosbox_x_keyboard_hook_v1",
            **result,
        }
    if action.duration_seconds > observation_seconds:
        raise SessionError(
            f"action duration {action.duration_seconds:.3f}s is longer than the "
            f"{observation_seconds:.3f}s observation"
        )
    if action.kind == "click":
        if action.x is None or action.y is None:
            raise SessionError("kind=click requires x and y")
        if not (
            0 <= action.x < session.profile.screen_width
            and 0 <= action.y < session.profile.screen_height
        ):
            raise SessionError(
                f"({action.x},{action.y}) is outside the {session.profile.screen_width}x"
                f"{session.profile.screen_height} guest screen"
            )
        writes = session.click(
            action.x, action.y, action.duration_seconds, action.buttons
        )
        return {
            "kind": "click",
            "transport": "profile_mouse_state",
            "x": action.x,
            "y": action.y,
            "buttons": action.buttons,
            "writes": writes,
        }
    result = session.hold(action.buttons, action.duration_seconds)
    return {
        "kind": "hold_buttons",
        "transport": "profile_mouse_state",
        "buttons": action.buttons,
        **result,
    }


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
    description=(
        "Run one bounded causal experiment in the live guest: capture a baseline "
        "indexed frame and requested memory, apply one existing valid input "
        "action, observe distinct frames and memory samples, then return exact "
        "visual differences, neutral region tracks, before/after fields, and an "
        "optional unknown-memory change scan.  kind=keys uses the BIOS ring; "
        "kind=hold_keys uses authentic make/break input through the versioned "
        "DOSBox-X companion hook.  "
        "Use action.kind=idle for a control experiment.  Give stem to retain "
        "raw indexed frames and before/after memory snapshots for later branch "
        "comparison with dosbox_compare_frames and dosbox_scan_changes."
    ),
)
async def dosbox_act_and_observe(
    action: ExperimentAction,
    duration_seconds: Annotated[float, Field(gt=0, le=30)] = 1.0,
    pid: int | None = None,
    interval_seconds: Annotated[
        float, Field(ge=0.01, le=1, description="Framebuffer/memory polling interval.")
    ] = 0.02,
    memory_fields: Annotated[
        list[MemoryProbe] | None,
        Field(max_length=64, description="Numeric fields sampled at each poll."),
    ] = None,
    memory_range: MemoryRangeProbe | None = None,
    reference: Annotated[
        str | None, Field(description="Current indexed frame path used to confirm the VRAM page.")
    ] = None,
    allow_unconfirmed: Annotated[
        bool, Field(description="Accept an unconfirmed framebuffer scan; verify it yourself.")
    ] = False,
    bridge_radius: Annotated[int, Field(ge=0, le=32)] = 1,
    min_changed_pixels: Annotated[int, Field(ge=1)] = 1,
    max_frames: Annotated[int, Field(ge=2, le=2000)] = 200,
    max_samples: Annotated[int, Field(ge=3, le=2000)] = 500,
    stem: Annotated[
        str | None, Field(description="Optional output stem for indexed/PNG frames and memory snapshots.")
    ] = None,
) -> dict[str, Any]:
    session = _session(pid)

    def experiment() -> dict[str, Any]:
        estimated_samples = int(duration_seconds / interval_seconds) + 2
        if estimated_samples > max_samples:
            raise SessionError(
                f"this interval can produce about {estimated_samples} samples; "
                f"increase interval_seconds or max_samples (currently {max_samples})"
            )
        _locate_framebuffer(session, reference, allow_unconfirmed)
        buffer = session.framebuffer()
        prepared_fields = _prepare_probes(session, memory_fields or [])
        prepared_range = _prepare_range(session, memory_range)
        if (
            memory_range is not None
            and memory_range.relation == "delta"
            and memory_range.expected_delta is None
        ):
            raise SessionError("memory_range relation='delta' requires expected_delta")

        baseline_page = buffer.read()
        baseline_values = _read_probes(session, prepared_fields)
        baseline_range = _read_range(session, prepared_range) if prepared_range else None
        frames: list[dict[str, Any]] = [
            {"t_ms": 0, "page": baseline_page, "memory": baseline_values}
        ]
        samples: list[dict[str, Any]] = [{"t_ms": 0, **baseline_values}]
        distinct_count = 1
        last_page = baseline_page
        dropped = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            action_future = executor.submit(
                _perform_action, session, action, duration_seconds
            )
            started = time.monotonic()
            next_poll = started + interval_seconds
            while True:
                now = time.monotonic()
                if now >= started + duration_seconds:
                    break
                if action_future.done() and action_future.exception() is not None:
                    action_future.result()
                if now < next_poll:
                    time.sleep(min(next_poll - now, interval_seconds))
                    continue
                elapsed_ms = int((now - started) * 1000)
                page = buffer.read()
                values = _read_probes(session, prepared_fields)
                samples.append({"t_ms": elapsed_ms, **values})
                if page != last_page:
                    distinct_count += 1
                    if len(frames) < max_frames:
                        frames.append({"t_ms": elapsed_ms, "page": page, "memory": values})
                    else:
                        dropped += 1
                    last_page = page
                next_poll += interval_seconds
            action_result = action_future.result()

        final_ms = int((time.monotonic() - started) * 1000)
        final_page = buffer.read()
        final_values = _read_probes(session, prepared_fields)
        samples.append({"t_ms": final_ms, **final_values})
        if final_page != last_page:
            distinct_count += 1
            if len(frames) < max_frames:
                frames.append({"t_ms": final_ms, "page": final_page, "memory": final_values})
            else:
                dropped += 1
        final_range = _read_range(session, prepared_range) if prepared_range else None

        comparisons = [
            perception.compare_frames(
                before["page"],
                after["page"],
                session.profile.screen_width,
                session.profile.screen_height,
                bridge_radius=bridge_radius,
                min_changed_pixels=min_changed_pixels,
            )
            for before, after in zip(frames, frames[1:])
        ]
        tracking = perception.track_regions(comparisons)
        for track in tracking["tracks"]:
            for observation in track["history"]:
                transition = int(observation["transition"])
                if transition < len(frames):
                    observation["t_ms"] = frames[transition]["t_ms"]
                    observation["memory"] = frames[transition]["memory"]
            if len(track["history"]) >= 3 and prepared_fields:
                memory_series = {
                    label: [entry["memory"][label] for entry in track["history"]]
                    for label, _, _, _ in prepared_fields
                }
                track["memory_correlations"] = {
                    "centroid_x": perception.rank_correlations(
                        [entry["centroid"][0] for entry in track["history"]], memory_series
                    ),
                    "centroid_y": perception.rank_correlations(
                        [entry["centroid"][1] for entry in track["history"]], memory_series
                    ),
                }

        baseline_final = perception.compare_frames(
            baseline_page,
            final_page,
            session.profile.screen_width,
            session.profile.screen_height,
            bridge_radius=bridge_radius,
            min_changed_pixels=min_changed_pixels,
        )
        fields = {
            label: {
                "address": (
                    f"DS:{offset:04X}" if segment is None else f"{segment:04X}:{offset:04X}"
                ),
                "size": size,
                "before": baseline_values[label],
                "after": final_values[label],
                "delta": final_values[label] - baseline_values[label],
            }
            for label, segment, offset, size in prepared_fields
        }
        memory_result: dict[str, Any] = {"fields": fields, "samples": samples}
        if memory_range is not None and baseline_range is not None and final_range is not None:
            candidates = perception.scan_memory_changes(
                baseline_range,
                final_range,
                value_size=memory_range.value_size,
                relation=memory_range.relation,
                expected_delta=memory_range.expected_delta,
                aligned=memory_range.aligned,
                offset_base=prepared_range[1],
            )
            memory_result["range"] = {
                "segment": (
                    "profile_data_segment"
                    if prepared_range[0] is None
                    else f"{prepared_range[0]:04X}"
                ),
                "offset": prepared_range[1],
                "length": prepared_range[2],
                "value_size": memory_range.value_size,
                "relation": memory_range.relation,
                "total_candidates": len(candidates),
                "truncated": len(candidates) > memory_range.candidate_limit,
                "candidates": candidates[: memory_range.candidate_limit],
            }

        files: list[str] = []
        if stem:
            target = _output(stem)
            palette = (
                vga.palette_bytes(session.profile.palette) if session.profile.palette else None
            )
            for index, entry in enumerate(frames):
                frame_stem = target.with_name(f"{target.name}_frame_{index:03d}")
                raw_path = frame_stem.with_suffix(".bin")
                png_path = frame_stem.with_suffix(".png")
                raw_path.write_bytes(entry["page"])
                png_path.write_bytes(
                    vga.png(
                        entry["page"],
                        palette,
                        session.profile.screen_width,
                        session.profile.screen_height,
                    )
                )
                files.extend((str(raw_path), str(png_path)))
            if baseline_range is not None and final_range is not None:
                before_path = target.with_name(f"{target.name}_memory_before.bin")
                after_path = target.with_name(f"{target.name}_memory_after.bin")
                before_path.write_bytes(baseline_range)
                after_path.write_bytes(final_range)
                files.extend((str(before_path), str(after_path)))

        return {
            "pid": session.pid,
            "action": action_result,
            "duration_ms": final_ms,
            "visual": {
                "distinct_frames": distinct_count,
                "kept_frames": len(frames),
                "dropped_frames": dropped,
                "frame_times_ms": [entry["t_ms"] for entry in frames],
                "baseline_to_final": baseline_final.as_dict(),
                "transitions": [comparison.as_dict() for comparison in comparisons],
                "tracking": tracking,
            },
            "memory": memory_result,
            "files": files,
            "epistemic_status": (
                "observed visual/memory evidence; track IDs and candidate offsets have no "
                "game semantics unless independently confirmed"
            ),
        }

    return await asyncio.to_thread(experiment)


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
    reference: Annotated[
        str | None,
        Field(description="Path to a 64,000-byte frame on screen now, to confirm the page."),
    ] = None,
    allow_unconfirmed: Annotated[bool, Field(description="Accept an unconfirmed blind scan.")] = False,
) -> dict[str, Any]:
    session = _session(pid)

    def watch() -> dict[str, Any]:
        _locate_framebuffer(session, reference, allow_unconfirmed)
        frames = session.framebuffer().watch(seconds, interval_seconds)
        kept = frames[:max_frames]
        result: dict[str, Any] = {
            "pid": session.pid,
            "distinct_frames": len(frames),
            "at_seconds": [when for when, _ in kept],
            "dropped": max(0, len(frames) - len(kept)),
        }
        if stem:
            target = _output(stem)
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
            target = _output(out)
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
