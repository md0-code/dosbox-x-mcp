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
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP, Image
from mcp.types import ToolAnnotations

from . import (
    bundle as experiment_bundle,
    emulator,
    evidence as evidence_store,
    guest as dosguest,
    perception,
    qualification,
    quality as experiment_quality,
    platform as host,
    trace as tracing,
    vga,
    world as world_model,
    writewatch,
)
from .profile import Profile, ProfileError, load_profile
from .companion import CompanionError
from .raw_keyboard import resolve_raw_key
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
    "dosbox_mcp",
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
        "dosbox_evidence_notebook persists observed, inferred, and confirmed "
        "claims separately from authoritative profiles.  "
        "dosbox_build_world_mosaic and dosbox_infer_movement_constraints add "
        "later-stage spatial evidence without asserting map semantics.  "
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


class BranchTrial(BaseModel):
    """One labelled action run from the same restored emulator state."""

    label: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    action: ExperimentAction


class TranslationStep(BaseModel):
    """Measured prior-frame pixel displacement into the next frame."""

    dx: int = Field(ge=-4096, le=4096)
    dy: int = Field(ge=-4096, le=4096)


class PositionSample(BaseModel):
    """One measured position in a consistent coordinate system."""

    x: float
    y: float


class MovementTrial(BaseModel):
    """A requested direction and measured positions from one controlled trial."""

    label: str = Field(min_length=1, max_length=64)
    direction: PositionSample
    positions: list[PositionSample] = Field(min_length=2, max_length=10000)


class TimelineEvent(BaseModel):
    """One action or observation scheduled against an explicit counter."""

    at: int = Field(ge=0, description="Requested counter value relative to timeline start.")
    kind: Literal["key_make", "key_break", "key_set", "capture_mark", "memory_snapshot", "presentation_snapshot", "trace_mark", "write_watch_start", "write_watch_stop"]
    key: str | None = Field(default=None, description="Physical key for key_make/key_break.")
    keys: list[str] = Field(default_factory=list, max_length=16, description="Simultaneous keys for key_set.")
    label: str | None = Field(default=None, min_length=1, max_length=128)
    physical_start: int | None = Field(default=None, ge=0, le=0x10FFFF, description="Bounded guest-physical start for memory_snapshot or presentation_snapshot.")
    length: int | None = Field(default=None, ge=1, le=65536, description="Bounded byte count for memory_snapshot or presentation_snapshot.")
    capture_bytes: int | None = Field(default=None, ge=1, le=1024 * 1024, description="Raw VGA backing bytes retained at a capture_mark or presentation_snapshot.")
    hit_limit: int | None = Field(default=None, ge=1, le=1000, description="Bounded raw hit limit for write_watch_start.")


class BundleArtifact(BaseModel):
    """One existing file copied and hashed into an experiment bundle."""

    artifact_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    source: str = Field(description="Existing input file to copy into the bundle.")
    bundle_path: str = Field(description="Portable relative path within the bundle.")
    media_type: str = Field(default="application/octet-stream", max_length=128)


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
    state_directory: Annotated[
        str | None,
        Field(
            description=(
                "Isolated DOSBox-X save-slot directory. Required by "
                "dosbox_try_actions so experiments never overwrite personal slots."
            )
        ),
    ] = None,
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

    state_path = Path(state_directory).resolve() if state_directory else None
    if state_path is not None:
        state_path.mkdir(parents=True, exist_ok=True)
    companion_token = secrets.token_urlsafe(32)
    process, display = emulator.launch(
        binary_path,
        config_path,
        working_directory,
        isolated=isolated,
        save_directory=state_path,
        companion_token=companion_token,
    )
    session = registry.attach(process.pid, resolved)
    session.display = display
    session.config = config_path
    session.state_directory = state_path
    session.companion_token = companion_token
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
        "companion_v2": session.companion(required=False) is not None,
        "state_directory": str(state_path) if state_path is not None else None,
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
    companion_token: Annotated[str | None, Field(description="Launch token for the optional protocol-v2 channel.")] = None,
) -> dict[str, Any]:
    resolved = _resolve_profile(profile)
    session = registry.attach(pid, resolved)
    session.companion_token = companion_token
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
        "companion_v2": session.companion(required=False) is not None,
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
                "state_directory": (
                    str(session.state_directory)
                    if session.state_directory is not None
                    else None
                ),
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
    result = host.capabilities().as_dict()
    result["companion_protocol"] = 2
    result["timeline_clocks"] = ["video_frame", "pic_us", "cpu_cycles", "wall_clock"]
    result["bundle_schema"] = experiment_bundle.BUNDLE_SCHEMA
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=False,
    ),
    description=(
        "List, record, revise, or delete entries in a project evidence notebook. "
        "Each entry is explicitly observed, inferred, or confirmed and retains "
        "its evidence/history. The notebook is a separate JSON file and never "
        "changes an authoritative game profile. Use concrete measurements as "
        "evidence; status and labels are caller assertions."
    ),
)
async def dosbox_evidence_notebook(
    operation: Literal["list", "upsert", "delete"] = "list",
    path: Annotated[
        str, Field(description="Notebook path, relative to DOSBOX_MCP_OUTPUT_DIR by default.")
    ] = "evidence.json",
    subject: Annotated[
        str | None, Field(description="Stable subject such as memory:DS:19A4 or visual:track:7.")
    ] = None,
    status: Literal["observed", "inferred", "confirmed"] | None = None,
    evidence: Annotated[list[str] | None, Field(max_length=32)] = None,
    label: Annotated[str | None, Field(max_length=128)] = None,
    confidence: Annotated[float | None, Field(ge=0, le=1)] = None,
    artifact_refs: Annotated[list[str] | None, Field(max_length=64)] = None,
    experiment_state: Literal["valid", "valid_with_ambiguity", "invalid", "incomplete"] | None = None,
) -> dict[str, Any]:
    target = _output(path)
    if operation == "list":
        found = await asyncio.to_thread(
            evidence_store.entries, target, status, subject
        )
        return {
            "path": str(target),
            "count": len(found),
            "entries": found,
            "note": "notebook claims are separate from authoritative profile facts",
        }
    if subject is None or not subject.strip():
        raise SessionError(f"operation={operation} requires subject")
    if operation == "delete":
        removed = await asyncio.to_thread(evidence_store.remove, target, subject)
        return {"path": str(target), "subject": subject, "removed": removed}
    if status is None:
        raise SessionError("operation=upsert requires status")
    if experiment_state in {"invalid", "incomplete"}:
        raise SessionError(
            f"experiment_state={experiment_state} cannot update the evidence notebook"
        )
    entry = await asyncio.to_thread(
        evidence_store.upsert,
        target,
        subject,
        status,
        evidence or [],
        label,
        confidence,
        artifact_refs,
    )
    return {"path": str(target), "entry": entry}


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
        "enemy, wall, or other game object.  Set detect_global_motion to search "
        "bounded whole-frame translations and report residual local changes."
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
    detect_global_motion: bool = False,
    max_translation_x: Annotated[int, Field(ge=0, le=128)] = 16,
    max_translation_y: Annotated[int, Field(ge=0, le=128)] = 16,
    translation_sample_stride: Annotated[int, Field(ge=1, le=16)] = 2,
    infer_screen_layout: bool = False,
    min_scroll_episodes: Annotated[int, Field(ge=1, le=99)] = 2,
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
        motions: list[dict[str, object]] = []
        for index, evidence in enumerate(comparisons):
            transition = {
                "from": str(loaded[index][0]),
                "to": str(loaded[index + 1][0]),
                **evidence.as_dict(),
            }
            if detect_global_motion or infer_screen_layout:
                motion = perception.detect_translation(
                    loaded[index][1],
                    loaded[index + 1][1],
                    width,
                    height,
                    max_dx=max_translation_x,
                    max_dy=max_translation_y,
                    sample_stride=translation_sample_stride,
                    bridge_radius=bridge_radius,
                    min_changed_pixels=min_changed_pixels,
                )
                motions.append(motion)
                transition["global_motion"] = motion
            transitions.append(transition)
        result: dict[str, Any] = {
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
        if infer_screen_layout:
            if len(loaded) < 3:
                raise SessionError("screen-layout inference needs at least three frames")
            result["screen_regions"] = perception.infer_screen_regions(
                [item[1] for item in loaded],
                [tuple(motion["best_translation"]) for motion in motions],
                width,
                height,
                min_episodes=min_scroll_episodes,
                min_region_pixels=min_changed_pixels,
            )
        return result

    return await asyncio.to_thread(compare)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    description=(
        "Stitch saved indexed frames into a raw world mosaic using supplied or "
        "automatically measured global translations. The first observed palette "
        "index is retained at each world coordinate; coverage and conflict maps "
        "show where later frames agreed or differed. Conflicts and ambiguous "
        "translations remain explicit evidence and are never silently repaired."
    ),
)
async def dosbox_build_world_mosaic(
    frames: Annotated[
        list[str], Field(min_length=2, max_length=100, description="Indexed frames in time order.")
    ],
    stem: Annotated[str, Field(description="Output stem for .bin, .png, and _coverage.bin.")],
    width: Annotated[int, Field(ge=1, le=4096)] = 320,
    height: Annotated[int, Field(ge=1, le=4096)] = 200,
    translations: Annotated[
        list[TranslationStep] | None,
        Field(description="Optional one measured prior-to-next translation per transition."),
    ] = None,
    max_translation_x: Annotated[int, Field(ge=0, le=128)] = 16,
    max_translation_y: Annotated[int, Field(ge=0, le=128)] = 16,
    translation_sample_stride: Annotated[int, Field(ge=1, le=16)] = 2,
    palette: Annotated[
        str | None, Field(description="Optional 768-byte six-bit DAC palette path.")
    ] = None,
    infer_tiles: bool = False,
    candidate_tile_sizes: Annotated[list[int], Field(min_length=1, max_length=16)] = [
        4,
        8,
        16,
        32,
    ],
    minimum_observed_tiles: Annotated[int, Field(ge=2, le=100000)] = 8,
    max_mosaic_pixels: Annotated[int, Field(ge=1, le=64 * 1024 * 1024)] = 64 * 1024 * 1024,
) -> dict[str, Any]:
    def stitch() -> dict[str, Any]:
        loaded = [_indexed_frame(path, width, height) for path in frames]
        motion_reports: list[dict[str, object]] = []
        camera_path: dict[str, object] | None = None
        if translations is None:
            camera_path = perception.estimate_camera_path(
                [item[1] for item in loaded],
                width,
                height,
                max_dx=max_translation_x,
                max_dy=max_translation_y,
                sample_stride=translation_sample_stride,
            )
            motion_reports = list(camera_path["pairwise"])
            steps = [tuple(step) for step in camera_path["accepted_translations"]]
            placement_methods = list(camera_path["placement_methods"])
            translation_source = "multi-frame temporal-consensus indexed-pixel camera path"
        else:
            if len(translations) != len(loaded) - 1:
                raise SessionError(
                    f"received {len(translations)} translations for {len(loaded)} frames; "
                    f"expected {len(loaded) - 1}"
                )
            steps = [(step.dx, step.dy) for step in translations]
            placement_methods = ["direct"] * len(loaded)
            translation_source = "caller-supplied measured translation"

        mosaic = world_model.build_mosaic(
            [item[1] for item in loaded],
            steps,
            width,
            height,
            max_mosaic_pixels=max_mosaic_pixels,
            placement_methods=placement_methods,
        )
        target = _output(stem)
        raw_path = target.with_suffix(".bin")
        png_path = target.with_suffix(".png")
        coverage_path = target.with_name(f"{target.name}_coverage.bin")
        palette_bytes = _input(palette).read_bytes() if palette else None
        if palette_bytes is not None and len(palette_bytes) != 768:
            raise SessionError(
                f"palette {_input(palette)} has {len(palette_bytes)} bytes; expected 768"
            )
        raw_path.write_bytes(mosaic.pixels)
        coverage_path.write_bytes(mosaic.coverage)
        png_path.write_bytes(
            vga.png(mosaic.pixels, palette_bytes, mosaic.width, mosaic.height)
        )
        result = {
            **mosaic.report(),
            "frames": [str(item[0]) for item in loaded],
            "translations": [list(step) for step in steps],
            "translation_source": translation_source,
            "translation_evidence": motion_reports,
            "camera_path": camera_path,
            "ambiguous_transitions": [
                index
                for index, motion in enumerate(motion_reports, start=1)
                if float(motion["confidence"]) < 0.5
            ],
            "files": {
                "indexed_mosaic": str(raw_path),
                "png": str(png_path),
                "coverage": str(coverage_path),
            },
        }
        if infer_tiles:
            structure = world_model.infer_tile_structure(
                mosaic.pixels,
                mosaic.coverage,
                mosaic.width,
                mosaic.height,
                candidate_sizes=candidate_tile_sizes,
                min_tiles=minimum_observed_tiles,
            )
            tiles_path = target.with_name(f"{target.name}_tiles.bin")
            tilemap_path = target.with_name(f"{target.name}_tilemap.json")
            tiles_path.write_bytes(structure.tile_bytes)
            tile_file_report = structure.report(include_tile_map=True)
            tilemap_path.write_text(
                json.dumps(tile_file_report, indent=2) + "\n", encoding="utf-8"
            )
            tile_report = structure.report(include_tile_map=False)
            result["tile_structure"] = tile_report
            result["files"]["tile_catalogue"] = str(tiles_path)
            result["files"]["tile_map"] = str(tilemap_path)
        return result

    return await asyncio.to_thread(stitch)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    description=(
        "Find repeated positions where a requested direction produced too little "
        "projected progress for several consecutive samples. Only candidates "
        "supported by the requested number of independently labelled trials are "
        "returned. These are movement constraints, not asserted walls or collisions."
    ),
)
def dosbox_infer_movement_constraints(
    trials: Annotated[list[MovementTrial], Field(min_length=1, max_length=100)],
    max_projected_progress: Annotated[float, Field(ge=0)] = 0.25,
    min_stalled_steps: Annotated[int, Field(ge=1, le=10000)] = 3,
    position_tolerance: Annotated[float, Field(ge=0)] = 2.0,
    min_trial_support: Annotated[int, Field(ge=1, le=100)] = 2,
) -> dict[str, object]:
    prepared = [
        {
            "label": trial.label,
            "direction": [trial.direction.x, trial.direction.y],
            "positions": [[sample.x, sample.y] for sample in trial.positions],
        }
        for trial in trials
    ]
    return world_model.infer_movement_constraints(
        prepared,
        max_projected_progress=max_projected_progress,
        min_stalled_steps=min_stalled_steps,
        position_tolerance=position_tolerance,
        min_trial_support=min_trial_support,
    )


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
        "comparison with dosbox_compare_frames and dosbox_scan_changes. Enable "
        "global motion to measure bounded baseline-to-final camera translation."
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
    detect_global_motion: bool = False,
    max_translation_x: Annotated[int, Field(ge=0, le=64)] = 16,
    max_translation_y: Annotated[int, Field(ge=0, le=64)] = 16,
    translation_sample_stride: Annotated[int, Field(ge=1, le=16)] = 2,
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
        global_motion = (
            perception.detect_translation(
                baseline_page,
                final_page,
                session.profile.screen_width,
                session.profile.screen_height,
                max_dx=max_translation_x,
                max_dy=max_translation_y,
                sample_stride=translation_sample_stride,
                bridge_radius=bridge_radius,
                min_changed_pixels=min_changed_pixels,
            )
            if detect_global_motion
            else None
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
                "global_motion": global_motion,
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
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=True,
    ),
    description=(
        "Run labelled actions from one identical DOSBox-X save-state baseline. "
        "The server saves isolated slot 100, restores it before every trial, "
        "collects the same exact visual/memory evidence as act_and_observe, and "
        "restores the baseline again before returning. This requires a Windows "
        "session launched with state_directory so personal save slots cannot be "
        "overwritten. Results are comparative evidence, not semantic labels."
    ),
)
async def dosbox_try_actions(
    trials: Annotated[list[BranchTrial], Field(min_length=1, max_length=32)],
    duration_seconds: Annotated[float, Field(gt=0, le=30)] = 1.0,
    pid: int | None = None,
    interval_seconds: Annotated[float, Field(ge=0.01, le=1)] = 0.02,
    memory_fields: Annotated[list[MemoryProbe] | None, Field(max_length=64)] = None,
    memory_range: MemoryRangeProbe | None = None,
    reference: str | None = None,
    allow_unconfirmed: bool = False,
    bridge_radius: Annotated[int, Field(ge=0, le=32)] = 1,
    min_changed_pixels: Annotated[int, Field(ge=1)] = 1,
    max_frames: Annotated[int, Field(ge=2, le=2000)] = 200,
    max_samples: Annotated[int, Field(ge=3, le=2000)] = 500,
    detect_global_motion: bool = False,
    max_translation_x: Annotated[int, Field(ge=0, le=64)] = 16,
    max_translation_y: Annotated[int, Field(ge=0, le=64)] = 16,
    translation_sample_stride: Annotated[int, Field(ge=1, le=16)] = 2,
    restore_settle_seconds: Annotated[float, Field(ge=0.05, le=5)] = 0.25,
    stem: Annotated[
        str | None, Field(description="Optional output stem; trial labels are appended.")
    ] = None,
) -> dict[str, Any]:
    session = _session(pid)
    if session.state_directory is None:
        raise SessionError(
            "controlled branches require an isolated save directory; launch this "
            "session with dosbox_launch(state_directory=...)"
        )
    host.require("menu_commands")
    labels = [trial.label for trial in trials]
    if len(set(labels)) != len(labels):
        raise SessionError("branch trial labels must be unique")

    session.release_all_keys()
    await asyncio.to_thread(
        _locate_framebuffer, session, reference, allow_unconfirmed
    )
    companion_method = getattr(session, "companion", None)
    companion_client = companion_method(required=False) if companion_method else None
    acknowledged_branching = bool(
        companion_client is not None
        and companion_client.capabilities.get("save_load_ack")
    )
    if acknowledged_branching:
        saved_ack = await asyncio.to_thread(companion_client.request, "save_state")
        state_candidate = session.state_directory / "100.sav"
        if not state_candidate.is_file() or state_candidate.stat().st_size == 0:
            raise SessionError(f"acknowledged save completed but state is missing: {state_candidate}")
        saved = {
            "state_file": str(state_candidate),
            "bytes": state_candidate.stat().st_size,
            "acknowledged": True,
            "events": saved_ack["events"],
        }
    else:
        saved = await asyncio.to_thread(
            emulator.save_branch_state, session.pid, session.state_directory
        )
    state_file = Path(saved["state_file"])
    branches: list[dict[str, Any]] = []
    restore_count = 0

    async def restore() -> dict[str, object]:
        nonlocal restore_count
        session.release_all_keys()
        if acknowledged_branching:
            result = await asyncio.to_thread(companion_client.request, "load_state")
            result = {**result, "state_file": str(state_file), "acknowledged": True}
        else:
            result = await asyncio.to_thread(
                emulator.load_branch_state,
                session.pid,
                state_file,
                restore_settle_seconds,
            )
        session.ds_base = 0
        session.ram_base = None
        await asyncio.to_thread(session.locate, True)
        restore_count += 1
        return result

    try:
        for trial in trials:
            restored = await restore()
            branch_stem = f"{stem}_{trial.label}" if stem else None
            observed = await dosbox_act_and_observe(
                action=trial.action,
                duration_seconds=duration_seconds,
                pid=session.pid,
                interval_seconds=interval_seconds,
                memory_fields=memory_fields,
                memory_range=memory_range,
                reference=None,
                allow_unconfirmed=allow_unconfirmed,
                bridge_radius=bridge_radius,
                min_changed_pixels=min_changed_pixels,
                max_frames=max_frames,
                max_samples=max_samples,
                detect_global_motion=detect_global_motion,
                max_translation_x=max_translation_x,
                max_translation_y=max_translation_y,
                translation_sample_stride=translation_sample_stride,
                stem=branch_stem,
            )
            branches.append(
                {"label": trial.label, "restore": restored, "evidence": observed}
            )
    finally:
        final_restore = await restore()

    return {
        "pid": session.pid,
        "baseline_state": saved,
        "trials": branches,
        "restore_count": restore_count,
        "baseline_restored": True,
        "acknowledged_branching": acknowledged_branching,
        "final_restore": final_restore,
        "epistemic_status": (
            "independent branch observations from one emulator state; action labels "
            "and measured differences do not establish game semantics"
        ),
    }


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=True,
    ),
    description=(
        "Probe candidate physical controls from one restored state, with repeated "
        "idle controls and authentic held-key trials. Ranks the strongest visual "
        "changed-region track response above idle motion by repeat support, "
        "direction consistency, and track confidence. Track IDs are local visual "
        "associations; this does not identify a player or assign control semantics."
    ),
)
async def dosbox_probe_controls(
    keys: Annotated[list[str], Field(min_length=1, max_length=12)],
    hold_seconds: Annotated[float, Field(gt=0, le=5)] = 0.35,
    observation_seconds: Annotated[float, Field(gt=0, le=10)] = 0.75,
    repeats: Annotated[int, Field(ge=1, le=5)] = 2,
    pid: int | None = None,
    interval_seconds: Annotated[float, Field(ge=0.01, le=0.5)] = 0.025,
    memory_fields: Annotated[list[MemoryProbe] | None, Field(max_length=32)] = None,
    reference: str | None = None,
    allow_unconfirmed: bool = False,
    stem: str | None = None,
    include_full_evidence: bool = False,
) -> dict[str, Any]:
    if hold_seconds > observation_seconds:
        raise SessionError("hold_seconds cannot exceed observation_seconds")
    try:
        canonical = [resolve_raw_key(key).name for key in keys]
    except ValueError as failure:
        raise SessionError(str(failure)) from None
    canonical = list(dict.fromkeys(canonical))
    trials: list[BranchTrial] = []
    metadata: dict[str, tuple[str, bool]] = {}
    for repeat in range(repeats):
        label = f"idle_r{repeat + 1}"
        trials.append(BranchTrial(label=label, action=ExperimentAction(kind="idle")))
        metadata[label] = ("idle", True)
    for key in canonical:
        for repeat in range(repeats):
            label = f"{key}_r{repeat + 1}"
            trials.append(
                BranchTrial(
                    label=label,
                    action=ExperimentAction(
                        kind="hold_keys", keys=[key], duration_seconds=hold_seconds
                    ),
                )
            )
            metadata[label] = (key, False)

    branched = await dosbox_try_actions(
        trials=trials,
        duration_seconds=observation_seconds,
        pid=pid,
        interval_seconds=interval_seconds,
        memory_fields=memory_fields,
        reference=reference,
        allow_unconfirmed=allow_unconfirmed,
        max_frames=200,
        max_samples=500,
        detect_global_motion=True,
        stem=stem,
    )
    control_trials: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for branch in branched["trials"]:
        label = branch["label"]
        action, control = metadata[label]
        evidence = branch["evidence"]
        tracks = evidence["visual"]["tracking"]["tracks"]
        motion = evidence["visual"].get("global_motion")
        trial_result = {"action": action, "control": control, "tracks": tracks}
        if motion is not None:
            trial_result["global_motion"] = motion
        trial_result["memory_deltas"] = {
            name: item["delta"] for name, item in evidence["memory"]["fields"].items()
        }
        trial_result["memory_samples"] = evidence["memory"].get("samples", [])
        trial_result["transition_fraction"] = evidence["visual"]["baseline_to_final"].get(
            "changed_fraction", 0.0
        )
        control_trials.append(trial_result)
        summaries.append(
            {
                "label": label,
                "action": action,
                "control": control,
                "track_count": len(tracks),
                "changed_pixels": evidence["visual"]["baseline_to_final"][
                    "changed_pixels"
                ],
                "global_translation": (
                    motion["best_translation"] if motion is not None else None
                ),
                "memory_deltas": {
                    name: item["delta"]
                    for name, item in evidence["memory"]["fields"].items()
                },
                "memory_activity": {
                    name: {
                        "minimum": min(values),
                        "maximum": max(values),
                        "distinct_values": len(set(values)),
                        "changed_samples": sum(value != values[0] for value in values),
                    }
                    for name in evidence["memory"]["fields"]
                    if (
                        values := [
                            sample[name]
                            for sample in evidence["memory"].get("samples", [])
                            if name in sample
                        ]
                    )
                },
            }
        )
    result: dict[str, Any] = {
        "pid": branched["pid"],
        "keys": canonical,
        "repeats": repeats,
        "baseline_state": branched["baseline_state"],
        "baseline_restored": branched["baseline_restored"],
        "ranking": perception.rank_control_responses(control_trials),
        "joint_ranking": perception.rank_joint_control_responses(
            control_trials, time_domain="wall_clock_ms"
        ),
        "trial_summaries": summaries,
        "epistemic_status": (
            "ranked repeatable visual response above idle; selected track IDs are "
            "trial-local evidence and are not semantic player identification"
        ),
    }
    if include_full_evidence:
        result["full_branch_evidence"] = branched["trials"]
    return result


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
        list[str], Field(max_length=24, description="Unlabelled values to record at each hit.")
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
    capture_schema: Annotated[dict[str, str] | None, Field(max_length=24, description="Optional stable label-to-register-or-memory-expression mapping for arguments and selected state.")] = None,
) -> dict[str, Any]:
    session = _session(pid)
    where = int(segment, 0) if isinstance(segment, str) else segment
    site = int(call_at, 0) if isinstance(call_at, str) else call_at
    cave = int(cave_at, 0) if isinstance(cave_at, str) else cave_at
    schema = dict(capture_schema or {})
    if any(not label.strip() or len(label) > 64 for label in schema):
        raise SessionError("trace schema labels must contain 1 to 64 characters")
    requested = list(dict.fromkeys([*captures, *schema.values()]))
    if not requested:
        raise SessionError("trace installation needs captures or a capture_schema")
    if len(requested) > 24:
        raise SessionError("trace installation supports at most 24 distinct capture expressions")

    def put() -> dict[str, Any]:
        installed = tracing.install(
            session.guest(),
            where,
            site,
            requested,
            cave,
            bytes.fromhex(expect_hex) if expect_hex else None,
            handle,
        )
        session.traces[installed.handle] = installed
        session.trace_schemas[installed.handle] = schema
        return {**installed.describe(), "capture_schema": schema}

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
    schema = session.trace_schemas.get(installed.handle, {})
    values = result.get("values", {})
    result["schema_values"] = {label: values[expression] for label, expression in schema.items()}
    result["capture_schema"] = schema
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
    session.trace_schemas.pop(installed.handle, None)
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


# ── post-perception reproducibility and emulator-time controls ────────────


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True),
    description=(
        "Run one ordered atomic timeline against an emulator-owned video-frame, "
        "PIC-microsecond, or CPU-cycle counter. Every event returns requested and "
        "actual counters with emulator acknowledgement. wall_clock is explicitly "
        "labelled compatibility mode. Held keys are released on timeout or disconnect."
    ),
)
async def dosbox_run_timeline(
    events: Annotated[list[TimelineEvent], Field(min_length=1, max_length=4096)],
    clock: Literal["video_frame", "pic_us", "cpu_cycles", "wall_clock"] = "video_frame",
    pid: int | None = None,
    timeout_seconds: Annotated[float, Field(gt=0, le=300)] = 30.0,
    expire_after_counter: Annotated[int | None, Field(ge=1)] = None,
) -> dict[str, Any]:
    session = _session(pid)
    client = session.companion(required=True)
    assert client is not None
    serialized: list[dict[str, Any]] = []
    capture_total = 0
    for order, event in enumerate(events):
        item = event.model_dump(exclude_none=True)
        item["order"] = order
        if event.kind in {"key_make", "key_break"}:
            if event.key is None:
                raise SessionError(f"{event.kind} requires key")
            item["key_id"] = resolve_raw_key(event.key).identifier
        elif event.kind == "key_set":
            if not event.keys:
                raise SessionError("key_set requires at least one key")
            item["key_ids"] = [resolve_raw_key(key).identifier for key in event.keys]
        elif event.kind == "memory_snapshot":
            if event.physical_start is None or event.length is None:
                raise SessionError("memory_snapshot requires physical_start and length")
        elif event.kind == "presentation_snapshot":
            if event.physical_start is None or event.length is None or event.capture_bytes is None:
                raise SessionError("presentation_snapshot requires physical_start, length, and capture_bytes")
            capture_total += event.length + event.capture_bytes
        elif event.kind == "write_watch_start":
            if event.physical_start is None or event.length is None or event.hit_limit is None:
                raise SessionError("write_watch_start requires physical_start, length, and hit_limit")
            capture_total += event.hit_limit * 256
        elif event.kind == "capture_mark" and event.capture_bytes is not None:
            capture_total += event.capture_bytes
        serialized.append(item)
    if capture_total > 1024 * 1024:
        raise SessionError("one timeline may retain at most 1 MiB of raw video observations")
    try:
        result = await asyncio.to_thread(client.run_timeline, clock, serialized, timeout_seconds, expire_after_counter)
    except Exception:
        session.release_all_keys()
        raise
    return {"pid": session.pid, "clock": clock, "timing_claim": "wall_clock_compatibility" if clock == "wall_clock" else "emulator_owned", **result}


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
    description="Negotiate the authenticated DOSBox-X companion protocol and return exact counters and capabilities.",
)
async def dosbox_companion_status(pid: int | None = None) -> dict[str, Any]:
    session = _session(pid)
    client = session.companion(required=True)
    assert client is not None
    result = await asyncio.to_thread(client.request, "status")
    return {"pid": session.pid, "protocol": 2, **result}


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=True),
    description=(
        "Run repeated idle and key trials from one acknowledged save state using "
        "VGA-frame timelines only. Captures raw video/DAC and selected memory at "
        "the same emulator events, applies joint channel ranking and validity "
        "gates, restores the baseline, and optionally emits a complete immutable bundle."
    ),
)
async def dosbox_qualify_controls(
    keys: Annotated[list[str], Field(min_length=1, max_length=12)],
    hold_frames: Annotated[int, Field(ge=1, le=10000)] = 25,
    observation_frames: Annotated[int, Field(ge=2, le=20000)] = 50,
    repeats: Annotated[int, Field(ge=2, le=10)] = 2,
    memory_fields: Annotated[list[MemoryProbe] | None, Field(max_length=32)] = None,
    reference: str | None = None,
    bundle_destination: str | None = None,
    game_files: Annotated[list[str] | None, Field(max_length=32)] = None,
    trace_files: Annotated[list[str] | None, Field(max_length=32, description="Optional bounded runtime trace reports to preserve in observations/traces.")] = None,
    progress: bool = False,
    pid: int | None = None,
) -> dict[str, Any]:
    def emit(message: str) -> None:
        if progress:
            print(f"  qualify: {message}", flush=True)

    if hold_frames < 2:
        raise SessionError("hold_frames must be at least 2 so make and held states are independently observable")
    if hold_frames >= observation_frames:
        raise SessionError("hold_frames must be smaller than observation_frames")
    session = _session(pid)
    if session.state_directory is None:
        raise SessionError("qualification requires an isolated state_directory")
    client = session.companion(required=True)
    assert client is not None
    if not all(client.capabilities.get(name) for name in ("timeline", "save_load_ack", "video_snapshot")):
        raise SessionError("qualification requires timeline, save/load acknowledgement, and video snapshot capabilities")
    canonical = list(dict.fromkeys(resolve_raw_key(key).name for key in keys))
    located = await asyncio.to_thread(_locate_framebuffer, session, reference, False)
    if not located.confirmed:
        raise SessionError("qualification refuses an unconfirmed framebuffer")
    emit("framebuffer confirmed")

    probes: list[dict[str, Any]] = []
    for probe in memory_fields or []:
        segment = _segment_number(probe.segment)
        if segment is None:
            offset, profile_size = session.profile.resolve(probe.offset)
            if session.profile.ds_segment is None:
                raise SessionError(f"memory field {probe.label or probe.offset!r} needs an explicit segment")
            segment = session.profile.ds_segment
            size = probe.size or profile_size
        else:
            offset = int(probe.offset, 0) if isinstance(probe.offset, str) else int(probe.offset)
            size = probe.size
        if offset < 0 or offset + size > 0x10000:
            raise SessionError(f"memory field {probe.label or probe.offset!r} crosses a segment boundary")
        probes.append({"label": probe.label or str(probe.offset), "physical": (segment << 4) + offset, "length": size})

    saved_ack = await asyncio.to_thread(client.request, "save_state")
    state_file = session.state_directory / "100.sav"
    if not state_file.is_file() or not state_file.stat().st_size:
        raise SessionError("acknowledged baseline state file is missing or empty")
    emit("baseline saved")

    trial_specs = [(f"idle_r{index + 1}", "idle", True) for index in range(repeats)]
    trial_specs += [(f"{key}_r{index + 1}", key, False) for key in canonical for index in range(repeats)]
    analyzed: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    memory_records: list[dict[str, Any]] = []
    timeline_records: list[dict[str, Any]] = []
    restore_events: list[dict[str, Any]] = []
    baseline_hashes: list[str] = []
    input_acknowledged = True
    rejected_attempts: list[dict[str, Any]] = []

    for label, action, control in trial_specs:
        emit(f"starting {label}")
        restored = await asyncio.to_thread(client.request, "load_state")
        restore_events.append({"trial": label, **restored})
        session.ds_base = 0
        session.ram_base = None
        await asyncio.to_thread(session.locate, True)
        probe_video = await asyncio.to_thread(client.request, "video_snapshot", {"after_frames": 0, "max_bytes": 1024 * 1024})
        probe_decoded = vga.decode_companion_snapshot(probe_video)
        width, height = int(probe_decoded["width"]), int(probe_decoded["height"])
        layout = str(probe_decoded["layout"])
        if probe_decoded["kind"] != "indexed":
            raise SessionError(f"control qualification requires indexed pixels; active layout is {layout}")
        if layout == "dosbox_chain4_sparse":
            capture_bytes = (int(probe_video["start_address"]) + width * height) * 4
        elif layout == "plane_interleaved":
            capture_bytes = (int(probe_video["start_address"]) + int(probe_video["plane_pitch"]) * height) * 4
        else:
            capture_bytes = int(probe_video["start_address"]) + int(probe_video["pitch"]) * height
        if capture_bytes * 3 > 1024 * 1024:
            raise SessionError(f"three atomic raw frame captures require {capture_bytes * 3} bytes, above the 1 MiB timeline limit")

        scheduled: list[dict[str, Any]] = []
        order_info: dict[int, tuple[str, str | None]] = {}

        def add(at: int, kind: str, mark: str, **values: Any) -> None:
            order = len(scheduled)
            field_label = values.pop("field_label", None)
            scheduled.append({"at": at, "order": order, "kind": kind, **values})
            order_info[order] = (mark, field_label)

        for at, mark in ((0, "baseline"), (hold_frames, "held"), (observation_frames, "final")):
            add(at, "capture_mark", mark, capture_bytes=capture_bytes)
            for probe in probes:
                add(at, "memory_snapshot", mark, physical_start=probe["physical"], length=probe["length"], field_label=probe["label"])
            if at == hold_frames and not control:
                add(at, "key_break", "break", key_id=resolve_raw_key(action).identifier)
        if not control:
            # Stable ordering makes the at=0 baseline pre-input evidence, while
            # the at=1 sample demonstrates that the guest observed the make.
            add(0, "key_make", "make", key_id=resolve_raw_key(action).identifier)
            for probe in probes:
                add(1, "memory_snapshot", "make", physical_start=probe["physical"], length=probe["length"], field_label=probe["label"])
        else:
            add(0, "trace_mark", "idle_start")
            for probe in probes:
                add(1, "memory_snapshot", "make", physical_start=probe["physical"], length=probe["length"], field_label=probe["label"])
        try:
            timeline = await asyncio.to_thread(client.run_timeline, "video_frame", scheduled, 60.0)
        except CompanionError as failure:
            rejected_attempts.append({"trial": label, "attempt": 1, "state": "incomplete", "reason": str(failure)})
            emit(f"retrying rejected {label}: {failure}")
            retry_restore = await asyncio.to_thread(client.request, "load_state")
            restore_events.append({"trial": label, "retry": True, **retry_restore})
            timeline = await asyncio.to_thread(client.run_timeline, "video_frame", scheduled, 60.0)
        emit(f"captured {label}")
        events = timeline["timeline_events"]
        if not control:
            key_events = [event for event in events if event["kind"] in {"key_make", "key_break"}]
            input_acknowledged &= len(key_events) == 2 and all(event.get("irq1_observed") and event["acknowledgement"] in {"key_applied", "key_released"} for event in key_events)
        input_acknowledged &= bool(timeline.get("released_all_keys"))
        frames: list[bytes] = []
        counters: list[int] = []
        samples: dict[str, dict[str, int]] = {mark: {"counter": at} for at, mark in ((0, "baseline"), (1, "make"), (hold_frames, "held"), (observation_frames, "final"))}
        sanitized_events: list[dict[str, Any]] = []
        for event in events:
            clean = {key: value for key, value in event.items() if key != "observation"}
            mark, field = order_info[int(event["order"])]
            observation = event.get("observation")
            if event["kind"] == "capture_mark" and isinstance(observation, dict):
                decoded = vga.decode_companion_snapshot(observation)
                frame = decoded["pixels"]
                assert isinstance(frame, bytes)
                frames.append(frame)
                counters.append(int(event["actual_counter"]))
                raw_records.append({"trial": label, "mark": mark, "counter": int(event["actual_counter"]), "frame": frame, "raw": decoded["raw"], "palette": decoded["palette"], "metadata": {key: value for key, value in observation.items() if key not in {"raw_hex", "dac_hex"}}})
                clean["artifact_ref"] = f"frame:{label}:{mark}:indexed"
            elif event["kind"] == "memory_snapshot" and isinstance(observation, dict) and field:
                raw = bytes.fromhex(str(observation["hex"]))
                samples[mark][field] = int.from_bytes(raw, "little")
                clean["artifact_ref"] = f"memory:{label}:{mark}:{field}"
                memory_records.append({"trial": label, "mark": mark, "field": field, "raw": raw})
            sanitized_events.append(clean)
        baseline_hashes.append(qualification.frame_hash(frames[0]))
        trial = qualification.analyze_trial(action, frames, counters, [samples[mark] for mark in ("baseline", "make", "held", "final")], width, height, control=control)
        trial["label"] = label
        analyzed.append(trial)
        timeline_records.append({"trial": label, "clock": "video_frame", "events": sanitized_events, "request_id": timeline["request_id"], "session_id": timeline["session_id"]})

    final_restore = await asyncio.to_thread(client.request, "load_state")
    final_video = await asyncio.to_thread(client.request, "video_snapshot", {"after_frames": 0, "max_bytes": 1024 * 1024})
    final_decoded = vga.decode_companion_snapshot(final_video)
    final_frame = final_decoded["pixels"]
    assert isinstance(final_frame, bytes)
    final_hash = qualification.frame_hash(final_frame)
    agreement = qualification.repeat_agreement(analyzed)
    ranking = perception.rank_joint_control_responses(analyzed, time_domain="video_frame")
    camera_confidences = [float((trial.get("global_motion") or {}).get("confidence", 0.0)) for trial in analyzed if not trial["control"]]
    gates = experiment_quality.evaluate(
        exact_framebuffer=True,
        baseline_hashes=baseline_hashes,
        restore_hash=final_hash,
        acknowledged_input=input_acknowledged,
        acknowledged_restore=all(["load_started", "load_completed", "post_restore_frame"] == [event["type"] for event in restore["events"]] for restore in restore_events) and ["load_started", "load_completed", "post_restore_frame"] == [event["type"] for event in final_restore["events"]],
        transition_fraction=max((float(trial["transition_fraction"]) for trial in analyzed if not trial["control"]), default=0.0),
        repeats=repeats,
        idle_repeats=repeats,
        camera_confidence=min(camera_confidences) if camera_confidences else None,
        cleanup_released=input_acknowledged,
        repeat_agreement=agreement,
        final_restore_ok=bool(baseline_hashes) and final_hash == baseline_hashes[0],
    )

    # Build neutral world evidence from the action trial with the greatest
    # accepted displacement.  This is evidence selection, not a semantic claim
    # that the selected action or memory fields have a particular meaning.
    action_trials = [trial for trial in analyzed if not trial["control"]]
    representative = max(
        action_trials,
        key=lambda trial: sum(abs(int(step[0])) + abs(int(step[1])) for step in trial["camera"]["accepted_translations"]),
    )
    representative_frames = [record["frame"] for record in raw_records if record["trial"] == representative["label"]]
    representative_steps = [tuple(step) for step in representative["camera"]["accepted_translations"]]
    mosaic = world_model.build_mosaic(
        representative_frames,
        representative_steps,
        width,
        height,
        placement_methods=representative["camera"]["placement_methods"],
    )
    screen_regions = perception.infer_screen_regions(
        representative_frames,
        representative_steps,
        width,
        height,
        min_episodes=1,
    )
    try:
        tile_structure = world_model.infer_tile_structure(
            mosaic.pixels,
            mosaic.coverage,
            mosaic.width,
            mosaic.height,
            candidate_sizes=[4, 8, 16, 32],
            min_tiles=8,
        )
        tile_report: dict[str, Any] = tile_structure.report()
    except world_model.WorldError as failure:
        tile_structure = None
        tile_report = {"status": "unresolved", "reason": str(failure), "epistemic_status": "no raw tile-grid candidate met the bounded evidence threshold"}

    probe_labels = {probe["label"] for probe in probes}
    x_field = next((name for name in ("player_x_copy", "player_x", "x") if name in probe_labels), None)
    y_field = next((name for name in ("player_y_copy", "player_y", "y") if name in probe_labels), None)
    direction_vectors = {"left": [-1, 0], "right": [1, 0], "up": [0, -1], "down": [0, 1]}
    movement_trials = []
    if x_field and y_field:
        for trial in action_trials:
            direction = direction_vectors.get(str(trial["action"]).lower())
            if direction is None:
                continue
            movement_trials.append({
                "label": trial["label"],
                "direction": direction,
                "positions": [[sample[x_field], sample[y_field]] for sample in trial["memory_samples"] if x_field in sample and y_field in sample],
            })
    constraints = (
        world_model.infer_movement_constraints(movement_trials, min_trial_support=2)
        if movement_trials
        else {"candidate_count": 0, "movement_constraints": [], "status": "unresolved", "reason": "no paired coordinate probes were supplied", "epistemic_status": "no movement semantics asserted"}
    )
    world_evidence = {
        "representative_trial": representative["label"],
        "camera": representative["camera"],
        "screen_regions": screen_regions,
        "mosaic": mosaic.report(),
        "tile_structure": tile_report,
        "constraints": constraints,
        "selection_rule": "greatest total accepted indexed-pixel displacement among non-idle restored trials",
        "epistemic_status": "camera, mosaic, raw tile, and requested-motion evidence only; no game semantics confirmed",
    }
    emit("world evidence derived")
    report: dict[str, Any] = {
        "pid": session.pid,
        "state": gates["state"],
        "clock": "video_frame",
        "hold_frames": hold_frames,
        "observation_frames": observation_frames,
        "repeats": repeats,
        "keys": canonical,
        "save": saved_ack,
        "final_restore": final_restore,
        "baseline_hashes": baseline_hashes,
        "final_restore_hash": final_hash,
        "trials": analyzed,
        "joint_ranking": ranking,
        "quality": gates,
        "timeline_count": len(timeline_records),
        "rejected_attempts": rejected_attempts,
    }

    if bundle_destination is not None:
        emit("writing bundle")
        destination = _output(bundle_destination)
        status = await asyncio.to_thread(client.request, "status")
        executable = session.process().executable_path()
        config_path = Path(session.config) if session.config is not None else None
        profile_path = session.profile.path
        game_sources = [_input(game_file) for game_file in (game_files or [])]
        metadata = {
            "experiment_type": "controlled_input_qualification",
            "emulator": status.get("build", {}),
            "protocol": 2,
            "mcp_version": "0.2.0",
            "executable_sha256": experiment_bundle.sha256(executable.read_bytes()),
            "effective_config_sha256": experiment_bundle.sha256(config_path.read_bytes()) if config_path and config_path.is_file() else None,
            "profile_sha256": experiment_bundle.sha256(profile_path.read_bytes()) if profile_path.is_file() else None,
            "framebuffer_confirmation": located.as_dict(),
            "frame_dimensions": {"width": width, "height": height},
            "quality_state": gates["state"],
            "post_perception_complete": bool(trace_files),
            "trace_artifact_count": len(trace_files or []),
            "game_files": [{"name": source.name, "sha256": experiment_bundle.sha256(source.read_bytes())} for source in game_sources],
        }
        with experiment_bundle.BundleWriter(destination, metadata) as writer:
            writer.add_json("emulator.build", "emulator/build.json", status.get("build", {}))
            if config_path and config_path.is_file():
                writer.add_file("emulator.config", "emulator/effective-config.conf", config_path, media_type="text/plain")
            writer.add_file("baseline.state", "baseline/state.sav", state_file)
            writer.add_bytes("baseline.frame", "baseline/frame.bin", next(record["frame"] for record in raw_records if record["mark"] == "baseline"))
            baseline_memory = b"".join(record["raw"] for record in memory_records if record["trial"] == analyzed[0]["label"] and record["mark"] == "baseline")
            writer.add_bytes("baseline.memory", "baseline/memory.bin", baseline_memory)
            writer.add_json("timeline", "timeline.json", timeline_records)
            for record in raw_records:
                prefix = f"observations/frames/{record['trial']}/{record['mark']}"
                writer.add_bytes(f"frame:{record['trial']}:{record['mark']}:raw", f"{prefix}.vga.bin", record["raw"])
                writer.add_bytes(f"frame:{record['trial']}:{record['mark']}:indexed", f"{prefix}.bin", record["frame"])
                writer.add_bytes(f"frame:{record['trial']}:{record['mark']}:dac", f"{prefix}.dac", record["palette"])
                writer.add_json(f"frame:{record['trial']}:{record['mark']}:metadata", f"{prefix}.json", record["metadata"])
            writer.add_json("observations.memory", "observations/memory/samples.json", [{"trial": trial["label"], "samples": trial["memory_samples"]} for trial in analyzed])
            for record in memory_records:
                writer.add_bytes(f"memory:{record['trial']}:{record['mark']}:{record['field']}", f"observations/memory/{record['trial']}/{record['mark']}-{record['field']}.bin", record["raw"])
            input_ids = [f"frame:{record['trial']}:{record['mark']}:indexed" for record in raw_records]
            input_hashes = [qualification.frame_hash(record["frame"]) for record in raw_records]
            derivation = {"tool": "dosbox_qualify_controls", "parameters": {"hold_frames": hold_frames, "observation_frames": observation_frames, "repeats": repeats}, "input_artifact_ids": input_ids, "input_hashes": input_hashes}
            writer.add_json("derived.comparisons", "derived/comparisons.json", [{"trial": trial["label"], "comparisons": trial["comparisons"]} for trial in analyzed], derivation=derivation)
            writer.add_json("derived.camera", "derived/camera.json", [{"trial": trial["label"], "camera": trial["camera"]} for trial in analyzed], derivation=derivation)
            writer.add_json("derived.world", "derived/world.json", world_evidence, derivation=derivation)
            writer.add_bytes("derived.mosaic.pixels", "derived/mosaic.bin", mosaic.pixels, derivation=derivation)
            writer.add_bytes("derived.mosaic.coverage", "derived/mosaic-coverage.bin", mosaic.coverage, derivation=derivation)
            if tile_structure is not None:
                writer.add_bytes("derived.tiles.raw", "derived/tiles.bin", tile_structure.tile_bytes, derivation=derivation)
                writer.add_json("derived.tiles.map", "derived/tilemap.json", tile_report, derivation=derivation)
            writer.add_json("derived.constraints", "derived/constraints.json", constraints, derivation=derivation)
            trace_ids = []
            for index, trace_path in enumerate(trace_files or []):
                source = _input(trace_path)
                artifact_id = f"observations.trace{index}"
                writer.add_file(artifact_id, f"observations/traces/{index}-{source.name}", source, media_type="application/json")
                trace_ids.append(artifact_id)
            writer.add_json("evidence", "evidence.json", {"joint_ranking": ranking, "quality": gates, "rejected_attempts": rejected_attempts, "world_evidence": ["derived.world", "derived.constraints"], "trace_evidence": trace_ids, "epistemic_status": "observed and inferred channels only; no control semantics are confirmed"}, derivation=derivation)
            for index, source in enumerate(game_sources):
                writer.add_file(f"game.file{index}", f"emulator/game/{source.name}", source)
            manifest = writer.finalize()
        verification = experiment_bundle.verify(destination)
        report["bundle"] = {"path": str(destination), "bundle_id": manifest["bundle_id"], "verification": verification}
        emit("bundle finalized and verified")
    return report


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False),
    description="Create an immutable, atomically finalized experiment bundle from explicitly named artifacts. Every file is SHA-256 hashed and machine paths are deterministically redacted.",
)
async def dosbox_create_experiment_bundle(
    destination: str,
    artifacts: Annotated[list[BundleArtifact], Field(min_length=1, max_length=2000)],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    target = _output(destination)

    def create() -> dict[str, Any]:
        with experiment_bundle.BundleWriter(target, metadata) as writer:
            for artifact in artifacts:
                writer.add_file(artifact.artifact_id, artifact.bundle_path, _input(artifact.source), media_type=artifact.media_type)
            manifest = writer.finalize()
        return {"bundle": str(target), "bundle_id": manifest["bundle_id"], "artifact_count": len(manifest["artifacts"]), "verified": experiment_bundle.verify(target)}

    return await asyncio.to_thread(create)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False),
    description="Verify every declared size and SHA-256 hash in a finalized experiment bundle offline, rejecting missing, altered, undeclared, unsafe, or partial content.",
)
def dosbox_verify_experiment_bundle(path: str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_absolute():
        target = DEFAULT_OUTPUT_DIR / target
    return experiment_bundle.verify(target)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False),
    description="Recompute qualification comparisons, camera, mosaic, raw tiles, and movement constraints from stored raw artifacts only and require byte-identical derived reports.",
)
def dosbox_reproduce_experiment_bundle(path: str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_absolute():
        target = DEFAULT_OUTPUT_DIR / target
    return experiment_bundle.reproduce(target)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False),
    description="Evaluate strict or exploratory validity gates. Invalid and incomplete experiments are explicitly barred from automatic evidence-notebook updates.",
)
def dosbox_evaluate_experiment(
    exact_framebuffer: bool | None,
    baseline_hashes: list[str],
    restore_hash: str | None,
    acknowledged_input: bool | None,
    acknowledged_restore: bool | None,
    transition_fraction: Annotated[float | None, Field(ge=0, le=1)],
    repeats: Annotated[int, Field(ge=0)],
    idle_repeats: Annotated[int, Field(ge=0)],
    camera_confidence: Annotated[float | None, Field(ge=0, le=1)],
    cleanup_released: bool | None,
    strict: bool = True,
) -> dict[str, Any]:
    return experiment_quality.evaluate(exact_framebuffer=exact_framebuffer, baseline_hashes=baseline_hashes, restore_hash=restore_hash, acknowledged_input=acknowledged_input, acknowledged_restore=acknowledged_restore, transition_fraction=transition_fraction, repeats=repeats, idle_repeats=idle_repeats, camera_confidence=camera_confidence, cleanup_released=cleanup_released, strict=strict)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False),
    description="Estimate a temporally consistent cumulative camera path from exact indexed frames, retaining top candidates, uncertainty, outliers, and exact-content loop-closure drift.",
)
async def dosbox_estimate_camera_path(
    frames: Annotated[list[str], Field(min_length=2, max_length=1000)],
    width: Annotated[int, Field(ge=1, le=4096)] = 320,
    height: Annotated[int, Field(ge=1, le=4096)] = 200,
    max_translation_x: Annotated[int, Field(ge=0, le=128)] = 16,
    max_translation_y: Annotated[int, Field(ge=0, le=128)] = 16,
    consensus_window: Annotated[int, Field(ge=1, le=15)] = 3,
    minimum_confidence: Annotated[float, Field(ge=0, le=1)] = 0.25,
    screen_fixed_mask: Annotated[str | None, Field(description="Optional width*height byte mask; non-zero pixels are excluded from translation scoring.")] = None,
) -> dict[str, Any]:
    loaded = [_indexed_frame(path, width, height)[1] for path in frames]
    mask = _input(screen_fixed_mask).read_bytes() if screen_fixed_mask else None
    return await asyncio.to_thread(perception.estimate_camera_path, loaded, width, height, max_dx=max_translation_x, max_dy=max_translation_y, consensus_window=consensus_window, minimum_confidence=minimum_confidence, screen_fixed_mask=mask)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=True),
    description="Observe writes to one bounded guest-physical range, returning writer CS:IP, before/after values, registers, emulator counters, hit counts, and timeline links. Requires protocol-v2 write_watch capability.",
)
async def dosbox_observe_writes(
    physical_start: Annotated[int, Field(ge=0, le=0x10FFFF)],
    length: Annotated[int, Field(ge=1, le=65536)],
    duration_counter: Annotated[int, Field(ge=1)],
    clock: Literal["video_frame", "pic_us", "cpu_cycles"] = "video_frame",
    hit_limit: Annotated[int, Field(ge=1, le=1_000_000)] = 10000,
    output_limit: Annotated[int, Field(ge=1, le=100000)] = 1000,
    registers: Annotated[list[Literal["ax", "bx", "cx", "dx", "si", "di", "bp", "sp", "ds", "es", "ss"]] | None, Field(max_length=11)] = None,
    pid: int | None = None,
) -> dict[str, Any]:
    session = _session(pid)
    client = session.companion(required=True)
    assert client is not None
    raw = await asyncio.to_thread(client.request, "observe_writes", {"physical_start": physical_start, "length": length, "duration_counter": duration_counter, "clock": clock, "hit_limit": hit_limit, "registers": registers or []})
    aggregated = writewatch.aggregate(raw.get("write_events", []), physical_start=physical_start, length=length, hit_limit=hit_limit, output_limit=output_limit)
    return {"pid": session.pid, "clock": clock, **aggregated, "request_id": raw["request_id"], "session_id": raw["session_id"]}


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True),
    description=(
        "Capture DOSBox-X's active raw VGA backing memory and DAC at one acknowledged "
        "emulated frame. Writes metadata, untouched raw memory, palette, and split "
        "planes where applicable; unsupported layouts are labelled and never guessed."
    ),
)
async def dosbox_capture_video_snapshot(
    output_directory: str,
    after_frames: Annotated[int, Field(ge=0, le=100000)] = 1,
    max_bytes: Annotated[int, Field(ge=1, le=1024 * 1024)] = 1024 * 1024,
    pid: int | None = None,
) -> dict[str, Any]:
    session = _session(pid)
    client = session.companion(required=True)
    assert client is not None
    snapshot = await asyncio.to_thread(
        client.request,
        "video_snapshot",
        {"after_frames": after_frames, "max_bytes": max_bytes},
    )
    try:
        raw = bytes.fromhex(snapshot.pop("raw_hex"))
        palette = bytes.fromhex(snapshot.pop("dac_hex"))
    except (KeyError, ValueError) as failure:
        raise SessionError(f"companion returned a malformed video snapshot: {failure}") from None
    directory = _output(str(Path(output_directory) / "metadata.json")).parent
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / "vga-memory.bin"
    palette_path = directory / "dac.bin"
    raw_path.write_bytes(raw)
    palette_path.write_bytes(palette)
    artifacts: list[dict[str, Any]] = [
        {"artifact_id": "video.raw", "path": str(raw_path), "bytes": len(raw), "sha256": experiment_bundle.sha256(raw)},
        {"artifact_id": "video.dac", "path": str(palette_path), "bytes": len(palette), "sha256": experiment_bundle.sha256(palette)},
    ]
    if snapshot.get("layout") in {"plane_interleaved", "text_plane_interleaved"}:
        for plane in range(4):
            data = raw[plane::4]
            plane_path = directory / f"plane-{plane}.bin"
            plane_path.write_bytes(data)
            artifacts.append({"artifact_id": f"video.plane{plane}", "path": str(plane_path), "bytes": len(data), "sha256": experiment_bundle.sha256(data)})
    metadata = {**snapshot, "pid": session.pid, "artifacts": artifacts}
    metadata_path = directory / "metadata.json"
    metadata_bytes = experiment_bundle.canonical_json(metadata)
    metadata_path.write_bytes(metadata_bytes)
    artifacts.append({"artifact_id": "video.metadata", "path": str(metadata_path), "bytes": len(metadata_bytes), "sha256": experiment_bundle.sha256(metadata_bytes)})
    return {**metadata, "artifacts": artifacts}


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False),
    description=(
        "Decode an exact raw Mode X, planar EGA/VGA, text, or packed VESA snapshot. "
        "Raw planes remain untouched; output is accepted as confirmed only when it "
        "matches the supplied exact reference. Unsupported layouts are refused."
    ),
)
async def dosbox_decode_video_snapshot(
    mode: Literal["mode_x", "planar_4bpp", "text", "vesa_linear"],
    sources: Annotated[list[str], Field(min_length=1, max_length=4)],
    output: str,
    width: Annotated[int, Field(ge=1, le=8192)],
    height: Annotated[int, Field(ge=1, le=8192)],
    pitch: Annotated[int | None, Field(ge=1)] = None,
    start_address: Annotated[int, Field(ge=0)] = 0,
    bytes_per_pixel: Annotated[int, Field(ge=1, le=4)] = 1,
    reference: str | None = None,
    allow_unconfirmed: bool = False,
) -> dict[str, Any]:
    paths = [_input(source) for source in sources]
    raw = [path.read_bytes() for path in paths]
    if mode in {"mode_x", "planar_4bpp"}:
        if len(raw) != 4:
            raise SessionError(f"{mode} requires exactly four raw plane files")
        planes = tuple(raw)
        decoded = (
            vga.decode_mode_x(planes, width, height, start_address=start_address, pitch=pitch)
            if mode == "mode_x"
            else vga.decode_planar_4bpp(planes, width, height, start_address=start_address, pitch=pitch)
        )
    elif mode == "vesa_linear":
        if len(raw) != 1 or pitch is None:
            raise SessionError("vesa_linear requires one source and explicit pitch")
        decoded = vga.decode_vesa_linear(raw[0], width, height, pitch=pitch, bytes_per_pixel=bytes_per_pixel, start_address=start_address)
    else:
        if len(raw) != 1:
            raise SessionError("text mode requires one interleaved character/attribute source")
        cells = vga.decode_text_cells(raw[0], width, height, start_address=start_address, pitch=pitch)
        decoded = experiment_bundle.canonical_json(cells)
    confirmed = False
    if reference is not None:
        confirmed = decoded == _input(reference).read_bytes()
        if not confirmed:
            raise SessionError("decoded video output does not exactly match the supplied reference")
    elif not allow_unconfirmed:
        raise SessionError("an exact reference is required; pass allow_unconfirmed only for exploratory decoding")
    target = _output(output)
    target.write_bytes(decoded)
    return {
        "mode": mode,
        "dimensions": [width, height],
        "pitch": pitch,
        "start_address": start_address,
        "bytes_per_pixel": bytes_per_pixel,
        "raw_sources": [{"path": str(path), "bytes": len(data), "sha256": experiment_bundle.sha256(data)} for path, data in zip(paths, raw)],
        "decoded": {"path": str(target), "bytes": len(decoded), "sha256": experiment_bundle.sha256(decoded)},
        "confirmed": confirmed,
        "note": "raw planes/pages are preserved at their source paths",
    }


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
