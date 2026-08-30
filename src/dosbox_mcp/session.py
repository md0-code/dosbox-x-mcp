"""Attached emulator sessions: find the guest's memory, then drive it.

A session binds one DOSBox-X process id to one profile -- or to no profile at
all, which is how a game nobody has described yet still gets driven.  On first use it scans
host memory for the profile's marker, validates the candidate segment, and
caches where it landed -- both the data segment and the guest's physical zero,
which is what the BIOS keyboard buffer is addressed from.  Every later call
revalidates the cached address in one small read and only rescans if the
emulator moved (it does not, in practice, but a stale write into a moved
allocation would be silent and terrible).

Guest physical zero is found by :mod:`dosbox_mcp.guest` from invariants every
DOS guest has, so it does not depend on the profile at all.  With it, any
segment is addressable; without a profile there is simply no *named* segment,
and the DS-relative calls say so rather than guessing.
"""
from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import guest as dosguest
from .platform import (
    Process,
    ProcessError,
    Window,
    find_windows,
    raw_keyboard_available,
    raw_keyboard_event,
)
from .profile import SEGMENT_SIZE, Profile, ProfileError
from .raw_keyboard import RawKey, resolve_raw_key
from .vga import Framebuffer

BDA_BASE = 0x0400
KEYBOARD_HEAD = 0x001A
KEYBOARD_TAIL = 0x001C
KEYBOARD_START = 0x001E
KEYBOARD_END = 0x003E
GUEST_SPAN = 0x120000  # conventional memory plus a margin for the HMA

# ASCII / scan-code pairs as INT 16h hands them out.  Letters, space, enter,
# escape and the arrows are the set proven against the game; the rest follow the
# same BIOS convention.
NAMED_KEYS: dict[str, tuple[int, int]] = {
    "space": (0x20, 0x39),
    "enter": (0x0D, 0x1C),
    "return": (0x0D, 0x1C),
    "esc": (0x1B, 0x01),
    "escape": (0x1B, 0x01),
    "tab": (0x09, 0x0F),
    "backspace": (0x08, 0x0E),
    "up": (0x00, 0x48),
    "down": (0x00, 0x50),
    "left": (0x00, 0x4B),
    "right": (0x00, 0x4D),
    "home": (0x00, 0x47),
    "end": (0x00, 0x4F),
    "pgup": (0x00, 0x49),
    "pgdn": (0x00, 0x51),
    "plus": (0x2B, 0x4E),
    "minus": (0x2D, 0x4A),
    **{f"f{n}": (0x00, 0x3A + n) for n in range(1, 11)},
}

# Scan codes follow the keyboard's rows, not the alphabet.
CHARACTER_SCANCODES: dict[str, int] = {
    **{character: 0x02 + index for index, character in enumerate("1234567890-=")},
    **{character: 0x10 + index for index, character in enumerate("qwertyuiop[]")},
    **{character: 0x1E + index for index, character in enumerate("asdfghjkl;'")},
    **{character: 0x2C + index for index, character in enumerate("zxcvbnm,./")},
    " ": 0x39,
    "\\": 0x2B,
    "`": 0x29,
}


class SessionError(RuntimeError):
    """The emulator could not be driven as asked."""


def key_codes(key: str) -> tuple[int, int]:
    """Resolve a key name or a single character to ``(ascii, scan)``."""
    lowered = key.lower()
    if lowered in NAMED_KEYS:
        return NAMED_KEYS[lowered]
    if len(key) == 1:
        scan = CHARACTER_SCANCODES.get(key.lower())
        if scan is not None:
            return ord(key), scan
    raise SessionError(
        f"unknown key {key!r}; use a single character or one of: "
        + ", ".join(sorted(NAMED_KEYS))
    )


@dataclass
class Condition:
    """A test against one little-endian field of the data segment."""

    offset: int
    size: int = 1
    equals: int | None = None
    not_equals: int | None = None
    at_least: int | None = None
    at_most: int | None = None
    any_of: tuple[int, ...] | None = None

    def value(self, segment: bytes) -> int:
        return int.from_bytes(segment[self.offset : self.offset + self.size], "little")

    def holds(self, segment: bytes) -> bool:
        seen = self.value(segment)
        if self.equals is not None and seen != self.equals:
            return False
        if self.not_equals is not None and seen == self.not_equals:
            return False
        if self.at_least is not None and seen < self.at_least:
            return False
        if self.at_most is not None and seen > self.at_most:
            return False
        if self.any_of is not None and seen not in self.any_of:
            return False
        return True

    def describe(self) -> str:
        parts = []
        for name, value in (
            ("==", self.equals),
            ("!=", self.not_equals),
            (">=", self.at_least),
            ("<=", self.at_most),
        ):
            if value is not None:
                parts.append(f"{name} {value}")
        if self.any_of:
            parts.append("in {" + ", ".join(str(v) for v in self.any_of) + "}")
        return f"DS:{self.offset:#06x}/{self.size} " + (" and ".join(parts) or "(no test)")


@dataclass
class Session:
    pid: int
    profile: Profile
    ds_base: int = 0
    ram_base: int | None = None
    scans: int = 0
    attached_at: float = field(default_factory=time.time)
    display: object | None = None
    config: object | None = None
    state_directory: Path | None = None
    traces: dict = field(default_factory=dict)
    _process: Process | None = None
    _framebuffer: Framebuffer | None = None
    _pressed_keys: dict[str, int] = field(default_factory=dict, repr=False)
    _bounded_keys: set[str] = field(default_factory=set, repr=False)
    _key_timers: dict[str, threading.Timer] = field(default_factory=dict, repr=False)
    _key_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # ── plumbing ───────────────────────────────────────────────────────────

    def process(self) -> Process:
        if self._process is None or not self._process.alive():
            self.close()
            self._process = Process(self.pid, write=True)
        return self._process

    def close(self) -> None:
        self.release_all_keys()
        if self._process is not None:
            self._process.close()
            self._process = None

    def window(self) -> Window | None:
        for candidate in find_windows(pid=self.pid):
            return candidate
        return None

    # ── locating the guest ─────────────────────────────────────────────────

    def locate(self, rescan: bool = False) -> None:
        """Ensure the guest is found, and the data segment still validates."""
        process = self.process()
        if self.profile.anonymous_profile:
            if self.ram_base is not None and not rescan:
                if dosguest.plausible(process, self.ram_base):
                    return
            self.ram_base = dosguest.find_one(process)
            self.scans += 1
            return
        if self.ds_base and not rescan:
            segment = process.read(self.ds_base, SEGMENT_SIZE)
            if segment is not None and self.profile.validates(segment):
                return
        self.ds_base = self._scan(process)
        self.ram_base = self._find_ram_base(process, self.ds_base)
        self.scans += 1

    # ── the guest, as the guest addresses itself ───────────────────────────

    def guest(self) -> dosguest.Guest:
        """Guest physical memory, for any segment rather than just the named one."""
        self.locate()
        if self.ram_base is None:
            raise SessionError(
                "the guest's physical memory could not be located, so only the "
                "profile's own data segment is reachable.  Set ds_segment in the "
                "profile to the game's runtime DS, or attach with no profile to "
                "search for the guest directly."
            )
        return dosguest.Guest(process=self.process(), ram_base=self.ram_base)

    def framebuffer(self) -> Framebuffer:
        if self._framebuffer is None:
            self._framebuffer = Framebuffer(
                self.process(), self.profile.screen_width, self.profile.screen_height
            )
        return self._framebuffer

    def _require_named_segment(self, what: str) -> None:
        if self.profile.anonymous_profile:
            raise SessionError(
                f"{what} needs a profile: with none attached there is no named data "
                f"segment.  Pass segment= to address the guest directly, or write a "
                f"profile (dosbox_find_guest shows what is loaded and where)."
            )

    def _scan(self, process: Process) -> int:
        candidates: list[int] = []
        for hit in process.find(self.profile.marker):
            base = hit - self.profile.marker_offset
            if any(base == seen for seen in candidates):
                continue
            segment = process.read(base, SEGMENT_SIZE)
            if segment is not None and self.profile.validates(segment):
                candidates.append(base)
        if not candidates:
            raise SessionError(
                f"no data segment matching profile {self.profile.name!r} in process "
                f"{self.pid}.  The game may not have reached the point where the "
                f"profile's marker exists yet, or this is not the right process."
            )
        if len(candidates) > 1:
            addresses = ", ".join(f"0x{value:X}" for value in candidates)
            raise SessionError(
                f"{len(candidates)} segments match profile {self.profile.name!r} "
                f"({addresses}); the profile's checks cannot tell them apart.  "
                f"Run one emulator at a time, or tighten the profile's checks."
            )
        return candidates[0]

    def _find_ram_base(self, process: Process, ds_base: int) -> int | None:
        """Guest physical address 0, needed to reach the BIOS data area."""
        tried: list[int] = []
        if self.profile.ds_segment is not None:
            tried.append(ds_base - (self.profile.ds_segment << 4))
        region = process.region_of(ds_base)
        if region:
            base, _, allocation = region
            tried.extend(value for value in (allocation, base) if value)
        for candidate in tried:
            if candidate <= ds_base and dosguest.plausible(process, candidate):
                return candidate
        # Last resort: find the BIOS keyboard buffer start/end words (0x001E,
        # 0x003E at absolute 0x480) below the segment.
        window_start = max(0, ds_base - GUEST_SPAN)
        span = process.read(window_start, ds_base - window_start)
        if span:
            position = span.rfind(b"\x1e\x00\x3e\x00")
            while position >= 0:
                candidate = window_start + position - 0x480
                if dosguest.plausible(process, candidate):
                    return candidate
                position = span.rfind(b"\x1e\x00\x3e\x00", 0, position)
        return None

    # ── memory ─────────────────────────────────────────────────────────────

    def segment(self) -> bytes:
        self._require_named_segment("reading the data segment")
        self.locate()
        data = self.process().read(self.ds_base, SEGMENT_SIZE)
        if data is None:
            raise SessionError("the data segment vanished mid-read; the emulator may have exited")
        return data

    def read(self, offset: int, length: int) -> bytes:
        self._require_named_segment("reading the data segment")
        self.locate()
        if offset + length > SEGMENT_SIZE:
            raise ProfileError(
                f"DS:{offset:#06x}+{length} runs past the end of the 64 KiB segment"
            )
        data = self.process().read(self.ds_base + offset, length)
        if data is None:
            raise SessionError(f"could not read DS:{offset:#06x}+{length}")
        return data

    def write(self, offset: int, data: bytes) -> None:
        self._require_named_segment("writing the data segment")
        self.locate()
        if offset + len(data) > SEGMENT_SIZE:
            raise ProfileError(
                f"DS:{offset:#06x}+{len(data)} runs past the end of the 64 KiB segment"
            )
        self.process().write(self.ds_base + offset, data)

    # ── input ──────────────────────────────────────────────────────────────

    def press(self, keys: list[str], pace: float) -> list[str]:
        """Append keys to the BIOS keyboard ring, one at a time, at ``pace``."""
        self.locate()
        if self.ram_base is None:
            raise SessionError(
                "the guest's BIOS data area could not be located, so keys cannot be "
                "injected.  Set ds_segment in the profile to the game's runtime DS "
                "(the segment its data lives at) and retry."
            )
        target = self.guest()
        report: list[str] = []
        for index, key in enumerate(keys):
            ascii_code, scan_code = key_codes(key)
            try:
                target.push_key(ascii_code, scan_code)
            except dosguest.GuestError as failure:
                raise SessionError(str(failure)) from None
            report.append(f"{key} (ascii {ascii_code:#04x}, scan {scan_code:#04x})")
            if index + 1 < len(keys):
                time.sleep(pace)
        return report

    def _raw_window(self) -> Window:
        windows = find_windows(pid=self.pid)
        for window in windows:
            if raw_keyboard_available(window.handle):
                return window
        if windows:
            raise SessionError(
                "this DOSBox-X build does not expose the MCP keyboard v1 hook; "
                "apply companion/dosbox-x-mcp-keyboard-v1.patch to DOSBox-X "
                "2026.07.02 and use that executable"
            )
        raise SessionError(f"process {self.pid} has no visible DOSBox-X window")

    @staticmethod
    def _resolve_raw_key(key: str) -> RawKey:
        try:
            return resolve_raw_key(key)
        except ValueError as failure:
            raise SessionError(str(failure)) from None

    def raw_keyboard_supported(self) -> bool:
        return any(
            raw_keyboard_available(window.handle) for window in find_windows(pid=self.pid)
        )

    def _cancel_key_timer(self, name: str) -> None:
        timer = self._key_timers.pop(name, None)
        if timer is not None:
            timer.cancel()

    def key_state(self, key: str, pressed: bool, lease_seconds: float = 5.0) -> dict[str, object]:
        """Set one authentic emulator key state, with an automatic release lease."""
        resolved = self._resolve_raw_key(key)
        window = self._raw_window()
        with self._key_lock:
            if resolved.name in self._bounded_keys:
                raise SessionError(
                    f"key {resolved.name} is owned by a bounded hold; wait for it to finish"
                )
            self._cancel_key_timer(resolved.name)
            was_pressed = resolved.name in self._pressed_keys
            if pressed:
                if not was_pressed:
                    raw_keyboard_event(window.handle, resolved.identifier, True)
                    self._pressed_keys[resolved.name] = resolved.identifier

                timer: threading.Timer

                def expire() -> None:
                    with self._key_lock:
                        if self._key_timers.get(resolved.name) is not timer:
                            return
                        self._key_timers.pop(resolved.name, None)
                        if resolved.name not in self._pressed_keys:
                            return
                        try:
                            raw_keyboard_event(window.handle, resolved.identifier, False)
                        except ProcessError:
                            pass
                        finally:
                            self._pressed_keys.pop(resolved.name, None)

                timer = threading.Timer(lease_seconds, expire)
                timer.daemon = True
                self._key_timers[resolved.name] = timer
                timer.start()
            else:
                # Send a break even when this session did not record the make;
                # this is a deliberate recovery path for an interrupted client.
                raw_keyboard_event(window.handle, resolved.identifier, False)
                self._pressed_keys.pop(resolved.name, None)
            return {
                "key": resolved.name,
                "identifier": resolved.identifier,
                "pressed": pressed,
                "changed": was_pressed != pressed,
                "lease_seconds": lease_seconds if pressed else None,
            }

    def hold_keys(self, keys: list[str], seconds: float) -> dict[str, object]:
        """Hold simultaneous emulator keys and release every make in reverse order."""
        resolved: list[RawKey] = []
        seen: set[str] = set()
        for key in keys:
            item = self._resolve_raw_key(key)
            if item.name not in seen:
                seen.add(item.name)
                resolved.append(item)
        if not resolved:
            raise SessionError("hold_keys requires at least one key")
        window = self._raw_window()
        pressed_now: list[RawKey] = []
        failures: list[str] = []
        with self._key_lock:
            overlap = [item.name for item in resolved if item.name in self._pressed_keys]
            if overlap:
                raise SessionError(
                    "cannot start a bounded hold while these keys already have active "
                    "leases: " + ", ".join(overlap)
                )
            try:
                for item in resolved:
                    raw_keyboard_event(window.handle, item.identifier, True)
                    self._pressed_keys[item.name] = item.identifier
                    self._bounded_keys.add(item.name)
                    pressed_now.append(item)
            except Exception:
                for item in reversed(pressed_now):
                    try:
                        raw_keyboard_event(window.handle, item.identifier, False)
                    except Exception:
                        pass
                    finally:
                        self._pressed_keys.pop(item.name, None)
                        self._bounded_keys.discard(item.name)
                raise
        try:
            time.sleep(seconds)
        finally:
            with self._key_lock:
                for item in reversed(pressed_now):
                    try:
                        raw_keyboard_event(window.handle, item.identifier, False)
                    except Exception as failure:
                        failures.append(f"{item.name}: {failure}")
                    finally:
                        self._pressed_keys.pop(item.name, None)
                        self._bounded_keys.discard(item.name)
        if failures:
            raise SessionError("key release failed: " + "; ".join(failures))
        return {
            "keys": [item.name for item in resolved],
            "identifiers": [item.identifier for item in resolved],
            "duration_seconds": seconds,
            "released": True,
        }

    def release_all_keys(self) -> None:
        """Best-effort break for every key owned by this session."""
        with self._key_lock:
            for timer in self._key_timers.values():
                timer.cancel()
            self._key_timers.clear()
            pressed = list(self._pressed_keys.items())
            self._pressed_keys.clear()
            self._bounded_keys.clear()
            if not pressed:
                return
            windows = find_windows(pid=self.pid)
            window = next(
                (item for item in windows if raw_keyboard_available(item.handle)), None
            )
            if window is None:
                return
            for _, identifier in reversed(pressed):
                try:
                    raw_keyboard_event(window.handle, identifier, False)
                except ProcessError:
                    pass

    def click(self, x: int, y: int, duration: float, button: int = 1) -> int:
        """Hold one controlled click in the game's post-INT-33 input state.

        The game samples its mouse state from the data segment whenever it feels
        like it, and its own interrupt handler keeps overwriting those words, so
        a single write usually loses the race.  Rewriting for a short while is
        what makes exactly one click land.
        """
        buttons_at, position_at = self._mouse_offsets()
        self.locate()
        process = self.process()
        coordinates = struct.pack("<HH", x, y)
        deadline = time.perf_counter() + duration
        writes = 0
        while time.perf_counter() < deadline:
            process.write(self.ds_base + position_at, coordinates)
            process.write(self.ds_base + buttons_at, struct.pack("<H", button))
            writes += 1
        process.write(self.ds_base + buttons_at, b"\x00\x00")
        return writes

    def hold(
        self,
        buttons: int,
        seconds: float,
        stop_when: Condition | None = None,
        poll_interval: float = 0.25,
    ) -> dict[str, object]:
        """Hold mouse buttons down by continuous write, optionally until a test passes."""
        buttons_at, _ = self._mouse_offsets()
        self.locate()
        process = self.process()
        pressed = struct.pack("<H", buttons)
        deadline = time.perf_counter() + seconds
        next_poll = time.perf_counter()
        writes = 0
        stopped = "timeout"
        observed: int | None = None
        try:
            while time.perf_counter() < deadline:
                process.write(self.ds_base + buttons_at, pressed)
                writes += 1
                if stop_when is not None and time.perf_counter() >= next_poll:
                    segment = process.read(self.ds_base, SEGMENT_SIZE)
                    if segment is not None:
                        observed = stop_when.value(segment)
                        if stop_when.holds(segment):
                            stopped = "condition"
                            break
                    next_poll = time.perf_counter() + poll_interval
        finally:
            process.write(self.ds_base + buttons_at, b"\x00\x00")
        return {"writes": writes, "stopped_on": stopped, "observed": observed}


    # ── sampling ───────────────────────────────────────────────────────────

    def sample(
        self,
        watch: list[tuple[str, int, int]],
        seconds: float,
        hz: float,
        pokes: list[tuple[int, int, int]] | None = None,
        settle: float = 0.0,
    ) -> list[dict[str, int]]:
        """Read a watch list repeatedly and return one row per instant.

        ``dosbox_read_memory`` answers what a value *is* and ``dosbox_wait_for``
        answers when it *becomes* something.  Neither answers how it moves,
        which is what most questions about a running program actually are.

        Each row is read from one whole-segment snapshot, so every field in a
        row is from the same instant -- fields read one at a time would drift
        against each other by exactly the amount that makes a trace unreadable.
        """
        self._require_named_segment("sampling")
        self.locate()
        process = self.process()
        rows: list[dict[str, int]] = []

        def take(elapsed: float) -> None:
            raw = process.read(self.ds_base, SEGMENT_SIZE)
            if raw is None:
                raise SessionError("the data segment vanished mid-sample")
            row = {"t_ms": int(elapsed * 1000)}
            for name, offset, size in watch:
                row[name] = int.from_bytes(raw[offset : offset + size], "little")
            rows.append(row)

        take(0.0)  # the entry instant, before anything is disturbed
        if settle:
            time.sleep(settle)
        for offset, value, size in pokes or ():
            process.write(self.ds_base + offset, int(value).to_bytes(size, "little"))

        interval = 1.0 / hz if hz > 0 else 0.05
        started = time.monotonic()
        while time.monotonic() - started < seconds:
            take(time.monotonic() - started)
            remaining = interval - ((time.monotonic() - started) % interval)
            time.sleep(min(remaining, interval))
        return rows

    def resolve_watch(self, names: list[str]) -> list[tuple[str, int, int]]:
        """Turn watch names -- symbols, watch-set names, or offsets -- into fields."""
        out: list[tuple[str, int, int]] = []
        for entry in names:
            expanded = self.profile.watch_sets.get(entry)
            if expanded is not None:
                out.extend(self.resolve_watch(list(expanded)))
                continue
            offset, size = self.profile.resolve(entry)
            label = entry if isinstance(entry, str) else f"{offset:#06x}"
            out.append((label, offset, max(1, size)))
        if not out:
            raise SessionError("give at least one symbol, watch set or offset to watch")
        return out

    def _mouse_offsets(self) -> tuple[int, int]:
        if self.profile.mouse_buttons is None or self.profile.mouse_position is None:
            raise SessionError(
                f"profile {self.profile.name!r} does not say where the game keeps its "
                "mouse state; add mouse.buttons and mouse.position to it"
            )
        return self.profile.mouse_buttons, self.profile.mouse_position

    def wait_for(
        self, condition: Condition, timeout: float, poll_interval: float
    ) -> dict[str, object]:
        started = time.perf_counter()
        last: int | None = None
        while time.perf_counter() - started < timeout:
            segment = self.segment()
            last = condition.value(segment)
            if condition.holds(segment):
                return {
                    "satisfied": True,
                    "value": last,
                    "waited_seconds": round(time.perf_counter() - started, 2),
                }
            time.sleep(poll_interval)
        return {
            "satisfied": False,
            "value": last,
            "waited_seconds": round(time.perf_counter() - started, 2),
        }


class Registry:
    """Every session this server has attached, by process id."""

    def __init__(self) -> None:
        self._sessions: dict[int, Session] = {}
        self._launched: set[int] = set()

    def attach(self, pid: int, profile: Profile) -> Session:
        session = self._sessions.get(pid)
        if session is None or session.profile.path != profile.path:
            if session is not None:
                session.close()
            session = Session(pid=pid, profile=profile)
            self._sessions[pid] = session
        return session

    def get(self, pid: int | None) -> Session:
        if pid is None:
            live = [s for s in self._sessions.values() if s.process().alive()]
            if len(live) == 1:
                return live[0]
            if not live:
                raise SessionError(
                    "no emulator is attached; call dosbox_launch or dosbox_attach first"
                )
            pids = ", ".join(str(s.pid) for s in live)
            raise SessionError(f"several emulators are attached ({pids}); pass pid")
        session = self._sessions.get(pid)
        if session is None:
            raise SessionError(f"process {pid} is not attached; call dosbox_attach first")
        return session

    def forget(self, pid: int) -> None:
        session = self._sessions.pop(pid, None)
        if session is not None:
            session.close()
        self._launched.discard(pid)

    def mark_launched(self, pid: int) -> None:
        self._launched.add(pid)

    def launched(self, pid: int) -> bool:
        return pid in self._launched

    def all(self) -> list[Session]:
        return list(self._sessions.values())
