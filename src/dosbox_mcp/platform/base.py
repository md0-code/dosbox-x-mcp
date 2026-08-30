"""What a platform backend has to provide, and what every backend shares.

The server reaches a running emulator through two very different channels: its
*memory*, which is an ordinary allocation in an ordinary host process, and its
*window*, which is not ordinary at all.  The first ports everywhere with only a
change of system call.  The second barely ports, and on Wayland does not port
at all.  Keeping them apart in the interface is what lets a platform support
one without the other, and lets a caller find that out by asking rather than by
failing.

Everything built on top of ``read()`` and ``regions()`` -- the marker scan, the
region lookup -- lives here, so a backend only has to supply the four calls
that are genuinely system-specific.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

#: Read in chunks rather than whole regions: an emulator with a large video
#: memory or a big EMS pool has allocations worth hundreds of megabytes, and
#: materialising one as a single bytes object is both slow and wasteful.
READ_CHUNK = 8 * 1024 * 1024


class ProcessError(RuntimeError):
    """The emulator process could not be opened, read or written."""


@dataclass(frozen=True)
class Window:
    """One of the emulator's windows.  ``handle`` is opaque to callers."""

    handle: int
    title: str
    pid: int


@dataclass(frozen=True)
class Capabilities:
    """What this platform can actually do, so a caller can ask instead of fail.

    ``memory`` is the one that matters: with it, every tool that reads or
    writes the guest works, which is nearly all of them.  The window entries
    only affect frame capture, and only when the frame cannot be read out of
    video memory instead.
    """

    platform: str
    memory: bool = False
    window_list: bool = False
    window_capture: bool = False
    window_resize: bool = False
    menu_commands: bool = False
    raw_keyboard: bool = False
    isolated_display: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "platform": self.platform,
            "memory": self.memory,
            "window_list": self.window_list,
            "window_capture": self.window_capture,
            "window_resize": self.window_resize,
            "menu_commands": self.menu_commands,
            "raw_keyboard": self.raw_keyboard,
            "isolated_display": self.isolated_display,
            "notes": list(self.notes),
        }


class BaseProcess:
    """The parts of a process backend that are the same everywhere.

    A backend supplies ``read``, ``write``, ``regions``, ``alive`` and
    ``close``; this supplies the searching built on them.
    """

    pid: int

    # ── to be provided by the backend ──────────────────────────────────────

    def read(self, address: int, size: int) -> bytes | None:  # pragma: no cover
        raise NotImplementedError

    def write(self, address: int, data: bytes) -> None:  # pragma: no cover
        raise NotImplementedError

    def regions(self) -> Iterator[tuple[int, int, int]]:  # pragma: no cover
        """``(base, size, allocation_base)`` for every readable committed span."""
        raise NotImplementedError

    def alive(self) -> bool:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def executable_path(self) -> Path:  # pragma: no cover
        raise NotImplementedError

    # ── shared ─────────────────────────────────────────────────────────────

    def __enter__(self) -> "BaseProcess":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def region_of(self, address: int) -> tuple[int, int, int] | None:
        """The region containing ``address``, or None.

        A backend with a cheaper answer than walking the list should override
        this; the generic version is correct but linear.
        """
        for base, size, allocation in self.regions():
            if base <= address < base + size:
                return base, size, allocation
        return None

    def find(self, marker: bytes, limit: int | None = None) -> list[int]:
        """Every host address at which ``marker`` occurs in readable memory.

        Reads in overlapping chunks so a match spanning a chunk boundary is
        still found, which matters because the markers that identify a data
        segment are typically 50-100 bytes long.
        """
        if not marker:
            return []
        hits: list[int] = []
        overlap = len(marker) - 1
        for base, size, _ in self.regions():
            position = 0
            tail = b""
            while position < size:
                count = min(READ_CHUNK, size - position)
                raw = self.read(base + position, count)
                if raw is None:
                    break
                haystack = tail + raw
                start = 0
                while True:
                    found = haystack.find(marker, start)
                    if found < 0:
                        break
                    hits.append(base + position - len(tail) + found)
                    if limit is not None and len(hits) >= limit:
                        return hits
                    start = found + 1
                tail = haystack[-overlap:] if overlap else b""
                position += count
        return hits


__all__ = [
    "READ_CHUNK",
    "BaseProcess",
    "Capabilities",
    "ProcessError",
    "Window",
]
