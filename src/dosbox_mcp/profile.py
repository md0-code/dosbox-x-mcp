"""Game profiles: how to recognise one game's data segment inside DOSBox-X.

A profile is a small JSON file.  It carries the byte pattern that identifies
the segment, the checks that prove a candidate really is that segment, the
input offsets the game reads its mouse from, the screen geometry and palette,
and any named offsets worth exposing to a caller by name instead of by number.
It may also name watch sets for sampling, describe the guest's video mode, and
record where the code segment has room for a trace cave.

Nothing else in this server knows anything game-specific.  A profile is also
**optional**: :meth:`Profile.anonymous` stands in for one so a guest that
nobody has profiled yet can still be found, searched and typed at, which is how
the first profile for a new game gets written.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SEGMENT_SIZE = 0x10000


class ProfileError(ValueError):
    """A profile is missing, malformed, or contradicts the running process."""


@dataclass(frozen=True)
class Symbol:
    name: str
    offset: int
    size: int = 1
    description: str = ""


@dataclass(frozen=True)
class Profile:
    name: str
    path: Path
    description: str
    marker: bytes
    marker_offset: int
    checks: tuple[dict[str, Any], ...]
    window_title: str
    ds_segment: int | None
    mouse_buttons: int | None
    mouse_position: int | None
    screen_width: int
    screen_height: int
    palette: tuple[tuple[int, int, int], ...]
    symbols: dict[str, Symbol] = field(default_factory=dict)
    watch_sets: dict[str, tuple[str, ...]] = field(default_factory=dict)
    video_mode: str = "13h"
    code_segment_offset: int | None = None
    caves: tuple[dict[str, Any], ...] = ()

    # ── an absent profile ──────────────────────────────────────────────────

    @classmethod
    def anonymous(cls) -> "Profile":
        """A profile for a game nobody has described yet.

        It has no marker, so nothing identifies a data segment and every
        DS-relative call refuses with an explanation.  Everything that works
        off guest physical memory -- finding the guest, reading any segment,
        searching, typing, the framebuffer -- works exactly as usual.
        """
        return cls(
            name="(none)",
            path=Path("(none)"),
            description=(
                "No profile.  The guest is addressed by segment; there is no "
                "named data segment and no symbols."
            ),
            marker=b"",
            marker_offset=0,
            checks=(),
            window_title="DOSBox-X",
            ds_segment=None,
            mouse_buttons=None,
            mouse_position=None,
            screen_width=320,
            screen_height=200,
            palette=(),
        )

    @property
    def anonymous_profile(self) -> bool:
        return not self.marker

    # ── segment identification ─────────────────────────────────────────────

    def validates(self, segment: bytes) -> bool:
        """Does this 64 KiB candidate satisfy every check in the profile?"""
        if len(segment) != SEGMENT_SIZE or not self.marker:
            return False
        if segment[self.marker_offset : self.marker_offset + len(self.marker)] != self.marker:
            return False
        for check in self.checks:
            if not _run_check(segment, check):
                return False
        return True

    # ── offsets ────────────────────────────────────────────────────────────

    def resolve(self, offset: int | str) -> tuple[int, int]:
        """Turn a number, a hex string, or a symbol name into ``(offset, size)``."""
        if isinstance(offset, int):
            return _checked(offset), 1
        text = offset.strip()
        symbol = self.symbols.get(text)
        if symbol is not None:
            return symbol.offset, symbol.size
        try:
            return _checked(int(text, 0)), 1
        except ValueError:
            known = ", ".join(sorted(self.symbols)) or "(none)"
            raise ProfileError(
                f"{offset!r} is neither a number nor a symbol in profile "
                f"{self.name!r}.  Known symbols: {known}"
            ) from None


def _checked(offset: int) -> int:
    if not 0 <= offset < SEGMENT_SIZE:
        raise ProfileError(f"offset {offset:#x} is outside the 64 KiB data segment")
    return offset


def _num(value: int | str) -> int:
    """Accept 4660, "4660" or "0x1234" -- JSON has no hex literals."""
    return int(value, 0) if isinstance(value, str) else int(value)


def _run_check(segment: bytes, check: dict[str, Any]) -> bool:
    kind = check.get("kind")
    if kind == "cstring_via_pointer":
        at = _num(check["pointer"])
        pointer = int.from_bytes(segment[at : at + 2], "little")
        if not 0 <= pointer < SEGMENT_SIZE:
            return False
        expected = check["value"].encode("latin-1") + b"\0"
        return segment[pointer : pointer + len(expected)] == expected
    if kind == "max":
        return segment[_num(check["offset"])] <= _num(check["value"])
    if kind == "max_range":
        return all(
            byte <= _num(check["value"])
            for byte in segment[_num(check["start"]) : _num(check["end"])]
        )
    if kind == "equals":
        expected = bytes.fromhex(check["bytes"])
        offset = _num(check["offset"])
        return segment[offset : offset + len(expected)] == expected
    raise ProfileError(f"unknown check kind {kind!r}")


def _scale6(value: int) -> int:
    """Expand a 6-bit VGA DAC component to 8 bits by bit replication."""
    return ((value << 2) | (value >> 4)) & 0xFF if value <= 63 else 255


def _load_palette(spec: dict[str, Any] | None, base: Path) -> tuple[tuple[int, int, int], ...]:
    if not spec:
        return ()
    raw = (base / spec["file"]).read_bytes()
    if spec.get("format", "rgb8") == "rgb6":
        raw = bytes(_scale6(component) for component in raw)
    entries = [tuple(raw[i * 3 : i * 3 + 3]) for i in range(len(raw) // 3)]
    return tuple(entries)  # type: ignore[return-value]


def load_profile(path: str | Path) -> Profile:
    """Read a profile from disk, resolving its marker and palette files."""
    path = Path(path).resolve()
    if not path.is_file():
        raise ProfileError(f"no profile at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent

    marker_spec = data.get("marker")
    if marker_spec is None:
        raise ProfileError(
            f"profile {path.name} has no marker, so nothing identifies the game's "
            f"data segment.  Give marker.bytes (hex) with marker.offset, or "
            f"marker.source_dump with marker.offset and marker.length.  To drive a "
            f"game with no profile at all, pass no profile instead of writing an "
            f"empty one."
        )
    offset = _num(marker_spec["offset"])
    if "bytes" in marker_spec:
        marker = bytes.fromhex(marker_spec["bytes"])
    else:
        dump = Path(marker_spec["source_dump"])
        if not dump.is_absolute():
            dump = (base / dump).resolve()
        if not dump.is_file():
            raise ProfileError(
                f"profile {path.name} points at a reference dump that is not there: "
                f"{dump}.  Capture one with dosbox_dump_segment, or inline the "
                f"pattern as marker.bytes."
            )
        raw = dump.read_bytes()
        if len(raw) != SEGMENT_SIZE:
            raise ProfileError(f"reference dump {dump} is {len(raw)} bytes, not {SEGMENT_SIZE}")
        marker = raw[offset : offset + _num(marker_spec["length"])]

    symbols = {
        name: Symbol(
            name=name,
            offset=_num(spec["offset"]),
            size=int(spec.get("size", 1)),
            description=spec.get("description", ""),
        )
        for name, spec in data.get("symbols", {}).items()
    }

    screen = data.get("screen", {})
    mouse = data.get("mouse", {})
    video = data.get("video", {})
    watch_sets = {
        name: tuple(str(entry) for entry in entries)
        for name, entries in data.get("watch_sets", {}).items()
    }
    return Profile(
        name=data.get("name", path.stem),
        path=path,
        description=data.get("description", ""),
        marker=marker,
        marker_offset=offset,
        checks=tuple(data.get("checks", ())),
        window_title=data.get("window_title", "DOSBox-X"),
        ds_segment=(
            int(data["ds_segment"], 0)
            if isinstance(data.get("ds_segment"), str)
            else data.get("ds_segment")
        ),
        mouse_buttons=_optional_offset(mouse.get("buttons")),
        mouse_position=_optional_offset(mouse.get("position")),
        screen_width=int(screen.get("width", 320)),
        screen_height=int(screen.get("height", 200)),
        palette=_load_palette(screen.get("palette"), base),
        symbols=symbols,
        watch_sets=watch_sets,
        video_mode=str(video.get("mode", "13h")),
        code_segment_offset=_optional_offset(video.get("code_segment_offset"))
        if "code_segment_offset" in video
        else _optional_offset(data.get("code_segment_offset")),
        caves=tuple(data.get("caves", ())),
    )


def _optional_offset(value: int | str | None) -> int | None:
    if value is None:
        return None
    return int(value, 0) if isinstance(value, str) else int(value)


def discover_profiles(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.json")) if directory.is_dir() else []
