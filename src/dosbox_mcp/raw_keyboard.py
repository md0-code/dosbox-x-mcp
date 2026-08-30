"""Versioned key identifiers for the DOSBox-X MCP keyboard hook.

The companion patch accepts the ``KBD_KEYS`` values from DOSBox-X 2026.07.02.
Keeping name resolution here makes it platform-independent and directly
testable; a backend only transports the resolved integer to a capable window.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RawKey:
    name: str
    identifier: int


# Exact KBD_KEYS order in DOSBox-X tag dosbox-x-v2026.07.02.  The hook property
# includes protocol version 1, so a future incompatible enum requires a new
# property/message version instead of silently sending the wrong key.
_KEY_NAMES = (
    "none",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    "q", "w", "e", "r", "t", "y", "u", "i", "o", "p",
    "a", "s", "d", "f", "g", "h", "j", "k", "l",
    "z", "x", "c", "v", "b", "n", "m",
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10",
    "f11", "f12", "esc", "tab", "backspace", "enter", "space",
    "leftalt", "rightalt", "leftctrl", "rightctrl", "leftshift", "rightshift",
    "capslock", "scrolllock", "numlock", "grave", "minus", "equals",
    "backslash", "leftbracket", "rightbracket", "semicolon", "quote",
    "period", "comma", "slash", "extra_lt_gt", "printscreen", "pause",
    "insert", "home", "pageup", "delete", "end", "pagedown",
    "left", "up", "down", "right",
    "kp1", "kp2", "kp3", "kp4", "kp5", "kp6", "kp7", "kp8", "kp9", "kp0",
    "kpdivide", "kpmultiply", "kpminus", "kpplus", "kpenter", "kpperiod",
    "lwindows", "rwindows", "rwinmenu", "kpequals",
    "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20", "f21", "f22",
    "f23", "f24",
)

_KEYS = {name: RawKey(name, index) for index, name in enumerate(_KEY_NAMES) if index}
_ALIASES = {
    "escape": "esc",
    "return": "enter",
    "alt": "leftalt",
    "ctrl": "leftctrl",
    "control": "leftctrl",
    "shift": "leftshift",
    "caps": "capslock",
    "scroll": "scrolllock",
    "num": "numlock",
    "pgup": "pageup",
    "pgdn": "pagedown",
    "del": "delete",
    "ins": "insert",
    "win": "lwindows",
    "menu": "rwinmenu",
    " ": "space",
    "`": "grave",
    "-": "minus",
    "=": "equals",
    "\\": "backslash",
    "[": "leftbracket",
    "]": "rightbracket",
    ";": "semicolon",
    "'": "quote",
    ".": "period",
    ",": "comma",
    "/": "slash",
}


def resolve_raw_key(key: str) -> RawKey:
    """Resolve one physical key name for protocol v1."""
    name = key.lower()
    name = _ALIASES.get(name, name)
    resolved = _KEYS.get(name)
    if resolved is None:
        raise ValueError(
            f"unknown raw key {key!r}; use a letter, digit, f1..f24, arrow, "
            "modifier, navigation key, or an explicit keypad name such as kpplus"
        )
    return resolved


__all__ = ["RawKey", "resolve_raw_key"]
