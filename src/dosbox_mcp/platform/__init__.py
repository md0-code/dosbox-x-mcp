"""Pick a backend for this host, and re-export what the rest of the server uses.

Importing this module is how every other module reaches the machine.  Nothing
above it mentions Win32 or `/proc`; the difference between the two shows up in
exactly one place a caller can see, :func:`capabilities`, which is what
``dosbox_capabilities`` reports so an agent learns the shape of its environment
by asking rather than by failing.
"""
from __future__ import annotations

import sys

from .base import READ_CHUNK, BaseProcess, Capabilities, ProcessError, Window

if sys.platform == "win32":
    from . import windows as backend
elif sys.platform.startswith("linux"):
    from . import linux as backend
else:  # pragma: no cover - macOS and the BSDs have neither backend
    backend = None  # type: ignore[assignment]


class _Unsupported:
    """Stands in for a backend on a platform that has none."""

    @staticmethod
    def capabilities() -> Capabilities:
        return Capabilities(
            platform=sys.platform,
            notes=(
                f"There is no backend for {sys.platform}.  Guest memory needs "
                f"ReadProcessMemory (Windows) or process_vm_readv (Linux); a "
                f"macOS backend would need task_for_pid and therefore either "
                f"root or a signed, entitled binary.",
            ),
        )

    def __getattr__(self, name: str):
        def refuse(*_: object, **__: object):
            raise ProcessError(self.capabilities().notes[0])

        return refuse


if backend is None:  # pragma: no cover
    backend = _Unsupported()  # type: ignore[assignment]


def _from_backend(name: str, default=None):
    return getattr(backend, name, default)


Process = _from_backend("Process")
find_windows = _from_backend("find_windows", lambda *a, **k: [])
client_size = _from_backend("client_size")
resize_client = _from_backend("resize_client")
grab_client = _from_backend("grab_client")
menu_window = _from_backend("menu_window", lambda pid: None)
menu_items = _from_backend("menu_items", lambda handle, depth=4: {})
resolve_command = _from_backend("resolve_command")
menu_command = _from_backend("menu_command")
raw_keyboard_available = _from_backend("raw_keyboard_available", lambda handle: False)
raw_keyboard_event = _from_backend("raw_keyboard_event")
terminate = _from_backend("terminate")


def capabilities() -> Capabilities:
    """What this host can do.  Cheap, and safe to call before anything else."""
    return backend.capabilities()


def name() -> str:
    return capabilities().platform


def require(what: str) -> None:
    """Raise with the reason when a capability this call needs is absent."""
    have = capabilities()
    if getattr(have, what, False):
        return
    detail = "  ".join(have.notes)
    raise ProcessError(
        f"{what.replace('_', ' ')} is not available on {have.platform}.  {detail}"
    )


__all__ = [
    "READ_CHUNK",
    "BaseProcess",
    "Capabilities",
    "Process",
    "ProcessError",
    "Window",
    "capabilities",
    "client_size",
    "find_windows",
    "grab_client",
    "menu_command",
    "menu_items",
    "menu_window",
    "name",
    "raw_keyboard_available",
    "raw_keyboard_event",
    "require",
    "resize_client",
    "resolve_command",
    "terminate",
]
