"""The Linux backend: `/proc` and `process_vm_readv` for memory, X11 for windows.

Memory ports cleanly and is where nearly all of this server's value lives.
`/proc/<pid>/maps` replaces `VirtualQueryEx` and is both cheaper and more
informative; `process_vm_readv` and `process_vm_writev` replace
`ReadProcessMemory` and `WriteProcessMemory` one for one.

**The access gate.**  Windows requires the emulator to run at the same
integrity level as the server.  Linux has the exact analogue in
``kernel.yama.ptrace_scope``:

===== ==========================================================
value effect
===== ==========================================================
0     any same-uid process -- attaching to a running guest works
1     descendants only -- launching works, attaching does not
2     ``CAP_SYS_PTRACE`` only
3     no attach at all
===== ==========================================================

One is the Debian and Ubuntu default, so on a stock desktop
``dosbox_launch`` works and ``dosbox_attach`` does not.  That is worth saying
in the failure text rather than surfacing a bare ``EPERM``, which is what
:func:`_denied` is for.

Windows are the part that does not port.  ``PrintWindow`` renders a window
without activating it and X11 has no equivalent: the nearest is
``XCompositeNameWindowPixmap`` plus ``XGetImage``, which needs a compositing
manager, and Wayland has no cross-client capture at all short of the PipeWire
portal.  The answer is not to emulate the mechanism but to remove the
constraint it satisfies -- run the emulator on its own ``Xvfb`` display, where
there is no desktop to protect.  See :mod:`dosbox_mcp.emulator`.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import signal
from typing import Iterator

from .base import BaseProcess, Capabilities, ProcessError, Window

#: `/proc/<pid>/maps` lines: start-end perms offset dev inode  pathname
MAPS_LINE = re.compile(r"^([0-9a-f]+)-([0-9a-f]+)\s+(\S{4})\s+\S+\s+\S+\s+\S+\s*(.*)$")

#: Mappings that are never guest RAM and are expensive or fatal to read.
SKIP_PATHS = {"[vvar]", "[vsyscall]", "[vdso]", "[vvar_vclock]"}

PTRACE_SCOPE = "/proc/sys/kernel/yama/ptrace_scope"


class iovec(ctypes.Structure):
    _fields_ = [("iov_base", ctypes.c_void_p), ("iov_len", ctypes.c_size_t)]


_libc_handle = None


def libc():
    """The C library, bound lazily.

    Lazily so that this module imports on any host: the maps parser and the
    permission diagnostics are ordinary logic worth testing from a Windows
    machine, and nothing but the two vm calls actually needs libc.
    """
    global _libc_handle
    if _libc_handle is not None:
        return _libc_handle
    name = ctypes.util.find_library("c") or "libc.so.6"
    handle = ctypes.CDLL(name, use_errno=True)
    for symbol in ("process_vm_readv", "process_vm_writev"):
        function = getattr(handle, symbol)
        function.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(iovec),
            ctypes.c_ulong,
            ctypes.POINTER(iovec),
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        function.restype = ctypes.c_ssize_t
    _libc_handle = handle
    return handle


def parse_maps(text: str):
    """``(base, size, readable, path)`` for each line of a `/proc/pid/maps`.

    Split out from :meth:`Process.regions` because it is the part with a rule
    in it, and a rule is worth testing without a live process to point at.
    """
    out = []
    for line in text.splitlines():
        match = MAPS_LINE.match(line)
        if match is None:
            continue
        start, end, perms, path = match.groups()
        base, limit = int(start, 16), int(end, 16)
        readable = perms[0] == "r" and path.strip() not in SKIP_PATHS
        if limit > base:
            out.append((base, limit - base, readable, path.strip()))
    return out


def ptrace_scope() -> int | None:
    """The current ``kernel.yama.ptrace_scope``, or None where Yama is absent."""
    try:
        with open(PTRACE_SCOPE, "r", encoding="ascii") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def _denied(pid: int, errno: int, what: str) -> ProcessError:
    """Turn an EPERM into the sentence that actually explains it."""
    if errno not in (1, 13):  # EPERM, EACCES
        return ProcessError(f"{what} on process {pid} failed: {os.strerror(errno)}")
    scope = ptrace_scope()
    if scope == 1:
        hint = (
            "kernel.yama.ptrace_scope is 1, the Debian and Ubuntu default, which "
            "allows this only for a descendant process.  Use dosbox_launch so the "
            "emulator is a child of this server, or set the sysctl to 0 "
            "(sudo sysctl kernel.yama.ptrace_scope=0) to allow attaching to a "
            "guest that is already running."
        )
    elif scope in (2, 3):
        hint = (
            f"kernel.yama.ptrace_scope is {scope}, which forbids this without "
            f"CAP_SYS_PTRACE ({'and forbids it entirely' if scope == 3 else 'even for a child'})."
        )
    else:
        hint = (
            "the process belongs to another user, or a security module (AppArmor, "
            "SELinux) is denying it.  Run the emulator from the same account."
        )
    return ProcessError(f"{what} on process {pid} was denied; {hint}")


class Process(BaseProcess):
    """A Linux emulator process, reached through `/proc` and the vm syscalls."""

    def __init__(self, pid: int, write: bool = False) -> None:
        if not os.path.isdir(f"/proc/{pid}"):
            raise ProcessError(f"no process {pid}; it has exited")
        self.pid = pid
        self.writable = write
        self._maps = f"/proc/{pid}/maps"
        # One cheap probe so a permission problem is reported at attach time,
        # with the explanation, rather than as a mysterious empty read later.
        first = self._first_readable()
        if first is None:
            raise ProcessError(
                f"process {pid} has no readable mappings in /proc/{pid}/maps; it is "
                f"a kernel thread, or it exited while being opened"
            )
        if self.read(first, 1) is None:
            raise _denied(pid, ctypes.get_errno() or 1, "reading memory")

    def _first_readable(self) -> int | None:
        for base, _, _ in self.regions():
            return base
        return None

    def close(self) -> None:
        pass  # `/proc` needs no handle; every call is self-contained.

    def alive(self) -> bool:
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return os.path.isdir(f"/proc/{self.pid}")

    # ── memory ─────────────────────────────────────────────────────────────

    def read(self, address: int, size: int) -> bytes | None:
        if size <= 0:
            return b""
        buffer = ctypes.create_string_buffer(size)
        local = iovec(ctypes.cast(buffer, ctypes.c_void_p), size)
        remote = iovec(ctypes.c_void_p(address), size)
        ctypes.set_errno(0)
        count = libc().process_vm_readv(
            self.pid, ctypes.byref(local), 1, ctypes.byref(remote), 1, 0
        )
        if count == size:
            return buffer.raw
        if count < 0 and ctypes.get_errno() in (1, 13):
            raise _denied(self.pid, ctypes.get_errno(), "reading memory")
        return None

    def write(self, address: int, data: bytes) -> None:
        if not self.writable:
            raise ProcessError(
                f"process {self.pid} was opened read-only; reattach with write access"
            )
        if not data:
            return
        buffer = ctypes.create_string_buffer(data, len(data))
        local = iovec(ctypes.cast(buffer, ctypes.c_void_p), len(data))
        remote = iovec(ctypes.c_void_p(address), len(data))
        ctypes.set_errno(0)
        count = libc().process_vm_writev(
            self.pid, ctypes.byref(local), 1, ctypes.byref(remote), 1, 0
        )
        if count == len(data):
            return
        errno = ctypes.get_errno()
        if errno in (1, 13):
            raise _denied(self.pid, errno, "writing memory")
        # process_vm_writev refuses a read-only mapping outright; `/proc/pid/mem`
        # goes through the same permission check but honours the write, which is
        # what patching a guest's code segment needs.
        try:
            with open(f"/proc/{self.pid}/mem", "wb", 0) as handle:
                handle.seek(address)
                handle.write(data)
            return
        except OSError as failure:
            raise ProcessError(
                f"writing {len(data)} bytes at 0x{address:X} failed: {failure}"
            ) from None

    def regions(self) -> Iterator[tuple[int, int, int]]:
        """``(base, size, allocation_base)`` for every readable private mapping.

        Linux has no notion of an allocation base distinct from the mapping, so
        the mapping's own start stands in for it -- which is what callers use it
        for anyway.
        """
        try:
            with open(self._maps, "r", encoding="ascii", errors="replace") as handle:
                text = handle.read()
        except OSError:
            return
        for base, size, readable, _ in parse_maps(text):
            if readable:
                yield base, size, base


# ── windows ────────────────────────────────────────────────────────────────
#
# Imported lazily and reported as unavailable rather than raising at import
# time, so a memory-only install needs no X libraries at all.

_display = None
_xlib_error: str | None = None


def _x11():
    """The X display, or None with ``_xlib_error`` explaining why not."""
    global _display, _xlib_error
    if _display is not None or _xlib_error is not None:
        return _display
    if not os.environ.get("DISPLAY"):
        _xlib_error = (
            "no DISPLAY is set.  On Wayland there is no cross-client window "
            "capture at all; run the emulator on an Xvfb display instead "
            "(dosbox_launch with isolated=true)."
        )
        return None
    try:
        from Xlib import display as xdisplay  # type: ignore

        _display = xdisplay.Display()
    except Exception as failure:  # pragma: no cover - depends on the host
        _xlib_error = (
            f"python-xlib could not open {os.environ['DISPLAY']}: {failure}.  "
            f"Install it with `pip install dosbox-x-mcp[x11]` for window tools; "
            f"every memory tool works without it."
        )
        return None
    return _display


def _pid_of(window) -> int:
    from Xlib import Xatom  # type: ignore

    try:
        atom = _display.get_atom("_NET_WM_PID")
        prop = window.get_full_property(atom, Xatom.CARDINAL)
        return int(prop.value[0]) if prop else 0
    except Exception:
        return 0


def _title_of(window) -> str:
    try:
        for name in ("_NET_WM_NAME", "WM_NAME"):
            prop = window.get_full_property(_display.get_atom(name), 0)
            if prop and prop.value:
                value = prop.value
                return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
    except Exception:
        pass
    return ""


def find_windows(title_substring: str | None = None, pid: int | None = None) -> list[Window]:
    """Mapped top-level windows, filtered by title substring and owning pid."""
    display = _x11()
    if display is None:
        return []
    found: list[Window] = []
    seen: set[int] = set()

    def walk(window, depth: int) -> None:
        if depth > 4:
            return
        try:
            children = window.query_tree().children
        except Exception:
            return
        for child in children:
            try:
                attributes = child.get_attributes()
                if attributes.map_state != 2:  # IsViewable
                    walk(child, depth + 1)
                    continue
            except Exception:
                continue
            owner, title = _pid_of(child), _title_of(child)
            if not owner and not title:
                walk(child, depth + 1)
                continue
            if child.id in seen:
                continue
            if pid is not None and owner != pid:
                walk(child, depth + 1)
                continue
            if title_substring and title_substring.lower() not in title.lower():
                walk(child, depth + 1)
                continue
            seen.add(child.id)
            found.append(Window(handle=int(child.id), title=title, pid=owner))

    walk(display.screen().root, 0)
    return found


def _window(handle: int):
    display = _x11()
    if display is None:
        raise ProcessError(_xlib_error or "no X display")
    return display.create_resource_object("window", handle)


def client_size(handle: int) -> tuple[int, int]:
    geometry = _window(handle).get_geometry()
    return int(geometry.width), int(geometry.height)


def resize_client(handle: int, width: int, height: int, settle: float) -> tuple[int, int]:
    """Resize, then wait for the server to actually report the new geometry.

    DOSBox-X rebuilds its drawing surface asynchronously, exactly as on
    Windows, and capturing before it settles samples the old surface stretched
    into the new size.
    """
    import time

    window = _window(handle)
    if client_size(handle) == (width, height):
        return width, height
    window.configure(width=width, height=height)
    _display.sync()
    deadline = time.time() + settle
    while time.time() < deadline:
        time.sleep(0.15)
        if client_size(handle) == (width, height):
            break
    return client_size(handle)


def grab_client(handle: int) -> tuple[int, int, bytes]:
    """Photograph a window's contents as BGRX rows.

    Unlike ``PrintWindow`` this reads what is on the screen, so an occluded
    window needs a compositing manager to have kept its contents.  Without one
    the result is whatever is in front of it -- which is why the framebuffer
    reader, not this, is the pixel path on Linux.
    """
    window = _window(handle)
    geometry = window.get_geometry()
    width, height = int(geometry.width), int(geometry.height)
    if width < 10 or height < 10:
        raise ProcessError(f"window is {width}x{height}; nothing to capture")
    try:
        raw = window.get_image(0, 0, width, height, 2, 0xFFFFFFFF)  # ZPixmap
    except Exception as failure:
        raise ProcessError(
            f"XGetImage on window {handle:#x} failed: {failure}.  An unmapped or "
            f"occluded window cannot be read without a compositing manager."
        ) from None
    data = raw.data
    if isinstance(data, str):  # older python-xlib
        data = data.encode("latin-1")
    expected = width * height * 4
    if len(data) < expected:  # 24-bit visual: widen to BGRX
        widened = bytearray(expected)
        for index in range(width * height):
            widened[index * 4 : index * 4 + 3] = data[index * 3 : index * 3 + 3]
        data = bytes(widened)
    return width, height, bytes(data[:expected])


def menu_window(pid: int) -> int | None:
    return None


def menu_items(handle: int, depth: int = 4) -> dict[str, int]:
    return {}


def resolve_command(handle: int, action: str) -> tuple[int, str]:
    raise ProcessError(
        "DOSBox-X draws its own menu on Linux, so there is no menu command to "
        "post.  Start the emulator with dosbox_launch(isolated=true), which puts "
        "it on its own Xvfb display where the host mapper's Ctrl+Alt+F6/F7/F8 "
        "can be driven directly without taking anything from the user."
    )


def menu_command(handle: int, identifier: int) -> None:
    resolve_command(handle, "")


def terminate(pid: int) -> int:
    """SIGTERM, then SIGKILL if it does not go."""
    import time

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return 0
    for _ in range(20):
        time.sleep(0.25)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return 0
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return 0


def capabilities() -> Capabilities:
    scope = ptrace_scope()
    notes = []
    if scope == 1:
        notes.append(
            "kernel.yama.ptrace_scope is 1: dosbox_launch works (the emulator is "
            "a child), dosbox_attach to an already-running guest does not."
        )
    elif scope in (2, 3):
        notes.append(
            f"kernel.yama.ptrace_scope is {scope}: guest memory is unreachable "
            f"without CAP_SYS_PTRACE."
        )
    windows_ok = _x11() is not None
    if not windows_ok and _xlib_error:
        notes.append(_xlib_error)
    notes.append(
        "Window capture reads the screen, so an occluded window needs a "
        "compositing manager.  Prefer dosbox_read_framebuffer, which reads the "
        "guest's video memory and does not involve the window at all."
    )
    return Capabilities(
        platform="linux",
        memory=scope in (None, 0, 1),
        window_list=windows_ok,
        window_capture=windows_ok,
        window_resize=windows_ok,
        menu_commands=False,
        isolated_display=bool(_which("Xvfb")),
        notes=tuple(notes),
    )


def _which(program: str) -> str | None:
    import shutil

    return shutil.which(program)
