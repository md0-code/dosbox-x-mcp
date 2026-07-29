"""Win32 primitives for reading, writing and photographing a DOSBox-X process.

Everything here is self-contained ctypes; nothing imports from the host
tree.  Two ideas carry the whole module:

* the emulator's guest RAM is an ordinary allocation inside the host process,
  so a guest segment can be found by searching host memory for a byte pattern
  that is known to sit at a fixed offset inside it, and

* the emulator's window can be photographed with ``PrintWindow``, which renders
  without activating -- no focus stealing, no desktop interaction.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
from dataclasses import dataclass
from typing import Iterator

PROCESS_VM_READ = 0x0010
PROCESS_VM_WRITE = 0x0020
PROCESS_VM_OPERATION = 0x0008
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_TERMINATE = 0x0001
ACCESS_READ = PROCESS_VM_READ | PROCESS_QUERY_INFORMATION
ACCESS_WRITE = ACCESS_READ | PROCESS_VM_WRITE | PROCESS_VM_OPERATION

MEM_COMMIT = 0x1000
PAGE_NOACCESS = 0x01
PAGE_GUARD = 0x100
MAX_USER_ADDRESS = 0x00007FFFFFFFFFFF
READ_CHUNK = 8 * 1024 * 1024
STILL_ACTIVE = 259


class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BaseAddress", ctypes.c_void_p),
        ("AllocationBase", ctypes.c_void_p),
        ("AllocationProtect", wt.DWORD),
        ("PartitionId", wt.WORD),
        ("RegionSize", ctypes.c_size_t),
        ("State", wt.DWORD),
        ("Protect", wt.DWORD),
        ("Type", wt.DWORD),
    ]


kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

kernel32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
kernel32.OpenProcess.restype = wt.HANDLE
kernel32.CloseHandle.argtypes = [wt.HANDLE]
kernel32.VirtualQueryEx.argtypes = [
    wt.HANDLE,
    ctypes.c_void_p,
    ctypes.POINTER(MEMORY_BASIC_INFORMATION),
    ctypes.c_size_t,
]
kernel32.VirtualQueryEx.restype = ctypes.c_size_t
kernel32.ReadProcessMemory.argtypes = [
    wt.HANDLE,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
kernel32.ReadProcessMemory.restype = wt.BOOL
kernel32.WriteProcessMemory.argtypes = [
    wt.HANDLE,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
kernel32.WriteProcessMemory.restype = wt.BOOL
kernel32.GetExitCodeProcess.argtypes = [wt.HANDLE, ctypes.POINTER(wt.DWORD)]
kernel32.GetExitCodeProcess.restype = wt.BOOL


class ProcessError(RuntimeError):
    """A process could not be opened, read or written."""


class Process:
    """A handle to one emulator process, opened for the access it needs."""

    def __init__(self, pid: int, write: bool = False) -> None:
        access = ACCESS_WRITE if write else ACCESS_READ
        handle = kernel32.OpenProcess(access, False, pid)
        if not handle:
            code = ctypes.get_last_error()
            hint = (
                "the process has exited"
                if code == 87
                else "it is running at a higher integrity level; "
                "start the emulator from the same account, without elevation"
            )
            raise ProcessError(f"OpenProcess({pid}) failed (error {code}); {hint}")
        self.pid = pid
        self.handle = handle

    def __enter__(self) -> "Process":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self.handle:
            kernel32.CloseHandle(self.handle)
            self.handle = None

    def alive(self) -> bool:
        code = wt.DWORD()
        if not kernel32.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            return False
        return code.value == STILL_ACTIVE

    def read(self, address: int, size: int) -> bytes | None:
        buffer = ctypes.create_string_buffer(size)
        count = ctypes.c_size_t()
        ok = kernel32.ReadProcessMemory(
            self.handle, ctypes.c_void_p(address), buffer, size, ctypes.byref(count)
        )
        if not ok or count.value != size:
            return None
        return buffer.raw

    def write(self, address: int, data: bytes) -> None:
        buffer = ctypes.create_string_buffer(data, len(data))
        count = ctypes.c_size_t()
        ok = kernel32.WriteProcessMemory(
            self.handle, ctypes.c_void_p(address), buffer, len(data), ctypes.byref(count)
        )
        if not ok or count.value != len(data):
            raise ProcessError(
                f"WriteProcessMemory(0x{address:X}, {len(data)} bytes) failed "
                f"(error {ctypes.get_last_error()})"
            )

    def regions(self) -> Iterator[tuple[int, int, int]]:
        """Yield ``(base, size, allocation_base)`` for readable committed memory."""
        address = 0
        info = MEMORY_BASIC_INFORMATION()
        while address < MAX_USER_ADDRESS:
            if not kernel32.VirtualQueryEx(
                self.handle, ctypes.c_void_p(address), ctypes.byref(info), ctypes.sizeof(info)
            ):
                break
            base = int(info.BaseAddress or 0)
            size = int(info.RegionSize)
            if (
                info.State == MEM_COMMIT
                and not (info.Protect & (PAGE_NOACCESS | PAGE_GUARD))
                and size
            ):
                yield base, size, int(info.AllocationBase or 0)
            following = base + size
            if following <= address:
                break
            address = following

    def region_of(self, address: int) -> tuple[int, int, int] | None:
        info = MEMORY_BASIC_INFORMATION()
        if not kernel32.VirtualQueryEx(
            self.handle, ctypes.c_void_p(address), ctypes.byref(info), ctypes.sizeof(info)
        ):
            return None
        return int(info.BaseAddress or 0), int(info.RegionSize), int(info.AllocationBase or 0)

    def find(self, marker: bytes) -> list[int]:
        """Every host address at which ``marker`` occurs in readable memory."""
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
                    start = found + 1
                tail = haystack[-overlap:] if overlap else b""
                position += count
        return hits


# ── windows ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Window:
    handle: int
    title: str
    pid: int


def find_windows(title_substring: str | None = None, pid: int | None = None) -> list[Window]:
    """Visible top-level windows filtered by title substring and/or owning pid."""
    found: list[Window] = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def callback(handle: int, _: int) -> bool:
        length = user32.GetWindowTextLengthW(handle)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(handle, buffer, length + 1)
        owner = wt.DWORD()
        user32.GetWindowThreadProcessId(handle, ctypes.byref(owner))
        if pid is not None and owner.value != pid:
            return True
        if title_substring and title_substring.lower() not in buffer.value.lower():
            return True
        if not user32.IsWindowVisible(handle):
            return True
        found.append(Window(handle=int(handle), title=buffer.value, pid=owner.value))
        return True

    user32.EnumWindows(callback, 0)
    return found


def client_size(handle: int) -> tuple[int, int]:
    rect = wt.RECT()
    user32.GetClientRect(handle, ctypes.byref(rect))
    return rect.right - rect.left, rect.bottom - rect.top


def resize_client(handle: int, width: int, height: int, settle: float) -> tuple[int, int]:
    """Resize so the client area is exactly ``width`` x ``height``.

    DOSBox-X rebuilds its drawing surface asynchronously; capturing before it
    settles samples the old surface stretched into the new size, which shifts
    every pixel.  So this waits for the client to actually report the new size.
    """
    import time

    have = client_size(handle)
    if have == (width, height):
        return have
    window = wt.RECT()
    user32.GetWindowRect(handle, ctypes.byref(window))
    frame_w = (window.right - window.left) - have[0]
    frame_h = (window.bottom - window.top) - have[1]
    user32.MoveWindow(
        handle, window.left, window.top, width + frame_w, height + frame_h, True
    )
    deadline = time.time() + settle
    while time.time() < deadline:
        time.sleep(0.15)
        if client_size(handle) == (width, height):
            break
    return client_size(handle)


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wt.DWORD),
        ("biWidth", wt.LONG),
        ("biHeight", wt.LONG),
        ("biPlanes", wt.WORD),
        ("biBitCount", wt.WORD),
        ("biCompression", wt.DWORD),
        ("biSizeImage", wt.DWORD),
        ("biXPelsPerMeter", wt.LONG),
        ("biYPelsPerMeter", wt.LONG),
        ("biClrUsed", wt.DWORD),
        ("biClrImportant", wt.DWORD),
    ]


def grab_client(handle: int) -> tuple[int, int, bytes]:
    """Photograph a window's client area as BGRX rows, without activating it."""
    import time

    user32.ShowWindow(handle, 4)  # SW_SHOWNOACTIVATE
    time.sleep(0.2)
    width, height = client_size(handle)
    if width < 10 or height < 10:
        raise ProcessError(f"window client area is {width}x{height}; nothing to capture")
    hdc = user32.GetDC(handle)
    memory_dc = gdi32.CreateCompatibleDC(hdc)
    bitmap = gdi32.CreateCompatibleBitmap(hdc, width, height)
    gdi32.SelectObject(memory_dc, bitmap)
    try:
        # PW_CLIENTONLY | PW_RENDERFULLCONTENT
        user32.PrintWindow(handle, memory_dc, 3)
        header = BITMAPINFOHEADER()
        header.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        header.biWidth = width
        header.biHeight = -height  # top-down
        header.biPlanes = 1
        header.biBitCount = 32
        header.biCompression = 0
        buffer = ctypes.create_string_buffer(width * height * 4)
        gdi32.GetDIBits(memory_dc, bitmap, 0, height, buffer, ctypes.byref(header), 0)
        return width, height, buffer.raw
    finally:
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(handle, hdc)
