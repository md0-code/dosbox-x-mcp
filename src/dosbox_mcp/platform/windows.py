"""The Windows backend: reading, writing and photographing a DOSBox-X process.

Everything here is self-contained ctypes; nothing imports from the host
tree.  Three ideas carry the whole module:

* the emulator's guest RAM is an ordinary allocation inside the host process,
  so a guest segment can be found by searching host memory for a byte pattern
  that is known to sit at a fixed offset inside it, and

* the emulator's window can be photographed with ``PrintWindow``, which renders
  without activating -- no focus stealing, no desktop interaction, and

* DOSBox-X's own menu items can be fired with ``PostMessage(WM_COMMAND)``,
  which reaches its capture facilities without a keystroke and without the host
  mapper.  That last one has no equivalent on any other platform, because
  DOSBox-X draws its own menu everywhere else.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
from typing import Iterator

from .base import BaseProcess, Capabilities, ProcessError, Window

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


class Process(BaseProcess):
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


# ── windows ────────────────────────────────────────────────────────────────


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


# ── the emulator's own menu ────────────────────────────────────────────────
#
# DOSBox-X binds its capture facilities to the host mapper (Ctrl+Alt+F6/F7/F8),
# which is unreachable from here: keys are injected into the BIOS ring, far
# below SDL.  But on Windows the same actions are native menu items, and one
# PostMessage fires one without focus, without a keystroke, and without
# touching the host desktop -- the constraint this whole server is built
# around.  Posting the same id again stops a recording.

WM_COMMAND = 0x0111
MF_BYPOSITION = 0x0400
MCP_KEYBOARD_MESSAGE = "DOSBox-X MCP keyboard v1"
MCP_KEYBOARD_PROPERTY = "DOSBoxXMCPKeyboardV1"

#: Fallbacks for a build whose menu cannot be walked (a hidden menu bar, say).
#: These are stable within a build; the walk below is what survives an upgrade.
KNOWN_COMMAND_IDS = {
    "record_wave": 0x145C,
    "record_midi": 0x145E,
    "record_opl": 0x145F,
}

#: Which menu labels each action means.  Every keyword must appear, case
#: -insensitively, in the item's text; the shortest match wins so that
#: "Record audio (WAV)" beats "Record audio (WAV) with ...".
COMMAND_KEYWORDS = {
    "record_wave": (("record", "wav"),),
    "record_midi": (("record", "midi"),),
    "record_opl": (("record", "opl"), ("raw", "opl")),
    "record_video": (("record", "video"), ("capture", "avi")),
    "screenshot": (("screenshot",), ("save", "png")),
    "save_state": (("save", "state"),),
    "load_state": (("load", "state"),),
}

user32.GetMenu.argtypes = [wt.HWND]
user32.GetMenu.restype = wt.HMENU
user32.GetSubMenu.argtypes = [wt.HMENU, ctypes.c_int]
user32.GetSubMenu.restype = wt.HMENU
user32.GetMenuItemCount.argtypes = [wt.HMENU]
user32.GetMenuItemCount.restype = ctypes.c_int
user32.GetMenuItemID.argtypes = [wt.HMENU, ctypes.c_int]
user32.GetMenuItemID.restype = wt.UINT
user32.GetMenuStringW.argtypes = [wt.HMENU, wt.UINT, wt.LPWSTR, ctypes.c_int, wt.UINT]
user32.GetMenuStringW.restype = ctypes.c_int
user32.PostMessageW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
user32.PostMessageW.restype = wt.BOOL
user32.RegisterWindowMessageW.argtypes = [wt.LPCWSTR]
user32.RegisterWindowMessageW.restype = wt.UINT
user32.GetPropW.argtypes = [wt.HWND, wt.LPCWSTR]
user32.GetPropW.restype = wt.HANDLE


def menu_window(pid: int) -> int | None:
    """The process's visible window that has a menu -- the emulator's own."""
    for window in find_windows(pid=pid):
        if user32.GetMenu(window.handle):
            return window.handle
    return None


def menu_items(handle: int, depth: int = 4) -> dict[str, int]:
    """Every command item in the window's menu tree, as ``label -> id``.

    Walked rather than hard-coded so the ids survive a DOSBox-X upgrade.
    Labels keep their accelerator markers stripped and their whitespace
    squeezed, because that is what a keyword match wants.
    """
    root = user32.GetMenu(handle)
    if not root:
        return {}
    found: dict[str, int] = {}

    def walk(menu: int, level: int) -> None:
        if level > depth:
            return
        for index in range(user32.GetMenuItemCount(menu)):
            buffer = ctypes.create_unicode_buffer(256)
            user32.GetMenuStringW(menu, index, buffer, 256, MF_BYPOSITION)
            label = " ".join(buffer.value.replace("&", "").split())
            child = user32.GetSubMenu(menu, index)
            if child:
                walk(child, level + 1)
                continue
            identifier = int(user32.GetMenuItemID(menu, index))
            if label and identifier not in (0, 0xFFFFFFFF):
                found.setdefault(label, identifier)

    walk(root, 0)
    return found


def resolve_command(handle: int, action: str) -> tuple[int, str]:
    """The command id for a named action, and the label it was matched from."""
    items = menu_items(handle)
    best: tuple[int, str] | None = None
    for keywords in COMMAND_KEYWORDS.get(action, ()):
        for label, identifier in items.items():
            lowered = label.lower()
            if all(word in lowered for word in keywords):
                if best is None or len(label) < len(best[1]):
                    best = (identifier, label)
        if best is not None:
            return best
    fallback = KNOWN_COMMAND_IDS.get(action)
    if fallback is not None:
        return fallback, f"(fallback id {fallback:#06x}; not found in the menu)"
    known = ", ".join(sorted(COMMAND_KEYWORDS))
    raise ProcessError(
        f"no menu item matches action {action!r}.  Known actions: {known}.  "
        f"The window's menu has {len(items)} items; if it is empty the menu bar "
        f"is hidden (set 'menu=true' in the [sdl] section of the config)."
    )


def menu_command(handle: int, identifier: int) -> None:
    """Fire one menu item without focus and without a keystroke."""
    if not user32.PostMessageW(handle, WM_COMMAND, identifier, 0):
        raise ProcessError(
            f"the emulator refused WM_COMMAND {identifier:#06x} "
            f"(error {ctypes.get_last_error()})"
        )


def raw_keyboard_available(handle: int) -> bool:
    """Whether this window exposes the versioned DOSBox-X keyboard hook."""
    return int(user32.GetPropW(handle, MCP_KEYBOARD_PROPERTY) or 0) == 1


def raw_keyboard_event(handle: int, identifier: int, pressed: bool) -> None:
    """Queue one authentic KEYBOARD_AddKey transition without activating a window."""
    if not raw_keyboard_available(handle):
        raise ProcessError(
            "this DOSBox-X window does not expose the MCP keyboard v1 hook; "
            "build DOSBox-X 2026.07.02 with companion/dosbox-x-mcp-keyboard-v1.patch"
        )
    message = user32.RegisterWindowMessageW(MCP_KEYBOARD_MESSAGE)
    if not message or not user32.PostMessageW(handle, message, identifier, int(pressed)):
        raise ProcessError(
            f"the emulator refused raw keyboard event {identifier} "
            f"(error {ctypes.get_last_error()})"
        )


def capabilities() -> Capabilities:
    return Capabilities(
        platform="windows",
        memory=True,
        window_list=True,
        window_capture=True,
        window_resize=True,
        menu_commands=True,
        raw_keyboard=True,
        isolated_display=False,
        notes=(
            "The emulator must run at the same integrity level as this server; "
            "an elevated DOSBox-X cannot be opened by an unelevated client.",
            "PrintWindow costs the guest real time: a filming loop stretches "
            "guest-visible phases by roughly 1.6x.  Prefer the framebuffer "
            "reader for anything quantitative.",
        ),
    )


def terminate(pid: int) -> int:
    """End the process and its children.  Returns taskkill's exit code."""
    import subprocess

    return subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True
    ).returncode
