"""A DOS guest built in a bytes buffer, so the tests need no emulator.

Everything this server does to a real guest -- finding it by its BIOS data
area, walking its memory chain, addressing a segment, reading its framebuffer,
patching a call into a cave -- is done to host memory that happens to contain a
guest.  So a plausible guest in a ``bytearray``, behind a process object with
the same four calls the real backends provide, exercises all of it offline.

The point is not to emulate DOS.  It is to lay down exactly the structures the
detection depends on, so that a test failing means the detection is wrong
rather than that the fixture drifted.
"""
from __future__ import annotations

import struct

from dosbox_mcp.guest import BDA, KEYBOARD_END, KEYBOARD_START, PARAGRAPH

#: A guest with a megabyte of address space, offset inside a larger host
#: allocation so nothing accidentally depends on the guest starting at zero.
GUEST_SPAN = 0x110000
HOST_PREFIX = 0x2000


class FakeProcess:
    """The four calls a backend has to provide, over one bytearray."""

    def __init__(self, memory: bytearray, base: int = 0x40000000) -> None:
        self.pid = 4242
        self.memory = memory
        self.base = base
        self.writes = 0

    # -- the backend interface --------------------------------------------

    def read(self, address: int, size: int) -> bytes | None:
        start = address - self.base
        if start < 0 or start + size > len(self.memory):
            return None
        return bytes(self.memory[start : start + size])

    def write(self, address: int, data: bytes) -> None:
        start = address - self.base
        if start < 0 or start + len(data) > len(self.memory):
            raise AssertionError(f"write outside the fixture at 0x{address:X}")
        self.memory[start : start + len(data)] = data
        self.writes += 1

    def regions(self):
        yield self.base, len(self.memory), self.base

    def region_of(self, address: int):
        if self.base <= address < self.base + len(self.memory):
            return self.base, len(self.memory), self.base
        return None

    def alive(self) -> bool:
        return True

    def close(self) -> None:
        pass

    # -- BaseProcess.find, so detection is exercised for real --------------

    def find(self, marker: bytes, limit: int | None = None) -> list[int]:
        from dosbox_mcp.platform.base import BaseProcess

        return BaseProcess.find(self, marker, limit)


def build_guest(
    programs: list[tuple[str, str, int]] | None = None,
    ram_offset: int = HOST_PREFIX,
) -> tuple[FakeProcess, int]:
    """A guest with a BIOS data area, an IVT and a DOS memory chain.

    ``programs`` are ``(mcb name, full path, paragraphs)``.  Returns the
    process and the host address of guest physical zero.
    """
    memory = bytearray(ram_offset + GUEST_SPAN)
    process = FakeProcess(memory)
    ram = process.base + ram_offset

    def put(physical: int, data: bytes) -> None:
        at = ram_offset + physical
        memory[at : at + len(data)] = data

    # The interrupt vector table: a live guest has INT 21h pointing somewhere.
    # A vector is offset then segment, so this is 00C8:1234.
    put(0x21 * 4, struct.pack("<HH", 0x1234, 0x00C8))
    put(0x08 * 4, struct.pack("<HH", 0x0100, 0x0070))

    # The BIOS data area: the keyboard ring's extent, its head and tail, and
    # the conventional memory size.  These three are what detection reads.
    put(BDA + 0x80, struct.pack("<HH", KEYBOARD_START, KEYBOARD_END))
    put(BDA + 0x1A, struct.pack("<HH", KEYBOARD_START, KEYBOARD_START))
    put(BDA + 0x13, struct.pack("<H", 640))

    # The DOS memory chain, starting above the kernel.
    chain = list(programs or [("COMMAND", "C:\\COMMAND.COM", 0x100)])
    segment = 0x0800
    for index, (name, path, paragraphs) in enumerate(chain):
        psp = segment + 1
        last = index == len(chain) - 1
        header = bytearray(PARAGRAPH)
        header[0] = 0x5A if last else 0x4D
        struct.pack_into("<HH", header, 1, psp, paragraphs)
        header[8 : 8 + len(name[:8])] = name[:8].encode("latin-1")
        put(segment * PARAGRAPH, bytes(header))

        # The PSP's environment pointer, and the environment block that holds
        # the program's full path after its variables.
        environment = psp + paragraphs - 0x40
        put(psp * PARAGRAPH + 0x2C, struct.pack("<H", environment))
        block = b"PATH=C:\\DOS\0COMSPEC=C:\\COMMAND.COM\0\0" + struct.pack("<H", 1)
        block += path.encode("latin-1") + b"\0"
        put(environment * PARAGRAPH, block)

        segment = segment + 1 + paragraphs
    return process, ram


def put_page(process: FakeProcess, at: int, page: bytes) -> None:
    """Write a screen into the fixture the way chain-4 video memory holds it.

    CPU offset ``o`` lands at linear ``4 * (o & ~3) + (o & 3)``, so four
    consecutive screen bytes are contiguous and the next twelve belong to
    nothing.  Laying a reference down this way and reading it back through
    ``dechain`` is the round trip that proves the interleave.
    """
    start = at - process.base
    for index in range(0, len(page), 4):
        linear = start + index * 4
        process.memory[linear : linear + 4] = page[index : index + 4]


def draw(width: int = 320, height: int = 200, seed: int = 7) -> bytes:
    """A picture: flat bands with edges, so it is coherent but not blank."""
    page = bytearray(width * height)
    for y in range(height):
        band = (y // 17 + seed) % 200 + 16
        for x in range(width):
            page[y * width + x] = band if x % 40 < 34 else (band + 9) % 256
    return bytes(page)
