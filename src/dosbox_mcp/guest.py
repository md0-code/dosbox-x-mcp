"""Find a DOS guest inside an emulator process, and address any of it.

The rest of this server used to reach exactly one segment: the one a profile
described.  That is a chicken and egg -- the profile's marker has to come from
a dump, and the dump comes from the server -- and it is the loop every new
project starts inside.

This module breaks it.  It finds guest physical address zero by the structures
*every* DOS guest has in the same place, needing no profile and no prior
knowledge:

* the BIOS keyboard ring's extent words at ``0040:0080``, which are always
  ``0x001E`` and ``0x003E``,
* a head and tail inside that extent and evenly aligned, and
* a non-null ``INT 21h`` vector, which only a live guest has.

From there every segment is addressable, so a second program in a launcher
chain, a TSR, an overlay, the interrupt vector table and the DOS memory chain
are all reachable -- and the memory chain is what turns "there is a guest here"
into "these programs are loaded, at these segments, from these paths".
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

SEGMENT_SIZE = 0x10000
PARAGRAPH = 0x10
BDA = 0x0400
IVT = 0x0000
CONVENTIONAL = 0xA0000

KEYBOARD_EXTENT_AT = 0x80  # within the BDA
KEYBOARD_HEAD = 0x1A
KEYBOARD_TAIL = 0x1C
KEYBOARD_START = 0x1E
KEYBOARD_END = 0x3E
MEMORY_SIZE_KB = 0x13  # BDA word: conventional memory in KiB

#: ``0x001E, 0x003E`` little-endian -- the needle that finds a DOS guest.
KEYBOARD_EXTENT = struct.pack("<HH", KEYBOARD_START, KEYBOARD_END)


class GuestError(RuntimeError):
    """No DOS guest could be found, or one could not be addressed as asked."""


# ── finding the guest ──────────────────────────────────────────────────────


def plausible(process, base: int) -> bool:
    """Is ``base`` really guest physical zero?  Three invariants, all cheap."""
    if base < 0:
        return False
    bda = process.read(base + BDA, 0x100)
    if bda is None:
        return False
    start, end = struct.unpack_from("<HH", bda, KEYBOARD_EXTENT_AT)
    if (start, end) != (KEYBOARD_START, KEYBOARD_END):
        return False
    head, tail = struct.unpack_from("<H", bda, KEYBOARD_HEAD)[0], struct.unpack_from(
        "<H", bda, KEYBOARD_TAIL
    )[0]
    if not all(
        KEYBOARD_START <= value < KEYBOARD_END and value % 2 == 0 for value in (head, tail)
    ):
        return False
    ivt = process.read(base + IVT, 0x400)
    if ivt is None or ivt[0x21 * 4 : 0x21 * 4 + 4] == b"\0\0\0\0":
        return False
    return True


def ram_bases(process) -> list[int]:
    """Every host address that plausibly holds guest physical zero.

    More than one is normal only when more than one guest is running; the
    caller decides what to do about that rather than this guessing.
    """
    found: list[int] = []
    for hit in process.find(KEYBOARD_EXTENT):
        base = hit - (BDA + KEYBOARD_EXTENT_AT)
        if base in found:
            continue
        if plausible(process, base):
            found.append(base)
    return sorted(found)


def find_one(process) -> int:
    """The single guest in this process, or an error saying why not."""
    bases = ram_bases(process)
    if not bases:
        raise GuestError(
            f"no DOS guest in process {process.pid}: nothing in its memory looks "
            f"like a BIOS data area.  The emulator may not have booted DOS yet, "
            f"or this is not an emulator."
        )
    if len(bases) > 1:
        listed = ", ".join(f"0x{value:X}" for value in bases)
        raise GuestError(
            f"{len(bases)} DOS guests in process {process.pid} ({listed}); pass "
            f"ram_base to say which."
        )
    return bases[0]


# ── addressing it ──────────────────────────────────────────────────────────


@dataclass
class Guest:
    """One DOS guest's physical memory, addressed the way the guest does."""

    process: object
    ram_base: int

    # -- linear ------------------------------------------------------------

    def linear(self, segment: int, offset: int = 0) -> int:
        """The host address of ``segment:offset``."""
        physical = ((segment & 0xFFFF) << 4) + (offset & 0xFFFF)
        return self.ram_base + physical

    def read_physical(self, address: int, size: int) -> bytes:
        data = self.process.read(self.ram_base + address, size)
        if data is None:
            raise GuestError(f"could not read guest physical 0x{address:05X}+{size}")
        return data

    # -- segmented ---------------------------------------------------------

    def read(self, segment: int, offset: int = 0, size: int = SEGMENT_SIZE) -> bytes:
        if offset + size > SEGMENT_SIZE:
            raise GuestError(
                f"{segment:04X}:{offset:04X}+{size} runs past the end of the segment; "
                f"a segment is 64 KiB and does not wrap here"
            )
        data = self.process.read(self.linear(segment, offset), size)
        if data is None:
            raise GuestError(f"could not read {segment:04X}:{offset:04X}+{size}")
        return data

    def write(self, segment: int, offset: int, data: bytes) -> None:
        if offset + len(data) > SEGMENT_SIZE:
            raise GuestError(
                f"{segment:04X}:{offset:04X}+{len(data)} runs past the end of the segment"
            )
        self.process.write(self.linear(segment, offset), data)

    def word(self, segment: int, offset: int) -> int:
        return int.from_bytes(self.read(segment, offset, 2), "little")

    # -- the parts every guest has -----------------------------------------

    def conventional(self) -> bytes:
        """All 640 KiB, in one read -- the working set for any scan."""
        data = self.process.read(self.ram_base, CONVENTIONAL)
        if data is None:
            raise GuestError("could not read the guest's conventional memory")
        return data

    def memory_kb(self) -> int:
        return int.from_bytes(self.read_physical(BDA + MEMORY_SIZE_KB, 2), "little")

    def vector(self, number: int) -> tuple[int, int]:
        """``(segment, offset)`` of one interrupt vector."""
        offset, segment = struct.unpack("<HH", self.read_physical(number * 4, 4))
        return segment, offset

    def search(self, needle: bytes, limit: int = 64) -> list[tuple[int, int]]:
        """Find ``needle`` in conventional memory, as ``(segment, offset)`` pairs.

        Reported at the paragraph the hit falls in, which is the form a profile
        and a debugger both want.
        """
        memory = self.conventional()
        out: list[tuple[int, int]] = []
        at = memory.find(needle)
        while at >= 0 and len(out) < limit:
            out.append((at >> 4, at & 0xF))
            at = memory.find(needle, at + 1)
        return out

    # -- keyboard ----------------------------------------------------------

    def push_key(self, ascii_code: int, scan_code: int) -> None:
        """Append one key to the BIOS ring, exactly as an interrupt would."""
        bda = self.read_physical(BDA, 0x100)
        head, tail = struct.unpack_from("<H", bda, KEYBOARD_HEAD)[0], struct.unpack_from(
            "<H", bda, KEYBOARD_TAIL
        )[0]
        if not all(
            KEYBOARD_START <= value < KEYBOARD_END and value % 2 == 0
            for value in (head, tail)
        ):
            raise GuestError(
                f"the BIOS keyboard ring looks wrong (head={head:#x} tail={tail:#x}); "
                f"the guest RAM base is probably misidentified"
            )
        following = tail + 2
        if following >= KEYBOARD_END:
            following = KEYBOARD_START
        if following == head:
            raise GuestError(
                "the BIOS keyboard ring is full; the guest is not consuming keys "
                "(is it waiting on the mouse instead, or has it installed its own "
                "INT 9 handler?)"
            )
        self.process.write(
            self.ram_base + BDA + tail, struct.pack("<BB", ascii_code, scan_code)
        )
        self.process.write(
            self.ram_base + BDA + KEYBOARD_TAIL, struct.pack("<H", following)
        )


# ── the DOS memory chain ───────────────────────────────────────────────────
#
# Every allocation DOS has made is a link in one chain of memory control
# blocks, and each link names the program that owns it.  Walking it answers
# the first question anybody profiling a new game has -- what is loaded, and
# where -- without a debugger and without knowing anything about the game.

MCB_LINK = 0x4D  # 'M'
MCB_LAST = 0x5A  # 'Z'


@dataclass(frozen=True)
class Block:
    """One memory control block."""

    segment: int          # the MCB's own paragraph
    data_segment: int     # the paragraph the owner actually gets
    owner: int            # owning PSP, 0 for free, 8 for DOS itself
    paragraphs: int
    name: str
    last: bool

    @property
    def size(self) -> int:
        return self.paragraphs * PARAGRAPH

    def describe(self) -> str:
        who = (
            "free"
            if self.owner == 0
            else "DOS"
            if self.owner == 8
            else f"PSP {self.owner:04X}"
        )
        label = f" {self.name}" if self.name else ""
        return f"{self.data_segment:04X} {self.size:>7} bytes  {who}{label}"


def _walk(memory: bytes, start: int, limit: int = 512) -> list[Block]:
    """Follow the chain from paragraph ``start``; empty if it is not one."""
    blocks: list[Block] = []
    segment = start
    while len(blocks) < limit:
        at = segment * PARAGRAPH
        if at + PARAGRAPH > len(memory):
            return []
        signature = memory[at]
        if signature not in (MCB_LINK, MCB_LAST):
            return []
        owner, paragraphs = struct.unpack_from("<HH", memory, at + 1)
        raw = memory[at + 8 : at + 16]
        name = raw.split(b"\0")[0].decode("latin-1", "replace")
        if not all(32 <= byte < 127 for byte in name.encode("latin-1", "replace")):
            name = ""
        blocks.append(
            Block(
                segment=segment,
                data_segment=segment + 1,
                owner=owner,
                paragraphs=paragraphs,
                name=name.strip(),
                last=signature == MCB_LAST,
            )
        )
        if signature == MCB_LAST:
            return blocks
        segment = segment + 1 + paragraphs
    return []


def memory_chain(guest: Guest) -> list[Block]:
    """Every DOS allocation, in order, found without asking DOS.

    ``INT 21h AH=52h`` would hand over the chain's head, but calling into the
    guest is not something this server does.  Instead the chain is found by the
    property that makes it a chain: from its real head, and from nowhere else,
    the links walk cleanly to a terminator.  The lowest such start wins, which
    is the head by construction -- a walk that begins mid-chain is a suffix,
    and a coincidence never produces a long one.
    """
    memory = guest.conventional()
    best: list[Block] = []
    segment = 0x40  # nothing below the BDA is ever an MCB
    while segment < 0xA000:
        if memory[segment * PARAGRAPH] in (MCB_LINK, MCB_LAST):
            chain = _walk(memory, segment)
            if len(chain) > len(best):
                best = chain
                # A clean walk from here reaches the top of memory; anything
                # starting later is a suffix of it and cannot be longer.
                segment += 1 + chain[0].paragraphs
                continue
        segment += 1
    return best


def program_path(guest: Guest, psp: int) -> str:
    """The full path a loaded program was started from, out of its environment.

    ``PSP:0x2C`` points at the program's environment block: NUL-terminated
    strings, an empty one to end them, a word, and then the ASCIIZ path.  It is
    the only place the guest keeps a program's own name at full length.
    """
    if psp in (0, 8) or psp >= 0xA000:
        return ""
    try:
        environment = guest.word(psp, 0x2C)
        if environment in (0, 0xFFFF):
            return ""
        block = guest.read(environment, 0, 0x8000)
    except GuestError:
        return ""
    end = block.find(b"\0\0")
    if end < 0:
        return ""
    tail = block[end + 2 :]
    if len(tail) < 3 or tail[0] != 1:
        return ""
    path = tail[2:].split(b"\0")[0]
    try:
        text = path.decode("latin-1")
    except UnicodeDecodeError:
        return ""
    return text if all(32 <= byte < 127 for byte in path) else ""


def programs(guest: Guest) -> list[dict[str, object]]:
    """Loaded programs, largest allocation first: segment, size, name, path."""
    out: list[dict[str, object]] = []
    for block in memory_chain(guest):
        # A program's own image is the block whose owner is itself: DOS builds
        # the PSP in the first paragraph of what it allocated.  Every other
        # block is free space, DOS's own, or a further allocation that program
        # asked for afterwards.
        if block.owner in (0, 8) or block.owner != block.data_segment:
            continue
        out.append(
            {
                "psp_segment": f"{block.owner:04X}",
                "load_segment": f"{block.data_segment:04X}",
                "bytes": block.size,
                "name": block.name,
                "path": program_path(guest, block.owner),
            }
        )
    out.sort(key=lambda entry: -int(entry["bytes"]))
    return out


def describe(guest: Guest) -> dict[str, object]:
    """Everything worth knowing about a guest before a profile exists."""
    chain = memory_chain(guest)
    interrupt_21 = guest.vector(0x21)
    return {
        "ram_base_host_address": f"0x{guest.ram_base:X}",
        "conventional_kb": guest.memory_kb(),
        "int21_vector": f"{interrupt_21[0]:04X}:{interrupt_21[1]:04X}",
        "memory_blocks": len(chain),
        "free_bytes": sum(block.size for block in chain if block.owner == 0),
        "programs": programs(guest),
    }
