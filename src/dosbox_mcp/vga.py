"""Read the guest's screen out of the emulator's video memory.

Photographing the window works, but it pays for every frame twice.  It costs
the *guest* real time -- a filming loop stretches guest-visible phases by
roughly 1.6x, which is enough to make a measured animation disagree with the
one the instruction stream implies -- and it costs *accuracy*, because a
screenshot taken between page flips can catch a half-drawn frame, and a pixel
has to be mapped back through the palette to recover the index the guest
actually wrote.

Reading video memory has none of those problems.  What comes back is the
palette indices the guest wrote, at the instant of the read, with no window, no
scaling, no palette round trip, and no compositor involved.  It is also the
only pixel path that works on a Linux host without a compositing manager, and
the only one that works at all with the emulator on an offscreen display.

**The interleave.**  DOSBox stores a chain-4 page the way the hardware does:
CPU offset ``o`` lands at linear ``4 * (o & ~3) + (o & 3)``, so four
consecutive screen bytes are contiguous and the next twelve linear bytes belong
to nothing.  Taking every sixteenth group of four turns a region back into a
screen.

**Finding the page.**  With a reference frame -- one the caller already has,
or one photographed from the window a moment earlier -- the page is found by
searching for a single row of it, which is exact.  Without one it is found by
what makes a picture a picture: adjacent pixels in a drawn frame agree with
each other far more often than bytes in unrelated memory do, horizontally and
vertically at once.  The two tests together separate a real page from garbage
by two orders of magnitude, and the score is reported so a caller can judge
rather than trust.
"""
from __future__ import annotations

import struct
import time
import zlib
from dataclasses import dataclass

WIDTH = 320
HEIGHT = 200

#: Only regions that could hold a VGA aperture are considered: a chain-4 page
#: occupies four times its own size, and nothing here lives in an allocation
#: bigger than an emulator's video memory.
MIN_REGION = 0x40000
MAX_REGION = 0x4000000

#: Mode 13h page-flipping moves the CRTC start address in dwords, so successive
#: pages sit one page-size apart in de-interleaved (CPU) space.
MAX_PAGES = 4

#: Below this, a candidate is not a drawn frame.  A real game screen scores
#: 0.3 to 0.9; unrelated memory scores about 1/256.
COHERENCE_FLOOR = 0.06


class VgaError(RuntimeError):
    """The guest's video memory could not be found or read."""


# ── the interleave ─────────────────────────────────────────────────────────


def dechain(raw: bytes) -> bytes:
    """The screen bytes inside one span of the emulator's video memory."""
    usable = len(raw) & ~0xF
    if not usable:
        return b""
    try:
        import numpy as np

        return (
            np.frombuffer(raw[:usable], dtype="u1")
            .reshape(-1, 16)[:, :4]
            .reshape(-1)
            .tobytes()
        )
    except ImportError:
        pass
    # Without numpy, do it with strided slices, which are still C-level: four
    # planes out, then interleaved back in.  A Python loop over megabytes is
    # not an option, and this is within a factor of three of numpy.
    view = raw[:usable]
    planes = [view[index::16] for index in range(4)]
    out = bytearray(len(planes[0]) * 4)
    for index, plane in enumerate(planes):
        out[index::4] = plane
    return bytes(out)


def rechain_offset(cpu_offset: int) -> int:
    """Where CPU offset ``o`` lands in the emulator's linear video memory."""
    return 4 * (cpu_offset & ~3) + (cpu_offset & 3)


# ── judging a candidate ────────────────────────────────────────────────────


def coherence(page: bytes, width: int = WIDTH, stride: int = 7) -> float:
    """How much this looks like a drawn frame rather than unrelated memory.

    The fraction of sampled pixels that equal their right-hand neighbour and
    their neighbour one row down.  Both have to hold: a run of one repeated
    byte scores perfectly horizontally and vertically, but so does a cleared
    page, which is why :func:`locate` also insists on some variety.
    """
    if len(page) < width * 4:
        return 0.0
    horizontal = vertical = 0
    samples = 0
    for index in range(0, len(page) - width - 1, stride):
        if index % width == width - 1:
            continue
        samples += 1
        if page[index] == page[index + 1]:
            horizontal += 1
        if page[index] == page[index + width]:
            vertical += 1
    if not samples:
        return 0.0
    return min(horizontal, vertical) / samples


def busiest_row(page: bytes, width: int = WIDTH, height: int = HEIGHT) -> int:
    """The row of a reference with the most non-zero bytes -- the best needle."""
    best, row = -1, 0
    for index in range(height):
        ink = sum(1 for byte in page[index * width : (index + 1) * width] if byte)
        if ink > best:
            best, row = ink, index
    return row


# ── the page itself ────────────────────────────────────────────────────────


@dataclass
class Located:
    """Where a page was found, and how confident that is."""

    region_base: int
    region_size: int
    page_offset: int          # in de-interleaved (CPU) bytes
    coherence: float
    matched_reference: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "region_host_address": f"0x{self.region_base:X}",
            "region_bytes": self.region_size,
            "page_offset": self.page_offset,
            "coherence": round(self.coherence, 4),
            "matched_reference": self.matched_reference,
        }


class Framebuffer:
    """One guest's video memory, located once and then read repeatedly."""

    def __init__(self, process, width: int = WIDTH, height: int = HEIGHT) -> None:
        self.process = process
        self.width = width
        self.height = height
        self.page = width * height
        self.found: Located | None = None

    # -- locating ----------------------------------------------------------

    def _candidate_regions(self):
        for base, size, _ in self.process.regions():
            if MIN_REGION <= size <= MAX_REGION:
                yield base, size & ~0xF

    def locate(self, reference: bytes | None = None) -> Located:
        """Find the page.  With a reference this is exact; without, it scores."""
        if reference is not None and len(reference) != self.page:
            raise VgaError(
                f"a reference is {self.page} bytes, one palette index per pixel; "
                f"got {len(reference)}"
            )
        best: Located | None = None
        for base, size in self._candidate_regions():
            raw = self.process.read(base, size)
            if raw is None:
                continue
            flat = dechain(raw)
            if reference is not None:
                found = self._match_reference(flat, base, size, reference)
                if found is not None:
                    self.found = found
                    return found
                continue
            found = self._score(flat, base, size)
            if found is not None and (best is None or found.coherence > best.coherence):
                best = found
        if best is None:
            raise VgaError(
                "no video memory page found in this process.  "
                + (
                    "The reference does not appear anywhere, so the guest is not "
                    "showing that frame -- capture the reference and locate in the "
                    "same moment."
                    if reference is not None
                    else "Nothing in memory scores as a drawn frame; the guest may be "
                    "in text mode, or showing a blank screen.  Pass a reference "
                    "frame, or use dosbox_capture_screen with source='window'."
                )
            )
        self.found = best
        return best

    def _match_reference(
        self, flat: bytes, base: int, size: int, reference: bytes
    ) -> Located | None:
        row = busiest_row(reference, self.width, self.height)
        needle = reference[row * self.width : (row + 1) * self.width]
        at = flat.find(needle)
        while at >= 0:
            start = at - row * self.width
            if start >= 0 and flat[start : start + self.page] == reference:
                return Located(base, size, start, 1.0, True)
            at = flat.find(needle, at + 1)
        return None

    def _score(self, flat: bytes, base: int, size: int) -> Located | None:
        best: Located | None = None
        for index in range(MAX_PAGES):
            start = index * self.page
            if start + self.page > len(flat):
                break
            page = flat[start : start + self.page]
            distinct = len(set(page[::13]))
            if distinct < 3:  # a cleared page is coherent and says nothing
                continue
            score = coherence(page, self.width)
            if score < COHERENCE_FLOOR:
                continue
            if best is None or score > best.coherence:
                best = Located(base, size, start, score, False)
        return best

    # -- reading -----------------------------------------------------------

    def read(self, page_offset: int | None = None) -> bytes:
        """The page as palette indices, top-left first."""
        if self.found is None:
            raise VgaError("locate() has not found a page yet")
        start = self.found.page_offset if page_offset is None else page_offset
        # Read only the linear span the page occupies -- four bytes of video
        # memory per screen byte -- rather than the whole region every time.
        raw = self.process.read(
            self.found.region_base + rechain_offset(start), self.page * 4
        )
        if raw is None:
            raise VgaError("the video memory region vanished mid-read")
        page = dechain(raw)
        if len(page) < self.page:
            raise VgaError("short read from video memory")
        return page[: self.page]

    def pages(self) -> list[int]:
        """The page offsets that fit inside the located region."""
        if self.found is None:
            return []
        room = (self.found.region_size // 4) - self.found.page_offset
        return [
            self.found.page_offset + index * self.page
            for index in range(max(1, min(MAX_PAGES, room // self.page)))
        ]

    def watch(
        self, seconds: float, interval: float = 0.02, page_offset: int | None = None
    ) -> list[tuple[float, bytes]]:
        """Every distinct page during ``seconds``, with when it first appeared."""
        started = time.monotonic()
        seen: list[tuple[float, bytes]] = []
        last: bytes | None = None
        while time.monotonic() - started < seconds:
            page = self.read(page_offset)
            if page != last:
                seen.append((round(time.monotonic() - started, 3), page))
                last = page
            time.sleep(interval)
        return seen


# ── output ─────────────────────────────────────────────────────────────────


def png(page: bytes, palette: bytes | None, width: int = WIDTH, height: int = HEIGHT) -> bytes:
    """A PNG of one page.  Without a palette, the indices as grey.

    A palette is 768 bytes of six-bit DAC values, which is what the hardware
    holds and what a guest's own palette read hands back.
    """
    rows = bytearray()
    for row in range(height):
        rows.append(0)  # filter: none
        line = page[row * width : (row + 1) * width]
        if palette:
            for index in line:
                rows += bytes(value << 2 for value in palette[index * 3 : index * 3 + 3])
        else:
            for index in line:
                rows += bytes((index, index, index))

    def chunk(tag: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + tag
            + body
            + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(rows), 6))
        + chunk(b"IEND", b"")
    )


def palette_bytes(entries) -> bytes:
    """Turn a profile's 8-bit RGB triples back into six-bit DAC bytes."""
    out = bytearray()
    for red, green, blue in entries:
        out += bytes((red >> 2, green >> 2, blue >> 2))
    return bytes(out)
