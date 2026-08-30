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

**Finding the page.**  With a reference frame the page is found by searching
for a single row of it and confirming all 64,000 bytes, which is exact.  That
is the path to prefer, and a port under development always has a reference: its
own render of the same screen.

Without one the page is found by what makes a picture a picture -- adjacent
pixels in a drawn frame agree with each other far more often than bytes in
unrelated memory do, horizontally and vertically at once -- scanned across
*every* four-byte-aligned offset in the region.  Scanning every offset is not
optional: a real page is where the CRTC start address puts it, which is a
multiple of four bytes and nothing more.  Measured against a live DOSBox-X the
page sat at offset **16**, so a scan that assumed page alignment found nothing
and a scan that assumed row alignment would have missed it too.

**The blind scan is a hint, not an answer, and says so.**  Measured against a
live DOSBox-X running a real game it returned a page scoring 0.999 that was not
the screen: an emulator's memory holds plenty of picture-like data that is not
the picture -- decoded sprite banks, a previous frame, the game's own offscreen
buffers.  Coherence cannot tell those from the front buffer, because they are
all pictures.

So a location is either **confirmed** -- every one of the 64,000 bytes equalled
a reference -- or it is not, and :attr:`Located.confirmed` is what says which.
Prefer a reference always.  A port under development has one for free: its own
render of the same screen.  Failing that, any earlier confirmed capture of the
same screen will re-confirm it.
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

#: The CRTC start address is in dwords, so a page begins on a four-byte
#: boundary of de-interleaved memory and on no coarser one.
PAGE_ALIGNMENT = 4

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


def _agreement(flat: bytes, width: int):
    """Prefix sums of "equals the next byte" and "equals the byte a row down".

    One pass over the region buys a constant-time score for every candidate
    offset afterwards, which is what makes scanning all of them affordable.
    numpy does it quickly; ``accumulate`` does it acceptably.
    """
    try:
        import numpy as np

        data = np.frombuffer(flat, dtype="u1")
        horizontal = np.zeros(len(data) + 1, dtype="i8")
        vertical = np.zeros(len(data) + 1, dtype="i8")
        np.cumsum(data[:-1] == data[1:], out=horizontal[2:])
        np.cumsum(data[:-width] == data[width:], out=vertical[width + 1 :])
        return horizontal, vertical
    except ImportError:
        pass
    from array import array
    from itertools import accumulate, chain

    def prefix(pairs, lead: int):
        return array(
            "l",
            accumulate(chain((0,) * (lead + 1), pairs)),
        )

    horizontal = prefix(
        (1 if a == b else 0 for a, b in zip(flat[:-1], flat[1:])), 1
    )
    vertical = prefix(
        (1 if a == b else 0 for a, b in zip(flat[:-width], flat[width:])), width
    )
    return horizontal, vertical


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
    confirmed: bool           # every byte equalled a reference
    storage: str = "chain4"   # DOSBox-X keeps VGA indices chained or linear

    #: Kept for callers that spelled it the old way.
    @property
    def matched_reference(self) -> bool:
        return self.confirmed

    def as_dict(self) -> dict[str, object]:
        return {
            "region_host_address": f"0x{self.region_base:X}",
            "region_bytes": self.region_size,
            "page_offset": self.page_offset,
            "coherence": round(self.coherence, 4),
            "confirmed": self.confirmed,
            "storage": self.storage,
            "note": (
                "every byte matched the reference"
                if self.confirmed
                else "UNCONFIRMED: the most picture-like page found, which on a real "
                "guest is often a sprite bank or an offscreen buffer rather than "
                "the screen.  Pass a reference frame to confirm it."
            ),
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
                # Current DOSBox-X builds can retain the indexed draw surface
                # linearly instead of in the older chain-4 backing layout.
                # The full-frame comparison keeps this path just as strict.
                found = self._match_reference(raw, base, size, reference,
                                              storage="linear")
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
                    "in text mode, or showing a blank screen.  A reference frame is "
                    "the reliable way in -- if a port of the game is being written, "
                    "its own render of the same screen is one."
                )
            )
        self.found = best
        return best

    def _match_reference(
        self, flat: bytes, base: int, size: int, reference: bytes,
        storage: str = "chain4",
    ) -> Located | None:
        row = busiest_row(reference, self.width, self.height)
        needle = reference[row * self.width : (row + 1) * self.width]
        at = flat.find(needle)
        while at >= 0:
            start = at - row * self.width
            if start >= 0 and flat[start : start + self.page] == reference:
                return Located(base, size, start, 1.0, True, storage)
            at = flat.find(needle, at + 1)
        return None

    def _score(self, flat: bytes, base: int, size: int) -> Located | None:
        """The most picture-like page in this region, over every valid offset.

        Every four-byte-aligned offset is tried, which is only affordable
        because the two agreement tests are computed once for the whole region
        and then summed over a window in constant time.
        """
        if len(flat) < self.page + self.width + 1:
            return None
        horizontal, vertical = _agreement(flat, self.width)
        best: Located | None = None
        span = self.page
        limit = len(flat) - span
        for start in range(0, limit + 1, PAGE_ALIGNMENT):
            same_h = horizontal[start + span] - horizontal[start]
            if same_h < span * COHERENCE_FLOOR:
                continue
            same_v = vertical[start + span] - vertical[start]
            score = min(same_h, same_v) / span
            if score < COHERENCE_FLOOR:
                continue
            # Ties go to the lowest offset -- the first page.  A score has to
            # beat the incumbent by more than noise to displace it, or a
            # picture with repeating structure picks an arbitrary row.
            if best is not None and score <= best.coherence * 1.01:
                continue
            # A cleared page agrees with itself perfectly and says nothing.
            if len(set(flat[start : start + span : 13])) < 3:
                continue
            best = Located(base, size, start, score, False)
        return best

    # -- reading -----------------------------------------------------------

    def read(self, page_offset: int | None = None) -> bytes:
        """The page as palette indices, top-left first."""
        if self.found is None:
            raise VgaError("locate() has not found a page yet")
        start = self.found.page_offset if page_offset is None else page_offset
        if self.found.storage == "linear":
            raw = self.process.read(self.found.region_base + start, self.page)
            if raw is None or len(raw) < self.page:
                raise VgaError("the linear video-memory region vanished mid-read")
            return raw[: self.page]
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
        divisor = 1 if self.found.storage == "linear" else 4
        room = (self.found.region_size // divisor) - self.found.page_offset
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


# ── exact offline decoders for additional VGA layouts ────────────────────


def decode_mode_x(planes: tuple[bytes, bytes, bytes, bytes], width: int, height: int, *, start_address: int = 0, pitch: int | None = None) -> bytes:
    """Decode four unchained 256-colour planes into indexed pixels."""
    if width <= 0 or height <= 0 or width % 4:
        raise VgaError("Mode X width must be positive and divisible by four")
    plane_pitch = pitch if pitch is not None else width // 4
    if plane_pitch < width // 4 or start_address < 0:
        raise VgaError("invalid Mode X pitch or start address")
    needed = start_address + plane_pitch * height
    if len(planes) != 4 or any(len(plane) < needed for plane in planes):
        raise VgaError("Mode X plane data is shorter than the requested page")
    output = bytearray(width * height)
    for y in range(height):
        row = start_address + y * plane_pitch
        for x in range(width):
            output[y * width + x] = planes[x & 3][row + (x >> 2)]
    return bytes(output)


def decode_planar_4bpp(planes: tuple[bytes, bytes, bytes, bytes], width: int, height: int, *, start_address: int = 0, pitch: int | None = None) -> bytes:
    """Decode EGA/VGA four-plane 16-colour storage into palette indices."""
    if width <= 0 or height <= 0 or width % 8:
        raise VgaError("planar width must be positive and divisible by eight")
    row_bytes = pitch if pitch is not None else width // 8
    if row_bytes < width // 8 or start_address < 0:
        raise VgaError("invalid planar pitch or start address")
    needed = start_address + row_bytes * height
    if len(planes) != 4 or any(len(plane) < needed for plane in planes):
        raise VgaError("planar data is shorter than the requested page")
    output = bytearray(width * height)
    for y in range(height):
        row = start_address + y * row_bytes
        for x in range(width):
            mask = 0x80 >> (x & 7)
            source = row + (x >> 3)
            output[y * width + x] = sum(((planes[plane][source] & mask) != 0) << plane for plane in range(4))
    return bytes(output)


def decode_text_cells(raw: bytes, columns: int, rows: int, *, start_address: int = 0, pitch: int | None = None) -> list[list[dict[str, int]]]:
    """Decode VGA text memory while preserving character and attribute bytes."""
    if columns <= 0 or rows <= 0 or start_address < 0:
        raise VgaError("invalid text dimensions or start address")
    row_bytes = pitch if pitch is not None else columns * 2
    if row_bytes < columns * 2 or len(raw) < start_address + row_bytes * rows:
        raise VgaError("text memory is shorter than the requested page")
    result: list[list[dict[str, int]]] = []
    for y in range(rows):
        row: list[dict[str, int]] = []
        base = start_address + y * row_bytes
        for x in range(columns):
            character, attribute = raw[base + x * 2 : base + x * 2 + 2]
            row.append({"character": character, "attribute": attribute, "foreground": attribute & 0x0F, "background": (attribute >> 4) & 0x07, "blink": (attribute >> 7) & 1})
        result.append(row)
    return result


def decode_vesa_linear(raw: bytes, width: int, height: int, *, pitch: int, bytes_per_pixel: int, start_address: int = 0) -> bytes:
    """Extract exact packed VESA pixels, removing row padding only."""
    if width <= 0 or height <= 0 or bytes_per_pixel not in (1, 2, 3, 4):
        raise VgaError("invalid VESA dimensions or pixel width")
    row_bytes = width * bytes_per_pixel
    if pitch < row_bytes or start_address < 0 or len(raw) < start_address + pitch * height:
        raise VgaError("VESA storage is shorter than the requested page")
    return b"".join(raw[start_address + y * pitch : start_address + y * pitch + row_bytes] for y in range(height))


def apply_dac_timeline(initial: bytes, writes: list[dict[str, int]]) -> list[dict[str, object]]:
    """Replay timestamped six-bit DAC component writes without losing history."""
    if len(initial) != 768:
        raise VgaError("a VGA DAC contains exactly 768 component bytes")
    palette = bytearray(initial)
    result: list[dict[str, object]] = []
    previous_counter = -1
    for write in writes:
        counter = int(write["counter"])
        index = int(write["index"])
        component = int(write["component"])
        value = int(write["value"])
        if counter < previous_counter or not 0 <= index < 256 or not 0 <= component < 3 or not 0 <= value < 64:
            raise VgaError("invalid or out-of-order DAC timeline write")
        palette[index * 3 + component] = value
        result.append({"counter": counter, "index": index, "component": component, "value": value, "palette": bytes(palette)})
        previous_counter = counter
    return result


def decode_companion_snapshot(snapshot: dict[str, object]) -> dict[str, object]:
    """Decode one acknowledged companion snapshot without discarding raw data."""
    try:
        raw = bytes.fromhex(str(snapshot["raw_hex"]))
        palette = bytes.fromhex(str(snapshot["dac_hex"]))
        layout = str(snapshot["layout"])
        width = int(snapshot["width"])
        height = int(snapshot["height"])
        pitch = int(snapshot["pitch"])
        plane_pitch = int(snapshot.get("plane_pitch", pitch))
        start = int(snapshot["start_address"])
        mode_id = int(snapshot["mode_id"])
        bpp = int(snapshot["bpp"])
    except (KeyError, TypeError, ValueError) as failure:
        raise VgaError(f"malformed companion snapshot: {failure}") from None
    if len(palette) != 768:
        raise VgaError("companion snapshot has an incomplete VGA DAC")
    if layout == "dosbox_chain4_sparse":
        flat = dechain(raw)
        needed = start + width * height
        if width <= 0 or height <= 0 or len(flat) < needed:
            raise VgaError("chain-4 snapshot is shorter than the presented page")
        decoded = flat[start:needed]
        kind = "indexed"
    elif layout == "plane_interleaved":
        planes = tuple(raw[index::4] for index in range(4))
        if mode_id == 3:
            decoded = decode_mode_x(planes, width, height, start_address=start, pitch=plane_pitch)
        elif mode_id in (2, 4, 21):
            decoded = decode_planar_4bpp(planes, width, height, start_address=start, pitch=plane_pitch)
        else:
            raise VgaError(f"unsupported planar companion mode {mode_id}")
        kind = "indexed"
    elif layout == "linear_packed":
        bytes_per_pixel = 1 if bpp == 8 else 2 if bpp in (15, 16) else 3 if bpp == 24 else 4 if bpp == 32 else 0
        if not bytes_per_pixel:
            raise VgaError(f"unsupported packed pixel width {bpp}")
        decoded = decode_vesa_linear(raw, width, height, pitch=pitch, bytes_per_pixel=bytes_per_pixel, start_address=start)
        kind = "indexed" if bpp == 8 else "packed"
    elif layout == "text_plane_interleaved":
        columns = int(snapshot.get("text_columns", 0))
        rows = int(snapshot.get("text_rows", 0))
        font_height = int(snapshot.get("font_height", 0))
        cell_stride = int(snapshot.get("text_cell_stride", 8))
        if columns <= 0 or rows <= 0 or plane_pitch < columns or cell_stride < 2:
            raise VgaError("text snapshot has invalid cell dimensions or plane pitch")
        needed = (start + plane_pitch * rows) * cell_stride
        if len(raw) < needed:
            raise VgaError("text snapshot is shorter than the presented cell page")
        cell_bytes = bytearray(columns * rows * 2)
        for y in range(rows):
            target = y * columns * 2
            for x in range(columns):
                source = (start + y * plane_pitch + x) * cell_stride
                cell_bytes[target + x * 2] = raw[source]
                cell_bytes[target + x * 2 + 1] = raw[source + 1]
        decoded = decode_text_cells(bytes(cell_bytes), columns, rows)
        kind = "text"
    else:
        raise VgaError(f"unsupported companion video layout {layout!r}")
    result = {"kind": kind, "pixels": decoded, "palette": palette, "raw": raw, "width": width, "height": height, "bpp": bpp, "layout": layout}
    if layout == "text_plane_interleaved":
        result.update({"columns": columns, "rows": rows, "font_height": font_height, "char9dot": bool(snapshot.get("char9dot", False)), "text_cell_stride": cell_stride, "font_plane": raw[2::4], "font_plane_source": "raw VGA plane 2 at the acknowledged frame"})
    return result
