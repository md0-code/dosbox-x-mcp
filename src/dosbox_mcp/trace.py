"""Watch a running guest's code without a debugger and without patching a file.

A `CALL` in the guest's live code segment is redirected into a stretch of
zero-filled padding -- a cave.  The cave calls the displaced target, preserves
the flags it returned, copies chosen registers and memory words into slots
after its own code, and returns.  The host then reads those slots whenever it
likes.

Nothing on disk changes.  The emulator dies with the run and takes the patch
with it, so a shipped executable stays byte-identical and a scratch drive stays
clean.  There is no breakpoint, no single-stepping and no pause: the guest runs
at full speed and does not know it is being watched, which matters when what is
being measured is timing.

**Two guards, both earned.**  The bytes at the call site are read back and must
be the instruction that was expected, so a trace never lands on a moved target;
and the cave must be genuinely zero, so a trace never overwrites code that a
different build put there.  Either check failing is a refusal, not a warning.

**One caveat worth stating.**  Memory captures are read through whatever ``DS``
holds when the cave runs.  For the overwhelmingly common case -- a game with
one data segment it keeps in ``DS`` -- that is exactly right.  For a routine
that has switched ``DS``, capture ``ds`` alongside and check it.
"""
from __future__ import annotations

import re
import struct
import time
from dataclasses import dataclass, field

WORD = 2

#: `mov r16, r/m16` register field codes, which are also `mov r/m16, r16`'s.
REGISTERS = {"ax": 0, "cx": 1, "dx": 2, "bx": 3, "sp": 4, "bp": 5, "si": 6, "di": 7}
SEGMENT_REGISTERS = {"es": 0, "cs": 1, "ss": 2, "ds": 3}
OVERRIDES = {"es": 0x26, "cs": 0x2E, "ss": 0x36, "ds": 0x3E}

#: Sources that come off the stack the cave itself built, not out of a register.
STACK_SOURCES = {"bp": 0x00, "ax": 0x02, "flags": 0x04}

NEAR_CALL = 0xE8
CALL_LENGTH = 3

MEMORY_SOURCE = re.compile(
    r"^(?P<width>word|byte)\s*:\s*(?:(?P<segment>[a-z]{2})\s*:\s*)?(?P<offset>0[xX][0-9a-fA-F]+|\d+)$"
)


class TraceError(RuntimeError):
    """A trace could not be assembled, installed, read or removed."""


# ── the emitter ────────────────────────────────────────────────────────────


def _word(value: int) -> bytes:
    return struct.pack("<H", value & 0xFFFF)


def _store_ax(slot: int) -> bytes:
    """``mov cs:[slot], ax`` -- the cave keeps its results in its own segment."""
    return b"\x2e\xa3" + _word(slot)


def _store_register(name: str, slot: int) -> bytes:
    """``mov cs:[slot], r16`` for a register the cave has not touched."""
    modrm = (REGISTERS[name] << 3) | 0b110  # mod=00, rm=110: disp16
    return b"\x2e\x89" + bytes((modrm,)) + _word(slot)


def _load_from_stack(displacement: int) -> bytes:
    """``mov ax, [bp+disp8]`` -- reach a value the cave pushed."""
    return b"\x8b\x46" + bytes((displacement,))


def _load_segment_register(name: str) -> bytes:
    """``mov ax, sreg``."""
    return b"\x8c" + bytes((0xC0 | (SEGMENT_REGISTERS[name] << 3),))


def _load_memory(width: str, segment: str | None, offset: int) -> bytes:
    """``mov ax, [imm16]`` or ``mov al, [imm16]``, with an optional override."""
    prefix = b"" if segment in (None, "ds") else bytes((OVERRIDES[segment],))
    if width == "word":
        return prefix + b"\xa1" + _word(offset)
    return prefix + b"\xa0" + _word(offset) + b"\x30\xe4"  # + xor ah,ah


def parse_source(text: str) -> tuple[str, tuple]:
    """Turn one capture specification into something :func:`assemble` can emit.

    Accepts a register (``ax``, ``bx``, ``cx``, ``dx``, ``si``, ``di``, ``bp``,
    ``sp``), a segment register (``ds``, ``es``, ``ss``, ``cs``), ``flags``, or
    a memory word or byte: ``word:0x9D10``, ``byte:DS:0x7A0E``,
    ``word:ES:0x0004``.  A memory source with no segment named uses ``DS``.
    """
    lowered = text.strip().lower()
    if lowered in STACK_SOURCES:
        return "stack", (lowered,)
    if lowered == "sp":
        return "sp", ()
    if lowered in REGISTERS:
        return "register", (lowered,)
    if lowered in SEGMENT_REGISTERS:
        return "segment", (lowered,)
    match = MEMORY_SOURCE.match(lowered)
    if match:
        segment = match.group("segment")
        if segment is not None and segment not in OVERRIDES:
            raise TraceError(
                f"{segment!r} is not a segment register; use ds, es, ss or cs"
            )
        return "memory", (match.group("width"), segment, int(match.group("offset"), 0))
    raise TraceError(
        f"{text!r} is not a capture source.  Use a register (ax bx cx dx si di bp sp), "
        f"a segment register (ds es ss cs), flags, or memory: word:0x1234, "
        f"byte:DS:0x1234, word:ES:0x1234."
    )


def assemble(
    captures: list[str], cave_at: int, target_from_cave: int
) -> tuple[bytes, int, dict[str, int]]:
    """Build a cave.  Returns ``(image, code_length, slot offsets by name)``.

    The image is code followed by its own data, so it is written in one go and
    the slots are addressed as absolute offsets in the code segment -- which is
    what ``mov cs:[imm16], ax`` needs and why the cave has to know where it
    will live before it can be built.
    """
    sources = [(name, *parse_source(name)) for name in captures]
    # The data area sits directly after the code, so its address depends on the
    # code's length, which depends on the addresses.  Two passes settle it: the
    # first measures with placeholder slots, the second emits for real.
    for _ in range(2):
        data_at = cave_at + _measure(sources)
        slots = {"__hits__": data_at}
        for index, (name, _, _) in enumerate(sources):
            slots[name] = data_at + WORD * (index + 1)
        image = _emit(sources, cave_at, target_from_cave, slots)
        if cave_at + len(image) == data_at:
            break
    code_length = len(image)
    return image + bytes(WORD * (len(sources) + 1)), code_length, slots


def _measure(sources) -> int:
    """The code length, which the two-pass build needs before the addresses."""
    return len(_emit(sources, 0, 0, {name: 0 for name, _, _ in sources} | {"__hits__": 0}))


def _emit(sources, cave_at: int, target: int, slots: dict[str, int]) -> bytes:
    out = bytearray()
    # The displaced call, first: everything captured is what it left behind.
    out += b"\xe8" + _word((target - (cave_at + CALL_LENGTH)) & 0xFFFF)
    out += b"\x9c"      # pushf   -- the callee's flags are part of its answer
    out += b"\x50"      # push ax
    out += b"\x55"      # push bp
    out += b"\x8b\xec"  # mov bp, sp
    out += b"\x2e\xff\x06" + _word(slots["__hits__"])  # inc word cs:[hits]
    for name, kind, arguments in sources:
        slot = slots[name]
        if kind == "stack":
            out += _load_from_stack(STACK_SOURCES[arguments[0]]) + _store_ax(slot)
        elif kind == "sp":
            # SP as the call site had it: bp, ax and flags pushed here, plus the
            # return address the cave's own call pushed.
            out += b"\x8d\x46\x08" + _store_ax(slot)  # lea ax,[bp+8]
        elif kind == "register":
            out += _store_register(arguments[0], slot)
        elif kind == "segment":
            out += _load_segment_register(arguments[0]) + _store_ax(slot)
        else:
            out += _load_memory(*arguments) + _store_ax(slot)
    out += b"\x5d"      # pop bp
    out += b"\x58"      # pop ax
    out += b"\x9d"      # popf
    out += b"\xc3"      # ret
    return bytes(out)


# ── caves ──────────────────────────────────────────────────────────────────


def find_caves(code: bytes, needed: int, limit: int = 8) -> list[tuple[int, int]]:
    """Runs of zero long enough to hold a cave, as ``(offset, length)``.

    Padding between routines and between a program's segments is the usual
    supply.  Longest first, because the longest run is the one least likely to
    be something a build put there deliberately.
    """
    runs: list[tuple[int, int]] = []
    start = None
    for index, byte in enumerate(code):
        if byte == 0:
            if start is None:
                start = index
        elif start is not None:
            if index - start >= needed:
                runs.append((start, index - start))
            start = None
    if start is not None and len(code) - start >= needed:
        runs.append((start, len(code) - start))
    runs.sort(key=lambda run: -run[1])
    return runs[:limit]


# ── installed traces ───────────────────────────────────────────────────────


@dataclass
class Trace:
    """One installed trace, and everything needed to read or undo it."""

    handle: str
    segment: int
    call_at: int
    cave_at: int
    original: bytes
    image_length: int
    code_length: int
    slots: dict[str, int]
    captures: list[str]
    target: int
    installed_at: float = field(default_factory=time.time)

    def describe(self) -> dict[str, object]:
        return {
            "handle": self.handle,
            "call_site": f"{self.segment:04X}:{self.call_at:04X}",
            "displaced_target": f"{self.segment:04X}:{self.target:04X}",
            "cave": f"{self.segment:04X}:{self.cave_at:04X}",
            "cave_bytes": self.image_length,
            "captures": list(self.captures),
        }


def install(
    guest,
    segment: int,
    call_at: int,
    captures: list[str],
    cave_at: int,
    expect: bytes | None = None,
    handle: str | None = None,
) -> Trace:
    """Redirect one near `CALL` through a cave that records what it wants."""
    if not captures:
        raise TraceError("a trace with no captures would record only its hit count")
    shipped = guest.read(segment, call_at, CALL_LENGTH)
    if expect is not None and shipped != expect:
        raise TraceError(
            f"{segment:04X}:{call_at:04X} holds {shipped.hex()}, not the expected "
            f"{expect.hex()}.  The build is not the one this trace was written for, "
            f"or the address is wrong.  Nothing was written."
        )
    if shipped[0] != NEAR_CALL:
        raise TraceError(
            f"{segment:04X}:{call_at:04X} is {shipped.hex()}, not a near call "
            f"(E8 rel16).  Only a near call can be displaced this way: it is the "
            f"one instruction whose target can be recomputed without changing its "
            f"length."
        )
    relative = struct.unpack_from("<h", shipped, 1)[0]
    target = (call_at + CALL_LENGTH + relative) & 0xFFFF

    image, code_length, slots = assemble(captures, cave_at, target)
    existing = guest.read(segment, cave_at, len(image))
    if existing != bytes(len(image)):
        raise TraceError(
            f"the cave at {segment:04X}:{cave_at:04X} is not {len(image)} zero bytes "
            f"-- it holds {existing[:16].hex()}...  Refusing to overwrite something "
            f"that may be code or data.  Use dosbox_find_cave to pick a run of zeros."
        )

    guest.write(segment, cave_at, image)
    displacement = (cave_at - (call_at + CALL_LENGTH)) & 0xFFFF
    guest.write(segment, call_at, bytes((NEAR_CALL,)) + _word(displacement))
    return Trace(
        handle=handle or f"{segment:04X}_{call_at:04X}",
        segment=segment,
        call_at=call_at,
        cave_at=cave_at,
        original=shipped,
        image_length=len(image),
        code_length=code_length,
        slots=slots,
        captures=list(captures),
        target=target,
    )


def read(guest, trace: Trace) -> dict[str, object]:
    """The slots as they stand now, plus how many times the cave has run."""
    data_at = trace.cave_at + trace.code_length
    raw = guest.read(trace.segment, data_at, trace.image_length - trace.code_length)
    words = struct.unpack_from("<" + "H" * (len(raw) // WORD), raw)
    hits = words[0]
    values = {name: words[index + 1] for index, name in enumerate(trace.captures)}
    return {
        "handle": trace.handle,
        "hits": hits,
        "fired": hits > 0,
        "values": values,
        "hex": {name: f"{value:#06x}" for name, value in values.items()},
    }


def remove(guest, trace: Trace) -> None:
    """Put the call site back, then blank the cave.

    In that order, and with a pause between: the guest may be inside the cave
    at this moment, and restoring the call site first means nothing new can
    enter it.
    """
    guest.write(trace.segment, trace.call_at, trace.original)
    time.sleep(0.05)
    guest.write(trace.segment, trace.cave_at, bytes(trace.image_length))
