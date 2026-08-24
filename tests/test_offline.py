"""Offline tests: everything that does not need a running emulator.

Every fixture here is built in memory, so the suite runs on any platform, in
CI, with no DOSBox-X and no game.  What it cannot cover is the two system calls
at the very bottom -- reading and writing another process -- and the window;
those are exercised by using the server against a real guest.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dosbox_mcp import emulator, guest as dosguest, trace as tracing, vga  # noqa: E402
from dosbox_mcp.platform import capabilities  # noqa: E402
from dosbox_mcp.profile import SEGMENT_SIZE, Profile, ProfileError, load_profile  # noqa: E402
from dosbox_mcp.session import Condition, SessionError, key_codes  # noqa: E402

from fake import FakeProcess, build_guest, draw, put_page  # noqa: E402

EXAMPLE = ROOT / "profiles" / "example.json"


# ── profiles ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def profile():
    return load_profile(EXAMPLE)


def test_the_shipped_example_profile_loads(profile):
    assert profile.name == "example"
    assert profile.marker == bytes.fromhex("6578616d706c652e64617400")
    assert profile.marker_offset == 0x0100
    assert profile.ds_segment == 0x1234
    assert profile.mouse_buttons == 0x00B2


def test_symbols_resolve_by_name_and_by_number(profile):
    assert profile.resolve("lives") == (0x1234, 1)
    assert profile.resolve("score") == (0x1236, 4)
    assert profile.resolve("0x40") == (0x40, 1)
    assert profile.resolve(64) == (64, 1)
    with pytest.raises(ProfileError):
        profile.resolve("treasury")
    with pytest.raises(ProfileError):
        profile.resolve(SEGMENT_SIZE)


def test_watch_sets_and_caves_come_through(profile):
    assert profile.watch_sets["player"] == ("lives", "score", "level")
    assert profile.caves[0]["length"] == 128
    assert profile.video_mode == "13h"


def test_a_profile_with_no_marker_is_refused(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text('{"name": "empty"}', encoding="utf-8")
    with pytest.raises(ProfileError) as failure:
        load_profile(path)
    assert "marker" in str(failure.value)


def test_checks_reject_a_segment_the_marker_alone_would_accept(profile):
    segment = bytearray(SEGMENT_SIZE)
    segment[0x0100 : 0x0100 + len(profile.marker)] = profile.marker
    assert not profile.validates(bytes(segment))          # the checks fail
    struct.pack_into("<H", segment, 0x0200, 0x0210)
    segment[0x0210:0x0215] = b"game\0"
    segment[0x0300] = 16
    assert profile.validates(bytes(segment))
    segment[0x0300] = 17                                   # past the max check
    assert not profile.validates(bytes(segment))


def test_an_anonymous_profile_names_nothing_and_validates_nothing():
    anonymous = Profile.anonymous()
    assert anonymous.anonymous_profile
    assert not anonymous.validates(bytes(SEGMENT_SIZE))
    assert anonymous.marker == b""


# ── conditions and keys ────────────────────────────────────────────────────


def test_conditions_read_little_endian_fields():
    segment = bytearray(SEGMENT_SIZE)
    struct.pack_into("<H", segment, 0x10, 0x1234)
    assert Condition(offset=0x10, size=2).value(bytes(segment)) == 0x1234
    assert Condition(offset=0x10, size=2, equals=0x1234).holds(bytes(segment))
    assert not Condition(offset=0x10, size=2, equals=0x1235).holds(bytes(segment))
    assert Condition(offset=0x10, size=2, at_least=0x1000, at_most=0x2000).holds(bytes(segment))
    assert Condition(offset=0x10, size=2, any_of=(1, 0x1234)).holds(bytes(segment))


def test_key_codes_cover_names_and_characters():
    assert key_codes("esc") == (0x1B, 0x01)
    assert key_codes("a") == (ord("a"), 0x1E)
    assert key_codes("F1") == (0x00, 0x3B)
    with pytest.raises(SessionError):
        key_codes("hyperspace")


# ── finding a guest ────────────────────────────────────────────────────────


def test_a_guest_is_found_by_its_bios_data_area():
    process, ram = build_guest()
    assert dosguest.ram_bases(process) == [ram]
    assert dosguest.find_one(process) == ram
    assert dosguest.plausible(process, ram)
    assert not dosguest.plausible(process, ram + 0x10)


def test_a_dead_interrupt_table_is_not_a_guest():
    process, ram = build_guest()
    process.write(ram + 0x21 * 4, b"\0\0\0\0")
    assert dosguest.ram_bases(process) == []


def test_a_segment_is_addressed_the_way_the_guest_addresses_it():
    process, ram = build_guest()
    target = dosguest.Guest(process=process, ram_base=ram)
    assert target.linear(0x1000, 0x0020) == ram + 0x10020
    target.write(0x1000, 0x20, b"\x34\x12")
    assert target.word(0x1000, 0x20) == 0x1234
    assert target.read(0x1000, 0x20, 2) == b"\x34\x12"
    # The same bytes, reached as a flat physical address.
    assert target.read_physical(0x10020, 2) == b"\x34\x12"
    with pytest.raises(dosguest.GuestError):
        target.read(0x1000, 0xFFFF, 2)


def test_the_interrupt_table_and_memory_size_read_back():
    process, ram = build_guest()
    target = dosguest.Guest(process=process, ram_base=ram)
    assert target.vector(0x21) == (0x00C8, 0x1234)
    assert target.memory_kb() == 640


def test_search_reports_guest_segment_and_offset():
    process, ram = build_guest()
    target = dosguest.Guest(process=process, ram_base=ram)
    target.write(0x2000, 0x0004, b"SPRITES.DBT\0")
    assert (0x2000, 0x4) in target.search(b"SPRITES.DBT")


def test_the_keyboard_ring_takes_keys_and_refuses_when_full():
    process, ram = build_guest()
    target = dosguest.Guest(process=process, ram_base=ram)
    for _ in range(15):
        target.push_key(0x20, 0x39)
    head = int.from_bytes(target.read_physical(dosguest.BDA + 0x1A, 2), "little")
    tail = int.from_bytes(target.read_physical(dosguest.BDA + 0x1C, 2), "little")
    assert head == dosguest.KEYBOARD_START and tail == dosguest.KEYBOARD_END - 2
    with pytest.raises(dosguest.GuestError) as failure:
        target.push_key(0x20, 0x39)
    assert "full" in str(failure.value)


# ── the DOS memory chain ───────────────────────────────────────────────────


def test_the_memory_chain_names_every_loaded_program():
    process, ram = build_guest(
        [
            ("COMMAND", "C:\\COMMAND.COM", 0x100),
            ("JP", "C:\\JP\\JP.EXE", 0x400),
            ("JP2D", "C:\\JP\\JP2D.EXE", 0x2000),
        ]
    )
    target = dosguest.Guest(process=process, ram_base=ram)
    chain = dosguest.memory_chain(target)
    assert [block.name for block in chain] == ["COMMAND", "JP", "JP2D"]
    assert chain[-1].last and not chain[0].last
    assert chain[1].size == 0x400 * 16

    loaded = dosguest.programs(target)
    assert [entry["name"] for entry in loaded] == ["JP2D", "JP", "COMMAND"]
    assert loaded[0]["path"] == "C:\\JP\\JP2D.EXE"
    assert loaded[0]["load_segment"] == f"{chain[2].data_segment:04X}"


def test_describe_summarises_a_guest_with_no_profile():
    process, ram = build_guest([("GAME", "C:\\GAME.EXE", 0x800)])
    summary = dosguest.describe(dosguest.Guest(process=process, ram_base=ram))
    assert summary["conventional_kb"] == 640
    assert summary["int21_vector"] == "00C8:1234"  # segment:offset, as the IVT holds it
    assert summary["programs"][0]["path"] == "C:\\GAME.EXE"


# ── the framebuffer ────────────────────────────────────────────────────────


def test_the_chain_four_interleave_round_trips():
    page = draw()
    memory = bytearray(len(page) * 4)
    process = FakeProcess(memory, base=0)
    put_page(process, 0, page)
    assert vga.dechain(bytes(memory)) == page


def test_rechain_offset_matches_the_hardware_rule():
    assert [vga.rechain_offset(o) for o in range(6)] == [0, 1, 2, 3, 16, 17]


def test_a_drawn_page_is_coherent_and_noise_is_not():
    page = draw()
    noise = bytes((index * 37 + index * index) % 256 for index in range(320 * 200))
    assert vga.coherence(page) > 0.5
    assert vga.coherence(noise) < vga.COHERENCE_FLOOR


def test_the_page_is_found_by_a_reference_and_read_back():
    page = draw()
    memory = bytearray(0x100000)
    process = FakeProcess(memory)
    put_page(process, process.base + 0x40000, page)
    buffer = vga.Framebuffer(process)
    found = buffer.locate(page)
    assert found.confirmed and found.page_offset == 0x10000
    assert buffer.read() == page


def test_a_blind_scan_finds_a_picture_and_admits_it_is_unconfirmed():
    page = draw(seed=3)
    memory = bytearray(0x100000)
    process = FakeProcess(memory)
    put_page(process, process.base, page)
    found = vga.Framebuffer(process).locate(None)
    assert found.coherence > vga.COHERENCE_FLOOR
    # Measured against a live guest, a blind scan returned a 0.999-scoring page
    # that was not the screen.  Saying so is the contract.
    assert not found.confirmed
    assert "UNCONFIRMED" in found.as_dict()["note"]


def test_a_page_at_an_unaligned_offset_is_still_found():
    """The measured live offset was 16, not 0 and not a multiple of the page."""
    page = draw(seed=5)
    memory = bytearray(0x100000)
    process = FakeProcess(memory)
    put_page(process, process.base + 16 * 4, page)
    buffer = vga.Framebuffer(process)
    found = buffer.locate(page)
    assert found.confirmed and found.page_offset == 16
    assert buffer.read() == page


def test_a_blank_page_is_not_mistaken_for_a_picture():
    memory = bytearray(0x100000)
    process = FakeProcess(memory)
    with pytest.raises(vga.VgaError):
        vga.Framebuffer(process).locate(None)


def test_png_is_written_with_or_without_a_palette():
    page = draw()
    grey = vga.png(page, None)
    coloured = vga.png(page, vga.palette_bytes([(i, i, i) for i in range(256)]))
    assert grey.startswith(b"\x89PNG\r\n\x1a\n") and coloured.startswith(b"\x89PNG")
    assert struct.unpack(">II", grey[16:24]) == (320, 200)


# ── trace caves ────────────────────────────────────────────────────────────


def test_capture_sources_parse_or_say_why_not():
    assert tracing.parse_source("ax") == ("stack", ("ax",))
    assert tracing.parse_source("bx") == ("register", ("bx",))
    assert tracing.parse_source("ds") == ("segment", ("ds",))
    assert tracing.parse_source("flags") == ("stack", ("flags",))
    assert tracing.parse_source("word:0x9D10") == ("memory", ("word", None, 0x9D10))
    assert tracing.parse_source("byte:DS:0x7A0E") == ("memory", ("byte", "ds", 0x7A0E))
    assert tracing.parse_source("word:ES:0x0004") == ("memory", ("word", "es", 4))
    with pytest.raises(tracing.TraceError):
        tracing.parse_source("rax")


def test_the_assembler_emits_the_bytes_it_claims_to():
    image, code_length, slots = tracing.assemble(["ax"], cave_at=0x0100, target_from_cave=0x0050)
    assert code_length == 24
    assert slots["__hits__"] == 0x0118 and slots["ax"] == 0x011A
    assert image.hex() == (
        "e84dff"      # call 0x0050, relative to the end of this instruction
        "9c"          # pushf   -- the callee's flags are part of its answer
        "50"          # push ax
        "55"          # push bp
        "8bec"        # mov bp, sp
        "2eff061801"  # inc word cs:[0x0118]
        "8b4602"      # mov ax, [bp+2]   -- the ax the callee returned
        "2ea31a01"    # mov cs:[0x011a], ax
        "5d58"        # pop bp / pop ax
        "9d"          # popf
        "c3"          # ret
        "00000000"    # the two data slots
    )


def test_every_capture_kind_assembles_to_its_documented_encoding():
    image, code_length, slots = tracing.assemble(
        ["bx", "ds", "sp", "word:0x9D10", "byte:ES:0x0004"], 0x0000, 0x0000
    )
    code = image[:code_length].hex()
    assert "2e891e" in code        # mov cs:[slot], bx
    assert "8cd8" in code          # mov ax, ds
    assert "8d4608" in code        # lea ax, [bp+8] -- sp as the call site had it
    assert "a1109d" in code        # mov ax, [0x9d10]
    assert "26a0040030e4" in code  # mov al, es:[4] / xor ah, ah
    assert set(slots) == {"__hits__", "bx", "ds", "sp", "word:0x9D10", "byte:ES:0x0004"}


def test_caves_are_the_longest_runs_of_zero():
    code = bytes(10) + b"\xcc" * 4 + bytes(40) + b"\xcc" + bytes(20)
    assert tracing.find_caves(code, 16) == [(14, 40), (55, 20)]
    assert tracing.find_caves(code, 64) == []


def _traceable_guest():
    """A guest whose segment 0x1000 holds a call at 0x0100 and a cave at 0x4000."""
    process, ram = build_guest()
    target = dosguest.Guest(process=process, ram_base=ram)
    # call 0x0400: +0x02FD from the end of the instruction at 0x0100.
    target.write(0x1000, 0x0100, b"\xe8\xfd\x02")
    target.write(0x1000, 0x4000, bytes(128))
    return target


def test_a_trace_displaces_the_call_and_can_be_undone():
    target = _traceable_guest()
    installed = tracing.install(
        target, 0x1000, 0x0100, ["ax", "word:0x9D10"], 0x4000, expect=b"\xe8\xfd\x02"
    )
    assert installed.target == 0x0400
    # The call site now points at the cave, and the cave calls the old target.
    assert target.read(0x1000, 0x0100, 3) == b"\xe8" + struct.pack("<H", 0x4000 - 0x0103)
    assert target.read(0x1000, 0x4000, 3) == b"\xe8" + struct.pack(
        "<H", (0x0400 - 0x4003) & 0xFFFF
    )

    reported = tracing.read(target, installed)
    assert reported["hits"] == 0 and not reported["fired"]
    assert set(reported["values"]) == {"ax", "word:0x9D10"}

    # A hit, written the way the guest would have written one.
    data_at = installed.cave_at + installed.code_length
    target.write(0x1000, data_at, struct.pack("<HHH", 3, 0x1234, 0xBEEF))
    reported = tracing.read(target, installed)
    assert reported["hits"] == 3 and reported["fired"]
    assert reported["values"]["ax"] == 0x1234
    assert reported["hex"]["word:0x9D10"] == "0xbeef"

    tracing.remove(target, installed)
    assert target.read(0x1000, 0x0100, 3) == b"\xe8\xfd\x02"
    assert target.read(0x1000, 0x4000, installed.image_length) == bytes(installed.image_length)


def test_a_trace_refuses_a_moved_target_and_a_dirty_cave():
    target = _traceable_guest()
    with pytest.raises(tracing.TraceError) as failure:
        tracing.install(target, 0x1000, 0x0100, ["ax"], 0x4000, expect=b"\xe8\x00\x00")
    assert "Nothing was written" in str(failure.value)

    target.write(0x1000, 0x4010, b"\x90")
    with pytest.raises(tracing.TraceError) as failure:
        tracing.install(target, 0x1000, 0x0100, ["ax"], 0x4000)
    assert "zero bytes" in str(failure.value)


def test_a_trace_refuses_an_instruction_that_is_not_a_near_call():
    target = _traceable_guest()
    target.write(0x1000, 0x0200, b"\x9a\x00\x00\x00\x00")  # a far call
    with pytest.raises(tracing.TraceError) as failure:
        tracing.install(target, 0x1000, 0x0200, ["ax"], 0x4000)
    assert "near call" in str(failure.value)


# ── the emulator's own configuration ───────────────────────────────────────


def test_the_captures_directory_is_relative_to_the_config_file(tmp_path):
    nested = tmp_path / "conf"
    nested.mkdir()
    config = nested / "game.conf"
    config.write_text("[dosbox]\ncaptures=shots\n", encoding="utf-8")
    assert emulator.captures_directory(config) == (nested / "shots").resolve()

    config.write_text("[dosbox]\nmachine=svga_s3\n", encoding="utf-8")
    assert emulator.captures_directory(config) == nested / "capture"


def test_recording_suffixes_are_not_all_dro():
    assert emulator.SUFFIXES["record_midi"] == (".mid", ".midi")
    assert emulator.SUFFIXES["record_wave"] == (".wav",)


# ── the platform layer ─────────────────────────────────────────────────────


def test_capabilities_answer_for_this_host():
    have = capabilities()
    assert have.platform in ("windows", "linux") or not have.memory
    assert isinstance(have.as_dict()["notes"], list)


def test_the_win32_shim_still_exports_what_scripts_import():
    from dosbox_mcp import win32

    for name in ("Process", "find_windows", "ProcessError", "Window"):
        assert hasattr(win32, name)


# ── the Linux backend, exercised from any host ─────────────────────────────


def test_the_maps_parser_keeps_guest_ram_and_drops_the_kernel_pages():
    from dosbox_mcp.platform import linux

    rows = linux.parse_maps(
        "00400000-00452000 r-xp 00000000 08:02 173521   /usr/bin/dosbox-x\n"
        "7f9c1e000000-7f9c1e200000 rw-p 00000000 00:00 0\n"
        "7f9c20000000-7f9c20021000 ---p 00000000 00:00 0\n"
        "7ffd4b7ff000-7ffd4b800000 r--p 00000000 00:00 0    [vvar]\n"
        "not a maps line at all\n"
    )
    assert [(size, readable) for _, size, readable, _ in rows] == [
        (0x52000, True),     # the binary
        (0x200000, True),    # an anonymous rw mapping -- guest RAM looks like this
        (0x21000, False),    # ---p, a guard
        (0x1000, False),     # [vvar], never read
    ]
    assert rows[0][0] == 0x00400000


def test_a_denial_explains_ptrace_scope_rather_than_printing_eperm(monkeypatch):
    from dosbox_mcp.platform import linux

    monkeypatch.setattr(linux, "ptrace_scope", lambda: 1)
    message = str(linux._denied(99, 1, "reading memory"))
    assert "ptrace_scope is 1" in message and "dosbox_launch" in message

    monkeypatch.setattr(linux, "ptrace_scope", lambda: 3)
    assert "CAP_SYS_PTRACE" in str(linux._denied(99, 13, "writing memory"))

    # Anything that is not a permission problem is reported as what it is.
    assert "denied" not in str(linux._denied(99, 3, "reading memory"))
