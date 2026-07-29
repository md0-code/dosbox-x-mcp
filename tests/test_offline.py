"""Offline tests: everything that does not need a running emulator."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dosbox_mcp.profile import SEGMENT_SIZE, ProfileError, load_profile  # noqa: E402
from dosbox_mcp.session import Condition, key_codes, SessionError  # noqa: E402

PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "openjp.json"


@pytest.fixture(scope="module")
def profile():
    return load_profile(PROFILE)


def test_profile_loads_marker_and_palette(profile):
    assert len(profile.marker) == 0x64
    assert profile.marker_offset == 0x5BDE
    assert len(profile.palette) == 256
    assert profile.mouse_buttons == 0xB2
    assert profile.ds_segment == 0x168A


def test_symbols_resolve_by_name_and_number(profile):
    assert profile.resolve("treasury") == (0x634A, 2)
    assert profile.resolve("0x634A") == (0x634A, 1)
    assert profile.resolve(0x634A) == (0x634A, 1)
    with pytest.raises(ProfileError, match="Known symbols"):
        profile.resolve("no_such_thing")
    with pytest.raises(ProfileError, match="outside"):
        profile.resolve(SEGMENT_SIZE)


def test_reference_dump_validates_and_a_corrupted_one_does_not(profile):
    dump = Path(__file__).resolve().parents[2] / "tools" / "ground_truth" / "turn0_ds_168A.bin"
    segment = dump.read_bytes()
    assert profile.validates(segment)

    # Break the region-name pointer: the marker still matches, the checks do not.
    broken = bytearray(segment)
    broken[0x5222:0x5224] = (0x1234).to_bytes(2, "little")
    assert not profile.validates(bytes(broken))

    # Break a difficulty byte's domain.
    broken = bytearray(segment)
    broken[0x0490] = 9
    assert not profile.validates(bytes(broken))

    assert not profile.validates(segment[:-1])


def test_conditions_read_little_endian_fields():
    segment = bytearray(SEGMENT_SIZE)
    segment[0x634A:0x634C] = (9000).to_bytes(2, "little")

    assert Condition(0x634A, 2, equals=9000).holds(segment)
    assert not Condition(0x634A, 2, equals=9001).holds(segment)
    assert Condition(0x634A, 2, at_least=8000, at_most=9000).holds(segment)
    assert Condition(0x634A, 2, not_equals=0).holds(segment)
    assert Condition(0x634A, 2, any_of=(1, 9000)).holds(segment)
    assert not Condition(0x634A, 2, any_of=(1, 2)).holds(segment)
    assert "DS:0x634a/2" in Condition(0x634A, 2, equals=9000).describe()


def test_key_codes_cover_names_and_characters():
    assert key_codes("space") == (0x20, 0x39)
    assert key_codes("enter") == (0x0D, 0x1C)
    assert key_codes("g") == (ord("g"), 0x22)
    assert key_codes("a") == (ord("a"), 0x1E)
    assert key_codes("1") == (ord("1"), 0x02)
    assert key_codes("F1") == (0x00, 0x3B)
    with pytest.raises(SessionError, match="unknown key"):
        key_codes("hyperspace")
