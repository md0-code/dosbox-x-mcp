"""Offline acceptance tests for Perception v1's pure evidence algorithms."""
from __future__ import annotations

import asyncio
import struct

import pytest

from dosbox_mcp.perception import (
    PerceptionError,
    compare_frames,
    correlate_series,
    rank_correlations,
    scan_memory_changes,
    track_regions,
)
from dosbox_mcp import server, vga


def frame(width: int, height: int, rectangles=()) -> bytes:
    pixels = bytearray(width * height)
    for x, y, box_width, box_height, colour in rectangles:
        for row in range(y, y + box_height):
            start = row * width + x
            pixels[start : start + box_width] = bytes([colour]) * box_width
    return bytes(pixels)


def transitions(frames: list[bytes], width: int, height: int, bridge_radius: int = 0):
    return [
        compare_frames(before, after, width, height, bridge_radius=bridge_radius)
        for before, after in zip(frames, frames[1:])
    ]


def test_identical_frames_have_no_change_or_regions():
    blank = frame(8, 6)
    result = compare_frames(blank, blank, 8, 6)
    assert result.changed_pixels == 0
    assert result.changed_fraction == 0
    assert result.overall_bbox is None
    assert result.regions == ()


def test_one_pixel_and_edge_corner_regions_are_exact():
    before = frame(5, 4)
    after = bytearray(before)
    after[0] = 1
    after[-1] = 2
    result = compare_frames(before, bytes(after), 5, 4, connectivity=4)
    assert result.changed_pixels == 2
    assert result.overall_bbox.as_list() == [0, 0, 5, 4]
    assert [region.bbox.as_list() for region in result.regions] == [
        [0, 0, 1, 1],
        [4, 3, 1, 1],
    ]


def test_rectangular_and_disconnected_changes_report_area_and_density():
    before = frame(12, 8)
    after = frame(12, 8, [(1, 2, 3, 2, 4), (9, 1, 2, 3, 7)])
    result = compare_frames(before, after, 12, 8)
    assert result.changed_pixels == 12
    assert [region.changed_pixels for region in result.regions] == [6, 6]
    assert all(region.density == 1 for region in result.regions)


def test_bridge_radius_joins_moving_old_and_new_footprints_without_inflating_counts():
    before = frame(16, 8, [(1, 2, 3, 3, 5)])
    after = frame(16, 8, [(6, 2, 3, 3, 5)])
    separate = compare_frames(before, after, 16, 8)
    joined = compare_frames(before, after, 16, 8, bridge_radius=1)
    assert len(separate.regions) == 2
    assert len(joined.regions) == 1
    assert joined.regions[0].bbox.as_list() == [1, 2, 8, 3]
    assert joined.regions[0].changed_pixels == 18


def test_small_noise_can_be_filtered_without_changing_total_evidence():
    before = frame(10, 6)
    after = bytearray(before)
    after[0] = 1
    for y in range(2, 4):
        for x in range(4, 7):
            after[y * 10 + x] = 2
    result = compare_frames(before, bytes(after), 10, 6, min_changed_pixels=2)
    assert result.changed_pixels == 7
    assert len(result.regions) == 1 and result.regions[0].changed_pixels == 6


@pytest.mark.parametrize(
    "before,after,width,height,message",
    [
        (b"", b"", 0, 1, "positive"),
        (b"\0", b"\0\0", 1, 1, "exactly"),
    ],
)
def test_frame_dimensions_and_lengths_are_validated(before, after, width, height, message):
    with pytest.raises(PerceptionError, match=message):
        compare_frames(before, after, width, height)


def test_one_visual_change_region_translates_with_a_stable_track_id():
    pictures = [
        frame(24, 10, [(x, 3, 3, 3, 9)])
        for x in (1, 4, 7, 10)
    ]
    result = track_regions(transitions(pictures, 24, 10, bridge_radius=1))
    assert result["track_count"] == 1
    track = result["tracks"][0]
    assert track["track_id"] == 1
    assert track["observations"] == 3
    assert track["dx"] == 6


def test_two_independent_visual_regions_remain_two_tracks():
    pictures = [
        frame(30, 15, [(2, 2, 2, 2, 3), (22, 10, 2, 2, 4)]),
        frame(30, 15, [(4, 2, 2, 2, 3), (20, 10, 2, 2, 4)]),
        frame(30, 15, [(6, 2, 2, 2, 3), (18, 10, 2, 2, 4)]),
    ]
    result = track_regions(transitions(pictures, 30, 15))
    assert result["track_count"] == 2
    assert sorted(track["dx"] for track in result["tracks"]) == [-2, 2]


def test_a_track_survives_one_transition_with_no_region():
    pictures = [
        frame(20, 10, [(2, 3, 2, 2, 5)]),
        frame(20, 10),
        frame(20, 10),
        frame(20, 10, [(5, 3, 2, 2, 5)]),
    ]
    result = track_regions(transitions(pictures, 20, 10), max_missed=1)
    assert result["track_count"] == 1
    assert result["tracks"][0]["observations"] == 2


def test_crossing_candidates_reduce_confidence_instead_of_claiming_certainty():
    pictures = [
        frame(24, 10, [(3, 3, 2, 2, 1), (17, 3, 2, 2, 2)]),
        frame(24, 10, [(8, 3, 2, 2, 1), (12, 3, 2, 2, 2)]),
        frame(24, 10, [(12, 3, 2, 2, 1), (8, 3, 2, 2, 2)]),
    ]
    result = track_regions(transitions(pictures, 24, 10), max_distance=20)
    assert result["track_count"] >= 2
    assert min(track["confidence"] for track in result["tracks"]) < 0.75


def test_a_large_screen_transition_is_flagged_and_does_not_seed_tracks():
    before = frame(20, 10)
    after = bytes([7]) * 200
    result = track_regions([compare_frames(before, after, 20, 10)])
    assert result["global_transitions"] == [1]
    assert result["track_count"] == 0


def snapshots():
    baseline = bytearray(16)
    experiment = bytearray(16)
    control = bytearray(16)
    opposite = bytearray(16)
    struct.pack_into("<H", baseline, 4, 100)
    struct.pack_into("<H", experiment, 4, 107)
    struct.pack_into("<H", control, 4, 100)
    struct.pack_into("<H", opposite, 4, 94)
    experiment[12] = 1
    control[12] = 1
    return bytes(baseline), bytes(experiment), bytes(control), bytes(opposite)


def test_memory_scanner_classifies_width_delta_control_and_opposite_direction():
    baseline, experiment, control, opposite = snapshots()
    candidates = scan_memory_changes(
        baseline,
        experiment,
        value_size=2,
        relation="increased",
        control=control,
        opposite=opposite,
        offset_base=0x100,
    )
    assert [candidate["offset"] for candidate in candidates] == [0x104]
    assert candidates[0]["delta"] == 7
    assert candidates[0]["opposite_delta"] == -6
    assert candidates[0]["score"] == 3


def test_memory_scanner_supports_exact_delta_and_unaligned_searches():
    before = bytes(8)
    after = bytearray(before)
    after[3:5] = (9).to_bytes(2, "little")
    candidates = scan_memory_changes(
        before,
        bytes(after),
        value_size=2,
        relation="delta",
        expected_delta=9,
        aligned=False,
    )
    assert candidates[0]["offset"] == 3


def test_memory_scanner_rejects_mismatched_snapshots():
    with pytest.raises(PerceptionError, match="baseline"):
        scan_memory_changes(b"\0\0", b"\0")


def test_correlations_report_raw_delta_inverse_and_fixed_scale_evidence():
    visual = [10, 12, 15, 14, 18]
    memory = [2560, 3072, 3840, 3584, 4608]
    result = correlate_series(visual, memory)
    assert result["raw_correlation"] == 1
    assert result["delta_correlation"] == 1
    assert result["fixed_scale"]["scale"] == 256

    inverse = correlate_series(visual, [-value for value in visual])
    assert inverse["direction"] == "inverse"
    assert inverse["delta_correlation"] == -1


def test_memory_correlations_are_ranked_by_numeric_evidence_only():
    result = rank_correlations(
        [0, 1, 2, 3, 4],
        {"noise": [3, 1, 4, 1, 5], "candidate": [0, 4, 8, 12, 16]},
    )
    assert result[0]["field"] == "candidate"
    assert "does not establish" in result[0]["limitation"]


def test_saved_frame_and_memory_tools_expose_the_pure_algorithms(tmp_path):
    first = frame(8, 6)
    second = frame(8, 6, [(2, 2, 2, 2, 7)])
    third = frame(8, 6, [(4, 2, 2, 2, 7)])
    paths = []
    for index, pixels in enumerate((first, second, third)):
        path = tmp_path / f"frame_{index}.bin"
        path.write_bytes(pixels)
        paths.append(str(path))
    compared = asyncio.run(
        server.dosbox_compare_frames(paths, width=8, height=6, bridge_radius=1)
    )
    assert compared["frame_count"] == 3
    assert compared["transitions"][0]["changed_pixels"] == 4
    assert compared["tracking"]["track_count"] == 1

    baseline, experiment, control, opposite = snapshots()
    snapshot_paths = []
    for name, data in zip(
        ("baseline", "experiment", "control", "opposite"),
        (baseline, experiment, control, opposite),
    ):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(data)
        snapshot_paths.append(str(path))
    scanned = asyncio.run(
        server.dosbox_scan_changes(
            snapshot_paths[0],
            snapshot_paths[1],
            value_size=2,
            relation="increased",
            control=snapshot_paths[2],
            opposite=snapshot_paths[3],
        )
    )
    assert scanned["total_candidates"] == 1
    assert scanned["candidates"][0]["offset"] == 4

    correlated = server.dosbox_correlate_memory(
        [0, 1, 2, 3], {"candidate": [0, 8, 16, 24]}
    )
    assert correlated["correlations"][0]["confidence"] == 1


class _ExperimentProfile:
    screen_width = 12
    screen_height = 8
    palette = ()
    name = "fixture"

    @staticmethod
    def resolve(offset):
        return (int(offset, 0) if isinstance(offset, str) else int(offset), 1)


class _SequenceFramebuffer:
    def __init__(self):
        self.found = vga.Located(0, 12 * 8, 0, 1.0, True, "linear")
        self.frames = [
            frame(12, 8),
            frame(12, 8, [(1, 3, 2, 2, 9)]),
            frame(12, 8, [(4, 3, 2, 2, 9)]),
        ]
        self.index = 0

    def read(self):
        result = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return result


class _ExperimentSession:
    pid = 4242
    profile = _ExperimentProfile()

    def __init__(self):
        self.buffer = _SequenceFramebuffer()
        self.memory_tick = 0

    def framebuffer(self):
        return self.buffer

    def segment(self):
        result = bytearray(0x10000)
        result[4] = self.memory_tick
        self.memory_tick += 1
        return bytes(result)

    def press(self, keys, pace):
        return [f"{key} fixture" for key in keys]


def test_act_and_observe_composes_action_frames_tracks_and_memory(monkeypatch):
    live = _ExperimentSession()
    monkeypatch.setattr(server, "_session", lambda pid: live)
    result = asyncio.run(
        server.dosbox_act_and_observe(
            server.ExperimentAction(kind="keys", keys=["right"]),
            duration_seconds=0.04,
            interval_seconds=0.01,
            memory_fields=[server.MemoryProbe(label="candidate", offset=4)],
            max_samples=10,
            max_frames=10,
        )
    )
    assert result["action"]["transport"] == "bios_keyboard_ring"
    assert "not held/raw" in result["action"]["limitation"]
    assert result["visual"]["distinct_frames"] >= 3
    assert result["visual"]["baseline_to_final"]["changed_pixels"] == 4
    assert result["memory"]["fields"]["candidate"]["delta"] > 0
    assert result["epistemic_status"].startswith("observed visual/memory evidence")
