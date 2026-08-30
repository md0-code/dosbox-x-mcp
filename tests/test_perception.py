"""Offline acceptance tests for Perception v1's pure evidence algorithms."""
from __future__ import annotations

import asyncio
import struct

import pytest

from dosbox_mcp.perception import (
    PerceptionError,
    compare_frames,
    correlate_series,
    detect_translation,
    infer_screen_regions,
    rank_control_responses,
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


def shifted(
    source: bytes, width: int, height: int, dx: int, dy: int, fill: int = 0
) -> bytes:
    result = bytearray([fill]) * (width * height)
    for y in range(height):
        for x in range(width):
            target_x, target_y = x + dx, y + dy
            if 0 <= target_x < width and 0 <= target_y < height:
                result[target_y * width + target_x] = source[y * width + x]
    return bytes(result)


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


def test_global_translation_and_local_residual_are_separated():
    width, height = 32, 20
    before = bytes((x * 17 + y * 31 + (x * y) % 19) % 251 for y in range(height) for x in range(width))
    after = bytearray(shifted(before, width, height, 3, -2))
    for y in range(8, 10):
        for x in range(14, 17):
            after[y * width + x] = 255
    result = detect_translation(
        before,
        bytes(after),
        width,
        height,
        max_dx=5,
        max_dy=4,
        sample_stride=2,
    )
    assert result["best_translation"] == [3, -2]
    assert result["match_ratio"] > 0.98
    assert result["confidence"] > 0.5
    assert result["residual"]["changed_pixels"] == 6
    assert result["residual"]["region_count"] == 1


def test_repeated_flat_frames_report_translation_ambiguity():
    before = bytes([7]) * (20 * 12)
    result = detect_translation(before, before, 20, 12, max_dx=3, max_dy=3)
    assert result["best_translation"] == [0, 0]
    assert result["ambiguity_margin"] == 0
    assert result["confidence"] == 0


def test_screen_fixed_and_changing_candidates_accumulate_across_scrolls():
    width, height = 28, 14
    world = bytes((x * 13 + y * 29 + x * y) % 199 for y in range(height) for x in range(width))
    frames = []
    for episode in range(4):
        pixels = bytearray(shifted(world, width, height, episode, 0))
        for y in range(2, 4):
            for x in range(6, 10):
                pixels[y * width + x] = 250
        for y in range(9, 11):
            for x in range(20, 23):
                pixels[y * width + x] = 210 + episode
        frames.append(bytes(pixels))
    result = infer_screen_regions(
        frames,
        [(1, 0), (1, 0), (1, 0)],
        width,
        height,
        min_episodes=3,
        min_region_pixels=2,
    )
    categories = result["categories"]
    assert categories["globally_translating"]["pixels"] > width * height // 2
    assert any(
        region["bbox"] == [6, 2, 1, 2]
        for region in categories["screen_fixed_candidate"]["regions"]
    )
    assert categories["uncertain"]["pixels"] >= 6  # flat overlay interior is ambiguous
    assert any(
        region["bbox"] == [20, 9, 4, 2]
        for region in categories["frequently_changing_fixed_position_candidate"]["regions"]
    )


def test_screen_region_reports_cap_components_without_losing_totals():
    width, height = 12, 4
    world = bytes((x + y * width) % 200 for y in range(height) for x in range(width))
    frames = []
    for episode in range(3):
        pixels = bytearray(shifted(world, width, height, episode, 0))
        for x in (2, 6, 10):
            pixels[width + x] = 250
        frames.append(bytes(pixels))
    result = infer_screen_regions(
        frames,
        [(1, 0), (1, 0)],
        width,
        height,
        min_episodes=2,
        min_region_pixels=1,
        max_regions_per_category=1,
    )
    fixed = result["categories"]["screen_fixed_candidate"]
    assert fixed["region_count"] > 1
    assert fixed["regions_truncated"] is True
    assert len(fixed["regions"]) == 1


def test_control_response_ranking_subtracts_idle_and_rewards_repeatable_direction():
    trials = [
        {"action": "idle", "control": True, "tracks": [{"track_id": 1, "dx": 1, "dy": 0, "confidence": 0.8, "observations": 3}]},
        {"action": "idle", "control": True, "tracks": [{"track_id": 2, "dx": -1, "dy": 0, "confidence": 0.8, "observations": 3}]},
        {"action": "right", "tracks": [{"track_id": 4, "dx": 8, "dy": 1, "confidence": 0.9, "observations": 4}]},
        {"action": "right", "tracks": [{"track_id": 7, "dx": 9, "dy": 0, "confidence": 0.85, "observations": 4}]},
        {"action": "random", "tracks": [{"track_id": 8, "dx": 5, "dy": 0, "confidence": 0.5, "observations": 2}]},
        {"action": "random", "tracks": [{"track_id": 9, "dx": -5, "dy": 0, "confidence": 0.5, "observations": 2}]},
    ]
    ranked = rank_control_responses(trials)
    assert ranked[0]["action"] == "right"
    assert ranked[0]["direction_consistency"] > 0.99
    assert ranked[0]["response_excess"] > 7
    assert ranked[1]["direction_consistency"] == 0


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


def test_compare_frames_optionally_reports_global_motion(tmp_path):
    width, height = 24, 16
    before = bytes((x * 11 + y * 23 + x * y) % 251 for y in range(height) for x in range(width))
    after = shifted(before, width, height, -2, 1)
    paths = []
    for name, data in (("before", before), ("after", after)):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(data)
        paths.append(str(path))
    result = asyncio.run(
        server.dosbox_compare_frames(
            paths,
            width=width,
            height=height,
            detect_global_motion=True,
            max_translation_x=4,
            max_translation_y=4,
        )
    )
    assert result["transitions"][0]["global_motion"]["best_translation"] == [-2, 1]


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

    def hold_keys(self, keys, seconds):
        return {"keys": keys, "duration_seconds": seconds, "released": True}


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


def test_act_and_observe_accepts_authentic_held_keys(monkeypatch):
    live = _ExperimentSession()
    monkeypatch.setattr(server, "_session", lambda pid: live)
    result = asyncio.run(
        server.dosbox_act_and_observe(
            server.ExperimentAction(
                kind="hold_keys", keys=["right", "space"], duration_seconds=0.02
            ),
            duration_seconds=0.04,
            interval_seconds=0.01,
            max_samples=10,
            max_frames=10,
        )
    )
    assert result["action"]["transport"] == "dosbox_x_keyboard_hook_v1"
    assert result["action"]["released"] is True


class _BranchSession:
    pid = 4242

    def __init__(self, directory):
        self.state_directory = directory
        self.ds_base = 123
        self.ram_base = 456
        self.releases = 0
        self.locates = 0

    def release_all_keys(self):
        self.releases += 1

    def locate(self, rescan=False):
        self.locates += 1


def test_try_actions_restores_each_branch_and_the_final_baseline(monkeypatch, tmp_path):
    live = _BranchSession(tmp_path)
    state = tmp_path / "100.sav"
    state.write_bytes(b"state")
    loads = []

    monkeypatch.setattr(server, "_session", lambda pid: live)
    monkeypatch.setattr(server.host, "require", lambda capability: None)
    monkeypatch.setattr(server, "_locate_framebuffer", lambda *args: None)
    monkeypatch.setattr(
        server.emulator,
        "save_branch_state",
        lambda pid, directory: {"state_file": str(state), "bytes": 5},
    )
    monkeypatch.setattr(
        server.emulator,
        "load_branch_state",
        lambda pid, path, settle: loads.append((path, settle)) or {"loaded": True},
    )

    async def observe(**kwargs):
        return {"action": kwargs["action"].kind, "stem": kwargs["stem"]}

    monkeypatch.setattr(server, "dosbox_act_and_observe", observe)
    result = asyncio.run(
        server.dosbox_try_actions(
            [
                server.BranchTrial(label="idle", action=server.ExperimentAction(kind="idle")),
                server.BranchTrial(
                    label="right",
                    action=server.ExperimentAction(
                        kind="hold_keys", keys=["right"], duration_seconds=0.1
                    ),
                ),
            ],
            duration_seconds=0.2,
            stem="branches/trial",
        )
    )
    assert [entry["label"] for entry in result["trials"]] == ["idle", "right"]
    assert result["restore_count"] == 3
    assert len(loads) == 3
    assert result["baseline_restored"] is True
    assert live.ds_base == 0 or live.locates == 3


def test_try_actions_restores_after_a_failed_trial(monkeypatch, tmp_path):
    live = _BranchSession(tmp_path)
    state = tmp_path / "100.sav"
    state.write_bytes(b"state")
    loads = []
    monkeypatch.setattr(server, "_session", lambda pid: live)
    monkeypatch.setattr(server.host, "require", lambda capability: None)
    monkeypatch.setattr(server, "_locate_framebuffer", lambda *args: None)
    monkeypatch.setattr(
        server.emulator,
        "save_branch_state",
        lambda pid, directory: {"state_file": str(state), "bytes": 5},
    )
    monkeypatch.setattr(
        server.emulator,
        "load_branch_state",
        lambda *args: loads.append(args) or {"loaded": True},
    )

    async def fail(**kwargs):
        raise RuntimeError("trial failed")

    monkeypatch.setattr(server, "dosbox_act_and_observe", fail)
    with pytest.raises(RuntimeError, match="trial failed"):
        asyncio.run(
            server.dosbox_try_actions(
                [server.BranchTrial(label="bad", action=server.ExperimentAction())]
            )
        )
    assert len(loads) == 2


def test_probe_controls_builds_idle_repeats_and_ranks_visual_response(monkeypatch):
    async def branched(**kwargs):
        branches = []
        for trial in kwargs["trials"]:
            moving = trial.label.startswith("right")
            tracks = (
                [{"track_id": 1, "dx": 8, "dy": 0, "confidence": 0.9, "observations": 4}]
                if moving
                else [{"track_id": 1, "dx": 1, "dy": 0, "confidence": 0.8, "observations": 3}]
            )
            branches.append(
                {
                    "label": trial.label,
                    "evidence": {
                        "visual": {
                            "tracking": {"tracks": tracks},
                            "baseline_to_final": {"changed_pixels": 4},
                        },
                        "memory": {"fields": {}},
                    },
                }
            )
        return {
            "pid": 4242,
            "baseline_state": {"state_file": "100.sav"},
            "baseline_restored": True,
            "trials": branches,
        }

    monkeypatch.setattr(server, "dosbox_try_actions", branched)
    result = asyncio.run(
        server.dosbox_probe_controls(
            ["right"], hold_seconds=0.1, observation_seconds=0.2, repeats=2
        )
    )
    assert [item["label"] for item in result["trial_summaries"]] == [
        "idle_r1",
        "idle_r2",
        "right_r1",
        "right_r2",
    ]
    assert result["ranking"][0]["action"] == "right"
    assert result["baseline_restored"] is True


def test_rank_control_responses_prefers_global_motion_when_available():
    trials = [
        {
            "action": "idle",
            "control": True,
            "tracks": [{"track_id": 8, "dx": 100, "dy": 0, "confidence": 1}],
            "global_motion": {"best_translation": [1, 0], "confidence": 0.9},
        },
        {
            "action": "right",
            "control": False,
            "tracks": [],
            "global_motion": {"best_translation": [-8, 0], "confidence": 0.9},
        },
        {
            "action": "right",
            "control": False,
            "tracks": [],
            "global_motion": {"best_translation": [-7, 0], "confidence": 0.8},
        },
    ]
    ranked = rank_control_responses(trials)
    assert ranked[0]["action"] == "right"
    assert ranked[0]["response_excess"] == 6.5
    assert all(
        item["response_kind"] == "global_translation"
        for item in ranked[0]["selected_tracks"]
    )


def test_probe_controls_rejects_unknown_raw_key():
    with pytest.raises(server.SessionError, match="unknown raw key"):
        asyncio.run(server.dosbox_probe_controls(["not-a-key"]))
