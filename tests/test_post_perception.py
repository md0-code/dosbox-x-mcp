from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosbox_mcp import bundle, companion, perception, quality, qualification, vga, writewatch  # noqa: E402


def shifted(frame: bytes, width: int, height: int, dx: int, dy: int) -> bytes:
    out = bytearray(width * height)
    for y in range(max(0, dy), min(height, height + dy)):
        for x in range(max(0, dx), min(width, width + dx)):
            out[y * width + x] = frame[(y - dy) * width + x - dx]
    return bytes(out)


def test_protocol_v2_framing_rejects_truncation_and_malformed_json():
    message = {"protocol": 2, "request_id": "abc", "payload": {"x": 1}}
    encoded = companion.encode_message(message)
    assert companion.decode_message(encoded) == message
    with pytest.raises(companion.CompanionError, match="declares"):
        companion.decode_message(encoded[:-1])
    with pytest.raises(companion.CompanionError, match="malformed"):
        companion.decode_message(struct.pack("<I", 1) + b"{")


def test_companion_client_rejects_stale_event_sequences(monkeypatch):
    responses = []

    class Stream:
        def __init__(self, reply):
            self.reply = reply
            self.written = b""

        def __enter__(self): return self
        def __exit__(self, *args): return None
        def write(self, data):
            self.written += data
            request = companion.decode_message(data)
            self.reply["request_id"] = request["request_id"]
        def read(self, count):
            encoded = companion.encode_message(self.reply)
            part, self.reply["_encoded"] = encoded[:count], encoded[count:]
            self.read = lambda size: self.reply["_encoded"][:size]
            return part

    client = companion.CompanionClient(7, "token")
    base = {"protocol": 2, "status": "ok", "session_id": "s", "result": {}, "events": [{"session_id": "s", "sequence": 1, "counters": {"pic_us": 1, "video_frame": 1, "cpu_cycles": 1}}]}

    class MemoryStream:
        def __init__(self, reply): self.reply, self.buffer = reply, b""
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def write(self, data):
            request = companion.decode_message(data)
            reply = dict(self.reply, request_id=request["request_id"])
            self.buffer = companion.encode_message(reply)
        def read(self, size):
            part, self.buffer = self.buffer[:size], self.buffer[size:]
            return part

    monkeypatch.setattr(client, "_connect", lambda timeout=15: MemoryStream(base))
    client.request("x")
    with pytest.raises(companion.CompanionError, match="stale"):
        client.request("x")


def test_atomic_bundle_round_trip_detects_tampering_and_redacts_paths(tmp_path):
    destination = tmp_path / "experiment"
    with bundle.BundleWriter(destination, {"config": r"C:\private\dosbox.conf"}) as writer:
        writer.add_json("timeline", "timeline.json", {"clock": "video_frame"})
        writer.add_bytes("frame", "observations/frame.bin", b"pixels")
        manifest = writer.finalize()
    assert manifest["finalized"] is True
    assert "<machine-path>" in json.loads((destination / "manifest.json").read_text())["metadata"]["config"]
    assert bundle.verify(destination)["valid"] is True
    (destination / "observations/frame.bin").write_bytes(b"changed")
    result = bundle.verify(destination)
    assert result["valid"] is False
    assert any("mismatch" in problem for problem in result["problems"])
    with pytest.raises(bundle.BundleError, match="already exists"):
        with bundle.BundleWriter(destination, {}):
            pass


def test_partial_bundle_is_removed_and_never_finalized(tmp_path):
    destination = tmp_path / "partial"
    with pytest.raises(RuntimeError):
        with bundle.BundleWriter(destination, {}) as writer:
            writer.add_bytes("x", "x.bin", b"x")
            raise RuntimeError("interrupt")
    assert not destination.exists()
    assert not list(tmp_path.glob(".partial.*"))


def test_quality_gates_distinguish_valid_ambiguous_invalid_and_incomplete():
    valid = quality.evaluate(exact_framebuffer=True, baseline_hashes=["a", "a"], restore_hash="a", acknowledged_input=True, acknowledged_restore=True, transition_fraction=0.1, repeats=2, idle_repeats=2, camera_confidence=0.8, cleanup_released=True)
    assert valid["state"] == "valid" and valid["notebook_update_allowed"]
    ambiguous = quality.evaluate(exact_framebuffer=True, baseline_hashes=["a", "a"], restore_hash="a", acknowledged_input=True, acknowledged_restore=True, transition_fraction=0.1, repeats=2, idle_repeats=2, camera_confidence=0.1, cleanup_released=True)
    assert ambiguous["state"] == "valid_with_ambiguity"
    invalid = quality.evaluate(exact_framebuffer=False, baseline_hashes=["a", "b"], restore_hash="x", acknowledged_input=False, acknowledged_restore=False, transition_fraction=0.9, repeats=1, idle_repeats=1, camera_confidence=0.8, cleanup_released=False)
    assert invalid["state"] == "invalid" and not invalid["notebook_update_allowed"]
    incomplete = quality.evaluate(exact_framebuffer=None, baseline_hashes=[], restore_hash=None, acknowledged_input=None, acknowledged_restore=None, transition_fraction=None, repeats=2, idle_repeats=2, camera_confidence=None, cleanup_released=None)
    assert incomplete["state"] == "incomplete"


def test_multi_frame_camera_consensus_path_and_exact_loop_closure():
    width, height = 32, 20
    base = bytes((x * 13 + y * 29 + x * y) % 251 for y in range(height) for x in range(width))
    frames = [base, shifted(base, width, height, 2, 0), shifted(base, width, height, 4, 0), base]
    result = perception.estimate_camera_path(frames, width, height, max_dx=6, max_dy=2, consensus_window=1, minimum_confidence=0.1)
    assert len(result["pairwise"][0]["top_candidates"]) >= 2
    assert result["cumulative_path"][2] == [4, 0]
    assert result["loop_closures"][0]["from_frame"] == 0
    assert result["loop_closures"][0]["to_frame"] == 3


def test_camera_mask_gap_recovery_and_placement_provenance():
    width, height = 32, 20
    base = bytes((x * 17 + y * 31 + x * y) % 251 for y in range(height) for x in range(width))
    mask = bytes(1 if x < 4 else 0 for y in range(height) for x in range(width))
    frames = [
        base,
        shifted(base, width, height, 2, 0),
        bytes(width * height),
        shifted(base, width, height, 6, 0),
        shifted(base, width, height, 8, 0),
    ]
    result = perception.estimate_camera_path(
        frames, width, height, max_dx=4, max_dy=1,
        consensus_window=1, minimum_confidence=0.1,
        screen_fixed_mask=mask,
    )
    assert result["cumulative_path"][-1] == [8, 0]
    assert result["gap_recoveries"][0]["anchor_frames"] == [1, 3]
    assert result["placement_methods"][2] == "unresolved"
    assert result["placement_methods"][3] == "consensus_corrected"
    assert result["settings"]["screen_fixed_mask_pixels"] == 80


def test_exact_loop_closure_retains_pre_and_post_correction(monkeypatch):
    frames = [bytes([1]) * 16, bytes([2]) * 16, bytes([3]) * 16, bytes([1]) * 16]
    steps = iter([(2, 0), (2, 0), (-3, 0)])

    def fake_translation(*args, **kwargs):
        dx, dy = next(steps)
        return {"best_translation": [dx, dy], "confidence": 1.0}

    monkeypatch.setattr(perception, "detect_translation", fake_translation)
    result = perception.estimate_camera_path(frames, 4, 4, consensus_window=1)
    assert result["pre_correction_path"][-1] == [1, 0]
    assert result["cumulative_path"][-1] == [0, 0]
    assert result["loop_closures"][0]["correction_applied"] is True


def test_joint_control_ranking_keeps_channels_and_transitions_separate():
    trials = [
        {"action": "idle", "control": True, "tracks": [], "memory_deltas": {"x": 1}, "transition_fraction": 0.01},
        {"action": "idle", "control": True, "tracks": [], "memory_deltas": {"x": 0}, "transition_fraction": 0.02},
        {"action": "right", "tracks": [{"dx": 4, "dy": 0, "confidence": 0.9}], "global_motion": {"best_translation": [-5, 0], "confidence": 0.8}, "memory_deltas": {"x": 20}, "transition_fraction": 0.1, "memory_samples": [{"counter": 0, "x": 0}, {"counter": 3, "x": 1}]},
        {"action": "right", "tracks": [{"dx": 5, "dy": 0, "confidence": 0.9}], "global_motion": {"best_translation": [-5, 0], "confidence": 0.8}, "memory_deltas": {"x": 22}, "transition_fraction": 0.1, "memory_samples": [{"counter": 0, "x": 0}, {"counter": 4, "x": 1}]},
    ]
    result = perception.rank_joint_control_responses(trials, time_domain="video_frame")
    assert set(result["channels"]) == {"camera", "local_visual", "memory", "transition"}
    assert result["combined"][0]["action"] == "right"
    assert result["combined"][0]["onset_latency"] == 3
    assert result["combined"][0]["transition_ranked_separately"] is True


def test_mode_x_planar_text_vesa_and_dac_decoders_are_exact():
    mode_x = tuple(bytes([plane + 1, plane + 5]) for plane in range(4))
    assert vga.decode_mode_x(mode_x, 8, 1) == bytes([1, 2, 3, 4, 5, 6, 7, 8])

    planes = (bytes([0b10100000]), bytes([0b01100000]), bytes([0]), bytes([0b00010000]))
    assert vga.decode_planar_4bpp(planes, 8, 1)[:4] == bytes([1, 2, 3, 8])

    cells = vga.decode_text_cells(b"A\x1fB\x80", 2, 1)
    assert cells[0][0] == {"character": 65, "attribute": 31, "foreground": 15, "background": 1, "blink": 0}
    assert cells[0][1]["blink"] == 1

    raw = b"abcdXXefghYY"
    assert vga.decode_vesa_linear(raw, 2, 2, pitch=6, bytes_per_pixel=2) == b"abcdefgh"

    palette = bytes(768)
    timeline = vga.apply_dac_timeline(palette, [{"counter": 2, "index": 3, "component": 1, "value": 63}])
    assert timeline[0]["palette"][10] == 63


def test_acknowledged_companion_snapshot_decodes_interleaved_planes():
    pixels = bytes(range(1, 9))
    planes = tuple(pixels[index::4] for index in range(4))
    interleaved = b"".join(bytes(values) for values in zip(*planes))
    snapshot = {
        "raw_hex": interleaved.hex(),
        "dac_hex": bytes(768).hex(),
        "layout": "plane_interleaved",
        "mode_id": 3,
        "width": 8,
        "height": 1,
        "pitch": 8,
        "plane_pitch": 2,
        "start_address": 0,
        "bpp": 8,
    }
    decoded = vga.decode_companion_snapshot(snapshot)
    assert decoded["pixels"] == pixels
    assert decoded["raw"] == interleaved


def test_acknowledged_mode_x_snapshot_honours_nonzero_presented_page_and_retains_raw_pages():
    raw = bytearray(32)
    page = bytes(range(1, 9))
    for index, value in enumerate(page):
        raw[(2 + index // 4) * 4 + index % 4] = value
    snapshot = {
        "raw_hex": bytes(raw).hex(), "dac_hex": bytes(768).hex(),
        "layout": "plane_interleaved", "mode_id": 3,
        "width": 8, "height": 1, "pitch": 8, "plane_pitch": 2,
        "start_address": 2, "bpp": 8,
    }
    decoded = vga.decode_companion_snapshot(snapshot)
    assert decoded["pixels"] == page
    assert decoded["raw"] == bytes(raw)


def test_acknowledged_text_snapshot_preserves_cells_attributes_and_font_plane():
    raw_data = bytearray(16)
    raw_data[0:2], raw_data[8:10] = b"A\x1f", b"B\x80"
    raw_data[2], raw_data[6], raw_data[10], raw_data[14] = b"font"
    raw = bytes(raw_data)
    snapshot = {
        "raw_hex": raw.hex(), "dac_hex": bytes(768).hex(),
        "layout": "text_plane_interleaved", "mode_id": 10,
        "width": 16, "height": 8, "pitch": 2, "plane_pitch": 2,
        "start_address": 0, "bpp": 0, "text_columns": 2,
        "text_rows": 1, "font_height": 8, "char9dot": False,
        "text_cell_stride": 8,
    }
    decoded = vga.decode_companion_snapshot(snapshot)
    assert decoded["kind"] == "text"
    assert decoded["pixels"][0][0]["character"] == ord("A")
    assert decoded["pixels"][0][1]["blink"] == 1
    assert decoded["font_plane"] == b"font"


def test_write_watch_deduplicates_hot_writers_and_preserves_boundaries():
    events = [
        {"physical_address": 0x100, "cs": 0x2000, "ip": 0x10, "before": 1, "after": 2, "counter": i, "registers": {"ax": i}}
        for i in range(5)
    ] + [{"physical_address": 0x101, "cs": 0x2000, "ip": 0x20, "before": 0, "after": 1, "counter": 8}]
    result = writewatch.aggregate(events, physical_start=0x100, length=2)
    assert result["hits"] == 6
    assert result["writers"][0]["hits"] == 5
    assert result["writers"][0]["first"]["counter"] == 0
    assert result["writers"][0]["last"]["counter"] == 4


def test_qualification_analysis_uses_guest_counters_and_repeat_channels():
    width, height = 16, 8
    base = bytes((x + y * 7) % 251 for y in range(height) for x in range(width))
    moved = shifted(base, width, height, 1, 0)
    trials = [
        qualification.analyze_trial("right", [base, moved, moved], [0, 3, 6], [{"counter": 0, "x": 1}, {"counter": 3, "x": 2}, {"counter": 6, "x": 2}], width, height, max_dx=2, max_dy=1),
        qualification.analyze_trial("right", [base, moved, moved], [0, 3, 6], [{"counter": 0, "x": 1}, {"counter": 3, "x": 2}, {"counter": 6, "x": 2}], width, height, max_dx=2, max_dy=1),
    ]
    assert trials[0]["frame_counters"] == [0, 3, 6]
    assert trials[0]["memory_deltas"] == {"x": 1}
    assert qualification.repeat_agreement(trials) == 1.0
