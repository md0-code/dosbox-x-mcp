#!/usr/bin/env python3
"""Exercise Perception v1 against a live original SecondStrike DOS guest.

This is intentionally outside the offline pytest suite.  It needs the sibling
SecondStrike checkout, DOSBox-X, its original game data, and a desktop.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path


MCP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SECONDSTRIKE = MCP_ROOT.parent


async def validate(root: Path, reach_live_mission: bool) -> dict[str, object]:
    sys.path.insert(0, str(root / "tools"))
    sys.path.insert(0, str(MCP_ROOT / "src"))

    from dosbox_mcp import server
    from drive import Game
    from oracle_runtime import locate_framebuffer
    from playthrough import reach_mission

    game = Game("desert")
    try:
        mission = reach_mission(game) if reach_live_mission else False
        if not reach_live_mission:
            # The profile appears while the protection screen is still fading.
            # Wait for its palette to settle before requiring an exact DAC match.
            time.sleep(5.0)
        framebuffer = locate_framebuffer(game)
        if framebuffer.found is None or not framebuffer.found.confirmed:
            raise RuntimeError("SecondStrike framebuffer was not confirmed against the live guest")

        # Reuse the already attached session through the actual MCP entrypoints.
        server.registry = game.registry
        output = root / "re_artifacts" / "perception_v1"
        output.mkdir(parents=True, exist_ok=True)
        stem = output / "secondstrike_idle"
        observed = await server.dosbox_act_and_observe(
            server.ExperimentAction(kind="idle"),
            duration_seconds=1.17,
            interval_seconds=0.037,
            memory_fields=[
                server.MemoryProbe(label="frame_subtick", offset="frame_subtick", size=2),
                server.MemoryProbe(label="tick_counter", offset="tick_counter", size=4),
                server.MemoryProbe(label="player_x_copy", offset="player_x_copy", size=2),
                server.MemoryProbe(label="player_y_copy", offset="player_y_copy", size=2),
            ],
            memory_range=server.MemoryRangeProbe(
                offset="0x0540", length=16, value_size=1, relation="changed"
            ),
            max_samples=64,
            max_frames=64,
            stem=str(stem),
        )

        frame_paths = sorted(output.glob("secondstrike_idle_frame_*.bin"))
        compared = None
        if len(frame_paths) >= 2:
            compared = await server.dosbox_compare_frames(
                [str(path) for path in frame_paths],
                width=320,
                height=200,
                bridge_radius=1,
                min_changed_pixels=2,
            )

        before_path = output / "secondstrike_idle_memory_before.bin"
        after_path = output / "secondstrike_idle_memory_after.bin"
        scanned = await server.dosbox_scan_changes(
            str(before_path),
            str(after_path),
            value_size=1,
            relation="changed",
            offset_base=0x0540,
        )

        correlation = None
        tracks = observed["visual"]["tracking"]["tracks"]
        for track in sorted(tracks, key=lambda item: -len(item["history"])):
            history = track["history"]
            if len(history) < 3:
                continue
            correlation = server.dosbox_correlate_memory(
                [entry["centroid"][0] for entry in history],
                {
                    "frame_subtick": [entry["memory"]["frame_subtick"] for entry in history],
                    "tick_counter": [entry["memory"]["tick_counter"] for entry in history],
                    "player_x_copy": [entry["memory"]["player_x_copy"] for entry in history],
                    "player_y_copy": [entry["memory"]["player_y_copy"] for entry in history],
                },
            )
            break

        summary = {
            "mission_reached": mission,
            "framebuffer": framebuffer.found.as_dict(),
            "action": observed["action"],
            "distinct_frames": observed["visual"]["distinct_frames"],
            "baseline_to_final_changed_pixels": observed["visual"]["baseline_to_final"][
                "changed_pixels"
            ],
            "live_track_count": observed["visual"]["tracking"]["track_count"],
            "saved_frame_transitions": len(compared["transitions"]) if compared else 0,
            "saved_frame_track_count": (
                compared["tracking"]["track_count"] if compared else 0
            ),
            "memory_field_deltas": {
                name: evidence["delta"]
                for name, evidence in observed["memory"]["fields"].items()
            },
            "memory_candidates": scanned["total_candidates"],
            "correlation_fields": correlation["total_fields"] if correlation else 0,
            "evidence_directory": str(output),
        }
        (output / "report.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary
    finally:
        game.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secondstrike-root", type=Path, default=DEFAULT_SECONDSTRIKE)
    parser.add_argument(
        "--mission",
        action="store_true",
        help="Reach a live mission before observing (slower and harder to confirm exactly).",
    )
    args = parser.parse_args()
    result = asyncio.run(validate(args.secondstrike_root.resolve(), args.mission))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
