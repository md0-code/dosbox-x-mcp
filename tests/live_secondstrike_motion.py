#!/usr/bin/env python3
"""Exercise control probing and global-motion evidence in a live mission."""
from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import time
from pathlib import Path


MCP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SECONDSTRIKE = MCP_ROOT.parent


async def validate(root: Path, executable: Path) -> dict[str, object]:
    sys.path.insert(0, str(root / "tools"))
    sys.path.insert(0, str(MCP_ROOT / "src"))
    from dosbox_mcp import server
    from drive import Game
    from oracle_runtime import locate_framebuffer
    from playthrough import reach_mission

    output = root / "re_artifacts" / "motion_v3"
    state_directory = output / "state"
    state_directory.mkdir(parents=True, exist_ok=True)
    game = Game("desert", executable=executable, save_directory=state_directory)
    try:
        # Confirm the backing page while the frontend is stable. Exact window
        # matching can race a continuously repainting mission even though the
        # already-confirmed backing region remains the same.
        time.sleep(5.0)
        framebuffer = locate_framebuffer(game)
        if framebuffer.found is None or not framebuffer.found.confirmed:
            raise RuntimeError("SecondStrike framebuffer is not exact-confirmed")
        if not reach_mission(game):
            raise RuntimeError("SecondStrike did not reach a live mission")
        while struct.unpack("<i", game.session.read(0x5C53, 4))[0] < 225:
            time.sleep(0.01)
        player = game.word(0x642A)
        motion = game.word(player + 0x16)
        server.registry = game.registry
        probed = await server.dosbox_probe_controls(
            ["left", "right"],
            hold_seconds=0.55,
            observation_seconds=0.85,
            repeats=2,
            pid=game.process.pid,
            interval_seconds=0.025,
            memory_fields=[
                server.MemoryProbe(label="player_x", offset=motion, size=4),
                server.MemoryProbe(label="player_y", offset=motion + 4, size=4),
                server.MemoryProbe(label="raw_left", offset=0x043D + 0xCB),
                server.MemoryProbe(label="raw_right", offset=0x043D + 0xCD),
            ],
            stem=str(output / "probe"),
        )

        right_frames = sorted(output.glob("probe_right_r1_frame_*.bin"))
        if len(right_frames) < 3:
            raise RuntimeError("RIGHT probe retained fewer than three distinct exact frames")
        motion = await server.dosbox_compare_frames(
            [str(path) for path in right_frames],
            width=320,
            height=200,
            bridge_radius=1,
            min_changed_pixels=2,
            detect_global_motion=True,
            max_translation_x=16,
            max_translation_y=16,
            translation_sample_stride=2,
            infer_screen_layout=True,
            min_scroll_episodes=2,
        )
        translations = [
            transition["global_motion"]["best_translation"]
            for transition in motion["transitions"]
        ]
        summary = {
            "pid": game.process.pid,
            "ranking": probed["ranking"],
            "trial_summaries": probed["trial_summaries"],
            "baseline_restored": probed["baseline_restored"],
            "right_frame_count": len(right_frames),
            "translations": translations,
            "nonzero_translations": sum(value != [0, 0] for value in translations),
            "screen_regions": motion["screen_regions"],
            "framebuffer": framebuffer.found.as_dict(),
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
    parser.add_argument("--dosbox-executable", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(
        validate(args.secondstrike_root.resolve(), args.dosbox_executable.resolve())
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
