#!/usr/bin/env python3
"""Validate isolated controlled branches against original SecondStrike."""
from __future__ import annotations

import argparse
import asyncio
import json
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

    output = root / "re_artifacts" / "branching_v1"
    state_directory = output / "state"
    state_directory.mkdir(parents=True, exist_ok=True)
    game = Game(
        "desert", executable=executable, save_directory=state_directory
    )
    try:
        time.sleep(5.0)
        framebuffer = locate_framebuffer(game)
        if framebuffer.found is None or not framebuffer.found.confirmed:
            raise RuntimeError("SecondStrike framebuffer is not exact-confirmed")
        server.registry = game.registry
        result = await server.dosbox_try_actions(
            [
                server.BranchTrial(
                    label="idle", action=server.ExperimentAction(kind="idle")
                ),
                server.BranchTrial(
                    label="right",
                    action=server.ExperimentAction(
                        kind="hold_keys",
                        keys=["right"],
                        duration_seconds=0.3,
                    ),
                ),
            ],
            duration_seconds=0.5,
            pid=game.process.pid,
            interval_seconds=0.025,
            memory_fields=[
                server.MemoryProbe(label="key_last_state", offset="key_last_state"),
                server.MemoryProbe(label="key_last_scancode", offset="key_last_scancode"),
                server.MemoryProbe(label="protect_count", offset="protect_count", size=2),
            ],
            max_samples=64,
            max_frames=64,
            stem=str(output / "trial"),
        )

        idle_baseline = output / "trial_idle_frame_000.bin"
        right_baseline = output / "trial_right_frame_000.bin"
        if idle_baseline.read_bytes() != right_baseline.read_bytes():
            raise RuntimeError("restored trial baseline frames differ")
        idle_samples = result["trials"][0]["evidence"]["memory"]["samples"]
        right_samples = result["trials"][1]["evidence"]["memory"]["samples"]
        if any(sample["key_last_scancode"] == 0xCD for sample in idle_samples):
            raise RuntimeError("idle control unexpectedly observed the RIGHT scan code")
        if not any(sample["key_last_scancode"] == 0xCD for sample in right_samples):
            raise RuntimeError("RIGHT branch did not observe its authentic scan code")

        summary = {
            "pid": game.process.pid,
            "state_file": result["baseline_state"]["state_file"],
            "state_bytes": result["baseline_state"]["bytes"],
            "trial_labels": [entry["label"] for entry in result["trials"]],
            "restore_count": result["restore_count"],
            "baseline_restored": result["baseline_restored"],
            "baseline_frames_equal": True,
            "idle_saw_right": False,
            "right_saw_right": True,
            "right_transport": result["trials"][1]["evidence"]["action"]["transport"],
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
