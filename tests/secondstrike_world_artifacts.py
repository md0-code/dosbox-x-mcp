#!/usr/bin/env python3
"""Validate P3 spatial evidence against captured SecondStrike runtime artifacts."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


MCP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SECONDSTRIKE = MCP_ROOT.parent


async def validate(root: Path) -> dict[str, object]:
    sys.path.insert(0, str(MCP_ROOT / "src"))
    from dosbox_mcp import server

    motion_root = root / "re_artifacts" / "motion_v3"
    motion_report = json.loads((motion_root / "report.json").read_text(encoding="utf-8"))
    if not motion_report["framebuffer"]["confirmed"]:
        raise RuntimeError("SecondStrike source frames were not exact-confirmed")
    frames = sorted(motion_root.glob("probe_right_r1_frame_*.bin"))
    translations = [
        server.TranslationStep(dx=dx, dy=dy)
        for dx, dy in motion_report["translations"]
    ]
    if len(translations) != len(frames) - 1:
        raise RuntimeError("SecondStrike frame/translation evidence is incomplete")

    output = root / "re_artifacts" / "world_v1"
    mosaic = await server.dosbox_build_world_mosaic(
        [str(path) for path in frames],
        stem=str(output / "secondstrike"),
        width=320,
        height=200,
        translations=translations,
        infer_tiles=True,
    )

    trials = []
    source_traces = []
    trace_root = root / "re_artifacts" / "verification" / "original_campaign1"
    for name in ("launch_trace.json", "flight_trace.json"):
        path = trace_root / name
        trace = json.loads(path.read_text(encoding="utf-8"))
        # Both independently captured traces apply the same UP stimulus through
        # tick 840. The controlled trace adds RIGHT only after that point, so
        # this shared window keeps requested direction and coordinates aligned.
        states = [state for state in trace["states"] if 220 <= state["tick"] <= 840]
        positions = [
            server.PositionSample(x=state["x"] / 65536, y=state["y"] / 65536)
            for state in states
        ]
        direction = server.PositionSample(
            x=positions[-1].x - positions[0].x,
            y=positions[-1].y - positions[0].y,
        )
        trials.append(
            server.MovementTrial(label=path.stem, direction=direction, positions=positions)
        )
        source_traces.append(str(path))
    constraints = server.dosbox_infer_movement_constraints(
        trials,
        max_projected_progress=0.25,
        min_stalled_steps=5,
        position_tolerance=8.0,
        min_trial_support=2,
    )

    report = {
        "source_framebuffer": motion_report["framebuffer"],
        "mosaic": mosaic,
        "movement_constraint_sources": source_traces,
        "movement_constraints": constraints,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secondstrike-root", type=Path, default=DEFAULT_SECONDSTRIKE)
    args = parser.parse_args()
    report = asyncio.run(validate(args.secondstrike_root.resolve()))
    print(
        json.dumps(
            {
                "mosaic": {
                    key: report["mosaic"][key]
                    for key in (
                        "width",
                        "height",
                        "observed_pixels",
                        "conflict_observations",
                    )
                },
                "tile_candidate": report["mosaic"]["tile_structure"][
                    "best_candidate"
                ],
                "movement_constraint_candidates": report["movement_constraints"][
                    "candidate_count"
                ],
                "exact_framebuffer": report["source_framebuffer"]["confirmed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
