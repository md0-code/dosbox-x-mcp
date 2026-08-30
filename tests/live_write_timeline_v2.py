#!/usr/bin/env python3
"""Attribute a known hot-loop write inside one acknowledged timeline."""
from __future__ import annotations

import argparse
import json
import secrets
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosbox_mcp import companion, emulator, writewatch  # noqa: E402


def run(executable: Path, report_path: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="dosbox-mcp-writer-") as temporary:
        root = Path(temporary)
        # ES=5000h, DI=0100h; INC byte ES:[DI]; STI; HLT; loop.
        (root / "WRITER.COM").write_bytes(bytes.fromhex("b800508ec0bf000126fe05fbf4ebf9"))
        config = root / "dosbox.conf"
        config.write_text(
            "[sdl]\noutput=surface\n[dosbox]\nmachine=svga_s3\n"
            "[cpu]\ncore=normal\ncycles=fixed 3000\n[autoexec]\n"
            f"mount C {root}\nC:\nWRITER.COM\n",
            encoding="ascii",
        )
        token = secrets.token_urlsafe(32)
        process, _ = emulator.launch(executable, config, save_directory=root / "state", companion_token=token)
        client = companion.CompanionClient(process.pid, token)
        try:
            deadline = time.monotonic() + 30
            while True:
                try:
                    client.negotiate()
                    break
                except Exception:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)
            client.request("video_snapshot", {"after_frames": 60, "max_bytes": 65536})
            timeline = client.run_timeline("video_frame", [
                {"at": 0, "order": 0, "kind": "write_watch_start", "physical_start": 0x50100, "length": 1, "hit_limit": 1000},
                {"at": 4, "order": 1, "kind": "capture_mark", "capture_bytes": 65536},
                {"at": 4, "order": 2, "kind": "trace_mark"},
                {"at": 8, "order": 3, "kind": "write_watch_stop"},
            ], 20.0)
            observation = timeline["timeline_events"][-1]["observation"]
            aggregate = writewatch.aggregate(observation["write_events"], physical_start=0x50100, length=1, hit_limit=1000, output_limit=20)
            if not observation["write_events"] or any(hit["physical_address"] != 0x50100 for hit in observation["write_events"]):
                raise RuntimeError("known physical write was not attributed")
            writers = {(hit["cs"], hit["ip"]) for hit in observation["write_events"]}
            if len(writers) != 1:
                raise RuntimeError(f"known single instruction produced inconsistent writers: {writers}")
            if next(iter(writers))[1] != 0x0108:
                raise RuntimeError(f"known INC instruction was attributed to the wrong IP: {writers}")
            if [event["actual_counter"] for event in timeline["timeline_events"]] != [0, 4, 4, 8]:
                raise RuntimeError("writer timeline did not retain exact requested frame counters")
            result = {
                "status": "pass", "physical_address": 0x50100,
                "writer": {"cs": next(iter(writers))[0], "ip": next(iter(writers))[1]},
                "raw_hits": observation["raw_hits"], "truncated": observation["truncated"],
                "aggregate": aggregate,
                "timeline_request_id": timeline["request_id"],
                "actual_counters": [event["actual_counter"] for event in timeline["timeline_events"]],
                "frame_mark_counter": timeline["timeline_events"][1]["actual_counter"],
                "trace_mark_counter": timeline["timeline_events"][2]["actual_counter"],
            }
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            return result
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.executable.resolve(), args.report.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
