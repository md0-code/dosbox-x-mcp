#!/usr/bin/env python3
"""Validate protocol-v2 guest-time timelines against original SecondStrike."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import secrets
import sys
import time
from pathlib import Path

MCP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SECONDSTRIKE = MCP_ROOT.parent


def validate(root: Path, executable: Path) -> dict[str, object]:
    sys.path.insert(0, str(root / "tools"))
    sys.path.insert(0, str(MCP_ROOT / "src"))

    from dosbox_mcp import emulator
    from dosbox_mcp.profile import load_profile
    from dosbox_mcp.session import Registry
    from dosbox_mcp.vga import dechain
    from oracle_runtime import locate_framebuffer

    token = secrets.token_urlsafe(32)
    profile = load_profile(root / "tools" / "profiles" / "desert.json")
    config = root / "tools" / "conf" / "desert.conf"
    output = root / "re_artifacts" / "post_perception_v2"
    output.mkdir(parents=True, exist_ok=True)
    state_directory = output / "state"
    state_directory.mkdir(parents=True, exist_ok=True)
    process, _ = emulator.launch(executable, config, save_directory=state_directory, companion_token=token)
    registry = Registry()
    session = registry.attach(process.pid, profile)
    session.companion_token = token
    try:
        print("stage: waiting for companion", flush=True)
        deadline = time.monotonic() + 60
        while True:
            try:
                session.locate(True)
                client = session.companion()
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.25)
        assert client is not None
        live = type("LiveGame", (), {"process": process, "session": session, "profile": profile})()
        framebuffer = locate_framebuffer(live)
        if framebuffer.found is None or not framebuffer.found.confirmed:
            raise RuntimeError("SecondStrike framebuffer is not exact-confirmed")
        print("stage: framebuffer confirmed", flush=True)

        events = [
            {"at": 2, "order": 0, "kind": "key_make", "key_id": 86, "key": "right"},
            {"at": 4, "order": 1, "kind": "capture_mark", "label": "during-right"},
            {"at": 6, "order": 2, "kind": "key_break", "key_id": 86, "key": "right"},
            {"at": 6, "order": 3, "kind": "memory_snapshot", "label": "released"},
        ]
        first = client.run_timeline("video_frame", events, 20.0)
        print("stage: first timeline complete", flush=True)
        second = client.run_timeline("video_frame", events, 20.0)
        print("stage: second timeline complete", flush=True)
        first_events = first["timeline_events"]
        second_events = second["timeline_events"]
        expected_order = [0, 1, 2, 3]
        if [item["order"] for item in first_events] != expected_order:
            raise RuntimeError("first timeline event ordering changed")
        if [item["order"] for item in second_events] != expected_order:
            raise RuntimeError("second timeline event ordering changed")
        if not first["released_all_keys"] or session.read(0x043D + 0xCD, 1)[0] != 0:
            raise RuntimeError("RIGHT remained held after timeline completion")
        if any(item["actual_counter"] < item["requested_counter"] for item in first_events):
            raise RuntimeError("an event was acknowledged before its requested frame")

        saved = client.request("save_state")
        print("stage: save complete", flush=True)
        if [event["type"] for event in saved["events"]] != ["save_started", "save_completed"]:
            raise RuntimeError("save acknowledgements are incomplete")
        loaded = client.request("load_state")
        print("stage: load complete", flush=True)
        if [event["type"] for event in loaded["events"]] != ["load_started", "load_completed", "post_restore_frame"]:
            raise RuntimeError("load acknowledgements are incomplete")

        video = client.request("video_snapshot", {"after_frames": 1, "max_bytes": 256000})
        video_raw = bytes.fromhex(video.pop("raw_hex"))
        video_dac = bytes.fromhex(video.pop("dac_hex"))
        direct_frame = framebuffer.read()
        if video["layout"] != "dosbox_chain4_sparse" or dechain(video_raw)[: len(direct_frame)] != direct_frame:
            raise RuntimeError("acknowledged raw VGA snapshot does not match the exact-confirmed SecondStrike page")
        print("stage: raw video snapshot confirmed", flush=True)

        right_offset = 0x043D + 0xCD
        physical = (profile.ds_segment << 4) + right_offset
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            watching = executor.submit(
                client.request,
                "observe_writes",
                {"physical_start": physical, "length": 1, "duration_counter": 12, "clock": "video_frame", "hit_limit": 1000, "registers": ["ax", "ds"]},
            )
            print("stage: write watch armed", flush=True)
            time.sleep(0.1)
            session.key_state("right", True, 2.0)
            time.sleep(0.15)
            session.key_state("right", False)
            writes = watching.result(timeout=20)
        print("stage: write watch complete", flush=True)
        if not writes["write_events"]:
            raise RuntimeError("bounded write observer saw no RIGHT-state writes")
        if any(event["physical_address"] != physical for event in writes["write_events"]):
            raise RuntimeError("write observer escaped its physical range")

        report = {
            "dosbox_executable": str(executable),
            "pid": process.pid,
            "protocol": 2,
            "transport": client.capabilities["transport"],
            "counters": client.request("status")["counters"],
            "framebuffer": framebuffer.found.as_dict(),
            "first": first,
            "second": second,
            "save": saved,
            "load": loaded,
            "video_snapshot": {
                **video,
                "raw_sha256": hashlib.sha256(video_raw).hexdigest(),
                "dac_sha256": hashlib.sha256(video_dac).hexdigest(),
                "decoded_matches_exact_framebuffer": True,
            },
            "write_watch": {
                "physical_address": physical,
                "hits": len(writes["write_events"]),
                "writers": sorted(set(f"{event['cs']:04X}:{event['ip']:04X}" for event in writes["write_events"])),
                "first": writes["write_events"][0],
                "last": writes["write_events"][-1],
            },
            "ordering_repeatable": True,
            "released_all_keys": True,
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "timeline_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        session.close()
        process.terminate()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secondstrike-root", type=Path, default=DEFAULT_SECONDSTRIKE)
    parser.add_argument("--dosbox-executable", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.secondstrike_root.resolve(), args.dosbox_executable.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
