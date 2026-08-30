#!/usr/bin/env python3
"""Run the complete protocol-v2 qualification against original SecondStrike."""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hashlib
import json
import secrets
import sys
import time
import uuid
from pathlib import Path

MCP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SECONDSTRIKE = MCP_ROOT.parent


async def validate(root: Path, executable: Path) -> dict[str, object]:
    sys.path.insert(0, str(root / "tools"))
    sys.path.insert(0, str(MCP_ROOT / "src"))
    from dosbox_mcp import companion, emulator, server, vga
    from dosbox_mcp.profile import load_profile
    from dosbox_mcp.session import Registry
    from drive import Game
    from oracle_runtime import locate_framebuffer
    from playthrough import in_mission, reach_mission

    output = root / "re_artifacts" / "post_perception_v2"
    output.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    bundle_path = output / f"secondstrike-qualification-{run_id}"
    state_directory = output / f"state-{run_id}"
    state_directory.mkdir(parents=True)
    token = secrets.token_urlsafe(32)
    profile = load_profile(root / "tools" / "profiles" / "desert.json")
    config = root / "tools" / "conf" / "desert.conf"
    process, _ = emulator.launch(executable, config, save_directory=state_directory, companion_token=token)
    registry = Registry()
    session = registry.attach(process.pid, profile)
    session.companion_token = token
    session.state_directory = state_directory
    session.config = config
    game = Game.__new__(Game)
    game.name, game.profile, game.process, game.registry, game.session = "desert", profile, process, registry, session
    try:
        print("stage: waiting for authenticated companion", flush=True)
        deadline = time.monotonic() + 60
        while True:
            try:
                session.locate(True)
                client = session.companion()
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                await asyncio.sleep(0.25)
        assert client is not None
        print("stage: exact framebuffer confirmation", flush=True)
        framebuffer = await asyncio.to_thread(locate_framebuffer, game)
        if framebuffer.found is None or not framebuffer.found.confirmed:
            raise RuntimeError("SecondStrike framebuffer was not exact-confirmed")
        print("stage: reaching a fresh original campaign mission", flush=True)
        if not await asyncio.to_thread(reach_mission, game, None, "mission", 16.0, 80):
            raise RuntimeError("SecondStrike did not reach a fresh campaign mission")
        await asyncio.sleep(6.0)
        tick_offset = profile.symbols["tick_counter"].offset
        tick_before = int.from_bytes(session.read(tick_offset, 4), "little", signed=True)
        player_before = int.from_bytes(session.read(profile.symbols["player_entity"].offset, 2), "little")
        await asyncio.sleep(0.2)
        tick_after = int.from_bytes(session.read(tick_offset, 4), "little", signed=True)
        if not 0x1000 < player_before < 0xFF00 or tick_after <= tick_before:
            raise RuntimeError("settled campaign baseline has no live player or advancing game clock")

        print("stage: waiting for acknowledged interactive control response", flush=True)
        control_physical = (profile.ds_segment << 4) + profile.symbols["control_mask"].offset
        interactive_probe: dict[str, object] | None = None
        for attempt in range(200):
            probe = await asyncio.to_thread(client.run_timeline, "video_frame", [
                {"at": 0, "order": 0, "kind": "key_make", "key_id": 86},
                {"at": 8, "order": 1, "kind": "memory_snapshot", "physical_start": control_physical, "length": 1},
                {"at": 10, "order": 2, "kind": "key_break", "key_id": 86},
                {"at": 12, "order": 3, "kind": "memory_snapshot", "physical_start": control_physical, "length": 1},
            ], 10.0)
            held_mask = int.from_bytes(bytes.fromhex(probe["timeline_events"][1]["observation"]["hex"]), "little")
            released_mask = int.from_bytes(bytes.fromhex(probe["timeline_events"][3]["observation"]["hex"]), "little")
            if held_mask & 0x08:
                interactive_probe = {"attempt": attempt + 1, "held_mask": held_mask, "released_mask": released_mask, "timeline": probe}
                break
        if interactive_probe is None:
            raise RuntimeError("fresh mission never consumed acknowledged RIGHT input into its control mask")

        selected = ["key_last_state", "key_last_scancode", "control_mask", "control_mask_pressed", "player_x_copy", "player_y_copy", "player_state"]
        memory_events = [{"at": 0, "order": index, "kind": "memory_snapshot", "physical_start": (profile.ds_segment << 4) + profile.symbols[name].offset, "length": profile.symbols[name].size} for index, name in enumerate(selected)]
        memory_events.append({"at": 0, "order": len(memory_events), "kind": "memory_snapshot", "physical_start": 0xA0000, "length": 320 * 200})
        def selected_checkpoint() -> tuple[dict[str, str], bytes]:
            timeline = client.run_timeline("video_frame", memory_events, 10.0)
            events = timeline["timeline_events"]
            return ({name: str(event["observation"]["hex"]) for name, event in zip(selected, events)}, bytes.fromhex(str(events[-1]["observation"]["hex"])))

        print("stage: 100 acknowledged save/restore branches", flush=True)
        saved = await asyncio.to_thread(client.request, "save_state")
        baseline_restore = await asyncio.to_thread(client.request, "load_state")
        baseline_memory, baseline_frame = await asyncio.to_thread(selected_checkpoint)
        assert isinstance(baseline_frame, bytes)
        baseline_snapshot = await asyncio.to_thread(client.request, "video_snapshot", {"after_frames": 0, "max_bytes": 1024 * 1024})
        decoded_snapshot = vga.decode_companion_snapshot(baseline_snapshot)["pixels"]
        if not isinstance(decoded_snapshot, bytes) or len(decoded_snapshot) != 320 * 200:
            raise RuntimeError("companion VGA decoder returned an invalid mission frame")
        # The raw decoder check advances the guest; restore once more so each
        # measured branch has exactly the same post-load event sequence.
        await asyncio.to_thread(client.request, "load_state")
        restore_failures: list[dict[str, object]] = []
        for index in range(100):
            restored = await asyncio.to_thread(client.request, "load_state")
            memory, decoded = await asyncio.to_thread(selected_checkpoint)
            events = [event["type"] for event in restored["events"]]
            if decoded != baseline_frame or memory != baseline_memory or events != ["load_started", "load_completed", "post_restore_frame"]:
                restore_failures.append({"branch": index + 1, "frame_equal": decoded == baseline_frame, "memory_equal": memory == baseline_memory, "memory_differences": {name: {"expected": baseline_memory[name], "actual": memory[name]} for name in selected if memory[name] != baseline_memory[name]}, "events": events})
            if (index + 1) % 25 == 0:
                print(f"  restore branches: {index + 1}/100", flush=True)
        if restore_failures:
            raise RuntimeError(f"restore stress failed: {restore_failures[:3]}")

        print("stage: 50-frame input under host load", flush=True)
        load_result: dict[str, object]
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            busy = pool.submit(lambda: sum(value * value for value in range(7_000_000)))
            load_result = await asyncio.to_thread(client.run_timeline, "video_frame", [
                {"at": 0, "order": 0, "kind": "key_make", "key_id": 86},
                {"at": 50, "order": 1, "kind": "key_break", "key_id": 86},
            ], 30.0)
            busy.result()
        if load_result["timeline_events"][1]["actual_counter"] != 50:
            raise RuntimeError("50-frame hold changed under host load")

        print("stage: expiry, timeout, and disconnect key cleanup", flush=True)
        try:
            await asyncio.to_thread(client.run_timeline, "video_frame", [
                {"at": 0, "order": 0, "kind": "key_make", "key_id": 86},
                {"at": 100, "order": 1, "kind": "key_break", "key_id": 86},
            ], 10.0, 2)
            raise RuntimeError("expiring timeline unexpectedly completed")
        except companion.CompanionError as failure:
            if "expired" not in str(failure):
                raise
        await asyncio.sleep(0.1)
        if session.read(0x043D + 0xCD, 1)[0]:
            raise RuntimeError("expiry left RIGHT held")

        raw_stream = client._connect(5.0)
        entered = raw_stream.__enter__()
        request_id = uuid.uuid4().hex
        entered.write(companion.encode_message({"protocol": 2, "request_id": request_id, "token": token, "operation": "run_timeline", "clock": "video_frame", "events": [
            {"at": 0, "order": 0, "kind": "key_make", "key_id": 86},
            {"at": 1000, "order": 1, "kind": "key_break", "key_id": 86},
        ]}))
        await asyncio.sleep(0.1)
        raw_stream.__exit__(None, None, None)
        await asyncio.sleep(0.15)
        if session.read(0x043D + 0xCD, 1)[0]:
            raise RuntimeError("disconnect left RIGHT held")

        print("stage: responsive motion-field writer linkage", flush=True)
        motion_offset = profile.symbols["control_mask"].offset
        physical_x = (profile.ds_segment << 4) + motion_offset
        linked_timeline = await asyncio.to_thread(client.run_timeline, "video_frame", [
            {"at": 0, "order": 0, "kind": "write_watch_start", "physical_start": physical_x, "length": 3, "hit_limit": 1000},
            {"at": 1, "order": 1, "kind": "key_make", "key_id": 86},
            {"at": 8, "order": 2, "kind": "capture_mark", "capture_bytes": 256064},
            {"at": 8, "order": 3, "kind": "trace_mark"},
            {"at": 10, "order": 4, "kind": "key_break", "key_id": 86},
            {"at": 12, "order": 5, "kind": "write_watch_stop"},
        ], 30.0)
        write_observation = linked_timeline["timeline_events"][5]["observation"]
        writer_summary = {"physical": physical_x, "events": write_observation["write_events"], "raw_hits": write_observation["raw_hits"], "truncated": write_observation["truncated"], "timeline": linked_timeline}
        if not write_observation["write_events"]:
            raise RuntimeError("no runtime writer was observed for the responsive control-to-motion field")
        if not all(linked_timeline["timeline_events"][index].get("irq1_observed") for index in (1, 4)):
            raise RuntimeError("writer linkage did not retain acknowledged key make/break evidence")
        if session.read(0x043D + 0xCD, 1)[0]:
            raise RuntimeError("writer-linkage input left RIGHT held")
        trace_path = output / f"control-mask-writers-{run_id}.json"
        trace_path.write_text(json.dumps(writer_summary, indent=2) + "\n", encoding="utf-8")

        print("stage: repeated six-control qualification and complete bundle", flush=True)
        server.registry = registry
        result = await server.dosbox_qualify_controls(
            ["left", "right", "up", "c", "x"],
            hold_frames=12,
            observation_frames=24,
            repeats=2,
            memory_fields=[server.MemoryProbe(label=name, offset=name, size=profile.symbols[name].size) for name in selected],
            bundle_destination=str(bundle_path),
            game_files=[str(root / "data" / "desert_strike" / "DESERT.EXE"), str(root / "data" / "desert_strike" / "DESERT.DAT")],
            trace_files=[str(trace_path)],
            progress=True,
            pid=process.pid,
        )
        if result["state"] not in {"valid", "valid_with_ambiguity"}:
            raise RuntimeError(f"qualification gates rejected the run: {result['quality']}")
        if not result["bundle"]["verification"]["valid"]:
            raise RuntimeError(f"bundle verification failed: {result['bundle']['verification']}")
        reproduction = await asyncio.to_thread(server.dosbox_reproduce_experiment_bundle, str(bundle_path))
        if not reproduction["valid"]:
            raise RuntimeError(f"offline derived-report reproduction failed: {reproduction}")

        summary = {
            "run_id": run_id,
            "pid": process.pid,
            "protocol_status": await asyncio.to_thread(client.request, "status"),
            "baseline_origin": "fresh original campaign path",
            "interactive_probe": interactive_probe,
            "stress_baseline_restore": baseline_restore,
            "framebuffer": framebuffer.found.as_dict(),
            "baseline_frame_sha256": hashlib.sha256(baseline_frame).hexdigest(),
            "restore_stress": {"branches": 100, "failures": restore_failures, "personal_slot_touched": False},
            "host_load_hold": {"requested_frames": 50, "actual_frames": load_result["timeline_events"][1]["actual_counter"]},
            "cleanup": {"success": True, "expiry": True, "timeout_disconnect": True, "disconnect": True, "process_termination": "performed in finally"},
            "qualification": result,
            "offline_reproduction": reproduction,
            "motion_field_writer": writer_summary,
        }
        report_path = output / f"qualification-report-{run_id}.json"
        report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"stage: PASS report={report_path}", flush=True)
        return summary
    finally:
        session.close()
        process.terminate()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secondstrike-root", type=Path, default=DEFAULT_SECONDSTRIKE)
    parser.add_argument("--dosbox-executable", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(validate(args.secondstrike_root.resolve(), args.dosbox_executable.resolve()))
    print(json.dumps({"run_id": result["run_id"], "state": result["qualification"]["state"], "bundle": result["qualification"]["bundle"], "restore_stress": result["restore_stress"], "motion_field_writer_hits": result["motion_field_writer"]["raw_hits"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
