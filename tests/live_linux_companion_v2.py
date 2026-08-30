#!/usr/bin/env python3
"""Exercise protocol-v2 event semantics against the native Linux build."""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dosbox_mcp import companion  # noqa: E402


def wait_client(process: subprocess.Popen[bytes], token: str, timeout: float = 30.0) -> companion.CompanionClient:
    client = companion.CompanionClient(process.pid, token)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            diagnostics = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            raise RuntimeError(f"Linux DOSBox-X exited {process.returncode} before endpoint: {diagnostics[-4000:]}")
        try:
            client.negotiate()
            return client
        except Exception:
            time.sleep(0.05)
    raise RuntimeError(f"Linux companion endpoint did not appear for pid {process.pid}")


def run(executable: Path, report: Path) -> dict[str, object]:
    token = secrets.token_urlsafe(32)
    with tempfile.TemporaryDirectory(prefix="dosbox-mcp-linux-") as temporary:
        root = Path(temporary)
        drive = root / "drive"
        state = root / "states"
        drive.mkdir()
        state.mkdir()
        # Delay the mode switch long enough for endpoint negotiation, then enter
        # a BIOS-cleared, static Mode 13h page and remain IRQ-responsive.
        (drive / "IDLE.COM").write_bytes(bytes.fromhex("b96400fbf4e2fdb81300cd10f4ebfd"))
        config = root / "linux.conf"
        config.write_text(
            "[sdl]\nfullscreen=false\nmenu=false\nvideodriver=x11\n"
            "[dosbox]\nmachine=svga_s3\nmemsize=16\n"
            "[cpu]\ncore=normal\ncycles=fixed 3000\n"
            "[autoexec]\nmount c " + str(drive) + "\nc:\nidle.com\n",
            encoding="utf-8",
        )
        environment = dict(os.environ)
        environment["DOSBOX_X_MCP_TOKEN"] = token
        environment.setdefault("DISPLAY", ":0")
        process = subprocess.Popen(
            [str(executable), "-conf", str(config), "-savedir", str(state), "-set", "dosbox saveslot=100", "-set", "dosbox saveremark=false"],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            client = wait_client(process, token)
            status = client.request("status")
            capabilities = status["capabilities"]
            required = {"timeline", "authentic_keyboard", "save_load_ack", "write_watch", "video_snapshot"}
            if not all(capabilities.get(name) is True for name in required):
                raise RuntimeError(f"Linux capabilities differ: {capabilities}")
            if capabilities.get("transport") != "unix_domain_socket":
                raise RuntimeError(f"Linux used wrong transport: {capabilities}")
            mode_deadline = time.monotonic() + 30
            while True:
                active_video = client.request("video_snapshot", {"after_frames": 1, "max_bytes": 262144})
                if active_video.get("width") == 320 and active_video.get("height") == 200 and active_video.get("bpp") == 8:
                    break
                if time.monotonic() >= mode_deadline:
                    raise RuntimeError(f"Linux fixture never reached static Mode 13h: {active_video}")
            try:
                companion.CompanionClient(process.pid, "wrong-token").request("status")
                raise RuntimeError("wrong Linux companion token was accepted")
            except companion.CompanionError as failure:
                if "authentication failed" not in str(failure):
                    raise

            timeline = client.run_timeline("video_frame", [
                {"at": 0, "order": 0, "kind": "key_make", "key_id": 86},
                {"at": 1, "order": 1, "kind": "memory_snapshot", "physical_start": 0x400, "length": 16},
                {"at": 2, "order": 2, "kind": "capture_mark", "capture_bytes": 65536},
                {"at": 3, "order": 3, "kind": "key_break", "key_id": 86},
            ], 15.0)
            events = timeline["timeline_events"]
            if [event["actual_counter"] for event in events] != [0, 1, 2, 3]:
                raise RuntimeError(f"Linux timeline counters differ: {events}")
            if events[0]["acknowledgement"] != "key_applied" or not events[0].get("irq1_observed"):
                raise RuntimeError(f"Linux key make was not IRQ-acknowledged: {events[0]}")
            if events[-1]["acknowledgement"] != "key_released" or not timeline.get("released_all_keys"):
                raise RuntimeError(f"Linux key cleanup differed: {timeline}")
            if len(bytes.fromhex(events[1]["observation"]["hex"])) != 16:
                raise RuntimeError("Linux memory snapshot length differed")
            if not events[2]["observation"].get("dac_hex"):
                raise RuntimeError("Linux atomic capture omitted same-frame DAC evidence")

            saved = client.request("save_state")
            state_file = state / "100.sav"
            if [event["type"] for event in saved["events"]] != ["save_started", "save_completed"] or not state_file.is_file():
                raise RuntimeError(f"Linux save acknowledgement differed: {saved}")
            loaded = client.request("load_state")
            if [event["type"] for event in loaded["events"]] != ["load_started", "load_completed", "post_restore_frame"]:
                raise RuntimeError(f"Linux restore acknowledgement differed: {loaded}")
            checkpoint_events = [
                {"at": 0, "order": 0, "kind": "memory_snapshot", "physical_start": 0x500, "length": 32},
                {"at": 0, "order": 1, "kind": "capture_mark", "capture_bytes": 65536},
            ]
            baseline_checkpoint = client.run_timeline("video_frame", checkpoint_events, 15.0)
            baseline_memory = baseline_checkpoint["timeline_events"][0]["observation"]["hex"]
            baseline_video = baseline_checkpoint["timeline_events"][1]["observation"]["raw_hex"]
            branch_failures = []
            for branch in range(10):
                branch_load = client.request("load_state")
                checkpoint = client.run_timeline("video_frame", checkpoint_events, 15.0)
                memory_equal = checkpoint["timeline_events"][0]["observation"]["hex"] == baseline_memory
                video_equal = checkpoint["timeline_events"][1]["observation"]["raw_hex"] == baseline_video
                acknowledged = [event["type"] for event in branch_load["events"]] == ["load_started", "load_completed", "post_restore_frame"]
                if not (memory_equal and video_equal and acknowledged):
                    branch_failures.append({"branch": branch + 1, "memory_equal": memory_equal, "video_equal": video_equal, "acknowledged": acknowledged})
            if branch_failures:
                raise RuntimeError(f"Linux controlled branches differed: {branch_failures}")

            raw = client._connect(5.0)
            stream = raw.__enter__()
            stream.write(companion.encode_message({
                "protocol": 2, "type": "request", "request_id": uuid.uuid4().hex,
                "token": token, "operation": "run_timeline",
                "payload": {"clock": "video_frame", "events": [
                    {"at": 0, "order": 0, "kind": "key_make", "key_id": 86},
                    {"at": 1000, "order": 1, "kind": "key_break", "key_id": 86},
                ]},
            }))
            time.sleep(0.1)
            raw.__exit__(None, None, None)
            time.sleep(0.1)
            after_disconnect = client.request("status")
            result = {
                "status": "pass",
                "pid": process.pid,
                "build": status["build"],
                "capabilities": capabilities,
                "timeline": {"actual_counters": [event["actual_counter"] for event in events], "event_types": [event["kind"] for event in events], "released_all_keys": timeline["released_all_keys"]},
                "save_events": [event["type"] for event in saved["events"]],
                "load_events": [event["type"] for event in loaded["events"]],
                "disconnect_recovered": after_disconnect["session_id"] == status["session_id"],
                "controlled_branches": {"count": 10, "failures": branch_failures},
                "state_slot": state_file.name,
            }
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            return result
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.executable.resolve(), args.report.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
