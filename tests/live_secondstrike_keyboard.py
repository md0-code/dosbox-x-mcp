#!/usr/bin/env python3
"""Validate raw keyboard v1 against SecondStrike's original IRQ1 state."""
from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import sys
import time
from pathlib import Path


MCP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SECONDSTRIKE = MCP_ROOT.parent


def wait_byte(session, offset: int, expected: int, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if session.read(offset, 1)[0] == expected:
            return
        time.sleep(0.01)
    seen = session.read(offset, 1)[0]
    raise RuntimeError(f"DS:{offset:04X} remained {seen:02X}, expected {expected:02X}")


async def validate(root: Path, executable: Path) -> dict[str, object]:
    sys.path.insert(0, str(MCP_ROOT / "src"))

    from dosbox_mcp import emulator, server
    from dosbox_mcp.profile import load_profile
    from dosbox_mcp.session import Registry

    profile = load_profile(root / "tools" / "profiles" / "desert.json")
    config = root / "tools" / "conf" / "desert.conf"
    process, _ = emulator.launch(executable, config)
    registry = Registry()
    session = registry.attach(process.pid, profile)
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                session.locate(True)
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.5)
        if not session.raw_keyboard_supported():
            raise RuntimeError("the launched DOSBox-X does not expose keyboard hook v1")
        server.registry = registry

        window = session.window()
        if window is None:
            raise RuntimeError("DOSBox-X has no visible window")
        user32 = ctypes.windll.user32
        user32.ShowWindow(window.handle, 6)  # SW_MINIMIZE; do not reactivate it.
        time.sleep(0.2)
        foreground = int(user32.GetForegroundWindow() or 0)
        if foreground == window.handle:
            raise RuntimeError("DOSBox-X remained foreground after minimization")

        key_state = 0x043D
        right = key_state + 0xCD
        left_ctrl = key_state + 0x1D
        space = key_state + 0x39
        left = key_state + 0xCB

        down = await server.dosbox_key_state("right", True, process.pid, 2.0)
        wait_byte(session, right, 0xFF)
        make = {
            "last_state": session.read(0x0438, 1)[0],
            "last_scancode": session.read(0x0439, 1)[0],
            "held": session.read(right, 1)[0],
        }
        up = await server.dosbox_key_state("right", False, process.pid)
        wait_byte(session, right, 0)
        release = {
            "last_state": session.read(0x0438, 1)[0],
            "last_scancode": session.read(0x0439, 1)[0],
            "held": session.read(right, 1)[0],
        }

        holding = asyncio.create_task(
            server.dosbox_hold_keys(["ctrl", "right"], 0.5, process.pid)
        )
        await asyncio.sleep(0.05)
        wait_byte(session, left_ctrl, 0xFF)
        wait_byte(session, right, 0xFF)
        simultaneous = {
            "ctrl": session.read(left_ctrl, 1)[0],
            "right": session.read(right, 1)[0],
        }
        held_result = await holding
        wait_byte(session, left_ctrl, 0)
        wait_byte(session, right, 0)

        await server.dosbox_key_state("space", True, process.pid, 0.2)
        wait_byte(session, space, 0xFF)
        wait_byte(session, space, 0, 1.0)

        await server.dosbox_key_state("left", True, process.pid, 2.0)
        wait_byte(session, left, 0xFF)
        session.close()
        wait_byte(session, left, 0)

        foreground_after = int(user32.GetForegroundWindow() or 0)
        if foreground_after != foreground:
            raise RuntimeError("raw keyboard calls changed the foreground window")

        report = {
            "dosbox_executable": str(executable),
            "pid": process.pid,
            "hook": "dosbox_x_keyboard_hook_v1",
            "make": make,
            "break": release,
            "simultaneous": simultaneous,
            "held_result": held_result,
            "lease_released": session.read(space, 1)[0] == 0,
            "session_close_released": session.read(left, 1)[0] == 0,
            "foreground_unchanged": foreground_after == foreground,
            "key_state_result": {"down": down, "up": up},
        }
        output = root / "re_artifacts" / "raw_keyboard_v1"
        output.mkdir(parents=True, exist_ok=True)
        (output / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return report
    finally:
        session.close()
        process.terminate()


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
