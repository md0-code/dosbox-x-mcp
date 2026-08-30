#!/usr/bin/env python3
"""Live exact-reference tests for companion planar, text, and VESA capture."""
from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from pathlib import Path

MCP_ROOT = Path(__file__).resolve().parents[1]


def plane_fill(mask: int, value: int, count: int) -> bytes:
    return bytes((0xBA, 0xC4, 0x03, 0xB8, 0x02, mask, 0xEF, 0x31, 0xFF, 0xB9, count & 0xFF, count >> 8, 0xB0, value, 0xF3, 0xAA))


def mode_x_program() -> bytes:
    code = bytearray((0xB8, 0x13, 0x00, 0xCD, 0x10))
    code += bytes((0xBA, 0xC4, 0x03, 0xB8, 0x04, 0x06, 0xEF))
    code += bytes((0xBA, 0xCE, 0x03, 0xB8, 0x05, 0x40, 0xEF, 0xB8, 0x06, 0x05, 0xEF))
    code += bytes((0xBA, 0xD4, 0x03, 0xB8, 0x14, 0x00, 0xEF, 0xB8, 0x17, 0xE3, 0xEF))
    code += bytes((0xB8, 0x00, 0xA0, 0x8E, 0xC0, 0xFC))
    for plane in range(4):
        code += plane_fill(1 << plane, plane + 1, 16000)
    code += b"\xEB\xFE"
    return bytes(code)


def ega_program() -> bytes:
    code = bytearray((0xB8, 0x0D, 0x00, 0xCD, 0x10, 0xB8, 0x00, 0xA0, 0x8E, 0xC0, 0xFC))
    for plane, value in enumerate((0xAA, 0xCC, 0xF0, 0x00)):
        code += plane_fill(1 << plane, value, 8000)
    code += b"\xEB\xFE"
    return bytes(code)


def text_program() -> bytes:
    # Mode 3; write A/1F and B/80 to the first two cells; retain the font plane.
    return bytes.fromhex("b80300cd10 b800b8 8ec0 b8411f 26a30000 b84280 26a30200 ebfe".replace(" ", ""))


def vesa_program() -> bytes:
    # VBE mode 101h (640x480x8), then four independently known pixels.
    return bytes.fromhex("b8024f bb0101 cd10 b800a0 8ec0 31ff b011aa b022aa b033aa b044aa ebfe".replace(" ", ""))


def run_case(executable: Path, root: Path, name: str, program: bytes, expected: bytes, expected_layout: str = "plane_interleaved") -> dict[str, object]:
    sys.path.insert(0, str(MCP_ROOT / "src"))
    from dosbox_mcp import emulator, vga
    from dosbox_mcp.companion import CompanionClient

    case = root / name
    case.mkdir(parents=True, exist_ok=True)
    (case / f"{name}.COM").write_bytes(program)
    config = case / "dosbox.conf"
    config.write_text(
        "[sdl]\noutput=surface\n[dosbox]\nmachine=svga_s3\n[cpu]\ncore=normal\ncycles=max\n[autoexec]\n"
        f"mount C {case}\nC:\n{name}.COM\n",
        encoding="ascii",
    )
    token = secrets.token_urlsafe(32)
    process, _ = emulator.launch(executable, config, save_directory=case / "state", companion_token=token)
    client = CompanionClient(process.pid, token)
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                status = client.negotiate()
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        snapshot = client.request("video_snapshot", {"after_frames": 200, "max_bytes": 1024 * 1024})
        decoded = vga.decode_companion_snapshot(snapshot)
        pixels = decoded["pixels"]
        if pixels != expected:
            first = next((index for index, (actual, wanted) in enumerate(zip(pixels, expected)) if actual != wanted), None)
            raw = decoded["raw"]
            raise RuntimeError(f"{name} decoded pixels do not match the independent exact reference: mode={snapshot['mode_name']} layout={snapshot['layout']} dimensions={snapshot['width']}x{snapshot['height']} pitch={snapshot['pitch']}/{snapshot['plane_pitch']} start={snapshot['start_address']} first_mismatch={first} raw_nonzero={sum(value != 0 for value in raw)} raw_first_nonzero={next((index for index, value in enumerate(raw) if value), None)} raw={raw[:64].hex()} actual={pixels[:32].hex()} expected={expected[:32].hex()}")
        if snapshot["layout"] != expected_layout or (expected_layout == "plane_interleaved" and snapshot["chained"]):
            raise RuntimeError(f"{name} exposed the wrong backing layout: {snapshot['layout']}")
        report = {
            "case": name,
            "status": status,
            "mode": {key: snapshot[key] for key in ("mode_id", "mode_name", "width", "height", "pitch", "plane_pitch", "start_address", "bpp", "layout", "chained")},
            "reference_bytes": len(expected),
            "exact_match": True,
        }
        (case / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        process.terminate()


def run_text_case(executable: Path, root: Path) -> dict[str, object]:
    sys.path.insert(0, str(MCP_ROOT / "src"))
    from dosbox_mcp import emulator, vga
    from dosbox_mcp.companion import CompanionClient

    case = root / "TEXT"
    case.mkdir(parents=True, exist_ok=True)
    (case / "TEXT.COM").write_bytes(text_program())
    config = case / "dosbox.conf"
    config.write_text(
        "[sdl]\noutput=surface\n[dosbox]\nmachine=svga_s3\n[cpu]\ncore=normal\ncycles=max\n[autoexec]\n"
        f"mount C {case}\nC:\nTEXT.COM\n",
        encoding="ascii",
    )
    token = secrets.token_urlsafe(32)
    process, _ = emulator.launch(executable, config, save_directory=case / "state", companion_token=token)
    client = CompanionClient(process.pid, token)
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
        snapshot = client.request("video_snapshot", {"after_frames": 200, "max_bytes": 1024 * 1024})
        decoded = vga.decode_companion_snapshot(snapshot)
        cells = decoded["pixels"]
        if decoded["kind"] != "text" or cells[0][0]["character"] != 65 or cells[0][0]["attribute"] != 0x1F or cells[0][1]["character"] != 66 or cells[0][1]["attribute"] != 0x80:
            raise RuntimeError(f"text cells/attributes do not match independent DOS writes: {cells[0][:2]} raw={decoded['raw'][:64].hex()}")
        if not decoded["font_plane"] or not any(decoded["font_plane"]):
            raise RuntimeError("text capture did not retain VGA font-plane evidence")
        report = {"case": "TEXT", "mode": {key: snapshot[key] for key in ("mode_id", "mode_name", "text_columns", "text_rows", "font_height", "char9dot", "layout")}, "exact_cells": True, "font_plane_bytes": len(decoded["font_plane"]), "font_plane_retained": True}
        (case / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        process.terminate()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dosbox-executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mode_x_reference = bytes((1, 2, 3, 4)) * 16000
    ega_reference = bytes((7, 6, 5, 4, 3, 2, 1, 0)) * 8000
    reports = [
        run_case(args.dosbox_executable.resolve(), args.output.resolve(), "MODEX", mode_x_program(), mode_x_reference),
        run_case(args.dosbox_executable.resolve(), args.output.resolve(), "EGA16", ega_program(), ega_reference),
        run_text_case(args.dosbox_executable.resolve(), args.output.resolve()),
        run_case(args.dosbox_executable.resolve(), args.output.resolve(), "VESA8", vesa_program(), bytes((0x11, 0x22, 0x33, 0x44)) + bytes(640 * 480 - 4), "linear_packed"),
    ]
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
