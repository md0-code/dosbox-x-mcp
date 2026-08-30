"""Starting DOSBox-X, and driving the facilities it does not expose to a guest.

Two things live here that are about the *emulator* rather than about the guest
inside it.

**Recording.**  DOSBox-X can write a `.dro` of every OPL register write, a
`.mid` of the MIDI stream and a `.wav` of the mixer -- which is the only
independent record of what a game's sound driver actually does.  It binds all
three to the host mapper, which is unreachable from here because keys are
injected into the BIOS ring far below SDL.  On Windows the same three actions
are native menu items and one ``PostMessage`` fires one without focus.

**An offscreen display.**  On Linux there is no menu to post to, and no way to
photograph an occluded window without a compositor.  Both problems dissolve if
the emulator has a display to itself: ``Xvfb`` costs nothing, takes the
emulator entirely off the user's desktop, and makes the host mapper usable
because there is nothing else there to take a keystroke from.

Two failures cost this project more time than everything else in the module,
so they are error text rather than documentation: ``captures=`` in the config
is relative to *the config file*, not the working directory; and DOSBox-X
starts an OPL recording at the guest's **first** write, so arming late does not
truncate the capture, it starts it somewhere else entirely.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from . import platform as host

CAPTURES_LINE = re.compile(r"^\s*captures\s*=\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)

#: What DOSBox-X names each recording, so the right new file is looked for.
#: Getting this wrong is silent: a MIDI capture searched for as `*.dro` always
#: reports "nothing recorded".
SUFFIXES = {
    "record_opl": (".dro",),
    "record_midi": (".mid", ".midi"),
    "record_wave": (".wav",),
    "record_video": (".avi",),
    "screenshot": (".png",),
}


class EmulatorError(RuntimeError):
    """The emulator could not be started or commanded."""


def captures_directory(config: Path) -> Path:
    """Where this config writes its captures.

    Relative to the config file, which is the trap: a `captures=` that looks
    right from the working directory produces one `Can't open dir` line per
    write, in a log nobody reads, and an empty capture.
    """
    try:
        text = config.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return config.parent / "capture"
    match = CAPTURES_LINE.search(text)
    if match is None:
        return config.parent / "capture"
    named = Path(match.group(1))
    return named if named.is_absolute() else (config.parent / named).resolve()


# ── an offscreen display ───────────────────────────────────────────────────


@dataclass
class Display:
    """An Xvfb display owned by one session, torn down with it."""

    number: int
    process: subprocess.Popen

    @property
    def name(self) -> str:
        return f":{self.number}"

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


def start_display(width: int = 1024, height: int = 768) -> Display:
    """Bring up an Xvfb the emulator can have to itself."""
    if not shutil.which("Xvfb"):
        raise EmulatorError(
            "Xvfb is not installed, so there is no offscreen display to run on.  "
            "Install it (apt install xvfb) or run the emulator on the user's own "
            "display and accept that window capture needs a compositor."
        )
    for number in range(90, 130):
        if os.path.exists(f"/tmp/.X{number}-lock"):
            continue
        process = subprocess.Popen(
            ["Xvfb", f":{number}", "-screen", "0", f"{width}x{height}x24", "-nolisten", "tcp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 10
        while time.time() < deadline:
            if process.poll() is not None:
                break
            if os.path.exists(f"/tmp/.X11-unix/X{number}"):
                return Display(number=number, process=process)
            time.sleep(0.2)
        if process.poll() is None:
            process.terminate()
    raise EmulatorError("could not start an Xvfb display on any of :90 to :129")


# ── launching ──────────────────────────────────────────────────────────────


def launch(
    executable: Path,
    config: Path,
    working_directory: str | None = None,
    isolated: bool = False,
    save_directory: Path | None = None,
) -> tuple[subprocess.Popen, Display | None]:
    """Start the emulator, optionally on a display of its own."""
    display: Display | None = None
    environment = dict(os.environ)
    if isolated:
        host.require("isolated_display")
        display = start_display()
        environment["DISPLAY"] = display.name
    try:
        arguments = [str(executable), "-conf", str(config)]
        if save_directory is not None:
            arguments.extend(
                [
                    "-savedir",
                    str(save_directory),
                    "-set",
                    "dosbox saveslot=100",
                    "-set",
                    "dosbox saveremark=false",
                ]
            )
        process = subprocess.Popen(
            arguments,
            cwd=working_directory or str(executable.parent),
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as failure:
        if display is not None:
            display.stop()
        raise EmulatorError(f"could not start {executable}: {failure}") from None
    return process, display


# ── commanding it ──────────────────────────────────────────────────────────


def command(pid: int, action: str) -> dict[str, object]:
    """Fire one of the emulator's own menu items, without focus."""
    host.require("menu_commands")
    handle = host.menu_window(pid)
    if handle is None:
        raise EmulatorError(
            f"process {pid} has no window with a menu.  The menu bar may be hidden; "
            f"set 'menu=true' in the config's [sdl] section."
        )
    identifier, label = host.resolve_command(handle, action)
    host.menu_command(handle, identifier)
    return {"action": action, "menu_item": label, "command_id": f"{identifier:#06x}"}


def save_branch_state(pid: int, directory: Path, timeout: float = 15.0) -> dict[str, object]:
    """Save isolated slot 100 and wait until its archive is complete."""
    target = directory / "100.sav"
    try:
        existing = target.stat()
        before = (existing.st_mtime_ns, existing.st_size)
    except OSError:
        before = None
    started = time.monotonic()
    result = command(pid, "save_state")
    previous: tuple[int, int] | None = None
    stable = 0
    while time.monotonic() - started < timeout:
        try:
            stat = target.stat()
            current = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            current = None
        if current is not None and current[1] > 0 and current != before and current == previous:
            stable += 1
            if stable >= 2:
                return {
                    **result,
                    "state_file": str(target),
                    "bytes": current[1],
                    "waited_seconds": round(time.monotonic() - started, 3),
                }
        else:
            stable = 0
        previous = current
        time.sleep(0.05)
    raise EmulatorError(f"DOSBox-X did not finish isolated state {target} within {timeout}s")


def load_branch_state(pid: int, state_file: Path, settle: float = 0.25) -> dict[str, object]:
    """Load isolated slot 100 after verifying the expected archive exists."""
    if not state_file.is_file() or state_file.stat().st_size == 0:
        raise EmulatorError(f"isolated branch state is missing or empty: {state_file}")
    result = command(pid, "load_state")
    time.sleep(settle)
    return {**result, "state_file": str(state_file), "settle_seconds": settle}


def record(
    pid: int,
    config: Path,
    action: str,
    seconds: float,
    settle: float = 0.0,
    captures: Path | None = None,
) -> dict[str, object]:
    """Arm a recording, wait, disarm, and report the file that appeared.

    Arming is the whole difficulty.  DOSBox-X begins an OPL capture at the
    guest's *first* register write, so a capture armed after the driver has
    installed does not start at the install -- it starts wherever the guest
    happens to be, and the deterministic part worth comparing is already gone.
    Arm as early as the window exists.
    """
    directory = captures or captures_directory(config)
    before = {path.name for path in directory.glob("*")} if directory.is_dir() else set()
    if settle:
        time.sleep(settle)
    started = command(pid, action)
    time.sleep(seconds)
    command(pid, action)
    time.sleep(2.0)  # DOSBox-X finishes the file after the second command

    suffixes = SUFFIXES.get(action, ())
    fresh = sorted(
        path
        for path in (directory.glob("*") if directory.is_dir() else [])
        if path.name not in before and (not suffixes or path.suffix.lower() in suffixes)
    )
    if not fresh:
        appeared = (
            sorted(p.name for p in directory.glob("*") if p.name not in before)
            if directory.is_dir()
            else []
        )
        raise EmulatorError(
            f"nothing new in {directory}.  Two usual causes: `captures=` in the "
            f"config is relative to the config file rather than the working "
            f"directory, so the emulator is writing somewhere else; or the guest "
            f"never touched the device being recorded -- an OPL capture needs the "
            f"guest to drive an AdLib, a MIDI one needs it to drive an MPU-401.  "
            f"The emulator's log says which."
            + (f"  What did appear: {', '.join(appeared)}." if appeared else "")
        )
    newest = max(fresh, key=lambda path: path.stat().st_mtime)
    return {
        **started,
        "seconds": seconds,
        "captures_directory": str(directory),
        "file": str(newest),
        "bytes": newest.stat().st_size,
        "also_appeared": [path.name for path in fresh if path != newest],
    }
