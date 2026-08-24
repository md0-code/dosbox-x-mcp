# dosbox-x-mcp

An MCP server that drives a DOSBox-X guest **without touching the desktop**: no
focus stealing, no synthetic host keystrokes, no pointer moves. A model can run
a DOS program start to finish while you keep working in front of it.

It began as the capture workflow for one reverse-engineering project and is now
a general aid. Given a running DOS guest it can find that guest with no prior
knowledge at all, tell you what programs are loaded and where, read and patch
any segment, sample values over time, read the screen out of video memory, drive
the emulator's own recorders, and watch code as it runs — without a debugger and
without changing a byte on disk.

## What it can do

| | |
| --- | --- |
| **Find a guest** | Locate a DOS guest from BIOS invariants alone — no profile, no marker, nothing known in advance. Walk the DOS memory chain and name every loaded program, its segment and the path it came from. |
| **Address any segment** | Another program in a launcher chain, a TSR, an overlay, the interrupt vector table, EMS pages. Read, patch, dump, search. |
| **Drive it** | Keys into the BIOS keyboard ring. Clicks into the game's own post-INT-33h mouse words. Neither goes near the host's input queue. |
| **See it** | Read the framebuffer straight out of the emulator's video memory: exact indices, no window, no scaling, no cost to the guest. Or photograph the window when you must. |
| **Measure it** | Block on a memory condition, or sample a watch list at up to 200 Hz into a TSV — every column from one snapshot, so they cannot drift apart. Watch video memory for every distinct frame and when it appeared. |
| **Record it** | The emulator's own OPL, MIDI and WAVE capture, fired without focus and without the host mapper. |
| **Instrument it** | Redirect a live `CALL` through a code cave that records registers and memory at every hit, then read it while the guest runs at full speed. Nothing on disk changes. |

## How it works

Nothing here involves the host's input queue or screen.

* **Guest discovery.** Every DOS guest has the BIOS keyboard ring's extent
  words `0x001E`/`0x003E` at `0040:0080`, a head and tail inside that extent,
  and a live `INT 21h` vector. Finding that triple in the emulator's memory
  finds guest physical zero, and from there every segment is addressable. This
  needs no profile, which is what breaks the chicken and egg a new project
  starts inside.
* **Keys** are appended to the guest's BIOS keyboard ring exactly as a real
  keyboard interrupt would.
* **Clicks** are written into the words the game's own INT 33h handler fills
  in — the state the game actually reads. They are rewritten continuously for a
  moment, because the game's handler keeps overwriting them and a single write
  loses that race.
* **Frames** come out of video memory. DOSBox stores a chain-4 page the way the
  hardware does — CPU offset `o` at linear `4 * (o & ~3) + (o & 3)` — so taking
  every sixteenth group of four turns a region back into a screen. The page is
  found either by matching a reference frame or by the fact that a drawn picture
  is far more self-similar, horizontally and vertically, than unrelated memory.
* **Traces** patch a near `CALL` to point at a run of zero padding. The cave
  calls the displaced target, preserves its flags, copies what you asked for
  into slots after its own code, and returns. The emulator dies with the run and
  takes the patch with it.

## Install

```bash
pip install "dosbox-x-mcp[all] @ git+https://github.com/md0-code/dosbox-x-mcp"
```

The extras are all optional, and named for what they buy:

| Extra | For |
| --- | --- |
| `window` | Pillow, for photographing the emulator's window |
| `fast` | numpy, for a quicker chain-4 de-interleave (there is a pure-Python fallback) |
| `x11` | python-xlib, for listing and photographing windows on Linux |

Register it with your MCP client. For Claude Code, an `.mcp.json` at your
project root:

```json
{
  "mcpServers": {
    "dosbox": {
      "command": "python",
      "args": ["-m", "dosbox_mcp.server"],
      "env": {
        "DOSBOX_MCP_PROFILE_DIR": "dosbox-x/profiles",
        "DOSBOX_MCP_EXECUTABLE": "dosbox-x/dosbox-x.exe"
      }
    }
  }
}
```

| Variable | Meaning |
| --- | --- |
| `DOSBOX_MCP_PROFILE_DIR` | where profiles live (default: the package's `profiles/`) |
| `DOSBOX_MCP_PROFILE` | default profile name; `none` for no profile |
| `DOSBOX_MCP_EXECUTABLE` | the `dosbox-x` binary to launch |
| `DOSBOX_MCP_OUTPUT_DIR` | default directory for captures |

## Tools

| Tool | What it does |
| --- | --- |
| `dosbox_capabilities` | what this host can and cannot do |
| `dosbox_profiles` | list profiles; show one profile's named offsets |
| `dosbox_launch` | start DOSBox-X and wait until the guest is drivable |
| `dosbox_attach` | attach to an already-running emulator by pid |
| `dosbox_sessions` | attached sessions, plus any other DOSBox-X windows |
| `dosbox_quit` | end a session |
| `dosbox_find_guest` | find the guest and list every loaded program — **no profile needed** |
| `dosbox_search_memory` | find a byte pattern, reported as guest `segment:offset` |
| `dosbox_send_keys` | type into the BIOS keyboard ring |
| `dosbox_click` | click through the game's own mouse words |
| `dosbox_hold_buttons` | hold buttons down, optionally until a memory test passes |
| `dosbox_read_memory` | read the data segment, or any segment |
| `dosbox_write_memory` | patch the data segment, or any segment |
| `dosbox_dump_segment` | write a whole 64 KiB segment to a file |
| `dosbox_wait_for` | block until a memory field satisfies a test |
| `dosbox_sample` | sample a watch list over time into a TSV |
| `dosbox_view_screen` | return the current frame as an image to look at |
| `dosbox_capture_screen` | save an exact frame, from video memory or the window |
| `dosbox_read_framebuffer` | read the page out of video memory |
| `dosbox_watch_frames` | every distinct frame in a stretch of time, with timings |
| `dosbox_emulator_command` | fire one of DOSBox-X's own menu items |
| `dosbox_record` | record OPL, MIDI or WAVE output to a file |
| `dosbox_find_cave` | find zero-filled runs big enough for a trace |
| `dosbox_install_trace` | redirect a live `CALL` through a recording cave |
| `dosbox_read_trace` | read what a trace has recorded |
| `dosbox_remove_trace` | restore the call site and blank the cave |

Anywhere an offset is accepted, a profile symbol name works too — `treasury`
rather than `0x634A`.

## Starting on a game nobody has profiled

A profile is not a precondition. This is the whole first session:

```
dosbox_launch(config="game.conf", profile="none")     → pid
dosbox_find_guest()
   → 640 KiB, INT 21h live, and:
       JP2D     load segment 2456    1.1 MB   C:\JP\JP2D.EXE
       JP       load segment 08A1     64 KB   C:\JP\JP.EXE
       COMMAND  load segment 0801     16 KB   C:\COMMAND.COM
dosbox_search_memory(text="sprites.dbt")   → 2456:027A
dosbox_dump_segment(path="jp2d.bin", segment="0x2456")
```

That dump is the profile's marker: pick 50–100 bytes of it that will not change
between runs, add two or three cheap checks, and every DS-relative tool starts
working by name.

## Profiles

A profile is one JSON file describing a game: how to recognise its data
segment, where it keeps its mouse state, its screen size and palette, named
offsets, watch sets and known caves.
[`profiles/example.json`](profiles/example.json) is an annotated template; the
[OpenJP](https://github.com/md0-code/OpenJP) repository has real ones written
against a shipped 1993 game.

```json
{
  "name": "example",
  "ds_segment": "0x1234",
  "marker": { "bytes": "6578616d706c652e64617400", "offset": "0x0100" },
  "checks": [ { "kind": "cstring_via_pointer", "pointer": "0x0200", "value": "game" } ],
  "mouse": { "buttons": "0x00B2", "position": "0x00B6" },
  "screen": { "width": 320, "height": 200 },
  "symbols": { "lives": { "offset": "0x1234", "size": 1, "description": "Lives left." } },
  "watch_sets": { "player": ["lives", "score", "level"] }
}
```

The marker can be inlined as `bytes` hex or sliced out of a reference dump
(`source_dump` + `offset` + `length`). Checks available: `cstring_via_pointer`,
`max`, `max_range`, `equals` — enough to make a false positive vanishingly
unlikely, which matters because the alternative is patching a random
allocation.

## Platforms

| | Windows | Linux | macOS |
| --- | --- | --- | --- |
| Guest memory, segments, search, keys, clicks | yes | yes | no backend |
| Framebuffer, sampling, traces | yes | yes | no backend |
| Window list, capture, resize | yes | with X11 | — |
| Emulator menu commands | yes | no — use an isolated display | — |
| Offscreen display | — | with `Xvfb` | — |

`dosbox_capabilities` reports all of this for the running host, so ask rather
than guess.

**On Linux**, `kernel.yama.ptrace_scope` gates access to another process's
memory exactly as integrity level does on Windows:

| Value | Effect |
| --- | --- |
| `0` | any same-uid process — `dosbox_attach` works |
| `1` (Debian and Ubuntu default) | descendants only — `dosbox_launch` works, `dosbox_attach` does not |
| `2` | `CAP_SYS_PTRACE` only |
| `3` | no attach at all |

Under the common default, launch rather than attach. The server reads the
sysctl and says so rather than surfacing a bare `EPERM`.

Linux has no way to photograph an occluded window without a compositing
manager, and Wayland has no cross-client capture at all. The answer is not to
emulate `PrintWindow` but to remove the constraint it exists to satisfy:
`dosbox_launch(isolated=true)` puts the emulator on its own `Xvfb` display,
where there is no desktop to protect and the emulator's own keyboard shortcuts
are usable without taking a keystroke from anyone.

macOS would need `task_for_pid`, and therefore either root or a signed, entitled
binary. There is no backend for it.

## Caveats

* The emulator must be reachable: same integrity level on Windows, an
  agreeable `ptrace_scope` on Linux.
* One emulator per profile at a time. Two guests running the same game make the
  segment scan ambiguous, and the server refuses rather than guessing.
* `dosbox_capture_screen` with `source="window"` resizes the emulator window to
  get an exact integer scale. That is the one visible effect this server has on
  the desktop — and the reason to prefer `source="vram"`, which is also faster
  and cannot catch a half-drawn frame.
* Photographing the window costs the *guest* real time: a filming loop stretches
  guest-visible phases by roughly 1.6x. Use the framebuffer for anything
  quantitative.
* A click can advance two "click to continue" pages in some games; send a key
  when paging through lists.
* Writes into a running game are not undoable, and the game may recompute a
  field right after you set it — patch at a moment where it will be read before
  it is recomputed.
* Trace memory captures are read through whatever `DS` holds when the cave
  runs. For a game with one data segment that is exactly right; for a routine
  that switches `DS`, capture `ds` alongside and check it.
* Only chain-4 linear mode 13h is read out of video memory. Anything else needs
  `source="window"`.

## Development

```bash
pip install -e ".[dev,all]"
pytest
```

The suite is entirely offline: a DOS guest, a memory chain, a chain-4
framebuffer and a traceable code segment are all built in a `bytearray`, so it
runs on any platform with no emulator and no game. What it does not cover is
the two system calls at the very bottom — reading and writing another process —
and the window.

## Licence

MIT.
