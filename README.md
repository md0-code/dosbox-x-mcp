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
| **Drive it** | Menu keys through the BIOS ring; authentic simultaneous make/break holds through the companion DOSBox-X hook; clicks through the game's post-INT-33h mouse words. None goes near the host's input queue. |
| **See it** | Read the framebuffer straight out of the emulator's video memory: exact indices, no window, no scaling, no cost to the guest — confirmed against a reference frame. Or photograph the window when you must. |
| **Experiment** | Compare exact frames, compensate camera motion, extract and track neutral changed regions, branch actions from one isolated save state, probe controls, scan unknown memory, and retain evidence separately from profiles. |
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
* **Menu keys** are appended to the guest's BIOS keyboard ring.
* **Held keys** enter DOSBox-X's own `KEYBOARD_AddKey` path through a small,
  versioned companion hook. This produces real make/break scan codes for IRQ1
  games, supports simultaneous keys, and never activates the emulator window.
* **Clicks** are written into the words the game's own INT 33h handler fills
  in — the state the game actually reads. They are rewritten continuously for a
  moment, because the game's handler keeps overwriting them and a single write
  loses that race.
* **Frames** come out of video memory. DOSBox stores a chain-4 page the way the
  hardware does — CPU offset `o` at linear `4 * (o & ~3) + (o & 3)` — so taking
  every sixteenth group of four turns a region back into a screen. Finding
  *which* bytes are the screen takes a reference frame: an emulator's memory is
  full of picture-like data that is not the picture, so a page is either
  confirmed byte for byte against a reference or reported as unconfirmed. See
  [Reading the screen](#reading-the-screen).
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
| `DOSBOX_MCP_OUTPUT_DIR` | where a relative output or reference path is taken from |

Every path a tool reads or writes follows one rule: absolute is taken as given,
relative is under `DOSBOX_MCP_OUTPUT_DIR`. Tools return the resolved path, so
there is never any doubt where a file went.

Captures can leave guest memory unchanged while creating parent directories
and creating or overwriting host files. `DOSBOX_MCP_OUTPUT_DIR` is the base for relative
paths, not an enforced boundary; absolute paths are supported. Use distinct
filenames to retain separate observations. Tool annotations describe supported
side effects; they do not enforce OS permissions or replace the client's approval
policy. See [tool annotation contracts](docs/TOOL_ANNOTATIONS.md).

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
| `dosbox_key_state` | set authentic make/break state with automatic lease release |
| `dosbox_hold_keys` | hold simultaneous authentic keys for a bounded duration |
| `dosbox_run_timeline` | schedule ordered input, capture, memory and trace events against emulator counters |
| `dosbox_companion_status` | report protocol-v2 capabilities, session identity and guest-time counters |
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
| `dosbox_compare_frames` | exact differences, changed regions and neutral tracks across saved indexed frames |
| `dosbox_act_and_observe` | one bounded action plus its visual and memory consequences |
| `dosbox_try_actions` | restore one isolated save-state baseline before each labelled trial |
| `dosbox_probe_controls` | rank repeated authentic held-key responses above idle motion |
| `dosbox_qualify_controls` | rank camera, local-visual, memory and transition evidence behind automatic quality gates |
| `dosbox_evaluate_experiment` | evaluate strict or exploratory experiment-quality gates without running a trial |
| `dosbox_estimate_camera_path` | build a multi-frame consensus camera path with ambiguity, outliers and exact loop closure |
| `dosbox_build_world_mosaic` | stitch indexed frames in measured camera coordinates, retaining conflicts |
| `dosbox_infer_movement_constraints` | find repeated requested-direction stalls without calling them collisions |
| `dosbox_scan_changes` | rank 1-, 2- or 4-byte fields across experiment/control snapshots |
| `dosbox_correlate_memory` | correlate a visual numeric series with candidate memory series |
| `dosbox_evidence_notebook` | persist observed, inferred and confirmed claims outside profiles |
| `dosbox_emulator_command` | fire one of DOSBox-X's own menu items |
| `dosbox_record` | record OPL, MIDI or WAVE output to a file |
| `dosbox_find_cave` | find zero-filled runs big enough for a trace |
| `dosbox_install_trace` | redirect a live `CALL` through a recording cave |
| `dosbox_read_trace` | read what a trace has recorded |
| `dosbox_remove_trace` | restore the call site and blank the cave |
| `dosbox_observe_writes` | attribute bounded guest-memory writes to exact runtime CS:IP writers |
| `dosbox_capture_video_snapshot` | capture acknowledged raw VGA planes/pages and same-frame DAC state |
| `dosbox_decode_video_snapshot` | exactly decode a stored Mode X, planar EGA/VGA, text or packed VESA snapshot |
| `dosbox_create_experiment_bundle` | finalise raw and derived evidence into an immutable, hashed bundle |
| `dosbox_verify_experiment_bundle` | verify every bundle artifact and manifest hash offline |
| `dosbox_reproduce_experiment_bundle` | reproduce stored comparisons, camera paths, mosaics, tiles and constraints offline |

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

## Reading the screen

`dosbox_read_framebuffer` returns the palette indices the guest wrote, at the
instant of the read. It is exact, it cannot catch a half-drawn frame, and it
costs the guest nothing — which is why it is the path to prefer for anything
being measured.

**It needs a reference frame, and it says so rather than guessing.** Locating
the page means finding 64,000 bytes among hundreds of megabytes, and coherence
alone is not enough: measured against a live game, a blind scan returned a page
scoring 0.999 that was a decoded sprite bank, not the screen. So pass
`reference=` a 64,000-byte file holding what is on screen right now, and the
page is confirmed byte for byte:

```
dosbox_read_framebuffer(reference="credits_logo.bin", stem="shots/logo")
   → page_offset 16, confirmed true, pages [16, 64016, 128016, 192016]
```

Where a reference comes from:

* **A port under development has one for free** — its own render of the same
  screen. That is also the comparison worth making: if the two agree byte for
  byte, the port's renderer is right.
* **Any earlier confirmed capture** of the same screen re-confirms it.
* **A window photograph**, when the profile carries a palette — this is
  automatic, and needs no argument.

Once confirmed the location is cached, so every later read and every frame of
`dosbox_watch_frames` is free. `allow_unconfirmed=true` takes the blind scan's
guess for anyone who wants to check it themselves.

The page is not necessarily where you would guess: it begins where the CRTC
start address puts it, which is a four-byte boundary and nothing coarser. The
one measured live sat at offset 16.

## Perception and controlled experiments

For the complete acknowledged workflow—from an atomic guest-time timeline to a
verified, reproducible bundle—see
[Controlled DOS experiments](docs/CONTROLLED_EXPERIMENTS.md). The exercised
platform, video-mode and SecondStrike acceptance matrix is recorded in
[Validation](docs/VALIDATION.md).

Perception tools report measurements, not game meanings. A region is an exact
connected set of changed pixels; a track is a deterministic association between
regions in successive transitions. Neither is silently called a player, enemy,
wall or sprite. Memory offsets are likewise candidates until independently
confirmed.

Compare two saved indexed frames:

```text
dosbox_compare_frames(
  frames=["trial_before.bin", "trial_after.bin"],
  width=320, height=200, bridge_radius=1, min_changed_pixels=3)
→ 428 changed pixels; region 1 bbox [115,92,23,27]
```

Pass a longer sequence to obtain stable track IDs and measured displacement:

```text
dosbox_compare_frames(
  frames=["walk_000.bin", "walk_001.bin", "walk_002.bin"],
  bridge_radius=2)
→ track 3: dx +7, dy 0, age 2, confidence 0.91
```

For scrolling scenes, enable bounded global translation and screen-position
evidence. The result includes the best `(dx, dy)`, match and ambiguity scores,
translation-compensated residual regions, and neutral region categories:

```text
dosbox_compare_frames(
  frames=["walk_000.bin", "walk_001.bin", "walk_002.bin"],
  detect_global_motion=true, infer_screen_layout=true,
  max_translation_x=16, max_translation_y=16)
```

Run an idle control or a bounded action and collect both evidence streams in one
round trip:

```text
dosbox_act_and_observe(
  action={"kind":"idle"}, duration_seconds=1.0,
  memory_fields=[{"label":"clock","offset":"tick_counter","size":4}],
  memory_range={"offset":"0x6000","length":1024,"value_size":2},
  stem="experiments/idle")

dosbox_act_and_observe(
  action={"kind":"hold_keys","keys":["right"],"duration_seconds":0.35},
  memory_fields=[{"label":"candidate_x","offset":"0x19A4","size":2}],
  stem="experiments/right")
```

`kind="keys"` appends BIOS key events. It does **not** hold a physical key or
manufacture make/break state by repeatedly filling the BIOS queue. Games that
read IRQ1, port `0x60`, or their own held-key table may ignore it; the tool says
so in every result. Use `kind="hold_keys"` with a
[companion-enabled DOSBox-X build](docs/raw-keyboard.md) for those games.

Run independent trials from one baseline by launching with a dedicated state
directory. The server reserves slot 100 there and restores it before every
trial and once more before returning, without touching personal save slots:

```text
dosbox_launch(config="game.conf", profile="none",
  state_directory="experiments/state")
dosbox_try_actions(
  trials=[
    {"label":"idle","action":{"kind":"idle"}},
    {"label":"right","action":{"kind":"hold_keys","keys":["right"],
      "duration_seconds":0.35}}
  ], duration_seconds=0.75, stem="experiments/branch")
```

`dosbox_probe_controls` automates repeated idle and held-key branches and ranks
repeatable visual responses. Rankings remain neutral evidence: they do not name
a track as the player.

Compare independent branches or repeated trials with an idle control and an
opposite action:

```text
dosbox_scan_changes(
  baseline="baseline.bin", experiment="right.bin",
  control="idle.bin", opposite="left.bin",
  value_size=2, relation="increased", aligned=true)
→ DS:19A4 increased during RIGHT, stayed stable while idle,
  and changed in the opposite direction during LEFT
```

Finally, rank sampled candidate fields against a measured track coordinate:

```text
dosbox_correlate_memory(
  visual_series=[115,120,126,132],
  memory_series={"DS:19A4":[1840,1920,2016,2112]})
→ raw/difference correlation, direction, and simple fixed-scale evidence
```

Store conclusions in a separate evidence notebook instead of turning a
hypothesis into a profile fact:

```text
dosbox_evidence_notebook(
  operation="upsert", path="experiments/evidence.json",
  subject="memory:DS:19A4", status="inferred", label="horizontal state",
  confidence=0.92,
  evidence=["increased in repeated RIGHT branches", "stable in idle branches"])
```

Notebook revisions retain their history and use the explicit statuses
`observed`, `inferred`, and `confirmed`.

For auditable work, finalise inputs, timelines, observations, derivations and
gate records into an experiment bundle. Bundles are immutable, hash every
artifact, can be verified or reproduced offline, and allow confirmed notebook
entries to cite exact artifact IDs. See
[Experiment bundles](docs/EXPERIMENT_BUNDLES.md).

After camera translation is understood, stitch frames into an indexed world
mosaic. Supply measured translations or let the tool run the same bounded
translation detector. It writes the raw mosaic, a PNG, and a coverage map;
overlap conflicts remain counted rather than being silently repaired:

```text
dosbox_build_world_mosaic(
  frames=["flight_000.bin", "flight_001.bin", "flight_002.bin"],
  translations=[{"dx":-4,"dy":0},{"dx":-4,"dy":0}],
  stem="experiments/world", infer_tiles=true,
  candidate_tile_sizes=[4,8,16,32])
```

Optional tile inference ranks exact repeated block grids, then writes a raw
tile catalogue and integer tile map. Those IDs carry no terrain or collision
meaning until supported by separate evidence.

Repeated position samples can identify places where requested motion made no
forward progress. The result deliberately calls these `movement_constraints`,
not walls or collisions:

```text
dosbox_infer_movement_constraints(
  trials=[
    {"label":"right_a","direction":{"x":1,"y":0},
     "positions":[{"x":10,"y":20},{"x":11,"y":20},{"x":11,"y":20},
                  {"x":11,"y":20},{"x":11,"y":20}]},
    {"label":"right_b","direction":{"x":1,"y":0},
     "positions":[{"x":9,"y":20},{"x":11,"y":20},{"x":11,"y":20},
                  {"x":11,"y":20},{"x":11,"y":20}]}
  ])
```

Frame processing is dependency-free and practical for 320×200 indexed modes.
Large whole-screen transitions reset associations instead of creating hundreds
of long-lived tracks; crossings and temporary disappearances reduce confidence.

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
* Indexed mode 13h pages stored in DOSBox-X's chain-4 or linear backing layout
  are read out of video memory. Anything else needs `source="window"`.
* A blind framebuffer scan is a hint, not an answer, and is reported as
  unconfirmed. Give a reference frame.

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
