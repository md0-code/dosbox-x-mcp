# dosbox-mcp

An MCP server that drives a DOSBox-X guest **without touching the desktop**: no
focus stealing, no synthetic host keystrokes, no pointer moves.  A model can run
a DOS program start to finish while the user keeps working in front of it.

It is the capture workflow this repository grew — key injection, click
injection, data-segment patching, exact frame capture — repackaged as tools an
agent can call directly, with the game-specific knowledge pushed out into a
profile file.

## How it works

Three mechanisms, none of which involve the host's input queue or screen:

* **Keys** are appended to the guest's BIOS keyboard ring buffer at
  `0040:001A`, exactly as a real keyboard interrupt would.
* **Clicks** are written into the words the game's own INT 33h handler fills
  in, which is the state the game actually reads.  They are rewritten
  continuously for a moment, because the game's handler keeps overwriting them
  and a single write loses that race.
* **Frames** are photographed with `PrintWindow`, which renders a window
  without activating it, then sampled one pixel per guest pixel so palette
  indices come back exactly as the guest drew them.

To do any of this, the server first finds the guest's data segment inside the
emulator's host memory by searching for a byte pattern the profile says is
always there, and validating candidates against the profile's checks.  The
address is cached and revalidated on every call.

## Install

Needs Windows, Python 3.11+, and an interpreter with `mcp` and `pillow`:

```powershell
pip install mcp pillow
```

Then register it.  For Claude Code, `.mcp.json` at the repository root already
does this:

```json
{
  "mcpServers": {
    "dosbox": {
      "command": "python",
      "args": ["-m", "dosbox_mcp.server"],
      "env": {
        "PYTHONPATH": "dosbox-mcp/src",
        "DOSBOX_MCP_PROFILE": "openjp",
        "DOSBOX_MCP_EXECUTABLE": "dosbox-x/dosbox-x.exe"
      }
    }
  }
}
```

| Variable | Meaning |
| --- | --- |
| `DOSBOX_MCP_PROFILE_DIR` | where profiles live (default `dosbox-mcp/profiles`) |
| `DOSBOX_MCP_PROFILE` | default profile name |
| `DOSBOX_MCP_EXECUTABLE` | `dosbox-x.exe` to launch |
| `DOSBOX_MCP_OUTPUT_DIR` | default directory for captures |

## Tools

| Tool | What it does |
| --- | --- |
| `dosbox_profiles` | list profiles; show one profile's named offsets |
| `dosbox_launch` | start DOSBox-X with a config and wait until the guest is drivable |
| `dosbox_attach` | attach to an already-running emulator by pid |
| `dosbox_sessions` | attached sessions plus any other DOSBox-X windows |
| `dosbox_quit` | end a session (only one this server launched, unless forced) |
| `dosbox_send_keys` | type into the BIOS keyboard ring |
| `dosbox_click` | click at a guest coordinate through the game's mouse words |
| `dosbox_hold_buttons` | hold buttons down, optionally until a memory test passes |
| `dosbox_read_memory` | read the data segment as hex, numbers or text |
| `dosbox_write_memory` | patch the data segment to set up a state |
| `dosbox_dump_segment` | write the whole 64 KiB segment to a file |
| `dosbox_wait_for` | block until a memory field satisfies a test |
| `dosbox_view_screen` | return the current frame as an image to look at |
| `dosbox_capture_screen` | save an exact frame: `.rgb` indices, `.png`, and the same-instant `_ds.bin` |

Anywhere an offset is accepted, a profile symbol name works too — `treasury`
rather than `0x634A`.

A typical session:

```
dosbox_launch(config="dosbox-x/ctrl.conf")     → pid
dosbox_view_screen()                            → see where the game is
dosbox_write_memory(writes=[{offset: "treasury", value: 9000, size: 2}])
dosbox_click(x=168, y=104)                      → open a province
dosbox_wait_for(offset="race_player_state", equals=7, timeout_seconds=300)
dosbox_capture_screen(stem="tools/ground_truth/screen_race_result")
```

## Profiles

A profile is one JSON file describing a game: how to recognise its data
segment, where it keeps its mouse state, its screen size and palette, and any
named offsets worth exposing.  `profiles/openjp.json` is this repository's profile.

```json
{
  "name": "openjp",
  "ds_segment": "0x1234",
  "marker": { "source_dump": "../../tools/ground_truth/jp2d_ds.bin",
              "offset": "0x0100", "length": "0x64" },
  "checks": [ { "kind": "cstring_via_pointer", "pointer": "0x0200", "value": "game" } ],
  "mouse": { "buttons": "0x00B2", "position": "0x00B6" },
  "screen": { "width": 320, "height": 200,
              "palette": { "file": "openjp_palette.bin", "format": "rgb6" } },
  "symbols": { "lives": { "offset": "0x1234", "size": 1, "description": "Lives left." } }
}
```

The marker can come from a reference dump (`source_dump` + `offset` + `length`)
or be inlined as `bytes` hex.  Checks available: `cstring_via_pointer`, `max`,
`max_range`, `equals` — enough to make a false positive vanishingly unlikely,
which matters because the alternative is patching a random allocation.

`ds_segment` is the guest segment the data lives at; it lets the server find
guest physical zero, and therefore the BIOS keyboard buffer.  If it is wrong or
absent the server falls back to locating the keyboard buffer by signature, and
only key injection is lost if that also fails.

To profile a new game: dump a known-good segment (any memory scanner will do,
or `dosbox_dump_segment` once a rough marker works), pick 50–100 bytes that
never change, and add two or three cheap checks over fields with narrow
domains.

## Caveats

* Windows only — it uses `ReadProcessMemory`, `WriteProcessMemory` and
  `PrintWindow`.
* The emulator must run at the same integrity level as the server; an elevated
  DOSBox-X cannot be opened by an unelevated client.
* One emulator per profile at a time.  Two guests running the same game make
  the segment scan ambiguous, and the server refuses rather than guessing.
* `dosbox_capture_screen` resizes the emulator window to get an exact integer
  scale.  That is the one visible effect it has on the desktop.
* A click can advance two "click to continue" pages in some games; send a key
  when paging through lists.
* Writes into a running game are not undoable, and the game may recompute a
  field right after you set it — patch at a moment where it will be read before
  it is recomputed.
