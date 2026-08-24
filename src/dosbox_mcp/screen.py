"""Photograph the emulator window and recover the guest's palette indices.

Reading the guest's own video memory is unreliable -- mode 13h's chain-4
interleave is not where a naive `A000:0000` read points, and the game's own
composition page holds only what it drew *before* copying to the visible page,
so anything finished afterwards (a command window, a confirm box, a card) is
missing.  The window is the only place the finished frame exists.

So: resize the client area to an exact integer multiple of the guest's
resolution, photograph it without activating it, and take one sample per guest
pixel -- no resampling, so the palette indices come back exactly.  A pixel that
is not a palette colour means the geometry or the palette is wrong, and an
exact capture fails rather than writing a frame that merely looks right.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .profile import Profile
from .platform import grab_client, resize_client


class CaptureError(RuntimeError):
    """The window could not be captured at the exactness that was asked for."""


@dataclass
class Capture:
    width: int
    height: int
    scale: int
    indices: bytes | None
    png_bytes: bytes
    distinct: int
    unmatched: int
    worst_colours: list[tuple[tuple[int, int, int], int]]


def _require_pillow():
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - environment problem
        raise CaptureError(
            "Pillow is required for screen capture; install it with "
            "`pip install pillow` into the interpreter running this server"
        ) from error
    return Image


def capture_window(
    handle: int,
    profile: Profile,
    scale: int | None = 2,
    settle: float = 4.0,
    exact: bool = True,
) -> Capture:
    """Capture one frame; with ``exact`` also recover the guest palette indices."""
    Image = _require_pillow()
    width, height = profile.screen_width, profile.screen_height

    if scale:
        resize_client(handle, width * scale, height * scale, settle)

    raw_w, raw_h, pixels = grab_client(handle)
    image = Image.frombuffer("RGB", (raw_w, raw_h), pixels, "raw", "BGRX", 0, 1)

    if not exact or not profile.palette:
        png = _to_png(image)
        return Capture(raw_w, raw_h, 0, None, png, 0, 0, [])

    if raw_w % width or raw_h % height or raw_w // width != raw_h // height:
        raise CaptureError(
            f"the client area is {raw_w}x{raw_h}, not an integer multiple of "
            f"{width}x{height}, so an exact capture would need resampling.  "
            f"Pass a scale that fits the screen, or exact=false for a plain "
            f"screenshot.  A client of {raw_w}x{raw_h} usually means the guest "
            f"is in text mode -- the program has exited to DOS."
        )
    got_scale = raw_w // width

    lookup: dict[tuple[int, int, int], int] = {}
    for index, rgb in enumerate(profile.palette):
        lookup.setdefault(tuple(rgb), index)

    source = image.load()
    indices = bytearray(width * height)
    missing: dict[tuple[int, int, int], int] = {}
    for y in range(height):
        row = y * width
        for x in range(width):
            rgb = source[x * got_scale, y * got_scale]
            index = lookup.get(rgb)
            if index is None:
                missing[rgb] = missing.get(rgb, 0) + 1
                index = 0
            indices[row + x] = index

    worst = sorted(missing.items(), key=lambda item: -item[1])[:6]
    if missing:
        detail = ", ".join(f"rgb{colour} x{count}" for colour, count in worst)
        raise CaptureError(
            f"{sum(missing.values())} of {width * height} pixels are not palette "
            f"colours ({len(missing)} distinct): {detail}.  The geometry or the "
            f"palette is wrong, or the frame was caught mid-fade; nothing was "
            f"written.  Use exact=false to keep the screenshot anyway."
        )

    flat = [component for rgb in profile.palette for component in rgb]
    preview = Image.frombytes("P", (width, height), bytes(indices))
    preview.putpalette(flat + [0] * (768 - len(flat)))
    png = _to_png(preview.convert("RGB"))
    return Capture(
        width=width,
        height=height,
        scale=got_scale,
        indices=bytes(indices),
        png_bytes=png,
        distinct=len(set(indices)),
        unmatched=0,
        worst_colours=worst,
    )


def _to_png(image) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def downscale_png(png: bytes, max_width: int) -> bytes:
    """Shrink a PNG so a screenshot returned to a model stays cheap."""
    Image = _require_pillow()
    from io import BytesIO

    image = Image.open(BytesIO(png))
    if image.width <= max_width:
        return png
    ratio = max_width / image.width
    resized = image.resize((max_width, max(1, round(image.height * ratio))), Image.NEAREST)
    return _to_png(resized.convert("RGB"))


def write_outputs(stem: Path, capture: Capture, segment: bytes | None) -> list[str]:
    """Write ``<stem>.rgb`` / ``.png`` / ``_ds.bin`` and report what was written."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    if capture.indices is not None:
        rgb = stem.with_suffix(".rgb")
        rgb.write_bytes(capture.indices)
        written.append(str(rgb))
    png = stem.with_suffix(".png")
    png.write_bytes(capture.png_bytes)
    written.append(str(png))
    if segment is not None:
        dump = stem.with_name(stem.name + "_ds.bin")
        dump.write_bytes(segment)
        written.append(str(dump))
    return written
