"""Dependency-free world stitching and movement-constraint evidence."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


class WorldError(ValueError):
    pass


@dataclass(frozen=True)
class Mosaic:
    pixels: bytes
    coverage: bytes
    width: int
    height: int
    placements: tuple[dict[str, object], ...]
    observed_pixels: int
    overlap_observations: int
    conflict_observations: int
    conflict_pixels: int

    def report(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "frame_count": len(self.placements),
            "placements": list(self.placements),
            "observed_pixels": self.observed_pixels,
            "coverage_fraction": round(
                self.observed_pixels / (self.width * self.height), 8
            ),
            "overlap_observations": self.overlap_observations,
            "conflict_observations": self.conflict_observations,
            "conflict_pixels": self.conflict_pixels,
            "conflict_fraction_of_overlap": round(
                (
                    self.conflict_observations / self.overlap_observations
                    if self.overlap_observations
                    else 0.0
                ),
                8,
            ),
            "composition": (
                "first observed palette index is retained; coverage and conflicts "
                "preserve where later observations agreed or differed"
            ),
            "epistemic_status": (
                "stitched indexed-frame evidence in inferred/provided camera coordinates; "
                "conflicts may be animation, entities, bad translation, or world change"
            ),
        }


@dataclass(frozen=True)
class TileStructure:
    tile_bytes: bytes
    tile_map: tuple[tuple[int, ...], ...]
    tile_width: int
    tile_height: int
    origin: tuple[int, int]
    candidates: tuple[dict[str, object], ...]

    def report(self, include_tile_map: bool = True) -> dict[str, object]:
        observed = sum(value >= 0 for row in self.tile_map for value in row)
        result = {
            "best_candidate": self.candidates[0],
            "candidates": list(self.candidates),
            "tile_width": self.tile_width,
            "tile_height": self.tile_height,
            "grid_origin": list(self.origin),
            "grid_width": len(self.tile_map[0]) if self.tile_map else 0,
            "grid_height": len(self.tile_map),
            "observed_grid_cells": observed,
            "tile_count": len(self.tile_bytes) // (self.tile_width * self.tile_height),
            "unobserved_tile_id": -1,
            "epistemic_status": (
                "ranked exact repeated indexed blocks; tile IDs are raw catalogue "
                "indices and do not imply terrain or collision semantics"
            ),
        }
        if include_tile_map:
            result["tile_map"] = [list(row) for row in self.tile_map]
        else:
            result["tile_map_omitted"] = True
        return result


def build_mosaic(
    frames: Sequence[bytes],
    translations: Sequence[tuple[int, int]],
    width: int,
    height: int,
    max_mosaic_pixels: int = 64 * 1024 * 1024,
) -> Mosaic:
    """Stitch frames using camera displacement derived from visual translation.

    A translation ``(dx, dy)`` means a pixel in the prior frame appears at
    that displacement in the next frame. The next camera origin therefore
    advances by ``(-dx, -dy)`` in mosaic coordinates.
    """
    if width <= 0 or height <= 0:
        raise WorldError("mosaic frame dimensions must be positive")
    if max_mosaic_pixels < 1:
        raise WorldError("max_mosaic_pixels must be positive")
    if len(frames) < 2:
        raise WorldError("a world mosaic needs at least two frames")
    if len(translations) != len(frames) - 1:
        raise WorldError("a world mosaic needs one translation per frame transition")
    expected = width * height
    if any(len(frame) != expected for frame in frames):
        raise WorldError("every mosaic frame must contain width*height indexed pixels")

    origins = [(0, 0)]
    camera_x = camera_y = 0
    for dx, dy in translations:
        if not isinstance(dx, int) or not isinstance(dy, int):
            raise WorldError("mosaic translations must be integer pixel displacements")
        camera_x -= dx
        camera_y -= dy
        origins.append((camera_x, camera_y))

    min_x = min(x for x, _ in origins)
    min_y = min(y for _, y in origins)
    max_x = max(x + width for x, _ in origins)
    max_y = max(y + height for _, y in origins)
    mosaic_width = max_x - min_x
    mosaic_height = max_y - min_y
    size = mosaic_width * mosaic_height
    if size > max_mosaic_pixels:
        raise WorldError(
            f"mosaic would contain {size} pixels; limit is {max_mosaic_pixels}"
        )
    pixels = bytearray(size)
    coverage = bytearray(size)
    conflicts = bytearray(size)
    overlap_observations = conflict_observations = 0
    placements: list[dict[str, object]] = []

    for frame_index, (frame, (origin_x, origin_y)) in enumerate(zip(frames, origins)):
        placed_x, placed_y = origin_x - min_x, origin_y - min_y
        placements.append(
            {
                "frame": frame_index,
                "camera_origin": [origin_x, origin_y],
                "mosaic_offset": [placed_x, placed_y],
            }
        )
        for y in range(height):
            source = y * width
            target = (placed_y + y) * mosaic_width + placed_x
            for x in range(width):
                at = target + x
                value = frame[source + x]
                if coverage[at] == 0:
                    pixels[at] = value
                else:
                    overlap_observations += 1
                    if pixels[at] != value:
                        conflict_observations += 1
                        conflicts[at] = 1
                coverage[at] = min(255, coverage[at] + 1)

    return Mosaic(
        pixels=bytes(pixels),
        coverage=bytes(coverage),
        width=mosaic_width,
        height=mosaic_height,
        placements=tuple(placements),
        observed_pixels=sum(value > 0 for value in coverage),
        overlap_observations=overlap_observations,
        conflict_observations=conflict_observations,
        conflict_pixels=sum(conflicts),
    )


def _blocks_for_grid(
    pixels: bytes,
    coverage: bytes,
    width: int,
    height: int,
    size: int,
    phase_x: int,
    phase_y: int,
) -> tuple[list[bytes], list[tuple[int, int]]]:
    blocks: list[bytes] = []
    cells: list[tuple[int, int]] = []
    grid_y = 0
    for y in range(phase_y, height - size + 1, size):
        grid_x = 0
        for x in range(phase_x, width - size + 1, size):
            rows = []
            complete = True
            for row in range(y, y + size):
                start = row * width + x
                if 0 in coverage[start : start + size]:
                    complete = False
                    break
                rows.append(pixels[start : start + size])
            if complete:
                blocks.append(b"".join(rows))
                cells.append((grid_x, grid_y))
            grid_x += 1
        grid_y += 1
    return blocks, cells


def infer_tile_structure(
    pixels: bytes,
    coverage: bytes,
    width: int,
    height: int,
    *,
    candidate_sizes: Sequence[int] = (4, 8, 16, 32),
    min_tiles: int = 8,
    max_inference_pixels: int = 4 * 1024 * 1024,
    max_grid_cells: int = 1_000_000,
) -> TileStructure:
    """Rank exact repeated square-block grids and retain a raw tile catalogue."""
    expected = width * height
    if width <= 0 or height <= 0 or len(pixels) != expected or len(coverage) != expected:
        raise WorldError("tile inference needs equal width*height pixels and coverage")
    if min_tiles < 2:
        raise WorldError("tile inference min_tiles must be at least two")
    if expected > max_inference_pixels:
        raise WorldError(
            f"tile inference received {expected} pixels; limit is {max_inference_pixels}"
        )
    if max_grid_cells < 1:
        raise WorldError("max_grid_cells must be positive")
    sizes = list(dict.fromkeys(candidate_sizes))
    if not sizes or any(not isinstance(size, int) or size < 2 for size in sizes):
        raise WorldError("candidate tile sizes must be integers of at least two pixels")
    if any(size > width or size > height for size in sizes):
        raise WorldError("candidate tile sizes cannot exceed the mosaic dimensions")

    ranked: list[dict[str, object]] = []
    for size in sizes:
        if (width // size) * (height // size) > max_grid_cells:
            continue
        best: dict[str, object] | None = None
        for phase_y in range(size):
            for phase_x in range(size):
                blocks, _ = _blocks_for_grid(
                    pixels, coverage, width, height, size, phase_x, phase_y
                )
                if len(blocks) < min_tiles:
                    continue
                counts: dict[bytes, int] = {}
                for block in blocks:
                    counts[block] = counts.get(block, 0) + 1
                repeated_instances = sum(count for count in counts.values() if count > 1)
                repeated_kinds = sum(count > 1 for count in counts.values())
                repeat_fraction = repeated_instances / len(blocks)
                support = min(1.0, len(blocks) / 32)
                score = repeat_fraction * support
                candidate = {
                    "tile_size": size,
                    "grid_phase": [phase_x, phase_y],
                    "observed_tiles": len(blocks),
                    "unique_tiles": len(counts),
                    "repeated_tile_kinds": repeated_kinds,
                    "repeated_instances": repeated_instances,
                    "repeat_fraction": round(repeat_fraction, 8),
                    "score": round(score, 8),
                }
                if best is None or (
                    float(candidate["score"]),
                    int(candidate["observed_tiles"]),
                    -phase_y,
                    -phase_x,
                ) > (
                    float(best["score"]),
                    int(best["observed_tiles"]),
                    -int(best["grid_phase"][1]),
                    -int(best["grid_phase"][0]),
                ):
                    best = candidate
        if best is not None:
            ranked.append(best)
    if not ranked:
        raise WorldError("no candidate grid had enough fully observed tiles")
    ranked.sort(
        key=lambda item: (
            -float(item["score"]),
            -int(item["observed_tiles"]),
            int(item["tile_size"]),
            item["grid_phase"],
        )
    )

    chosen = ranked[0]
    size = int(chosen["tile_size"])
    phase_x, phase_y = (int(value) for value in chosen["grid_phase"])
    blocks, cells = _blocks_for_grid(
        pixels, coverage, width, height, size, phase_x, phase_y
    )
    grid_width = max(0, (width - phase_x) // size)
    grid_height = max(0, (height - phase_y) // size)
    tile_map = [[-1] * grid_width for _ in range(grid_height)]
    catalogue: list[bytes] = []
    identifiers: dict[bytes, int] = {}
    for block, (x, y) in zip(blocks, cells):
        identifier = identifiers.get(block)
        if identifier is None:
            identifier = len(catalogue)
            identifiers[block] = identifier
            catalogue.append(block)
        tile_map[y][x] = identifier
    return TileStructure(
        tile_bytes=b"".join(catalogue),
        tile_map=tuple(tuple(row) for row in tile_map),
        tile_width=size,
        tile_height=size,
        origin=(phase_x, phase_y),
        candidates=tuple(ranked),
    )


def infer_movement_constraints(
    trials: Sequence[dict[str, object]],
    *,
    max_projected_progress: float = 0.25,
    min_stalled_steps: int = 3,
    position_tolerance: float = 2.0,
    min_trial_support: int = 2,
    max_reported_episodes: int = 1000,
    max_evidence_per_candidate: int = 100,
) -> dict[str, object]:
    """Find repeated locations where requested motion makes no forward progress."""
    if max_projected_progress < 0:
        raise WorldError("max_projected_progress cannot be negative")
    if min_stalled_steps < 1 or min_trial_support < 1:
        raise WorldError("stall length and trial support must be positive")
    if position_tolerance < 0:
        raise WorldError("position_tolerance cannot be negative")
    if max_reported_episodes < 1 or max_evidence_per_candidate < 1:
        raise WorldError("movement evidence report limits must be positive")

    episodes: list[dict[str, object]] = []
    for trial in trials:
        label = str(trial.get("label", "")).strip()
        direction = trial.get("direction")
        positions = trial.get("positions")
        if not label:
            raise WorldError("every movement trial needs a label")
        if not isinstance(direction, (list, tuple)) or len(direction) != 2:
            raise WorldError(f"movement trial {label!r} needs a two-value direction")
        if not isinstance(positions, (list, tuple)) or len(positions) < 2:
            raise WorldError(f"movement trial {label!r} needs at least two positions")
        dx, dy = float(direction[0]), float(direction[1])
        magnitude = math.hypot(dx, dy)
        if not math.isfinite(magnitude) or magnitude == 0:
            raise WorldError(f"movement trial {label!r} direction must be finite and non-zero")
        unit_x, unit_y = dx / magnitude, dy / magnitude
        parsed: list[tuple[float, float]] = []
        for position in positions:
            if not isinstance(position, (list, tuple)) or len(position) != 2:
                raise WorldError(f"movement trial {label!r} positions must have two values")
            x, y = float(position[0]), float(position[1])
            if not math.isfinite(x) or not math.isfinite(y):
                raise WorldError(f"movement trial {label!r} positions must be finite")
            parsed.append((x, y))

        projected = [
            (after[0] - before[0]) * unit_x + (after[1] - before[1]) * unit_y
            for before, after in zip(parsed, parsed[1:])
        ]
        run_start: int | None = None
        for index in range(len(projected) + 1):
            stalled = (
                index < len(projected)
                and projected[index] <= max_projected_progress
            )
            if stalled and run_start is None:
                run_start = index
            if not stalled and run_start is not None:
                length = index - run_start
                if length >= min_stalled_steps:
                    segment = projected[run_start:index]
                    point = parsed[run_start]
                    episodes.append(
                        {
                            "trial": label,
                            "direction": [round(unit_x, 6), round(unit_y, 6)],
                            "start_step": run_start,
                            "end_step": index,
                            "stalled_steps": length,
                            "position": [round(point[0], 6), round(point[1], 6)],
                            "mean_projected_progress": round(sum(segment) / length, 8),
                        }
                    )
                run_start = None

    groups: list[dict[str, object]] = []
    for episode in episodes:
        px, py = episode["position"]
        direction = episode["direction"]
        match = None
        for group in groups:
            if group["direction"] != direction:
                continue
            gx, gy = group["position_sum"]
            count = int(group["episode_count"])
            if math.hypot(px - gx / count, py - gy / count) <= position_tolerance:
                match = group
                break
        if match is None:
            match = {
                "direction": direction,
                "position_sum": [px, py],
                "episode_count": 1,
                "trials": {episode["trial"]},
                "stall_steps": int(episode["stalled_steps"]),
                "episodes": [episode],
            }
            groups.append(match)
        else:
            match["position_sum"][0] += px
            match["position_sum"][1] += py
            match["episode_count"] += 1
            match["trials"].add(episode["trial"])
            match["stall_steps"] += int(episode["stalled_steps"])
            match["episodes"].append(episode)

    candidates = []
    for group in groups:
        trial_labels = sorted(group["trials"])
        if len(trial_labels) < min_trial_support:
            continue
        count = int(group["episode_count"])
        candidates.append(
            {
                "direction": group["direction"],
                "position": [
                    round(group["position_sum"][0] / count, 6),
                    round(group["position_sum"][1] / count, 6),
                ],
                "trial_support": len(trial_labels),
                "supporting_trials": trial_labels,
                "episode_count": count,
                "stalled_steps": group["stall_steps"],
                "evidence_count": len(group["episodes"]),
                "evidence_truncated": (
                    len(group["episodes"]) > max_evidence_per_candidate
                ),
                "evidence": group["episodes"][:max_evidence_per_candidate],
            }
        )
    candidates.sort(
        key=lambda item: (
            -int(item["trial_support"]),
            -int(item["stalled_steps"]),
            item["direction"],
            item["position"],
        )
    )
    return {
        "candidate_count": len(candidates),
        "movement_constraints": candidates,
        "stall_episode_count": len(episodes),
        "stall_episodes_truncated": len(episodes) > max_reported_episodes,
        "all_stall_episodes": episodes[:max_reported_episodes],
        "settings": {
            "max_projected_progress": max_projected_progress,
            "min_stalled_steps": min_stalled_steps,
            "position_tolerance": position_tolerance,
            "min_trial_support": min_trial_support,
        },
        "epistemic_status": (
            "repeated requested-direction stalls in supplied coordinates; candidates "
            "are not identified as walls, collisions, or implementation semantics"
        ),
    }


__all__ = [
    "Mosaic",
    "TileStructure",
    "WorldError",
    "build_mosaic",
    "infer_movement_constraints",
    "infer_tile_structure",
]
