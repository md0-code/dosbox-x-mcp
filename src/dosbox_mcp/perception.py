"""Pure visual and memory evidence algorithms for DOS experiments.

Nothing in this module knows about games, DOSBox-X, or MCP.  It compares
indexed frames, associates changed regions over time, classifies changes in
memory snapshots, and measures simple numeric correlations.  The results use
neutral terms deliberately: a changed region or correlated offset is evidence,
not a semantic claim about a player, enemy, score, or wall.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence


class PerceptionError(ValueError):
    """Inputs cannot be compared without inventing or discarding evidence."""


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def centroid(self) -> tuple[float, float]:
        return (self.x + (self.width - 1) / 2, self.y + (self.height - 1) / 2)

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.width, self.height]


@dataclass(frozen=True)
class ChangedRegion:
    bbox: BoundingBox
    changed_pixels: int

    @property
    def density(self) -> float:
        return self.changed_pixels / self.bbox.area

    @property
    def centroid(self) -> tuple[float, float]:
        return self.bbox.centroid

    def as_dict(self) -> dict[str, object]:
        return {
            "bbox": self.bbox.as_list(),
            "x": self.bbox.x,
            "y": self.bbox.y,
            "width": self.bbox.width,
            "height": self.bbox.height,
            "changed_pixels": self.changed_pixels,
            "density": round(self.density, 6),
            "centroid": [round(value, 3) for value in self.centroid],
        }


@dataclass(frozen=True)
class FrameComparison:
    width: int
    height: int
    changed_pixels: int
    overall_bbox: BoundingBox | None
    regions: tuple[ChangedRegion, ...]
    bridge_radius: int = 0
    min_changed_pixels: int = 1

    @property
    def changed_fraction(self) -> float:
        return self.changed_pixels / (self.width * self.height)

    def as_dict(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "changed_pixels": self.changed_pixels,
            "changed_fraction": round(self.changed_fraction, 8),
            "overall_bbox": self.overall_bbox.as_list() if self.overall_bbox else None,
            "region_count": len(self.regions),
            "regions": [region.as_dict() for region in self.regions],
            "settings": {
                "bridge_radius": self.bridge_radius,
                "min_changed_pixels": self.min_changed_pixels,
            },
        }


def _bbox(indices: Iterable[int], width: int) -> BoundingBox | None:
    points = list(indices)
    if not points:
        return None
    xs = [position % width for position in points]
    ys = [position // width for position in points]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    return BoundingBox(left, top, right - left + 1, bottom - top + 1)


def compare_frames(
    before: bytes,
    after: bytes,
    width: int,
    height: int,
    *,
    bridge_radius: int = 0,
    min_changed_pixels: int = 1,
    connectivity: Literal[4, 8] = 8,
) -> FrameComparison:
    """Compare two equal-size indexed frames and extract changed regions.

    ``bridge_radius`` dilates the binary difference mask before component
    labelling.  This joins nearby old/new footprints while reported boxes and
    counts remain based only on pixels that actually changed.
    """
    if width <= 0 or height <= 0:
        raise PerceptionError("frame width and height must both be positive")
    expected = width * height
    if len(before) != expected or len(after) != expected:
        raise PerceptionError(
            f"both indexed frames must be exactly {expected} bytes for "
            f"{width}x{height}; got {len(before)} and {len(after)}"
        )
    if bridge_radius < 0 or bridge_radius > 32:
        raise PerceptionError("bridge_radius must be between 0 and 32 pixels")
    if min_changed_pixels < 1:
        raise PerceptionError("min_changed_pixels must be at least 1")
    if connectivity not in (4, 8):
        raise PerceptionError("connectivity must be 4 or 8")

    changed = bytearray(expected)
    changed_indices: list[int] = []
    for index, (old, new) in enumerate(zip(before, after)):
        if old != new:
            changed[index] = 1
            changed_indices.append(index)

    if not changed_indices:
        return FrameComparison(width, height, 0, None, (), bridge_radius, min_changed_pixels)

    analysis = changed if bridge_radius == 0 else bytearray(expected)
    if bridge_radius:
        radius = bridge_radius
        for position in changed_indices:
            x, y = position % width, position // width
            for near_y in range(max(0, y - radius), min(height, y + radius + 1)):
                row = near_y * width
                left, right = max(0, x - radius), min(width, x + radius + 1)
                analysis[row + left : row + right] = b"\x01" * (right - left)

    neighbours = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if connectivity == 8:
        neighbours += ((-1, -1), (-1, 1), (1, -1), (1, 1))

    visited = bytearray(expected)
    regions: list[ChangedRegion] = []
    for seed in range(expected):
        if not analysis[seed] or visited[seed]:
            continue
        visited[seed] = 1
        stack = [seed]
        actual: list[int] = []
        while stack:
            position = stack.pop()
            if changed[position]:
                actual.append(position)
            x, y = position % width, position // width
            for dx, dy in neighbours:
                near_x, near_y = x + dx, y + dy
                if not (0 <= near_x < width and 0 <= near_y < height):
                    continue
                near = near_y * width + near_x
                if analysis[near] and not visited[near]:
                    visited[near] = 1
                    stack.append(near)
        if len(actual) < min_changed_pixels:
            continue
        box = _bbox(actual, width)
        if box is not None:
            regions.append(ChangedRegion(box, len(actual)))

    regions.sort(key=lambda region: (region.bbox.y, region.bbox.x, -region.changed_pixels))
    return FrameComparison(
        width,
        height,
        len(changed_indices),
        _bbox(changed_indices, width),
        tuple(regions),
        bridge_radius,
        min_changed_pixels,
    )


def _translation_bounds(
    width: int, height: int, dx: int, dy: int
) -> tuple[int, int, int, int]:
    """After-frame bounds whose source coordinate exists in the before frame."""
    return max(0, dx), max(0, dy), min(width, width + dx), min(height, height + dy)


def _translation_score(
    before: bytes,
    after: bytes,
    width: int,
    height: int,
    dx: int,
    dy: int,
    stride: int,
    ignore_mask: bytes | None = None,
) -> tuple[float, int, int]:
    x0, y0, x1, y1 = _translation_bounds(width, height, dx, dy)
    matches = samples = 0
    for y in range(y0, y1, stride):
        before_row = (y - dy) * width
        after_row = y * width
        for x in range(x0, x1, stride):
            source = before_row + x - dx
            target = after_row + x
            if ignore_mask is not None and (ignore_mask[source] or ignore_mask[target]):
                continue
            matches += before[source] == after[target]
            samples += 1
    return (matches / samples if samples else 0.0), matches, samples


def detect_translation(
    before: bytes,
    after: bytes,
    width: int,
    height: int,
    *,
    max_dx: int = 16,
    max_dy: int = 16,
    sample_stride: int = 2,
    bridge_radius: int = 0,
    min_changed_pixels: int = 1,
    ignore_mask: bytes | None = None,
) -> dict[str, object]:
    """Find global frame translation and residual changes without optical flow.

    ``(dx, dy)`` means a pixel in ``before`` appears at that displacement in
    ``after``. All bounded candidates are sampled; the strongest candidates
    and their immediate neighbours are then scored at every overlap pixel.
    """
    expected = width * height
    if width <= 0 or height <= 0 or len(before) != expected or len(after) != expected:
        raise PerceptionError("translation inputs must be equal width*height indexed frames")
    if not 0 <= max_dx < width or not 0 <= max_dy < height:
        raise PerceptionError("translation bounds must be non-negative and smaller than the frame")
    if not 1 <= sample_stride <= 16:
        raise PerceptionError("sample_stride must be between 1 and 16")
    if ignore_mask is not None and len(ignore_mask) != expected:
        raise PerceptionError("translation ignore_mask must contain width*height bytes")
    if ignore_mask is not None and all(ignore_mask):
        raise PerceptionError("translation ignore_mask cannot exclude the entire frame")

    sampled: list[tuple[float, int, int]] = []
    for dy in range(-max_dy, max_dy + 1):
        for dx in range(-max_dx, max_dx + 1):
            ratio, _, _ = _translation_score(
                before, after, width, height, dx, dy, sample_stride, ignore_mask
            )
            sampled.append((ratio, dx, dy))
    sampled.sort(key=lambda item: (-item[0], abs(item[1]) + abs(item[2]), item[2], item[1]))

    refine: set[tuple[int, int]] = {(0, 0)}
    for _, dx, dy in sampled[:32]:
        for near_y in range(max(-max_dy, dy - 1), min(max_dy, dy + 1) + 1):
            for near_x in range(max(-max_dx, dx - 1), min(max_dx, dx + 1) + 1):
                refine.add((near_x, near_y))
    exact: list[dict[str, object]] = []
    for dx, dy in refine:
        ratio, matches, overlap = _translation_score(
            before, after, width, height, dx, dy, 1, ignore_mask
        )
        exact.append(
            {
                "dx": dx,
                "dy": dy,
                "match_ratio": ratio,
                "matching_pixels": matches,
                "overlap_pixels": overlap,
                "overlap_fraction": overlap / expected,
            }
        )
    exact.sort(
        key=lambda item: (
            -float(item["match_ratio"]),
            -float(item["overlap_fraction"]),
            abs(int(item["dx"])) + abs(int(item["dy"])),
            int(item["dy"]),
            int(item["dx"]),
        )
    )
    best = exact[0]
    second = exact[1] if len(exact) > 1 else exact[0]
    margin = float(best["match_ratio"]) - float(second["match_ratio"])
    confidence = float(best["match_ratio"]) * min(1.0, max(0.0, margin) / 0.05)
    dx, dy = int(best["dx"]), int(best["dy"])
    x0, y0, x1, y1 = _translation_bounds(width, height, dx, dy)

    # Outside the overlap there is no source evidence. Copy after there so the
    # residual comparison reports only independently changed overlap pixels.
    compensated = bytearray(after)
    for y in range(y0, y1):
        source = (y - dy) * width + x0 - dx
        target = y * width + x0
        compensated[target : target + x1 - x0] = before[source : source + x1 - x0]
    residual = compare_frames(
        bytes(compensated),
        after,
        width,
        height,
        bridge_radius=bridge_radius,
        min_changed_pixels=min_changed_pixels,
    )
    return {
        "best_translation": [dx, dy],
        "match_ratio": round(float(best["match_ratio"]), 8),
        "matching_pixels": best["matching_pixels"],
        "overlap_pixels": best["overlap_pixels"],
        "overlap_fraction": round(float(best["overlap_fraction"]), 8),
        "overlap_bbox": [x0, y0, x1 - x0, y1 - y0],
        "second_best": {
            "translation": [second["dx"], second["dy"]],
            "match_ratio": round(float(second["match_ratio"]), 8),
        },
        "top_candidates": [
            {
                "translation": [item["dx"], item["dy"]],
                "match_ratio": round(float(item["match_ratio"]), 8),
                "overlap_fraction": round(float(item["overlap_fraction"]), 8),
            }
            for item in exact[:16]
        ],
        "ambiguity_margin": round(margin, 8),
        "confidence": round(confidence, 8),
        "residual": residual.as_dict(),
        "settings": {
            "max_dx": max_dx,
            "max_dy": max_dy,
            "sample_stride": sample_stride,
            "refined_candidates": len(exact),
            "masked_pixels": sum(bool(value) for value in ignore_mask) if ignore_mask is not None else 0,
        },
        "epistemic_status": (
            "global indexed-pixel translation evidence; residual regions are "
            "changes after compensation, not semantic objects"
        ),
    }


def estimate_camera_path(
    frames: Sequence[bytes],
    width: int,
    height: int,
    *,
    max_dx: int = 16,
    max_dy: int = 16,
    sample_stride: int = 2,
    consensus_window: int = 3,
    minimum_confidence: float = 0.25,
    screen_fixed_mask: bytes | None = None,
) -> dict[str, object]:
    """Build a confidence-bearing camera trajectory from consecutive frames.

    Pairwise translation stays available as the low-level primitive.  This
    layer rejects isolated low-confidence jumps, applies a bounded temporal
    median to plausible steps, accumulates coordinates, and reports exact
    frame loop closures plus the resulting drift.
    """
    if len(frames) < 2:
        raise PerceptionError("camera-path estimation requires at least two frames")
    if not 1 <= consensus_window <= 15 or consensus_window % 2 == 0:
        raise PerceptionError("consensus_window must be an odd number from 1 to 15")
    expected = width * height
    if any(len(frame) != expected for frame in frames):
        raise PerceptionError("every camera-path frame must contain width*height bytes")
    if screen_fixed_mask is not None and len(screen_fixed_mask) != expected:
        raise PerceptionError("screen_fixed_mask must contain width*height bytes")

    pairwise = [
        detect_translation(
            before,
            after,
            width,
            height,
            max_dx=max_dx,
            max_dy=max_dy,
            sample_stride=sample_stride,
            ignore_mask=screen_fixed_mask,
        )
        for before, after in zip(frames, frames[1:])
    ]
    raw = [tuple(int(value) for value in result["best_translation"]) for result in pairwise]
    confidences = [float(result["confidence"]) for result in pairwise]
    radius = consensus_window // 2
    accepted: list[tuple[int, int]] = []
    outliers: list[dict[str, object]] = []
    for index, ((dx, dy), confidence) in enumerate(zip(raw, confidences)):
        start, stop = max(0, index - radius), min(len(raw), index + radius + 1)
        plausible = [raw[item] for item in range(start, stop) if confidences[item] >= minimum_confidence]
        if confidence < minimum_confidence or not plausible:
            accepted.append((0, 0))
            outliers.append({"transition": index + 1, "translation": [dx, dy], "confidence": confidence, "reason": "below confidence threshold"})
            continue
        xs = sorted(item[0] for item in plausible)
        ys = sorted(item[1] for item in plausible)
        consensus = (xs[len(xs) // 2], ys[len(ys) // 2])
        if abs(dx - consensus[0]) > 1 or abs(dy - consensus[1]) > 1:
            accepted.append(consensus)
            outliers.append({"transition": index + 1, "translation": [dx, dy], "consensus": list(consensus), "confidence": confidence, "reason": "temporal-consensus outlier"})
        else:
            accepted.append((dx, dy))

    gap_recoveries: list[dict[str, object]] = []
    for index in range(len(accepted) - 1):
        if confidences[index] >= minimum_confidence or confidences[index + 1] >= minimum_confidence:
            continue
        skipped = detect_translation(
            frames[index],
            frames[index + 2],
            width,
            height,
            max_dx=min(width - 1, max_dx * 2),
            max_dy=min(height - 1, max_dy * 2),
            sample_stride=sample_stride,
            ignore_mask=screen_fixed_mask,
        )
        if float(skipped["confidence"]) < minimum_confidence:
            continue
        recovered = tuple(int(value) for value in skipped["best_translation"])
        accepted[index] = (0, 0)
        accepted[index + 1] = recovered
        gap_recoveries.append(
            {
                "unresolved_frame": index + 1,
                "anchor_frames": [index, index + 2],
                "aggregate_translation": list(recovered),
                "confidence": skipped["confidence"],
                "epistemic_status": "aggregate displacement across one unresolved frame; no position is assigned inside the gap",
            }
        )

    path = [[0, 0]]
    uncertainty = [[0.0, 0.0]]
    for (dx, dy), confidence in zip(accepted, confidences):
        path.append([path[-1][0] + dx, path[-1][1] + dy])
        spread = max(0.0, 1.0 - confidence)
        uncertainty.append([round(uncertainty[-1][0] + spread, 6), round(uncertainty[-1][1] + spread, 6)])

    pre_correction_path = [point[:] for point in path]
    first_seen: dict[bytes, int] = {}
    closures: list[dict[str, object]] = []
    for index, frame in enumerate(frames):
        if frame in first_seen:
            origin = first_seen[frame]
            drift = [path[index][0] - path[origin][0], path[index][1] - path[origin][1]]
            closures.append({"from_frame": origin, "to_frame": index, "exact_content": True, "drift": drift, "within_one_pixel": abs(drift[0]) <= 1 and abs(drift[1]) <= 1, "correction_applied": drift != [0, 0]})
        else:
            first_seen[frame] = index

    # Exact byte-identical revisits are the only evidence allowed to close a
    # loop.  Distribute the measured drift over the closed interval, then keep
    # later coordinates in the corrected reference frame.  Raw coordinates
    # remain available verbatim in pre_correction_path.
    for closure in closures:
        drift_x, drift_y = closure["drift"]
        if drift_x == 0 and drift_y == 0:
            continue
        origin, end = int(closure["from_frame"]), int(closure["to_frame"])
        span = end - origin
        for position in range(origin + 1, end + 1):
            fraction = (position - origin) / span
            path[position][0] -= round(drift_x * fraction)
            path[position][1] -= round(drift_y * fraction)
        for position in range(end + 1, len(path)):
            path[position][0] -= drift_x
            path[position][1] -= drift_y

    corrected_translations = [
        [path[index + 1][0] - path[index][0], path[index + 1][1] - path[index][1]]
        for index in range(len(path) - 1)
    ]
    outlier_transitions = {int(item["transition"]): item for item in outliers}
    recovery_targets = {int(item["anchor_frames"][1]) for item in gap_recoveries}
    placement_methods = ["direct"]
    for index, (before_step, corrected_step) in enumerate(zip(raw, corrected_translations), start=1):
        outlier = outlier_transitions.get(index)
        if index in recovery_targets:
            placement_methods.append("consensus_corrected")
        elif outlier and outlier["reason"] == "below confidence threshold":
            placement_methods.append("unresolved")
        elif list(before_step) != corrected_step or outlier is not None:
            placement_methods.append("consensus_corrected")
        else:
            placement_methods.append("direct")
    return {
        "frame_count": len(frames),
        "pairwise": pairwise,
        "accepted_translations": corrected_translations,
        "pre_correction_translations": [list(item) for item in accepted],
        "pre_correction_path": pre_correction_path,
        "cumulative_path": path,
        "placement_methods": placement_methods,
        "uncertainty_pixels": uncertainty,
        "outliers": outliers,
        "gap_recoveries": gap_recoveries,
        "loop_closures": closures,
        "settings": {"consensus_window": consensus_window, "minimum_confidence": minimum_confidence, "screen_fixed_mask_pixels": sum(bool(value) for value in screen_fixed_mask) if screen_fixed_mask is not None else 0},
        "epistemic_status": "multi-frame indexed-pixel camera evidence; low-confidence steps are rejected, not guessed",
    }


def infer_screen_regions(
    frames: Sequence[bytes],
    translations: Sequence[tuple[int, int]],
    width: int,
    height: int,
    *,
    min_episodes: int = 2,
    support_ratio: float = 0.8,
    ambiguity_margin: float = 0.15,
    min_region_pixels: int = 4,
    max_regions_per_category: int = 64,
) -> dict[str, object]:
    """Accumulate screen-fixed/global-motion evidence across scroll episodes."""
    expected = width * height
    if len(frames) < 3 or len(translations) != len(frames) - 1:
        raise PerceptionError("screen-region inference needs at least 3 frames and one translation per transition")
    if any(len(frame) != expected for frame in frames):
        raise PerceptionError("every screen-region frame must be width*height bytes")
    if min_episodes < 1:
        raise PerceptionError("min_episodes must be positive")
    if max_regions_per_category < 1:
        raise PerceptionError("max_regions_per_category must be positive")
    if not 0 < support_ratio <= 1 or not 0 <= ambiguity_margin <= 1:
        raise PerceptionError("support ratios must be between 0 and 1")

    observed = [0] * expected
    moved = [0] * expected
    fixed = [0] * expected
    changed = [0] * expected
    used_episodes = 0
    for before, after, (dx, dy) in zip(frames, frames[1:], translations):
        if dx == 0 and dy == 0:
            continue
        used_episodes += 1
        x0, y0, x1, y1 = _translation_bounds(width, height, dx, dy)
        for y in range(y0, y1):
            source_row = (y - dy) * width
            row = y * width
            for x in range(x0, x1):
                at = row + x
                observed[at] += 1
                moved[at] += before[source_row + x - dx] == after[at]
                fixed[at] += before[at] == after[at]
                changed[at] += before[at] != after[at]

    masks = {
        "globally_translating": bytearray(expected),
        "screen_fixed_candidate": bytearray(expected),
        "frequently_changing_fixed_position_candidate": bytearray(expected),
        "uncertain": bytearray(expected),
    }
    for at, count in enumerate(observed):
        if count < min_episodes:
            continue
        moved_ratio = moved[at] / count
        fixed_ratio = fixed[at] / count
        changed_ratio = changed[at] / count
        if moved_ratio >= support_ratio and moved_ratio - fixed_ratio >= ambiguity_margin:
            category = "globally_translating"
        elif fixed_ratio >= support_ratio and fixed_ratio - moved_ratio >= ambiguity_margin:
            category = "screen_fixed_candidate"
        elif changed_ratio >= support_ratio and moved_ratio < support_ratio:
            category = "frequently_changing_fixed_position_candidate"
        else:
            category = "uncertain"
        masks[category][at] = 1

    categories: dict[str, object] = {}
    blank = bytes(expected)
    for name, mask in masks.items():
        comparison = compare_frames(
            blank,
            bytes(mask),
            width,
            height,
            min_changed_pixels=min_region_pixels,
        )
        regions = sorted(
            comparison.regions,
            key=lambda region: (-region.changed_pixels, region.bbox.y, region.bbox.x),
        )
        categories[name] = {
            "pixels": sum(mask),
            "region_count": len(regions),
            "regions_truncated": len(regions) > max_regions_per_category,
            "regions": [
                region.as_dict() for region in regions[:max_regions_per_category]
            ],
        }
    return {
        "scroll_episodes": used_episodes,
        "required_episode_support": min_episodes,
        "categories": categories,
        "settings": {
            "support_ratio": support_ratio,
            "ambiguity_margin": ambiguity_margin,
            "min_region_pixels": min_region_pixels,
            "max_regions_per_category": max_regions_per_category,
        },
        "epistemic_status": (
            "screen-position evidence accumulated across measured translations; "
            "fixed candidates are not semantically identified as HUD"
        ),
    }


def rank_control_responses(trials: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Rank repeatable visual-track displacement above idle-control motion."""
    grouped: dict[str, list[dict[str, object] | None]] = {}
    controls: list[dict[str, object] | None] = []
    for trial in trials:
        action = str(trial.get("action", ""))
        tracks = trial.get("tracks", [])
        if not isinstance(tracks, list):
            raise PerceptionError("control trial tracks must be a list")
        candidates = []
        motion = trial.get("global_motion")
        if isinstance(motion, dict):
            translation = motion.get("best_translation", [0, 0])
            if isinstance(translation, list) and len(translation) == 2:
                dx, dy = float(translation[0]), float(translation[1])
                confidence = float(motion.get("confidence", 0))
                candidates.append(
                    {
                        "dx": dx,
                        "dy": dy,
                        "magnitude": math.hypot(dx, dy),
                        "confidence": confidence,
                        "observations": 1.0,
                        "track_id": 0.0,
                        "response_kind": "global_translation",
                        "selection_score": math.hypot(dx, dy) * confidence,
                    }
                )
        else:
            for track in tracks:
                if not isinstance(track, dict):
                    continue
                dx = float(track.get("dx", 0))
                dy = float(track.get("dy", 0))
                confidence = float(track.get("confidence", 0))
                observations = int(track.get("observations", 0))
                magnitude = math.hypot(dx, dy)
                candidates.append(
                    {
                        "dx": dx,
                        "dy": dy,
                        "magnitude": magnitude,
                        "confidence": confidence,
                        "observations": float(observations),
                        "track_id": float(int(track.get("track_id", 0))),
                        "response_kind": "changed_region_track",
                        "selection_score": magnitude * confidence * max(1, observations),
                    }
                )
        selected = max(candidates, key=lambda item: item["selection_score"], default=None)
        if bool(trial.get("control", False)):
            controls.append(selected)
        else:
            grouped.setdefault(action, []).append(selected)

    idle_magnitudes = [item["magnitude"] for item in controls if item is not None]
    idle_mean = sum(idle_magnitudes) / len(idle_magnitudes) if idle_magnitudes else 0.0
    ranked: list[dict[str, object]] = []
    for action, repeats in grouped.items():
        present = [item for item in repeats if item is not None]
        if present:
            mean_dx = sum(item["dx"] for item in present) / len(present)
            mean_dy = sum(item["dy"] for item in present) / len(present)
            mean_magnitude = sum(item["magnitude"] for item in present) / len(present)
            mean_confidence = sum(item["confidence"] for item in present) / len(present)
            direction_consistency = (
                math.hypot(mean_dx, mean_dy) / mean_magnitude if mean_magnitude else 0.0
            )
        else:
            mean_dx = mean_dy = mean_magnitude = mean_confidence = direction_consistency = 0.0
        support = len(present) / len(repeats) if repeats else 0.0
        excess = max(0.0, mean_magnitude - idle_mean)
        score = excess * direction_consistency * mean_confidence * support
        ranked.append(
            {
                "action": action,
                "repeat_count": len(repeats),
                "responding_repeats": len(present),
                "mean_dx": round(mean_dx, 4),
                "mean_dy": round(mean_dy, 4),
                "mean_magnitude": round(mean_magnitude, 4),
                "idle_mean_magnitude": round(idle_mean, 4),
                "response_excess": round(excess, 4),
                "direction_consistency": round(direction_consistency, 6),
                "mean_track_confidence": round(mean_confidence, 6),
                "score": round(score, 6),
                "selected_tracks": [
                    (
                        {
                            "track_id": int(item["track_id"]),
                            "response_kind": item["response_kind"],
                            "dx": round(item["dx"], 4),
                            "dy": round(item["dy"], 4),
                            "confidence": round(item["confidence"], 6),
                        }
                        if item is not None
                        else None
                    )
                    for item in repeats
                ],
            }
        )
    ranked.sort(key=lambda item: (-float(item["score"]), str(item["action"])))
    return ranked


def rank_joint_control_responses(
    trials: Sequence[dict[str, object]], *, time_domain: str = "wall_clock_ms"
) -> dict[str, object]:
    """Rank camera, local-visual, memory and transition evidence separately.

    Trial dictionaries are intentionally neutral observations.  An ``idle``
    action supplies the per-channel noise floor; repeats establish support.
    A weak camera channel is capped before combination so it cannot overrule
    stronger memory or local-track evidence.
    """
    if not trials:
        return {"channels": {}, "combined": [], "time_domain": time_domain}

    def channel_values(trial: dict[str, object]) -> dict[str, float]:
        motion = trial.get("global_motion") or {}
        translation = motion.get("best_translation", [0, 0]) if isinstance(motion, dict) else [0, 0]
        camera_confidence = float(motion.get("confidence", 0)) if isinstance(motion, dict) else 0.0
        camera = (abs(float(translation[0])) + abs(float(translation[1]))) * camera_confidence
        tracks = trial.get("tracks", [])
        local = max((abs(float(track.get("dx", 0))) + abs(float(track.get("dy", 0)))) * float(track.get("confidence", 0)) for track in tracks) if tracks else 0.0
        memory_deltas = trial.get("memory_deltas", {})
        memory = max((abs(float(value)) for value in memory_deltas.values()), default=0.0) if isinstance(memory_deltas, dict) else 0.0
        transition = float(trial.get("transition_fraction", 0.0))
        return {"camera": camera, "local_visual": local, "memory": memory, "transition": transition}

    controls = [channel_values(trial) for trial in trials if bool(trial.get("control"))]
    idle = {name: (sum(item[name] for item in controls) / len(controls) if controls else 0.0) for name in ("camera", "local_visual", "memory", "transition")}
    grouped: dict[str, list[tuple[dict[str, object], dict[str, float]]]] = {}
    for trial in trials:
        if bool(trial.get("control")):
            continue
        grouped.setdefault(str(trial.get("action", "")), []).append((trial, channel_values(trial)))

    channels: dict[str, list[dict[str, object]]] = {name: [] for name in idle}
    combined: list[dict[str, object]] = []
    opposite_names = {"left": "right", "right": "left", "up": "down", "down": "up"}
    for action, repeats in grouped.items():
        per_channel: dict[str, dict[str, object]] = {}
        for name in idle:
            values = [item[1][name] for item in repeats]
            mean = sum(values) / len(values)
            excess = max(0.0, mean - idle[name])
            repeat_support = sum(value > idle[name] for value in values) / len(values)
            confidence = repeat_support * (excess / (excess + idle[name] + 1.0))
            if name == "camera":
                raw_confidence = [float((item[0].get("global_motion") or {}).get("confidence", 0)) for item in repeats]
                confidence *= sum(raw_confidence) / len(raw_confidence)
            entry = {"action": action, "mean_response": round(mean, 8), "idle_mean": round(idle[name], 8), "excess": round(excess, 8), "repeat_support": round(repeat_support, 8), "confidence": round(confidence, 8), "repeats": len(values)}
            channels[name].append(entry)
            per_channel[name] = entry

        onset_values: list[float] = []
        release_values: list[float] = []
        for trial, _ in repeats:
            samples = trial.get("memory_samples", [])
            if not isinstance(samples, list) or len(samples) < 2:
                continue
            numeric = [key for key in samples[0] if key not in {"t_ms", "counter"}]
            baseline = samples[0]
            changed = [sample for sample in samples[1:] if any(sample.get(key) != baseline.get(key) for key in numeric)]
            if changed:
                onset_values.append(float(changed[0].get("counter", changed[0].get("t_ms", 0))))
                returned = [sample for sample in changed[1:] if all(sample.get(key) == baseline.get(key) for key in numeric)]
                if returned:
                    release_values.append(float(returned[0].get("counter", returned[0].get("t_ms", 0))))
        non_transition = [per_channel[name]["confidence"] for name in ("camera", "local_visual", "memory")]
        strongest = max(float(value) for value in non_transition)
        score = sum(float(value) for value in non_transition) / 3.0
        combined.append({"action": action, "score": round(score, 8), "confidence": round(strongest, 8), "channels": per_channel, "transition_ranked_separately": True, "onset_latency": min(onset_values) if onset_values else None, "release_latency": min(release_values) if release_values else None, "time_domain": time_domain, "opposite_action": opposite_names.get(action), "opposite_available": opposite_names.get(action) in grouped})

    for entries in channels.values():
        entries.sort(key=lambda item: (-float(item["confidence"]), -float(item["excess"]), str(item["action"])))
    combined.sort(key=lambda item: (-float(item["score"]), -float(item["confidence"]), str(item["action"])))
    return {"idle_baseline": idle, "channels": channels, "combined": combined, "time_domain": time_domain, "epistemic_status": "channel-specific response evidence above repeated idle; transitions remain separate and no control semantics are asserted"}


def _iou(left: BoundingBox, right: BoundingBox) -> float:
    x0, y0 = max(left.x, right.x), max(left.y, right.y)
    x1 = min(left.x + left.width, right.x + right.width)
    y1 = min(left.y + left.height, right.y + right.height)
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    union = left.area + right.area - intersection
    return intersection / union if union else 0.0


@dataclass
class _Track:
    track_id: int
    bbox: BoundingBox
    start_centroid: tuple[float, float]
    last_centroid: tuple[float, float]
    age: int = 1
    missed: int = 0
    confidences: list[float] = field(default_factory=lambda: [0.5])
    history: list[dict[str, object]] = field(default_factory=list)

    def report(self) -> dict[str, object]:
        dx = self.last_centroid[0] - self.start_centroid[0]
        dy = self.last_centroid[1] - self.start_centroid[1]
        return {
            "track_id": self.track_id,
            "bbox": self.bbox.as_list(),
            "centroid": [round(value, 3) for value in self.last_centroid],
            "dx": round(dx, 3),
            "dy": round(dy, 3),
            "age": self.age,
            "observations": len(self.history),
            "missed": self.missed,
            "confidence": round(sum(self.confidences) / len(self.confidences), 4),
            "history": self.history,
        }


def track_regions(
    comparisons: Sequence[FrameComparison],
    *,
    max_distance: float = 48.0,
    max_missed: int = 1,
    global_change_fraction: float = 0.35,
) -> dict[str, object]:
    """Associate changed regions between transitions using deterministic scores."""
    if max_distance <= 0:
        raise PerceptionError("max_distance must be positive")
    if max_missed < 0:
        raise PerceptionError("max_missed cannot be negative")
    if not 0 < global_change_fraction <= 1:
        raise PerceptionError("global_change_fraction must be in (0, 1]")

    active: dict[int, _Track] = {}
    all_tracks: dict[int, _Track] = {}
    next_id = 1
    global_transitions: list[int] = []

    for transition, comparison in enumerate(comparisons, start=1):
        if comparison.changed_fraction >= global_change_fraction:
            global_transitions.append(transition)
            active.clear()
            continue

        scores: list[tuple[float, int, int]] = []
        by_region: dict[int, list[float]] = {}
        for track_id, track in active.items():
            for region_index, region in enumerate(comparison.regions):
                overlap = _iou(track.bbox, region.bbox)
                cx, cy = region.centroid
                distance = math.hypot(cx - track.last_centroid[0], cy - track.last_centroid[1])
                if overlap == 0 and distance > max_distance:
                    continue
                distance_score = max(0.0, 1.0 - distance / max_distance)
                size_score = min(track.bbox.area, region.bbox.area) / max(
                    track.bbox.area, region.bbox.area
                )
                score = 0.45 * overlap + 0.35 * distance_score + 0.20 * size_score
                scores.append((score, track_id, region_index))
                by_region.setdefault(region_index, []).append(score)

        matched_tracks: set[int] = set()
        matched_regions: set[int] = set()
        for score, track_id, region_index in sorted(
            scores, key=lambda item: (-item[0], item[1], item[2])
        ):
            if track_id in matched_tracks or region_index in matched_regions:
                continue
            region = comparison.regions[region_index]
            track = active[track_id]
            alternatives = sorted(by_region.get(region_index, ()), reverse=True)
            second = alternatives[1] if len(alternatives) > 1 else 0.0
            ambiguity_margin = max(0.0, score - second)
            confidence = score * (0.5 + 0.5 * min(1.0, ambiguity_margin / 0.25))
            previous = track.last_centroid
            track.bbox = region.bbox
            track.last_centroid = region.centroid
            track.age += 1
            track.missed = 0
            track.confidences.append(confidence)
            track.history.append(
                {
                    "transition": transition,
                    "bbox": region.bbox.as_list(),
                    "centroid": [round(value, 3) for value in region.centroid],
                    "step_dx": round(region.centroid[0] - previous[0], 3),
                    "step_dy": round(region.centroid[1] - previous[1], 3),
                    "confidence": round(confidence, 4),
                }
            )
            matched_tracks.add(track_id)
            matched_regions.add(region_index)

        for track_id in list(active):
            if track_id in matched_tracks:
                continue
            track = active[track_id]
            track.age += 1
            track.missed += 1
            track.confidences.append(0.0)
            if track.missed > max_missed:
                del active[track_id]

        for region_index, region in enumerate(comparison.regions):
            if region_index in matched_regions:
                continue
            track = _Track(next_id, region.bbox, region.centroid, region.centroid)
            track.history.append(
                {
                    "transition": transition,
                    "bbox": region.bbox.as_list(),
                    "centroid": [round(value, 3) for value in region.centroid],
                    "step_dx": 0.0,
                    "step_dy": 0.0,
                    "confidence": 0.5,
                }
            )
            active[next_id] = track
            all_tracks[next_id] = track
            next_id += 1

    reports = [all_tracks[track_id].report() for track_id in sorted(all_tracks)]
    return {
        "tracks": reports,
        "track_count": len(reports),
        "global_transitions": global_transitions,
        "limitations": (
            "Track IDs associate visual-change regions only; crossings and global "
            "screen changes are explicitly ambiguous and carry reduced confidence."
        ),
    }


MemoryRelation = Literal["changed", "unchanged", "increased", "decreased", "delta"]


def scan_memory_changes(
    baseline: bytes,
    experiment: bytes,
    *,
    value_size: Literal[1, 2, 4] = 1,
    relation: MemoryRelation = "changed",
    expected_delta: int | None = None,
    control: bytes | None = None,
    opposite: bytes | None = None,
    require_control_stable: bool = True,
    require_opposite_direction: bool = True,
    aligned: bool = True,
    offset_base: int = 0,
) -> list[dict[str, object]]:
    """Classify numeric fields in equal-size memory snapshots.

    One width is considered per call.  Aligned scanning is the default so a
    16- or 32-bit field does not flood the result with overlapping byte-shifted
    interpretations.
    """
    if value_size not in (1, 2, 4):
        raise PerceptionError("value_size must be 1, 2, or 4")
    snapshots = [("experiment", experiment), ("control", control), ("opposite", opposite)]
    for name, snapshot in snapshots:
        if snapshot is not None and len(snapshot) != len(baseline):
            raise PerceptionError(
                f"{name} snapshot has {len(snapshot)} bytes; baseline has {len(baseline)}"
            )
    if len(baseline) < value_size:
        raise PerceptionError("snapshot is shorter than the requested value size")
    if relation == "delta" and expected_delta is None:
        raise PerceptionError("relation='delta' requires expected_delta")

    stride = value_size if aligned else 1
    candidates: list[dict[str, object]] = []
    for offset in range(0, len(baseline) - value_size + 1, stride):
        before = int.from_bytes(baseline[offset : offset + value_size], "little")
        after = int.from_bytes(experiment[offset : offset + value_size], "little")
        delta = after - before
        matches = {
            "changed": delta != 0,
            "unchanged": delta == 0,
            "increased": delta > 0,
            "decreased": delta < 0,
            "delta": delta == expected_delta,
        }[relation]
        if not matches:
            continue

        evidence = [f"experiment {relation}: {before} -> {after} (delta {delta:+d})"]
        score = 1.0
        control_value = None
        if control is not None:
            control_value = int.from_bytes(control[offset : offset + value_size], "little")
            stable = control_value == before
            if require_control_stable and not stable:
                continue
            evidence.append("control stable" if stable else f"control changed to {control_value}")
            score += 1.0 if stable else 0.0

        opposite_value = None
        opposite_delta = None
        if opposite is not None:
            opposite_value = int.from_bytes(opposite[offset : offset + value_size], "little")
            opposite_delta = opposite_value - before
            is_opposite = delta * opposite_delta < 0
            if require_opposite_direction and not is_opposite:
                continue
            evidence.append(
                "opposite experiment changed in the opposite direction"
                if is_opposite
                else f"opposite experiment delta {opposite_delta:+d}"
            )
            score += 1.0 if is_opposite else 0.0

        candidates.append(
            {
                "offset": offset_base + offset,
                "offset_hex": f"0x{offset_base + offset:04X}",
                "size": value_size,
                "before": before,
                "after": after,
                "delta": delta,
                "control": control_value,
                "opposite": opposite_value,
                "opposite_delta": opposite_delta,
                "score": score,
                "evidence": evidence,
            }
        )
    candidates.sort(key=lambda item: (-float(item["score"]), int(item["offset"])))
    return candidates


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_delta)
        * sum(value * value for value in right_delta)
    )
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(left_delta, right_delta)) / denominator


def _differences(values: Sequence[float]) -> list[float]:
    return [after - before for before, after in zip(values, values[1:])]


def correlate_series(visual: Sequence[float], memory: Sequence[float]) -> dict[str, object]:
    """Explain cheap linear evidence between equally sampled visual/memory series."""
    if len(visual) != len(memory):
        raise PerceptionError(
            f"visual and memory series must have equal length; got {len(visual)} and {len(memory)}"
        )
    if len(visual) < 3:
        raise PerceptionError("correlation needs at least three paired samples")
    if not all(math.isfinite(float(value)) for value in (*visual, *memory)):
        raise PerceptionError("correlation samples must all be finite numbers")

    visual_values = [float(value) for value in visual]
    memory_values = [float(value) for value in memory]
    raw = _pearson(visual_values, memory_values)
    visual_delta = _differences(visual_values)
    memory_delta = _differences(memory_values)
    delta = _pearson(visual_delta, memory_delta)

    best_scale: dict[str, object] | None = None
    for factor in (1.0, 2.0, 4.0, 8.0, 16.0, 256.0):
        for sign in (1.0, -1.0):
            signed = factor * sign
            checks = (
                ("memory_delta = visual_delta * scale", memory_delta, visual_delta, signed),
                ("visual_delta = memory_delta * scale", visual_delta, memory_delta, signed),
            )
            for equation, target, source, scale in checks:
                error = sum(abs(a - b * scale) for a, b in zip(target, source)) / len(target)
                normalizer = max(1.0, sum(abs(value) for value in target) / len(target))
                relative = error / normalizer
                if best_scale is None or relative < float(best_scale["relative_error"]):
                    best_scale = {
                        "equation": equation,
                        "scale": scale,
                        "relative_error": relative,
                    }

    evidence: list[str] = []
    if raw is not None:
        evidence.append(
            f"raw values have {'inverse ' if raw < 0 else ''}Pearson correlation {raw:.4f}"
        )
    if delta is not None:
        evidence.append(
            f"first differences have {'inverse ' if delta < 0 else ''}Pearson correlation {delta:.4f}"
        )
    if best_scale is not None and float(best_scale["relative_error"]) <= 0.10:
        evidence.append(
            f"{best_scale['equation']} fits scale {best_scale['scale']:g} "
            f"with relative error {best_scale['relative_error']:.4f}"
        )
    else:
        best_scale = None

    correlations = [abs(value) for value in (raw, delta) if value is not None]
    confidence = max(correlations, default=0.0)
    if best_scale is not None:
        confidence = max(confidence, 1.0 - float(best_scale["relative_error"]))
    return {
        "samples": len(visual_values),
        "raw_correlation": round(raw, 6) if raw is not None else None,
        "delta_correlation": round(delta, 6) if delta is not None else None,
        "direction": (
            "same" if (delta if delta is not None else raw or 0) > 0
            else "inverse" if (delta if delta is not None else raw or 0) < 0
            else "undetermined"
        ),
        "fixed_scale": (
            {
                **best_scale,
                "relative_error": round(float(best_scale["relative_error"]), 6),
            }
            if best_scale is not None
            else None
        ),
        "confidence": round(confidence, 6),
        "evidence": evidence,
        "limitation": "Correlation is numeric evidence only and does not establish game semantics or causality.",
    }


def rank_correlations(
    visual: Sequence[float], memory_series: dict[str, Sequence[float]]
) -> list[dict[str, object]]:
    ranked = []
    for name, values in memory_series.items():
        result = correlate_series(visual, values)
        ranked.append({"field": name, **result})
    ranked.sort(key=lambda item: (-float(item["confidence"]), str(item["field"])))
    return ranked
