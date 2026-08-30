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
