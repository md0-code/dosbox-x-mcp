"""Pure analysis and canonical artifact helpers for controlled qualifications."""
from __future__ import annotations

import hashlib
from typing import Any, Sequence

from . import perception


def frame_hash(frame: bytes) -> str:
    return hashlib.sha256(frame).hexdigest()


def analyze_trial(
    action: str,
    frames: Sequence[bytes],
    counters: Sequence[int],
    memory_samples: Sequence[dict[str, int]],
    width: int,
    height: int,
    *,
    control: bool = False,
    max_dx: int = 16,
    max_dy: int = 16,
) -> dict[str, Any]:
    if len(frames) < 2 or len(frames) != len(counters):
        raise ValueError("a qualification trial needs aligned frame counters and at least two frames")
    comparisons = [perception.compare_frames(frames[index - 1], frames[index], width, height) for index in range(1, len(frames))]
    camera = perception.estimate_camera_path(frames, width, height, max_dx=max_dx, max_dy=max_dy)
    tracking = perception.track_regions(comparisons)
    fields = sorted({key for sample in memory_samples for key in sample if key != "counter"})
    memory_deltas = {
        field: int(memory_samples[-1].get(field, memory_samples[0].get(field, 0))) - int(memory_samples[0].get(field, 0))
        for field in fields
    } if memory_samples else {}
    transition_fraction = max((comparison.changed_fraction for comparison in comparisons), default=0.0)
    global_motion = camera["pairwise"][-1] if camera.get("pairwise") else None
    return {
        "action": action,
        "control": control,
        "frame_counters": list(counters),
        "frame_hashes": [frame_hash(frame) for frame in frames],
        "comparisons": [comparison.as_dict() for comparison in comparisons],
        "global_motion": global_motion,
        "camera": camera,
        "tracks": tracking["tracks"],
        "memory_samples": list(memory_samples),
        "memory_deltas": memory_deltas,
        "transition_fraction": transition_fraction,
    }


def repeat_agreement(trials: Sequence[dict[str, Any]]) -> float:
    """Agreement of channel presence across repeats, independent of semantics."""
    grouped: dict[str, list[tuple[bool, bool, bool]]] = {}
    for trial in trials:
        if trial.get("control"):
            continue
        motion = trial.get("global_motion") or {}
        camera = bool(motion) and float(motion.get("confidence", 0.0)) >= 0.25
        local = any(float(track.get("confidence", 0.0)) >= 0.25 for track in trial.get("tracks", []))
        memory = any(int(value) != 0 for value in trial.get("memory_deltas", {}).values())
        grouped.setdefault(str(trial.get("action", "")), []).append((camera, local, memory))
    scores: list[float] = []
    for signatures in grouped.values():
        if len(signatures) < 2:
            scores.append(0.0)
            continue
        for channel in range(3):
            present = sum(signature[channel] for signature in signatures)
            scores.append(max(present, len(signatures) - present) / len(signatures))
    return sum(scores) / len(scores) if scores else 0.0


__all__ = ["analyze_trial", "frame_hash", "repeat_agreement"]
