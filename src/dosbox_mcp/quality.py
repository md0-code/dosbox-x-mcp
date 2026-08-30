"""Machine-readable validity gates for controlled DOS experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ResultState = Literal["valid", "valid_with_ambiguity", "invalid", "incomplete"]


@dataclass(frozen=True)
class Gate:
    name: str
    status: Literal["pass", "ambiguous", "fail", "missing"]
    observed: object
    threshold: object
    message: str

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "status": self.status, "observed": self.observed, "threshold": self.threshold, "message": self.message}


def evaluate(*, exact_framebuffer: bool | None, baseline_hashes: list[str], restore_hash: str | None, acknowledged_input: bool | None, acknowledged_restore: bool | None, transition_fraction: float | None, repeats: int, idle_repeats: int, camera_confidence: float | None, cleanup_released: bool | None, strict: bool = True, max_transition_fraction: float = 0.6, min_camera_confidence: float = 0.5, repeat_agreement: float = 1.0, min_repeat_agreement: float = 0.75, output_within_limits: bool | None = True, artifacts_complete: bool | None = True, final_restore_ok: bool | None = True) -> dict[str, Any]:
    gates: list[Gate] = []

    def boolean(name: str, value: bool | None, message: str) -> None:
        gates.append(Gate(name, "missing" if value is None else ("pass" if value else "fail"), value, True, message))

    boolean("exact_framebuffer", exact_framebuffer, "framebuffer backing must be reference-confirmed")
    baseline_stable = bool(baseline_hashes) and len(set(baseline_hashes)) == 1
    gates.append(Gate("baseline_stability", "missing" if not baseline_hashes else ("pass" if baseline_stable else "fail"), baseline_hashes, "all hashes equal", "repeated idle baselines must be byte-identical"))
    restore_ok = restore_hash is not None and bool(baseline_hashes) and restore_hash == baseline_hashes[0]
    gates.append(Gate("restore_integrity", "missing" if restore_hash is None or not baseline_hashes else ("pass" if restore_ok else "fail"), restore_hash, baseline_hashes[0] if baseline_hashes else None, "post-restore frame must equal the baseline"))
    boolean("input_acknowledged", acknowledged_input, "input application and release require emulator acknowledgements")
    boolean("restore_acknowledged", acknowledged_restore, "state load and first post-restore frame require emulator acknowledgements")
    gates.append(Gate("transition_overlap", "missing" if transition_fraction is None else ("pass" if transition_fraction <= max_transition_fraction else "fail"), transition_fraction, max_transition_fraction, "state transitions must not dominate the response window"))
    gates.append(Gate("repeats", "pass" if repeats >= 2 and idle_repeats >= 2 else "fail", {"action": repeats, "idle": idle_repeats}, {"action": 2, "idle": 2}, "action and idle controls require repeats"))
    gates.append(Gate("repeat_agreement", "pass" if repeat_agreement >= min_repeat_agreement else "fail", repeat_agreement, min_repeat_agreement, "independent restored repeats must agree on channel presence"))
    camera_status = "missing" if camera_confidence is None else ("pass" if camera_confidence >= min_camera_confidence else "ambiguous")
    gates.append(Gate("camera_confidence", camera_status, camera_confidence, min_camera_confidence, "weak camera evidence remains explicitly ambiguous"))
    boolean("cleanup", cleanup_released, "all held inputs must be released")
    boolean("final_restore", final_restore_ok, "the exact baseline must be restored after the final trial")
    boolean("output_limits", output_within_limits, "all samples and derived outputs must remain within configured bounds")
    boolean("artifact_completeness", artifacts_complete, "all raw inputs required by derived results must be retained")

    statuses = {gate.status for gate in gates}
    if "fail" in statuses:
        state: ResultState = "invalid"
    elif "missing" in statuses:
        state = "incomplete" if strict else "valid_with_ambiguity"
    elif "ambiguous" in statuses:
        state = "valid_with_ambiguity"
    else:
        state = "valid"
    return {"state": state, "policy": "strict" if strict else "exploratory", "gates": [gate.as_dict() for gate in gates], "notebook_update_allowed": state in {"valid", "valid_with_ambiguity"}}


__all__ = ["Gate", "ResultState", "evaluate"]
