"""Immutable, portable experiment bundles with complete hash provenance."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BUNDLE_SCHEMA = "dosbox-x-mcp-experiment-v1"
MANIFEST = "manifest.json"
MACHINE_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|/home/|/Users/)[^\s\"']+")


class BundleError(RuntimeError):
    """An experiment bundle is incomplete, mutable, or corrupt."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def redact_paths(value: object) -> object:
    if isinstance(value, str):
        return MACHINE_PATH.sub("<machine-path>", value)
    if isinstance(value, list):
        return [redact_paths(item) for item in value]
    if isinstance(value, dict):
        return {str(key): redact_paths(item) for key, item in value.items()}
    return value


@dataclass
class BundleWriter:
    destination: Path
    metadata: dict[str, Any]
    bundle_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    _temporary: Path | None = field(default=None, init=False)
    _artifacts: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    _finalized: bool = field(default=False, init=False)

    def __enter__(self) -> "BundleWriter":
        self.destination = self.destination.resolve()
        if self.destination.exists():
            raise BundleError(f"bundle destination already exists: {self.destination}")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self._temporary = Path(tempfile.mkdtemp(prefix=f".{self.destination.name}.", dir=self.destination.parent))
        return self

    @property
    def root(self) -> Path:
        if self._temporary is None or self._finalized:
            raise BundleError("bundle writer is not open")
        return self._temporary

    def add_bytes(self, artifact_id: str, relative_path: str, data: bytes, *, media_type: str = "application/octet-stream", derivation: dict[str, Any] | None = None) -> dict[str, Any]:
        if artifact_id in self._artifacts:
            raise BundleError(f"duplicate artifact id {artifact_id!r}")
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts or relative.name == MANIFEST:
            raise BundleError(f"unsafe bundle path {relative_path!r}")
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        record = {
            "artifact_id": artifact_id,
            "path": relative.as_posix(),
            "bytes": len(data),
            "sha256": sha256(data),
            "media_type": media_type,
        }
        if derivation is not None:
            record["derivation"] = redact_paths(derivation)
        self._artifacts[artifact_id] = record
        return dict(record)

    def add_json(self, artifact_id: str, relative_path: str, value: object, *, derivation: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.add_bytes(artifact_id, relative_path, canonical_json(redact_paths(value)), media_type="application/json", derivation=derivation)

    def add_file(self, artifact_id: str, relative_path: str, source: Path, *, media_type: str = "application/octet-stream", derivation: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.add_bytes(artifact_id, relative_path, source.read_bytes(), media_type=media_type, derivation=derivation)

    def finalize(self) -> dict[str, Any]:
        manifest = {
            "schema": BUNDLE_SCHEMA,
            "bundle_id": self.bundle_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "finalized": True,
            "metadata": redact_paths(self.metadata),
            "artifacts": sorted(self._artifacts.values(), key=lambda item: item["artifact_id"]),
        }
        (self.root / MANIFEST).write_bytes(canonical_json(manifest))
        os.replace(self.root, self.destination)
        self._temporary = None
        self._finalized = True
        return manifest

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._temporary is not None:
            shutil.rmtree(self._temporary, ignore_errors=True)
            self._temporary = None


def verify(path: Path) -> dict[str, Any]:
    root = path.resolve()
    problems: list[str] = []
    try:
        manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as failure:
        return {"valid": False, "bundle": str(root), "problems": [f"manifest unavailable: {failure}"]}
    if manifest.get("schema") != BUNDLE_SCHEMA:
        problems.append(f"unsupported schema {manifest.get('schema')!r}")
    if manifest.get("finalized") is not True:
        problems.append("bundle is not finalized")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        problems.append("artifacts is not a list")
        artifacts = []
    seen_ids: set[str] = set()
    records_by_id: dict[str, dict[str, Any]] = {}
    declared_paths: set[str] = {MANIFEST}
    for record in artifacts:
        if not isinstance(record, dict):
            problems.append("artifact record is not an object")
            continue
        artifact_id = str(record.get("artifact_id", ""))
        relative = Path(str(record.get("path", "")))
        if not artifact_id or artifact_id in seen_ids:
            problems.append(f"missing or duplicate artifact id {artifact_id!r}")
        seen_ids.add(artifact_id)
        records_by_id[artifact_id] = record
        if relative.is_absolute() or ".." in relative.parts:
            problems.append(f"unsafe artifact path {relative}")
            continue
        declared_paths.add(relative.as_posix())
        target = root / relative
        try:
            data = target.read_bytes()
        except OSError:
            problems.append(f"missing artifact {relative.as_posix()}")
            continue
        if len(data) != record.get("bytes"):
            problems.append(f"size mismatch for {relative.as_posix()}")
        if sha256(data) != record.get("sha256"):
            problems.append(f"hash mismatch for {relative.as_posix()}")
    for artifact_id, record in records_by_id.items():
        derivation = record.get("derivation")
        if derivation is None:
            continue
        if not isinstance(derivation, dict):
            problems.append(f"invalid derivation for {artifact_id}")
            continue
        input_ids = derivation.get("input_artifact_ids")
        input_hashes = derivation.get("input_hashes")
        if not isinstance(input_ids, list) or not isinstance(input_hashes, list) or len(input_ids) != len(input_hashes):
            problems.append(f"incomplete derivation inputs for {artifact_id}")
            continue
        for input_id, expected_hash in zip(input_ids, input_hashes):
            source = records_by_id.get(str(input_id))
            if source is None:
                problems.append(f"derivation {artifact_id} references missing artifact {input_id}")
            elif source.get("sha256") != expected_hash:
                problems.append(f"derivation {artifact_id} has wrong input hash for {input_id}")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        problems.append("manifest metadata is not an object")
        metadata = {}
    if metadata.get("experiment_type") == "controlled_input_qualification":
        required_ids = {
            "emulator.build", "baseline.state", "baseline.frame", "baseline.memory",
            "timeline", "observations.memory", "derived.comparisons", "derived.camera",
            "derived.world", "derived.mosaic.pixels", "derived.mosaic.coverage",
            "derived.constraints", "evidence",
        }
        missing = sorted(required_ids - seen_ids)
        if missing:
            problems.append("missing qualification artifacts: " + ", ".join(missing))
        for field in ("executable_sha256", "profile_sha256", "mcp_version", "protocol", "framebuffer_confirmation"):
            if metadata.get(field) in (None, ""):
                problems.append(f"missing qualification metadata {field}")
        if metadata.get("post_perception_complete"):
            if not any(artifact_id.startswith("observations.trace") for artifact_id in seen_ids):
                problems.append("complete post-perception qualification has no runtime trace artifact")
            if "derived.tiles.raw" not in seen_ids or "derived.tiles.map" not in seen_ids:
                problems.append("complete post-perception qualification has no reproducible raw tile evidence")
    if root.is_dir():
        extras = sorted(file.relative_to(root).as_posix() for file in root.rglob("*") if file.is_file() and file.relative_to(root).as_posix() not in declared_paths)
        if extras:
            problems.append("undeclared files: " + ", ".join(extras))
    return {"valid": not problems, "bundle": str(root), "bundle_id": manifest.get("bundle_id"), "artifact_count": len(artifacts), "problems": problems}


def reproduce(path: Path) -> dict[str, Any]:
    """Recompute qualification-derived reports from only stored raw artifacts."""
    from . import perception
    from . import world as world_model

    root = path.resolve()
    verification = verify(root)
    if not verification["valid"]:
        raise BundleError("cannot reproduce an invalid bundle: " + "; ".join(verification["problems"]))
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("metadata", {}).get("experiment_type") != "controlled_input_qualification":
        raise BundleError("offline reproduction currently requires a controlled input qualification bundle")
    dimensions = manifest["metadata"].get("frame_dimensions", {})
    width, height = int(dimensions.get("width", 0)), int(dimensions.get("height", 0))
    if width <= 0 or height <= 0:
        raise BundleError("qualification bundle is missing positive frame_dimensions")
    records = {record["artifact_id"]: record for record in manifest["artifacts"]}

    def read(artifact_id: str) -> bytes:
        try:
            return (root / records[artifact_id]["path"]).read_bytes()
        except KeyError as failure:
            raise BundleError(f"reproduction needs missing artifact {artifact_id}") from failure

    def read_json(artifact_id: str) -> Any:
        return json.loads(read(artifact_id))

    expected_comparisons = read_json("derived.comparisons")
    trial_labels = [item["trial"] for item in expected_comparisons]
    marks = ("baseline", "held", "final")
    frames = {
        label: [read(f"frame:{label}:{mark}:indexed") for mark in marks]
        for label in trial_labels
    }
    reproduced_comparisons = [
        {
            "trial": label,
            "comparisons": [
                perception.compare_frames(before, after, width, height).as_dict()
                for before, after in zip(frames[label], frames[label][1:])
            ],
        }
        for label in trial_labels
    ]
    reproduced_camera = [
        {"trial": label, "camera": perception.estimate_camera_path(frames[label], width, height)}
        for label in trial_labels
    ]
    camera_by_trial = {item["trial"]: item["camera"] for item in reproduced_camera}
    action_labels = [label for label in trial_labels if not label.lower().startswith("idle_")]
    representative = max(
        action_labels,
        key=lambda label: sum(abs(int(step[0])) + abs(int(step[1])) for step in camera_by_trial[label]["accepted_translations"]),
    )
    camera = camera_by_trial[representative]
    steps = [tuple(step) for step in camera["accepted_translations"]]
    mosaic = world_model.build_mosaic(
        frames[representative], steps, width, height,
        placement_methods=camera["placement_methods"],
    )
    screen_regions = perception.infer_screen_regions(
        frames[representative], steps, width, height, min_episodes=1,
    )
    try:
        tile_structure = world_model.infer_tile_structure(
            mosaic.pixels, mosaic.coverage, mosaic.width, mosaic.height,
            candidate_sizes=[4, 8, 16, 32], min_tiles=8,
        )
        tile_report: dict[str, Any] = tile_structure.report()
    except world_model.WorldError as failure:
        tile_structure = None
        tile_report = {"status": "unresolved", "reason": str(failure), "epistemic_status": "no raw tile-grid candidate met the bounded evidence threshold"}

    memory = read_json("observations.memory")
    samples_by_trial = {item["trial"]: item["samples"] for item in memory}
    all_fields = {key for samples in samples_by_trial.values() for sample in samples for key in sample}
    x_field = next((name for name in ("player_x_copy", "player_x", "x") if name in all_fields), None)
    y_field = next((name for name in ("player_y_copy", "player_y", "y") if name in all_fields), None)
    directions = {"left": [-1, 0], "right": [1, 0], "up": [0, -1], "down": [0, 1]}
    movement_trials = []
    if x_field and y_field:
        for label in action_labels:
            action = label.rsplit("_r", 1)[0].lower()
            if action not in directions:
                continue
            movement_trials.append({"label": label, "direction": directions[action], "positions": [[sample[x_field], sample[y_field]] for sample in samples_by_trial[label] if x_field in sample and y_field in sample]})
    constraints = (
        world_model.infer_movement_constraints(movement_trials, min_trial_support=2)
        if movement_trials
        else {"candidate_count": 0, "movement_constraints": [], "status": "unresolved", "reason": "no paired coordinate probes were supplied", "epistemic_status": "no movement semantics asserted"}
    )
    reproduced_world = {
        "representative_trial": representative,
        "camera": camera,
        "screen_regions": screen_regions,
        "mosaic": mosaic.report(),
        "tile_structure": tile_report,
        "constraints": constraints,
        "selection_rule": "greatest total accepted indexed-pixel displacement among non-idle restored trials",
        "epistemic_status": "camera, mosaic, raw tile, and requested-motion evidence only; no game semantics confirmed",
    }

    checks: dict[str, dict[str, Any]] = {}
    json_results = {
        "derived.comparisons": reproduced_comparisons,
        "derived.camera": reproduced_camera,
        "derived.world": reproduced_world,
        "derived.constraints": constraints,
    }
    for artifact_id, actual in json_results.items():
        expected = read(artifact_id)
        generated = canonical_json(actual)
        checks[artifact_id] = {"exact": generated == expected, "stored_sha256": sha256(expected), "reproduced_sha256": sha256(generated)}
    byte_results = {
        "derived.mosaic.pixels": mosaic.pixels,
        "derived.mosaic.coverage": mosaic.coverage,
    }
    if tile_structure is not None:
        byte_results["derived.tiles.raw"] = tile_structure.tile_bytes
        json_results["derived.tiles.map"] = tile_report
        expected = read("derived.tiles.map")
        generated = canonical_json(tile_report)
        checks["derived.tiles.map"] = {"exact": generated == expected, "stored_sha256": sha256(expected), "reproduced_sha256": sha256(generated)}
    for artifact_id, generated in byte_results.items():
        expected = read(artifact_id)
        checks[artifact_id] = {"exact": generated == expected, "stored_sha256": sha256(expected), "reproduced_sha256": sha256(generated)}
    return {
        "valid": all(check["exact"] for check in checks.values()),
        "bundle": str(root),
        "bundle_id": manifest["bundle_id"],
        "checks": checks,
        "input_source": "stored raw bundle artifacts only",
    }


__all__ = ["BUNDLE_SCHEMA", "BundleError", "BundleWriter", "canonical_json", "redact_paths", "reproduce", "sha256", "verify"]
