"""Small persistent notebooks that keep evidence separate from profiles."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


Status = Literal["observed", "inferred", "confirmed"]
STATUSES = {"observed", "inferred", "confirmed"}


class EvidenceError(RuntimeError):
    pass


def _empty() -> dict[str, object]:
    return {"version": 1, "entries": {}}


def load(path: Path) -> dict[str, object]:
    if not path.exists():
        return _empty()
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as failure:
        raise EvidenceError(f"could not read evidence notebook {path}: {failure}") from None
    if not isinstance(notebook, dict) or notebook.get("version") != 1:
        raise EvidenceError(f"{path} is not an evidence notebook version 1")
    entries = notebook.get("entries")
    if not isinstance(entries, dict):
        raise EvidenceError(f"{path} has no valid entries object")
    for subject, entry in entries.items():
        if not isinstance(subject, str) or not isinstance(entry, dict):
            raise EvidenceError(f"{path} contains an invalid evidence entry")
        if entry.get("status") not in STATUSES:
            raise EvidenceError(f"{path} entry {subject!r} has an invalid status")
    return notebook


def _write(path: Path, notebook: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(notebook, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def upsert(
    path: Path,
    subject: str,
    status: Status,
    evidence: list[str],
    label: str | None = None,
    confidence: float | None = None,
    now: str | None = None,
) -> dict[str, object]:
    subject = subject.strip()
    if not subject:
        raise EvidenceError("evidence subject cannot be empty")
    if status not in STATUSES:
        raise EvidenceError(f"invalid evidence status {status!r}")
    if confidence is not None and not 0 <= confidence <= 1:
        raise EvidenceError("confidence must be between 0 and 1")
    added = [item.strip() for item in evidence if item.strip()]
    if not added:
        raise EvidenceError("record at least one concrete evidence statement")

    notebook = load(path)
    entries = notebook["entries"]
    assert isinstance(entries, dict)
    previous = entries.get(subject, {})
    previous_evidence = previous.get("evidence", []) if isinstance(previous, dict) else []
    combined = list(dict.fromkeys([*previous_evidence, *added]))
    timestamp = now or datetime.now(timezone.utc).isoformat()
    history = list(previous.get("history", [])) if isinstance(previous, dict) else []
    history.append(
        {
            "at": timestamp,
            "status": status,
            "label": label,
            "confidence": confidence,
            "evidence_added": added,
        }
    )
    entry = {
        "subject": subject,
        "label": label,
        "status": status,
        "confidence": confidence,
        "evidence": combined,
        "created_at": previous.get("created_at", timestamp) if isinstance(previous, dict) else timestamp,
        "updated_at": timestamp,
        "history": history,
    }
    entries[subject] = entry
    _write(path, notebook)
    return entry


def remove(path: Path, subject: str) -> bool:
    notebook = load(path)
    entries = notebook["entries"]
    assert isinstance(entries, dict)
    existed = entries.pop(subject, None) is not None
    if existed:
        _write(path, notebook)
    return existed


def entries(
    path: Path, status: Status | None = None, subject: str | None = None
) -> list[dict[str, object]]:
    notebook = load(path)
    found = notebook["entries"]
    assert isinstance(found, dict)
    values = [entry for entry in found.values() if isinstance(entry, dict)]
    if status is not None:
        values = [entry for entry in values if entry.get("status") == status]
    if subject is not None:
        values = [entry for entry in values if entry.get("subject") == subject]
    return sorted(values, key=lambda entry: str(entry.get("subject", "")))


__all__ = ["EvidenceError", "Status", "entries", "load", "remove", "upsert"]
