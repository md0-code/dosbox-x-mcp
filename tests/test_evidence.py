"""Persistent evidence stays explicit, auditable, and profile-independent."""
from __future__ import annotations

import asyncio
import json

import pytest

from dosbox_mcp import evidence, server


def test_upsert_merges_evidence_and_retains_revision_history(tmp_path):
    path = tmp_path / "evidence.json"
    first = evidence.upsert(
        path,
        "memory:DS:19A4",
        "observed",
        ["increased during RIGHT"],
        now="2026-01-01T00:00:00+00:00",
    )
    second = evidence.upsert(
        path,
        "memory:DS:19A4",
        "inferred",
        ["increased during RIGHT", "decreased during LEFT"],
        label="candidate_x",
        confidence=0.9,
        now="2026-01-02T00:00:00+00:00",
    )
    assert first["status"] == "observed"
    assert second["status"] == "inferred"
    assert second["evidence"] == ["increased during RIGHT", "decreased during LEFT"]
    assert len(second["history"]) == 2
    assert second["created_at"] == "2026-01-01T00:00:00+00:00"
    assert evidence.entries(path, status="inferred") == [second]


def test_notebook_rejects_empty_evidence_and_invalid_documents(tmp_path):
    path = tmp_path / "evidence.json"
    with pytest.raises(evidence.EvidenceError, match="at least one"):
        evidence.upsert(path, "memory:DS:1", "observed", [])
    path.write_text('{"version":2,"entries":{}}', encoding="utf-8")
    with pytest.raises(evidence.EvidenceError, match="version 1"):
        evidence.load(path)


def test_remove_is_explicit_and_preserves_other_entries(tmp_path):
    path = tmp_path / "evidence.json"
    evidence.upsert(path, "a", "observed", ["A"])
    evidence.upsert(path, "b", "confirmed", ["B"])
    assert evidence.remove(path, "a") is True
    assert evidence.remove(path, "missing") is False
    assert [entry["subject"] for entry in evidence.entries(path)] == ["b"]


def test_mcp_notebook_round_trip(tmp_path):
    path = tmp_path / "project-evidence.json"
    recorded = asyncio.run(
        server.dosbox_evidence_notebook(
            operation="upsert",
            path=str(path),
            subject="visual:track:2",
            status="inferred",
            evidence=["moves right during RIGHT branch"],
            label="input-responsive track",
            confidence=0.75,
        )
    )
    listed = asyncio.run(server.dosbox_evidence_notebook(path=str(path)))
    assert recorded["entry"]["status"] == "inferred"
    assert listed["count"] == 1
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1
