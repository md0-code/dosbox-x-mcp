"""Offline tests for world reconstruction and movement-constraint evidence."""
from __future__ import annotations

import asyncio

import pytest

from dosbox_mcp import server
from dosbox_mcp.world import (
    WorldError,
    build_mosaic,
    infer_movement_constraints,
    infer_tile_structure,
)


def test_mosaic_places_camera_motion_in_world_coordinates():
    first = bytes((1, 2, 3, 4, 5, 6, 7, 8))
    second = bytes((2, 3, 4, 9, 6, 7, 8, 10))
    mosaic = build_mosaic([first, second], [(-1, 0)], 4, 2)
    assert (mosaic.width, mosaic.height) == (5, 2)
    assert mosaic.pixels == bytes((1, 2, 3, 4, 9, 5, 6, 7, 8, 10))
    assert mosaic.coverage == bytes((1, 2, 2, 2, 1, 1, 2, 2, 2, 1))
    assert mosaic.conflict_observations == 0
    assert mosaic.placements[1]["camera_origin"] == [1, 0]
    assert mosaic.placements[1]["placement_method"] == "direct"


def test_mosaic_records_consensus_and_unresolved_placement_provenance():
    mosaic = build_mosaic(
        [bytes((1, 2)), bytes((2, 3)), bytes((2, 3))],
        [(-1, 0), (0, 0)],
        2,
        1,
        placement_methods=["direct", "consensus_corrected", "unresolved"],
    )
    assert [item["placement_method"] for item in mosaic.placements] == [
        "direct", "consensus_corrected", "unresolved"
    ]


def test_mosaic_preserves_first_observation_and_reports_conflicts():
    first = bytes((1, 2, 3, 4))
    second = bytes((9, 3, 4, 5))
    mosaic = build_mosaic([first, second], [(-1, 0)], 4, 1)
    assert mosaic.pixels == bytes((1, 2, 3, 4, 5))
    assert mosaic.conflict_observations == 1
    assert mosaic.conflict_pixels == 1
    assert mosaic.report()["conflict_fraction_of_overlap"] == pytest.approx(1 / 3)


def test_mosaic_validates_dimensions_translations_and_size():
    with pytest.raises(WorldError, match="one translation"):
        build_mosaic([bytes(4), bytes(4)], [], 2, 2)
    with pytest.raises(WorldError, match="limit"):
        build_mosaic([bytes(4), bytes(4)], [(-2, 0)], 2, 2, max_mosaic_pixels=4)


def test_tile_structure_ranks_repeated_blocks_and_keeps_raw_ids():
    tile_a = ((1, 2), (3, 4))
    tile_b = ((5, 6), (7, 8))
    rows = []
    for tile_y in range(4):
        for inner_y in range(2):
            row = []
            for tile_x in range(4):
                tile = tile_a if (tile_x + tile_y) % 2 == 0 else tile_b
                row.extend(tile[inner_y])
            rows.extend(row)
    pixels = bytes(rows)
    structure = infer_tile_structure(
        pixels, bytes([1]) * len(pixels), 8, 8, candidate_sizes=[2, 4], min_tiles=4
    )
    report = structure.report()
    assert report["tile_width"] == 2
    assert report["tile_count"] == 2
    assert report["best_candidate"]["repeat_fraction"] == 1
    assert len(structure.tile_bytes) == 8
    assert set(value for row in structure.tile_map for value in row) == {0, 1}


def test_repeated_stalls_become_neutral_movement_constraints():
    result = infer_movement_constraints(
        [
            {
                "label": "right_a",
                "direction": [1, 0],
                "positions": [[0, 0], [1, 0], [2, 0], [2.1, 0], [2.1, 0], [2.1, 0]],
            },
            {
                "label": "right_b",
                "direction": [2, 0],
                "positions": [[0, 1], [1, 1], [2, 1], [2, 1], [2, 1], [2, 1]],
            },
            {
                "label": "unrepeated",
                "direction": [1, 0],
                "positions": [[8, 8], [9, 8], [9, 8], [9, 8], [9, 8]],
            },
        ],
        min_stalled_steps=3,
        position_tolerance=1.5,
    )
    assert result["candidate_count"] == 1
    candidate = result["movement_constraints"][0]
    assert candidate["supporting_trials"] == ["right_a", "right_b"]
    assert candidate["position"] == [2.0, 0.5]
    assert "not identified as walls" in result["epistemic_status"]


def test_movement_constraints_validate_trial_evidence():
    with pytest.raises(WorldError, match="non-zero"):
        infer_movement_constraints(
            [{"label": "bad", "direction": [0, 0], "positions": [[0, 0], [0, 0]]}]
        )


def test_world_mosaic_and_constraint_mcp_tools_round_trip(tmp_path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(bytes((1, 2, 3, 4)))
    second.write_bytes(bytes((2, 3, 4, 5)))
    result = asyncio.run(
        server.dosbox_build_world_mosaic(
            [str(first), str(second)],
            stem=str(tmp_path / "world"),
            width=4,
            height=1,
            translations=[server.TranslationStep(dx=-1, dy=0)],
        )
    )
    assert result["width"] == 5
    assert (tmp_path / "world.bin").read_bytes() == bytes((1, 2, 3, 4, 5))
    assert (tmp_path / "world.png").read_bytes().startswith(b"\x89PNG")

    constraints = server.dosbox_infer_movement_constraints(
        [
            server.MovementTrial(
                label="a",
                direction=server.PositionSample(x=1, y=0),
                positions=[server.PositionSample(x=2, y=3)] * 4,
            )
        ],
        min_trial_support=1,
    )
    assert constraints["candidate_count"] == 1
