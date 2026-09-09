import asyncio

import pytest

from dosbox_mcp import server


HINTS = (
    "readOnlyHint",
    "destructiveHint",
    "idempotentHint",
    "openWorldHint",
)
EXPORTERS = ("dosbox_dump_segment", "dosbox_read_framebuffer")


@pytest.fixture(scope="module")
def published_tools():
    # Exercise the same public SDK method used by the existing CI smoke test.
    listed = asyncio.run(server.mcp.list_tools())
    assert listed, "The MCP server registered no tools"
    names = [tool.name for tool in listed]
    assert len(names) == len(set(names)), "Duplicate tool names"
    return {tool.name: tool for tool in listed}


@pytest.mark.parametrize("name", EXPORTERS)
def test_exporters_advertise_their_side_effects(published_tools, name):
    tool = published_tools[name]
    actual = tool.model_dump(mode="json", exclude_none=True).get("annotations", {})
    expected = {
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": True,
    }
    for hint, value in expected.items():
        assert actual.get(hint) is value, (name, hint, actual)


def test_all_published_tools_have_explicit_boolean_hints(published_tools):
    # Repository policy, not a claim that base MCP requires all four fields.
    problems = []
    for name, tool in sorted(published_tools.items()):
        descriptor = tool.model_dump(mode="json", exclude_none=True)
        annotations = descriptor.get("annotations", {})
        for hint in HINTS:
            if type(annotations.get(hint)) is not bool:
                problems.append(f"{name}: {hint} is absent or not a boolean")
    assert not problems, "\n".join(problems)


def test_published_schemas_match_pre_patch_contract(published_tools):
    # Fingerprints from public list_tools() at d5b0867, before annotation edits.
    # An intentional API change must review and update this contract fixture.
    import hashlib
    import json
    from pathlib import Path

    expected = json.loads(
        (Path(__file__).parent / 'fixtures/tool_schema_hashes.json').read_text()
    )
    actual = {}
    for name, tool in published_tools.items():
        descriptor = tool.model_dump(mode='json', exclude_none=True)
        schemas = {key: descriptor.get(key) for key in ('inputSchema', 'outputSchema')}
        payload = json.dumps(schemas, sort_keys=True, separators=(',', ':')).encode()
        actual[name] = hashlib.sha256(payload).hexdigest()
    assert actual == expected


@pytest.mark.parametrize('name', [
    'dosbox_capture_screen', 'dosbox_emulator_command',
    'dosbox_build_world_mosaic', 'dosbox_act_and_observe',
    'dosbox_watch_frames', 'dosbox_sample',
    'dosbox_capture_video_snapshot', 'dosbox_decode_video_snapshot',
])
def test_additional_overwrite_and_restore_branches_are_destructive(published_tools, name):
    annotations = published_tools[name].annotations
    assert annotations.readOnlyHint is False
    assert annotations.destructiveHint is True
    assert annotations.idempotentHint is (
        name in {'dosbox_build_world_mosaic', 'dosbox_decode_video_snapshot'}
    )
