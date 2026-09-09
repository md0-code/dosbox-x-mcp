import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dosbox_mcp import server


@pytest.fixture
def capture_root(monkeypatch, tmp_path):
    root = tmp_path / "captures"
    monkeypatch.setattr(server, "DEFAULT_OUTPUT_DIR", root)
    return root


@pytest.fixture
def dump_source(monkeypatch):
    data = bytes(range(256)) * 256  # Exactly one 64 KiB segment.
    guest = SimpleNamespace(read=Mock(return_value=data))
    session = SimpleNamespace(
        pid=4242,
        ds_base=0x90000,
        segment=Mock(return_value=data),
        guest=Mock(return_value=guest),
    )
    monkeypatch.setattr(server, "_session", lambda pid: session)
    return session, guest, data


@pytest.mark.parametrize("segment", [None, 0x1234, "0x1234"])
def test_dump_writes_exact_bytes_and_preserves_result(
    capture_root, dump_source, segment
):
    session, guest, data = dump_source
    result = asyncio.run(
        server.dosbox_dump_segment(
            path="nested/state.bin", pid=session.pid, segment=segment
        )
    )
    target = capture_root / "nested" / "state.bin"
    assert target.read_bytes() == data
    assert result == {
        "pid": session.pid,
        "path": str(target),
        "bytes": 65536,
        "source": "0x90000" if segment is None else "1234:0000",
    }
    if segment is None:
        session.segment.assert_called_once_with()
        guest.read.assert_not_called()
    else:
        session.segment.assert_not_called()
        guest.read.assert_called_once_with(0x1234, 0, 0x10000)


def test_dump_preserves_absolute_output_paths(
    capture_root, dump_source, tmp_path
):
    session, _, data = dump_source
    target = tmp_path / "outside-default" / "state.bin"
    result = asyncio.run(
        server.dosbox_dump_segment(path=str(target), pid=session.pid)
    )
    assert target.read_bytes() == data
    assert result["path"] == str(target)
    assert not capture_root.exists()


def test_dump_keeps_existing_overwrite_behavior_in_metadata_only_patch(
    capture_root, dump_source
):
    session, _, first = dump_source
    second = b"Z" * 65536
    session.segment.side_effect = [first, second]
    for expected in (first, second):
        asyncio.run(server.dosbox_dump_segment(path="state.bin", pid=session.pid))
        assert (capture_root / "state.bin").read_bytes() == expected


@pytest.fixture
def framebuffer_source(monkeypatch):
    page = bytes(range(256)) * 250  # Exactly 320 x 200 indexed pixels.
    found = SimpleNamespace(as_dict=lambda: {"fixture_location": True})
    buffer = SimpleNamespace(
        read=Mock(return_value=page),
        pages=Mock(return_value=[0]),
        found=found,
    )
    session = SimpleNamespace(
        pid=4242,
        profile=SimpleNamespace(palette=None, screen_width=320, screen_height=200),
        framebuffer=lambda: buffer,
    )
    monkeypatch.setattr(server, "_session", lambda pid: session)
    monkeypatch.setattr(server, "_locate_framebuffer", Mock(return_value=found))
    return session, buffer, page


def test_framebuffer_without_stem_does_not_export(
    capture_root, framebuffer_source, monkeypatch
):
    session, buffer, _ = framebuffer_source
    output = Mock(side_effect=AssertionError("No output path should be resolved"))
    monkeypatch.setattr(server, "_output", output)
    result = asyncio.run(server.dosbox_read_framebuffer(pid=session.pid))
    output.assert_not_called()
    assert not capture_root.exists()
    assert "files" not in result
    assert result["bytes"] == 64000
    buffer.read.assert_called_once_with(None)


@pytest.mark.parametrize("absolute", [False, True])
def test_framebuffer_exports_bin_and_png(
    capture_root, framebuffer_source, tmp_path, absolute
):
    session, _, page = framebuffer_source
    target = (
        tmp_path / "outside-default" / "frame"
        if absolute else capture_root / "nested" / "frame"
    )
    argument = str(target) if absolute else "nested/frame"
    result = asyncio.run(
        server.dosbox_read_framebuffer(pid=session.pid, stem=argument)
    )
    binary_path = target.with_suffix(".bin")
    image_path = target.with_suffix(".png")
    assert binary_path.read_bytes() == page
    assert image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert result["files"] == [str(binary_path), str(image_path)]
    assert result["bytes"] == 64000
    assert result["pid"] == session.pid


def test_framebuffer_keeps_existing_overwrite_behavior_in_metadata_only_patch(
    capture_root, framebuffer_source
):
    session, buffer, first = framebuffer_source
    second = b"Q" * 64000
    buffer.read.side_effect = [first, second]
    for expected in (first, second):
        asyncio.run(server.dosbox_read_framebuffer(pid=session.pid, stem="frame"))
        assert (capture_root / "frame.bin").read_bytes() == expected
        # Exercise the real encoder; detailed PNG correctness belongs to VGA tests.
        assert (capture_root / "frame.png").read_bytes() == server.vga.png(
            expected, None, 320, 200
        )


@pytest.mark.parametrize('segment', [0x10000, '0x10000', -1])
def test_invalid_dump_segment_creates_no_output(capture_root, dump_source, segment):
    session, _, _ = dump_source
    with pytest.raises(server.SessionError, match='segments are 16-bit'):
        asyncio.run(server.dosbox_dump_segment('nested/state.bin', session.pid, segment))
    assert not capture_root.exists()


@pytest.mark.parametrize('segment', [None, 0x1234])
def test_dump_read_failure_propagates_without_output(capture_root, dump_source, segment):
    session, guest, _ = dump_source
    error = server.ProcessError('guest read failed')
    (session.segment if segment is None else guest.read).side_effect = error
    with pytest.raises(server.ProcessError) as raised:
        asyncio.run(server.dosbox_dump_segment('nested/state.bin', session.pid, segment))
    assert raised.value is error
    assert not capture_root.exists()


@pytest.mark.parametrize('capture', ['dump', 'framebuffer'])
def test_host_write_failure_propagates(
    capture_root, dump_source, framebuffer_source, monkeypatch, capture
):
    from pathlib import Path

    session = dump_source[0] if capture == 'dump' else framebuffer_source[0]
    monkeypatch.setattr(server, '_session', lambda pid: session)
    target = capture_root / 'denied.bin'
    real_write = Path.write_bytes
    error = PermissionError('fixture write denied')

    def write(path, data):
        if path == target:
            raise error
        return real_write(path, data)

    monkeypatch.setattr(Path, 'write_bytes', write)
    with pytest.raises(PermissionError) as raised:
        asyncio.run(
            server.dosbox_dump_segment('denied.bin', session.pid)
            if capture == 'dump'
            else server.dosbox_read_framebuffer(session.pid, stem='denied')
        )
    assert raised.value is error
    assert not target.exists()
    assert not target.with_suffix('.png').exists()


def test_framebuffer_locator_error_propagates_without_output(
    capture_root, framebuffer_source, monkeypatch
):
    session, _, _ = framebuffer_source
    error = server.vga.VgaError('page could not be confirmed')
    monkeypatch.setattr(server, '_locate_framebuffer', Mock(side_effect=error))
    with pytest.raises(server.vga.VgaError) as raised:
        asyncio.run(server.dosbox_read_framebuffer(session.pid, stem='nested/frame'))
    assert raised.value is error
    assert not capture_root.exists()


def test_framebuffer_read_failure_creates_no_output(capture_root, framebuffer_source):
    session, buffer, _ = framebuffer_source
    buffer.read.side_effect = server.ProcessError('video read failed')
    with pytest.raises(server.ProcessError, match='video read failed'):
        asyncio.run(server.dosbox_read_framebuffer(session.pid, stem='nested/frame'))
    assert not capture_root.exists()


def test_framebuffer_preserves_page_and_location_options(
    capture_root, framebuffer_source, monkeypatch
):
    session, buffer, _ = framebuffer_source
    found = buffer.found

    def locate(current, reference, allow_unconfirmed):
        assert current is session
        assert buffer.found is None  # relocate cleared the cache before locating.
        assert reference == 'reference.bin'
        assert allow_unconfirmed is True
        return found

    monkeypatch.setattr(server, '_locate_framebuffer', locate)
    asyncio.run(server.dosbox_read_framebuffer(
        session.pid, page_offset=65536, relocate=True,
        reference='reference.bin', allow_unconfirmed=True,
    ))
    buffer.read.assert_called_once_with(65536)
    assert not capture_root.exists()


@pytest.mark.parametrize('absolute', [False, True])
def test_framebuffer_paths_suffixes_and_result(
    capture_root, framebuffer_source, tmp_path, absolute
):
    session, _, page = framebuffer_source
    stem = tmp_path / 'outside-default/frame.old' if absolute else 'nested/frame.old'
    target = stem if absolute else capture_root / stem
    result = asyncio.run(server.dosbox_read_framebuffer(session.pid, stem=str(stem)))
    assert result == {
        'pid': session.pid, 'fixture_location': True, 'pages': [0],
        'distinct_indices': 256, 'bytes': 64000,
        'files': [str(target.with_suffix('.bin')), str(target.with_suffix('.png'))],
    }
    assert target.with_suffix('.bin').read_bytes() == page
    assert target.with_suffix('.png').read_bytes() == server.vga.png(page, None, 320, 200)
    assert not target.exists()
    if absolute:
        assert not capture_root.exists()


@pytest.mark.parametrize('capture', ['screen', 'watch', 'observe'])
def test_other_frame_exporters_replace_existing_files(
    capture_root, framebuffer_source, capture
):
    session, buffer, page = framebuffer_source
    capture_root.mkdir()
    if capture == 'screen':
        raw = capture_root / 'frame.rgb'
        png = capture_root / 'frame.png'
        raw.write_bytes(b'old raw')
        call = server.dosbox_capture_screen(
            'frame', session.pid, source='vram', with_segment=False,
        )
    elif capture == 'watch':
        raw = None
        png = capture_root / 'frame_000.png'
        buffer.watch = Mock(return_value=[(0.0, page)])
        call = server.dosbox_watch_frames(0.01, 'frame', session.pid)
    else:
        raw = capture_root / 'frame_frame_000.bin'
        png = capture_root / 'frame_frame_000.png'
        raw.write_bytes(b'old raw')
        call = server.dosbox_act_and_observe(
            server.ExperimentAction(kind='idle'), duration_seconds=0.01,
            interval_seconds=0.01, pid=session.pid, stem='frame',
        )
    png.write_bytes(b'old png')
    result = asyncio.run(call)
    assert str(png) in result['files']
    assert png.read_bytes() == server.vga.png(page, None, 320, 200)
    if raw is not None:
        assert raw.read_bytes() == page


def test_sample_replaces_tsv_and_segment(capture_root, dump_source):
    session, _, data = dump_source
    session.resolve_watch = Mock(return_value=[('value', 0, 1)])
    session.sample = Mock(return_value=[{'t_ms': 0, 'value': 7}])
    capture_root.mkdir()
    tsv = capture_root / 'sample.tsv'
    dump = capture_root / 'sample_entry_ds.bin'
    tsv.write_bytes(b'old rows')
    dump.write_bytes(b'old segment')
    result = asyncio.run(server.dosbox_sample(['value'], pid=session.pid, out='sample'))
    assert result['files'] == [str(tsv), str(dump)]
    assert tsv.read_text() == 't_ms\tvalue\n0\t7\n'
    assert dump.read_bytes() == data


@pytest.mark.parametrize('action', ['save_state', 'load_state'])
def test_emulator_command_dispatches_state_mutations(dump_source, monkeypatch, action):
    session, _, _ = dump_source
    command = Mock(return_value={'action': action})
    monkeypatch.setattr(server.emulator, 'command', command)
    result = asyncio.run(server.dosbox_emulator_command(action, session.pid))
    command.assert_called_once_with(session.pid, action)
    assert result == {'pid': session.pid, 'action': action}


def test_video_snapshot_replaces_raw_palette_planes_and_metadata(
    capture_root, dump_source
):
    import json

    session, _, _ = dump_source
    raw = bytes(range(16))
    palette = bytes(768)
    client = SimpleNamespace(request=Mock(return_value={
        'raw_hex': raw.hex(), 'dac_hex': palette.hex(), 'layout': 'plane_interleaved',
    }))
    session.companion = Mock(return_value=client)
    directory = capture_root / 'video'
    directory.mkdir(parents=True)
    for name in ['vga-memory.bin', 'dac.bin', 'metadata.json'] + [f'plane-{p}.bin' for p in range(4)]:
        (directory / name).write_bytes(b'old artifact')
    result = asyncio.run(server.dosbox_capture_video_snapshot('video', pid=session.pid))
    assert (directory / 'vga-memory.bin').read_bytes() == raw
    assert (directory / 'dac.bin').read_bytes() == palette
    for plane in range(4):
        assert (directory / f'plane-{plane}.bin').read_bytes() == raw[plane::4]
    assert json.loads((directory / 'metadata.json').read_text())['layout'] == 'plane_interleaved'
    assert all(item['path'].startswith(str(directory)) for item in result['artifacts'])


def test_decode_replaces_output_and_retries_with_same_content(capture_root):
    capture_root.mkdir()
    source = capture_root / 'source.bin'
    output = capture_root / 'decoded.bin'
    source.write_bytes(bytes(range(4)))
    output.write_bytes(b'old output')
    for _ in range(2):
        result = asyncio.run(server.dosbox_decode_video_snapshot(
            'vesa_linear', ['source.bin'], 'decoded.bin', width=2, height=2,
            pitch=2, reference='source.bin',
        ))
        assert result['confirmed'] is True
        assert output.read_bytes() == source.read_bytes()


def test_mosaic_replaces_artifacts_and_retries_with_same_content(capture_root):
    capture_root.mkdir()
    source = capture_root / 'source.bin'
    source.write_bytes(bytes(range(16)))
    outputs = [capture_root / name for name in ('mosaic.bin', 'mosaic.png', 'mosaic_coverage.bin')]
    for path in outputs:
        path.write_bytes(b'old output')
    first = None
    for _ in range(2):
        result = asyncio.run(server.dosbox_build_world_mosaic(
            ['source.bin', 'source.bin'], 'mosaic', width=4, height=4,
            translations=[server.TranslationStep(dx=0, dy=0)],
        ))
        assert set(result['files'].values()) == {str(path) for path in outputs}
        assert outputs[0].read_bytes() == source.read_bytes()
        assert outputs[1].read_bytes() == server.vga.png(source.read_bytes(), None, 4, 4)
        assert outputs[2].read_bytes() != b'old output'
        contents = [path.read_bytes() for path in outputs]
        if first is not None:
            assert contents == first
        first = contents
