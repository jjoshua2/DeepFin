import hashlib
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from scripts.benchmark_zarr_zip_decoder import owned_decode, pack


def test_pack_preserves_every_metadata_and_compressed_chunk_byte(tmp_path):
    source = tmp_path / "source.zarr"
    source.mkdir()
    (source / ".zgroup").write_bytes(b'{"zarr_format":2}')
    (source / ".zattrs").write_bytes(b'{"row_schema":3}')
    (source / "x").mkdir()
    (source / "x/.zarray").write_bytes(b'{"compressor":{"id":"blosc"}}')
    (source / "x/0.0").write_bytes(bytes(range(256)))
    archive = tmp_path / "source.zarr.zip"
    proof = pack(source, archive)
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {".zgroup", ".zattrs", "x/.zarray", "x/0.0"}
        for entry in z.infolist():
            assert entry.compress_type == zipfile.ZIP_STORED
            assert z.read(entry) == (source / entry.filename).read_bytes()
    assert proof["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="fresh"):
        pack(source, archive)


def test_pack_rejects_linked_payload(tmp_path):
    source = tmp_path / "source.zarr"
    source.mkdir()
    (source / ".zgroup").write_text("{}")
    (source / ".zattrs").write_text("{}")
    (source / "linked").symlink_to(source / ".zgroup")
    with pytest.raises(ValueError, match="linked"):
        pack(source, tmp_path / "output.zip")


@pytest.mark.parametrize("fail", [False, True])
def test_decoder_closes_all_owned_stores_and_restores_opener(fail):
    closed = []
    store = SimpleNamespace(close=lambda: closed.append("closed"))

    def original(*_args, **_kwargs):
        return SimpleNamespace(store=store)

    zarr = SimpleNamespace(open_group=original)

    def loader(path, **kwargs):
        assert kwargs == {"lazy": False, "validate": True}
        zarr.open_group(path)
        zarr.open_group(path)
        if fail:
            raise RuntimeError("decode error")
        return {}, {}

    if fail:
        with pytest.raises(RuntimeError, match="decode error"):
            owned_decode(Path("dummy.zip"), loader, zarr)
    else:
        assert owned_decode(Path("dummy.zip"), loader, zarr) == ({}, {})
    assert closed == ["closed"]
    assert zarr.open_group is original


def test_guard_exits_even_when_failure_receipt_cannot_be_written(tmp_path, monkeypatch):
    from scripts import benchmark_zarr_zip_decoder as pilot

    blocker = tmp_path / "not-a-directory"
    blocker.write_text("file")

    def exit_process(code):
        assert code == 9
        raise SystemExit(code)

    monkeypatch.setattr(pilot.os, "_exit", exit_process)
    with pytest.raises(SystemExit):
        pilot.abort(blocker, "measure", RuntimeError("RSS limit"))


@pytest.mark.parametrize("statm", ["100 7 2 0 0 0 0", "100 0 0 0 0 0 0"])
def test_linux_rss_uses_current_resident_pages(monkeypatch, statm):
    from scripts import benchmark_zarr_zip_decoder as pilot

    monkeypatch.setattr(pilot.Path, "read_text", lambda _path: statm)
    monkeypatch.setattr(pilot.os, "sysconf", lambda _key: 4096)
    assert pilot.current_rss_bytes() == int(statm.split()[1]) * 4096


@pytest.mark.parametrize("statm", ["", "100", "100 invalid", "100 -1"])
def test_linux_rss_rejects_unreadable_or_invalid_samples(monkeypatch, statm):
    from scripts import benchmark_zarr_zip_decoder as pilot

    monkeypatch.setattr(pilot.Path, "read_text", lambda _path: statm)
    with pytest.raises((ValueError, IndexError)):
        pilot.current_rss_bytes()


def test_linux_rss_propagates_missing_proc_read(monkeypatch):
    from scripts import benchmark_zarr_zip_decoder as pilot

    def missing(_path):
        raise FileNotFoundError("statm unavailable")

    monkeypatch.setattr(pilot.Path, "read_text", missing)
    with pytest.raises(FileNotFoundError, match="unavailable"):
        pilot.current_rss_bytes()


def test_required_external_path_refuses_missing_value(tmp_path):
    from scripts.benchmark_zarr_zip_decoder import required_path

    with pytest.raises(ValueError, match="external output required"):
        required_path(None, "external output required")
    assert required_path(tmp_path, "external output required") == tmp_path


@pytest.mark.parametrize(("error", "expected"), [(None, True), (ProcessLookupError, False),
                                             (PermissionError, True)])
def test_predecessor_pid_gate_is_conservative(monkeypatch, error, expected):
    from scripts import benchmark_zarr_zip_decoder as pilot

    def probe(pid, signal):
        assert (pid, signal) == (12345, 0)
        if error is not None:
            raise error()

    monkeypatch.setattr(pilot.os, "kill", probe)
    assert pilot.pid_exists(12345) is expected


@pytest.mark.parametrize("pid", [0, -1])
def test_predecessor_pid_gate_refuses_nonpositive_pid(pid):
    from scripts.benchmark_zarr_zip_decoder import pid_exists

    with pytest.raises(ValueError, match="positive"):
        pid_exists(pid)

