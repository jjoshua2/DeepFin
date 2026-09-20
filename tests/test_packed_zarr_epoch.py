from __future__ import annotations

import json
from pathlib import Path
import stat
import struct
import warnings
import zipfile

import numpy as np
import pytest
import zarr
from zarr.storage import ZipStore

from chess_anti_engine.replay import packed_zarr as packed
from chess_anti_engine.replay.game_epoch import GameAwareEpochBuffer
from chess_anti_engine.replay.shard import load_shard_arrays, open_shard_arrays
from tests.test_game_aware_epoch_replay import _write


def pack(source, target):
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source).as_posix())
    return target


@pytest.fixture
def corpus(tmp_path):
    source = _write(
        tmp_path / "directory",
        [[(i % 4, i) for i in range(8)], [(i % 4, i) for i in range(8, 16)]],
    )
    target = tmp_path / "packed"
    target.mkdir()
    for path in source.glob("*.zarr"):
        pack(path, target / (path.name + ".zip"))
    return source, target


def buffer(path, **kwargs):
    return GameAwareEpochBuffer(
        shard_dir=path,
        batch_size=2,
        seed=121,
        input_planes=146,
        input_history_encoding="legacy",
        history_rep_fix=False,
        mirror_augmentation=True,
        load_workers=1,
        plan_workers=2,
        **kwargs,
    )


def test_opt_in_same_ordered_batches_and_source_partitions(corpus, tmp_path):
    source, target = corpus
    # A second independent source reuses game ids. Stage through separate links
    # to preserve the same corpus partition in both representations.
    extra = _write(tmp_path / "other", [[(i, i + 30) for i in range(4)]])
    zipped_extra = tmp_path / "other-packed"
    zipped_extra.mkdir()
    pack(extra / "shard_000000.zarr", zipped_extra / "shard_000000.zarr.zip")
    for name, base, other, ext in [
        ("stage-dir", source, extra, ".zarr"),
        ("stage-zip", target, zipped_extra, ".zarr.zip"),
    ]:
        stage = tmp_path / name
        stage.mkdir()
        for i, shard in enumerate([*sorted(base.iterdir()), *sorted(other.iterdir())]):
            (stage / f"shard_{i:06d}{ext}").symlink_to(shard)
    with pytest.raises(ValueError, match=r"shard|empty|corpus"):
        buffer(target)
    a = buffer(tmp_path / "stage-dir")
    b = buffer(tmp_path / "stage-zip", allow_packed_zarr=True)
    try:
        assert a.plan.rows == b.plan.rows == 20
        assert a.plan.source_count == b.plan.source_count == 2
        for _ in range(a.num_batches):
            x, y = a.sample_batch_arrays(2), b.sample_batch_arrays(2)
            assert x.keys() == y.keys()
            for key in x:
                np.testing.assert_array_equal(x[key], y[key], err_msg=key)
            # Trainer mirror RNG is a separate unchanged stream.
            np.testing.assert_array_equal(a.rng.random(2), b.rng.random(2))
    finally:
        a.close()
        b.close()


def test_duplicate_discovery_indices_refused(corpus):
    source, target = corpus
    (target / "shard_000000.zarr").symlink_to(source / "shard_000000.zarr")
    with pytest.raises(ValueError, match="duplicate shard index"):
        buffer(target, allow_packed_zarr=True)


def test_lazy_planning_never_decodes_wide_arrays_before_memory_qualification(
    corpus, monkeypatch
):
    _, target = corpus
    original = zarr.Array.__getitem__

    def read(self, selection):
        if self.path in ("x", "policy_target"):
            pytest.fail("wide array decoded before working-set qualification")
        return original(self, selection)

    monkeypatch.setattr(zarr.Array, "__getitem__", read)
    with pytest.raises(ValueError, match=r"working.set|bytes|limit"):
        buffer(target, allow_packed_zarr=True, max_working_set_bytes=1)


def test_objective_census_and_lazy_errors_close_store(corpus, monkeypatch):
    _, target = corpus
    opened = []
    original = ZipStore.__init__

    def init(self, *a, **k):
        original(self, *a, **k)
        opened.append(self)

    monkeypatch.setattr(ZipStore, "__init__", init)

    def census(arrays):
        assert np.asarray(arrays["has_game_id"]).all()
        raise RuntimeError("census failed")

    with pytest.raises(RuntimeError, match="census failed"):
        buffer(target, allow_packed_zarr=True, objective_mask_counter=census)
    assert opened
    assert all(store.zf.fp is None for store in opened)
    path = next(target.glob("*.zip"))
    with pytest.raises(ValueError, match="context"):
        load_shard_arrays(path, lazy=True)
    def consume():
        with open_shard_arrays(path, lazy=True) as (arrays, _):
            assert arrays["x"].shape[0] == 8
            raise RuntimeError("consumer failure")

    with pytest.raises(RuntimeError, match="consumer failure"):
        consume()
    assert all(store.zf.fp is None for store in opened)


@pytest.mark.parametrize(
    "name",
    [
        "../x",
        "/x",
        "x\\0",
        "x/../0",
        "x//0",
        ".zattrs",
        "target_overlay.json",
        "base-binding.json",
    ],
)
def test_archive_members_refused(corpus, name):
    _, target = corpus
    path = next(target.glob("*.zip"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr(name, "{}")
    with pytest.raises(ValueError, match="member"):
        packed.content_sha256(path)


@pytest.mark.parametrize("kind", ["symlink", "compressed"])
def test_special_or_recompressed_members_refused(corpus, kind):
    _, target = corpus
    path = next(target.glob("*.zip"))
    entry = zipfile.ZipInfo("extra/0")
    if kind == "symlink":
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    else:
        entry.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr(entry, "x")
    with pytest.raises(ValueError, match="member"):
        packed.content_sha256(path)


def test_hash_refuses_concurrent_rewrite(corpus, monkeypatch):
    _, target = corpus
    path = next(target.glob("*.zip"))
    original = packed._validate_members

    def mutate(archive):
        original(archive)
        with path.open("ab") as f:
            f.write(b"changed")

    monkeypatch.setattr(packed, "_validate_members", mutate)
    with pytest.raises(RuntimeError, match="changed while hashing"):
        packed.content_sha256(path)


def test_change_after_plan_refused(corpus):
    _, target = corpus
    b = buffer(target, allow_packed_zarr=True)
    try:
        for path in target.glob("*.zip"):
            with path.open("ab") as f:
                f.write(b"changed")
        with pytest.raises(RuntimeError, match="changed after exact-epoch preflight"):
            b.sample_batch_arrays(2)
    finally:
        b.close()


def test_corrupt_crc_during_decode_closes_actual_fd(corpus):
    _, target = corpus
    path = next(target.glob("*.zip"))
    with zipfile.ZipFile(path) as archive:
        entry = archive.getinfo("x/0.0.0.0")
    with path.open("r+b") as f:
        f.seek(entry.header_offset + 26)
        name_len, extra_len = struct.unpack("<HH", f.read(4))
        f.seek(entry.header_offset + 30 + name_len + extra_len)
        value = f.read(1)
        f.seek(-1, 1)
        f.write(bytes([value[0] ^ 1]))
    before = len(list(Path("/proc/self/fd").iterdir()))
    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        load_shard_arrays(path)
    assert len(list(Path("/proc/self/fd").iterdir())) == before


def test_truncated_archive_rejected_without_fd_leak(corpus):
    _, target = corpus
    path = next(target.glob("*.zip"))
    with path.open("r+b") as f:
        f.truncate(path.stat().st_size - 40)
    before = len(list(Path("/proc/self/fd").iterdir()))
    with pytest.raises(zipfile.BadZipFile):
        load_shard_arrays(path)
    assert len(list(Path("/proc/self/fd").iterdir())) == before


def test_unknown_codec_refused_before_chunks(corpus, tmp_path):
    source, _ = corpus
    directory = source / "shard_000000.zarr"
    meta = directory / "x/.zarray"
    attrs = json.loads(meta.read_text())
    attrs["compressor"] = {"id": "unregistered-codec"}
    meta.write_text(json.dumps(attrs))
    path = pack(directory, tmp_path / "shard_000000.zarr.zip")
    before = len(list(Path("/proc/self/fd").iterdir()))
    with pytest.raises((ValueError, RuntimeError), match=r"codec|compressor"):
        load_shard_arrays(path)
    assert len(list(Path("/proc/self/fd").iterdir())) == before


def test_qualification_uses_real_exact_sampler(corpus):
    from scripts.qualify_packed_zarr_epoch import qualify

    source, target = corpus
    result = qualify(
        source,
        target,
        {
            "batch_size": 2,
            "seed": 121,
            "input_planes": 146,
            "input_history_encoding": "legacy",
            "history_rep_fix": False,
            "mirror_augmentation": True,
            "plan_workers": 1,
            "load_workers": 1,
            "max_working_set_bytes": 16 * 2**20,
        },
    )
    assert result["status"] == "PASS_MATCHED_PACKED_ZARR_SAMPLER"
    assert (
        result["runs"][0]["plan"]["corpus_sha256"]
        != result["runs"][1]["plan"]["corpus_sha256"]
    )
    assert result["runs"][0]["sequence_sha256"] == result["runs"][1]["sequence_sha256"]
    assert result["runs"][1]["rows"] == 16


def test_qualification_refuses_mismatched_rosters(corpus):
    from scripts.qualify_packed_zarr_epoch import qualify

    source, target = corpus
    next(target.glob("*.zip")).unlink()
    with pytest.raises(ValueError, match="rosters"):
        qualify(source, target, {})
