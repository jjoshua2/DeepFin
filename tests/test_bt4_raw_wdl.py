"""Actual tiny raw-shard writes/replay, with a deterministic named-output session."""

import argparse
from dataclasses import replace
import gzip
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import zarr

from scripts import bt4_raw_corpus_sidecar as tool
from tests.test_bt4_raw_corpus_sidecar import make_source

REMAP = {"commit": "fixture", "dirty": False, "blobs": {"x": "y"}}


class Session:
    def __init__(self, dtype="float16", kind="logits", bad=None):
        self.dtype, self.kind, self.bad = dtype, kind, bad
        self.requests = []
        self.saved = []
        self.cursor = 0

    def get_outputs(self):
        native = {"float16": "float16", "float32": "float", "float64": "double"}[
            self.dtype
        ]
        name = "absent" if self.bad == "missing_head" else "value_head"
        shape = [None, 4] if self.bad == "head_shape" else [None, 3]
        return [
            SimpleNamespace(name="policy", shape=[None, 1858], type="tensor(float)"),
            SimpleNamespace(name=name, shape=shape, type=f"tensor({native})"),
        ]

    def run(self, names, feed):
        self.requests.append(list(names))
        n = next(iter(feed.values())).shape[0]
        out: list[np.ndarray] = [np.zeros((n, 1858), dtype=np.float32)]
        if len(names) == 2:
            assert names == ["policy", "value_head"]
            if self.kind == "probabilities":
                values = np.tile([0.125, 0.25, 0.625], (n, 1)).astype(self.dtype)
            else:
                values = (np.arange(n * 3).reshape(n, 3) + self.cursor * 3 - 4).astype(
                    self.dtype
                )
            if self.bad == "nonfinite":
                values[0, 0] = np.nan
            elif self.bad == "mass":
                values[:] = 0.5
            elif self.bad == "negative":
                values[0] = [-0.25, 0.25, 1]
            elif self.bad == "dtype":
                values = values.astype("float64")
            elif self.bad == "rows":
                values = values[:-1]
            elif self.bad == "width":
                values = values[:, :2]
            self.saved.append(values.copy())
            out.append(values)
        self.cursor += n
        return out


def source3(tmp_path):
    source = make_source(tmp_path)
    path = source.inventory.shards[0]
    with gzip.open(path, "rt") as handle:
        original = json.loads(handle.readline())
    with gzip.open(path, "wt") as handle:
        for i in range(3):
            handle.write(
                json.dumps({**original, "game_id": 17 + i, "ply": 3 + i}) + "\n"
            )
    source = replace(
        source, inventory=replace(source.inventory, shard_rows=(3,), rows_claimed=3)
    )
    return source, tool.PendingShard(
        source, path, 3, source.out_dir / tool.sidecar_name(path.name)
    )


def produce(pending, session, contract):
    return tool.label_shard(
        pending,
        sess=session,
        input_name="input",
        input_dtype=np.dtype("float32"),
        providers=["fixture"],
        policy_name="policy",
        onnx_path=pending.path.parent / "fake.onnx",
        onnx_sha256="teacher",
        remap_stamp=REMAP,
        batch_size=2,
        wdl_output=contract,
    )


def verify(pending, contract=None):
    return tool.verify_shard(
        pending,
        onnx_sha256="teacher",
        expected_policy_output="policy",
        expected_providers=["fixture"],
        expected_remap=REMAP,
        batch_size=2,
        expected_wdl=contract,
    )


@pytest.mark.parametrize(
    ("dtype", "kind"),
    [
        ("float16", "logits"),
        ("float32", "logits"),
        ("float64", "logits"),
        ("float16", "probabilities"),
    ],
)
def test_named_raw_value_reaches_same_rows_and_keeps_policy_bytes(
    tmp_path, dtype, kind
):
    base = tmp_path / "base"
    base.mkdir()
    value = tmp_path / "value"
    value.mkdir()
    _, old = source3(base)
    source, pending = source3(value)
    ordinary = Session()
    old_attrs = produce(old, ordinary, None)
    session = Session(dtype, kind)
    requested = {"output": "value_head", "kind": kind}
    contract = tool.resolve_wdl_output(session, requested, "policy")
    attrs = produce(pending, session, contract)
    assert ordinary.requests == [["policy"], ["policy"]]
    assert session.requests == [["policy", "value_head"], ["policy", "value_head"]]
    original: Any = zarr.open_group(str(old.target), mode="r")
    group: Any = zarr.open_group(str(pending.target), mode="r")
    assert set(original.array_keys()) == {
        tool.POLICY_FIELD,
        tool.SOURCE_KEY_FIELD,
        tool.INPUT_KEY_FIELD,
        tool.GAME_ID_FIELD,
        tool.PLY_FIELD,
    }
    assert "wdl" not in old_attrs
    assert tool.WDL_FIELD not in original
    for name in original.array_keys():
        np.testing.assert_array_equal(group[name][:], original[name][:])
        # The old compressed array payload/layout bytes also remain identical.
        for path in (old.target / name).iterdir():
            if path.is_file():
                assert (
                    path.read_bytes()
                    == (pending.target / name / path.name).read_bytes()
                )
    raw = np.asarray(group[tool.WDL_FIELD][:])
    assert raw.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(raw, np.concatenate(session.saved))
    assert attrs["wdl"]["sha256"] == tool.sha_array(raw)
    assert attrs["wdl"]["rows"] == 3
    assert attrs["teacher_evaluations_per_position"] == 1
    assert verify(pending, requested)["wdl"] == attrs["wdl"]
    tool.append_receipt(
        source.out_dir / tool.PROGRESS_NAME,
        tool.receipt_from_attrs(attrs, pending.target),
    )
    assert (
        tool.pending_shards([source], onnx_sha256="teacher", expected_wdl=requested)[0]
        == []
    )


@pytest.mark.parametrize(
    "bad", ["nonfinite", "mass", "negative", "dtype", "rows", "width"]
)
def test_bad_actual_value_never_publishes_completed_sidecar(tmp_path, bad):
    _, pending = source3(tmp_path)
    session = Session("float16", "probabilities", bad)
    contract = tool.resolve_wdl_output(
        session, {"output": "value_head", "kind": "probabilities"}, "policy"
    )
    with pytest.raises(ValueError, match="WDL"):
        produce(pending, session, contract)
    assert not pending.target.exists()


@pytest.mark.parametrize("bad", ["missing_head", "head_shape"])
def test_requested_head_metadata_refuses_before_teacher_call(bad):
    session = Session(bad=bad)
    with pytest.raises(ValueError, match="WDL"):
        tool.resolve_wdl_output(
            session, {"output": "value_head", "kind": "logits"}, "policy"
        )
    assert not session.requests


@pytest.mark.parametrize(
    "mutation", ["payload", "kind", "pov", "receipt", "missing_array"]
)
def test_optional_value_replay_and_lineage_refuse_tampering(tmp_path, mutation):
    source, pending = source3(tmp_path)
    session = Session()
    contract = tool.resolve_wdl_output(
        session, {"output": "value_head", "kind": "logits"}, "policy"
    )
    attrs = produce(pending, session, contract)
    receipt = tool.receipt_from_attrs(attrs, pending.target)
    group: Any = zarr.open_group(str(pending.target), mode="a")
    if mutation == "payload":
        group[tool.WDL_FIELD][0, 0] += 1
    elif mutation in ("kind", "pov"):
        wdl = dict(group.attrs["wdl"])
        wdl[mutation] = "probabilities" if mutation == "kind" else "white"
        group.attrs["wdl"] = wdl
    elif mutation == "missing_array":
        del group[tool.WDL_FIELD]
    else:
        receipt["wdl"] = {**receipt["wdl"], "output": "other_head"}
    tool.append_receipt(source.out_dir / tool.PROGRESS_NAME, receipt)
    if mutation == "receipt":
        with pytest.raises(ValueError, match="WDL"):
            tool.pending_shards([source], onnx_sha256="teacher")
    else:
        with pytest.raises(ValueError, match="WDL"):
            verify(pending, {"output": "value_head", "kind": "logits"})


def test_existing_policy_only_remains_complete_but_cannot_claim_value(
    tmp_path, monkeypatch
):
    source, pending = source3(tmp_path)
    attrs = produce(pending, Session(), None)
    tool.append_receipt(
        source.out_dir / tool.PROGRESS_NAME,
        tool.receipt_from_attrs(attrs, pending.target),
    )
    wanted = {"output": "value_head", "kind": "logits"}
    assert tool.pending_shards(
        [source], onnx_sha256="teacher", expected_wdl=wanted
    ) == ([], {source.source_id: 1})
    assert "wdl" not in verify(pending)
    with pytest.raises(ValueError, match="required WDL coverage"):
        verify(pending, wanted)
    monkeypatch.setattr(tool, "load_sources", lambda *_: [source])
    monkeypatch.setattr(tool, "remap_provenance", lambda: REMAP)
    args = argparse.Namespace(
        source=[],
        batch_size=2,
        gpu_mem_gb=0,
        wdl_output="value_head",
        wdl_output_kind="logits",
    )
    with pytest.raises(ValueError, match="required WDL coverage"):
        tool.verify_all(args, out_root=source.out_dir.parent, onnx_sha="teacher")
    assert (
        json.loads((source.out_dir.parent / tool.VERIFY_NAME).read_text())["verdict"]
        == "FAIL"
    )


def test_group_cli_contract_reaches_actual_output_and_coverage(
    tmp_path, monkeypatch, capsys
):
    source, old_pending = source3(tmp_path)
    old_attrs = produce(old_pending, Session(), None)
    tool.append_receipt(
        source.out_dir / tool.PROGRESS_NAME,
        tool.receipt_from_attrs(old_attrs, old_pending.target),
    )
    original_bytes = {
        str(p.relative_to(old_pending.target)): p.read_bytes()
        for p in old_pending.target.rglob("*")
        if p.is_file()
    }
    second = source.corpus_dir / "w00-00001.jsonl.gz"
    second.write_bytes(old_pending.path.read_bytes())
    source = replace(
        source,
        inventory=replace(
            source.inventory,
            shards=(old_pending.path, second),
            shard_rows=(3, 3),
            rows_claimed=6,
        ),
    )
    pending = tool.PendingShard(
        source, second, 3, source.out_dir / tool.sidecar_name(second.name)
    )
    session = Session()
    monkeypatch.setattr(tool, "load_sources", lambda *_: [source])
    monkeypatch.setattr(
        tool,
        "open_session",
        lambda *a, **k: (session, "input", np.dtype("float32"), ["fixture"]),
    )
    monkeypatch.setattr(tool, "file_sha256", lambda _: "teacher")
    monkeypatch.setattr(tool, "remap_provenance", lambda: REMAP)
    args = tool.build_parser().parse_args(
        [
            "--source",
            "x=" + str(source.corpus_dir),
            "--out-root",
            str(source.out_dir.parent),
            "--gpu-lock",
            str(tmp_path / "lock"),
            "--gpu-mem-gb",
            "0",
            "--min-free-gib",
            "0",
            "--batch-size",
            "2",
            "--wdl-output",
            "value_head",
            "--wdl-output-kind",
            "logits",
        ]
    )
    assert tool.run_label_group(args) == 0
    assert session.requests == [["policy", "value_head"], ["policy", "value_head"]]
    status = json.loads((source.out_dir.parent / tool.STATUS_NAME).read_text())
    assert status["sources"][source.source_id]["wdl_coverage"] == {
        "complete_shards": 1,
        "complete_rows": 3,
        "policy_only_shards": 1,
        "policy_only_rows": 3,
    }
    assert original_bytes == {
        str(p.relative_to(old_pending.target)): p.read_bytes()
        for p in old_pending.target.rglob("*")
        if p.is_file()
    }
    monkeypatch.setattr(
        tool,
        "run_label_child_under_gpu_lease",
        lambda _: pytest.fail("caught-up future mode must not rerun teacher"),
    )
    capsys.readouterr()
    assert tool.run(args) == 0
    printed = capsys.readouterr().out
    assert '"backfill": false' in printed
    assert '"policy_only_rows": 3' in printed
    assert (
        tool.read_receipts(source.out_dir / tool.PROGRESS_NAME)[pending.path.name][
            "wdl"
        ]["output"]
        == "value_head"
    )


@pytest.mark.parametrize(
    ("name", "kind"), [("value_head", None), (None, "logits"), ("", "logits")]
)
def test_value_options_are_paired_and_explicit(name, kind):
    with pytest.raises(ValueError, match=r"WDL|wdl"):
        tool.requested_wdl(argparse.Namespace(wdl_output=name, wdl_output_kind=kind))


@pytest.mark.parametrize("where", ["attrs", "receipt"])
def test_null_value_metadata_cannot_forge_coverage(tmp_path, where):
    source, pending = source3(tmp_path)
    attrs = produce(pending, Session(), None)
    receipt = tool.receipt_from_attrs(attrs, pending.target)
    if where == "attrs":
        group: Any = zarr.open_group(str(pending.target), mode="a")
        group.attrs["wdl"] = None
    else:
        receipt["wdl"] = None
    tool.append_receipt(source.out_dir / tool.PROGRESS_NAME, receipt)
    with pytest.raises(ValueError, match="WDL"):
        tool.pending_shards([source], onnx_sha256="teacher")
