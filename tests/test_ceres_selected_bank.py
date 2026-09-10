"""Selected-bank joins and shared fixed32 collection, with no real teacher."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import zarr

from scripts import ceres_derived_sidecar as tool, ceres_selected_bank as selected
from tests.test_ceres_derived_sidecar import Session, setup
from tests.test_sf_policy_rewrite import raw_row
from tests.test_derive_corpus_targets import FEN_W
from scripts import derive_corpus_targets as derive


def sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    args = setup(tmp_path, monkeypatch, 66)
    original = Path(args.source)
    source: Any = zarr.open_group(str(original / "shard_000000.zarr"), mode="r")
    bank = tmp_path / "qualified"
    bank.mkdir()
    choices, raw = [], []
    # Uneven fragments exercise local tail padding while retaining global selection order.
    orders = [np.r_[32, np.arange(32)], np.arange(33, 65)[::-1]]
    for part, order in enumerate(orders):
        name = f"shard_{part:06d}.zarr"
        group: dict[str, Any] = {
            k: np.asarray(source[k][:])[order] for k in tool.COLUMNS
        }
        for i, row in enumerate(order):
            item = {
                "source_dir": str(tmp_path / "raw_source"),
                "derived_shard": name,
                "derived_row": int(row),
                "game_id": int(group["game_id"][i]),
                "ply": int(group["ply_index"][i]),
                "stratum": part,
                "weight": float(40 + part),
            }
            choices.append(item)
            # Real re-encoding of distinct clocks keeps raw-history/input keys coherent.
            root_fields = cast(list[str], FEN_W.split())
            root_fields[-2] = str(int(row))
            record = raw_row(game_id=item["game_id"], fen=" ".join(root_fields))
            group["x"][i] = np.asarray(
                derive.encode_position(
                    derive.board_from_row(record),
                    add_features=True,
                    input_history_encoding=tool.shared.HISTORY,
                    input_extra_features="v2_threats",
                ),
                dtype=np.float16,
            )
            record["ply"] = item["ply"]
            raw.append(
                {
                    **item,
                    "raw_shard": f"w00-{part:05d}.jsonl.zst",
                    "physical_row": int(row),
                    "worker_id": record["worker_id"],
                    "input_key": record["input_key"],
                    "raw": record,
                }
            )
        np.savez_compressed(bank / (name + ".npz"), **group, selected_rows=order)
    (bank / "selection.json").write_text(json.dumps({"selected": choices}))
    (bank / "raw_rows.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in reversed(raw))
    )
    repin(bank, monkeypatch)
    args.source = str(bank)
    args.expected_source_summary_sha256 = None
    args.selected_bank_qualification = str(bank / "complete.json")
    args.expected_selected_bank_qualification_sha256 = selected.QUALIFICATION_SHA
    args.max_shards = 2
    args.pad_final_batch = args.retain_value2 = True
    monkeypatch.setattr(selected, "ROWS", 65)
    monkeypatch.setattr(selected, "FRAGMENTS", 2)
    return args, choices


def repin(bank: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    complete = {
        "status": "COMPLETE_QUALIFIED_TRAINING_SAMPLE",
        "rows": 65,
        "derived_shards": 2,
        "outputs": {
            f.name: {"bytes": f.stat().st_size, "sha256": tool.shared.file_sha256(f)}
            for f in bank.iterdir()
            if f.name != "complete.json"
        },
    }
    (bank / "complete.json").write_text(json.dumps(complete))
    monkeypatch.setattr(
        selected, "QUALIFICATION_SHA", tool.shared.file_sha256(bank / "complete.json")
    )


class OrderedSession(Session):
    def run(self, names, feed):
        outputs = super().run(names, feed)
        marker = feed["squares_byte"][:, 0, tool.tpg.MOVE50_COUNT].astype("float32")
        outputs[0] = (np.arange(1858)[None, :] / 2048 + marker[:, None]).astype(
            "float16"
        )
        self.policies[-1] = outputs[0].copy()
        return outputs


def test_selected_order_two_fragments_single_session_and_cache(tmp_path, monkeypatch):
    args, choices = sample(tmp_path, monkeypatch)
    session = OrderedSession(tmp_path)
    opens = []
    monkeypatch.setattr(tool, "open_teacher", lambda a: opens.append(a) or session)
    tool.produce(args)
    assert len(opens) == 1
    assert session.requests == [["policy", "value", "value2"]] * 3
    # The first fragment's final row is repeated, not the next fragment's first row.
    np.testing.assert_array_equal(
        session.feeds[1], np.repeat(session.feeds[1][:1], 32, axis=0)
    )
    for part, (start, end) in enumerate([(0, 33), (33, 65)]):
        output: Any = zarr.open_group(
            str(Path(args.out) / f"selection_{part:06d}.zarr"), mode="r"
        )
        rows = choices[start:end]
        np.testing.assert_array_equal(
            output["row_index"][:], [r["derived_row"] for r in rows]
        )
        np.testing.assert_array_equal(
            output["selection_index"][:], np.arange(start, end)
        )
        np.testing.assert_array_equal(
            output["game_id"][:], [r["game_id"] for r in rows]
        )
        np.testing.assert_array_equal(
            output["value2_logits"][:], output["value_logits"][:] + 0.5
        )
        bound = output.attrs["binding"]
        assert bound["profile"] == selected.PROFILE
        assert bound["summary_sha256"] is None
        for actual, expected in zip(
            bound["selected_rows"]["identities"], rows, strict=True
        ):
            assert all(actual[k] == expected[k] for k in selected.KEYS)
            assert (
                len(actual["history_sha256"]) == len(actual["raw_record_sha256"]) == 64
            )
        with np.load(Path(args.source) / f"shard_{part:06d}.zarr.npz") as source:
            np.testing.assert_array_equal(
                output["x_sha256"][:], tool.shared.row_digests(source["x"])
            )
            feed = tool.tpg.stored_x_to_ceres_tpg_bytes(
                source["x"],
                input_history_encoding=tool.shared.HISTORY,
                history_rep_fix=True,
            )
        np.testing.assert_array_equal(
            output["tpg_feed_sha256"][:], tool.shared.row_digests(feed)
        )
        assert len({bytes(row) for row in tool.shared.row_digests(feed)}) == len(rows)
        first_call = 0 if part == 0 else 2
        for local_start in range(0, len(rows), 32):
            real_feed = feed[local_start : local_start + 32]
            np.testing.assert_array_equal(
                session.feeds[first_call + local_start // 32][: len(real_feed)],
                real_feed,
            )
        gather = tool.mapping.leela_gather_indices(
            *tool.tpg.ceres_tpg_gather_context(feed)
        )
        offsets = output["legal_offsets"][:]
        for row in range(len(rows)):
            lo, hi = int(offsets[row]), int(offsets[row + 1])
            indices = output["legal_indices"][lo:hi]
            expected = session.policies[first_call + row // 32][
                row % 32, gather[row, indices]
            ]
            np.testing.assert_array_equal(output["policy_logits"][lo:hi], expected)

    receipt = json.loads((Path(args.invocation) / "child_completed.json").read_text())
    assert "shards" not in receipt
    assert receipt["fragments"] == 2
    assert receipt["rows"] == 65
    assert receipt["collection_counts"] == {
        "real_rows": 65,
        "padding_rows": 31,
        "calls": 3,
        "input_rows": 96,
    }
    monkeypatch.setattr(
        tool, "open_teacher", lambda _a: pytest.fail("cache reached teacher")
    )
    tool.produce(args)
    # A forged array digest cannot turn whole-shard row numbering into selected numbering.
    output = zarr.open_group(str(Path(args.out) / "selection_000000.zarr"), mode="a")
    output["row_index"][:] = np.arange(33)
    digests = dict(output.attrs["array_sha256"])
    digests["row_index"] = tool.shared.raw.sha_array(output["row_index"][:])
    output.attrs["array_sha256"] = digests
    with pytest.raises(ValueError, match="row order differs"):
        tool.produce(args)


@pytest.mark.parametrize(
    "defect",
    [
        "missing_pin",
        "unqualified",
        "partial_range",
        "summary",
        "no_value2",
        "no_padding",
    ],
)
def test_selected_contract_refuses_before_session(tmp_path, monkeypatch, defect):
    args, _ = sample(tmp_path, monkeypatch)
    if defect == "missing_pin":
        args.expected_selected_bank_qualification_sha256 = None
    elif defect == "unqualified":
        args.expected_selected_bank_qualification_sha256 = "0" * 64
    elif defect == "partial_range":
        args.max_shards = 1
    elif defect == "summary":
        args.expected_source_summary_sha256 = "0" * 64
    elif defect == "no_value2":
        args.retain_value2 = False
    else:
        args.pad_final_batch = False
    monkeypatch.setattr(
        tool,
        "open_teacher",
        lambda _a: pytest.fail("invalid selection reached teacher"),
    )
    with pytest.raises(
        ValueError, match=r"selected|qualification|fragment|raw history"
    ):
        tool.produce(args)
    assert not (Path(args.invocation) / "child_completed.json").exists()


@pytest.mark.parametrize(
    "defect", ["selection", "raw_join", "npz_order", "npz_game", "missing_fragment"]
)
def test_selected_metadata_and_fragment_refusals(tmp_path, monkeypatch, defect):
    args, _ = sample(tmp_path, monkeypatch)
    bank = Path(args.source)
    if defect == "selection":
        path = bank / "selection.json"
        data = json.loads(path.read_text())
        data["selected"][0] = data["selected"][1]
        path.write_text(json.dumps(data))
    elif defect == "raw_join":
        path = bank / "raw_rows.jsonl"
        lines = path.read_text().splitlines()
        data = json.loads(lines[0])
        data["input_key"] = "wrong"
        lines[0] = json.dumps(data)
        path.write_text("\n".join(lines) + "\n")
    elif defect == "missing_fragment":
        (bank / "shard_000000.zarr.npz").unlink()
    else:
        path = bank / "shard_000000.zarr.npz"
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
        key = "selected_rows" if defect == "npz_order" else "game_id"
        arrays[key] = arrays[key][::-1]
        np.savez_compressed(path, **arrays)
    repin(bank, monkeypatch)
    args.expected_selected_bank_qualification_sha256 = selected.QUALIFICATION_SHA
    session = Session(tmp_path)
    monkeypatch.setattr(tool, "open_teacher", lambda _a: session)
    with pytest.raises(
        ValueError, match=r"selected|qualification|fragment|raw history"
    ):
        tool.produce(args)
    assert not session.requests
    assert not (Path(args.out) / "selection_000000.zarr").exists()


def test_source_mutation_and_namespace_mixing_refuse(tmp_path, monkeypatch):
    args, _ = sample(tmp_path, monkeypatch)
    session = Session(tmp_path)

    def mutate_after_creation(_args):
        path = Path(args.source) / "selection.json"
        path.write_text(path.read_text() + " ")
        return session

    monkeypatch.setattr(tool, "open_teacher", mutate_after_creation)
    with pytest.raises(ValueError, match="selected bank changed"):
        tool.produce(args)
    assert not session.requests
    # The existing whole-shard path must reject this selected namespace as well.
    (tmp_path / "other").mkdir()
    original_args = setup(tmp_path / "other", monkeypatch)
    original_args.out = args.out
    monkeypatch.setattr(
        tool, "open_teacher", lambda _a: pytest.fail("mixed namespace reached teacher")
    )
    with pytest.raises(ValueError, match="namespace differs"):
        tool.produce(original_args)


def test_selected_cli_reaches_ceres_child_with_explicit_contract(tmp_path, monkeypatch):
    args, _ = sample(tmp_path, monkeypatch)
    observed = []
    monkeypatch.setattr(
        tool.shared,
        "run",
        lambda a, *, child_target: observed.append((a, child_target)) or 0,
    )
    argv = [
        "--source",
        args.source,
        "--out",
        args.out,
        "--onnx",
        args.onnx,
        "--expected-onnx-sha256",
        args.expected_onnx_sha256,
        "--wdl-output",
        "value",
        "--wdl-output-kind",
        "logits",
        "--max-shards",
        "2",
        "--gpu-lock",
        args.gpu_lock,
        "--pad-final-batch",
        "--retain-value2",
        "--selected-bank-qualification",
        args.selected_bank_qualification,
        "--expected-selected-bank-qualification-sha256",
        args.expected_selected_bank_qualification_sha256,
    ]
    assert tool.main(argv) == 0
    actual, target = observed[0]
    assert target is tool.child
    assert tool.profile(actual) == selected.PROFILE
    assert actual.expected_source_summary_sha256 is None
    with pytest.raises(ValueError, match="source summary"):
        tool.main(argv[:-4])


def test_selected_cache_profile_cannot_claim_primary_only_backend(
    tmp_path, monkeypatch
):
    args, _ = sample(tmp_path, monkeypatch)
    bound = tool.namespace(args)
    bound["backend"] = tool.BACKEND
    with pytest.raises(ValueError, match="selected Ceres backend"):
        tool.binding_options(bound)
