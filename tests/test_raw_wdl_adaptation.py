"""Tiny real raw writer/history/derivation/adaptation and value consumer, no ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from scripts import adapt_raw_bt4_sidecars as adapter, raw_wdl_adaptation as reuse
from scripts import bt4_value_rewrite as value, bt4_policy_mix as mix
from tests.test_adapt_raw_bt4_sidecars import prepare, pinned
from tests.test_bt4_policy_mix import _write_audit_receipt


def run_adapter(path: Path, out: Path) -> None:
    assert (
        adapter.main(
            [
                "--manifest",
                str(path),
                "--expected-manifest-sha256",
                pinned(path)["sha256"],
                "--out",
                str(out),
            ]
        )
        == 0
    )


@pytest.mark.parametrize("three_shards", [False, True])
def test_real_shuffled_join_native_values_feed_hashes_and_value_consumer(
    tmp_path: Path, three_shards: bool
) -> None:
    manifest_path, derived, raw_dir, _manifest = prepare(
        tmp_path, three_shards=three_shards, wdl=True
    )
    out = tmp_path / "adapted"
    run_adapter(manifest_path, out)
    order = []
    for path in sorted(derived.glob("shard_*.zarr")):
        g: Any = zarr.open_group(str(path), mode="r")
        refs = adapter.provenance.read(
            path / adapter.provenance.FILENAME, rows=len(g["x"])
        )
        side: Any = zarr.open_group(str(out / "wdl" / path.name), mode="r")
        for i, ref in enumerate(refs):
            original: Any = zarr.open_group(
                str(raw_dir / adapter.raw.sidecar_name(ref["source_shard"])), mode="r"
            )
            np.testing.assert_array_equal(
                side["bt4_wdl_raw"][i], original["bt4_wdl_raw"][ref["source_row"]]
            )
            order.append(ref["game_id"])
        assert side["bt4_wdl_raw"].dtype == np.dtype("float32")
        np.testing.assert_array_equal(
            side["lc0_feed_sha256"][:], reuse.canonical_hashes(g["x"][:])
        )
        assert side.attrs["binding"]["profile"] == reuse.PROFILE
        assert "No raw-history replay" not in side.attrs["binding"]["history_lineage"]
        # The unchanged direct-inference contract refuses these adapted values.
        expected = value.wdl.binding(
            argparse.Namespace(
                source=str(derived),
                onnx=str(tmp_path / "synthetic.onnx"),
                expected_source_summary_sha256=pinned(derived / value.DERIVE_SUMMARY)[
                    "sha256"
                ],
                expected_onnx_sha256="a" * 64,
                wdl_output="value",
                wdl_output_kind="probabilities",
            ),
            {"path": path.name, "rows": len(g["x"])},
            value.wdl.storage_identity(path),
        )
        with pytest.raises(ValueError, match="binding differs"):
            value.wdl.verify_cached(out / "wdl" / path.name, expected, 2)
        value.wdl.verify_cached(
            out / "wdl" / path.name,
            reuse.expected_binding(
                out / "wdl" / path.name, expected, pinned(manifest_path)
            ),
            2,
        )
    assert sorted(order) == (
        [0, 1, 3, 4, 5, 6, 7, 8] if three_shards else [0, 1, 3, 4, 5, 6, 7]
    )
    assert order != sorted(order)
    if three_shards:
        return  # This fixture deliberately models an incomplete raw corpus; training admission stays closed.
    # Actual global B100 policy writer followed by explicit adapted-WDL value admission.
    b100 = tmp_path / "B100"
    audit_path = _write_audit_receipt(tmp_path, scope="global", bt4_temperature=0.5)
    audit = json.loads(audit_path.read_text())
    audit["treatment_invariants"]["mass_reference"] = "normalized_total_legal_mass"
    audit_path.write_text(json.dumps(audit))
    assert (
        mix.mix_corpus(
            argparse.Namespace(
                shards=derived,
                sidecar=out,
                out=b100,
                alpha=1.0,
                scope="global",
                bt4_temperature=0.5,
                near_max_ratio=0.5,
                expected_rows=7,
                expected_shards=3,
                expected_source_summary_sha256=pinned(derived / mix.DERIVE_SUMMARY)[
                    "sha256"
                ],
                audit_receipt=audit_path,
            )
        )
        == 0
    )
    args = value.build_parser().parse_args(
        [
            "--source",
            str(b100),
            "--sf-source",
            str(derived),
            "--wdl",
            str(out / "wdl"),
            "--out",
            str(tmp_path / "value"),
            "--expected-source-summary-sha256",
            pinned(b100 / value.DERIVE_SUMMARY)["sha256"],
            "--expected-policy-summary-sha256",
            pinned(b100 / value.POLICY_SUMMARY)["sha256"],
            "--expected-sf-summary-sha256",
            pinned(derived / value.DERIVE_SUMMARY)["sha256"],
            "--expected-onnx-sha256",
            "a" * 64,
            "--wdl-output",
            "value",
            "--minimum-free-gib",
            "0",
            "--wdl-adapter-manifest",
            str(manifest_path),
            "--expected-wdl-adapter-manifest-sha256",
            pinned(manifest_path)["sha256"],
        ]
    )
    result = value.rewrite(args)
    assert result["rows"] == 7
    assert result["wdl_adaptation"]["profile"] == reuse.PROFILE
    from chess_anti_engine.replay.shard import load_shard_arrays

    for spec in result["outputs"]:
        old, _ = load_shard_arrays(b100 / spec["path"])
        new, _ = load_shard_arrays(Path(args.out) / spec["path"])
        side = zarr.open_group(str(out / "wdl" / spec["path"]), mode="r")
        np.testing.assert_array_equal(
            new["search_wdl"], value.target(old["search_wdl"], side["bt4_wdl_raw"][:])
        )
        for name in value.ARRAYS - {"search_wdl"}:
            np.testing.assert_array_equal(new[name], old[name])


@pytest.mark.parametrize(
    ("defect", "match"),
    [
        ("missing", "coverage"),
        ("model", "onnx_sha256"),
        ("kind", "contract"),
        ("offset", "physical source row"),
        ("feed", "LC0 feed mismatch"),
    ],
)
def test_refuses_wrong_raw_value_or_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str, match: str
) -> None:
    mp, derived, raw_dir, m = prepare(tmp_path, wdl=defect != "missing")
    m["wdl"] = {"output": "value", "kind": "probabilities", "dtype": "float32"}
    if defect in ("model", "kind"):
        path = next(raw_dir.glob("*.zarr"))
        g: Any = zarr.open_group(str(path), mode="a")
        if defect == "model":
            g.attrs["onnx_sha256"] = "b" * 64
        else:
            w = dict(g.attrs["wdl"])
            w["kind"] = "logits"
            g.attrs["wdl"] = w
    if defect == "offset":
        shard = next(derived.glob("*.zarr"))
        rp = shard / adapter.provenance.FILENAME
        with np.load(rp, allow_pickle=False) as a:
            sources, records = a["sources"], a["records"]
        records["row"][0] = 999
        with rp.open("wb") as f:
            np.savez(f, sources=sources, records=records)
        g = zarr.open_group(str(shard), mode="a")
        stamp = dict(g.attrs["derive_row_provenance"])
        stamp["sha256"] = pinned(rp)["sha256"]
        g.attrs["derive_row_provenance"] = stamp
        summary_path = derived / mix.DERIVE_SUMMARY
        summary = json.loads(summary_path.read_text())
        for s in summary["shards"]:
            if s["path"] == shard.name:
                s["row_provenance"] = stamp
        summary_path.write_text(json.dumps(summary))
        m["derived_summary"] = pinned(summary_path)
    if defect == "feed":
        verify = adapter.raw.verify_shard

        def altered(*args, **kwargs):
            attrs = verify(*args, **kwargs)
            kwargs["canonical_feed_records"][0, 0] ^= 1
            return attrs

        monkeypatch.setattr(adapter.raw, "verify_shard", altered)
    mp.write_text(json.dumps(m))
    with pytest.raises(ValueError, match=match):
        run_adapter(mp, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_explicit_value_mode_does_not_change_policy_only_output(tmp_path: Path) -> None:
    mp, _derived, _raw, m = prepare(tmp_path, wdl=True)
    run_adapter(mp, tmp_path / "with_value")
    m.pop("wdl")
    mp.write_text(json.dumps(m))
    run_adapter(mp, tmp_path / "policy_only")
    assert not (tmp_path / "policy_only" / "wdl").exists()
    for path in (tmp_path / "policy_only").glob("shard_*.zarr"):
        a: Any = zarr.open_group(str(path), mode="r")
        b: Any = zarr.open_group(str(tmp_path / "with_value" / path.name), mode="r")
        for name in a.array_keys():
            np.testing.assert_array_equal(a[name][:], b[name][:])


def test_native_contract_rejects_logits_and_null() -> None:
    for w in (None, {"output": "value", "kind": "logits", "dtype": "float32"}):
        with pytest.raises(ValueError, match="native WDL"):
            reuse.contract({"wdl": w, "teacher": {"policy_output": "policy"}})


@pytest.mark.parametrize("field", ["input", "providers", "dtype"])
def test_consumer_requires_native_contract_and_canonical_provenance(
    tmp_path: Path, field: str
) -> None:
    mp, _derived, _raw, _manifest = prepare(tmp_path, wdl=True)
    out = tmp_path / "adapted"
    run_adapter(mp, out)
    path = sorted((out / "wdl").glob("*.zarr"))[0]
    g: Any = zarr.open_group(str(path), mode="a")
    expected = dict(g.attrs["binding"])
    if field == "input":
        metadata = dict(g.attrs["input"])
        metadata["kind"] = "historical_actual_feed"
        g.attrs["input"] = metadata
    elif field == "providers":
        g.attrs["providers"] = ["invented_provider"]
    else:
        metadata = dict(g.attrs["wdl"])
        metadata["dtype"] = "float16"
        g.attrs["wdl"] = metadata
    with pytest.raises(ValueError, match=r"provenance differs|contract differs"):
        reuse.expected_binding(path, expected, pinned(mp))
