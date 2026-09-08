"""Actual shuffled derivation, WDL bank, rewrite, replay loader and loss; CPU only."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pytest
import torch
import zarr

from chess_anti_engine.replay.shard import load_shard_arrays
from chess_anti_engine.replay.dataset import collate_arrays
from chess_anti_engine.train.losses import compute_loss
from scripts import bt4_policy_mix as policy
from scripts import bt4_value_rewrite as tool
from tests.test_bt4_derived_wdl_sidecar import Session, install, setup


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    wargs, _ = setup(tmp_path)
    wargs.max_shards = 100
    install(monkeypatch, Session())
    tool.wdl.produce(wargs)
    sf = Path(wargs.source)
    source = tmp_path / "B100"
    shutil.copytree(sf, source)
    summary = json.loads((sf / tool.DERIVE_SUMMARY).read_text())
    recipe = {
        "kind": "global",
        "algorithm": "legal-normalized-global-arithmetic-v1",
        "alpha": 1.0,
        "bt4_temperature": 0.5,
        "rows": 6,
        "expected_shards": 2,
        "source_dir": str(sf),
        "source_derive_summary_sha256": wargs.expected_source_summary_sha256,
        "mutated_arrays": ["policy_target"],
    }
    for path in source.glob("*.zarr"):
        group: Any = zarr.open_group(str(path), mode="a")
        legal = group["legal_mask"][:]
        external = np.broadcast_to(np.arange(1, 1859), legal.shape).astype("float32")
        group["policy_target"][:] = policy.mix_policy_targets(
            group["policy_target"][:],
            external,
            legal,
            alpha=1.0,
            scope="global",
            bt4_temperature=0.5,
        )
        group.attrs.update(
            policy_target_mix_kind="global",
            policy_target_mix_alpha=1.0,
            policy_target_mix_bt4_temperature=0.5,
        )
    summary["policy_target_postprocess"] = recipe
    (source / tool.POLICY_SUMMARY).write_text(json.dumps(recipe))
    (source / tool.DERIVE_SUMMARY).write_text(json.dumps(summary))
    return tool.build_parser().parse_args(
        [
            "--source",
            str(source),
            "--sf-source",
            str(sf),
            "--wdl",
            str(wargs.out),
            "--out",
            str(tmp_path / "value10"),
            "--expected-source-summary-sha256",
            tool.wdl.file_sha256(source / tool.DERIVE_SUMMARY),
            "--expected-policy-summary-sha256",
            tool.wdl.file_sha256(source / tool.POLICY_SUMMARY),
            "--expected-sf-summary-sha256",
            wargs.expected_source_summary_sha256,
            "--expected-onnx-sha256",
            wargs.expected_onnx_sha256,
            "--wdl-output",
            "value",
            "--batch-size",
            "2",
            "--minimum-free-gib",
            "0",
        ]
    )


def test_real_rewrite_loader_and_value_gradient(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    before = tool.wdl.storage_identity(Path(args.source))
    result = tool.rewrite(args)
    assert result["rows"] == result["changed_rows"] == 6
    assert result["shards"] == 2
    assert tool.wdl.storage_identity(Path(args.source)) == before
    for spec in result["outputs"]:
        name = spec["path"]
        old, _ = load_shard_arrays(Path(args.source) / name)
        new, metadata = load_shard_arrays(Path(args.out) / name)
        assert metadata["derive_schema"] == 2
        assert metadata["derive_value_scheme"] == tool.VALUE_SCHEME
        assert "categorical_target" not in new
        for key in tool.ARRAYS - {"search_wdl"}:
            np.testing.assert_array_equal(old[key], new[key])
        bt4: Any = zarr.open_group(str(Path(args.wdl) / name), mode="r")
        expected = (
            0.9
            * old["search_wdl"].astype("float64")
            / old["search_wdl"].astype("float64").sum(1, keepdims=True)
            + 0.1 * bt4["bt4_wdl_raw"][:].astype("float64")
        ).astype("float16")
        np.testing.assert_array_equal(new["search_wdl"], expected)
        n = len(expected)
        grads = []
        for arrays in (old, new):
            batch = collate_arrays(arrays, device="cpu")
            logits = torch.tensor([[0.6, -0.2, 0.3]] * n, requires_grad=True)
            outputs = {
                "policy": torch.zeros((n, 1858), requires_grad=True),
                "wdl": logits,
            }
            loss = compute_loss(
                outputs, batch, search_wdl_frac=1.0, sf_wdl_frac=0.0, w_policy=0.0
            )
            target = torch.as_tensor(arrays["search_wdl"].astype("float32"))
            target /= target.sum(1, keepdim=True)
            assert loss["wdl_ce"].item() == pytest.approx(
                float(-(target * logits.detach().log_softmax(1)).sum(1).mean()),
                abs=1e-6,
            )
            assert loss["search_wdl_effective_rows"].item() == n
            loss["wdl_ce"].backward()
            torch.testing.assert_close(
                logits.grad, (logits.detach().softmax(1) - target) / n
            )
            grads.append(logits.grad)
        assert not torch.equal(*grads)
    with pytest.raises(ValueError, match="output or partial"):
        tool.rewrite(args)


@pytest.mark.parametrize(
    "defect",
    [
        "coverage",
        "teacher",
        "output",
        "side_payload",
        "side_identity",
        "side_feed",
        "source_value",
        "policy_recipe",
        "presence",
        "chunk",
    ],
)
def test_refuses_incomplete_or_mismatched_value_inputs(tmp_path, monkeypatch, defect):
    args = fixture(tmp_path, monkeypatch)
    source: Any = zarr.open_group(
        str(Path(args.source) / "shard_000000.zarr"), mode="a"
    )
    side: Any = zarr.open_group(str(Path(args.wdl) / "shard_000000.zarr"), mode="a")
    if defect == "coverage":
        shutil.rmtree(Path(args.wdl) / "shard_000001.zarr")
    elif defect == "teacher":
        args.expected_onnx_sha256 = "a" * 64
    elif defect == "output":
        args.wdl_output = "other"
    elif defect == "side_payload":
        side["bt4_wdl_raw"][0] = [0.25, 0.25, 0.5]
    elif defect == "side_identity":
        side["game_id"][0] += 1
    elif defect == "side_feed":
        side["lc0_feed_sha256"][0, 0] += 1
    elif defect == "source_value":
        source["search_wdl"][0] = [0.25, 0.25, 0.5]
    elif defect == "policy_recipe":
        args.expected_policy_summary_sha256 = "b" * 64
    elif defect == "presence":
        source["has_search_wdl"][0] = 0
    else:
        del source["search_wdl"].chunk_store[source["search_wdl"]._chunk_key((0, 0))]
    with pytest.raises(
        ValueError, match=r"coverage|binding|content|nonpolicy|pin|stored chunk"
    ):
        tool.rewrite(args)
    assert not Path(args.out).exists()


@pytest.mark.parametrize("defect", ["source_mutation", "stop"])
def test_late_failure_preserves_partial_and_never_retries(
    tmp_path, monkeypatch, defect
):
    args = fixture(tmp_path, monkeypatch)
    original = tool.target

    def changed(sf, bt4):
        result = original(sf, bt4)
        if defect == "stop":
            (tmp_path / "STOP").touch()
        else:
            group: Any = zarr.open_group(
                str(Path(args.source) / "shard_000000.zarr"), mode="a"
            )
            group["priority"][0] += 1
        return result

    monkeypatch.setattr(tool, "target", changed)
    with pytest.raises(ValueError, match=r"STOP|source or sidecar changed"):
        tool.rewrite(args)
    assert not Path(args.out).exists()
    assert Path(str(args.out) + ".writing/failed.json").exists()
    with pytest.raises(ValueError, match="output or partial"):
        tool.rewrite(args)


@pytest.mark.parametrize("column", ["game_id", "lc0_feed_sha256"])
def test_semantic_join_refuses_even_with_self_consistent_sidecar_hashes(
    tmp_path, monkeypatch, column
):
    args = fixture(tmp_path, monkeypatch)
    side: Any = zarr.open_group(str(Path(args.wdl) / "shard_000000.zarr"), mode="a")
    values = side[column][:]
    values.flat[0] += 1
    side[column][:] = values
    import hashlib

    hashes = dict(side.attrs["array_sha256"])
    hashes[column] = hashlib.sha256(values.tobytes(order="C")).hexdigest()
    side.attrs["array_sha256"] = hashes
    with pytest.raises(ValueError, match=r"WDL identity|WDL feed identity"):
        tool.rewrite(args)
    assert not Path(args.out).exists()


def test_actual_direct_cli_publishes_same_value_recipe(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys

    args = fixture(tmp_path, monkeypatch)
    argv = [sys.executable, str(Path(tool.__file__).resolve())]
    for key, value in vars(args).items():
        argv += ["--" + key.replace("_", "-"), str(value)]
    result = subprocess.run(
        argv,
        cwd=tmp_path,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    recipe = json.loads((Path(args.out) / tool.SUMMARY).read_text())
    assert recipe["rows"] == recipe["changed_rows"] == 6
    assert recipe["algorithm"] == tool.ALGORITHM
    assert recipe["mutated_arrays"] == ["search_wdl"]


def test_trainer_identity_refuses_different_model_or_named_head(tmp_path):
    from scripts.lc0_control_train import value_scheme_identity_problems

    paths = []
    for i, (model, head) in enumerate(
        [("a" * 64, "value"), ("b" * 64, "value"), ("a" * 64, "value2")]
    ):
        path = tmp_path / str(i)
        shard = path / "shard_000000.zarr"
        group = zarr.open_group(str(shard), mode="w")
        group.attrs.update(
            derive_schema=2,
            derive_value_scheme=tool.VALUE_SCHEME,
            derive_value_source=tool.value_source(model, head),
        )
        paths.append(path)
        assert value_scheme_identity_problems([path]) == []
    assert value_scheme_identity_problems(paths[:2])
    assert value_scheme_identity_problems([paths[0], paths[2]])
