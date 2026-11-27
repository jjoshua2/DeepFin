"""Exercise the bank -> strict checkpoint -> paired readout path on CPU."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.encoding import encode_position, rep_fix
from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig, build_model
from chess_anti_engine.moves import legal_move_mask_for_encoding
from chess_anti_engine.replay.shard import save_local_shard_arrays
from scripts import input_encoding_screen as screen


def _source(tmp_path: Path, *, meta_change: dict | None = None,
            missing_flag: str | None = None) -> tuple[Path, dict[str, np.ndarray]]:
    rep_fix.apply(True, boards_discarded=True)
    board = chess.Board()
    boards = []
    for move in ("e2e4", "c7c5", "g1f3", "d7d6"):
        board.push_uci(move)
        boards.append(board.copy(stack=True))
    x = np.stack([encode_position(
        b, input_history_encoding="lc0_root_legacy_meta",
        input_extra_features="v2_threats",
    ) for b in boards])
    legal = np.stack([legal_move_mask_for_encoding(b, policy_encoding="lc0_1858") for b in boards]).astype(np.uint8)
    arrs = {
        "x": x, "policy_target": legal / legal.sum(axis=1, keepdims=True),
        "wdl_target": np.ones(4, dtype=np.int8), "legal_mask": legal,
        "has_policy": np.ones(4, dtype=np.uint8),
        "has_legal_mask": np.ones(4, dtype=np.uint8),
        "game_id": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "has_game_id": np.ones(4, dtype=np.uint8),
    }
    if missing_flag:
        arrs[missing_flag][2] = 0
    meta = {**{k: v for k, v in screen.ENCODING.items() if k != "input_extra_features"},
            **(meta_change or {})}
    shard = save_local_shard_arrays(tmp_path / "rows.zarr", arrs=arrs, meta=meta)
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps({"schema_version": 1, "sources": [
        {"source_id": "conversion-A", "shards": [str(shard)]},
    ]}))
    return manifest, arrs


def _checkpoint(tmp_path: Path, **changes) -> Path:
    options: dict[str, Any] = {**screen.ENCODING, **changes}
    cfg = ModelConfig(
        embed_dim=32, num_layers=1, num_heads=4, ffn_mult=1,
        use_smolgen=False, **options,
    )
    torch.manual_seed(7)
    model = build_model(cfg)
    path = tmp_path / "model.pt"
    torch.save({"arch": {"_schema_version": ARCH_SCHEMA_VERSION, **asdict(cfg)},
                "model": model.state_dict()}, path)
    return path


def test_freeze_and_real_transformer_readout(tmp_path: Path) -> None:
    manifest, original = _source(tmp_path)
    bank = tmp_path / "bank.npz"
    screen.main(["freeze", "--manifest", str(manifest), "--out", str(bank),
                 "--max-rows", "4", "--max-source-rows", "4"])
    arrays, meta = screen.load_bank(bank)
    np.testing.assert_array_equal(arrays["x"], original["x"])
    np.testing.assert_array_equal(arrays["legal"], original["legal_mask"])
    assert meta["exposure_status"].startswith("unknown")
    assert len(set(arrays["cluster_key"])) == 2
    checkpoint = _checkpoint(tmp_path)
    out = tmp_path / "readout"
    screen.main(["run", "--bank", str(bank), "--checkpoint", str(checkpoint),
                 "--out-dir", str(out), "--families", "attacks_typed",
                 "--batch-size", "2", "--bootstrap", "20"])
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "complete"
    assert set(report["cases"]) == {
        "identity", "attacks_typed__zero", "attacks_typed__spatial_shuffle",
    }
    with np.load(out / "identity.npz") as identity:
        assert np.max(identity["abs_q_change"]) == 0
        assert np.max(identity["top1_changed"]) == 0
    with np.load(out / "attacks_typed__zero.npz") as altered:
        assert np.max(altered["policy_total_variation"]) > 1e-7
        np.testing.assert_array_equal(altered["row_id"], arrays["row_id"])
    assert report["checkpoint"]["sha256"] == screen._sha(checkpoint)
    assert report["bank"]["sha256"] == screen._sha(bank)


@pytest.mark.parametrize("change", [
    {"history_rep_fix": False}, {"history_rep_fix": "true"},
    {"input_history_encoding": "legacy"},
])
def test_freeze_refuses_mismatched_history(tmp_path: Path, change: dict) -> None:
    manifest, _ = _source(tmp_path, meta_change=change)
    with pytest.raises(ValueError, match="recorded corrected"):
        screen.freeze_bank(manifest, tmp_path / "bad.npz", max_rows=4)
    assert not (tmp_path / "bad.npz").exists()


@pytest.mark.parametrize("flag", ["has_game_id", "has_legal_mask"])
def test_freeze_refuses_missing_selected_provenance(tmp_path: Path, flag: str) -> None:
    manifest, _ = _source(tmp_path, missing_flag=flag)
    with pytest.raises(ValueError, match=flag):
        screen.freeze_bank(manifest, tmp_path / "bad.npz", max_rows=4)


def test_freeze_is_repeatable_bounded_and_exclusive(tmp_path: Path) -> None:
    manifest, _ = _source(tmp_path)
    first, second = tmp_path / "one.npz", tmp_path / "two.npz"
    screen.freeze_bank(manifest, first, max_rows=2, seed=6)
    screen.freeze_bank(manifest, second, max_rows=2, seed=6)
    a, _ = screen.load_bank(first)
    b, _ = screen.load_bank(second)
    np.testing.assert_array_equal(a["row_id"], b["row_id"])
    before = first.read_bytes()
    with pytest.raises(ValueError, match=r"new \.npz"):
        screen.freeze_bank(manifest, first, max_rows=2)
    assert first.read_bytes() == before
    with pytest.raises(ValueError, match="source rows exceed"):
        screen.freeze_bank(manifest, tmp_path / "cap.npz", max_rows=2, max_source_rows=3)


def test_source_namespaces_keep_colliding_game_ids_separate(tmp_path: Path) -> None:
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    first, _ = _source(left)
    second, _ = _source(right)
    manifest = json.loads(first.read_text())
    other = json.loads(second.read_text())["sources"][0]
    other["source_id"] = "conversion-B"
    manifest["sources"].append(other)
    first.write_text(json.dumps(manifest))
    bank = tmp_path / "bank.npz"
    screen.freeze_bank(first, bank, max_rows=8)
    arrays, meta = screen.load_bank(bank)
    assert len(set(arrays["cluster_key"])) == 4
    assert meta["duplicate_input_rows"] == 4


@pytest.mark.parametrize(("field", "match"), [
    ("x", "input content identity"), ("legal", "row content identity"),
    ("cluster_key", "game cluster identity"),
])
def test_bank_rechecks_saved_content_and_cluster_identity(
    tmp_path: Path, field: str, match: str,
) -> None:
    manifest, _ = _source(tmp_path)
    bank = tmp_path / "bank.npz"
    screen.freeze_bank(manifest, bank, max_rows=4)
    with np.load(bank, allow_pickle=False) as saved:
        payload = {name: np.array(saved[name]) for name in saved.files}
    if field == "x":
        payload[field][0, 150, 2, 2] += 0.5
    elif field == "legal":
        payload[field][0, 0] = 1 - payload[field][0, 0]
    else:
        payload[field][0] = payload[field][2]
    changed = tmp_path / "changed.npz"
    screen._write_npz(changed, **payload)
    with pytest.raises(ValueError, match=match):
        screen.load_bank(changed)


@pytest.mark.parametrize(("changes", "match"), [
    ({"history_rep_fix": False}, "complete input/history/policy"),
    ({"use_dynamic_relations": True}, "relation inputs are unsupported"),
])
def test_run_rejects_incompatible_checkpoint_before_output(
    tmp_path: Path, changes: dict, match: str,
) -> None:
    manifest, _ = _source(tmp_path)
    bank = tmp_path / "bank.npz"
    screen.freeze_bank(manifest, bank, max_rows=4)
    checkpoint = _checkpoint(tmp_path, **changes)
    out = tmp_path / "rejected"
    with pytest.raises(ValueError, match=match):
        screen.run_screen(bank, checkpoint, out, families=["extras_all"], modes=["zero"])
    assert not out.exists()


def test_incomplete_checkpoint_is_not_scored(tmp_path: Path) -> None:
    manifest, _ = _source(tmp_path)
    bank = tmp_path / "bank.npz"
    screen.freeze_bank(manifest, bank, max_rows=4)
    checkpoint = _checkpoint(tmp_path)
    payload = torch.load(checkpoint, weights_only=True)
    del payload["model"]["embed.weight"]
    torch.save(payload, checkpoint)
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="Missing key"):
        screen.run_screen(bank, checkpoint, out, families=["extras_all"], modes=["zero"])
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["cases"] == {}
    assert not (out / "baseline.npz").exists()


def test_mislabeled_v1_checkpoint_cannot_be_zero_expanded(tmp_path: Path) -> None:
    manifest, _ = _source(tmp_path)
    bank = tmp_path / "bank.npz"
    screen.freeze_bank(manifest, bank, max_rows=4)
    checkpoint = _checkpoint(tmp_path)
    payload = torch.load(checkpoint, weights_only=True)
    payload["model"]["embed.weight"] = payload["model"]["embed.weight"][:, :146].clone()
    torch.save(payload, checkpoint)
    out = tmp_path / "mislabeled"
    with pytest.raises(RuntimeError, match=r"size mismatch for embed\.weight"):
        screen.run_screen(bank, checkpoint, out, families=["threats_all"], modes=["zero"])
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "failed"
    assert not (out / "baseline.npz").exists()


def test_compiled_prefix_is_only_name_normalization(tmp_path: Path) -> None:
    manifest, _ = _source(tmp_path)
    bank = tmp_path / "bank.npz"
    screen.freeze_bank(manifest, bank, max_rows=4)
    checkpoint = _checkpoint(tmp_path)
    payload = torch.load(checkpoint, weights_only=True)
    payload["model"] = {f"_orig_mod.{key}": value for key, value in payload["model"].items()}
    torch.save(payload, checkpoint)
    report = screen.run_screen(bank, checkpoint, tmp_path / "compiled",
                               families=["threats_all"], modes=["zero"], bootstrap=0)
    assert report["status"] == "complete"


def test_identity_control_stops_unstable_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _ = _source(tmp_path)
    bank = tmp_path / "bank.npz"
    screen.freeze_bank(manifest, bank, max_rows=4)
    checkpoint = _checkpoint(tmp_path)
    original_predict = screen._predict
    calls = 0

    def unstable(*args, **kwargs):
        nonlocal calls
        result = original_predict(*args, **kwargs)
        calls += 1
        if calls > 1:
            result["wdl"][:, 0] += 1
        return result

    monkeypatch.setattr(screen, "_predict", unstable)
    out = tmp_path / "unstable"
    with pytest.raises(ValueError, match="identity repeat-forward"):
        screen.run_screen(bank, checkpoint, out, families=["extras_all"], modes=["zero"], bootstrap=0)
    report = json.loads((out / "report.json").read_text())
    assert calls == 2
    assert report["status"] == "failed"
    assert report["cases"] == {}
