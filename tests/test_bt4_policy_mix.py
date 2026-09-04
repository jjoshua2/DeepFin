from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import chess
import numpy as np
import pytest
import zarr

from chess_anti_engine.moves.leela_index import compact_index_for_move
from scripts import bt4_policy_mix as tool


def test_top_max_ties_preserves_mass_and_unique_maxima() -> None:
    source = np.asarray(
        [[0.5, 0.5, 0.0, 0.0], [0.7, 0.2, 0.0, 0.0]],
        dtype=np.float16,
    )
    bt4 = np.asarray(
        [[0.1, 0.9, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0]],
        dtype=np.float32,
    )
    legal = np.asarray([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=np.uint8)

    mixed = tool.mix_policy_targets(
        source,
        bt4,
        legal,
        alpha=0.2,
        scope="top-max-ties",
    )

    assert mixed[0] == pytest.approx([0.42, 0.58, 0.0, 0.0])
    assert np.array_equal(mixed[1], source[1].astype(np.float32))
    assert mixed[0, :2].sum() == pytest.approx(source[0, :2].sum())


def test_global_scope_is_the_literal_arithmetic_mix() -> None:
    source = np.asarray([[0.8, 0.2, 0.4]], dtype=np.float32)
    bt4 = np.asarray([[0.1, 0.9, 0.8]], dtype=np.float32)
    legal = np.asarray([[1, 1, 0]], dtype=np.uint8)

    mixed = tool.mix_policy_targets(
        source,
        bt4,
        legal,
        alpha=0.25,
        scope="global",
    )

    assert mixed[0] == pytest.approx([0.625, 0.375, 0.0])


@pytest.mark.parametrize("alpha", [0.0, 1.1, -0.1, float("nan")])
def test_invalid_alpha_is_refused(alpha: float) -> None:
    with pytest.raises(ValueError, match="positive, and at most 1"):
        tool.validate_alpha(alpha)


@pytest.mark.parametrize("temperature", [0.0, -0.1, float("nan")])
def test_invalid_bt4_temperature_is_refused(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature must be finite and positive"):
        tool.validate_bt4_temperature(temperature)


@pytest.mark.parametrize("ratio", [0.0, 1.0, 1.1, -0.1, float("nan")])
def test_invalid_near_max_ratio_is_refused(ratio: float) -> None:
    with pytest.raises(ValueError, match="positive, and below 1"):
        tool.validate_near_max_ratio(ratio)


def test_full_bt4_tie_break_stays_soft_and_preserves_non_tie_mass() -> None:
    source = np.asarray([[0.4, 0.4, 0.2]], dtype=np.float16)
    bt4 = np.asarray([[0.10, 0.08, 0.82]], dtype=np.float32)
    legal = np.ones((1, 3), dtype=np.uint8)

    mixed = tool.mix_policy_targets(
        source,
        bt4,
        legal,
        alpha=1.0,
        scope="top-max-ties",
    )

    stored_source = source.astype(np.float32)
    top_mass = float(stored_source[0, :2].sum())
    assert mixed[0] == pytest.approx(
        [top_mass * 5.0 / 9.0, top_mass * 4.0 / 9.0, stored_source[0, 2]],
    )
    assert mixed[0, :2].sum() == pytest.approx(top_mass)
    assert mixed[0, 2] == stored_source[0, 2]
    assert float(mixed.max()) < 0.5


def test_bt4_temperature_sharpens_only_within_top_tie() -> None:
    source = np.asarray([[0.4, 0.4, 0.2]], dtype=np.float16)
    bt4 = np.asarray([[0.10, 0.08, 0.82]], dtype=np.float32)
    legal = np.ones((1, 3), dtype=np.uint8)

    mixed = tool.mix_policy_targets(
        source,
        bt4,
        legal,
        alpha=1.0,
        scope="top-max-ties",
        bt4_temperature=0.5,
    )

    stored_source = source.astype(np.float32)
    top_mass = float(stored_source[0, :2].sum())
    assert mixed[0] == pytest.approx(
        [top_mass * 25.0 / 41.0, top_mass * 16.0 / 41.0, stored_source[0, 2]],
    )
    assert mixed[0, :2].sum() == pytest.approx(top_mass)
    assert mixed[0, 2] == stored_source[0, 2]


def test_near_max_ratio_extends_set_and_preserves_outside_mass() -> None:
    source = np.asarray(
        [[0.60, 0.35, 0.05], [0.80, 0.19, 0.01]],
        dtype=np.float16,
    )
    bt4 = np.asarray(
        [[0.20, 0.80, 0.00], [0.20, 0.79, 0.01]],
        dtype=np.float32,
    )
    legal = np.ones((2, 3), dtype=np.uint8)

    mixed = tool.mix_policy_targets(
        source,
        bt4,
        legal,
        alpha=1.0,
        scope="near-max-ratio",
        near_max_ratio=0.5,
    )

    stored_source = source.astype(np.float32)
    selected_mass = float(stored_source[0, :2].sum())
    assert mixed[0] == pytest.approx(
        [selected_mass * 0.2, selected_mass * 0.8, stored_source[0, 2]],
    )
    assert np.array_equal(mixed[1], stored_source[1])


def test_sf_cp_window_uses_true_rank_gap_not_cold_source_probability() -> None:
    source = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float16)
    bt4 = np.asarray([[0.10, 0.60, 0.25, 0.05]], dtype=np.float32)
    legal = np.ones((1, 4), dtype=np.uint8)
    rank_indices = np.asarray([[0, 1, 2]], dtype=np.uint16)
    rank_gaps = np.asarray([[0.0, 8.0, 19.0]], dtype=np.float32)

    mixed = tool.mix_policy_targets(
        source,
        bt4,
        legal,
        alpha=0.5,
        scope="sf-cp-window",
        bt4_temperature=1.0,
        sf_rank_indices=rank_indices,
        sf_rank_gaps_cp=rank_gaps,
        sf_rank_cap=3,
        sf_cp_window=10.0,
    )

    # Only top-1 and the true 8cp runner-up are eligible. Half the selected
    # mass keeps the cold SF target and half follows conditional BT4 (1:6).
    assert mixed[0] == pytest.approx([0.5 + 0.5 / 7.0, 3.0 / 7.0, 0.0, 0.0])


def test_sf_cp_window_unions_all_stored_top_ties_with_rank_cap() -> None:
    source = np.asarray([[0.5, 0.5, 0.0, 0.0]], dtype=np.float16)
    bt4 = np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    legal = np.ones((1, 4), dtype=np.uint8)

    mixed = tool.mix_policy_targets(
        source,
        bt4,
        legal,
        alpha=1.0,
        scope="sf-cp-window",
        sf_rank_indices=np.asarray([[0, 2]], dtype=np.uint16),
        sf_rank_gaps_cp=np.asarray([[0.0, 5.0]], dtype=np.float32),
        sf_rank_cap=2,
        sf_cp_window=10.0,
    )

    # Index 1 is retained because it is a stored top tie even though it is not
    # in the two supplied d9 ranks; index 3 remains outside the treatment.
    assert mixed[0] == pytest.approx([1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0, 0.0])


def test_top_tie_break_refuses_zero_bt4_mass_on_tied_moves() -> None:
    source = np.asarray([[0.4, 0.4, 0.2]], dtype=np.float16)
    bt4 = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)
    legal = np.ones((1, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="no mass on the source top-tie"):
        tool.mix_policy_targets(
            source,
            bt4,
            legal,
            alpha=1.0,
            scope="top-max-ties",
        )


def test_functional_remap_identity_ignores_only_repository_head() -> None:
    current = tool.remap_provenance()
    older_head = dict(current)
    older_head["git_head"] = "0" * 40

    assert tool.functional_remap_identity(older_head) == (
        tool.functional_remap_identity(current)
    )

    changed_blob = json.loads(json.dumps(current))
    first_source = next(iter(changed_blob["blobs"]))
    changed_blob["blobs"][first_source] = "different"
    assert tool.functional_remap_identity(changed_blob) != (
        tool.functional_remap_identity(current)
    )


def _write_source(root: Path) -> tuple[Path, Path, dict[str, np.ndarray]]:
    source_dir = root / "source"
    shard_path = source_dir / "shard_000000.zarr"
    group = zarr.open_group(str(shard_path), mode="w")
    policy = np.zeros((2, 1858), dtype=np.float16)
    policy[0, :3] = [0.4, 0.4, 0.2]
    policy[1, :2] = [0.60, 0.40]
    legal_mask = np.zeros((2, 1858), dtype=np.uint8)
    legal_mask[:, :3] = 1
    arrays = {
        "x": np.zeros((2, 175, 8, 8), dtype=np.float16),
        "policy_target": policy,
        "legal_mask": legal_mask,
        "has_policy": np.ones(2, dtype=np.uint8),
        "has_legal_mask": np.ones(2, dtype=np.uint8),
        "wdl_target": np.asarray([0, 2], dtype=np.int8),
        "search_wdl": np.asarray(
            [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]],
            dtype=np.float16,
        ),
        "has_search_wdl": np.ones(2, dtype=np.uint8),
        "game_id": np.asarray([11, 12], dtype=np.int64),
        "ply_index": np.asarray([3, 4], dtype=np.int32),
        "has_game_id": np.ones(2, dtype=np.uint8),
        "has_ply_index": np.ones(2, dtype=np.uint8),
    }
    # Make the row identities distinct without requiring a decodable board.
    arrays["x"][1, 0, 0, 0] = 1.0
    for name, value in arrays.items():
        chunks = (1, *value.shape[1:])
        group.create_dataset(name, data=value, chunks=chunks)
    group.attrs.update(
        {
            "input_history_encoding": "lc0_root_legacy_meta",
            "policy_encoding": "lc0_1858",
            "positions": 2,
        }
    )
    (source_dir / tool.DERIVE_SUMMARY).write_text(
        json.dumps(
            {
                "schema": 1,
                "scheme": {"canonical": "uniform-d9"},
                "temp_requested": 0.0005,
                "floor_requested": 0.0,
                "cp_map": {"cp_slope": 0.006, "cp_draw_width": 120.0},
                "policy": {"encoding": "lc0_1858"},
                "realized": {"realized_base_depth_histogram": {"9": 2}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return source_dir, shard_path, arrays


def _write_sidecar(root: Path, source_dir: Path, shard_path: Path) -> Path:
    source = zarr.open_group(str(shard_path), mode="r")
    encoding, keys, key_sha, policy_sha = tool._sidecar_identity(source, shard_path)
    sidecar_dir = root / "sidecar"
    sidecar_path = sidecar_dir / shard_path.name
    sidecar = zarr.open_group(str(sidecar_path), mode="w")
    bt4 = np.zeros((2, 1858), dtype=np.float32)
    bt4[0, :2] = [0.1, 0.9]
    bt4[1, :2] = [0.2, 0.8]
    sidecar.create_dataset(tool.SIDECAR_KEY_FIELD, data=keys, chunks=keys.shape)
    sidecar.create_dataset(tool.SIDECAR_POLICY_FIELD, data=bt4, chunks=(1, 1858))
    sidecar.attrs.update(
        {
            "bt4_policy_sidecar_schema": tool.SIDECAR_SCHEMA,
            "source_shard": shard_path.name,
            "positions": 2,
            "source_key_sha256": key_sha,
            "source_policy_sha256": policy_sha,
            "input_history_encoding": encoding,
            "policy_encoding": "lc0_1858",
            "policy_size": 1858,
            "onnx_sha256": "fake-onnx-sha",
            "providers": ["fake"],
            "policy_output": "policy",
            "teacher_evaluations_per_position": 1,
            "search_nodes": 0,
            "stored_dtype": "float32",
        }
    )
    (sidecar_dir / tool.SIDECAR_SUMMARY).write_text(
        json.dumps(
            {
                "schema": tool.SIDECAR_SCHEMA,
                "source_dir": str(source_dir.resolve()),
                "source_shards": 1,
                "sidecar_shards": 1,
                "rows": 2,
                "kind": "bt4_raw_legal_policy_sidecar",
                "policy_encoding": "lc0_1858",
                "onnx": {"path": "fake.onnx", "sha256": "fake-onnx-sha"},
                "providers": ["fake"],
                "policy_output": "policy",
                "teacher_evaluations_per_position": 1,
                "search_nodes": 0,
                "stored_dtype": "float32",
                "remap": tool.remap_provenance(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return sidecar_dir


def _write_audit_receipt(
    root: Path,
    *,
    scope: str = "top-max-ties",
    bt4_temperature: float = 1.0,
    near_max_ratio: float = 0.5,
    sf_rank_cap: int = 3,
    sf_cp_window: float = 10.0,
) -> Path:
    path = root / "audit.json"
    near = scope in {"near-max-ratio", "sf-cp-window"}
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "bt4_policy_mix_frozen_deep_sf_audit",
                "treatment": tool.treatment_spec(
                    scope=scope,
                    alpha=1.0,
                    bt4_temperature=bt4_temperature,
                    near_max_ratio=near_max_ratio,
                    sf_rank_cap=sf_rank_cap,
                    sf_cp_window=sf_cp_window,
                ),
                "source_target_contract": tool.SOURCE_TARGET_CONTRACT,
                "ruler": tool.AUDIT_RULER_CONTRACT,
                "treatment_invariants": {
                    "candidate_set_wider_rows": 1 if near else 0,
                    "changed_unique_max_rows": 1 if near else 0,
                    "temperature_one_top1_mismatch_rows": 0,
                    "temperature_prestorage_top1_mismatch_rows": 0,
                    "near_max_extended": True,
                    "top_tie_unique_max_identity": True,
                    "temperature_one_top1_preserved": True,
                    "temperature_rank_preserved_before_storage": True,
                    "selected_mass_drift_within_bounds": True,
                },
                "gate": {
                    "training_permitted": True,
                    "treatment_invariants_passed": True,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_rank_sidecar(root: Path, source_dir: Path, shard_path: Path) -> Path:
    source = zarr.open_group(str(shard_path), mode="r")
    rank_dir = root / "sf-ranks"
    rank_path = rank_dir / shard_path.name
    group = zarr.open_group(str(rank_path), mode="w")
    indices = np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.uint16)
    gaps = np.asarray([[0.0, 0.0, 20.0], [0.0, 8.0, 30.0]], dtype=np.float32)
    counts = np.asarray([3, 3], dtype=np.uint8)
    group.create_dataset(tool.sf_ranks.INDEX_FIELD, data=indices)
    group.create_dataset(tool.sf_ranks.GAP_FIELD, data=gaps)
    group.create_dataset(tool.sf_ranks.COUNT_FIELD, data=counts)
    source_summary_sha = tool.file_sha256(source_dir / tool.DERIVE_SUMMARY)
    group.attrs.update(
        {
            "sf_d9_rank_sidecar_schema": tool.sf_ranks.SCHEMA,
            "source_shard": shard_path.name,
            "source_rows": 2,
            "source_row_identity_sha256": tool._source_game_ply_sha(
                source,
                shard_path,
            ),
            "source_derive_summary_sha256": source_summary_sha,
            "raw_config_sha256": "raw-config",
            "depth": 9,
            "top_k": 3,
            "index_encoding": "lc0_1858",
            "gap_definition": "rank1_effective_cp-minus-ranked_effective_cp",
            "payload_sha256": tool.sf_ranks._sha_arrays(indices, gaps, counts),
        }
    )
    (rank_dir / tool.sf_ranks.SUMMARY_NAME).write_text(
        json.dumps(
            {
                "schema": tool.sf_ranks.SCHEMA,
                "kind": "sf_d9_rank_gap_sidecar",
                "source_dir": str(source_dir),
                "source_derive_summary_sha256": source_summary_sha,
                "rows": 2,
                "shards": 1,
                "depth": 9,
                "top_k": 3,
                "index_encoding": "lc0_1858",
                "gap_definition": "rank1_effective_cp-minus-ranked_effective_cp",
                "outputs": [
                    {
                        "path": shard_path.name,
                        "rows": 2,
                        "source_row_identity_sha256": group.attrs[
                            "source_row_identity_sha256"
                        ],
                        "payload_sha256": group.attrs["payload_sha256"],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return rank_dir


def test_audit_receipt_requires_exact_treatment_algorithm(tmp_path: Path) -> None:
    path = _write_audit_receipt(tmp_path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    del receipt["treatment"]["algorithm"]
    path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"treatment\.algorithm"):
        tool._load_audit_receipt(
            path,
            alpha=1.0,
            scope="top-max-ties",
            bt4_temperature=1.0,
            near_max_ratio=0.5,
        )


@pytest.mark.parametrize(
    ("verdict", "invariants_passed", "expected"),
    [
        ("graduate_win", True, True),
        ("graduate_tie", True, True),
        ("kill", True, False),
        ("graduate_win", False, False),
        ("graduate_tie", False, False),
    ],
)
def test_audit_exit_admission_requires_value_and_fidelity_gates(
    verdict: str,
    invariants_passed: bool,
    expected: bool,
) -> None:
    assert (
        tool.audit_training_permitted(
            verdict=verdict,
            treatment_invariants_passed=invariants_passed,
        )
        is expected
    )


def _run_synthetic_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_prefix: list[float],
    bt4_prefix: list[float],
    scope: str,
    bt4_temperature: float,
) -> tuple[int, dict[str, Any]]:
    board = chess.Board()
    legal_ucis = [move.uci() for move in board.legal_moves]
    source = np.zeros(len(legal_ucis), dtype=np.float32)
    source[: len(source_prefix)] = source_prefix
    bt4 = np.zeros(len(legal_ucis), dtype=np.float64)
    bt4[: len(bt4_prefix)] = bt4_prefix
    positions = [
        SimpleNamespace(
            key=f"p{index}",
            fen=board.fen(),
            phase=index % 3,
            source=index % 2,
            move_cp=dict.fromkeys(legal_ucis, 0.0),
            best_cp=0.0,
        )
        for index in range(6)
    ]
    d9_rows = {
        position.key: {
            "key": position.key,
            "timed_out": False,
            "depths": [{"depth": 9, "lines": []}],
        }
        for position in positions
    }
    bt4_rows = {
        position.key: {"key": position.key, "topk": list(zip(legal_ucis, bt4))}
        for position in positions
    }
    monkeypatch.setattr(tool, "load_audit_set", lambda _path: positions)
    monkeypatch.setattr(
        tool,
        "_load_keyed_jsonl",
        lambda path: d9_rows if "d9" in Path(path).name else bt4_rows,
    )
    monkeypatch.setattr(
        tool,
        "_stored_d9_policy",
        lambda _legal, _lines: source.copy(),
    )
    monkeypatch.setattr(tool, "file_sha256", lambda path: Path(path).name)
    out_path = tmp_path / "audit-result.json"
    result = tool.audit_mix(
        argparse.Namespace(
            audit_set=tmp_path / "audit.jsonl",
            d9_labels=tmp_path / "d9.jsonl",
            bt4_cache=tmp_path / "bt4.jsonl",
            json=out_path,
            alpha=1.0,
            scope=scope,
            bt4_temperature=bt4_temperature,
            near_max_ratio=0.5,
            boot=200,
            seed=20260903,
        )
    )
    return result, json.loads(out_path.read_text(encoding="utf-8"))


def test_near_max_audit_proves_it_changes_unique_max_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, report = _run_synthetic_audit(
        tmp_path,
        monkeypatch,
        source_prefix=[0.6, 0.4],
        bt4_prefix=[0.25, 0.75],
        scope="near-max-ratio",
        bt4_temperature=1.0,
    )

    assert result == 0
    invariants = report["treatment_invariants"]
    assert isinstance(invariants, dict)
    assert invariants["candidate_set_wider_rows"] == 6
    assert invariants["changed_unique_max_rows"] == 6
    assert invariants["near_max_extended"] is True
    assert report["gate"]["training_permitted"] is True


def test_sharpened_audit_rejects_float16_top1_change_from_temperature_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, report = _run_synthetic_audit(
        tmp_path,
        monkeypatch,
        source_prefix=[0.5, 0.5],
        bt4_prefix=[0.4999, 0.5001],
        scope="top-max-ties",
        bt4_temperature=0.5,
    )

    assert result == 2
    invariants = report["treatment_invariants"]
    assert isinstance(invariants, dict)
    assert invariants["temperature_one_top1_mismatch_rows"] == 6
    assert invariants["temperature_one_top1_preserved"] is False
    assert report["gate"]["verdict"] == "graduate_tie"
    assert report["gate"]["training_permitted"] is False


def test_audit_rejects_selected_mass_drift_over_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool, "SELECTED_MASS_DRIFT_MEAN_MAX", -1.0)
    result, report = _run_synthetic_audit(
        tmp_path,
        monkeypatch,
        source_prefix=[0.5, 0.5],
        bt4_prefix=[0.2, 0.8],
        scope="top-max-ties",
        bt4_temperature=1.0,
    )

    assert result == 2
    invariants = report["treatment_invariants"]
    assert isinstance(invariants, dict)
    assert invariants["selected_mass_drift_within_bounds"] is False
    assert report["gate"]["verdict"] == "graduate_tie"
    assert report["gate"]["training_permitted"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [("bt4_temperature", 0.5), ("near_max_ratio", 0.25)],
)
def test_audit_receipt_requires_exact_new_treatment_fields(
    tmp_path: Path,
    field: str,
    value: float,
) -> None:
    path = _write_audit_receipt(
        tmp_path,
        scope="near-max-ratio",
        near_max_ratio=0.5,
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["treatment"][field] = value
    path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"treatment\.{field}"):
        tool._load_audit_receipt(
            path,
            alpha=1.0,
            scope="near-max-ratio",
            bt4_temperature=1.0,
            near_max_ratio=0.5,
        )


@pytest.mark.parametrize(
    ("scope", "bt4_temperature", "field", "value"),
    [
        ("near-max-ratio", 1.0, "candidate_set_wider_rows", 0),
        ("near-max-ratio", 1.0, "changed_unique_max_rows", 0),
        ("top-max-ties", 0.5, "temperature_one_top1_mismatch_rows", 1),
        ("top-max-ties", 1.0, "selected_mass_drift_within_bounds", False),
    ],
)
def test_audit_receipt_enforces_treatment_invariants(
    tmp_path: Path,
    scope: str,
    bt4_temperature: float,
    field: str,
    value: int | bool,
) -> None:
    path = _write_audit_receipt(
        tmp_path,
        scope=scope,
        bt4_temperature=bt4_temperature,
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["treatment_invariants"][field] = value
    path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"treatment_invariants\.{field}"):
        tool._load_audit_receipt(
            path,
            alpha=1.0,
            scope=scope,
            bt4_temperature=bt4_temperature,
            near_max_ratio=0.5,
        )


def test_mix_corpus_changes_only_the_policy_target(tmp_path: Path) -> None:
    source_dir, shard_path, arrays = _write_source(tmp_path)
    sidecar_dir = _write_sidecar(tmp_path, source_dir, shard_path)
    audit_receipt = _write_audit_receipt(tmp_path)
    out_dir = tmp_path / "mixed"

    result = tool.mix_corpus(
        argparse.Namespace(
            shards=source_dir,
            sidecar=sidecar_dir,
            out=out_dir,
            alpha=1.0,
            scope="top-max-ties",
            bt4_temperature=1.0,
            near_max_ratio=0.5,
            expected_rows=2,
            expected_shards=1,
            expected_source_summary_sha256=tool.file_sha256(
                source_dir / tool.DERIVE_SUMMARY
            ),
            audit_receipt=audit_receipt,
        )
    )

    assert result == 0
    source = zarr.open_group(str(shard_path), mode="r")
    mixed = zarr.open_group(str(out_dir / shard_path.name), mode="r")
    assert np.array_equal(
        np.asarray(source["policy_target"][:]),
        arrays["policy_target"],
    )
    assert mixed["policy_target"][0, :4] == pytest.approx(
        [0.08, 0.72, 0.2, 0.0],
        abs=1e-3,
    )
    assert (
        np.asarray(mixed["policy_target"][0, 2]).tobytes()
        == np.asarray(source["policy_target"][0, 2]).tobytes()
    )
    assert np.array_equal(
        np.asarray(mixed["policy_target"][1]),
        arrays["policy_target"][1],
    )
    for name, expected in arrays.items():
        if name != "policy_target":
            assert np.array_equal(np.asarray(mixed[name][:]), expected), name
    assert mixed.attrs["policy_target_mix_kind"] == "top-max-ties"
    assert (
        mixed.attrs["policy_target_mix_algorithm"]
        == tool.TREATMENT_ALGORITHMS["top-max-ties"]
    )
    assert mixed.attrs["policy_target_mix_value_columns_unchanged"] is True
    assert mixed.attrs["policy_target_mix_bt4_temperature"] == 1.0
    assert mixed.attrs["policy_target_mix_near_max_ratio"] is None
    receipt = json.loads((out_dir / tool.MIX_SUMMARY).read_text(encoding="utf-8"))
    assert receipt["changed_rows"] == 1
    assert receipt["source_top_tied_rows"] == 1
    assert receipt["changed_unique_max_rows"] == 0
    assert receipt["value_columns_unchanged"] == ["wdl_target", "search_wdl"]
    assert receipt["top1_ge_0_99_fraction"] == {"source": 0.0, "mixed": 0.0}
    assert receipt["selected_mass_abs_drift"]["mean"] <= 0.0002
    assert receipt["selected_mass_abs_drift"]["max"] <= 0.005


def test_mix_corpus_near_max_changes_declared_unique_maxima(tmp_path: Path) -> None:
    source_dir, shard_path, arrays = _write_source(tmp_path)
    sidecar_dir = _write_sidecar(tmp_path, source_dir, shard_path)
    audit_receipt = _write_audit_receipt(
        tmp_path,
        scope="near-max-ratio",
        near_max_ratio=0.5,
    )
    out_dir = tmp_path / "near-mixed"

    result = tool.mix_corpus(
        argparse.Namespace(
            shards=source_dir,
            sidecar=sidecar_dir,
            out=out_dir,
            alpha=1.0,
            scope="near-max-ratio",
            bt4_temperature=1.0,
            near_max_ratio=0.5,
            expected_rows=2,
            expected_shards=1,
            expected_source_summary_sha256=tool.file_sha256(
                source_dir / tool.DERIVE_SUMMARY
            ),
            audit_receipt=audit_receipt,
        )
    )

    assert result == 0
    source = zarr.open_group(str(shard_path), mode="r")
    mixed = zarr.open_group(str(out_dir / shard_path.name), mode="r")
    assert np.array_equal(np.asarray(source["policy_target"][:]), arrays["policy_target"])
    assert mixed["policy_target"][1, :4] == pytest.approx(
        [0.2, 0.8, 0.0, 0.0],
        abs=1e-3,
    )
    receipt = json.loads((out_dir / tool.MIX_SUMMARY).read_text(encoding="utf-8"))
    assert receipt["changed_unique_max_rows"] == 1
    assert receipt["candidate_set_wider_rows"] == 2
    assert receipt["near_max_ratio"] == 0.5


def test_mix_corpus_uses_verified_sf_cp_rank_sidecar(tmp_path: Path) -> None:
    source_dir, shard_path, _arrays = _write_source(tmp_path)
    sidecar_dir = _write_sidecar(tmp_path, source_dir, shard_path)
    rank_dir = _write_rank_sidecar(tmp_path, source_dir, shard_path)
    audit_receipt = _write_audit_receipt(
        tmp_path,
        scope="sf-cp-window",
        sf_rank_cap=2,
        sf_cp_window=10.0,
    )
    out_dir = tmp_path / "cp-window-mixed"

    result = tool.mix_corpus(
        argparse.Namespace(
            shards=source_dir,
            sidecar=sidecar_dir,
            sf_rank_sidecar=rank_dir,
            out=out_dir,
            alpha=1.0,
            scope="sf-cp-window",
            bt4_temperature=1.0,
            near_max_ratio=0.5,
            sf_rank_cap=2,
            sf_cp_window=10.0,
            expected_rows=2,
            expected_shards=1,
            expected_source_summary_sha256=tool.file_sha256(
                source_dir / tool.DERIVE_SUMMARY
            ),
            audit_receipt=audit_receipt,
        )
    )

    assert result == 0
    mixed = zarr.open_group(str(out_dir / shard_path.name), mode="r")
    assert np.asarray(mixed["policy_target"][0, :3]) == pytest.approx(
        [0.08, 0.72, 0.20],
        abs=1e-3,
    )
    assert np.asarray(mixed["policy_target"][1, :3]) == pytest.approx(
        [0.20, 0.80, 0.0],
        abs=1e-3,
    )
    assert mixed.attrs["policy_target_mix_sf_rank_cap"] == 2
    assert mixed.attrs["policy_target_mix_sf_cp_window"] == 10.0
    receipt = json.loads((out_dir / tool.MIX_SUMMARY).read_text(encoding="utf-8"))
    assert receipt["candidate_set_wider_rows"] == 1
    assert receipt["changed_unique_max_rows"] == 1


def test_mix_corpus_refuses_rank_summary_receipt_drift(tmp_path: Path) -> None:
    source_dir, shard_path, _arrays = _write_source(tmp_path)
    sidecar_dir = _write_sidecar(tmp_path, source_dir, shard_path)
    rank_dir = _write_rank_sidecar(tmp_path, source_dir, shard_path)
    summary_path = rank_dir / tool.sf_ranks.SUMMARY_NAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["outputs"][0]["payload_sha256"] = "0" * 64
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    audit_receipt = _write_audit_receipt(
        tmp_path,
        scope="sf-cp-window",
        sf_rank_cap=2,
        sf_cp_window=10.0,
    )

    with pytest.raises(ValueError, match="summary output receipt mismatch"):
        tool.mix_corpus(
            argparse.Namespace(
                shards=source_dir,
                sidecar=sidecar_dir,
                sf_rank_sidecar=rank_dir,
                out=tmp_path / "bad-rank-summary-mixed",
                alpha=1.0,
                scope="sf-cp-window",
                bt4_temperature=1.0,
                near_max_ratio=0.5,
                sf_rank_cap=2,
                sf_cp_window=10.0,
                expected_rows=2,
                expected_shards=1,
                expected_source_summary_sha256=tool.file_sha256(
                    source_dir / tool.DERIVE_SUMMARY
                ),
                audit_receipt=audit_receipt,
            )
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("expected_rows", 3, "source corpus size mismatch"),
        ("expected_shards", 2, "source corpus size mismatch"),
        (
            "expected_source_summary_sha256",
            "0" * 64,
            "source derive-summary SHA-256 mismatch",
        ),
    ],
)
def test_mix_corpus_requires_exact_source_bank(
    tmp_path: Path,
    field: str,
    value: int | str,
    message: str,
) -> None:
    source_dir, shard_path, _arrays = _write_source(tmp_path)
    sidecar_dir = _write_sidecar(tmp_path, source_dir, shard_path)
    audit_receipt = _write_audit_receipt(tmp_path)
    arguments: dict[str, object] = {
        "shards": source_dir,
        "sidecar": sidecar_dir,
        "out": tmp_path / "wrong-bank",
        "alpha": 1.0,
        "scope": "top-max-ties",
        "bt4_temperature": 1.0,
        "near_max_ratio": 0.5,
        "expected_rows": 2,
        "expected_shards": 1,
        "expected_source_summary_sha256": tool.file_sha256(
            source_dir / tool.DERIVE_SUMMARY
        ),
        "audit_receipt": audit_receipt,
    }
    arguments[field] = value

    with pytest.raises(SystemExit, match=message):
        tool.mix_corpus(argparse.Namespace(**arguments))

    assert not (tmp_path / "wrong-bank").exists()
    assert not (tmp_path / "wrong-bank.writing").exists()


def test_mix_corpus_refuses_post_storage_mass_drift_over_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir, shard_path, _arrays = _write_source(tmp_path)
    sidecar_dir = _write_sidecar(tmp_path, source_dir, shard_path)
    audit_receipt = _write_audit_receipt(tmp_path)
    out_dir = tmp_path / "mass-drift"
    monkeypatch.setattr(tool, "SELECTED_MASS_DRIFT_MEAN_MAX", -1.0)

    with pytest.raises(ValueError, match="selected-set mass drift"):
        tool.mix_corpus(
            argparse.Namespace(
                shards=source_dir,
                sidecar=sidecar_dir,
                out=out_dir,
                alpha=1.0,
                scope="top-max-ties",
                bt4_temperature=1.0,
                near_max_ratio=0.5,
                expected_rows=2,
                expected_shards=1,
                expected_source_summary_sha256=tool.file_sha256(
                    source_dir / tool.DERIVE_SUMMARY
                ),
                audit_receipt=audit_receipt,
            )
        )

    assert not out_dir.exists()
    assert out_dir.with_name(out_dir.name + ".writing").is_dir()


def test_mix_refuses_a_sidecar_after_source_identity_changes(tmp_path: Path) -> None:
    source_dir, shard_path, _arrays = _write_source(tmp_path)
    sidecar_dir = _write_sidecar(tmp_path, source_dir, shard_path)
    audit_receipt = _write_audit_receipt(tmp_path)
    source = zarr.open_group(str(shard_path), mode="a")
    changed = np.asarray(source["x"][0])
    changed[1, 0, 0] = 1.0
    source["x"][0] = changed
    out_dir = tmp_path / "mismatch"

    with pytest.raises(ValueError, match="provenance mismatch"):
        tool.mix_corpus(
            argparse.Namespace(
                shards=source_dir,
                sidecar=sidecar_dir,
                out=out_dir,
                alpha=1.0,
                scope="top-max-ties",
                bt4_temperature=1.0,
                near_max_ratio=0.5,
                expected_rows=2,
                expected_shards=1,
                expected_source_summary_sha256=tool.file_sha256(
                    source_dir / tool.DERIVE_SUMMARY
                ),
                audit_receipt=audit_receipt,
            )
        )

    assert not out_dir.exists()
    assert out_dir.with_name(out_dir.name + ".writing").is_dir()


def test_mix_refuses_non_float16_source_policy(tmp_path: Path) -> None:
    source_dir, shard_path, _arrays = _write_source(tmp_path)
    source = zarr.open_group(str(shard_path), mode="a")
    source_policy = np.asarray(source["policy_target"][:], dtype=np.float32)
    del source["policy_target"]
    source.create_dataset("policy_target", data=source_policy, chunks=(1, 1858))
    sidecar_dir = _write_sidecar(tmp_path, source_dir, shard_path)
    audit_receipt = _write_audit_receipt(tmp_path)

    with pytest.raises(ValueError, match="storage_dtype"):
        tool.mix_corpus(
            argparse.Namespace(
                shards=source_dir,
                sidecar=sidecar_dir,
                out=tmp_path / "wrong-dtype",
                alpha=1.0,
                scope="top-max-ties",
                bt4_temperature=1.0,
                near_max_ratio=0.5,
                expected_rows=2,
                expected_shards=1,
                expected_source_summary_sha256=tool.file_sha256(
                    source_dir / tool.DERIVE_SUMMARY
                ),
                audit_receipt=audit_receipt,
            )
        )


def test_label_banks_one_true_history_eval_per_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    shard_path = source_dir / "shard_000000.zarr"
    group = zarr.open_group(str(shard_path), mode="w")
    board = chess.Board()
    legal_indices = [compact_index_for_move(board, move) for move in board.legal_moves]
    legal_mask = np.zeros((2, 1858), dtype=np.uint8)
    legal_mask[:, legal_indices] = 1
    policy = legal_mask.astype(np.float16)
    policy /= policy.sum(axis=1, keepdims=True)
    arrays = {
        "x": np.zeros((2, 175, 8, 8), dtype=np.float16),
        "policy_target": policy,
        "legal_mask": legal_mask,
        "has_policy": np.ones(2, dtype=np.uint8),
        "has_legal_mask": np.ones(2, dtype=np.uint8),
        "wdl_target": np.asarray([1, 1], dtype=np.int8),
        "search_wdl": np.asarray([[0.2, 0.6, 0.2]] * 2, dtype=np.float16),
        "has_search_wdl": np.ones(2, dtype=np.uint8),
    }
    for name, value in arrays.items():
        group.create_dataset(name, data=value, chunks=(1, *value.shape[1:]))
    group.attrs["input_history_encoding"] = "lc0_root_legacy_meta"

    class FakeSession:
        def get_outputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="policy")]

        def run(
            self, names: list[str], feeds: dict[str, np.ndarray]
        ) -> list[np.ndarray]:
            assert names == ["policy"]
            batch = next(iter(feeds.values())).shape[0]
            return [np.zeros((batch, 1858), dtype=np.float32)]

    seen_planes: list[np.ndarray] = []

    def fake_planes(x: np.ndarray, *, input_history_encoding: str) -> np.ndarray:
        assert input_history_encoding == "lc0_root_legacy_meta"
        result = np.zeros((x.shape[0], 112, 8, 8), dtype=np.float32)
        seen_planes.append(result)
        return result

    monkeypatch.setattr(tool, "x_to_lc0_planes", fake_planes)
    monkeypatch.setattr(tool, "board_from_stored_x", lambda *_a, **_kw: board.copy())
    monkeypatch.setattr(
        tool,
        "open_session",
        lambda *_a, **_kw: (FakeSession(), "input", np.float32, ["fake"]),
    )
    monkeypatch.setattr(tool, "resolve_policy_output", lambda *_a, **_kw: 0)
    monkeypatch.setattr(tool, "file_sha256", lambda _path: "fake-onnx-sha")
    monkeypatch.setattr(tool, "remap_provenance", lambda: {"test": True})
    out_dir = tmp_path / "sidecar"

    result = tool.label_sidecar(
        argparse.Namespace(
            shards=source_dir,
            out=out_dir,
            onnx=tmp_path / "fake.onnx",
            batch_size=2,
            gpu_mem_gb=0.0,
            threads=1,
            policy_output=None,
            resume=False,
        )
    )

    assert result == 0
    assert len(seen_planes) == 1
    sidecar = zarr.open_group(str(out_dir / shard_path.name), mode="r")
    raw = np.asarray(sidecar[tool.SIDECAR_POLICY_FIELD][:])
    assert raw.shape == (2, 1858)
    assert np.all(raw[legal_mask == 0] == 0.0)
    assert np.allclose(raw.sum(axis=1), 1.0)
    assert sidecar.attrs["teacher_evaluations_per_position"] == 1
    summary = json.loads((out_dir / tool.SIDECAR_SUMMARY).read_text(encoding="utf-8"))
    assert summary["rows"] == 2
    assert summary["teacher_evaluations_per_position"] == 1
