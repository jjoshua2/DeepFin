from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

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


def _write_source(root: Path) -> tuple[Path, Path, dict[str, np.ndarray]]:
    source_dir = root / "source"
    shard_path = source_dir / "shard_000000.zarr"
    group = zarr.open_group(str(shard_path), mode="w")
    policy = np.zeros((2, 1858), dtype=np.float16)
    policy[0, :2] = [0.5, 0.5]
    policy[1, :2] = [0.75, 0.25]
    legal_mask = np.zeros((2, 1858), dtype=np.uint8)
    legal_mask[:, :2] = 1
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


def _write_audit_receipt(root: Path) -> Path:
    path = root / "audit.json"
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "bt4_policy_mix_frozen_deep_sf_audit",
                "treatment": {
                    "scope": "top-max-ties",
                    "alpha": 1.0,
                    "algorithm": tool.TREATMENT_ALGORITHMS["top-max-ties"],
                },
                "source_target_contract": tool.SOURCE_TARGET_CONTRACT,
                "ruler": tool.AUDIT_RULER_CONTRACT,
                "gate": {"training_permitted": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_audit_receipt_requires_exact_treatment_algorithm(tmp_path: Path) -> None:
    path = _write_audit_receipt(tmp_path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    del receipt["treatment"]["algorithm"]
    path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"treatment\.algorithm"):
        tool._load_audit_receipt(path, alpha=1.0, scope="top-max-ties")


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
        [0.1, 0.9, 0.0, 0.0],
        abs=1e-3,
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
    receipt = json.loads((out_dir / tool.MIX_SUMMARY).read_text(encoding="utf-8"))
    assert receipt["changed_rows"] == 1
    assert receipt["source_top_tied_rows"] == 1
    assert receipt["changed_unique_max_rows"] == 0
    assert receipt["value_columns_unchanged"] == ["wdl_target", "search_wdl"]
    assert receipt["top1_ge_0_99_fraction"] == {"source": 0.0, "mixed": 0.0}


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
