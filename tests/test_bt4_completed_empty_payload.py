"""Inert payload/NPZ regression, collected normally by hosted pytest.

The direct unittest entry point supports separately authorized isolated checks.
Only audited pure AST fragments execute; no encoding or game code executes.
"""
from __future__ import annotations

import argparse
import ast
import functools
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def tree(relative):
    return ast.parse((ROOT / relative).read_text())


def call_name(node):
    return ast.unparse(node.func)


def execute(nodes, namespace, allowed_calls):
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
                raise AssertionError("unexpected action in extracted fragment")
            if isinstance(child, ast.Call) and call_name(child) not in allowed_calls:
                raise AssertionError(f"unreviewed call: {call_name(child)}")
    fragment = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(fragment), "<audited-payload-fragment>", "exec"), namespace)


def literal(relative, name):
    for node in tree(relative).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing literal {name}")


def load_plane_count():
    features = tree("chess_anti_engine/encoding/features.py")
    names = {"EXTRA_FEATURE_VERSIONS", "_EXTRA_FEATURE_PLANES"}
    constants = [node for node in features.body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name)
        and (target.id.startswith("EXTRA_FEATURES_") or target.id in names)
        for target in node.targets
    )]
    functions = [node for node in features.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"extra_feature_plane_count", "normalize_extra_features_encoding"}]
    encoder = [node for node in tree("chess_anti_engine/encoding/encode.py").body
               if isinstance(node, ast.FunctionDef) and node.name == "input_plane_count"]
    assert len(functions) == 2
    assert len(encoder) == 1
    namespace = {"functools": functools}
    execute([*constants, *functions, *encoder], namespace, {
        "functools.lru_cache", "str", "str(value).strip", "str(value).strip().lower",
        "ValueError", "normalize_extra_features_encoding", "extra_feature_plane_count",
    })
    return namespace["input_plane_count"]


class Discarded(SimpleNamespace):
    pass


def inert_board(fen):
    assert fen == "inert-root"
    return SimpleNamespace(occupied=32)


def inert_popcount(occupied):
    assert occupied == 32
    return occupied


def load_payload():
    nodes = [node for node in tree("scripts/bt4_root_policy_worker.py").body
             if isinstance(node, ast.FunctionDef) and node.name == "_game_payload"]
    assert len(nodes) == 1
    namespace = {
        "np": np, "SCHEMA": literal("scripts/bt4_root_policy_worker.py", "SCHEMA"),
        "OUTCOME_MODE": literal("scripts/bt4_root_policy_worker.py", "OUTCOME_MODE"),
        "COMPACT_POLICY_SIZE": literal("chess_anti_engine/moves/encode.py", "COMPACT_POLICY_SIZE"),
        "INPUT_EXTRA_FEATURES": literal("scripts/gen_sf_rooted_corpus.py", "INPUT_EXTRA_FEATURES"),
        "input_plane_count": load_plane_count(), "BT4DiscardedGame": Discarded,
        "chess": SimpleNamespace(Board=inert_board, popcount=inert_popcount),
    }
    execute(nodes, namespace, {
        "ValueError", "isinstance", "vars", "enumerate", "chess.popcount", "chess.Board",
        "teacher.source_key.hex", "played.move.uci", "rows.append", "len",
        "np.stack", "np.empty", "input_plane_count",
    })
    return namespace["_game_payload"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def consumer_layout(arrays, n):
    # Execute the exact consumer layout assertions, not a weakened local ruler.
    nodes = []
    for node in ast.walk(tree("scripts/bt4_own_game_teacher_source.py")):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (isinstance(call.func, ast.Name) and call.func.id == "require" and len(call.args) == 2
                    and isinstance(call.args[1], ast.Constant) and call.args[1].value in {
                        "x layout differs", "policy layout differs", "WDL layout differs",
                    }):
                nodes.append(node)
    assert len(nodes) == 3
    execute(nodes, {"require": require, "np": np, "n": n,
                    "x": arrays["x"], "policy": arrays["policy_t1"], "wdl": arrays["wdl_raw"]},
            {"require", "np.dtype"})


def completed(records=(), result="1/2-1/2"):
    return SimpleNamespace(slot_id=7, result=result, termination="natural",
                           detail="inert terminal witness", records=records,
                           outcome_provenance=SimpleNamespace(mode="rule50_match_v1", marker="unchanged"))


def record(index):
    teacher = SimpleNamespace(
        fen="inert-root", input_key=f"key-{index}", source_key=bytes([index]),
        model_sha256="model", policy_output="policy", wdl_output="wdl", wdl_kind="probabilities",
        input_name="x", input_dtype="float32", input_history_encoding="lc0_root_legacy_meta",
        input_extra_features="v2_threats", history_rep_fix=True,
        policy_t1=np.full((1858,), 1 / 1858, dtype=np.float32),
        wdl_raw=np.array([0.25, 0.5, 0.25], dtype=np.float32),
    )
    played = SimpleNamespace(teacher=teacher, x=np.full((175, 8, 8), index, dtype=np.float32),
                             ply_index=index, pov_white=True, temperature=1.0,
                             move=SimpleNamespace(uci=lambda: "inert-move"))
    return SimpleNamespace(played=played, wdl_target=1)


def roundtrip(meta, arrays):
    with io.BytesIO() as buffer:
        np.savez_compressed(buffer, **arrays,
                            metadata=np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8))
        assert buffer.tell() < 32768
        buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as payload:
            assert set(payload.files) == {"x", "policy_t1", "wdl_raw", "metadata"}
            restored = {key: payload[key].copy() for key in arrays}
            restored_meta = json.loads(payload["metadata"].tobytes())
    assert restored_meta == meta
    for key, value in arrays.items():
        np.testing.assert_array_equal(restored[key], value, strict=True)
    return restored


class PayloadContract(unittest.TestCase):
    def setUp(self):
        self.payload = load_payload()

    def test_completed_empty_contract_and_serialization(self):
        for result in ("1-0", "0-1", "1/2-1/2"):
            with self.subTest(result=result):
                game = completed(result=result)
                meta, arrays = self.payload(game, initial_fen="inert-initial")
                assert meta == {
                    "schema": "bt4_root_policy_games_v1", "game_id": 7,
                    "initial_fen": "inert-initial", "status": "completed", "result": result,
                    "termination": "natural", "detail": "inert terminal witness",
                    "attempted_plies": 0, "discarded_rows": 0, "rows": [],
                    "outcome_provenance": vars(game.outcome_provenance),
                }
                print("EMPTY-WITNESS", result, arrays["x"].shape, arrays["x"].dtype, flush=True)
                restored = roundtrip(meta, arrays)
                consumer_layout(restored, 0)

    def test_nonempty_rows_unchanged(self):
        records = (record(0), record(1))
        meta, arrays = self.payload(completed(records), initial_fen="inert-initial")
        assert meta["status"] == "completed"
        assert meta["attempted_plies"] == 2
        assert meta["discarded_rows"] == 0
        assert [row["index"] for row in meta["rows"]] == [0, 1]
        assert [row["source_key"] for row in meta["rows"]] == ["00", "01"]
        assert [row["move_uci"] for row in meta["rows"]] == ["inert-move"] * 2
        assert [row["teacher"]["input_extra_features"] for row in meta["rows"]] == ["v2_threats"] * 2
        for key, expected in {
            "x": [row.played.x for row in records],
            "policy_t1": [row.played.teacher.policy_t1 for row in records],
            "wdl_raw": [row.played.teacher.wdl_raw for row in records],
        }.items():
            np.testing.assert_array_equal(arrays[key], np.stack(expected), strict=True)
        consumer_layout(roundtrip(meta, arrays), 2)

    def test_discarded_unfinished_shape_and_accounting_unchanged(self):
        game = Discarded(slot_id=8, termination="max_plies", detail="unfinished",
                         attempted_plies=2, discarded_rows=2,
                         outcome_provenance=SimpleNamespace(mode="rule50_match_v1"))
        meta, arrays = self.payload(game, initial_fen="inert-initial")
        assert meta["status"] == "discarded"
        assert meta["result"] is None
        assert meta["rows"] == []
        assert (meta["attempted_plies"], meta["discarded_rows"]) == (2, 2)
        assert arrays["x"].shape == (0, 0, 8, 8)
        assert arrays["policy_t1"].shape == (0, 1858)
        assert arrays["wdl_raw"].shape == (0, 3)
        assert all(value.dtype == np.float32 for value in arrays.values())
        roundtrip(meta, arrays)
        game.discarded_rows = 1
        with pytest.raises(ValueError, match="discard must account"):
            self.payload(game, initial_fen="inert-initial")

    def test_mode_rejection_preserved(self):
        game = completed()
        game.outcome_provenance.mode = "unsupported"
        with pytest.raises(ValueError, match="outcome mode differs"):
            self.payload(game, initial_fen="inert-initial")

    def test_consumer_rejects_wrong_empty_layouts(self):
        valid = {"x": np.empty((0, 175, 8, 8), np.float32),
                 "policy_t1": np.empty((0, 1858), np.float32), "wdl_raw": np.empty((0, 3), np.float32)}
        consumer_layout(valid, 0)
        for key, invalid in (
            ("x", np.empty((0, 0, 8, 8), np.float32)),
            ("x", np.empty((0, 175, 8, 8), np.float64)),
            ("policy_t1", np.empty((0, 0), np.float32)),
            ("wdl_raw", np.empty((0, 0), np.float32)),
        ):
            with (self.subTest(key=key, shape=invalid.shape, dtype=invalid.dtype),
                  pytest.raises(ValueError, match="layout differs")):
                consumer_layout({**valid, key: invalid}, 0)

    def test_bound_plane_count_helper_semantics(self):
        helper = load_plane_count()
        feature = literal("scripts/gen_sf_rooted_corpus.py", "INPUT_EXTRA_FEATURES")
        assert feature == "v2_threats"
        assert helper(feature) == 175
        assert helper("v1") == 146
        assert helper("v3_checks") == 179
        with pytest.raises(ValueError, match="unknown input_extra_features"):
            helper("invalid")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT)
    args, remaining = parser.parse_known_args()
    ROOT = args.source_root.resolve()
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
