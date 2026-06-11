"""Frozen deep-SF audit set: decode round-trip, scoring math, build/score flows.

The decode contract is side-to-move CANONICAL: every reconstructed board has
white to move; black-to-move originals come back as their color-mirrored
twin (chess is exactly symmetric under that transform, and the legacy input
encoding does not store absolute color at all).
"""
from __future__ import annotations

import json
import random

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard
from chess_anti_engine.eval.audit import (
    AuditPosition,
    decode_board_from_planes,
    expected_and_top1_regret,
    move_regrets,
    parse_audit_record,
    phase_bucket,
    position_key,
    wdl_brier,
    wdl_ece,
)


def _canonical_key(b: chess.Board) -> str:
    return position_key(b if b.turn == chess.WHITE else b.mirror())


@pytest.mark.parametrize("enc", ["legacy", "lc0_root_legacy_meta"])
def test_decode_round_trip_random_games(enc):
    rng = random.Random(7)
    checked = 0
    for _ in range(40):
        b = chess.Board()
        for _ in range(rng.randrange(0, 70)):
            moves = list(b.legal_moves)
            if not moves or b.is_game_over():
                break
            b.push(rng.choice(moves))
        x = encode_cboard(
            CBoard.from_board(b),
            input_history_encoding=enc, input_extra_features="v1",
        )
        dec = decode_board_from_planes(x, input_history_encoding=enc)
        assert dec is not None, b.fen()
        assert dec.turn == chess.WHITE
        expect = _canonical_key(b)
        got = position_key(dec)
        # An EP square without a legal capturer is legitimately dropped.
        if got != expect:
            assert got.rsplit(" ", 1)[0] == expect.rsplit(" ", 1)[0]
            assert got.endswith(" -")
        canon = b if b.turn == chess.WHITE else b.mirror()
        assert dec.halfmove_clock == min(canon.halfmove_clock, 100)
        checked += 1
    assert checked == 40


def test_decode_rejects_corrupt_planes():
    b = chess.Board()
    x = encode_cboard(
        CBoard.from_board(b),
        input_history_encoding="lc0_root_legacy_meta", input_extra_features="v1",
    ).copy()
    x[0, 0, 4] = 1.0  # pawn on top of the king square -> overlapping planes
    assert decode_board_from_planes(x, input_history_encoding="lc0_root_legacy_meta") is None


def test_phase_bucket_thresholds_match_probe():
    from scripts.probe_policy_targets import _PHASE_THRESHOLDS

    from chess_anti_engine.eval.audit import PHASE_THRESHOLDS

    assert tuple(PHASE_THRESHOLDS) == tuple(_PHASE_THRESHOLDS)
    assert [phase_bucket(n) for n in (2, 13, 14, 22, 23, 32)] == [0, 0, 1, 1, 2, 2]


def _audit_pos(move_cp: dict[str, float], wdl=(0.5, 0.3, 0.2)) -> AuditPosition:
    return AuditPosition(
        key="k", fen=chess.Board().fen(), phase=1, source=0,
        move_cp=move_cp, best_cp=max(move_cp.values()),
        deep_wdl=wdl, sf_nodes=1_000_000, sf_depth=30,
    )


def test_move_regret_goldens():
    pos = _audit_pos({"e2e4": 30.0, "d2d4": 25.0, "g1f3": -10.0})
    legal = ["e2e4", "d2d4", "g1f3", "a2a3"]  # a2a3 unlisted -> worst-line floor
    r = move_regrets(pos, legal)
    np.testing.assert_allclose(r, [0.0, 5.0, 40.0, 40.0])

    probs = np.array([0.5, 0.3, 0.1, 0.1])
    exp_r, top1 = expected_and_top1_regret(probs, r)
    assert exp_r == pytest.approx(0.5 * 0 + 0.3 * 5 + 0.1 * 40 + 0.1 * 40)
    assert top1 == 0.0
    # degenerate all-zero distribution falls back to uniform
    exp_u, _ = expected_and_top1_regret(np.zeros(4), r)
    assert exp_u == pytest.approx(r.mean())


def test_wdl_brier_and_ece_goldens():
    assert wdl_brier(np.array([1, 0, 0.0]), np.array([1, 0, 0.0])) == 0.0
    assert wdl_brier(np.array([1, 0, 0.0]), np.array([0, 0, 1.0])) == pytest.approx(2.0)
    # Perfectly confident + correct predictions -> |conf - acc| = 0
    preds = np.eye(3)[np.array([0, 1, 2, 0])].astype(float)
    assert wdl_ece(preds, preds) == pytest.approx(0.0)
    # Confident but always wrong -> ECE ~= 1
    wrong = np.eye(3)[np.array([1, 2, 0, 1])].astype(float)
    assert wdl_ece(preds, wrong) == pytest.approx(1.0)


def test_parse_audit_record_mate_and_dedup():
    rec = {
        "key": "k", "fen": chess.Board().fen(), "phase": 2, "source": 1,
        "multipv": [
            {"move": "e2e4", "cp": 30, "mate": None, "wdl": None},
            {"move": "d2d4", "cp": None, "mate": 3, "wdl": None},
            {"move": "e2e4", "cp": -999, "mate": None, "wdl": None},  # dup: first wins
        ],
        "wdl": [500.0, 300.0, 200.0],
        "nodes": 1_000_000, "depth": 35,
    }
    pos = parse_audit_record(json.dumps(rec))
    assert pos.move_cp["e2e4"] == 30.0
    assert pos.move_cp["d2d4"] > 1000.0  # mate folded to large effective cp
    assert pos.best_cp == pos.move_cp["d2d4"]
    np.testing.assert_allclose(pos.deep_wdl, (0.5, 0.3, 0.2))


# ---------------------------------------------------------------------------
# build_audit_set sampling + freeze semantics (no Stockfish required)
# ---------------------------------------------------------------------------


def _write_synthetic_shard(tmp_path, *, n_games: int = 6, plies: int = 24):
    from chess_anti_engine.replay.buffer import ReplaySample
    from chess_anti_engine.replay.shard import ShardMeta, samples_to_arrays, save_local_shard_arrays

    rng = random.Random(3)
    samples = []
    pol = np.zeros(4672, np.float32)
    pol[0] = 1.0
    for g in range(n_games):
        b = chess.Board()
        for _ in range(plies):
            moves = list(b.legal_moves)
            if not moves or b.is_game_over():
                break
            b.push(rng.choice(moves))
            x = encode_cboard(
                CBoard.from_board(b),
                input_history_encoding="lc0_root_legacy_meta",
                input_extra_features="v1",
            )
            samples.append(ReplaySample(
                x=x.astype(np.float32), policy_target=pol, wdl_target=1,
                is_selfplay=bool(g % 2),
            ))
    arrs = samples_to_arrays(samples)
    arrs["_input_history_encoding"] = np.array(["lc0_root_legacy_meta"])
    save_local_shard_arrays(
        tmp_path / "shard_000000.zarr", arrs=arrs, meta=ShardMeta(positions=len(samples)),
    )
    return len(samples)


def test_sample_manifest_stratifies_and_dedups(tmp_path):
    from scripts.build_audit_set import _sample_manifest

    _write_synthetic_shard(tmp_path)
    manifest = _sample_manifest([tmp_path], positions=12, seed=0)
    assert manifest
    keys = [row["key"] for row in manifest]
    assert len(keys) == len(set(keys))
    srcs = {row["source"] for row in manifest}
    assert srcs == {0, 1}
    for row in manifest:
        board = chess.Board(row["fen"])
        assert board.turn == chess.WHITE  # canonical POV
        assert row["phase"] == phase_bucket(len(board.piece_map()))


def test_build_refuses_overwrite_and_frozen(tmp_path, monkeypatch, capsys):
    import scripts.build_audit_set as bas

    out = tmp_path / "audit_set_v1.jsonl"
    out.write_text('{"key": "x"}\n', encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["build_audit_set.py", "--replay-dir", str(tmp_path), "--out", str(out),
         "--stockfish", "/nonexistent"],
    )
    with pytest.raises(SystemExit, match="FROZEN"):
        bas.main()

    readme = tmp_path / "audit_set_v1_README.md"
    readme.write_text("# Audit set\n- FROZEN: yes\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["build_audit_set.py", "--replay-dir", str(tmp_path), "--out", str(out),
         "--stockfish", "/nonexistent", "--resume"],
    )
    with pytest.raises(SystemExit, match="FROZEN"):
        bas.main()
    del capsys


def test_build_labels_resume_skips_done(tmp_path, monkeypatch):
    """Resumable labeling appends only missing keys; a complete set freezes."""
    import scripts.build_audit_set as bas

    _write_synthetic_shard(tmp_path)
    out = tmp_path / "audit_set_v1.jsonl"

    calls: list[str] = []

    def fake_label(eng, row, *, nodes):
        del eng, nodes
        calls.append(row["key"])
        return {**row, "multipv": [{"move": "e2e4", "cp": 10, "mate": None, "wdl": None}],
                "wdl": [400.0, 300.0, 300.0], "bestmove": "e2e4",
                "nodes": 123, "depth": 9}

    class _FakeEngine:
        def close(self):
            pass

    monkeypatch.setattr(bas, "_label_one", fake_label)
    monkeypatch.setattr(bas, "StockfishUCI", lambda *a, **k: _FakeEngine())
    argv = ["build_audit_set.py", "--replay-dir", str(tmp_path), "--out", str(out),
            "--stockfish", "fake", "--positions", "8", "--sf-workers", "2"]

    # First pass: build the manifest, then simulate a mid-run kill by
    # pre-labeling a strict subset of the manifest before running.
    monkeypatch.setattr("sys.argv", argv)
    bas.main()
    n_total = len(out.read_text(encoding="utf-8").splitlines())
    assert n_total > 1
    assert len(calls) == n_total

    manifest_path = tmp_path / "audit_set_v1.jsonl.manifest.jsonl"
    assert manifest_path.exists()
    lines = out.read_text(encoding="utf-8").splitlines()
    out.write_text("\n".join(lines[: n_total // 2]) + "\n", encoding="utf-8")
    (tmp_path / "audit_set_v1_README.md").write_text("FROZEN: no\n", encoding="utf-8")

    calls.clear()
    monkeypatch.setattr("sys.argv", [*argv, "--resume"])
    bas.main()
    # Only the missing half was relabeled, and the file is complete again.
    assert len(calls) == n_total - n_total // 2
    assert len(out.read_text(encoding="utf-8").splitlines()) == n_total
    readme = (tmp_path / "audit_set_v1_README.md").read_text(encoding="utf-8")
    assert "FROZEN: yes" in readme

    # A COMPLETE (frozen) set refuses even --resume.
    monkeypatch.setattr("sys.argv", [*argv, "--resume"])
    with pytest.raises(SystemExit, match="FROZEN"):
        bas.main()

    for line in out.read_text(encoding="utf-8").splitlines():
        parse_audit_record(line)


# ---------------------------------------------------------------------------
# audit_targets end-to-end smoke on a synthetic audit set (CPU, tiny model)
# ---------------------------------------------------------------------------


def test_audit_targets_smoke(tmp_path, monkeypatch):
    import torch

    from chess_anti_engine.model import ModelConfig, build_model
    import scripts.audit_targets as at

    # Synthetic audit set: 3 positions with deep labels.
    rng = random.Random(1)
    rows = []
    shallow_rows = []
    for i in range(3):
        b = chess.Board()
        for _ in range(2 * i):
            b.push(rng.choice(list(b.legal_moves)))
        if b.turn == chess.BLACK:
            b = b.mirror()
        legal = [m.uci() for m in b.legal_moves]
        rows.append({
            "key": position_key(b), "fen": b.fen(), "phase": 2, "source": i % 2,
            "multipv": [
                {"move": u, "cp": 50 - 10 * j, "mate": None, "wdl": None}
                for j, u in enumerate(legal[:10])
            ],
            "wdl": [400.0, 400.0, 200.0], "bestmove": legal[0],
            "nodes": 1_000_000, "depth": 30,
        })
        shallow_rows.append({
            "key": position_key(b), "nodes_requested": 500, "multipv": 4,
            "cp": 25, "mate": None, "wdl": [380.0, 420.0, 200.0],
            "pvs": [
                {"move": u, "cp": 25 - 5 * j, "mate": None, "wdl": [380.0, 420.0, 200.0]}
                for j, u in enumerate(legal[:4])
            ],
        })
    audit = tmp_path / "audit.jsonl"
    audit.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (tmp_path / "audit.jsonl.shallow_sf.jsonl").write_text(
        "\n".join(json.dumps(r) for r in shallow_rows) + "\n", encoding="utf-8",
    )

    torch.manual_seed(0)
    model = build_model(ModelConfig(embed_dim=32, num_layers=1, num_heads=2, use_smolgen=False))
    import dataclasses

    from chess_anti_engine.model import ARCH_SCHEMA_VERSION

    ckpt = tmp_path / "model.pt"
    torch.save({
        "model": model.state_dict(),
        "arch": {"_schema_version": ARCH_SCHEMA_VERSION,
                 **dataclasses.asdict(ModelConfig(
                     embed_dim=32, num_layers=1, num_heads=2, use_smolgen=False))},
    }, ckpt)

    out_dir = tmp_path / "runs"
    monkeypatch.setattr(
        "sys.argv",
        ["audit_targets.py", "--audit-set", str(audit), "--checkpoint", str(ckpt),
         "--device", "cpu", "--sims", "4", "--batch-size", "2",
         "--sf-soft-nodes", "500", "--sf-soft-multipv", "4",
         "--out-dir", str(out_dir)],
    )
    at.main()
    reports = list(out_dir.glob("target_audit_*.md"))
    assert len(reports) == 1
    text = reports[0].read_text(encoding="utf-8")
    for label in ("net raw policy", "Gumbel search", "SF MultiPV soft target",
                  "production training target", "Brier vs deep WDL"):
        assert label in text
    # Every regret cell is finite and non-negative by construction; spot-check
    # the table parsed numbers exist.
    assert "cp" in text
