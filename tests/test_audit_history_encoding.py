"""audit-v2 input encoding: default preservation, and the stored path taking effect.

Two properties carry this feature, and each is written so that BREAKING IT MAKES
A TEST FAIL rather than making a number quietly move:

1. `fen_only` is the default and is bit-identical to the pre-audit-v2 ruler.
   The rulers' encoding hook must be the IDENTITY on that branch — proven
   against `encode_cboard` directly, which is literally what the old code
   called. Splice anything into the fen_only branch and
   `test_fen_only_root_is_identity` / `test_fen_only_child_is_identity` /
   `test_value_regret_fen_only_feeds_canonical_encodings` fail.

2. `stored` actually reaches the net. It is not enough that the helper
   returns different planes: the ruler has to feed them. The
   `test_value_regret_stored_*` pair captures every array the evaluator was
   handed inside `value_1ply_regret` and checks it against the transform.
   Neuter the stored branch and they fail.

The `pov_flip_slot` bit-exactness is pinned against REAL production rows via a
checked-in fixture (16 consecutive-ply pairs spanning parent plies 4..390,
emitted by `scripts/verify_audit_history_transform.py`). Tests never read
`runs/`. The full-snapshot run behind the fixture: 396,733 pairs,
2,777,131/2,777,131 piece-history slots and 396,733/396,733 colour flags exact.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard
from chess_anti_engine.eval.audit_history import (
    N_SLOTS,
    STM_PLANE,
    STORED_EXTRA_FEATURES,
    STORED_HISTORY_ENCODING,
    STORED_PLANES,
    child_input_planes,
    child_planes_for_encoding,
    normalize_input_encoding,
    pov_flip_slot,
    root_input_planes,
    slot_bounds,
    verify_child_transform,
)

FIXTURE = Path(__file__).parent / "data" / "audit_history_pairs.npz"

ENC_KWARGS = {
    "input_history_encoding": STORED_HISTORY_ENCODING,
    "input_extra_features": STORED_EXTRA_FEATURES,
}

FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "8/8/4k3/8/2R5/4K3/8/8 w - - 0 1",
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
]

# Audit boards are ALWAYS white-to-move canonical, so every 1-ply child of one
# is black to move and carries colour flag 1.0. A fen_only branch that forced
# plane 108 to 1.0 would therefore be invisible on the audit set alone — the
# identity tests must not be able to miss it, so they also run on children that
# come out white to move.
BLACK_TO_MOVE_FENS = [
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 5 4",
]
IDENTITY_FENS = [*FENS, *BLACK_TO_MOVE_FENS]


def _fixture_pairs() -> list[tuple[np.ndarray, np.ndarray]]:
    data = np.load(FIXTURE, allow_pickle=False)
    parents = np.asarray(data["parent"], dtype=np.float32)
    children = np.asarray(data["child"], dtype=np.float32)
    assert parents.shape[1:] == (STORED_PLANES, 8, 8)
    return list(zip(parents, children, strict=True))


# ---------------------------------------------------------------------------
# 1. the default must not move
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fen", IDENTITY_FENS)
def test_fen_only_root_is_identity(fen: str) -> None:
    """The fen_only root hook returns exactly what the old code fed the net."""
    canonical = encode_cboard(CBoard.from_board(chess.Board(fen)), **ENC_KWARGS)
    stored = np.arange(STORED_PLANES * 64, dtype=np.float32).reshape(
        STORED_PLANES, 8, 8
    )
    got = root_input_planes(
        encoding="fen_only", fen_planes=canonical, stored_planes=stored,
    )
    # Bit-identical, and identical even with a stored row available: the
    # default must not be reachable by the stored machinery at all.
    assert np.array_equal(got, canonical)


@pytest.mark.parametrize("fen", IDENTITY_FENS)
def test_fen_only_child_is_identity(fen: str) -> None:
    board = chess.Board(fen)
    move = next(iter(board.legal_moves))
    board.push(move)
    canonical = encode_cboard(CBoard.from_board(board), **ENC_KWARGS)
    parent, _child = _fixture_pairs()[0]
    got = child_planes_for_encoding(
        encoding="fen_only", child_fen_planes=canonical, stored_parent=parent,
    )
    assert np.array_equal(got, canonical)
    # Name the planes explicitly: array_equal alone would still pass if a
    # future edit spliced a plane whose value happened to match on these
    # boards, which is exactly how a colour-flag mutation slipped through the
    # first version of this test.
    for plane in (STM_PLANE, 12, 13, 103):
        assert np.array_equal(got[plane], canonical[plane]), plane


def test_fen_only_is_the_default() -> None:
    assert normalize_input_encoding(None) == "fen_only"
    assert normalize_input_encoding("") == "fen_only"
    with pytest.raises(ValueError, match="unknown input encoding"):
        normalize_input_encoding("v2")


def test_fen_only_needs_no_stored_row() -> None:
    """A fen_only run must never require the audit-v2 machinery to exist."""
    canonical = encode_cboard(CBoard.from_board(chess.Board()), **ENC_KWARGS)
    assert np.array_equal(
        root_input_planes(
            encoding="fen_only", fen_planes=canonical, stored_planes=None,
        ),
        canonical,
    )
    with pytest.raises(ValueError, match="needs the matched stored row"):
        root_input_planes(
            encoding="stored", fen_planes=canonical, stored_planes=None,
        )


# ---------------------------------------------------------------------------
# 2. pov_flip_slot, against REAL stored rows
# ---------------------------------------------------------------------------


def test_pov_flip_slot_bit_exact_on_real_consecutive_plies() -> None:
    pairs = _fixture_pairs()
    assert len(pairs) >= 8
    report = verify_child_transform(pairs)
    assert report["piece_slots_exact"] == report["piece_slots_total"]
    assert report["piece_slots_total"] == len(pairs) * (N_SLOTS - 1)
    assert report["stm_flag_exact"] == report["stm_flag_total"] == len(pairs)


def test_pov_flip_slot_negative_control() -> None:
    """A transform that does NOT flip must FAIL the same check.

    Without this, `verify_child_transform` passing would be evidence only that
    the rows are comparable — a no-op shift would score just as well on slots
    that happen to be symmetric.
    """
    pairs = _fixture_pairs()
    unflipped_hits = 0
    for parent, child in pairs:
        for k in range(N_SLOTS - 1):
            src_lo, src_hi = slot_bounds(k)
            dst_lo, dst_hi = slot_bounds(k + 1)
            if np.array_equal(parent[src_lo:src_hi][:12], child[dst_lo:dst_hi][:12]):
                unflipped_hits += 1
    total = len(pairs) * (N_SLOTS - 1)
    # Measured on the checked-in fixture: 3/112. `< total` would leave ~97%
    # slack and still pass on a fixture degenerate enough to be useless, so
    # pin the DISCRIMINATION this test is named for, not merely its sign.
    assert unflipped_hits < total // 4, (
        f"the un-flipped shift matched {unflipped_hits}/{total} slots; the "
        "fixture cannot discriminate a POV flip from a plain shift"
    )


def test_pov_flip_slot_is_an_involution() -> None:
    parent, _child = _fixture_pairs()[0]
    lo, hi = slot_bounds(0)
    slot = parent[lo:hi]
    assert np.array_equal(pov_flip_slot(pov_flip_slot(slot)), slot)


def test_child_input_planes_only_touches_history_and_stm() -> None:
    parent, _child = _fixture_pairs()[0]
    board = chess.Board(FENS[1])
    board.push(next(iter(board.legal_moves)))
    child_fen = np.asarray(
        encode_cboard(CBoard.from_board(board), **ENC_KWARGS), dtype=np.float32,
    )
    spliced = child_input_planes(parent, child_fen)
    untouched = [p for p in range(STORED_PLANES) if p < 13 or (104 <= p != STM_PLANE)]
    assert np.array_equal(spliced[untouched], child_fen[untouched])
    # And planes 13..103 came from the parent, not from the child's own encode.
    assert not np.array_equal(spliced[13:104], child_fen[13:104])


# ---------------------------------------------------------------------------
# 3. the stored branch takes effect ON THE RULER'S REAL PATH
# ---------------------------------------------------------------------------


class _RecordingEvaluator:
    """Captures every encoded batch handed to the net inside the ruler."""

    def __init__(self) -> None:
        self.seen: list[np.ndarray] = []

    def evaluate_encoded(
        self, xs: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        _ = relations
        arr = np.asarray(xs, dtype=np.float32)
        self.seen.append(arr.copy())
        n = arr.shape[0]
        # Deterministic, position-dependent WDL so argmax is stable.
        w = np.linspace(0.0, 1.0, n, dtype=np.float64)
        return np.zeros((n, 1858), dtype=np.float32), np.stack(
            [w, 1.0 - w, np.zeros(n)], axis=1,
        )


class _FakeMatched:
    def __init__(self, key: str, row: np.ndarray) -> None:
        self._key, self._row = key, row

    def __contains__(self, key: object) -> bool:
        return str(key) == self._key

    def stored_row(self, key: str) -> np.ndarray:
        assert key == self._key
        return self._row

    def require_model_compatible(self, enc_kwargs: dict[str, str]) -> None:
        assert enc_kwargs == ENC_KWARGS


def _run_value_regret(
    monkeypatch: pytest.MonkeyPatch,
    input_encoding: str = "fen_only",
    matched_rows: Any = None,
) -> tuple[_RecordingEvaluator, chess.Board]:
    """Drive scripts/value_regret.value_1ply_regret with a stubbed model."""
    import scripts.value_regret as vr

    import chess_anti_engine.uci.model_loader as model_loader
    from chess_anti_engine.eval.audit import AuditPosition
    from scripts.net_source import NetSource

    class _Model:
        input_history_encoding = STORED_HISTORY_ENCODING
        input_extra_features = STORED_EXTRA_FEATURES
        use_dynamic_relations = False

        def eval(self) -> None:
            return None

    evaluator = _RecordingEvaluator()
    monkeypatch.setattr(model_loader, "load_model_from_checkpoint",
                        lambda *a, **k: _Model())
    monkeypatch.setattr(vr, "LocalModelEvaluator", lambda *a, **k: evaluator)

    board = chess.Board(FENS[1])
    pos = AuditPosition(
        key="k", fen=board.fen(), phase=1, source=0,
        move_cp={m.uci(): 0.0 for m in board.legal_moves},
        best_cp=0.0, deep_wdl=(0.3, 0.4, 0.3), sf_nodes=1, sf_depth=1,
    )
    vr.value_1ply_regret(
        net=NetSource(checkpoint="unused"), positions=[pos], device="cpu",
        batch_size=64, pos_chunk=8,
        input_encoding=input_encoding, matched_rows=matched_rows,
    )
    return evaluator, board


def test_value_regret_fen_only_feeds_canonical_encodings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The DEFAULT path feeds exactly `encode_cboard` of each pushed child."""
    evaluator, board = _run_value_regret(monkeypatch)
    fed = np.concatenate(evaluator.seen, axis=0)
    want = []
    for move in board.legal_moves:
        board.push(move)
        if not board.is_game_over():
            want.append(encode_cboard(CBoard.from_board(board), **ENC_KWARGS))
        board.pop()
    assert fed.shape[0] == len(want)
    assert np.array_equal(fed, np.stack(want).astype(np.float32))


def test_value_regret_stored_feeds_spliced_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--input-encoding stored` reaches the net, and with the RIGHT planes."""
    parent, _child = _fixture_pairs()[0]
    evaluator, board = _run_value_regret(
        monkeypatch,
        input_encoding="stored",
        matched_rows=_FakeMatched("k", parent),
    )
    fed = np.concatenate(evaluator.seen, axis=0)
    want = []
    for move in board.legal_moves:
        board.push(move)
        if not board.is_game_over():
            want.append(child_input_planes(
                parent,
                np.asarray(
                    encode_cboard(CBoard.from_board(board), **ENC_KWARGS),
                    dtype=np.float32,
                ),
            ))
        board.pop()
    assert fed.shape[0] == len(want)
    assert np.array_equal(fed, np.stack(want))
    # The history block really is populated — a stored branch that silently
    # fell back to the FEN-only encoding would leave planes 13..103 at zero.
    assert np.abs(fed[:, 13:104]).max() > 0.0


def test_value_regret_stored_differs_from_fen_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two encodings are not accidentally the same object.

    A stored branch that was wired up but never took effect would make this
    pass trivially, so it is asserted alongside the exact-planes test above,
    not instead of it.
    """
    parent, _child = _fixture_pairs()[0]
    fen_ev, _ = _run_value_regret(monkeypatch)
    stored_ev, _ = _run_value_regret(
        monkeypatch, input_encoding="stored",
        matched_rows=_FakeMatched("k", parent),
    )
    fen_fed = np.concatenate(fen_ev.seen, axis=0)
    stored_fed = np.concatenate(stored_ev.seen, axis=0)
    assert fen_fed.shape == stored_fed.shape
    assert not np.array_equal(fen_fed, stored_fed)
    # ...and it differs exactly where audit-v2 says it should.
    differing = np.flatnonzero((fen_fed != stored_fed).any(axis=(0, 2, 3)))
    assert set(differing.tolist()) <= set(range(13, 104)) | {STM_PLANE}


def test_value_regret_stored_requires_the_index() -> None:
    import scripts.value_regret as vr

    from chess_anti_engine.eval.audit import AuditPosition
    from scripts.net_source import NetSource

    pos = AuditPosition(
        key="k", fen=chess.STARTING_FEN, phase=2, source=0,
        move_cp={"e2e4": 0.0}, best_cp=0.0, deep_wdl=(0.3, 0.4, 0.3),
        sf_nodes=1, sf_depth=1,
    )
    with pytest.raises(ValueError, match="matched_rows"):
        vr.value_1ply_regret(
            net=NetSource(checkpoint="unused"), positions=[pos], device="cpu",
            batch_size=8, pos_chunk=8, input_encoding="stored",
        )


def test_matched_rows_exposes_source_shard_for_cluster_identity(tmp_path: Path) -> None:
    from chess_anti_engine.eval.audit_history import MatchedAuditRows

    path = tmp_path / "matched.npz"
    np.savez(
        path,
        x_stored=np.zeros((1, 175, 8, 8), dtype=np.float16),
        game_id=np.asarray([7], dtype=np.int64),
        has_game_id=np.asarray([1], dtype=np.uint8),
        source_cluster_ambiguous=np.asarray([0], dtype=np.uint8),
        found=np.asarray([True]),
        key=np.asarray(["k"]),
        src_shard=np.asarray(["shard_000123.zarr"]),
    )

    matched = MatchedAuditRows(path)

    assert matched.game_id("k") == 7
    assert matched.has_game_id("k")
    assert matched.source_shard("k") == "shard_000123.zarr"
    assert matched.source_cluster_is_unique("k")


def test_matched_rows_reports_ambiguous_source_cluster(tmp_path: Path) -> None:
    from chess_anti_engine.eval.audit_history import MatchedAuditRows

    path = tmp_path / "matched.npz"
    np.savez(
        path,
        x_stored=np.zeros((1, 175, 8, 8), dtype=np.float16),
        game_id=np.asarray([7], dtype=np.int64),
        has_game_id=np.asarray([1], dtype=np.uint8),
        source_cluster_ambiguous=np.asarray([1], dtype=np.uint8),
        found=np.asarray([True]),
        key=np.asarray(["k"]),
        src_shard=np.asarray(["shard_000123.zarr"]),
    )

    matched = MatchedAuditRows(path)

    matched.require_index_layout(require_game_ids=True)
    assert not matched.source_cluster_is_unique("k")


def test_matched_rows_ignores_zero_filled_game_id_when_presence_is_false(
    tmp_path: Path,
) -> None:
    from chess_anti_engine.eval.audit_history import MatchedAuditRows

    path = tmp_path / "matched.npz"
    np.savez(
        path,
        x_stored=np.zeros((1, STORED_PLANES, 8, 8), dtype=np.float16),
        game_id=np.asarray([0], dtype=np.int64),
        has_game_id=np.asarray([0], dtype=np.uint8),
        source_cluster_ambiguous=np.asarray([0], dtype=np.uint8),
        found=np.asarray([True]),
        key=np.asarray(["k"]),
    )

    matched = MatchedAuditRows(path)

    assert not matched.has_game_id("k")
    assert matched.game_id("k") is None
    matched.require_index_layout(require_game_ids=True)


def test_match_audit_rows_honours_source_game_id_presence_mask() -> None:
    from scripts.match_audit_rows import _game_ids_and_presence

    values, present = _game_ids_and_presence(
        {
            "game_id": np.asarray([0, 41], dtype=np.int64),
            "has_game_id": np.asarray([0, 1], dtype=np.uint8),
        },
        source=Path("source.zarr"),
    )

    assert values is not None
    assert present is not None
    assert values.tolist() == [0, 41]
    assert present.tolist() == [False, True]


def test_match_audit_rows_preserves_legacy_unmasked_game_ids() -> None:
    from scripts.match_audit_rows import _game_ids_and_presence

    values, present = _game_ids_and_presence(
        {"game_id": np.asarray([0, 41], dtype=np.int64)},
        source=Path("legacy.zarr"),
    )

    assert values is not None
    assert present is not None
    assert present.tolist() == [True, True]


def test_match_audit_rows_unifies_the_same_game_across_shards() -> None:
    from scripts.match_audit_rows import _source_game_cluster_is_ambiguous

    assert not _source_game_cluster_is_ambiguous(
        existing_has_game=True,
        existing_game=41,
        candidate_has_game=True,
        candidate_game=41,
    )
    assert _source_game_cluster_is_ambiguous(
        existing_has_game=True,
        existing_game=41,
        candidate_has_game=True,
        candidate_game=42,
    )
    assert _source_game_cluster_is_ambiguous(
        existing_has_game=True,
        existing_game=41,
        candidate_has_game=False,
        candidate_game=-1,
    )


def _matched_origin_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    claimed_game_id: int = 7,
    claimed_source_row: int = 0,
    claimed_duplicate_count: int = 1,
    claimed_ambiguous: bool = False,
    source_game_ids: tuple[int, ...] = (7,),
) -> tuple[Path, Path, Path, list[dict[str, str]]]:
    from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard
    from chess_anti_engine.eval.audit import position_key
    from scripts import match_audit_rows as matcher

    board = chess.Board()
    key = position_key(board)
    stored = encode_cboard(
        CBoard.from_board(board),
        input_history_encoding=STORED_HISTORY_ENCODING,
        input_extra_features=STORED_EXTRA_FEATURES,
    ).astype(np.float16)
    audit = tmp_path / "audit.jsonl"
    audit.write_text(json.dumps({"key": key, "fen": board.fen()}) + "\n")
    snapshot = tmp_path / "snapshot"
    shard = snapshot / "s0.zarr"
    shard.mkdir(parents=True)
    matched = tmp_path / "matched.npz"
    np.savez(
        matched,
        x_stored=stored[None],
        found=np.asarray([True]),
        key=np.asarray([key]),
        src_shard=np.asarray([shard.name]),
        src_row=np.asarray([claimed_source_row], dtype=np.int64),
        game_id=np.asarray([claimed_game_id], dtype=np.int64),
        has_game_id=np.asarray([1], dtype=np.uint8),
        source_cluster_ambiguous=np.asarray([claimed_ambiguous], dtype=np.uint8),
        dup_count=np.asarray([claimed_duplicate_count], dtype=np.int64),
        snapshot=np.asarray([str(snapshot.resolve())]),
        audit_set=np.asarray([str(audit.resolve())]),
        input_history_encoding=np.asarray([STORED_HISTORY_ENCODING]),
        input_extra_features=np.asarray([STORED_EXTRA_FEATURES]),
    )

    def fake_load(path: Path, *, lazy: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
        assert lazy
        assert Path(path) == shard
        return {
            "x": np.stack([stored] * len(source_game_ids)),
            "game_id": np.asarray(source_game_ids, dtype=np.int64),
            "has_game_id": np.ones(len(source_game_ids), dtype=np.uint8),
            "_input_history_encoding": np.asarray([STORED_HISTORY_ENCODING]),
        }, {}

    monkeypatch.setattr(matcher, "load_shard_arrays", fake_load)
    monkeypatch.setattr(matcher, "_report_builder_is_bound", lambda _report: True)
    report = matcher.default_matched_rows_report_path(matched)
    row_origins = [{
        "key": key,
        "shard": shard.name,
        "row": claimed_source_row,
        "stored_x_sha256": matcher._row_sha256(stored),
        "has_game_id": True,
        "game_id": claimed_game_id,
        "source_cluster_ambiguous": claimed_ambiguous,
        "duplicate_count": claimed_duplicate_count,
    }]
    row_origins_document = json.dumps(
        row_origins, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
    ).encode()
    report.write_text(json.dumps({
        "schema": matcher.MATCHED_ROWS_REPORT_SCHEMA,
        "complete": True,
        "verification_passed": True,
        "builder": {},
        "audit_set": matcher._file_artifact(audit),
        "output": matcher._file_artifact(matched),
        "snapshot_inventory": matcher.snapshot_inventory(snapshot),
        "row_origin_count": 1,
        "row_origins_sha256": matcher._sha256_bytes(row_origins_document),
        "row_origins": row_origins,
    }))
    return audit, matched, report, [{"key": key, "fen": board.fen()}]


def test_selected_origin_verification_requires_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.match_audit_rows import verify_selected_row_origins

    audit, matched, report, selected = _matched_origin_fixture(tmp_path, monkeypatch)
    report.unlink()
    with pytest.raises(SystemExit, match="regular file"):
        verify_selected_row_origins(
            audit_set=audit, matched_rows=matched, report_path=report, selected=selected,
        )


def test_selected_origin_verification_rejects_forged_report_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.match_audit_rows import verify_selected_row_origins

    audit, matched, report, selected = _matched_origin_fixture(tmp_path, monkeypatch)
    payload = json.loads(report.read_text())
    payload["output"]["sha256"] = "0" * 64
    report.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="does not bind the consumed NPZ"):
        verify_selected_row_origins(
            audit_set=audit, matched_rows=matched, report_path=report, selected=selected,
        )


def test_selected_origin_verification_rejects_fabricated_unique_game_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.match_audit_rows import verify_selected_row_origins

    audit, matched, report, selected = _matched_origin_fixture(
        tmp_path, monkeypatch, claimed_game_id=99,
    )
    with pytest.raises(SystemExit, match="game_id"):
        verify_selected_row_origins(
            audit_set=audit, matched_rows=matched, report_path=report, selected=selected,
        )


def test_selected_origin_verification_rejects_one_field_origin_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.match_audit_rows import verify_selected_row_origins

    audit, matched, report, selected = _matched_origin_fixture(
        tmp_path, monkeypatch, claimed_source_row=1,
    )
    with pytest.raises(SystemExit, match="src_row"):
        verify_selected_row_origins(
            audit_set=audit, matched_rows=matched, report_path=report, selected=selected,
        )


def test_selected_origin_verification_recomputes_duplicate_cluster_ambiguity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.match_audit_rows import verify_selected_row_origins

    audit, matched, report, selected = _matched_origin_fixture(
        tmp_path,
        monkeypatch,
        claimed_duplicate_count=2,
        claimed_ambiguous=False,
        source_game_ids=(7, 8),
    )
    with pytest.raises(SystemExit, match="source_cluster_ambiguous"):
        verify_selected_row_origins(
            audit_set=audit, matched_rows=matched, report_path=report, selected=selected,
        )


def test_matched_rows_requires_presence_mask_for_decision_grade_only(
    tmp_path: Path,
) -> None:
    from chess_anti_engine.eval.audit_history import MatchedAuditRows

    path = tmp_path / "legacy.npz"
    np.savez(
        path,
        x_stored=np.zeros((1, STORED_PLANES, 8, 8), dtype=np.float16),
        game_id=np.asarray([-1], dtype=np.int64),
        found=np.asarray([True]),
        key=np.asarray(["k"]),
    )

    matched = MatchedAuditRows(path)

    # Legacy audit-v2 rulers remain reproducible: their score never used this
    # optional clustering identity.
    matched.require_index_layout()
    assert matched.has_game_id("k")
    assert matched.game_id("k") == -1
    with pytest.raises(SystemExit, match="has_game_id presence mask"):
        matched.require_index_layout(require_game_ids=True)


@pytest.mark.parametrize(
    ("mask", "ambiguity", "message"),
    [
        (
            np.asarray([[1]], dtype=np.uint8),
            np.asarray([0], dtype=np.uint8),
            "malformed has_game_id shape",
        ),
        (
            np.asarray([2], dtype=np.uint8),
            np.asarray([0], dtype=np.uint8),
            "non-binary has_game_id",
        ),
        (
            np.asarray([1], dtype=np.uint8),
            np.asarray([[0]], dtype=np.uint8),
            "malformed source_cluster_ambiguous shape",
        ),
        (
            np.asarray([1], dtype=np.uint8),
            np.asarray([2], dtype=np.uint8),
            "non-binary source_cluster_ambiguous",
        ),
    ],
)
def test_matched_rows_validates_decision_grade_game_presence_mask(
    tmp_path: Path, mask: np.ndarray, ambiguity: np.ndarray, message: str,
) -> None:
    from chess_anti_engine.eval.audit_history import MatchedAuditRows

    path = tmp_path / "matched.npz"
    np.savez(
        path,
        x_stored=np.zeros((1, STORED_PLANES, 8, 8), dtype=np.float16),
        game_id=np.asarray([7], dtype=np.int64),
        has_game_id=mask,
        source_cluster_ambiguous=ambiguity,
        found=np.asarray([True]),
        key=np.asarray(["k"]),
    )

    with pytest.raises(SystemExit, match=message):
        MatchedAuditRows(path).require_index_layout(require_game_ids=True)


# ---------------------------------------------------------------------------
# 4. audit_targets: row (a) is the only row the flag moves
# ---------------------------------------------------------------------------


def _run_net_candidates(
    monkeypatch: pytest.MonkeyPatch, stored_x: np.ndarray | None,
) -> _RecordingEvaluator:
    """Drive audit_targets._net_candidates with a stubbed model and search."""
    import scripts.audit_targets as at

    import chess_anti_engine.inference as inference
    import chess_anti_engine.mcts.gumbel_c as gumbel_c
    import chess_anti_engine.uci.model_loader as model_loader
    from scripts.net_source import NetSource

    class _Model:
        input_history_encoding = STORED_HISTORY_ENCODING
        input_extra_features = STORED_EXTRA_FEATURES
        policy_encoding = "lc0_1858"
        use_dynamic_relations = False

        def eval(self) -> None:
            return None

    evaluator = _RecordingEvaluator()
    monkeypatch.setattr(model_loader, "load_model_from_checkpoint",
                        lambda *a, **k: _Model())
    monkeypatch.setattr(inference, "LocalModelEvaluator", lambda *a, **k: evaluator)

    def _fake_search(*, boards: list[chess.Board], **_kw: object) -> tuple[object, ...]:
        n = len(boards)
        probs = np.zeros((n, 1858), dtype=np.float64)
        probs[:, 0] = 1.0
        return probs, None, np.zeros(n, dtype=np.float64), None, None, None

    monkeypatch.setattr(gumbel_c, "run_gumbel_root_many_c", _fake_search)

    profile = at.build_search_profiles({}, play_sims=1, play_topk=2)["search"]
    at._net_candidates(
        [chess.Board(f) for f in FENS], net=NetSource(checkpoint="unused"),
        device="cpu",
        batch_size=4, seed=0, profiles={"search": profile}, stored_x=stored_x,
        requested_gumbel_overrides=(),
    )
    return evaluator


def test_audit_targets_stored_encoding_reaches_the_raw_policy_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--input-encoding stored` changes what row (a)'s forward is fed.

    The searches build their own encodings, so the ONLY observation that proves
    the flag took effect is the extra forward's input — assert it exactly.
    """
    stored = np.stack([_fixture_pairs()[i][0] for i in range(len(FENS))])
    fen_only_ev = _run_net_candidates(monkeypatch, None)
    stored_ev = _run_net_candidates(monkeypatch, stored)

    fen_batches = list(fen_only_ev.seen)
    stored_batches = list(stored_ev.seen)
    # stored runs one EXTRA forward per batch (the raw-policy one); the
    # search-facing forward is unchanged.
    assert len(stored_batches) == 2 * len(fen_batches)
    assert np.array_equal(
        np.concatenate(stored_batches[0::2], axis=0),
        np.concatenate(fen_batches, axis=0),
    )
    assert np.array_equal(np.concatenate(stored_batches[1::2], axis=0), stored)


def test_audit_targets_fen_only_runs_exactly_one_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default must not gain the audit-v2 forward, or its cost and its
    numbers both move."""
    evaluator = _run_net_candidates(monkeypatch, None)
    fed = np.concatenate(evaluator.seen, axis=0)
    want = np.stack([
        encode_cboard(CBoard.from_board(chess.Board(f)), **ENC_KWARGS) for f in FENS
    ])
    assert np.array_equal(fed, want.astype(np.float32))


def test_audit_targets_labels_every_row_with_its_own_encoding() -> None:
    import scripts.audit_targets as at

    for encoding in ("fen_only", "stored"):
        labels = at.candidate_labels(encoding)
        assert set(labels) == set(at._CANDIDATE_NAMES)
        assert f"[enc={encoding}]" in labels["raw"]
        # The searches encode internally; claiming otherwise would be the
        # defect this labelling exists to prevent.
        for cand in ("search", "train", "train_fast"):
            assert "[enc=fen_only, search-internal]" in labels[cand]
        assert "[no net input]" in labels["sf_soft"]


# ---------------------------------------------------------------------------
# 5. score_audit_v2 — the rig that produced the LEDGER'S PUBLISHED NUMBERS
# ---------------------------------------------------------------------------
#
# This file has its own arm dispatcher, so its `v1` BASELINE arm is a branch a
# silent regression would zero out: point `v1` at the stored row and every
# v1-vs-v2 contrast reads exactly 0.00 while the run, the tests and the lint
# gate all look healthy. That is mutation B's failure shape one file over, and
# these are the tests that make it fail.


def _score_arm_inputs(fen: str, parent: np.ndarray):
    """(root fen planes, child fen planes, stored parent) for one position."""
    board = chess.Board(fen)
    root_fen = np.asarray(
        encode_cboard(CBoard.from_board(board), **ENC_KWARGS), dtype=np.float32,
    )
    board.push(next(iter(board.legal_moves)))
    child_fen = np.asarray(
        encode_cboard(CBoard.from_board(board), **ENC_KWARGS), dtype=np.float32,
    )
    return root_fen, child_fen, parent


@pytest.mark.parametrize("fen", IDENTITY_FENS)
def test_score_audit_v2_v1_arm_is_the_fen_only_identity(fen: str) -> None:
    """The BASELINE arm must be the untouched FEN-only encoding.

    Kills the mutation that makes `v1` return the stored row: the contrast
    would read 0.00 everywhere and nothing else in the repo would notice.
    """
    import scripts.score_audit_v2 as sa

    root_fen, child_fen, parent = _score_arm_inputs(fen, _fixture_pairs()[0][0])
    assert np.array_equal(sa.root_planes("v1", root_fen, parent), root_fen)
    assert np.array_equal(sa.child_planes("v1", child_fen, parent), child_fen)


@pytest.mark.parametrize("fen", IDENTITY_FENS)
def test_score_audit_v2_v2_arm_is_the_stored_encoding(fen: str) -> None:
    import scripts.score_audit_v2 as sa

    root_fen, child_fen, parent = _score_arm_inputs(fen, _fixture_pairs()[0][0])
    assert np.array_equal(sa.root_planes("v2", root_fen, parent), parent)
    assert np.array_equal(
        sa.child_planes("v2", child_fen, parent),
        child_input_planes(parent, child_fen),
    )
    # ...and the v2 arm is not accidentally the v1 arm.
    assert not np.array_equal(sa.root_planes("v2", root_fen, parent), root_fen)


def test_score_audit_v2_v1_stm_arm_touches_only_the_colour_flag() -> None:
    """The attribution arm must not drift into either neighbour.

    Its whole job is to separate the colour flag from the history frames, so
    a version that also spliced history would silently answer a different
    question — and would still produce plausible numbers.
    """
    import scripts.score_audit_v2 as sa

    exact_hits = 0
    for parent, _child in _fixture_pairs():
        root_fen, child_fen, _ = _score_arm_inputs(FENS[1], parent)
        for got, base in (
            (sa.root_planes("v1_stm", root_fen, parent), root_fen),
            (sa.child_planes("v1_stm", child_fen, parent), child_fen),
        ):
            differing = set(np.flatnonzero((got != base).any(axis=(1, 2))).tolist())
            assert differing <= {STM_PLANE}, differing
            exact_hits += int(differing == {STM_PLANE})
    # The fixture spans both colour-flag directions, so the arm must actually
    # HAVE changed the flag somewhere; a no-op would satisfy the subset check.
    assert exact_hits > 0


def test_score_audit_v2_arm_tags_name_the_encoding() -> None:
    """`arm_tag` is the provenance stamp printed on every published line."""
    import scripts.score_audit_v2 as sa

    assert sa.arm_tag("v1", 256, 256) == "[arm=v1 enc=fen_only pb=256 vb=256]"
    assert sa.arm_tag("v2", 256, 256) == "[arm=v2 enc=stored pb=256 vb=256]"
    assert sa.arm_tag("v1_stm", 128, 64) == (
        "[arm=v1_stm enc=fen_only+true_stm pb=64 vb=128]"
    )
    assert set(sa.ARM_ENCODING) == {"v1", "v1_stm", "v2"}
    assert sa.ARM_ENCODING["v1"] == "fen_only"
    assert sa.ARM_ENCODING["v2"] == "stored"


def test_score_audit_v2_refuses_an_arm_with_no_encoding() -> None:
    import scripts.score_audit_v2 as sa

    with pytest.raises(ValueError, match="no --input-encoding equivalent"):
        sa._require_encoding("v1_stm")


# ---------------------------------------------------------------------------
# 6. match_audit_rows.require_canonical
# ---------------------------------------------------------------------------


def test_require_canonical_refuses_a_black_to_move_audit_board() -> None:
    """The fingerprint join assumes white-to-move canonical audit boards.

    Without this guard a black-to-move row would simply never match and would
    be reported as an unmatched row — a silent shortfall rather than an error.
    """
    import scripts.match_audit_rows as mar

    with pytest.raises(SystemExit, match="not side-to-move canonical"):
        mar.require_canonical([
            chess.Board(FENS[0]), chess.Board(BLACK_TO_MOVE_FENS[0]),
        ])


def test_require_canonical_accepts_the_real_audit_shape() -> None:
    import scripts.match_audit_rows as mar

    mar.require_canonical([chess.Board(f) for f in FENS])


def test_board_fingerprint_separates_distinct_positions() -> None:
    """A fingerprint that collided everywhere would make the join meaningless."""
    import scripts.match_audit_rows as mar

    prints = {mar.board_fingerprint(chess.Board(f)) for f in IDENTITY_FENS}
    assert len(prints) == len(IDENTITY_FENS)
    # Colour/castling/EP are NOT in the fingerprint by design (it is a superset
    # filter refined by the position-key compare), so the two starting-position
    # spellings must collide — pinning that the join cannot rely on it alone.
    assert mar.board_fingerprint(chess.Board(chess.STARTING_FEN)) == (
        mar.board_fingerprint(chess.Board(BLACK_TO_MOVE_FENS[0].replace("4P3", "8")
                                          .replace("PPPP1PPP", "PPPPPPPP")))
    )
