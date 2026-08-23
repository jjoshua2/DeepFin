"""The BT4 policy dump's gather + renormalisation, without an ONNX session.

The pieces under test are the ones that can be wrong SILENTLY: which 1858 slot
each legal move reads, and whether the emitted dict is a distribution over the
legal moves. A synthetic policy row whose value at slot ``i`` IS ``i`` makes a
mis-gather visible as a wrong number rather than a wrong-looking one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from chess_anti_engine.moves.lc0_1858_movestrs import LC0_1858_UCI_TO_IDX
from scripts.bt4_policy_dump import (
    Row,
    run,
    board_from_stored_x,
    source_rows,
    x_to_lc0_planes,
    batched,
    castling_probability,
    entropy_nats,
    iter_fen_rows,
    legal_move_policy,
    load_done_keys,
    read_header,
    remap_provenance,
    resolve_policy_output,
    shard_encoding,
    truncate_torn_tail,
)

# Both castles legal for white, plus a 7th-rank pawn and pieces that can slide
# onto the back rank -- every family the static remap got wrong, in one board.
KITCHEN_SINK = "r3k2r/P6P/8/8/8/8/8/R3K2R w KQkq - 0 1"
# Black to move AND castling legal: under the static-remap mutant the wrongly
# read slots reorder the probabilities here, whereas on KITCHEN_SINK they
# happen to preserve the order.
ORDER_SENSITIVE = "r3k2r/8/8/8/8/8/6PP/R3K2R b KQkq - 0 1"


def _identity_row() -> np.ndarray:
    """policy[i] == i, so a gathered logit names the slot it came from."""
    return np.arange(COMPACT_POLICY_SIZE, dtype=np.float32)


@pytest.mark.parametrize("fen", [KITCHEN_SINK, ORDER_SENSITIVE])
def test_each_legal_move_reads_its_own_leela_slot(fen: str) -> None:
    """Recovers the EXACT slot each move was gathered from, not merely its rank.

    With policy[i] == i the softmax gives log p_i = idx_i - logsumexp, so
    log p_i - log p_0 == idx_i - idx_0 exactly. Comparing those differences
    pins the gathered INDEX per move.

    ⚑ Regression: the first version of this test compared only the probability
    RANKING, and it PASSED under the static-remap mutant -- on the kitchen-sink
    board the wrongly-read slots happened to preserve the order, so a test
    written to catch that mutant could not see it.
    """
    board = chess.Board(fen)
    ucis, probs = legal_move_policy(board, _identity_row())
    assert len(ucis) == board.legal_moves.count()
    want = np.array([
        LC0_1858_UCI_TO_IDX[_expected_leela_uci(board, chess.Move.from_uci(u))]
        for u in ucis
    ], dtype=np.float64)
    assert len(set(want.tolist())) == len(want), "two legal moves cannot share a slot"
    got = np.log(probs) - np.log(probs[0])
    assert np.allclose(got, want - want[0], atol=1e-6), (
        f"gathered slots off by {np.round(got - (want - want[0]), 3)}"
    )


def _expected_leela_uci(board: chess.Board, move: chess.Move) -> str:
    """LC0's spelling of ``move``, written out here rather than imported.

    A test that asked the remap what the remap thinks would pass with the
    remap broken; this restates the two conventions independently.
    """
    to_sq = move.to_square
    if board.is_castling(move):
        back = chess.BB_RANK_1 if board.turn == chess.WHITE else chess.BB_RANK_8
        rooks = board.castling_rights & back
        kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        to_sq = chess.msb(rooks) if kingside else chess.lsb(rooks)
    flip = board.turn == chess.BLACK
    f = chess.square_mirror(move.from_square) if flip else move.from_square
    t = chess.square_mirror(to_sq) if flip else to_sq
    suffix = {chess.QUEEN: "q", chess.ROOK: "r", chess.BISHOP: "b", chess.KNIGHT: ""}
    return (chess.square_name(f) + chess.square_name(t)
            + ("" if move.promotion is None else suffix[int(move.promotion)]))


def test_castling_reads_the_king_takes_rook_slot_not_the_slide() -> None:
    """The fingerprint of the fix: O-O must read ``e1h1``, not ``e1g1``.

    With the static remap this gathered LC0's ordinary ``e1g1`` slide logit --
    a real number for an unrelated move, which is why the defect never raised.
    """
    board = chess.Board(KITCHEN_SINK)
    row = np.full((COMPACT_POLICY_SIZE,), -20.0, dtype=np.float32)
    row[LC0_1858_UCI_TO_IDX["e1h1"]] = 10.0  # the castling slot
    row[LC0_1858_UCI_TO_IDX["e1g1"]] = -20.0  # the decoy slide slot
    ucis, probs = legal_move_policy(board, row)
    best = ucis[int(np.argmax(probs))]
    assert best == "e1g1", "O-O should dominate when its LC0 slot holds the mass"
    assert probs.max() > 0.9
    castles = castling_probability(board, ucis, probs)
    assert set(castles) == {"e1g1", "e1c1"}
    assert castles["e1g1"] > 0.9


def test_black_to_move_moves_are_oriented_before_lookup() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    row = np.full((COMPACT_POLICY_SIZE,), -20.0, dtype=np.float32)
    row[LC0_1858_UCI_TO_IDX["e1h1"]] = 10.0  # oriented: black's O-O
    ucis, probs = legal_move_policy(board, row)
    assert ucis[int(np.argmax(probs))] == "e8g8"


def test_promotion_pieces_do_not_collide() -> None:
    """=Q/=R/=B have their own slots and =N aliases the bare entry; all four
    must come back as four DIFFERENT probabilities."""
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    ucis, probs = legal_move_policy(board, _identity_row())
    promos = {u: p for u, p in zip(ucis, probs.tolist(), strict=True) if len(u) == 5}
    assert set(promos) == {"a7a8q", "a7a8r", "a7a8b", "a7a8n"}
    assert len(set(promos.values())) == 4


def test_output_is_a_distribution_over_legal_moves_only() -> None:
    for fen in (KITCHEN_SINK, chess.STARTING_FEN,
                "8/8/6r1/1np5/p2k4/P7/8/2K5 w - - 0 1"):
        board = chess.Board(fen)
        ucis, probs = legal_move_policy(board, _identity_row())
        assert len(probs) == board.legal_moves.count()
        assert probs.min() > 0.0
        assert float(probs.sum()) == pytest.approx(1.0)
        assert chess.Move.from_uci(ucis[int(np.argmax(probs))]) in board.legal_moves


def test_masked_slots_do_not_poison_the_softmax() -> None:
    board = chess.Board(chess.STARTING_FEN)
    row = np.full((COMPACT_POLICY_SIZE,), -np.inf, dtype=np.float32)
    row[LC0_1858_UCI_TO_IDX["e2e4"]] = 1.0
    ucis, probs = legal_move_policy(board, row)
    assert np.isfinite(probs).all()
    assert ucis[int(np.argmax(probs))] == "e2e4"
    assert float(probs.sum()) == pytest.approx(1.0)


def test_terminal_position_yields_no_moves() -> None:
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # checkmate
    ucis, probs = legal_move_policy(board, _identity_row())
    assert ucis == []
    assert probs.shape == (0,)


def test_entropy_matches_a_hand_computed_uniform() -> None:
    assert entropy_nats(np.full(4, 0.25)) == pytest.approx(np.log(4.0))
    assert entropy_nats(np.array([1.0, 0.0])) == pytest.approx(0.0)


def test_iter_fen_rows_keeps_the_key_but_builds_from_the_full_fen(tmp_path: Path) -> None:
    """rule50 is plane 109 and the 4-field key does not carry it, so the board
    must come from `fen` while the output stays keyed by `key`."""
    key = "8/8/6r1/1np5/p2k4/P7/8/2K5 w - -"
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text(
        json.dumps({"key": key, "fen": key + " 89 1"}) + "\n"
        + json.dumps({"fen": chess.STARTING_FEN}) + "\n",
        encoding="utf-8",
    )
    rows = list(iter_fen_rows(jsonl))
    assert [r.key for r in rows] == [key, chess.STARTING_FEN]
    assert rows[0].board.halfmove_clock == 89  # from `fen`, not the clock-less key
    assert rows[0].identity() == {}  # FEN rows carry no shard identity
    assert chess.Board(key + " 89 1").halfmove_clock == 89
    assert chess.Board(key).halfmove_clock == 0  # what keying off `key` would feed
    plain = tmp_path / "rows.txt"
    plain.write_text(f"{chess.STARTING_FEN}\n\n{chess.STARTING_FEN}\n", encoding="utf-8")
    assert [r.key for r in iter_fen_rows(plain)] == [chess.STARTING_FEN] * 2


def test_resume_ignores_a_torn_trailing_line(tmp_path: Path) -> None:
    out = tmp_path / "dump.jsonl"
    out.write_text(
        json.dumps({"record": "header"}) + "\n"
        + json.dumps({"key": "a", "n_legal": 1, "policy": {}}) + "\n"
        + '{"key": "b", "n_leg',  # killed mid-write
        encoding="utf-8",
    )
    assert load_done_keys(out) == {"a"}


def test_provenance_pins_the_remap_sources() -> None:
    prov = remap_provenance()
    assert set(prov["blobs"]) == {
        "chess_anti_engine/moves/leela_index.py",
        "chess_anti_engine/moves/lc0_1858_movestrs.py",
        "chess_anti_engine/encoding/lc0.py",
    }
    # A dump must be attributable even from a dirty tree, so the blob hashes
    # (not just the commit) have to be real.
    assert all(len(v) == 40 for v in prov["blobs"].values())


# --------------------------------------------------------- resume / streaming
def _torn_dump(tmp_path: Path) -> Path:
    out = tmp_path / "dump.jsonl"
    out.write_text(
        json.dumps({"record": "header", "onnx": {"sha256": "abc"}}) + "\n"
        + json.dumps({"key": "a", "n_legal": 1, "policy": {}}) + "\n"
        + '{"key": "b", "n_leg',  # killed mid-write
        encoding="utf-8",
    )
    return out


def test_resuming_a_torn_file_leaves_it_well_formed(tmp_path: Path) -> None:
    """The torn record must be CUT, not appended onto.

    ⚑ Regression: opening in "a" without truncating welds the half-written
    record to the first new record, producing one unparseable line -- so the
    torn row and a good row are both lost, and nothing reports it.
    """
    out = _torn_dump(tmp_path)
    dropped = truncate_torn_tail(out)
    assert dropped == len('{"key": "b", "n_leg')
    with out.open("a", encoding="utf-8") as fh:  # what run() does next
        fh.write(json.dumps({"key": "c", "n_legal": 1, "policy": {}}) + "\n")
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for line in lines:
        json.loads(line)  # every line parses -- the actual contract
    assert [json.loads(ln).get("key") for ln in lines] == [None, "a", "c"]


def test_truncate_is_a_noop_on_a_clean_file(tmp_path: Path) -> None:
    out = tmp_path / "clean.jsonl"
    out.write_text(json.dumps({"key": "a"}) + "\n", encoding="utf-8")
    before = out.read_bytes()
    assert truncate_torn_tail(out) == 0
    assert out.read_bytes() == before


def test_read_header_returns_the_provenance_record(tmp_path: Path) -> None:
    out = _torn_dump(tmp_path)
    header = read_header(out)
    assert header is not None
    assert header["onnx"]["sha256"] == "abc"
    assert read_header(tmp_path / "missing.jsonl") is None


def test_batched_emits_fixed_size_groups_and_a_short_tail() -> None:
    rows = [Row(key=str(i), board=chess.Board()) for i in range(7)]
    assert [len(b) for b in batched(rows, 3)] == [3, 3, 1]
    assert [len(b) for b in batched([], 3)] == []


def test_shard_encoding_refuses_a_missing_or_wrong_layout() -> None:
    """A shard read under the wrong layout yields a valid tensor and a silently
    wrong rule50, so the layout is never assumed."""
    class _Group:
        def __init__(self, **attrs: object) -> None:
            self.attrs = attrs

    path = Path("shard_000.zarr")
    assert shard_encoding(
        _Group(input_history_encoding="lc0_root_legacy_meta"), path,
    ) == "lc0_root_legacy_meta"
    assert shard_encoding(_Group(input_history_encoding="lc0_root"), path) == "lc0_root"
    with pytest.raises(SystemExit, match="no 'input_history_encoding'"):
        shard_encoding(_Group(), path)
    with pytest.raises(SystemExit, match="unknown input_history_encoding"):
        shard_encoding(_Group(input_history_encoding="not_a_layout"), path)
    with pytest.raises(SystemExit, match="not an LC0-root layout"):
        shard_encoding(_Group(input_history_encoding="legacy"), path)


class _FakeOut:
    def __init__(self, name: str, width: int) -> None:
        self.name = name
        self.shape = ["batch", width]


class _FakeSess:
    def __init__(self, *outs: _FakeOut) -> None:
        self._outs = list(outs)

    def get_outputs(self) -> list[_FakeOut]:
        return self._outs


def test_resolve_policy_output_refuses_an_ambiguous_graph() -> None:
    """Two 1858-wide heads must raise, matching scripts/net_source.py. Picking
    one silently is the defect class this file exists to avoid."""
    one = _FakeSess(_FakeOut("wdl", 3), _FakeOut("policy", 1858))
    assert resolve_policy_output(one, None) == 1
    two = _FakeSess(_FakeOut("policy", 1858), _FakeOut("policy2", 1858))
    with pytest.raises(SystemExit, match="2 1858-wide outputs"):
        resolve_policy_output(two, None)
    assert resolve_policy_output(two, "policy2") == 1


def test_resolve_policy_output_names_the_available_outputs() -> None:
    sess = _FakeSess(_FakeOut("wdl", 3), _FakeOut("policy", 1858))
    with pytest.raises(SystemExit, match="is not an output of this graph"):
        resolve_policy_output(sess, "nope")
    with pytest.raises(SystemExit, match="no 1858-wide policy output"):
        resolve_policy_output(_FakeSess(_FakeOut("wdl", 3)), None)


def test_shard_rows_carry_the_join_identity() -> None:
    """The arm-B join needs shard/row/game_id/ply_index on every record, plus the
    canonical FEN now that ``key`` is the fingerprint and no longer readable."""
    row = Row(key="k", board=chess.Board(), fen="a-fen", shard="shard_1.zarr",
              row_index=7, game_id=123, ply_index=42)
    assert row.identity() == {
        "fen": "a-fen", "shard": "shard_1.zarr", "row": 7, "game_id": 123,
        "ply_index": 42,
    }


# ------------------------------------------------- the frame contract, pinned
BLACK_TO_MOVE_CASTLING = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1"


def test_shard_path_emits_the_CANONICAL_frame_end_to_end() -> None:
    """⚑⚑ CONTRACT TEST — the shard path must emit CANONICAL (mirrored) FENs and
    UCIs for a black-to-move row, and this is DELIBERATE.

    The downstream rig builds its labels on the same white-to-move canonical
    board reconstructed from the same planes, so canonical<->canonical is the
    coherent join; "fixing" the shard path to emit true-board coordinates would
    silently break arm B's coordinate system.

    What this adds over ``test_board_decodes_back_out_of_the_stored_tensor``:
    that test already pins the decoded BOARD (its ``turn == WHITE`` assertion
    does fail under a ``.mirror()`` mutant, on all 7 parametrisations). What it
    does NOT check is the EMITTED NAMING — the key string and the UCI keys that
    actually land in the JSONL. This asserts those: for a black-to-move row the
    key is the mirrored FEN and black's kingside castle is named ``e1g1``, with
    ``e8g8`` absent from the record entirely.
    """
    true_board = chess.Board(BLACK_TO_MOVE_CASTLING)
    assert true_board.turn == chess.BLACK
    assert chess.Move.from_uci("e8g8") in true_board.legal_moves

    # exactly what a shard stores, then exactly what the dump does to it
    stored = encode_position(true_board, add_features=True,
                             input_history_encoding="lc0_root_legacy_meta")
    planes = x_to_lc0_planes(stored, input_history_encoding="lc0_root_legacy_meta")
    decoded = board_from_stored_x(stored, planes,
                                  input_history_encoding="lc0_root_legacy_meta")

    assert decoded.turn == chess.WHITE, "canonical frame is always White to move"
    assert decoded.fen() == true_board.mirror().fen()
    key = decoded.fen()
    assert key.split()[1] == "w"

    row = np.full((COMPACT_POLICY_SIZE,), -20.0, dtype=np.float32)
    row[LC0_1858_UCI_TO_IDX["e1h1"]] = 10.0  # LC0's castling slot
    ucis, probs = legal_move_policy(decoded, row)
    assert "e1g1" in ucis, "black O-O must be named e1g1 in the canonical frame"
    assert "e8g8" not in ucis, "true-frame UCIs must NOT appear in a shard record"
    assert ucis[int(np.argmax(probs))] == "e1g1"


def test_fen_path_keeps_the_TRUE_frame() -> None:
    """The other half of the contract: FEN input is echoed verbatim, in true
    coordinates -- so the two modes genuinely differ and the header says which."""
    board = chess.Board(BLACK_TO_MOVE_CASTLING)
    row = np.full((COMPACT_POLICY_SIZE,), -20.0, dtype=np.float32)
    row[LC0_1858_UCI_TO_IDX["e1h1"]] = 10.0
    ucis, probs = legal_move_policy(board, row)
    assert "e8g8" in ucis
    assert "e1g1" not in ucis
    assert ucis[int(np.argmax(probs))] == "e8g8"


# ----------------------------------------------------------- resume integrity
def test_a_record_without_a_policy_is_not_counted_as_done(tmp_path: Path) -> None:
    """⚑ A write torn at the right byte can leave VALID JSON that stops before
    the policy field. Trusting it would skip that position forever."""
    out = tmp_path / "dump.jsonl"
    out.write_text(
        json.dumps({"key": "complete", "n_legal": 2, "policy": {"e2e4": 1.0}}) + "\n"
        + json.dumps({"key": "truncated", "n_legal": 2}) + "\n",
        encoding="utf-8",
    )
    assert load_done_keys(out) == {"complete"}


def test_transposition_collisions_are_counted(tmp_path: Path) -> None:
    """First-wins skip must be COUNTED, not silent."""
    rows = tmp_path / "rows.txt"
    rows.write_text("\n".join([chess.STARTING_FEN, chess.STARTING_FEN,
                                BLACK_TO_MOVE_CASTLING]) + "\n", encoding="utf-8")
    args = argparse.Namespace(input_source="fens", rows=str(rows),
                              check_legal_mask=True, limit=0)
    stats: dict[str, int] = {}
    emitted = list(source_rows(args, set(), stats))
    assert [r.key for r in emitted] == [chess.STARTING_FEN, BLACK_TO_MOVE_CASTLING]
    assert stats["seen_rows"] == 3
    assert stats["distinct_keys"] == 2
    assert stats["collisions"] == 1
    assert stats["resume_skipped"] == 0


def test_resume_skipped_rows_are_counted_separately(tmp_path: Path) -> None:
    rows = tmp_path / "rows.txt"
    rows.write_text(chess.STARTING_FEN + "\n", encoding="utf-8")
    args = argparse.Namespace(input_source="fens", rows=str(rows),
                              check_legal_mask=True, limit=0)
    stats: dict[str, int] = {}
    assert list(source_rows(args, {chess.STARTING_FEN}, stats)) == []
    assert stats["resume_skipped"] == 1
    assert stats["collisions"] == 0


def _resume_args(out: Path, rows: Path) -> argparse.Namespace:
    return argparse.Namespace(
        out=str(out), rows=str(rows), input_source="fens", resume=True,
        onnx="/nonexistent-should-never-be-opened.onnx", batch_size=8, threads=1,
        gpu_mem_gb=0.0, policy_output=None, limit=0, castle_examples=0,
        check_legal_mask=True, restrict_keys=None,
    )


def test_resume_refuses_a_dump_with_no_readable_header(tmp_path: Path) -> None:
    """⚑ A headerless/corrupt file is exactly the case the provenance guard
    exists for, so it must be refused rather than quietly appended to.

    Also pins the ORDER: the refusal happens before the file is touched, so the
    dump is byte-identical afterwards. (Truncating the torn tail first meant a
    run that refused to write had already mutated the file.)
    """
    rows = tmp_path / "rows.txt"
    rows.write_text(chess.STARTING_FEN + "\n", encoding="utf-8")
    out = tmp_path / "headerless.jsonl"
    out.write_text(
        json.dumps({"key": "a", "n_legal": 1, "policy": {"e2e4": 1.0}}) + "\n"
        + '{"key": "b", "n_leg',  # torn tail, and no header record
        encoding="utf-8",
    )
    before = out.read_bytes()
    with pytest.raises(SystemExit, match="no readable header record"):
        run(_resume_args(out, rows))
    assert out.read_bytes() == before, "a refused resume must not mutate the file"


def test_resume_onto_a_missing_file_is_allowed(tmp_path: Path) -> None:
    """--resume on a not-yet-existing dump is a normal first run, not a refusal;
    it must get past the header check (and then fail on the bogus onnx path)."""
    rows = tmp_path / "rows.txt"
    rows.write_text(chess.STARTING_FEN + "\n", encoding="utf-8")
    out = tmp_path / "fresh.jsonl"
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - ORT's own error type
        run(_resume_args(out, rows))
    assert "no readable header record" not in str(excinfo.value)
