"""``scripts/derive_corpus_targets.py`` -- the corpus bank -> training shards tool.

Every corpus row here is built through the GENERATOR'S OWN row-writing helpers
(``PhaseResult.as_row``, ``ShardWriter``) rather than through a hand-written
dict.  A fixture that spelled the block keys itself would keep passing after a
row-schema change that broke the real reader, which is the one failure these
tests exist to catch.

The banked values are chosen so that every assertion below is hand-computable:
each test names the cp it put in a block and the cp it expects to come back out,
and the distributions are compared against ``softmax(q / temp)`` recomputed from
the SHARED cp->q map rather than against a stored golden array.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest
import zarr

from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from chess_anti_engine.moves.leela_index import compact_index_for_move
from chess_anti_engine.replay.sample import ReplaySample
from chess_anti_engine.replay.shard import arrays_to_samples, iter_shard_paths, load_shard_arrays
from chess_anti_engine.train.value_blend_guard import (
    ValueBlendMisconfigured,
    assert_pid_cannot_reassert_sf_wdl,
)
from scripts import audit_label_candidates as gate
from scripts import derive_corpus_targets as derive
from scripts import gen_random_selfplay_shards as gen
from scripts import gen_sf_rooted_corpus as corpus
from scripts.lc0_data_to_rows import (
    run_config_problems,
    shard_dir_search_wdl_coverage,
    shard_dir_sf_wdl_coverage,
)

# ── the synthetic corpus ─────────────────────────────────────────────────────

#: 7 pieces (the generator's banking floor), 9 legal moves for White and 5 for
#: Black -- small enough to enumerate in an assertion, big enough for a top-K.
FEN_W = "7k/6pp/8/8/8/8/6PP/5NK1 w - - 0 1"
FEN_B = "7k/6pp/8/8/8/8/6PP/5NK1 b - - 0 1"

SLOPE = float(gen.NNUE_CP_SLOPE)
DRAW_WIDTH = float(gen.NNUE_CP_DRAW_WIDTH)

#: The generation config both of the corpus's own records stamp.
CONFIG_REQUESTED: dict[str, Any] = {
    "cp_slope": SLOPE,
    "cp_draw_width": DRAW_WIDTH,
    "run_id": "test_corpus",
}

#: ⚑ HASHED BY THE GENERATOR'S OWN STAMPER, not invented.  It used to be
#: ``"0" * 64``, which is fine for a summary (nothing hashes it) and fatal for a
#: manifest: ``load_resume_manifest`` refuses a manifest whose ``config_sha256``
#: does not hash its own ``config_requested``, and a corpus that must be
#: readable through BOTH records therefore has to carry the real stamp.  Every
#: assertion below compares against this name rather than against the literal,
#: so the value moving is not a behaviour change.
CONFIG_SHA = corpus.stamp_sha256(CONFIG_REQUESTED)

#: The staircase the synthetic summary declares: a full-width depth-9 scout and
#: one narrowed depth-11 rung, i.e. the production shape with one rung removed.
STAIRCASE = [{"width": "all", "depth": 9}, {"width": 3, "depth": 11}]


#: ``played_move=None`` is a real corpus state (a row whose game ended without
#: one), so the default cannot be ``None`` -- it has to be distinguishable from
#: "the caller asked for no played move".
_UNSET: Any = object()


def legal_ucis(fen: str) -> list[str]:
    return sorted(move.uci() for move in chess.Board(fen).legal_moves)


def depth_block(
    depth: int,
    values: dict[str, float],
    *,
    complete: bool = True,
    nodes: int | None = None,
) -> tuple[int, dict[str, float], bool, int | None]:
    return depth, values, complete, nodes


def phase_row(
    *,
    index: int,
    width_requested: str,
    width_realized: int,
    depth_requested: int,
    searchmoves: tuple[str, ...] | None,
    blocks: list[tuple[int, dict[str, float], bool, int | None]],
) -> dict[str, Any]:
    """One phase, serialized by ``PhaseResult.as_row`` -- the generator's own writer."""
    parse = corpus.StreamParse(
        blocks=tuple(
            corpus.DepthBlock(
                depth=depth,
                lines=tuple(
                    corpus.PvLine(
                        rank=rank,
                        move=move,
                        effective_cp=float(cp),
                        nodes=None if nodes is None else int(nodes),
                    )
                    for rank, (move, cp) in enumerate(values.items(), start=1)
                ),
                emissions=len(values),
                complete=complete,
                nodes_at_depth=nodes,
            )
            for depth, values, complete, nodes in blocks
        ),
        re_emissions=0,
        re_emissions_disagreeing=0,
        bound_lines=0,
        unscored_lines=0,
        emission_count_violations=0,
        duplicate_iteration_flushes=0,
    )
    return corpus.PhaseResult(
        index=index,
        width_requested=width_requested,
        width_realized=width_realized,
        width_streamed=width_realized,
        depth_requested=depth_requested,
        searchmoves=searchmoves,
        parse=parse,
    ).as_row()


def full_width_phase(
    fen: str, per_depth: dict[int, dict[str, float]], *, nodes: dict[int, int] | None = None,
    complete: dict[int, bool] | None = None, depth_requested: int = 9,
) -> dict[str, Any]:
    board = chess.Board(fen)
    return phase_row(
        index=0,
        width_requested=corpus.WIDTH_ALL,
        width_realized=board.legal_moves.count(),
        depth_requested=depth_requested,
        searchmoves=None,
        blocks=[
            depth_block(
                depth,
                values,
                complete=True if complete is None else complete.get(depth, True),
                nodes=None if nodes is None else nodes.get(depth),
            )
            for depth, values in sorted(per_depth.items())
        ],
    )


def narrowed_phase(
    per_depth: dict[int, dict[str, float]], *, index: int = 1, depth_requested: int = 11,
) -> dict[str, Any]:
    width = len(next(iter(per_depth.values())))
    return phase_row(
        index=index,
        width_requested=str(width),
        width_realized=width,
        depth_requested=depth_requested,
        searchmoves=tuple(next(iter(per_depth.values())).keys()),
        blocks=[
            depth_block(depth, values, nodes=None)
            for depth, values in sorted(per_depth.items())
        ],
    )


def corpus_row(
    *,
    fen: str,
    phases: list[dict[str, Any]],
    result: float | None,
    result_pgn: str | None = None,
    game_id: int = 0,
    ply: int = 0,
    config_sha: str = CONFIG_SHA,
    played_move: str | None = _UNSET,
    worker_id: int = 0,
) -> dict[str, Any]:
    board = chess.Board(fen)
    played = (
        next(iter(board.legal_moves)).uci() if played_move is _UNSET else played_move
    )
    return {
        "schema": corpus.ROW_SCHEMA,
        "run": {
            "run_id": "test_corpus",
            "config_sha256": config_sha,
            corpus.KEY_TT_CARRIED: True,
        },
        "fen": fen,
        "dedup_key": " ".join(fen.split(" ")[:5]),
        "worker_id": worker_id,
        "game_id": game_id,
        "ply": ply,
        "stm": "w" if board.turn == chess.WHITE else "b",
        "piece_count": int(chess.popcount(board.occupied)),
        "game_phase": corpus.PHASE_MIDDLEGAME,
        "played_move": played,
        "selection": {
            "temp": 0.3,
            "schedule_phase": "low",
            "temp_plies": 20,
            "value_depth": 9,
            "value_width": board.legal_moves.count(),
            "value_full_width": True,
            "legal_moves": board.legal_moves.count(),
            "seed_material": [1, 0, game_id, 1, ply],
        },
        "phases": phases,
        "result": result,
        "result_pgn": result_pgn,
        "adjudication": None,
    }


def write_corpus(
    tmp_path: Path,
    rows: list[dict[str, Any]],
    *,
    staircase: list[dict[str, Any]] | None = None,
    config_sha: str = CONFIG_SHA,
    name: str = "corpus",
    drop_shard_from_summary: bool = False,
    complete: bool = True,
) -> Path:
    """A corpus directory, written by the GENERATOR'S OWN writer.

    ``complete=True`` is a run that finished: ``manifest.json`` (banked at
    launch), the per-worker ``w00.progress.jsonl`` ``ShardWriter`` appends to as
    it closes shards, and ``summary.json``.  ``complete=False`` is the same
    corpus without the summary -- a run that is live, was killed, or sits
    between ``--resume`` sessions, which is the state ``manifest+progress`` mode
    exists for.

    ⚑ ``end_game`` IS CALLED, and it is not decoration.  ``ShardWriter.close``
    ABANDONS a shard holding rows of a game that never ended -- it is left
    unlisted, no progress line is written and it never enters ``writer.shards``
    -- so a fixture that only ``write``\\ s produces a corpus whose inventory is
    empty and whose progress file does not exist.  That is what the generator's
    ``--resume`` work (``48ac57471``) changed under this file, and it is why
    every inventory-checking test here was failing before this call was added.
    """
    out = tmp_path / name
    out.mkdir()
    writer = corpus.ShardWriter(out_dir=out, worker_id=0, shard_rows=1000)
    open_game: int | None = None
    for row in rows:
        game_id = int(row["game_id"])
        if open_game is not None and game_id != open_game:
            writer.end_game(open_game)
        writer.write(row)
        open_game = game_id
    if open_game is not None:
        writer.end_game(open_game)
    writer.close()
    write_manifest(out, staircase=staircase, config_sha=config_sha)
    if not complete:
        return out
    summary = {
        "schema": corpus.SUMMARY_SCHEMA,
        "row_schema": corpus.ROW_SCHEMA,
        "run_id": "test_corpus",
        "config_sha256": config_sha,
        "config_requested": dict(CONFIG_REQUESTED),
        "staircase_parsed": staircase if staircase is not None else STAIRCASE,
        "shards": [] if drop_shard_from_summary else list(writer.shards),
        "banked_rows_min_piece_count": corpus.MIN_BANKED_PIECES,
    }
    (out / corpus.SUMMARY_NAME).write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    return out


def write_manifest(
    out: Path,
    *,
    staircase: list[dict[str, Any]] | None = None,
    config_sha: str = CONFIG_SHA,
    config_requested: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The launch record, in the shape ``write_launch_manifest`` writes it."""
    requested = (
        dict(CONFIG_REQUESTED) if config_requested is None else dict(config_requested)
    )
    manifest = {
        "schema": corpus.MANIFEST_SCHEMA,
        "row_schema": corpus.ROW_SCHEMA,
        "complete": False,
        "config_requested": requested,
        "config_sha256": config_sha,
        "staircase_parsed": staircase if staircase is not None else STAIRCASE,
        "banked_rows_min_piece_count": corpus.MIN_BANKED_PIECES,
        "started_utc": "2026-08-28T00:00:00+00:00",
    }
    (out / corpus.MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )
    return manifest


def strip_summary(corpus_dir: Path, tmp_path: Path, *, name: str = "partial") -> Path:
    """THE SAME CORPUS, minus its ``summary.json``.

    A copy rather than a delete, so one test can derive from both records and
    compare -- the rows on either side are then the same bytes, not merely two
    fixtures built from the same list.
    """
    partial = tmp_path / name
    shutil.copytree(corpus_dir, partial)
    (partial / corpus.SUMMARY_NAME).unlink()
    return partial


def read_manifest(corpus_dir: Path) -> dict[str, Any]:
    return json.loads(
        (corpus_dir / corpus.MANIFEST_NAME).read_text(encoding="utf-8"),
    )


def progress_lines(corpus_dir: Path, worker_id: int = 0) -> list[dict[str, Any]]:
    """One worker's progress records, read with the GENERATOR'S own reader."""
    records, _ = corpus.read_worker_progress(
        corpus_dir / corpus.progress_name(worker_id),
    )
    return records


def run_derive(
    corpus_dir: Path, out_dir: Path, scheme: str, *extra: str, temp: float = 1.0,
) -> dict[str, Any]:
    code = derive.main([
        "--corpus", str(corpus_dir),
        "--out", str(out_dir),
        "--scheme", scheme,
        "--temp", str(temp),
        *extra,
    ])
    assert code == 0
    return json.loads((out_dir / derive.SUMMARY_NAME).read_text(encoding="utf-8"))


def read_rows(out_dir: Path) -> tuple[list[Any], dict[str, Any]]:
    """Read the emitted shards back with the RIG'S OWN reader.

    ``DiskReplayBuffer._try_load_shard`` calls ``load_shard_arrays``; nothing in
    this file re-implements the decode, so an assertion here is an assertion
    about what the trainer would see.
    """
    samples: list[Any] = []
    meta: dict[str, Any] = {}
    for path in iter_shard_paths(out_dir):
        arrs, shard_meta = load_shard_arrays(path, lazy=False)
        samples.extend(arrays_to_samples(arrs))
        meta = shard_meta
    return samples, meta


def hand_policy(cps: dict[str, float], *, temp: float) -> dict[str, float]:
    """``softmax(q / temp)`` recomputed from the shared map, per uci."""
    moves = list(cps)
    q = gate.q_from_effective_cp(
        np.array([cps[m] for m in moves], dtype=np.float64),
        slope=SLOPE,
        draw_width_cp=DRAW_WIDTH,
    )
    scaled = np.exp(q / temp - np.max(q / temp))
    probs = scaled / scaled.sum()
    return dict(zip(moves, (float(p) for p in probs)))


def emitted_policy(sample: Any, fen: str) -> dict[str, float]:
    """The emitted row's policy, keyed back by uci through the shared index map."""
    board = chess.Board(fen)
    return {
        move.uci(): float(sample.policy_target[compact_index_for_move(board, move)])
        for move in board.legal_moves
    }


def ramp(fen: str, best: str, *, best_cp: float = 400.0, step: float = 25.0) -> dict[str, float]:
    """A distinct cp per legal move, ``best`` on top, deterministic by uci order."""
    values = {best: best_cp}
    for offset, move in enumerate(m for m in legal_ucis(fen) if m != best):
        values[move] = best_cp - step * (offset + 1)
    return {move: values[move] for move in legal_ucis(fen)}


# ── schemes ──────────────────────────────────────────────────────────────────


def test_uniform_scheme_reads_the_depth_it_names(tmp_path: Path) -> None:
    """d5 and d9 disagree in the bank, so the two corpora must disagree too."""
    at5 = ramp(FEN_W, "g2g4")
    at9 = ramp(FEN_W, "f1e3")
    row = corpus_row(
        fen=FEN_W,
        phases=[full_width_phase(FEN_W, {5: at5, 9: at9})],
        result=1.0,
        result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row])

    five = run_derive(corpus_dir, tmp_path / "d5", "uniform-d5")
    nine = run_derive(corpus_dir, tmp_path / "d9", "uniform-d9")

    got5, _ = read_rows(tmp_path / "d5")
    got9, _ = read_rows(tmp_path / "d9")
    p5, p9 = emitted_policy(got5[0], FEN_W), emitted_policy(got9[0], FEN_W)

    assert max(p5, key=lambda m: p5[m]) == "g2g4"
    assert max(p9, key=lambda m: p9[m]) == "f1e3"
    for move, expected in hand_policy(at5, temp=1.0).items():
        assert p5[move] == pytest.approx(expected, abs=2e-3)
    for move, expected in hand_policy(at9, temp=1.0).items():
        assert p9[move] == pytest.approx(expected, abs=2e-3)
    assert five["realized"]["realized_base_depth_histogram"] == {"5": 1}
    assert nine["realized"]["realized_base_depth_histogram"] == {"9": 1}


def test_uniform_reads_a_move_from_the_deepest_phase_that_covers_it(
    tmp_path: Path,
) -> None:
    """A narrowed rung's depth-5 block wins over phase 0's, for the moves it has."""
    at5 = ramp(FEN_W, "g2g4")
    # The narrowed phase re-reports two of the moves at the SAME depth with a
    # different number -- the warm-table reading of the same cell.
    narrowed = {5: {"f1e3": 900.0, "g2g4": 10.0}, 11: {"f1e3": 950.0, "g2g4": 20.0}}
    row = corpus_row(
        fen=FEN_W,
        phases=[
            full_width_phase(FEN_W, {5: at5, 9: ramp(FEN_W, "f1d2")}),
            narrowed_phase(narrowed),
        ],
        result=0.0,
        result_pgn="1/2-1/2",
    )
    corpus_dir = write_corpus(tmp_path, [row])
    summary = run_derive(corpus_dir, tmp_path / "out", "uniform-d5")

    expected_cps = dict(at5)
    expected_cps.update(narrowed[5])
    got, _ = read_rows(tmp_path / "out")
    policy = emitted_policy(got[0], FEN_W)
    for move, expected in hand_policy(expected_cps, temp=1.0).items():
        assert policy[move] == pytest.approx(expected, abs=2e-3)
    assert max(policy, key=lambda m: policy[m]) == "f1e3"
    # Two of the nine values came from phase 1; the summary says so -- and it
    # also says the row HAD a phase 1, without which that split is unreadable.
    assert summary["realized"]["values_by_phase"] == {"0": 7, "1": 2}
    assert summary["realized"]["phases_per_row"] == {"2": 1}


def test_topk_reads_the_deep_depth_for_exactly_the_top_k_by_d1(tmp_path: Path) -> None:
    at9 = ramp(FEN_W, "f1e3")            # f1e3 400, then 375, 350, ... by uci order
    top_two = sorted(at9, key=lambda m: (-at9[m], m))[:2]
    # Deep block: both top-2 moves move, and they SWAP order.
    deep = {top_two[0]: -200.0, top_two[1]: 800.0, "g2g3": 700.0}
    row = corpus_row(
        fen=FEN_W,
        phases=[full_width_phase(FEN_W, {9: at9}), narrowed_phase({11: deep})],
        result=1.0,
        result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row])
    summary = run_derive(corpus_dir, tmp_path / "out", "top2-d11-rest-d9")

    expected_cps = dict(at9)
    for move in top_two:
        expected_cps[move] = deep[move]
    # ⚑ g2g3 is IN the deep block but NOT in the top 2 by d9, so it must keep
    # its d9 value: a scheme that read the deep block for everything it happened
    # to cover would be a different scheme with the same name.
    assert "g2g3" not in top_two
    assert expected_cps["g2g3"] == at9["g2g3"]

    got, _ = read_rows(tmp_path / "out")
    policy = emitted_policy(got[0], FEN_W)
    for move, expected in hand_policy(expected_cps, temp=1.0).items():
        assert policy[move] == pytest.approx(expected, abs=2e-3)
    assert max(policy, key=lambda m: policy[m]) == top_two[1]
    assert summary["realized"]["deep_tier_moves"] == 2
    assert summary["realized"]["base_tier_moves"] == len(legal_ucis(FEN_W)) - 2


def test_topk_refuses_a_top_move_the_narrowing_never_carried(tmp_path: Path) -> None:
    at9 = ramp(FEN_W, "f1e3")
    top_two = sorted(at9, key=lambda m: (-at9[m], m))[:2]
    # The generator narrowed on something else, so the SECOND-best move by d9
    # has no depth-11 block anywhere.
    deep = {top_two[0]: 500.0, "h2h4": 100.0}
    row = corpus_row(
        fen=FEN_W,
        phases=[full_width_phase(FEN_W, {9: at9}), narrowed_phase({11: deep})],
        result=1.0,
        result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row])
    with pytest.raises(derive.CorpusIntegrityError, match="top 2 by depth 9"):
        run_derive(corpus_dir, tmp_path / "out", "top2-d11-rest-d9")


def test_nodes_scheme_takes_the_deepest_affordable_complete_depth(
    tmp_path: Path,
) -> None:
    per_depth = {d: ramp(FEN_W, m) for d, m in ((1, "h2h4"), (3, "g2g3"), (5, "f1e3"))}
    nodes = {1: 100, 3: 2_000, 5: 40_000}
    row = corpus_row(
        fen=FEN_W,
        phases=[full_width_phase(FEN_W, per_depth, nodes=nodes)],
        result=1.0,
        result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row])
    summary = run_derive(corpus_dir, tmp_path / "out", "nodes-3000")

    assert summary["realized"]["realized_base_depth_histogram"] == {"3": 1}
    assert summary["realized"]["nodes_floor_hits"] == 0
    got, _ = read_rows(tmp_path / "out")
    policy = emitted_policy(got[0], FEN_W)
    for move, expected in hand_policy(per_depth[3], temp=1.0).items():
        assert policy[move] == pytest.approx(expected, abs=2e-3)


def test_nodes_scheme_skips_an_incomplete_block(tmp_path: Path) -> None:
    """An aborted iteration is not what a node budget would have bought."""
    per_depth = {d: ramp(FEN_W, m) for d, m in ((1, "h2h4"), (3, "g2g3"), (5, "f1e3"))}
    row = corpus_row(
        fen=FEN_W,
        phases=[full_width_phase(
            FEN_W, per_depth, nodes={1: 100, 3: 2_000, 5: 2_500},
            complete={5: False},
        )],
        result=1.0,
        result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row])
    # Depth 5 is affordable at 3000 nodes but incomplete, so depth 3 wins.
    summary = run_derive(corpus_dir, tmp_path / "out", "nodes-3000")
    assert summary["realized"]["realized_base_depth_histogram"] == {"3": 1}


def test_nodes_scheme_floors_and_counts_the_event(tmp_path: Path) -> None:
    per_depth = {1: ramp(FEN_W, "h2h4"), 3: ramp(FEN_W, "g2g3")}
    row = corpus_row(
        fen=FEN_W,
        phases=[full_width_phase(FEN_W, per_depth, nodes={1: 5_000, 3: 90_000})],
        result=1.0,
        result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row])
    summary = run_derive(corpus_dir, tmp_path / "out", "nodes-100")

    assert summary["realized"]["nodes_floor_hits"] == 1
    assert summary["realized"]["realized_base_depth_histogram"] == {"1": 1}
    got, _ = read_rows(tmp_path / "out")
    policy = emitted_policy(got[0], FEN_W)
    assert max(policy, key=lambda m: policy[m]) == "h2h4"


def test_nodes_scheme_reads_phase_zero_only(tmp_path: Path) -> None:
    """A narrowed rung covering the same depth must NOT displace phase 0's value.

    The scheme's whole claim is "what a full-width ``go nodes N`` would have
    produced"; a depth-11 reading spliced in would make that claim false while
    every other stamp stayed identical.
    """
    per_depth = {1: ramp(FEN_W, "h2h4"), 3: ramp(FEN_W, "g2g3")}
    row = corpus_row(
        fen=FEN_W,
        phases=[
            full_width_phase(FEN_W, per_depth, nodes={1: 100, 3: 2_000}),
            # Same DEPTH 3, wildly different numbers, from a narrowed rung.
            narrowed_phase({3: {"f1e3": 5_000.0, "g2g4": 4_000.0}}),
        ],
        result=1.0,
        result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row])
    summary = run_derive(corpus_dir, tmp_path / "out", "nodes-3000")

    assert summary["scheme"]["value_source"] == derive.VALUE_SOURCE_PHASE0
    assert summary["realized"]["values_by_phase"] == {"0": len(legal_ucis(FEN_W))}
    # ⚑ And phase 1 WAS there to be read -- otherwise "phase 0 only" would be a
    # property of the fixture rather than of the scheme.
    assert summary["realized"]["phases_per_row"] == {"2": 1}
    got, _ = read_rows(tmp_path / "out")
    policy = emitted_policy(got[0], FEN_W)
    for move, expected in hand_policy(per_depth[3], temp=1.0).items():
        assert policy[move] == pytest.approx(expected, abs=2e-3)
    assert max(policy, key=lambda m: policy[m]) == "g2g3"


# ── temperature ──────────────────────────────────────────────────────────────


def test_temp_changes_the_target_and_is_recoverable_from_it(tmp_path: Path) -> None:
    at9 = ramp(FEN_W, "f1e3")
    row = corpus_row(
        fen=FEN_W, phases=[full_width_phase(FEN_W, {9: at9})],
        result=1.0, result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row])

    hot = run_derive(corpus_dir, tmp_path / "hot", "uniform-d9", temp=2.0)
    cold = run_derive(corpus_dir, tmp_path / "cold", "uniform-d9", temp=0.05)

    got_hot, _ = read_rows(tmp_path / "hot")
    got_cold, _ = read_rows(tmp_path / "cold")
    p_hot, p_cold = emitted_policy(got_hot[0], FEN_W), emitted_policy(got_cold[0], FEN_W)

    assert p_cold["f1e3"] > p_hot["f1e3"] + 0.2
    for move, expected in hand_policy(at9, temp=2.0).items():
        assert p_hot[move] == pytest.approx(expected, abs=2e-3)
    # ⚑ The REALIZED stamp: the temperature read back off the emitted policy,
    # not echoed from the flag.
    for summary, requested in ((hot, 2.0), (cold, 0.05)):
        recovered = summary["realized"]["temp_recovered_from_emitted_policy"]
        assert recovered["n"] == 1
        assert recovered["min"] == pytest.approx(requested, rel=1e-6)
        assert recovered["max"] == pytest.approx(requested, rel=1e-6)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_a_non_positive_temperature_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        derive.softmax_at_temp(np.array([0.1, 0.2]), temp=bad)


def test_a_bad_temperature_is_refused_before_the_corpus_is_touched(
    tmp_path: Path,
) -> None:
    """The refusal must not wait for the first row that reaches the softmax."""
    with pytest.raises(ValueError, match="finite and positive"):
        derive.main([
            "--corpus", str(tmp_path / "does_not_exist"),
            "--out", str(tmp_path / "out"),
            "--scheme", "uniform-d9",
            "--temp", "0",
        ])
    assert not (tmp_path / "out").exists()


def test_the_float16_cast_cost_is_counted(tmp_path: Path) -> None:
    """A cold target's tail dies in the shard's float16, and the run says so.

    ⚑ The support stamp is measured on the far side of the cast for exactly
    this case: in float64 all nine moves carry mass, and the trainer sees three.
    """
    at9 = ramp(FEN_W, "f1e3", best_cp=400.0, step=25.0)
    row = corpus_row(
        fen=FEN_W, phases=[full_width_phase(FEN_W, {9: at9})],
        result=1.0, result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row])
    summary = run_derive(corpus_dir, tmp_path / "out", "uniform-d9", temp=0.005)

    realized = summary["realized"]
    assert realized["policy_support_lost_to_float16"] > 0
    assert realized["policy_support_max"] < len(legal_ucis(FEN_W))
    samples, _ = read_rows(tmp_path / "out")
    stored_support = int((np.asarray(samples[0].policy_target) > 0).sum())
    assert stored_support == realized["policy_support_max"]
    # The legal mask still names every legal move -- which is the asymmetry the
    # counter exists to disclose.
    assert int(np.asarray(samples[0].legal_mask).sum()) == len(legal_ucis(FEN_W))


# ── the rig's format ─────────────────────────────────────────────────────────


def test_policy_lands_where_the_rig_reader_expects(tmp_path: Path) -> None:
    """Round trip: write a shard, read it with the rig's reader, check the argmax."""
    at9 = ramp(FEN_W, "f1e3")
    row = corpus_row(
        fen=FEN_W, phases=[full_width_phase(FEN_W, {9: at9})],
        result=1.0, result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row])
    summary = run_derive(corpus_dir, tmp_path / "out", "uniform-d9")

    samples, meta = read_rows(tmp_path / "out")
    assert len(samples) == 1
    sample = samples[0]
    board = chess.Board(FEN_W)

    assert meta["policy_encoding"] == "lc0_1858"
    assert int(meta["policy_size"]) == COMPACT_POLICY_SIZE
    assert meta["input_history_encoding"] == derive.INPUT_HISTORY_ENCODING
    assert sample.x.shape == (175, 8, 8)
    assert sample.policy_target.shape == (COMPACT_POLICY_SIZE,)

    # The hand-picked best move, mapped through the SHARED index map.
    expected_index = compact_index_for_move(board, chess.Move.from_uci("f1e3"))
    assert int(np.argmax(sample.policy_target)) == expected_index
    assert float(sample.policy_target.sum()) == pytest.approx(1.0, abs=2e-3)

    # Support and legal mask are exactly the legal moves, nothing else.
    legal_indices = {
        compact_index_for_move(board, move) for move in board.legal_moves
    }
    assert set(np.nonzero(sample.policy_target)[0].tolist()) == legal_indices
    assert set(np.nonzero(sample.legal_mask)[0].tolist()) == legal_indices
    assert summary["realized"]["policy_support_min"] == len(legal_indices)
    assert summary["realized"]["policy_width"] == COMPACT_POLICY_SIZE
    assert summary["realized"]["x_planes"] == 175


def test_value_target_sign_follows_the_rows_own_side_to_move(tmp_path: Path) -> None:
    """A black-to-move row of a game WHITE won is a LOSS row, and reads +cp as +W."""
    at9 = ramp(FEN_B, "h8g8", best_cp=400.0)
    row = corpus_row(
        fen=FEN_B,
        phases=[full_width_phase(FEN_B, {9: at9})],
        # `result_from_pov("1-0", white_to_move=False)` is -1.0: the row's mover lost.
        result=-1.0,
        result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row])
    run_derive(corpus_dir, tmp_path / "out", "uniform-d9")

    samples, _ = read_rows(tmp_path / "out")
    sample = samples[0]
    assert int(sample.wdl_target) == 2  # 0=W / 1=D / 2=L, side-to-move POV

    # +400 cp is banked from the ROOT MOVER's seat, so the searched value is a
    # WIN for the row's side to move -- nothing is negated on the way in.
    expected = gen.cp_to_wdl_array(
        np.array([400.0]), slope=SLOPE, draw_width_cp=DRAW_WIDTH,
    ).reshape(-1)
    assert np.allclose(np.asarray(sample.search_wdl, dtype=np.float64), expected, atol=2e-3)
    assert float(sample.search_wdl[0]) > float(sample.search_wdl[2])


def test_a_win_row_and_a_draw_row_map_to_their_own_wdl_slots(tmp_path: Path) -> None:
    rows = [
        corpus_row(
            fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
            result=result, result_pgn=pgn, game_id=index, ply=index,
        )
        for index, (result, pgn) in enumerate(((1.0, "1-0"), (0.0, "1/2-1/2")))
    ]
    corpus_dir = write_corpus(tmp_path, rows)
    run_derive(corpus_dir, tmp_path / "out", "uniform-d9", "--seed", "0")
    samples, _ = read_rows(tmp_path / "out")
    assert sorted(int(s.wdl_target) for s in samples) == [0, 1]


def test_sf_wdl_is_absent_and_search_wdl_is_present(tmp_path: Path) -> None:
    """The channel choice is a format fact, so it is pinned as one."""
    row = corpus_row(
        fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
        result=1.0, result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row])
    out = tmp_path / "out"
    run_derive(corpus_dir, out, "uniform-d9")

    sf_labelled, rows = shard_dir_sf_wdl_coverage(out)
    search_labelled, _ = shard_dir_search_wdl_coverage(out)
    assert (sf_labelled, search_labelled, rows) == (0, 1, 1)


def test_the_manifests_overrides_pass_the_rigs_own_guards(tmp_path: Path) -> None:
    """The recommended config is checked against the rig's REAL guards.

    ⚑ Not against a restatement of them: ``run_config_problems`` and
    ``assert_pid_cannot_reassert_sf_wdl`` are the exact objects
    ``lc0_control_train.py`` calls at launch, driven by the MEASURED coverage of
    the shards this tool just wrote.  A manifest whose overrides the rig would
    reject is a manifest nobody can use.
    """
    row = corpus_row(
        fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
        result=1.0, result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row])
    out = tmp_path / "out"
    summary = run_derive(corpus_dir, out, "uniform-d9")

    sf_labelled, rows = shard_dir_sf_wdl_coverage(out)
    search_labelled, _ = shard_dir_search_wdl_coverage(out)
    overrides = summary["required_training_overrides"]
    cfg = {
        "sf_wdl_frac": float(overrides["sf_wdl_frac"]),
        "sf_wdl_frac_floor": float(overrides["sf_wdl_frac_floor"]),
        "search_wdl_frac": 1.0,
    }
    assert run_config_problems(
        cfg,
        shards_have_sf_wdl=sf_labelled == rows,
        shards_have_search_wdl=search_labelled == rows,
    ) == []
    assert_pid_cannot_reassert_sf_wdl(
        sf_wdl_frac=cfg["sf_wdl_frac"], sf_wdl_frac_floor=cfg["sf_wdl_frac_floor"],
    )
    # And the config the manifest talks a reader OUT of is genuinely refused,
    # which is why the searched value is not in `sf_wdl`.
    with pytest.raises(ValueBlendMisconfigured):
        assert_pid_cannot_reassert_sf_wdl(sf_wdl_frac=0.69, sf_wdl_frac_floor=0.0)


def test_history_is_zero_and_the_summary_measures_it(tmp_path: Path) -> None:
    row = corpus_row(
        fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
        result=1.0, result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row])
    summary = run_derive(corpus_dir, tmp_path / "out", "uniform-d9")

    assert summary["input"]["zero_history"] is True
    assert summary["realized"]["history_slots_nonzero_max"] == 1
    assert summary["realized"]["repetition_planes_nonzero_rows"] == 0

    samples, _ = read_rows(tmp_path / "out")
    planes = np.asarray(samples[0].x)
    # Slot 0 carries the position; slots 1..7 (planes 13..103) are empty.
    assert bool(np.any(planes[0:12]))
    assert not bool(np.any(planes[13:104]))


def test_shards_carry_the_scheme_and_schema_stamps(tmp_path: Path) -> None:
    row = corpus_row(
        fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
        result=1.0, result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row])
    run_derive(corpus_dir, tmp_path / "out", "uniform-d9", temp=0.7)

    path = iter_shard_paths(tmp_path / "out")[0]
    attrs = dict(zarr.open_group(str(path), mode="r").attrs.asdict())
    assert attrs["derive_scheme"] == "uniform-d9"
    assert attrs["derive_schema"] == derive.DERIVE_SCHEMA
    assert attrs["derive_temp"] == pytest.approx(0.7)
    assert attrs["derive_scheme_params"]["depth"] == 9
    assert attrs["derive_corpus_config_sha256"] == CONFIG_SHA
    assert attrs["derive_corpus_row_schema"] == corpus.ROW_SCHEMA
    # The extra attrs must not disturb the reader that ignores them.
    _arrs, meta = load_shard_arrays(path, lazy=False)
    assert meta["run_id"] == derive.SHARD_RUN_ID


# ── refusals and counters ────────────────────────────────────────────────────


def test_rows_without_a_result_are_skipped_and_counted(tmp_path: Path) -> None:
    rows = [
        corpus_row(
            fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
            result=1.0, result_pgn="1-0", game_id=0, ply=0,
        ),
        corpus_row(
            fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
            result=None, result_pgn=None, game_id=1, ply=0,
        ),
    ]
    corpus_dir = write_corpus(tmp_path, rows)
    summary = run_derive(corpus_dir, tmp_path / "out", "uniform-d9")

    assert summary["realized"]["rows_read"] == 2
    assert summary["realized"]["rows_written"] == 1
    assert summary["realized"]["rows_dropped_no_result"] == 1
    samples, _ = read_rows(tmp_path / "out")
    assert len(samples) == 1


def test_a_depth_above_the_rows_envelope_is_refused(tmp_path: Path) -> None:
    """The staircase promises depth 9; this row's phase 0 stopped at 5."""
    row = corpus_row(
        fen=FEN_W, phases=[full_width_phase(FEN_W, {1: ramp(FEN_W, "h2h4"), 5: ramp(FEN_W, "f1e3")})],
        result=1.0, result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row])
    with pytest.raises(derive.CorpusIntegrityError, match="no complete block at depth 9"):
        run_derive(corpus_dir, tmp_path / "strict", "uniform-d9")


def test_an_envelope_miss_can_be_tolerated_but_never_silently(tmp_path: Path) -> None:
    rows = [
        corpus_row(
            fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
            result=1.0, result_pgn="1-0", game_id=0,
        ),
        corpus_row(
            fen=FEN_W, phases=[full_width_phase(FEN_W, {5: ramp(FEN_W, "f1e3")})],
            result=1.0, result_pgn="1-0", game_id=1,
        ),
    ]
    corpus_dir = write_corpus(tmp_path, rows)
    summary = run_derive(
        corpus_dir, tmp_path / "out", "uniform-d9", "--max-envelope-misses", "1",
    )
    assert summary["realized"]["rows_dropped_envelope"] == 1
    assert summary["realized"]["rows_written"] == 1
    assert summary["realized"]["envelope_miss_examples"]
    assert "game 1 ply 0" in summary["realized"]["envelope_miss_examples"][0]


def test_a_scheme_beyond_the_corpus_staircase_is_refused_before_any_row(
    tmp_path: Path,
) -> None:
    row = corpus_row(
        fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
        result=1.0, result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row])
    out = tmp_path / "out"
    with pytest.raises(derive.CorpusIntegrityError, match="exceeds the corpus envelope"):
        run_derive(corpus_dir, out, "uniform-d13")
    assert iter_shard_paths(out) == []


def test_the_deep_depth_is_bounded_by_the_deepest_rung_not_the_full_width_one() -> None:
    """The two envelope bounds are different numbers and must not be swapped."""
    scheme = derive.parse_scheme("top3-d11-rest-d9")
    assert derive.scheme_vs_staircase_problems(scheme, STAIRCASE) == []
    assert derive.scheme_vs_staircase_problems(
        derive.parse_scheme("top3-d13-rest-d9"), STAIRCASE,
    )
    assert derive.scheme_vs_staircase_problems(
        derive.parse_scheme("top3-d11-rest-d10"), STAIRCASE,
    )


def test_a_populated_out_dir_is_refused(tmp_path: Path) -> None:
    row = corpus_row(
        fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
        result=1.0, result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row])
    out = tmp_path / "out"
    out.mkdir()
    (out / "already_here.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already holds files"):
        run_derive(corpus_dir, out, "uniform-d9")
    assert iter_shard_paths(out) == []


def test_limit_caps_the_rows_read_and_is_stamped(tmp_path: Path) -> None:
    rows = [
        corpus_row(
            fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
            result=1.0, result_pgn="1-0", game_id=index, ply=index,
        )
        for index in range(5)
    ]
    corpus_dir = write_corpus(tmp_path, rows)
    summary = run_derive(corpus_dir, tmp_path / "out", "uniform-d9", "--limit", "2")

    assert summary["limit_requested"] == 2
    assert summary["realized"]["rows_read"] == 2
    assert summary["realized"]["rows_written"] == 2
    samples, _ = read_rows(tmp_path / "out")
    assert len(samples) == 2


def test_a_row_from_another_run_is_refused(tmp_path: Path) -> None:
    rows = [
        corpus_row(
            fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
            result=1.0, result_pgn="1-0",
        ),
        corpus_row(
            fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
            result=1.0, result_pgn="1-0", game_id=1, config_sha="f" * 64,
        ),
    ]
    corpus_dir = write_corpus(tmp_path, rows)
    with pytest.raises(derive.CorpusIntegrityError, match="does not match the corpus summary"):
        run_derive(corpus_dir, tmp_path / "out", "uniform-d9")


def test_a_banked_move_set_that_is_not_the_legal_set_is_refused(tmp_path: Path) -> None:
    values = ramp(FEN_W, "f1e3")
    values.pop("h2h4")  # a "complete" block that is missing a legal move
    board = chess.Board(FEN_W)
    phase = phase_row(
        index=0,
        width_requested=corpus.WIDTH_ALL,
        width_realized=board.legal_moves.count(),
        depth_requested=9,
        searchmoves=None,
        blocks=[depth_block(9, values)],
    )
    corpus_dir = write_corpus(
        tmp_path, [corpus_row(fen=FEN_W, phases=[phase], result=1.0, result_pgn="1-0")],
    )
    with pytest.raises(derive.CorpusIntegrityError, match="not the legal move set"):
        run_derive(corpus_dir, tmp_path / "out", "uniform-d9")


def test_a_shard_missing_from_the_summary_inventory_is_refused(tmp_path: Path) -> None:
    row = corpus_row(
        fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
        result=1.0, result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row])
    summary = json.loads((corpus_dir / "summary.json").read_text(encoding="utf-8"))
    summary["shards"].append({"path": "/elsewhere/w01-00000.jsonl.zst", "rows": 7})
    (corpus_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(derive.CorpusIntegrityError, match="are not the ones"):
        run_derive(corpus_dir, tmp_path / "out", "uniform-d9")


def test_an_empty_shard_inventory_is_refused(tmp_path: Path) -> None:
    """review finding 2: ``shards: []`` used to skip the inventory check.

    Mutation caught: restoring the lenient ``if named and`` guard — an empty
    inventory then trains on whatever is in the directory with no gate fired.
    Uses the fixture flag that existed for exactly this and was never exercised
    (review finding 3).
    """
    row = corpus_row(
        fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
        result=1.0, result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row], drop_shard_from_summary=True)
    with pytest.raises(derive.CorpusIntegrityError, match="names no shards"):
        run_derive(corpus_dir, tmp_path / "out", "uniform-d9")


def test_a_realized_cp_map_disagreeing_with_the_request_is_refused(
    tmp_path: Path,
) -> None:
    """review finding 1: the requested map was trusted without a cross-check.

    Mutation caught: deleting the realized-stamp loop from ``cp_map_params`` —
    a generator that selected under one map while stamping another then
    derives targets nothing played under, silently.
    """
    row = corpus_row(
        fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
        result=1.0, result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row])
    summary = json.loads((corpus_dir / "summary.json").read_text(encoding="utf-8"))
    summary["config_realized_by_worker"] = {
        "0": {"cp_slope": SLOPE * 2.0, "cp_draw_width": DRAW_WIDTH},
        # A dead worker's placeholder must be SKIPPED, not tripped over.
        "1": {"unavailable_worker_process_died": True},
    }
    (corpus_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    # ⚑ Not "realized cp map": ``tmp_path`` is named after the test, so that
    # phrase is inside every path the message quotes and the pattern would
    # match a refusal from an entirely different gate.
    with pytest.raises(
        derive.CorpusIntegrityError, match="disagrees with config_requested",
    ):
        run_derive(corpus_dir, tmp_path / "out", "uniform-d9")


def test_an_agreeing_realized_cp_map_passes(tmp_path: Path) -> None:
    """The cross-check must not refuse a healthy corpus (and must skip dead
    workers' placeholders on the passing path too)."""
    row = corpus_row(
        fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
        result=1.0, result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row])
    summary = json.loads((corpus_dir / "summary.json").read_text(encoding="utf-8"))
    summary["config_realized_by_worker"] = {
        "0": {"cp_slope": SLOPE, "cp_draw_width": DRAW_WIDTH},
        "1": {"unavailable_worker_process_died": True},
    }
    (corpus_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    summary = run_derive(corpus_dir, tmp_path / "out", "uniform-d9")
    assert summary["realized"]["rows_written"] == 1


def test_a_corpus_without_cp_map_parameters_is_refused(tmp_path: Path) -> None:
    row = corpus_row(
        fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
        result=1.0, result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row])
    summary = json.loads((corpus_dir / "summary.json").read_text(encoding="utf-8"))
    summary["config_requested"].pop("cp_slope")
    (corpus_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(derive.CorpusIntegrityError, match="cp_slope"):
        run_derive(corpus_dir, tmp_path / "out", "uniform-d9")


# ── the shared mapping and the seed ──────────────────────────────────────────


def test_the_cp_map_is_one_shared_function_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace the ONE object and watch this file's targets move.

    ⚑ ``gate.q_from_effective_cp`` resolves ``gen.cp_to_wdl_array`` as a module
    attribute at call time, so patching that single name is what proves the
    generator's selection, the label gate's arms and this tool's targets are the
    same mapping rather than three that agree today.
    """
    at9 = ramp(FEN_W, "f1e3")
    row = corpus_row(
        fen=FEN_W, phases=[full_width_phase(FEN_W, {9: at9})],
        result=1.0, result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row])
    baseline = run_derive(corpus_dir, tmp_path / "before", "uniform-d9")

    original = gen.cp_to_wdl_array

    def inverted(eff_cp: np.ndarray, **kwargs: float) -> np.ndarray:
        return original(-np.asarray(eff_cp, dtype=np.float64), **kwargs)

    monkeypatch.setattr(gen, "cp_to_wdl_array", inverted)
    patched = run_derive(corpus_dir, tmp_path / "after", "uniform-d9")

    before, _ = read_rows(tmp_path / "before")
    after, _ = read_rows(tmp_path / "after")
    p_before = emitted_policy(before[0], FEN_W)
    p_after = emitted_policy(after[0], FEN_W)
    # The best move under the inverted map is the WORST under the real one.
    assert max(p_before, key=lambda m: p_before[m]) == "f1e3"
    assert max(p_after, key=lambda m: p_after[m]) == min(at9, key=lambda m: at9[m])
    # ... and the VALUE channel moved with it, through the same object.
    assert float(before[0].search_wdl[0]) > float(before[0].search_wdl[2])
    assert float(after[0].search_wdl[0]) < float(after[0].search_wdl[2])
    assert baseline["cp_map"]["wdl_function"].endswith("cp_to_wdl_array")
    assert patched["cp_map"]["cp_slope"] == pytest.approx(SLOPE)


def test_the_seed_permutes_rows_without_changing_a_target(tmp_path: Path) -> None:
    rows = [
        corpus_row(
            fen=FEN_W,
            phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, move)})],
            result=1.0, result_pgn="1-0", game_id=index, ply=index,
        )
        for index, move in enumerate(legal_ucis(FEN_W))
    ]
    corpus_dir = write_corpus(tmp_path, rows)
    run_derive(corpus_dir, tmp_path / "a", "uniform-d9", "--seed", "1")
    run_derive(corpus_dir, tmp_path / "b", "uniform-d9", "--seed", "2")

    a, _ = read_rows(tmp_path / "a")
    b, _ = read_rows(tmp_path / "b")
    order_a = [int(s.game_id) for s in a]
    order_b = [int(s.game_id) for s in b]
    assert order_a != order_b
    assert sorted(order_a) == sorted(order_b) == list(range(len(rows)))
    # Same row, same target, whatever order it was written in.
    by_a = {int(s.game_id): s for s in a}
    by_b = {int(s.game_id): s for s in b}
    for game_id, sample in by_a.items():
        assert np.array_equal(sample.policy_target, by_b[game_id].policy_target)
        assert int(sample.wdl_target) == int(by_b[game_id].wdl_target)


# ── unit-level readings ──────────────────────────────────────────────────────


def test_recover_temp_returns_none_on_a_flat_row() -> None:
    q = np.array([0.25, 0.25, 0.25])
    probs = derive.softmax_at_temp(q, temp=0.4)
    assert derive.recover_temp(q, probs) is None


@pytest.mark.parametrize("temp", [0.05, 0.5, 1.0, 3.0])
def test_recover_temp_inverts_the_softmax(temp: float) -> None:
    q = np.array([-0.9, -0.1, 0.3, 0.8])
    recovered = derive.recover_temp(q, derive.softmax_at_temp(q, temp=temp))
    assert recovered is not None
    assert recovered == pytest.approx(temp, rel=1e-9)


@pytest.mark.parametrize(
    ("result", "expected"), [(1.0, 0), (0.0, 1), (-1.0, 2)],
)
def test_wdl_target_from_result(result: float, expected: int) -> None:
    assert derive.wdl_target_from_result(result) == expected


def test_wdl_target_refuses_a_value_result_from_pov_cannot_produce() -> None:
    with pytest.raises(derive.CorpusIntegrityError, match="not one of"):
        derive.wdl_target_from_result(0.5)


def test_scheme_canonical_is_rebuilt_not_echoed() -> None:
    for spelling in ("uniform-d9", "top4-d13-rest-d9", "nodes-200000"):
        assert derive.parse_scheme(spelling).canonical == spelling
    assert derive.parse_scheme("  uniform-d9  ").canonical == "uniform-d9"


def test_summary_passes_through_the_corpus_identity(tmp_path: Path) -> None:
    row = corpus_row(
        fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
        result=1.0, result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row])
    summary = run_derive(corpus_dir, tmp_path / "out", "uniform-d9")

    assert summary["corpus"]["config_sha256"] == CONFIG_SHA
    assert summary["corpus"]["row_schema"] == corpus.ROW_SCHEMA
    assert summary["corpus"][corpus.KEY_TT_CARRIED] == [True]
    assert summary["corpus"]["staircase_parsed"] == STAIRCASE
    assert summary["schema"] == derive.DERIVE_SCHEMA
    assert not math.isnan(summary["realized"]["temp_recovered_from_emitted_policy"]["mean"])


# ── the corpus's two records: summary vs manifest+progress ───────────────────
#
# ``summary.json`` is written ONCE, at run END.  A live, killed or resumed
# corpus has none -- for days -- while carrying every fact this tool needs in
# ``manifest.json`` (banked at launch) and the per-worker progress files
# (appended as each shard CLOSES).  These tests pin that the fallback reads the
# same corpus to the same rows, refuses the same damage, and never lets a
# partial read pass for a whole one.


def one_row_corpus(tmp_path: Path, *, complete: bool = True, name: str = "corpus") -> Path:
    """The smallest corpus every test below shares: one White row, one shard."""
    row = corpus_row(
        fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
        result=1.0, result_pgn="1-0",
    )
    return write_corpus(tmp_path, [row], complete=complete, name=name)


def test_summary_mode_is_used_and_stamped_when_the_summary_exists(
    tmp_path: Path,
) -> None:
    """(a) A COMPLETE corpus is read exactly as before, and says so.

    The manifest sits beside the summary here -- the generator writes one on
    every run -- so this also pins the precedence: the summary wins, and the
    progress files are not consulted at all.

    Mutation caught: inverting ``read_corpus_record``'s branch (falling back to
    manifest+progress whenever a manifest exists) drops the summary's
    both-directions inventory check and its realized cp cross-check on a corpus
    that has both.
    """
    corpus_dir = one_row_corpus(tmp_path)
    assert (corpus_dir / corpus.MANIFEST_NAME).exists()
    assert (corpus_dir / corpus.progress_name(0)).exists()

    summary = run_derive(corpus_dir, tmp_path / "out", "uniform-d9")

    assert summary["corpus"]["corpus_record"] == derive.CORPUS_RECORD_SUMMARY
    detail = summary["corpus"]["corpus_record_detail"]
    assert detail["run_finished"] is True
    assert detail["document"] == corpus.SUMMARY_NAME
    # ⚑ Not merely "summary mode": the progress files were not read at all.
    assert detail["progress_files_read"] == []
    assert detail["facts_only_in_summary"] == {}
    assert summary["cp_map"]["realized_cross_check_available"] is True


def test_derive_reads_the_record_itself_when_none_is_threaded_in(
    tmp_path: Path,
) -> None:
    """The direct-caller path: ``derive()`` with no ``corpus_record``.

    ⚑ Tested because ``main`` ALWAYS passes one, so a broken default here would
    be invisible from the CLI -- an argument accepted and then not used is this
    codebase's signature defect, and a default nothing exercises is the same
    shape. Both records go through it, so the fallback is covered too.
    """
    for name, complete, expected in (
        ("whole", True, derive.CORPUS_RECORD_SUMMARY),
        ("partial", False, derive.CORPUS_RECORD_PARTIAL),
    ):
        corpus_dir = one_row_corpus(tmp_path, complete=complete, name=name)
        out = derive.derive(
            corpus_dir=corpus_dir,
            out_dir=tmp_path / f"out-{name}",
            options=derive.DeriveOptions(
                scheme=derive.parse_scheme("uniform-d9"),
                temp=1.0,
                cp_slope=SLOPE,
                cp_draw_width=DRAW_WIDTH,
                limit=0,
                seed=derive.DEFAULT_SEED,
                rows_per_shard=derive.DEFAULT_ROWS_PER_SHARD,
                max_envelope_misses=0,
            ),
        )
        assert out["corpus"]["corpus_record"] == expected
        assert out["realized"]["rows_written"] == 1


def test_a_summary_corpus_with_no_shards_on_disk_keeps_its_own_message(
    tmp_path: Path,
) -> None:
    """(a) An EMPTY corpus directory and a corpus MISSING one shard differ.

    Both are refusals and both were before this change; the point is that they
    kept different messages, and moving the inventory check earlier must not
    re-route the first into the second's wording. An operator reading "the
    shards on disk are not the ones summary.json names: missing [everything]"
    goes looking for a partial copy; "holds no shards" says the directory is
    empty.
    """
    corpus_dir = one_row_corpus(tmp_path)
    for shard in derive.corpus_shard_paths(corpus_dir):
        shard.unlink()

    with pytest.raises(derive.CorpusIntegrityError, match="holds no"):
        run_derive(corpus_dir, tmp_path / "out", "uniform-d9")


def test_manifest_and_progress_derive_what_the_summary_derives(
    tmp_path: Path,
) -> None:
    """(b) THE SAME CORPUS, both records, byte-identical rows.

    ``strip_summary`` copies the directory and deletes only ``summary.json``,
    so the two derivations read the SAME shard bytes -- any difference is the
    record, not the fixture.  The rows are compared array by array through the
    rig's own shard reader, which is what makes "byte-identical" a claim about
    what the trainer would load rather than about a json blob.

    Mutation caught: making partial mode read ``corpus_shard_paths`` (the
    directory glob) instead of the progress inventory still passes on a corpus
    with nothing in flight -- so the real proof is the pair of assertions here
    PLUS ``test_a_shard_outside_the_snapshot_is_counted_and_never_read``, which
    is the case where the two lists differ.
    """
    rows = [
        corpus_row(
            fen=fen, phases=[full_width_phase(fen, {9: ramp(fen, best)})],
            result=result, result_pgn=pgn, game_id=index, ply=index,
        )
        for index, (fen, best, result, pgn) in enumerate([
            (FEN_W, "f1e3", 1.0, "1-0"),
            (FEN_B, "h8g8", -1.0, "1-0"),
            (FEN_W, "g2g4", 0.0, "1/2-1/2"),
        ])
    ]
    whole = write_corpus(tmp_path, rows)
    partial = strip_summary(whole, tmp_path)

    from_summary = run_derive(whole, tmp_path / "a", "uniform-d9")
    from_partial = run_derive(partial, tmp_path / "b", "uniform-d9")

    assert from_summary["corpus"]["corpus_record"] == derive.CORPUS_RECORD_SUMMARY
    assert from_partial["corpus"]["corpus_record"] == derive.CORPUS_RECORD_PARTIAL

    a, meta_a = read_rows(tmp_path / "a")
    b, meta_b = read_rows(tmp_path / "b")
    assert len(a) == len(b) == len(rows)
    assert meta_a == meta_b
    by_a = {int(s.game_id): s for s in a}
    by_b = {int(s.game_id): s for s in b}
    assert sorted(by_a) == sorted(by_b) == [0, 1, 2]
    for game_id, left in by_a.items():
        right = by_b[game_id]
        assert np.array_equal(left.x, right.x)
        assert np.array_equal(left.policy_target, right.policy_target)
        assert np.array_equal(left.legal_mask, right.legal_mask)
        assert np.array_equal(left.search_wdl, right.search_wdl)
        assert int(left.wdl_target) == int(right.wdl_target)
        assert int(left.ply_index) == int(right.ply_index)

    # The corpus identity the two records supply must agree too, or the rows
    # would be identical while their provenance stamps disagreed.
    for key in ("config_sha256", "run_id", "staircase_parsed", "row_schema"):
        assert from_summary["corpus"][key] == from_partial["corpus"][key]
    assert from_summary["cp_map"]["cp_slope"] == from_partial["cp_map"]["cp_slope"]
    assert (
        from_summary["cp_map"]["cp_draw_width"]
        == from_partial["cp_map"]["cp_draw_width"]
    )
    assert from_summary["realized"]["rows_written"] == (
        from_partial["realized"]["rows_written"]
    )


def test_a_listed_shard_missing_from_disk_is_refused(tmp_path: Path) -> None:
    """(c) The inventory claims rows that are gone -- a hard refusal.

    The same rule ``resume_worker_state`` applies for the same reason: deriving
    from what is left would train on a subset nothing recorded, and the row
    count would look like a smaller corpus rather than a damaged one.

    Mutation caught: replacing the ``path.exists()`` refusal with a ``continue``
    -- the derivation then silently succeeds on the shards that happen to
    remain.
    """
    partial = one_row_corpus(tmp_path, complete=False)
    listed = [record for record in progress_lines(partial) if record["path"] is not None]
    assert len(listed) == 1
    (partial / Path(str(listed[0]["path"])).name).unlink()

    with pytest.raises(derive.CorpusIntegrityError, match="it is not in"):
        run_derive(partial, tmp_path / "out", "uniform-d9")


def test_a_null_path_progress_line_is_a_game_record_not_a_shard(
    tmp_path: Path,
) -> None:
    """(d) ``path: null`` indexes GAMES, not a file, and must be skipped.

    ``ShardWriter.close`` writes one when a worker ends on games that banked no
    rows (every position dedup-served, or adjudicated before it banked). There
    is no file behind it.

    Mutation caught: dropping the ``if raw_path is None: continue`` -- the
    reader then resolves ``Path("None").name`` and refuses the corpus with
    "lists None and it is not in", i.e. a healthy corpus becomes underivable
    because one of its games was cheap.
    """
    partial = one_row_corpus(tmp_path, complete=False)
    baseline = run_derive(partial, tmp_path / "before", "uniform-d9")

    with open(partial / corpus.progress_name(0), "a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "path": None, "rows": 0, "codec": "zstd", "games": [41, 42],
        }, sort_keys=True) + "\n")
    # The generator's own reader accepts it as a record; it is this tool that
    # must decline to treat it as a shard.
    assert progress_lines(partial)[-1]["path"] is None

    after = run_derive(partial, tmp_path / "after", "uniform-d9")

    before_detail = baseline["corpus"]["corpus_record_detail"]
    after_detail = after["corpus"]["corpus_record_detail"]
    assert after_detail["shards_adopted"] == before_detail["shards_adopted"] == 1
    assert after_detail["rows_claimed_by_inventory"] == (
        before_detail["rows_claimed_by_inventory"]
    )
    assert after["realized"]["rows_written"] == baseline["realized"]["rows_written"]


def test_a_tampered_manifest_is_refused_by_its_own_sha(tmp_path: Path) -> None:
    """(e) ``config_sha256`` must hash ``config_requested`` -- the generator's check.

    The tamper is the one that matters: a cp map edited AFTER the fact. Every
    target in the corpus was selected under the original map, so a derivation
    that accepted the edit would produce a policy whose ranking disagrees with
    the play that produced the positions -- silently, and only in the tail.

    Mutation caught: replacing the ``load_resume_manifest`` call with a plain
    ``json.loads`` of the manifest -- the run then succeeds and derives under
    the tampered slope.
    """
    partial = one_row_corpus(tmp_path, complete=False)
    manifest = read_manifest(partial)
    assert manifest["config_requested"]["cp_slope"] == pytest.approx(SLOPE)
    manifest["config_requested"]["cp_slope"] = SLOPE * 2.0  # sha left untouched
    (partial / corpus.MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )

    with pytest.raises(derive.CorpusIntegrityError, match="inconsistent with itself"):
        run_derive(partial, tmp_path / "out", "uniform-d9")


def test_the_sha_check_refuses_disagreement_not_rewriting(tmp_path: Path) -> None:
    """The companion to the test above: the gate must not fire on everything.

    A manifest rewritten CONSISTENTLY (config and sha both) clears the sha
    check -- and is then caught one gate later by the row-identity join, because
    the rows still carry the sha they were banked under. Two gates, two
    different messages; a check that refused both would be indistinguishable
    from one that refused nothing useful.
    """
    partial = one_row_corpus(tmp_path, complete=False)
    requested = dict(CONFIG_REQUESTED)
    requested["cp_slope"] = SLOPE * 2.0
    write_manifest(
        partial,
        config_requested=requested,
        config_sha=corpus.stamp_sha256(requested),
    )
    # The rows still carry the ORIGINAL sha, so the row-identity join is what
    # refuses now -- a different gate, and the one that should fire here.
    with pytest.raises(derive.CorpusIntegrityError, match="does not match the corpus"):
        run_derive(partial, tmp_path / "out", "uniform-d9")


def test_the_partial_mode_stamp_is_in_the_output_manifest(tmp_path: Path) -> None:
    """(f) A derivation from a partial corpus is VISIBLY partial.

    Not a comment: the mode, the shard count, the rows the inventory claims and
    the facts a manifest cannot carry are all in the derived manifest, and the
    cp cross-check reports the ZERO workers it actually covered rather than
    looking exactly like a run that cross-checked every one.

    Mutation caught: hard-coding ``CORPUS_RECORD_SUMMARY`` into
    ``build_summary`` -- every row is still correct and the output claims to
    come from a finished run.
    """
    partial = one_row_corpus(tmp_path, complete=False)
    out = run_derive(partial, tmp_path / "out", "uniform-d9")

    assert out["corpus"]["corpus_record"] == derive.CORPUS_RECORD_PARTIAL
    assert out["corpus"]["corpus_record"] == "manifest+progress"
    detail = out["corpus"]["corpus_record_detail"]
    assert detail["run_finished"] is False
    assert detail["shards_adopted"] == 1
    assert detail["rows_claimed_by_inventory"] == 1
    assert detail["progress_files_read"] == [corpus.progress_name(0)]
    assert detail["progress_torn_tail_files"] == []
    assert detail["shards_on_disk_not_in_inventory"] == []
    assert corpus.MANIFEST_NAME in detail["document"]

    # ⚑ The weakening is REPORTED, not inferred from silence.
    assert "config_realized_by_worker" in detail["facts_only_in_summary"]
    assert out["cp_map"]["realized_workers_cross_checked"] == 0
    assert out["cp_map"]["realized_cross_check_available"] is False
    # And on a whole corpus the same reading is nonzero, so the 0 above is a
    # measurement rather than a field nothing ever fills.
    whole = one_row_corpus(tmp_path, name="whole")
    summary = json.loads((whole / corpus.SUMMARY_NAME).read_text(encoding="utf-8"))
    summary["config_realized_by_worker"] = {
        "0": {"cp_slope": SLOPE, "cp_draw_width": DRAW_WIDTH},
    }
    (whole / corpus.SUMMARY_NAME).write_text(json.dumps(summary), encoding="utf-8")
    checked = run_derive(whole, tmp_path / "whole-out", "uniform-d9")
    assert checked["cp_map"]["realized_workers_cross_checked"] == 1


def test_a_corpus_with_neither_record_is_refused(tmp_path: Path) -> None:
    """No summary AND no manifest: nothing to derive from, and it says which.

    The pre-``manifest.json`` corpora are exactly this shape, and the refusal
    has to name both documents -- a message that only mentions the summary sends
    the operator looking for the wrong file.
    """
    partial = one_row_corpus(tmp_path, complete=False)
    (partial / corpus.MANIFEST_NAME).unlink()

    with pytest.raises(derive.CorpusIntegrityError) as excinfo:
        run_derive(partial, tmp_path / "out", "uniform-d9")
    message = str(excinfo.value)
    assert corpus.SUMMARY_NAME in message
    assert corpus.MANIFEST_NAME in message


def test_a_shard_outside_the_snapshot_is_counted_and_never_read(
    tmp_path: Path,
) -> None:
    """⚑ A LIVE CORPUS IS A MOVING TARGET: only CLOSED shards are read.

    The unlisted file is what every live worker is holding open right now --
    opened ``"x"`` at its first write, listed only once it closes. Reading it
    would read a JSONL that is still being appended to. So the derivation takes
    the inventory's shards and COUNTS the rest.

    Mutation caught: sourcing partial mode's shard list from
    ``corpus_shard_paths(corpus_dir)`` (the directory glob) instead of the
    progress inventory -- the in-flight shard's rows are then silently derived,
    and the run's row count cannot be reproduced by any later read of the
    corpus.
    """
    listed_row = corpus_row(
        fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
        result=1.0, result_pgn="1-0", game_id=0,
    )
    partial = write_corpus(tmp_path, [listed_row], complete=False)

    # A second worker's shard, on disk and in NO progress file: the in-flight
    # shard, written with the generator's own writer and left unlisted.
    in_flight = corpus.ShardWriter(out_dir=partial, worker_id=9, shard_rows=1000)
    in_flight.write(corpus_row(
        fen=FEN_B, phases=[full_width_phase(FEN_B, {9: ramp(FEN_B, "h8g8")})],
        result=-1.0, result_pgn="1-0", game_id=77,
    ))
    in_flight.close()  # abandons: the game never ended, so nothing lists it
    assert in_flight.shards == []
    assert not (partial / corpus.progress_name(9)).exists()
    assert len(derive.corpus_shard_paths(partial)) == 2

    out = run_derive(partial, tmp_path / "out", "uniform-d9")

    detail = out["corpus"]["corpus_record_detail"]
    assert detail["shards_adopted"] == 1
    assert len(detail["shards_on_disk_not_in_inventory"]) == 1
    assert detail["shards_on_disk_not_in_inventory"][0].startswith("w09-")
    assert out["realized"]["rows_read"] == 1
    assert out["realized"]["rows_written"] == 1
    samples, _ = read_rows(tmp_path / "out")
    assert [int(s.game_id) for s in samples] == [0]


def test_the_inventory_resolves_shards_by_name_not_by_stored_path(
    tmp_path: Path,
) -> None:
    """⚑ The stored path is the PRODUCING machine's; the file is the local one.

    A progress line records an absolute path, and a corpus is routinely read
    from somewhere else -- copied off the box, mounted at another root, or (as
    here) worked on through a copy. ``resume_worker_state`` resolves by name for
    exactly this reason and so does this.

    Mutation caught: resolving ``record["path"]`` as written instead of taking
    its ``.name`` -- the derivation then reads the ORIGINAL directory's shards
    while claiming to have read this one, or refuses a corpus that is entirely
    intact.
    """
    original = one_row_corpus(tmp_path, complete=False, name="original")
    listed = [r for r in progress_lines(original) if r["path"] is not None]
    stored = Path(str(listed[0]["path"]))
    assert stored.is_absolute()
    assert stored.parent == original

    moved = tmp_path / "moved"
    shutil.copytree(original, moved)
    shutil.rmtree(original)  # the stored path now points at nothing
    assert not stored.exists()

    out = run_derive(moved, tmp_path / "out", "uniform-d9")
    assert out["corpus"]["corpus_record"] == derive.CORPUS_RECORD_PARTIAL
    assert out["corpus"]["corpus_record_detail"]["shards_adopted"] == 1
    assert out["realized"]["rows_written"] == 1


def test_a_progress_file_damaged_past_its_torn_tail_is_refused(
    tmp_path: Path,
) -> None:
    """The tolerance is the generator's, and it is as narrow as the failure.

    ``read_worker_progress`` accepts a final line cut short of its newline -- the
    only thing a ``kill -9`` during the single append can leave -- and refuses
    damage anywhere else. Imported rather than reimplemented, so this test is
    also the proof that the two readers cannot drift apart.

    ⚑ The pattern is a PHRASE, not the word "damaged". ``tmp_path`` is named
    after the test, so this test's own name is inside every path the error
    message quotes -- ``match="damaged"`` passed against the refusal for an
    entirely different reason, and the mutation that removes the check went
    uncaught until the pattern stopped overlapping the directory name.
    """
    partial = one_row_corpus(tmp_path, complete=False)
    progress = partial / corpus.progress_name(0)
    intact = progress.read_text(encoding="utf-8")
    progress.write_text("{not json at all\n" + intact, encoding="utf-8")

    with pytest.raises(
        derive.CorpusIntegrityError, match="the inventory it carries cannot be trusted",
    ):
        run_derive(partial, tmp_path / "out", "uniform-d9")


def test_a_torn_final_progress_line_is_tolerated_and_reported(
    tmp_path: Path,
) -> None:
    """The one tolerated damage, and it is never silent.

    A ``kill -9`` mid-append leaves the LAST line short of its newline. The
    shards listed above it are intact and are derived from; the torn line names
    a shard whose closure was never recorded, and the file it was in is NAMED in
    the output rather than dropped quietly.
    """
    rows = [
        corpus_row(
            fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
            result=1.0, result_pgn="1-0", game_id=index, ply=index,
        )
        for index in range(2)
    ]
    partial = write_corpus(tmp_path, rows, complete=False)
    progress = partial / corpus.progress_name(0)
    with open(progress, "a", encoding="utf-8") as handle:
        handle.write('{"codec": "zstd", "path": "w00-0000')  # no newline

    out = run_derive(partial, tmp_path / "out", "uniform-d9")

    detail = out["corpus"]["corpus_record_detail"]
    assert detail["progress_torn_tail_files"] == [corpus.progress_name(0)]
    assert detail["shards_adopted"] == 1
    assert out["realized"]["rows_written"] == 2


def test_a_manifest_with_no_progress_file_is_refused(tmp_path: Path) -> None:
    """A launched run that has closed no shard yet has nothing to derive from."""
    partial = one_row_corpus(tmp_path, complete=False)
    (partial / corpus.progress_name(0)).unlink()

    with pytest.raises(derive.CorpusIntegrityError, match="no shard inventory"):
        run_derive(partial, tmp_path / "out", "uniform-d9")


def test_a_shard_listed_by_two_workers_is_refused(tmp_path: Path) -> None:
    """One name, two inventories: two runs' progress files in one directory.

    Shard names are unique per worker and a resume continues the index rather
    than reusing it, so a repeat cannot happen inside one run.
    """
    partial = one_row_corpus(tmp_path, complete=False)
    listed = [r for r in progress_lines(partial) if r["path"] is not None]
    shutil.copyfile(
        partial / corpus.progress_name(0), partial / corpus.progress_name(1),
    )
    assert len(listed) == 1

    with pytest.raises(derive.CorpusIntegrityError, match="listed twice"):
        run_derive(partial, tmp_path / "out", "uniform-d9")


def test_a_partial_corpus_still_refuses_a_scheme_its_staircase_cannot_answer(
    tmp_path: Path,
) -> None:
    """The manifest's staircase is load-bearing, not decoration.

    Every run-level refusal a summary powers must still fire off a manifest, or
    "partial mode" would quietly be "fewer checks mode".
    """
    partial = one_row_corpus(tmp_path, complete=False)
    with pytest.raises(derive.CorpusIntegrityError, match="exceeds the corpus envelope"):
        run_derive(partial, tmp_path / "out", "uniform-d20")


def test_a_partial_corpus_with_a_bumped_row_schema_is_refused(
    tmp_path: Path,
) -> None:
    """The manifest's ``row_schema`` is checked exactly as the summary's is."""
    partial = one_row_corpus(tmp_path, complete=False)
    manifest = read_manifest(partial)
    manifest["row_schema"] = corpus.ROW_SCHEMA + 1
    manifest["config_sha256"] = corpus.stamp_sha256(manifest["config_requested"])
    (partial / corpus.MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )

    with pytest.raises(derive.CorpusIntegrityError, match="row schema"):
        run_derive(partial, tmp_path / "out", "uniform-d9")


# ── the realized --temp stamp, and what saturation does to it ────────────────


#: Two effective-cp values deep in the win-saturated tail whose q values differ
#: by ONE float64 ULP.  Swept off the production cp map, not invented; the test
#: below asserts the property rather than trusting the sweep.
SATURATED_BEST_CP = 6119.0
SATURATED_REST_CP = 5936.0


def test_a_saturated_row_is_held_out_of_the_temp_stamp_and_counted(
    tmp_path: Path,
) -> None:
    """⚑ F4: a won position's tau is quantisation, not a reading.

    In a position won outright every legal move's q is ±1.0, and the smallest
    nonzero spread the cp map can produce there is one float64 ULP. ``tau =
    gap / log_gap`` then returns a function of the MOVE COUNT -- this 5-move row
    reads **0.5** under ``--temp 1.0`` -- which is how 7 rows in 240k made a
    healthy run stamp ``min=0.625 max=1.007``.

    ⚑ THE TARGETS ARE UNTOUCHED: the row is still written, its policy is still
    ``softmax(q / temp)``, and only the temperature READ BACK off it is held
    out. The count is reported so the hold-out is visible.

    Mutation caught: deleting the ``q_spread`` guard from ``derive_row`` --
    the run then stamps ``min=0.5`` for a ``--temp 1.0`` derivation and counts
    zero skips.
    """
    saturated = {
        move: (SATURATED_BEST_CP if index == 0 else SATURATED_REST_CP)
        for index, move in enumerate(legal_ucis(FEN_B))
    }
    q = gate.q_from_effective_cp(
        np.array(list(saturated.values()), dtype=np.float64),
        slope=SLOPE, draw_width_cp=DRAW_WIDTH,
    )
    # The fixture's own premise, asserted rather than assumed: saturated at
    # +1.0, with a spread that is nonzero and far below the epsilon.
    assert np.allclose(q, 1.0, atol=1e-12)
    spread = derive.q_spread(q)
    assert 0.0 < spread < derive.TEMP_RECOVERY_MIN_Q_SPREAD
    # And the tau it would have contributed is nothing like the requested one.
    bogus = derive.recover_temp(q, derive.softmax_at_temp(q, temp=1.0))
    assert bogus is not None
    assert not math.isclose(bogus, 1.0, rel_tol=0.1)

    rows = [
        corpus_row(
            fen=FEN_B, phases=[full_width_phase(FEN_B, {9: saturated})],
            result=1.0, result_pgn="1-0", game_id=0, ply=0,
        ),
        # One healthy row, so the stamp has something real to report and the
        # test distinguishes "held out" from "recorded nothing at all".
        corpus_row(
            fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
            result=1.0, result_pgn="1-0", game_id=1, ply=1,
        ),
    ]
    corpus_dir = write_corpus(tmp_path, rows)
    out = run_derive(corpus_dir, tmp_path / "out", "uniform-d9", temp=1.0)

    realized = out["realized"]
    assert realized["rows_written"] == 2  # ⚑ the row is DERIVED, not dropped
    assert realized["temp_recovery_skipped_saturated"] == 1
    assert realized["temp_recovery_saturation_q_spread_epsilon"] == (
        derive.TEMP_RECOVERY_MIN_Q_SPREAD
    )
    recovered = realized["temp_recovered_from_emitted_policy"]
    assert recovered["n"] == 1
    assert recovered["min"] == pytest.approx(1.0, rel=1e-6)
    assert recovered["max"] == pytest.approx(1.0, rel=1e-6)
    assert recovered["mean"] == pytest.approx(1.0, rel=1e-6)
    assert "skipped(saturated)=1" in derive.format_summary(out)


def test_an_unsaturated_row_is_never_held_out(tmp_path: Path) -> None:
    """The hold-out must not fire on a healthy corpus.

    A guard that quietly swallowed ordinary rows would shrink ``n`` -- the very
    reading that proves ``--temp`` was applied -- and nothing would say so.
    """
    rows = [
        corpus_row(
            fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
            result=1.0, result_pgn="1-0", game_id=index, ply=index,
        )
        for index in range(3)
    ]
    corpus_dir = write_corpus(tmp_path, rows)
    out = run_derive(corpus_dir, tmp_path / "out", "uniform-d9", temp=0.5)

    realized = out["realized"]
    assert realized["temp_recovery_skipped_saturated"] == 0
    assert realized["temp_recovered_from_emitted_policy"]["n"] == 3
    assert realized["temp_recovered_from_emitted_policy"]["min"] == pytest.approx(0.5)
    assert realized["temp_recovered_from_emitted_policy"]["max"] == pytest.approx(0.5)


# ── the exploration floor ────────────────────────────────────────────────────
#
# ⚑ The arm this exists for is the ledger's floor arm -- `qtemp_0.04 + floor
# 0.002` as the shape re-analysis measured it (`3e655b762`), scheduled as arm 5
# `qtemp_0.067 + floor 0.002` after the reorder (`66f29b703`), which keeps the
# head at Gumbel sigma and makes the floor the single treatment.  T80's shape is
# a search-sharpened head plus an exploration floor, and a single temperature
# misses at both ends.  Every assertion below is about the EMITTED rows -- the
# flag is never read back out of the summary and called a measurement.

#: The ladder's floor, which is 0.002 in every version of the arm -- the
#: temperature is NOT: the reorder (``66f29b703``) runs arm 5 at the unchanged
#: Gumbel-sigma head 0.067 with the floor as the single treatment.
ARM_FLOOR = 0.002

#: The temperature these fixtures derive at, and it is a fixture choice rather
#: than an arm: 0.04 is sharp enough that the ninth move's unfloored mass
#: (1.6e-09) is stored as EXACTLY ZERO by the shard's float16, which is the
#: effect the floor exists to undo. A test at the arm's own 0.067 would assert
#: the same formula against a tail float16 can still represent.
SHARP_TEMP = 0.04


def hand_floored_policy(
    cps: dict[str, float], *, temp: float, floor: float,
) -> dict[str, float]:
    """``(1 - floor * n) * softmax(q / temp) + floor``, recomputed by hand."""
    plain = hand_policy(cps, temp=temp)
    n_legal = len(cps)
    return {
        move: (1.0 - floor * n_legal) * prob + floor for move, prob in plain.items()
    }


def ramp_corpus(tmp_path: Path, *, name: str = "corpus") -> tuple[Path, dict[str, float]]:
    """One nine-move row with a distinct cp per move -- the floor's test bed.

    ⚑ The 50cp step is chosen, not incidental: at ``--temp 0.04`` it puts the
    ninth move's unfloored mass at 1.6e-09, which the shard's float16 stores as
    EXACTLY ZERO. That is the tail the floor exists to keep alive, so the
    fixture has to contain one.
    """
    at9 = ramp(FEN_W, "f1e3", best_cp=400.0, step=50.0)
    row = corpus_row(
        fen=FEN_W, phases=[full_width_phase(FEN_W, {9: at9})],
        result=1.0, result_pgn="1-0",
    )
    return write_corpus(tmp_path, [row], name=name), at9


def test_floor_zero_emits_exactly_what_the_unfloored_tool_emitted(
    tmp_path: Path,
) -> None:
    """⚑ THE REGRESSION. ``--floor 0`` must not perturb one bit.

    Three ways, because "the identity is arithmetically exact" is an argument
    and this is a corpus: the array is not touched at all (object identity), a
    ``--floor 0`` run and a run with no flag at all agree bit for bit, and both
    agree with the softmax recomputed independently from the shared cp map.

    Mutation caught: dropping ``apply_floor``'s ``value == 0.0`` short circuit
    so the zero path computes ``1.0 * p + 0.0`` -- which is exact here, so the
    array-equality assertions still pass and the ``is`` assertion fails; and
    making the floor ``max(floor, 1e-12)`` -- which the ``is`` assertion and
    both equalities catch.
    """
    corpus_dir, at9 = ramp_corpus(tmp_path)
    plain = derive.softmax_at_temp(
        np.array([at9[m] for m in at9], dtype=np.float64), temp=SHARP_TEMP,
    )
    assert derive.apply_floor(plain, floor=0.0, n_legal=len(at9)) is plain

    unflagged = run_derive(corpus_dir, tmp_path / "none", "uniform-d9", temp=SHARP_TEMP)
    zero = run_derive(
        corpus_dir, tmp_path / "zero", "uniform-d9", "--floor", "0", temp=SHARP_TEMP,
    )
    got_none, _ = read_rows(tmp_path / "none")
    got_zero, _ = read_rows(tmp_path / "zero")
    assert np.array_equal(
        np.asarray(got_none[0].policy_target), np.asarray(got_zero[0].policy_target),
    )
    # And against the map, not against the other run: two identical bugs would
    # agree with each other.
    emitted = emitted_policy(got_zero[0], FEN_W)
    for move, expected in hand_policy(at9, temp=SHARP_TEMP).items():
        assert emitted[move] == float(np.float16(np.float32(expected)))
    # Every pre-existing reading is untouched; the new keys are additions.
    for key, value in unflagged["realized"].items():
        assert zero["realized"][key] == value or (
            isinstance(value, float) and math.isnan(value)
        )
    assert unflagged["realized"]["temp_recovered_from_emitted_policy"]["n"] == 1
    assert zero["floor_requested"] == 0.0
    assert zero["policy"]["temp_recovery_estimator"] == "closed_form_two_move"


def test_the_floor_reaches_every_legal_move_and_keeps_the_scheme_argmax(
    tmp_path: Path,
) -> None:
    """The floor's whole shape: a sharp head at 0.04 and 0.002 under every move.

    ⚑ The point of the floor is the TAIL: at temp 0.04 this row's ninth move
    carries 1.6e-09 unfloored and the shard's float16 stores it as EXACTLY
    ZERO -- the move is unrecoverable for the net, and the legal mask still
    names it. Floored it stores 0.00200. The argmax is the same move either
    way, because the floor is affine with a positive coefficient.

    Mutation caught: making ``apply_floor`` return ``probs + floor``
    (unnormalised) -- the row then sums to 1.018 and the mass assertion fails;
    and ``(1 - floor) * probs + floor`` (the n_legal dropped) -- the row sums
    to 1.016 and the same assertion fails.
    """
    corpus_dir, at9 = ramp_corpus(tmp_path)
    run_derive(corpus_dir, tmp_path / "flat", "uniform-d9", temp=SHARP_TEMP)
    run_derive(
        corpus_dir, tmp_path / "floored", "uniform-d9",
        "--floor", str(ARM_FLOOR), temp=SHARP_TEMP,
    )
    flat, _ = read_rows(tmp_path / "flat")
    floored, _ = read_rows(tmp_path / "floored")
    p_flat = emitted_policy(flat[0], FEN_W)
    p_floored = emitted_policy(floored[0], FEN_W)

    # ⚑ The bar is the floor AS FLOAT16 STORES IT (0.002 happens to round UP,
    # to 0.00200081; roughly half of all floors round down instead) -- the
    # cast is the only thing allowed to move it, in either direction.
    stored_floor = float(np.float16(ARM_FLOOR))
    assert min(p_floored.values()) >= stored_floor
    assert min(p_flat.values()) == 0.0  # the tail the float16 cast kills
    assert sorted(p_flat.values())[1] < ARM_FLOOR / 10.0
    assert max(p_floored, key=lambda m: p_floored[m]) == "f1e3"
    assert max(p_flat, key=lambda m: p_flat[m]) == "f1e3"
    assert sum(p_floored.values()) == pytest.approx(1.0, abs=2e-3)
    for move, expected in hand_floored_policy(
        at9, temp=SHARP_TEMP, floor=ARM_FLOOR,
    ).items():
        assert p_floored[move] == pytest.approx(expected, rel=1e-3)
    # The head paid for the tail: the best move gave up exactly what the eight
    # others gained.
    assert p_floored["f1e3"] < p_flat["f1e3"]


@pytest.mark.parametrize(
    "bad", [-1e-9, -1.0, float("nan"), float("inf"), 1.0 / 218.0, 0.005, 0.5],
)
def test_a_floor_that_could_starve_the_head_is_refused(bad: float) -> None:
    """The startup bound, at and above ``1 / MAX_LEGAL_MOVES``.

    ⚑ The refusal is against the CHESS-THEORETIC maximum, not against the
    corpus in hand: a nine-move row could carry ``--floor 0.05`` happily, and
    accepting it would mean a floor that is legal for one corpus and fatal for
    the next. See ``validate_floor``.

    Mutation caught: deleting the ``value * MAX_LEGAL_MOVES >= 1.0`` branch --
    every value from ``1/218`` up is then accepted and a 218-move row would
    emit a policy with a non-positive head coefficient.
    """
    with pytest.raises(ValueError, match="--floor"):
        derive.validate_floor(bad)


@pytest.mark.parametrize("good", [0.0, 1e-6, 0.002, 0.004])
def test_a_floor_inside_the_bound_is_accepted_unchanged(good: float) -> None:
    """The other side of the same gate -- and the arm's 0.002 is inside it."""
    assert derive.validate_floor(good) == good
    assert good * derive.MAX_LEGAL_MOVES < 1.0


def test_a_bad_floor_is_refused_before_the_corpus_is_touched(
    tmp_path: Path,
) -> None:
    """Startup, not first row: nothing is written and nothing is read.

    Mutation caught: moving ``validate_floor`` out of ``main`` and leaving it
    only in ``apply_floor`` -- the refusal then happens after the corpus record
    has been read and the out-dir created, which is the failure mode the temp
    validator has its own test for.
    """
    with pytest.raises(ValueError, match="--floor"):
        derive.main([
            "--corpus", str(tmp_path / "does_not_exist"),
            "--out", str(tmp_path / "out"),
            "--scheme", "uniform-d9",
            "--floor", "0.9",
        ])
    assert not (tmp_path / "out").exists()


def test_a_row_beyond_the_theoretic_bound_is_fatal_and_never_a_dropped_row() -> None:
    """``apply_floor``'s per-row check is the BOUND's falsifier.

    It cannot fire while ``MAX_LEGAL_MOVES`` really bounds chess, which is why
    it is written as a hard failure rather than as an envelope-style drop: a
    dropped row would make the emitted row set a function of ``--floor``, and
    two arms of a floor ladder would then differ in which positions they hold.

    Mutation caught: turning the raise into ``return probs`` -- the call below
    returns a policy whose head coefficient is -0.2 (mass on the best move goes
    NEGATIVE) instead of refusing.
    """
    probs = derive.softmax_at_temp(np.linspace(1.0, 0.0, 300), temp=0.5)
    with pytest.raises(derive.CorpusIntegrityError, match="falsifies that bound"):
        derive.apply_floor(probs, floor=0.004, n_legal=300)


@pytest.mark.parametrize("temp", [0.04, 0.3, 1.0])
@pytest.mark.parametrize("floor", [0.0, 0.0005, 0.002, 0.004])
def test_recover_floor_and_temp_inverts_the_floored_policy(
    temp: float, floor: float,
) -> None:
    """Both knobs, read back out of the emitted row with neither echoed in.

    ⚑ ``floor=0.0`` is in the grid on purpose: the estimator run against an
    UNFLOORED policy must read the floor as 0, which is precisely what makes it
    able to notice a floor that was parsed and never applied.

    Mutation caught: replacing the recovered scale with ``1.0`` (so the floor
    is reported as 0 always) -- every nonzero row of this grid fails.
    """
    q = np.linspace(0.9, -0.6, 12)
    probs = derive.apply_floor(
        derive.softmax_at_temp(q, temp=temp), floor=floor, n_legal=12,
    )
    reading = derive.recover_floor_and_temp(q, probs, n_legal=12)
    assert reading is not None
    got_temp, got_floor = reading
    assert got_temp == pytest.approx(temp, rel=1e-6)
    assert got_floor == pytest.approx(floor, abs=1e-9)


def test_the_floored_estimator_refuses_a_row_it_cannot_read() -> None:
    """Fewer than three distinct values, and a difference that is pure rounding.

    Both return None rather than a number: an estimator that answered anyway
    would put rounding noise into the stamp a reader uses to decide whether the
    floor applied at all.

    Mutation caught: dropping the ``FLOOR_RECOVERY_MIN_REL_GAP`` conditioning
    check (leaving only ``> 0``) -- the pinned row below then reads a
    temperature of **0.0141938 for a 0.014 derivation, 1.4% high**, off a
    difference that is a single float64 ULP, and the run stamps it as a
    measurement instead of counting the row as unreadable. ⚑ The FLOOR it
    recovers there is still 0.002: the two readings are not equally
    conditioned, and the gate is sized for the fragile one.
    """
    two_valued = np.array([0.9, 0.9, 0.1, 0.1, 0.1], dtype=np.float64)
    assert derive.distinct_values(two_valued) == 2
    assert derive.recover_floor_and_temp(
        two_valued,
        derive.apply_floor(
            derive.softmax_at_temp(two_valued, temp=0.5), floor=0.002, n_legal=5,
        ),
        n_legal=5,
    ) is None

    # Four distinct values, but at temp 0.014 the second and fourth are 4.3e-19
    # apart on top of a floor of 2e-3 -- ONE float64 ULP, strictly positive,
    # and 2.2e-16 of the probability it came out of. The row is arithmetic
    # noise wearing the shape of a reading.
    pinned = np.array([1.0, 0.4, 0.4, 0.0, -0.5], dtype=np.float64)
    probs = derive.apply_floor(
        derive.softmax_at_temp(pinned, temp=0.014), floor=0.002, n_legal=5,
    )
    assert derive.distinct_values(pinned) == 4
    gap = float(probs[1] - probs[3])
    assert 0.0 < gap / float(probs[1]) < derive.FLOOR_RECOVERY_MIN_REL_GAP
    assert derive.recover_floor_and_temp(pinned, probs, n_legal=5) is None


def test_the_floor_take_effect_stamp_reads_the_rows_not_the_flag(
    tmp_path: Path,
) -> None:
    """⚑⚑ THE TAKE-EFFECT PROOF, and it must distinguish two corpora.

    The same corpus, the same temperature, derived twice: with the arm's floor
    and without it. Both instruments have to separate them -- the algebraic
    recovery (exact, on the rows that can carry it) and the stored minimum mass
    (coarse, on every row) -- and the recovered floor has to be the floor that
    was EMITTED, not the one that was asked for.

    Mutation caught: making ``apply_floor`` a no-op (``return probs``) with the
    flag still parsed, stamped in the summary and stamped on the shards -- the
    run then looks floored in every echoed field, and this test fails on the
    recovered floor (0 instead of 0.002) and on the stored min mass (0 instead
    of 0.00199).
    """
    corpus_dir, _ = ramp_corpus(tmp_path)
    floored = run_derive(
        corpus_dir, tmp_path / "floored", "uniform-d9",
        "--floor", str(ARM_FLOOR), temp=SHARP_TEMP,
    )
    flat = run_derive(
        corpus_dir, tmp_path / "flat", "uniform-d9", temp=SHARP_TEMP,
    )

    read = floored["realized"]["floor_recovered_from_emitted_policy"]
    assert read["n"] == 1
    assert read["min"] == pytest.approx(ARM_FLOOR, abs=1e-9)
    assert read["max"] == pytest.approx(ARM_FLOOR, abs=1e-9)
    assert floored["realized"]["floor_recovery_skipped_few_values"] == 0
    assert floored["realized"]["floor_recovery_skipped_ill_conditioned"] == 0
    # The temperature survives the floor: the joint estimator recovers it too.
    recovered = floored["realized"]["temp_recovered_from_emitted_policy"]
    assert recovered["n"] == 1
    assert recovered["min"] == pytest.approx(SHARP_TEMP, rel=1e-6)
    assert floored["policy"]["temp_recovery_estimator"] == "floored_three_move_bisection"

    # The unfloored run: the algebraic estimator does not run at all, and the
    # coarse one reads the softmax's own tail instead of the floor.
    assert flat["realized"]["floor_recovered_from_emitted_policy"]["n"] == 0
    assert math.isnan(flat["realized"]["floor_recovered_from_emitted_policy"]["min"])
    floored_mass = floored["realized"]["policy_min_legal_prob_stored"]
    flat_mass = flat["realized"]["policy_min_legal_prob_stored"]
    assert floored_mass["n"] == flat_mass["n"] == 1
    assert floored_mass["min"] == pytest.approx(float(np.float16(ARM_FLOOR)))
    assert flat_mass["min"] == 0.0
    assert "floor recovered from the emitted policy: n=1" in derive.format_summary(
        floored,
    )


def test_the_floor_is_banked_in_the_summary_and_on_every_shard(
    tmp_path: Path,
) -> None:
    """Attribution: a derived corpus says which floor made it, shard by shard.

    ⚑ Both places, because they answer different questions. The summary
    attributes the RUN; the zarr attrs travel with a shard that gets copied
    somewhere else, where a floored and an unfloored shard are otherwise
    identical in every stamp they carry.

    Mutation caught: dropping ``derive_floor`` from ``_stamp_shard_attrs`` --
    the summary still reports the floor and the shards become unattributable;
    dropping ``floor_requested`` from ``build_summary`` -- the reverse.
    """
    corpus_dir, _ = ramp_corpus(tmp_path)
    out = run_derive(
        corpus_dir, tmp_path / "out", "uniform-d9",
        "--floor", str(ARM_FLOOR), temp=SHARP_TEMP,
    )
    assert out["floor_requested"] == ARM_FLOOR
    assert out["policy"]["floor"] == ARM_FLOOR
    assert out["policy"]["floor_max_legal_moves_bound"] == derive.MAX_LEGAL_MOVES
    assert "floor" in out["policy"]["construction"]
    for path in iter_shard_paths(tmp_path / "out"):
        attrs = dict(zarr.open_group(str(path), mode="r").attrs)
        assert attrs["derive_floor"] == ARM_FLOOR
        assert attrs["derive_temp"] == SHARP_TEMP
    assert f"floor={ARM_FLOOR}" in derive.format_summary(out)


def test_a_row_the_floored_estimator_cannot_read_is_counted_and_still_written(
    tmp_path: Path,
) -> None:
    """Coverage is reported, never implied.

    A two-valued row cannot determine (tau, floor) jointly, so it is held out
    of the reading and counted -- and it is still DERIVED, with the floor on
    every one of its legal moves. A corpus of such rows would stamp ``n=0``
    beside a nonzero skip count, which is a different fact from a floor that
    failed to apply, and the coarse stamp still covers it.

    Mutation caught: counting the hold-out into ``floor_recovered_n`` with a
    0.0 reading instead of skipping it -- the summary then reports a floor of
    0.001 (the mean of a real 0.002 and a fabricated 0) on a corpus whose rows
    all carry 0.002, and the skip counter reads zero.
    """
    two_valued = {
        move: (400.0 if index == 0 else 100.0)
        for index, move in enumerate(legal_ucis(FEN_W))
    }
    rows = [
        corpus_row(
            fen=FEN_W, phases=[full_width_phase(FEN_W, {9: two_valued})],
            result=1.0, result_pgn="1-0", game_id=0, ply=0,
        ),
        corpus_row(
            fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
            result=1.0, result_pgn="1-0", game_id=1, ply=1,
        ),
    ]
    corpus_dir = write_corpus(tmp_path, rows)
    out = run_derive(
        corpus_dir, tmp_path / "out", "uniform-d9",
        "--floor", str(ARM_FLOOR), temp=SHARP_TEMP,
    )

    realized = out["realized"]
    assert realized["rows_written"] == 2  # ⚑ held out of the STAMP, not dropped
    assert realized["floor_recovery_skipped_few_values"] == 1
    assert realized["floor_recovered_from_emitted_policy"]["n"] == 1
    assert realized["floor_recovered_from_emitted_policy"]["min"] == pytest.approx(
        ARM_FLOOR, abs=1e-9,
    )
    # The coarse stamp covers BOTH rows, including the one held out above.
    assert realized["policy_min_legal_prob_stored"]["n"] == 2
    assert realized["policy_min_legal_prob_stored"]["min"] == pytest.approx(
        float(np.float16(ARM_FLOOR)),
    )


# ── PR #486 review hardening: the estimator must refuse what it cannot read ──


def test_tiny_spread_rows_are_skipped_not_stamped() -> None:
    """⚑⚑ THE ILL-CONDITIONED BAND, found independently by two reviewers.

    A q spread just above the saturation gate sits in the softmax's LINEAR
    regime, where the emitted row carries two numbers (slope, offset) and can
    never determine three (tau, floor, scale) -- the three-point ratio lies on
    ``R(tau)``'s flat hot tail and the inverse returns rounding noise.  Before
    the ``FLOOR_RECOVERY_MIN_SPREAD_PER_TAU`` gate these rows STAMPED the
    noise: spread 1e-9 at true (0.067, 0.002) recovered (0.0398, 0.035), and
    the cp-band row below recovered (0.087, -0.022) -- a negative floor no
    emission can carry.

    Mutation caught: deleting the spread-per-tau gate re-stamps every row in
    the band and each ``is None`` below fails.
    """
    for spread in (1e-9, 3e-9, 1e-8, 1e-7):
        q = np.linspace(0.0, spread, 12)
        probs = derive.apply_floor(
            derive.softmax_at_temp(q, temp=0.067), floor=ARM_FLOOR, n_legal=12,
        )
        assert derive.recover_floor_and_temp(q, probs, n_legal=12) is None

    # The production shape, not a synthetic one: every move winning by a mile
    # (cp 3000..3200) saturates the cp->q map to spread ~6.5e-8 -- distinct
    # values, above the saturation gate, and unreadable.
    cps = np.linspace(3000.0, 3200.0, 12)
    q = np.asarray(
        gate.q_from_effective_cp(cps, slope=SLOPE, draw_width_cp=DRAW_WIDTH),
        dtype=np.float64,
    )
    assert derive.q_spread(q) > derive.TEMP_RECOVERY_MIN_Q_SPREAD
    probs = derive.apply_floor(
        derive.softmax_at_temp(q, temp=0.067), floor=ARM_FLOOR, n_legal=12,
    )
    assert derive.recover_floor_and_temp(q, probs, n_legal=12) is None

    # And the gate does not eat honest rows: a real spread recovers exactly.
    q = np.linspace(0.0, 1.5, 12)
    probs = derive.apply_floor(
        derive.softmax_at_temp(q, temp=0.067), floor=ARM_FLOOR, n_legal=12,
    )
    reading = derive.recover_floor_and_temp(q, probs, n_legal=12)
    assert reading is not None
    tau, floor = reading
    assert tau == pytest.approx(0.067, rel=1e-6)
    assert floor == pytest.approx(ARM_FLOOR, abs=1e-9)


def test_an_impossible_recovered_floor_is_refused_not_stamped() -> None:
    """Belt and braces under the spread gate: no emission can carry a negative
    floor (``validate_floor`` refuses it at startup), so a reading that lands
    below ``-FLOOR_RECOVERY_FLOOR_TOL`` is the arithmetic failing, not a fact
    about the corpus.  The take-effect zero must SURVIVE the bound: an
    unfloored row's honest reading is rounding residue of either sign.
    """
    q = np.linspace(0.0, 1.5, 12)
    s = derive.softmax_at_temp(q, temp=0.067)
    # Hand-build the emission apply_floor would refuse to make.
    impossible = (1.0 - (-0.01) * 12) * s + (-0.01)
    assert derive.recover_floor_and_temp(q, impossible, n_legal=12) is None
    # The unapplied-floor reading (~0, either sign) still passes.
    reading = derive.recover_floor_and_temp(q, s, n_legal=12)
    assert reading is not None
    assert reading[1] == pytest.approx(0.0, abs=1e-9)


def test_ill_conditioned_rows_reach_the_summary_counter(tmp_path: Path) -> None:
    """⚑ End to end, because the increment was a surviving mutant: ``+= 1`` ->
    ``+= 0`` passed the whole file before this test.  A row every move of
    which wins by a mile is derived and WRITTEN, held out of the stamp, and
    counted -- next to a readable row that keeps the run's take-effect proof
    alive.
    """
    winning = {
        move: 3000.0 + 25.0 * index
        for index, move in enumerate(legal_ucis(FEN_W))
    }
    rows = [
        corpus_row(
            fen=FEN_W, phases=[full_width_phase(FEN_W, {9: winning})],
            result=1.0, result_pgn="1-0", game_id=0, ply=0,
        ),
        corpus_row(
            fen=FEN_W, phases=[full_width_phase(FEN_W, {9: ramp(FEN_W, "f1e3")})],
            result=1.0, result_pgn="1-0", game_id=1, ply=1,
        ),
    ]
    corpus_dir = write_corpus(tmp_path, rows)
    out = run_derive(
        corpus_dir, tmp_path / "out", "uniform-d9",
        "--floor", str(ARM_FLOOR), temp=SHARP_TEMP,
    )
    realized = out["realized"]
    assert realized["rows_written"] == 2
    assert realized["floor_recovery_skipped_ill_conditioned"] == 1
    assert realized["floor_recovered_from_emitted_policy"]["n"] == 1
    assert realized["floor_recovered_from_emitted_policy"]["mean"] == pytest.approx(
        ARM_FLOOR, abs=1e-9,
    )
    # The four buckets partition the written rows (the accounting the summary
    # comment promises).
    assert (
        realized["temp_recovery_skipped_saturated"]
        + realized["floor_recovery_skipped_few_values"]
        + realized["floor_recovery_skipped_ill_conditioned"]
        + realized["floor_recovered_from_emitted_policy"]["n"]
    ) == realized["rows_written"]


def test_a_floor_that_does_not_take_effect_kills_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑⚑ ENFORCED, not published: the exact failure the stamp exists for --
    parsed, validated, echoed everywhere, never applied -- must now kill the
    run before the summary is written, not hand a human two numbers to diff.

    Mutation caught: deleting the ``enforce_take_effect`` call in ``derive``
    turns this back into a run that exits 0 with a lying corpus on disk.
    """
    corpus_dir, _ = ramp_corpus(tmp_path)
    monkeypatch.setattr(
        derive, "apply_floor", lambda probs, *, floor, n_legal: probs,
    )
    with pytest.raises(derive.CorpusIntegrityError, match="did not take effect"):
        derive.main([
            "--corpus", str(corpus_dir),
            "--out", str(tmp_path / "out"),
            "--scheme", "uniform-d9",
            "--temp", str(SHARP_TEMP),
            "--floor", str(ARM_FLOOR),
        ])
    # Died BEFORE the summary: the documented "this run DIED" state.
    assert not (tmp_path / "out" / derive.SUMMARY_NAME).exists()


def test_a_floored_run_no_row_can_vouch_for_is_refused(tmp_path: Path) -> None:
    """A floored run whose every row is unreadable has NO exact take-effect
    proof, and absent is not passed.  (A real corpus cannot trip this --
    run02's first 20k rows leave 17k+ readable.)
    """
    winning = {
        move: 3000.0 + 25.0 * index
        for index, move in enumerate(legal_ucis(FEN_W))
    }
    row = corpus_row(
        fen=FEN_W, phases=[full_width_phase(FEN_W, {9: winning})],
        result=1.0, result_pgn="1-0",
    )
    corpus_dir = write_corpus(tmp_path, [row])
    with pytest.raises(derive.CorpusIntegrityError, match="floor_recovered_n == 0"):
        derive.main([
            "--corpus", str(corpus_dir),
            "--out", str(tmp_path / "out"),
            "--scheme", "uniform-d9",
            "--temp", str(SHARP_TEMP),
            "--floor", str(ARM_FLOOR),
        ])


@pytest.mark.parametrize("bad", [1e-9, 2.9e-8])
def test_a_floor_that_vanishes_in_shard_storage_is_refused(bad: float) -> None:
    """A positive floor below half of float16's smallest subnormal serializes
    to exactly zero on the shard path, so the CLI would accept a flag the
    trainer can never see.  Refused at startup, like every other bad floor.
    """
    with pytest.raises(ValueError, match="vanishes in shard storage"):
        derive.validate_floor(bad)
    # The smallest storable floor survives.
    assert derive.validate_floor(6e-8) == 6e-8


def test_the_stamps_use_the_shards_two_step_cast() -> None:
    """⚑ Just above 2**-25 is the trap: the direct float64->float16 cast
    rounds it UP to float16's smallest subnormal, but float32 first rounds it
    DOWN onto the tie exactly, and the tie goes to even -- ZERO.  A one-step
    stamp there claims a tail the trainer reads as nothing.  ``shard_stored``
    is the one cast every stamp goes through, so it must take the shard's
    two-step path.
    """
    tricky = np.asarray([2.0 ** -25 * (1.0 + 2.0 ** -30)], dtype=np.float64)
    assert float(np.float16(tricky[0])) > 0.0  # the one-step cast says "alive"
    assert float(derive.shard_stored(tricky)[0]) == 0.0  # the shard says "dead"


# ── the value round: V0 / A / B / C ──────────────────────────────────────────
#
# ⚑ EVERY EXPECTED VECTOR BELOW IS RECOMPUTED FROM THE SHARED cp->WDL MAP, never
# stored as a golden array -- the same rule the policy tests above follow, and
# for the same reason: a golden array pins the map as well as the arm, so a
# deliberate change to one would read as a failure of the other.  What IS pinned
# as a literal is the ledger's frozen PARAMETERS, which is the opposite kind of
# claim and belongs in a test that fails when they drift.

#: The clean/blunder cps used throughout, and the q they land on (slope 0.006,
#: draw width 120): q(120) = 0.30846, q(100) = 0.25922, q(0) = 0.  So a played
#: move at 100 against a top of 120 is a regret of 0.049 -- ordinary sampling
#: noise, well under the 0.27 boundary -- and one at 0 against the same top is
#: 0.308, over it.  Both are asserted in ``test_the_toy_games_regrets_bracket_the_boundary``
#: rather than left as arithmetic in a comment.
CP_TOP = 120.0
CP_CLEAN = 100.0
CP_BLUNDER = 0.0


def wdl_of_cp(cp: float) -> np.ndarray:
    """One cp -> (W, D, L), through the same object the tool uses."""
    return np.asarray(
        gen.cp_to_wdl_array(
            np.array([float(cp)], dtype=np.float64),
            slope=SLOPE,
            draw_width_cp=DRAW_WIDTH,
        ),
        dtype=np.float64,
    ).reshape(-1)[:3]


def q_of_cp(cp: float) -> float:
    return float(
        gate.q_from_effective_cp(
            np.array([float(cp)], dtype=np.float64),
            slope=SLOPE,
            draw_width_cp=DRAW_WIDTH,
        )[0],
    )


def onehot(index: int) -> np.ndarray:
    return derive.onehot_wdl(index)


def flipped(vector: np.ndarray) -> np.ndarray:
    return derive.flip_wdl(vector)


class Ply:
    """One banked ply of a synthetic game, spelled in cp.

    ``top9``/``top8``/``top7`` are the BEST value at each full-width rung, which
    is what the instability ``u`` is built from; ``played`` is the cp the row's
    played move carries at d9, which is what the regret is built from.  Anything
    left at its default is the "nothing interesting here" reading: a settled
    teacher (all three rungs agree) that played a best move (zero regret).
    """

    def __init__(
        self,
        *,
        top9: float = CP_TOP,
        top8: float | None = None,
        top7: float | None = None,
        played: float | None = None,
        drop_d7: bool = False,
        played_off_book: bool = False,
        played_absent: bool = False,
    ) -> None:
        self.top9 = top9
        self.top8 = top9 if top8 is None else top8
        self.top7 = (self.top8 if top7 is None else top7)
        self.played = top9 if played is None else played
        self.drop_d7 = drop_d7
        self.played_off_book = played_off_book
        self.played_absent = played_absent

    @property
    def u(self) -> float:
        """The instability this ply's rungs produce, in q units."""
        return abs(q_of_cp(self.top9) - q_of_cp(self.top8)) + abs(
            q_of_cp(self.top8) - q_of_cp(self.top7),
        )

    @property
    def regret(self) -> float:
        return q_of_cp(self.top9) - q_of_cp(self.played)

    @property
    def weight(self) -> float:
        """The frozen map's blend weight for this ply."""
        if self.drop_d7:
            return 0.0
        return derive.QZ_W_MIN + derive.QZ_W_SPAN * min(
            self.u / derive.QZ_U_SCALE, 1.0,
        )


def build_game(
    plies: list[Ply],
    *,
    result_white: float,
    game_id: int = 0,
    worker_id: int = 0,
    start_fen: str = FEN_W,
    step: float = 20.0,
) -> list[dict[str, Any]]:
    """A synthetic GAME: consecutive real positions, one banked row per ply.

    ⚑ The positions are reached by PUSHING the played move, so ``stm``
    alternates for real and the POV rotations under test are the ones a corpus
    would produce.  ``result`` is stored per row in that row's own seat, exactly
    as ``result_from_pov`` stores it -- which is why the arms never rotate ``Z``
    and why a test that hand-rotated it would be testing the fixture.
    """
    board = chess.Board(start_fen)
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(plies):
        legal = legal_ucis(board.fen())
        played = legal[0]
        best = legal[1]
        per_depth: dict[int, dict[str, float]] = {}
        for depth, top in ((7, spec.top7), (8, spec.top8), (9, spec.top9)):
            if depth == 7 and spec.drop_d7:
                continue
            values = ramp(board.fen(), best, best_cp=top, step=step)
            if depth == 9:
                values[played] = spec.played
            per_depth[depth] = values
        if spec.played_absent:
            named: str | None = None
        elif spec.played_off_book:
            # A uci no block ever carried: the row banked a played move and the
            # d9 block cannot price it.  Legality is irrelevant here -- the
            # support check compares the BANKED set against the legal set, and
            # this touches neither.
            named = "a1a2"
        else:
            named = played
        rows.append(
            corpus_row(
                fen=board.fen(),
                phases=[full_width_phase(board.fen(), per_depth)],
                result=result_white if board.turn == chess.WHITE else -result_white,
                game_id=game_id,
                ply=index,
                played_move=named,
                worker_id=worker_id,
            ),
        )
        board.push(chess.Move.from_uci(played))
    return rows


def facts_for(rows: list[dict[str, Any]]) -> list[Any]:
    """The value facts the tool would derive for these rows, via the tool."""
    options = derive.DeriveOptions(
        scheme=derive.parse_scheme("uniform-d9"),
        temp=1.0,
        cp_slope=SLOPE,
        cp_draw_width=DRAW_WIDTH,
        limit=0,
        seed=1,
        rows_per_shard=64,
        max_envelope_misses=0,
    )
    deriver = derive.TargetDeriver(options)
    derived = [deriver.derive_row(row) for row in rows]
    return [item.facts for item in derived if item is not None]


def targets_for(
    rows: list[dict[str, Any]],
    *,
    value_scheme: str = derive.VALUE_SCHEME_QZSEGMENT,
    **qz: Any,
) -> list[np.ndarray]:
    targets, _ = derive.game_value_targets(
        facts_for(rows), value_scheme=value_scheme, params=derive.QzParams(**qz),
    )
    return targets


def value_rows(out_dir: Path) -> dict[tuple[int, int], Any]:
    """The emitted rows, keyed back by (game_id, ply) through the RIG's reader."""
    samples, _ = read_rows(out_dir)
    return {(int(s.game_id), int(s.ply_index)): s for s in samples}


# ── the frozen parameters ────────────────────────────────────────────────────


def test_the_frozen_value_round_parameters_are_the_ledgers() -> None:
    """⚑ THE LITERALS, pinned against the ledger's 2026-08-31 prereg and its
    two same-day amendments.  Every one of them is a flag with a default, so
    the only thing standing between a quiet edit here and four arms derived
    under parameters no entry describes is this assertion.
    """
    assert derive.QZ_R_BOUNDARY == 0.27
    assert derive.QZ_U_SCALE == 0.05
    assert derive.QZ_W_MIN == 0.5
    assert derive.QZ_W_SPAN == 0.45
    assert derive.QZ_R_FREE_CALIBRATED == 0.06
    assert derive.QZ_TAU_R_CALIBRATED == 0.25
    assert (
        derive.QZ_GATE_DEPTH_TOP,
        derive.QZ_GATE_DEPTH_MID,
        derive.QZ_GATE_DEPTH_LOW,
    ) == (9, 8, 7)
    # ⚑ The frozen values live on QzParams, and the CLI carries PRESENCE
    # SENTINELS rather than a second copy of them -- so "was this knob passed"
    # is answerable, and there is exactly one place the literals can drift from.
    frozen = derive.QzParams()
    assert frozen.r_boundary == derive.QZ_R_BOUNDARY
    assert frozen.u_scale == derive.QZ_U_SCALE
    assert frozen.w_const is None
    assert frozen.no_boundary is False
    args = derive.build_parser().parse_args(
        ["--corpus", "c", "--out", "o", "--scheme", "uniform-d9"],
    )
    assert args.value_scheme == derive.VALUE_SCHEME_SEARCH
    assert args.qz_r_boundary is None
    assert args.qz_u_scale is None
    assert args.qz_w_const is None
    assert args.qz_no_boundary is None


def test_the_boundary_is_the_frozen_gates_half_credit_point() -> None:
    """0.27 is not a free parameter: it is where the prereg's soft gate
    ``exp(-[max(0, r - r_free)/tau_r]^2)`` reaches c = 0.5.  The derivation is
    recomputed here rather than asserted in prose, so a future edit to either
    calibrated constant has to face the number it implies.
    """
    derived = derive.QZ_R_FREE_CALIBRATED + derive.QZ_TAU_R_CALIBRATED * math.sqrt(
        math.log(2.0),
    )
    assert math.isclose(derived, 0.2681386527894244, rel_tol=1e-12)
    assert abs(derived - derive.QZ_R_BOUNDARY) < 0.002
    # And it really is the half-credit point of the gate it came from.
    gate_at_boundary = math.exp(
        -((max(0.0, derived - derive.QZ_R_FREE_CALIBRATED)
           / derive.QZ_TAU_R_CALIBRATED) ** 2),
    )
    assert math.isclose(gate_at_boundary, 0.5, rel_tol=1e-12)


def test_the_toy_games_regrets_bracket_the_boundary() -> None:
    """The fixture's two cps must land on opposite sides of 0.27, or every
    boundary test below would be asserting the same branch twice.
    """
    assert Ply(top9=CP_TOP, played=CP_CLEAN).regret < derive.QZ_R_BOUNDARY
    assert Ply(top9=CP_TOP, played=CP_BLUNDER).regret > derive.QZ_R_BOUNDARY


@pytest.mark.parametrize(
    ("u", "expected"),
    [
        (0.0, 0.5),
        (0.025, 0.725),
        (0.05, 0.95),
        (0.5, 0.95),
    ],
)
def test_the_instability_map_is_the_frozen_one(u: float, expected: float) -> None:
    """``w = 0.5 + 0.45 * min(u / 0.05, 1)``, saturating at the top."""
    assert math.isclose(
        derive.segment_blend_weight(u, u_scale=derive.QZ_U_SCALE), expected,
    )


def test_an_unmeasurable_instability_is_pure_q_not_the_maps_floor() -> None:
    """⚑ The fail direction. ``u`` unknown is w = 0 (all Q), NOT u = 0's w = 0.5,
    which would hand half the target to a retrospective value on exactly the
    rows whose reliability could not be established.
    """
    assert derive.segment_blend_weight(None, u_scale=derive.QZ_U_SCALE) == 0.0
    assert derive.segment_blend_weight(0.0, u_scale=derive.QZ_U_SCALE) == 0.5


# ── the hand-computed toy game ───────────────────────────────────────────────

#: A four-ply game White wins.  Ply 1's teacher is UNSTABLE (d9 120 against d8
#: and d7 at 0), so its blend weight saturates at 0.95 while every other ply's
#: rungs agree and weight 0.5 -- one game exercising both ends of the map.
#: Every played move is clean (regret 0.049 < 0.27), so arm C's scan runs to the
#: end from every row.
TOY_PLIES = [
    Ply(top9=CP_TOP, played=CP_CLEAN),
    Ply(top9=CP_TOP, top8=CP_BLUNDER, top7=CP_BLUNDER, played=CP_CLEAN),
    Ply(top9=CP_TOP, played=CP_CLEAN),
    Ply(top9=CP_TOP, played=CP_CLEAN),
]


def toy_rows(**kwargs: Any) -> list[dict[str, Any]]:
    return build_game(TOY_PLIES, result_white=1.0, **kwargs)


def test_the_toy_games_seats_alternate_and_carry_their_own_result() -> None:
    """The fixture's premise, checked before anything is built on it: White
    wins, so the row-own-seat result alternates +1 / -1 down the game.
    """
    rows = toy_rows()
    assert [row["stm"] for row in rows] == ["w", "b", "w", "b"]
    assert [row["result"] for row in rows] == [1.0, -1.0, 1.0, -1.0]
    assert [derive.wdl_target_from_result(row["result"]) for row in rows] == [0, 2, 0, 2]


def test_the_control_arm_writes_the_searched_value_and_nothing_else() -> None:
    """V0: every row is ``Q_t`` = the d9 top's WDL, hand-computed."""
    targets = targets_for(toy_rows(), value_scheme=derive.VALUE_SCHEME_SEARCH)
    expected = wdl_of_cp(CP_TOP)
    assert len(targets) == 4
    for target in targets:
        np.testing.assert_allclose(target, expected, atol=1e-12)


def test_arm_a_is_half_the_searched_value_and_half_the_outcome() -> None:
    """A: ``0.5 * Q_t + 0.5 * Z_t``, and ``Z`` is the row's OWN seat -- so the
    White rows blend toward a win and the Black rows toward a loss.
    """
    targets = targets_for(toy_rows(), value_scheme=derive.VALUE_SCHEME_QZ50)
    q = wdl_of_cp(CP_TOP)
    for index, target in enumerate(targets):
        z = onehot(0 if index % 2 == 0 else 2)
        np.testing.assert_allclose(target, 0.5 * q + 0.5 * z, atol=1e-12)


def test_arm_b_ramps_the_outcome_in_by_ply() -> None:
    """B: ``w = ply / terminal_ply`` over the game's last BANKED ply (3 here),
    so row 0 is pure ``Q`` and row 3 is pure ``Z``.
    """
    targets = targets_for(toy_rows(), value_scheme=derive.VALUE_SCHEME_QZPHASE)
    q = wdl_of_cp(CP_TOP)
    for index, target in enumerate(targets):
        weight = index / 3.0
        z = onehot(0 if index % 2 == 0 else 2)
        np.testing.assert_allclose(target, (1.0 - weight) * q + weight * z, atol=1e-12)
    np.testing.assert_allclose(targets[0], q, atol=1e-12)
    np.testing.assert_allclose(targets[3], onehot(2), atol=1e-12)


def test_arm_c_blends_the_terminal_outcome_at_the_instability_weight() -> None:
    """C on a wholly clean game: EVERY row's future is the game result in its
    own seat, at the weight its own ``u`` sets -- 0.95 on the unstable ply 1,
    0.5 on the three settled ones.  ⚑ The weights are per ROW; a scheme that
    used the game's mean ``u``, or the terminal row's, would pass every other
    assertion in this file and fail this one.
    """
    targets = targets_for(toy_rows())
    q = wdl_of_cp(CP_TOP)
    weights = [0.5, 0.95, 0.5, 0.5]
    assert [round(ply.weight, 12) for ply in TOY_PLIES] == weights
    for index, (target, weight) in enumerate(zip(targets, weights)):
        z = onehot(0 if index % 2 == 0 else 2)
        np.testing.assert_allclose(target, (1.0 - weight) * q + weight * z, atol=1e-12)


def test_a_long_clean_segment_reaches_row_zero_undiminished() -> None:
    """⚑⚑ THE AMENDMENT'S WHOLE POINT, and the mutation target.

    Ten clean plies.  Row 0's future must be the TERMINAL outcome at full
    strength -- identical to the four-ply game's row 0, because the segment
    carries no length term at all.  The per-step lambda-return this arm replaced
    would deliver ``lambda**9`` of it here and ``lambda**3`` there, so any
    reintroduced per-step factor makes these two numbers differ and fails.
    """
    long_game = build_game([Ply(played=CP_CLEAN) for _ in range(10)], result_white=1.0)
    short_game = build_game([Ply(played=CP_CLEAN) for _ in range(4)], result_white=1.0)
    long_targets = targets_for(long_game)
    short_targets = targets_for(short_game)
    q = wdl_of_cp(CP_TOP)
    expected = 0.5 * q + 0.5 * onehot(0)
    np.testing.assert_allclose(long_targets[0], expected, atol=1e-12)
    np.testing.assert_allclose(long_targets[0], short_targets[0], atol=1e-12)
    # ... and the row nine plies from the end is not weaker than the row one
    # ply from the end, which is exactly what decay would make it.
    np.testing.assert_allclose(long_targets[0], long_targets[8], atol=1e-12)


def test_a_blunder_stops_the_scan_before_it_and_not_after() -> None:
    """One high-regret move at ply 2 of a five-ply game.

    Rows 0 and 1 must blend toward ply 2's PRE-BOUNDARY searched value (rotated
    into their own seat), not toward the outcome; rows 2 onward are past it and
    still reach the terminal result.  ⚑ Row 2 itself is the blunder: its own
    scan stops at itself, so its target collapses to pure ``Q`` -- the row that
    threw the game learns nothing retrospective from it.
    """
    plies = [
        Ply(played=CP_CLEAN),
        Ply(played=CP_CLEAN),
        Ply(top9=CP_TOP, played=CP_BLUNDER),
        Ply(played=CP_CLEAN),
        Ply(played=CP_CLEAN),
    ]
    rows = build_game(plies, result_white=1.0)
    targets = targets_for(rows)
    q = wdl_of_cp(CP_TOP)
    # Rows 0 and 1: F is ply 2's Q, rotated by seat.  Ply 2 is White (even ply),
    # so row 0 needs no rotation and row 1 does.
    np.testing.assert_allclose(targets[0], 0.5 * q + 0.5 * q, atol=1e-12)
    np.testing.assert_allclose(targets[1], 0.5 * q + 0.5 * flipped(q), atol=1e-12)
    # ⚑ Row 0's target is Q only because ply 2's Q happens to equal it; the
    # ROTATED one at row 1 is visibly not the outcome, which is the assertion
    # that the boundary fired rather than the terminal being reached.
    assert not np.allclose(targets[1], 0.5 * q + 0.5 * onehot(2))
    # Row 2 is the blunder: pure Q.
    np.testing.assert_allclose(targets[2], q, atol=1e-12)
    # Rows 3 and 4 are past it and reach the terminal outcome.
    np.testing.assert_allclose(targets[3], 0.5 * q + 0.5 * onehot(2), atol=1e-12)
    np.testing.assert_allclose(targets[4], 0.5 * q + 0.5 * onehot(0), atol=1e-12)


def test_a_pre_boundary_value_is_rotated_into_the_reading_rows_seat() -> None:
    """The POV rotation on its own, with a value that cannot be mistaken for
    its own mirror: the pre-boundary row's searched value is strongly winning,
    so the row one ply earlier must read it as strongly LOSING.
    """
    plies = [
        Ply(played=CP_CLEAN),
        Ply(top9=500.0, played=500.0),
        Ply(top9=500.0, played=CP_BLUNDER),
        Ply(played=CP_CLEAN),
    ]
    rows = build_game(plies, result_white=1.0)
    targets = targets_for(rows)
    boundary_q = wdl_of_cp(500.0)
    assert boundary_q[0] > boundary_q[2]  # the pre-boundary seat is winning
    # Row 1 (Black) and the boundary row 2 (White) are opposite seats.
    row1_q = wdl_of_cp(500.0)
    np.testing.assert_allclose(
        targets[1], 0.5 * row1_q + 0.5 * flipped(boundary_q), atol=1e-12,
    )
    # Row 0 (White) shares the boundary's seat, so no rotation.
    np.testing.assert_allclose(
        targets[0], 0.5 * wdl_of_cp(CP_TOP) + 0.5 * boundary_q, atol=1e-12,
    )


# ── missing data, and the direction it fails in ──────────────────────────────


def test_a_row_that_cannot_price_its_played_move_is_pure_q_and_counted() -> None:
    """A row whose banked played move is absent from its own d9 block cannot
    say whether the move it made was a blunder, so it must not inherit anything
    from what came after -- and the corpus has to say how often that happened.
    """
    plies = [
        Ply(played=CP_CLEAN),
        Ply(played=CP_CLEAN, played_off_book=True),
        Ply(played=CP_CLEAN),
        Ply(played=CP_CLEAN),
    ]
    rows = build_game(plies, result_white=1.0)
    facts = facts_for(rows)
    assert [f.missing_played_move for f in facts] == [False, True, False, False]
    assert [f.missing_depths for f in facts] == [False] * 4
    assert facts[1].played_regret is None
    assert facts[1].instability is not None  # its rungs were all banked
    targets = targets_for(rows)
    np.testing.assert_allclose(targets[1], wdl_of_cp(CP_TOP), atol=1e-12)


def test_a_row_with_no_played_move_at_all_is_pure_q() -> None:
    """The other spelling of the same absence: the generator banked no played
    move for the row.  Same fail direction, same bucket.
    """
    plies = [Ply(played=CP_CLEAN), Ply(played_absent=True), Ply(played=CP_CLEAN)]
    rows = build_game(plies, result_white=1.0)
    facts = facts_for(rows)
    assert facts[1].missing_played_move is True
    assert facts[1].played_regret is None
    np.testing.assert_allclose(targets_for(rows)[1], wdl_of_cp(CP_TOP), atol=1e-12)


def test_a_row_missing_a_gate_depth_is_pure_q_and_counted_separately() -> None:
    """A row without d7 cannot measure its teacher's instability, so its own
    weight is 0 -- pure Q.  ⚑ It lands in the DEPTH bucket, not the played-move
    one: the two counters are disjoint so that a reader comparing either against
    ``rows_written`` is not reading the same failure twice.
    """
    plies = [
        Ply(played=CP_CLEAN),
        Ply(played=CP_CLEAN, drop_d7=True),
        Ply(played=CP_CLEAN),
    ]
    rows = build_game(plies, result_white=1.0)
    facts = facts_for(rows)
    assert [f.missing_depths for f in facts] == [False, True, False]
    assert [f.missing_played_move for f in facts] == [False, False, False]
    assert facts[1].instability is None
    assert facts[1].played_regret is not None  # d9 was there; only d7 was not
    np.testing.assert_allclose(targets_for(rows)[1], wdl_of_cp(CP_TOP), atol=1e-12)


def test_a_row_that_cannot_price_itself_stops_the_scans_passing_through_it() -> None:
    """⚑ The fail direction applies to OTHER rows too.  Row 0 cannot certify
    that row 1's move was clean, so it must not inherit the outcome across it --
    it stops at row 1's searched value instead.
    """
    plies = [
        Ply(played=CP_CLEAN),
        Ply(top9=500.0, played=500.0, played_off_book=True),
        Ply(played=CP_CLEAN),
    ]
    rows = build_game(plies, result_white=1.0)
    targets = targets_for(rows)
    # Row 0 (White) stops at row 1 (Black): row 1's Q, rotated.
    np.testing.assert_allclose(
        targets[0],
        0.5 * wdl_of_cp(CP_TOP) + 0.5 * flipped(wdl_of_cp(500.0)),
        atol=1e-12,
    )
    # Row 2 is past it and still reaches the terminal outcome.
    np.testing.assert_allclose(
        targets[2], 0.5 * wdl_of_cp(CP_TOP) + 0.5 * onehot(0), atol=1e-12,
    )


def test_a_single_row_game_is_all_outcome_under_both_ramps() -> None:
    """``ply / terminal_ply`` is 0/0 for a game with one banked row at ply 0.
    It is the game's terminal row, so arm B reads w = 1 -- the same place arm C
    puts it -- rather than dividing by zero or silently reading 0.
    """
    rows = build_game([Ply(played=CP_CLEAN)], result_white=1.0)
    ramped = targets_for(rows, value_scheme=derive.VALUE_SCHEME_QZPHASE)
    np.testing.assert_allclose(ramped[0], onehot(0), atol=1e-12)
    segment = targets_for(rows)
    np.testing.assert_allclose(
        segment[0], 0.5 * wdl_of_cp(CP_TOP) + 0.5 * onehot(0), atol=1e-12,
    )


# ── the two diagnostic ablations ─────────────────────────────────────────────


def test_qz_w_const_replaces_the_map_on_every_row() -> None:
    """C-no-u: the constant is the weight everywhere, INCLUDING the unstable
    ply the map would have pushed to 0.95.  The segment logic is untouched, so
    the futures are the same ones C-full found.

    Mutation caught: dropping the ``params.w_const is None`` branch in
    ``game_value_targets`` -- ply 1 then reads 0.95 again and this fails.
    """
    rows = toy_rows()
    targets = targets_for(rows, w_const=0.725)
    q = wdl_of_cp(CP_TOP)
    for index, target in enumerate(targets):
        z = onehot(0 if index % 2 == 0 else 2)
        np.testing.assert_allclose(target, 0.275 * q + 0.725 * z, atol=1e-12)
    # ... and it really is a departure: C-full disagrees on the unstable ply.
    assert not np.allclose(targets[1], targets_for(rows)[1])


def test_qz_w_const_still_lets_a_boundary_stop_the_scan() -> None:
    """⚑ The ablation removes ONE mechanism.  A blunder still hard-stops the
    segment under a constant weight, or C-no-u would be testing two changes.
    """
    plies = [Ply(played=CP_CLEAN), Ply(top9=500.0, played=CP_BLUNDER), Ply(played=CP_CLEAN)]
    rows = build_game(plies, result_white=1.0)
    targets = targets_for(rows, w_const=0.725)
    np.testing.assert_allclose(
        targets[0], 0.275 * wdl_of_cp(CP_TOP) + 0.725 * flipped(wdl_of_cp(500.0)),
        atol=1e-12,
    )


def test_qz_no_boundary_carries_the_outcome_past_a_blunder() -> None:
    """C-no-segment: row 0 takes the game result even though row 1 threw it
    away, where the default scheme stops at row 1's searched value.  The u map
    still sets the weight.

    Mutation caught: deleting the ``params.no_boundary`` branch -- row 0 falls
    back to the boundary's rotated Q and this fails.
    """
    plies = [Ply(played=CP_CLEAN), Ply(top9=500.0, played=CP_BLUNDER), Ply(played=CP_CLEAN)]
    rows = build_game(plies, result_white=1.0)
    q = wdl_of_cp(CP_TOP)
    default = targets_for(rows)
    ablated = targets_for(rows, no_boundary=True)
    np.testing.assert_allclose(ablated[0], 0.5 * q + 0.5 * onehot(0), atol=1e-12)
    assert not np.allclose(default[0], ablated[0])
    # The blunder row itself is unchanged: it is past its own boundary either
    # way, so the ablation is visible on the rows BEFORE the blunder only.
    np.testing.assert_allclose(
        ablated[1], 0.5 * wdl_of_cp(500.0) + 0.5 * onehot(2), atol=1e-12,
    )


def test_qz_no_boundary_still_fails_toward_q_on_an_unpriceable_row() -> None:
    """⚑ The ablation removes BLUNDER boundaries, not the fail-toward-Q rule.
    A row that cannot price its own move still gets pure Q for itself, while the
    rows before it now scan straight past it to the outcome.
    """
    plies = [
        Ply(played=CP_CLEAN),
        Ply(played=CP_CLEAN, played_off_book=True),
        Ply(played=CP_CLEAN),
    ]
    rows = build_game(plies, result_white=1.0)
    ablated = targets_for(rows, no_boundary=True)
    q = wdl_of_cp(CP_TOP)
    np.testing.assert_allclose(ablated[1], q, atol=1e-12)
    np.testing.assert_allclose(ablated[0], 0.5 * q + 0.5 * onehot(0), atol=1e-12)


def test_both_ablations_at_once_are_refused() -> None:
    """Constant weight AND no boundaries is ``(1-c) Q + c Z`` -- arm A with a
    different constant, and a cell no prereg describes.
    """
    with pytest.raises(ValueError, match="second amendment"):
        derive.QzParams(w_const=0.725, no_boundary=True)


@pytest.mark.parametrize("bad", [-0.01, 1.5])
def test_a_blend_weight_outside_the_unit_interval_is_refused(bad: float) -> None:
    """Outside [0, 1] the target extrapolates past both endpoints instead of
    mixing them; that is not a weaker ablation, it is a different object.
    """
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        derive.QzParams(w_const=bad)


# ── the flag reaching the bytes ──────────────────────────────────────────────


def game_corpus(
    tmp_path: Path, plies: list[Ply] | None = None, *, name: str = "corpus",
) -> Path:
    """A one-game corpus whose rows are a real, consecutive ply sequence."""
    return write_corpus(
        tmp_path, build_game(plies or TOY_PLIES, result_white=1.0), name=name,
    )


def test_the_value_scheme_reaches_the_emitted_shards(tmp_path: Path) -> None:
    """⚑ THE END-TO-END TAKE-EFFECT PROOF: the vectors the pure function
    computes are the vectors the RIG'S OWN READER finds in the shards.  Every
    other test in this section reads ``game_value_targets``; this one reads the
    bytes, so a scheme that was computed and then not written would fail here
    and nowhere else.
    """
    rows = build_game(TOY_PLIES, result_white=1.0)
    corpus_dir = write_corpus(tmp_path, rows)
    expected, _ = derive.game_value_targets(
        facts_for(rows),
        value_scheme=derive.VALUE_SCHEME_QZSEGMENT,
        params=derive.QzParams(),
    )
    run_derive(
        corpus_dir, tmp_path / "out", "uniform-d9",
        "--value-scheme", derive.VALUE_SCHEME_QZSEGMENT,
    )
    emitted = value_rows(tmp_path / "out")
    assert len(emitted) == 4
    for ply, want in enumerate(expected):
        # float16 is the shard's storage for this column, so the tolerance is
        # the cast's and not the arithmetic's.
        np.testing.assert_allclose(emitted[(0, ply)].search_wdl, want, atol=2e-3)


def test_the_control_arm_writes_the_bare_searched_value(tmp_path: Path) -> None:
    """V0 through the same path: every emitted row is the d9 top's WDL, and the
    summary's own delta-against-V0 reading is exactly zero.  ⚑ The control's
    zero is MEASURED, which is what makes the same instrument's nonzero reading
    on the other arms mean something.
    """
    out = run_derive(game_corpus(tmp_path), tmp_path / "out", "uniform-d9")
    emitted = value_rows(tmp_path / "out")
    for sample in emitted.values():
        np.testing.assert_allclose(sample.search_wdl, wdl_of_cp(CP_TOP), atol=2e-3)
    realized = out["realized"]["value_scheme_realized"]
    assert realized["rows_differing_from_search_value"] == 0
    assert realized["l1_delta_vs_search_value"]["max"] == 0.0
    assert realized["l1_delta_vs_search_value"]["n"] == 4
    # V0 never assembles a game, so the grouped counters stay at zero.
    assert realized["games"] == 0
    assert out["value_blend"]["baked_into_rows"] is False


def test_a_value_scheme_that_never_reaches_the_bytes_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑⚑ THE GATE THIS REPO EXISTS TO HAVE.  Simulate the signature defect --
    the flag parses, the summary would stamp it, and the emitted rows are V0's
    -- and the run must die rather than produce a directory that trains for
    9,680 steps and reads as a clean null because it WAS the control.
    """
    monkeypatch.setattr(
        derive,
        "game_value_targets",
        lambda facts, **_: ([f.q_wdl for f in facts], [None] * len(facts)),
    )
    with pytest.raises(derive.CorpusIntegrityError, match="did not reach the bytes"):
        derive.main([
            "--corpus", str(game_corpus(tmp_path)),
            "--out", str(tmp_path / "out"),
            "--scheme", "uniform-d9",
            "--value-scheme", derive.VALUE_SCHEME_QZ50,
        ])


def test_a_control_arm_that_blended_anything_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same instrument in the other direction: if ``search`` ever stopped
    writing the bare searched value, every arm measured against it would be
    measured against nothing.
    """
    monkeypatch.setattr(
        derive,
        "game_value_targets",
        lambda facts, **_: (
            [0.5 * f.q_wdl + 0.5 * derive.onehot_wdl(f.z_index) for f in facts],
            [None] * len(facts),
        ),
    )
    with pytest.raises(derive.CorpusIntegrityError, match="control arm"):
        derive.main([
            "--corpus", str(game_corpus(tmp_path)),
            "--out", str(tmp_path / "out"),
            "--scheme", "uniform-d9",
        ])


def test_an_ablation_that_changed_no_row_is_refused(tmp_path: Path) -> None:
    """``--qz-no-boundary`` against a corpus with no blunder in it removes
    nothing, so its rows ARE C-full's and the comparison would be empty.  A
    positive reading is required, not the absence of a negative one.
    """
    with pytest.raises(
        derive.CorpusIntegrityError, match="differs from the target C-full",
    ):
        derive.main([
            "--corpus", str(game_corpus(tmp_path)),
            "--out", str(tmp_path / "out"),
            "--scheme", "uniform-d9",
            "--value-scheme", derive.VALUE_SCHEME_QZSEGMENT,
            "--qz-no-boundary",
        ])


def test_an_ablation_that_changed_rows_is_stamped(tmp_path: Path) -> None:
    """And with a blunder present it runs, and says how many rows it moved."""
    plies = [Ply(played=CP_CLEAN), Ply(top9=500.0, played=CP_BLUNDER), Ply(played=CP_CLEAN)]
    out = run_derive(
        game_corpus(tmp_path, plies), tmp_path / "out", "uniform-d9",
        "--value-scheme", derive.VALUE_SCHEME_QZSEGMENT, "--qz-no-boundary",
    )
    realized = out["realized"]["value_scheme_realized"]
    # TWO rows change, not one: row 0 would have stopped at the blunder AHEAD of
    # it, and the blunder row itself would have stopped at ITSELF (pure Q). The
    # ablation lifts both, which is what "no boundaries" means.
    assert realized["boundary_suppressed_by_flag"] == 2
    assert realized["stop_boundary_ahead"] == 0  # the flag removed them all
    assert out["value_scheme"]["params"]["variant"] == "C-no-segment"
    assert out["value_scheme"]["params"]["adoption_candidate"] is False


def test_a_constant_weight_run_is_stamped_and_measured(tmp_path: Path) -> None:
    """C-no-u through the CLI: the variant name, the constant, and how many
    rows it actually moved off the frozen map."""
    out = run_derive(
        game_corpus(tmp_path), tmp_path / "out", "uniform-d9",
        "--value-scheme", derive.VALUE_SCHEME_QZSEGMENT, "--qz-w-const", "0.725",
    )
    params = out["value_scheme"]["params"]
    assert params["variant"] == "C-no-u"
    assert params["w_const"] == 0.725
    realized = out["realized"]["value_scheme_realized"]
    # Three settled plies at 0.5 and one unstable at 0.95: all four move.
    assert realized["w_const_differs_from_map"] == 4
    assert realized["blend_weight_w"]["min"] == 0.725
    assert realized["blend_weight_w"]["max"] == 0.725


def test_a_constant_weight_that_the_map_would_have_given_anyway_is_refused(
    tmp_path: Path,
) -> None:
    """⚑ A positive take-effect reading again.  A settled-teacher corpus maps
    every row to 0.5, so ``--qz-w-const 0.5`` is not an ablation of anything --
    and it would otherwise produce a C-no-u directory identical to C-full's.
    """
    plies = [Ply(played=CP_CLEAN) for _ in range(4)]
    with pytest.raises(
        derive.CorpusIntegrityError, match="differs from the target C-full",
    ):
        derive.main([
            "--corpus", str(game_corpus(tmp_path, plies)),
            "--out", str(tmp_path / "out"),
            "--scheme", "uniform-d9",
            "--value-scheme", derive.VALUE_SCHEME_QZSEGMENT,
            "--qz-w-const", "0.5",
        ])


def test_the_shards_carry_the_value_scheme_stamp(tmp_path: Path) -> None:
    """⚑ ON THE SHARD, not only in the summary.  Four arms of the value round
    share a scheme, a temp and a floor and differ only in this column; without
    the stamp their shards are indistinguishable once copied out of their
    directories, and comparing them is the entire round.
    """
    run_derive(
        game_corpus(tmp_path), tmp_path / "out", "uniform-d9",
        "--value-scheme", derive.VALUE_SCHEME_QZSEGMENT,
    )
    for path in iter_shard_paths(tmp_path / "out"):
        attrs = dict(zarr.open_group(str(path), mode="r").attrs)
        assert attrs["derive_value_scheme"] == derive.VALUE_SCHEME_QZSEGMENT
        assert attrs["derive_value_scheme_params"]["r_boundary"] == derive.QZ_R_BOUNDARY
        assert attrs["derive_value_scheme_params"]["u_scale"] == derive.QZ_U_SCALE
        assert attrs["derive_value_scheme_params"]["variant"] == "C-full"


def test_a_baked_blend_demands_game_frac_zero_in_the_manifest(
    tmp_path: Path,
) -> None:
    """⚑ ``wdl_target`` still holds the RAW outcome, so a trainer run at
    ``game_frac > 0`` against these shards mixes the outcome in a second time on
    top of the share the scheme already chose -- producing a target that is no
    arm of the round.  The manifest has to say so, because nothing else can.
    """
    out = run_derive(
        game_corpus(tmp_path), tmp_path / "out", "uniform-d9",
        "--value-scheme", derive.VALUE_SCHEME_QZ50,
    )
    assert out["value_blend"]["baked_into_rows"] is True
    assert out["required_training_overrides"]["game_frac"] == 0.0
    assert "REQUIRED" in out["required_training_overrides"]["game_frac_note"]
    assert "SECOND time" in out["value_blend"]["note"]
    # And the column's own description no longer claims to be the bare search.
    assert "TARGET" in out["value_channels"]["search_wdl"]


def test_the_summary_records_where_every_scan_stopped(tmp_path: Path) -> None:
    """The three stop kinds are reported separately.  ⚑ ``self`` and ``ahead``
    are never summed: a row that stopped at ITSELF learned nothing
    retrospective, and a corpus of those would be V0 with a segment scheme's
    name on it -- which is exactly what adding the two would hide.
    """
    plies = [Ply(played=CP_CLEAN), Ply(top9=500.0, played=CP_BLUNDER), Ply(played=CP_CLEAN)]
    out = run_derive(
        game_corpus(tmp_path, plies), tmp_path / "out", "uniform-d9",
        "--value-scheme", derive.VALUE_SCHEME_QZSEGMENT,
    )
    realized = out["realized"]["value_scheme_realized"]
    assert realized["stop_boundary_ahead"] == 1  # row 0 -> the blunder at row 1
    assert realized["stop_boundary_self"] == 1   # row 1 is the blunder
    assert realized["stop_terminal"] == 1        # row 2 is past it
    assert realized["games"] == 1
    assert realized["game_rows_min"] == realized["game_rows_max"] == 3
    assert realized["ply_gaps_nonunit"] == 0
    assert realized["played_regret"]["n"] == 3
    assert realized["instability_u"]["n"] == 3


# ── the refusals ─────────────────────────────────────────────────────────────


def test_the_pre_amendment_scheme_name_is_refused_by_name() -> None:
    """A driver written against the FIRST prereg asked for ``qzlambda``.  It
    must be told which amendment replaced it, not merely that the name is
    unknown -- the two are different targets and the next move after a generic
    error is a guess.
    """
    with pytest.raises(ValueError, match="2026-08-31 amendment"):
        derive.parse_value_scheme("qzlambda")
    with pytest.raises(ValueError, match="qzsegment"):
        derive.parse_value_scheme("qzlambda")


def test_an_unknown_value_scheme_is_refused() -> None:
    with pytest.raises(ValueError, match="not one of"):
        derive.parse_value_scheme("qz75")


@pytest.mark.parametrize(
    "extra",
    [
        ["--qz-w-const", "0.725"],
        ["--qz-no-boundary"],
        ["--qz-r-boundary", "0.4"],
        ["--qz-u-scale", "0.1"],
        ["--qz-lambda-u-scale", "0.1"],
    ],
)
def test_a_qz_knob_without_the_segment_scheme_is_refused(
    tmp_path: Path, extra: list[str],
) -> None:
    """⚑ ACCEPTED-AND-IGNORED IS THE DEFECT.  ``--value-scheme qz50
    --qz-w-const 0.725`` looks exactly like a configured ablation and would
    derive arm A, stamping a summary that names a knob which touched no row.
    """
    with pytest.raises(ValueError, match="only affect --value-scheme"):
        derive.main([
            "--corpus", str(game_corpus(tmp_path)),
            "--out", str(tmp_path / "out"),
            "--scheme", "uniform-d9",
            "--value-scheme", derive.VALUE_SCHEME_QZ50,
            *extra,
        ])


def test_the_alias_and_the_new_spelling_are_one_knob() -> None:
    """``--qz-lambda-u-scale`` survives from the first prereg's drivers as an
    ALIAS onto the same value -- never as a second knob that is parsed and
    dropped, which is the failure this file's whole take-effect apparatus
    exists for.
    """
    parser = derive.build_parser()
    base = ["--corpus", "c", "--out", "o", "--scheme", "uniform-d9"]
    assert parser.parse_args(base).qz_u_scale is None
    assert parser.parse_args([*base, "--qz-u-scale", "0.07"]).qz_u_scale == 0.07
    assert parser.parse_args([*base, "--qz-lambda-u-scale", "0.07"]).qz_u_scale == 0.07


def test_a_staircase_that_cannot_answer_the_gate_is_refused(tmp_path: Path) -> None:
    """⚑ WITHOUT THIS, ARM C ON A SHALLOW CORPUS IS ARM V0 WITH A COUNTER.
    Every row would miss d9/d8/d7, every scan would stop where it started, every
    weight would be 0 -- a complete derivation, fully counted, that trains as
    the control.  Refused at startup, before the first row.
    """
    shallow = [{"width": "all", "depth": 5}]
    rows = [
        corpus_row(
            fen=FEN_W,
            phases=[full_width_phase(FEN_W, {5: ramp(FEN_W, "g2g4")}, depth_requested=5)],
            result=1.0,
        ),
    ]
    corpus_dir = write_corpus(tmp_path, rows, staircase=shallow)
    # The POLICY scheme is answerable; only the value gate is not.
    run_derive(corpus_dir, tmp_path / "ok", "uniform-d5")
    with pytest.raises(derive.CorpusIntegrityError, match="silently be V0"):
        derive.main([
            "--corpus", str(corpus_dir),
            "--out", str(tmp_path / "out"),
            "--scheme", "uniform-d5",
            "--value-scheme", derive.VALUE_SCHEME_QZSEGMENT,
        ])


# ── assembling games out of a streamed corpus ────────────────────────────────


def grouper_over(rows: list[dict[str, Any]]) -> tuple[Any, list[list[Any]]]:
    """Feed rows through a real :class:`GameGrouper` and collect the games."""
    options = derive.DeriveOptions(
        scheme=derive.parse_scheme("uniform-d9"),
        temp=1.0,
        cp_slope=SLOPE,
        cp_draw_width=DRAW_WIDTH,
        limit=0,
        seed=1,
        rows_per_shard=64,
        max_envelope_misses=0,
        value_scheme=derive.VALUE_SCHEME_QZSEGMENT,
    )
    deriver = derive.TargetDeriver(options)
    grouper = derive.GameGrouper(deriver)
    games: list[list[Any]] = []
    for row in rows:
        derived = deriver.derive_row(row)
        assert derived is not None
        closed = grouper.add(row, derived)
        if closed:
            games.append(closed)
    return grouper, games


def test_games_are_closed_when_the_next_ones_first_row_arrives() -> None:
    """The stream is game-contiguous, so a game ends where the next begins."""
    rows = build_game([Ply(played=CP_CLEAN)] * 3, result_white=1.0, game_id=0)
    rows += build_game([Ply(played=CP_CLEAN)] * 2, result_white=-1.0, game_id=1)
    grouper, games = grouper_over(rows)
    assert [len(game) for game in games] == [3]
    assert [len(game) for game in [grouper.flush(cut_by_limit=False)]] == [2]


def test_a_re_opened_game_is_refused_rather_than_split() -> None:
    """⚑⚑ THE SILENT FAILURE THIS GUARD EXISTS FOR.  If a game's rows were ever
    interleaved with another's, closing on key change would split it into
    fragments -- and each fragment's LAST row would be handed the game's outcome
    as a terminal position.  The result is a corpus that looks entirely normal
    and whose arm-C targets are wrong in the middle of every game.  MEASURED on
    run02_snap_20260829: 124 game-runs over the first three shards, zero
    re-openings -- so the invariant holds, and it is checked rather than assumed.
    """
    rows = build_game([Ply(played=CP_CLEAN)] * 2, result_white=1.0, game_id=0)
    rows += build_game([Ply(played=CP_CLEAN)] * 2, result_white=-1.0, game_id=1)
    rows += build_game([Ply(played=CP_CLEAN)] * 2, result_white=1.0, game_id=0)
    with pytest.raises(derive.CorpusIntegrityError, match="re-opens worker 0 game 0"):
        grouper_over(rows)


def test_the_same_game_id_from_two_workers_is_two_games() -> None:
    """⚑ The key is ``(worker_id, game_id)``.  Every worker numbers its games
    from 0, so a corpus of 14 workers holds 14 game 0s -- and joining them into
    one 3,000-ply "game" would put one worker's outcome into another's rows.
    """
    rows = build_game([Ply(played=CP_CLEAN)] * 2, result_white=1.0, worker_id=0)
    rows += build_game([Ply(played=CP_CLEAN)] * 2, result_white=-1.0, worker_id=1)
    grouper, games = grouper_over(rows)
    assert [len(game) for game in games] == [2]
    assert len(grouper.flush(cut_by_limit=False)) == 2


def test_a_gap_in_the_banked_plies_is_counted_and_still_rotates(
    tmp_path: Path,
) -> None:
    """A corpus row is banked on a dedup MISS only, so a game's plies need not
    be contiguous.  The gap is counted, and the seat is read off ``stm`` rather
    than off ply parity so the rotation across it is the real one.
    """
    rows = build_game([Ply(played=CP_CLEAN)] * 4, result_white=1.0)
    del rows[2]  # plies 0, 1, 3 survive
    out = run_derive(
        write_corpus(tmp_path, rows), tmp_path / "out", "uniform-d9",
        "--value-scheme", derive.VALUE_SCHEME_QZSEGMENT,
    )
    realized = out["realized"]["value_scheme_realized"]
    assert realized["ply_gaps_nonunit"] == 1
    assert realized["games"] == 1
    assert realized["game_rows_max"] == 3
    emitted = value_rows(tmp_path / "out")
    assert sorted(emitted) == [(0, 0), (0, 1), (0, 3)]
    # Ply 3 is Black and the game's last banked row: pure outcome at w = 0.5.
    np.testing.assert_allclose(
        emitted[(0, 3)].search_wdl,
        0.5 * wdl_of_cp(CP_TOP) + 0.5 * onehot(2),
        atol=2e-3,
    )


def test_a_game_cut_by_the_row_limit_is_counted(tmp_path: Path) -> None:
    """⚑ ``--limit`` takes a PREFIX, so its last game's last banked row is
    treated as terminal and handed an outcome the derivation never read that
    far to see.  A limited run is a smoke test and the counter is why nobody
    mistakes one for an arm.
    """
    out = run_derive(
        game_corpus(tmp_path), tmp_path / "out", "uniform-d9",
        "--value-scheme", derive.VALUE_SCHEME_QZSEGMENT, "--limit", "3",
    )
    realized = out["realized"]["value_scheme_realized"]
    assert realized["games"] == 1
    assert realized["games_cut_by_limit"] == 1
    assert realized["game_rows_max"] == 3
    assert out["realized"]["rows_written"] == 3


def test_a_whole_corpus_run_reports_no_game_cut_by_the_limit(
    tmp_path: Path,
) -> None:
    """The converse, so the counter above is a reading and not a constant."""
    out = run_derive(
        game_corpus(tmp_path), tmp_path / "out", "uniform-d9",
        "--value-scheme", derive.VALUE_SCHEME_QZSEGMENT,
    )
    assert out["realized"]["value_scheme_realized"]["games_cut_by_limit"] == 0


def test_grouping_does_not_move_a_row_into_another_shard(tmp_path: Path) -> None:
    """⚑ The shard LAYOUT must not depend on the value scheme, or two arms of
    the round would differ in how their rows were batched as well as in their
    targets.  A grouped path pushing a whole game at once could otherwise
    overshoot ``--rows-per-shard``; it cuts at exactly the same boundary.
    """
    rows: list[dict[str, Any]] = []
    for game_id in range(4):
        rows += build_game(
            [Ply(played=CP_CLEAN)] * 3, result_white=1.0, game_id=game_id,
        )
    corpus_dir = write_corpus(tmp_path, rows)
    control = run_derive(
        corpus_dir, tmp_path / "v0", "uniform-d9", "--rows-per-shard", "5",
    )
    grouped = run_derive(
        corpus_dir, tmp_path / "c", "uniform-d9", "--rows-per-shard", "5",
        "--value-scheme", derive.VALUE_SCHEME_QZSEGMENT,
    )
    assert [s["rows"] for s in control["shards"]] == [5, 5, 2]
    assert [s["rows"] for s in grouped["shards"]] == [s["rows"] for s in control["shards"]]


# ── review round 1: the gate must read the WRITTEN column ────────────────────


def test_a_dropped_write_is_refused_even_though_the_computation_was_right(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑⚑ THE BAD *WRITE* PATH, which the computation-side test cannot reach.

    ``test_a_value_scheme_that_never_reaches_the_bytes_is_refused`` monkeypatches
    ``game_value_targets``, so it exercises the gate on a bad COMPUTATION only.
    An independent reviewer replaced the ``search_wdl`` assignment with the
    baseline -- the flag parses, the scheme computes correctly, the WRITE drops
    it -- and the gate stayed silent, because it was measuring the vector it had
    just computed rather than the column.

    Here the computation is left completely alone and only the write is broken:
    ``write_value_target`` reports honestly what it stored and simply does not
    store the target.  The gate must fire on that.
    """
    # Every row of this fixture has the same searched value, so "the write
    # stored V0's vector instead of the target" is one constant.
    searched = wdl_of_cp(CP_TOP).astype(np.float32)

    def dropped_write(sample: Any, _target: np.ndarray) -> np.ndarray:
        # ⚑ The reviewer's mutation, as a stub: the scheme's target is computed
        # correctly and then DISCARDED by the write, which stores the searched
        # value and reports honestly what it stored.
        sample.search_wdl = searched.copy()
        return np.asarray(
            derive.shard_stored(np.asarray(sample.search_wdl, dtype=np.float64)),
            dtype=np.float64,
        )

    monkeypatch.setattr(derive, "write_value_target", dropped_write)
    with pytest.raises(derive.CorpusIntegrityError, match="did not reach the bytes"):
        derive.main([
            "--corpus", str(game_corpus(tmp_path)),
            "--out", str(tmp_path / "out"),
            "--scheme", "uniform-d9",
            "--value-scheme", derive.VALUE_SCHEME_QZSEGMENT,
        ])


def test_the_take_effect_reading_comes_off_the_row_not_off_the_target() -> None:
    """``write_value_target``'s contract, in isolation: it hands back what it
    READ off the sample, so a writer that stores the wrong thing reports the
    wrong thing.  A version that returned ``target`` would make every gate above
    it unfalsifiable.
    """
    sample = ReplaySample(
        x=np.zeros((1, 8, 8), dtype=np.float32),
        policy_target=np.zeros((COMPACT_POLICY_SIZE,), dtype=np.float32),
        wdl_target=0,
    )
    want = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    got = derive.write_value_target(sample, want)
    np.testing.assert_allclose(got, want, atol=1e-3)
    np.testing.assert_allclose(np.asarray(sample.search_wdl, dtype=float), want, atol=1e-6)
    # ⚑ And it is the SHARD's reading, not float64: the value comes back through
    # the float32-then-float16 cast, so a target the column cannot hold reads
    # back as what the column holds.
    tiny = np.array([1.0 - 2.0 ** -20, 2.0 ** -20, 0.0], dtype=np.float64)
    back = derive.write_value_target(sample, tiny)
    assert float(back[1]) == float(np.float16(np.float32(2.0 ** -20)))


def test_the_control_arms_delta_is_exactly_zero_through_the_shard_cast(
    tmp_path: Path,
) -> None:
    """The baseline goes through the SAME cast as the read-back, so V0's delta
    is an exact 0 rather than 0-to-within-float16 -- which is what lets
    ``enforce_value_scheme_take_effect`` demand equality instead of a tolerance.
    """
    out = run_derive(game_corpus(tmp_path), tmp_path / "out", "uniform-d9")
    delta = out["realized"]["value_scheme_realized"]["l1_delta_vs_search_value"]
    assert delta["max"] == 0.0
    assert delta["min"] == 0.0
    assert delta["mean"] == 0.0


def test_a_shard_whose_value_column_did_not_survive_the_write_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ The BYTES, one level below the row.  ``search_wdl`` is an OPTIONAL shard
    column gated by ``has_search_wdl``; dropped, the shard stays well formed,
    every row reads ``has_search_wdl=0``, and the trainer falls the search
    component back to the raw one-hot outcome.  ``_flush`` reads the column back
    off the written file, so such a shard never leaves this process.
    """
    real = derive.samples_to_arrays

    def strip_value(samples: Any) -> Any:
        arrs = dict(real(samples))
        arrs["has_search_wdl"] = np.zeros_like(np.asarray(arrs["has_search_wdl"]))
        return arrs

    monkeypatch.setattr(derive, "samples_to_arrays", strip_value)
    with pytest.raises(derive.CorpusIntegrityError, match="value target is gone"):
        derive.main([
            "--corpus", str(game_corpus(tmp_path)),
            "--out", str(tmp_path / "out"),
            "--scheme", "uniform-d9",
            "--value-scheme", derive.VALUE_SCHEME_QZ50,
        ])


# ── review round 1: the ablation must differ in exactly ONE way ──────────────


def test_the_last_banked_move_is_priced_like_any_other() -> None:
    """⚑⚑ THE LAST BANKED ROW'S MOVE IS A PLAYED MOVE, and arm C must price it.

    ``gen_sf_rooted_corpus.play_game`` writes ``played_move`` and then PUSHES
    it, and adjudicates at the TOP of its loop -- so the last banked move is
    usually the move that ENDS the game, banked and full-width like every
    other. A backward pass that started at ``count - 2`` never looked at it, so
    a game thrown away on its final banked move stopped nothing: that row and
    every clean row behind it blended toward the outcome the blunder had just
    produced, which is the exact attribution error this arm exists to prevent.

    Here the LAST move is the blunder and nothing else is. No row may reach the
    game result.
    """
    plies = [
        Ply(played=CP_CLEAN),
        Ply(played=CP_CLEAN),
        Ply(top9=CP_TOP, played=CP_BLUNDER),  # the final banked move throws it
    ]
    rows = build_game(plies, result_white=1.0)
    facts = facts_for(rows)
    assert facts[2].played_regret is not None
    assert facts[2].played_regret > derive.QZ_R_BOUNDARY
    targets, readings = derive.game_value_targets(
        facts, value_scheme=derive.VALUE_SCHEME_QZSEGMENT, params=derive.QzParams(),
    )
    assert [r.stop for r in readings if r is not None] == [
        derive.SEGMENT_BOUNDARY_AHEAD,
        derive.SEGMENT_BOUNDARY_AHEAD,
        derive.SEGMENT_BOUNDARY_SELF,
    ]
    q = wdl_of_cp(CP_TOP)
    # Row 2 is the blunder: pure Q. Rows 0 and 1 stop there -- row 0 shares its
    # seat (both White), row 1 does not.
    np.testing.assert_allclose(targets[2], q, atol=1e-12)
    np.testing.assert_allclose(targets[0], 0.5 * q + 0.5 * q, atol=1e-12)
    np.testing.assert_allclose(targets[1], 0.5 * q + 0.5 * flipped(q), atol=1e-12)
    # ⚑ And NOTHING reached the outcome, which is the whole claim.
    for index, target in enumerate(targets):
        z = onehot(0 if index % 2 == 0 else 2)
        assert not np.allclose(target, 0.5 * q + 0.5 * z)


def test_a_clean_final_move_still_reaches_the_outcome() -> None:
    """The converse, so the test above pins the gate and not a constant."""
    rows = build_game([Ply(played=CP_CLEAN)] * 3, result_white=1.0)
    _, readings = derive.game_value_targets(
        facts_for(rows), value_scheme=derive.VALUE_SCHEME_QZSEGMENT,
        params=derive.QzParams(),
    )
    assert all(r is not None and r.stop == derive.SEGMENT_TERMINAL for r in readings)


def test_an_unpriceable_last_row_stops_at_itself_under_both_c_cells() -> None:
    """⚑ The fail-toward-Q rule is not a boundary and neither ablation
    suppresses it.  With C-full now pricing the last banked move, an
    UNPRICEABLE last row stops at itself under C-full -- and C-no-segment must
    agree, because its licensed difference is suppressing BLUNDER boundaries
    only.
    """
    plies = [
        Ply(played=CP_CLEAN),
        Ply(played=CP_CLEAN),
        Ply(played=CP_CLEAN, played_off_book=True),  # the LAST row, unpriceable
    ]
    rows = build_game(plies, result_white=1.0)
    q = wdl_of_cp(CP_TOP)
    for params in (derive.QzParams(), derive.QzParams(no_boundary=True)):
        targets, readings = derive.game_value_targets(
            facts_for(rows), value_scheme=derive.VALUE_SCHEME_QZSEGMENT, params=params,
        )
        assert readings[2] is not None
        assert readings[2].stop == derive.SEGMENT_BOUNDARY_SELF
        np.testing.assert_allclose(targets[2], q, atol=1e-12)
    # A MID-game unpriceable row behaves the same way under both, so the rule
    # is about certifiability and not about position in the game.
    mid = build_game(
        [Ply(played=CP_CLEAN), Ply(played=CP_CLEAN, played_off_book=True),
         Ply(played=CP_CLEAN)],
        result_white=1.0,
    )
    np.testing.assert_allclose(targets_for(mid)[1], q, atol=1e-12)
    np.testing.assert_allclose(targets_for(mid, no_boundary=True)[1], q, atol=1e-12)


def test_a_gap_in_the_banked_plies_stops_a_segment() -> None:
    """⚑⚑ AN UNOBSERVED TRANSITION IS NOT A CLEAN ONE.  A dedup hit (or a
    tolerated drop) leaves a non-unit ply gap, and the moves inside it were
    never banked -- they cannot be priced even in principle.  Scanning across a
    gap lets an earlier row inherit the terminal outcome through an unobserved
    blunder, inside a stretch the summary is calling clean.  The gap is a hard
    boundary under every C cell, ablations included.
    """
    rows = build_game([Ply(played=CP_CLEAN)] * 4, result_white=1.0)
    contiguous = targets_for(rows)
    del rows[2]  # plies 0, 1, 3 -- the transition out of ply 1 is now unobserved
    gapped = targets_for(rows)
    q = wdl_of_cp(CP_TOP)
    # Contiguous: every row reaches the outcome.
    np.testing.assert_allclose(contiguous[1], 0.5 * q + 0.5 * onehot(2), atol=1e-12)
    # Gapped: ply 1 is the boundary, so it is pure Q and ply 0 stops there.
    np.testing.assert_allclose(gapped[1], q, atol=1e-12)
    np.testing.assert_allclose(gapped[0], 0.5 * q + 0.5 * flipped(q), atol=1e-12)
    # Ply 3 is past the gap and is the last banked row with a clean move.
    np.testing.assert_allclose(gapped[2], 0.5 * q + 0.5 * onehot(2), atol=1e-12)
    # ⚑ The ablations keep the carve-out for the row's OWN target ...
    np.testing.assert_allclose(targets_for(rows, no_boundary=True)[1], q, atol=1e-12)
    np.testing.assert_allclose(targets_for(rows, w_const=0.725)[1], q, atol=1e-12)
    # ... and C-no-segment lets the row BEHIND it scan past, which is exactly
    # what that cell ablates. Pinned rather than left accidental: "the ablation
    # preserves fail-toward-Q" is true of a row's own target and false of its
    # neighbours', and those are different claims.
    np.testing.assert_allclose(
        targets_for(rows, no_boundary=True)[0], 0.5 * q + 0.5 * onehot(0), atol=1e-12,
    )
    # C-no-u keeps the segment logic, so its row 0 still stops at the gap.
    np.testing.assert_allclose(
        targets_for(rows, w_const=0.725)[0], 0.275 * q + 0.725 * flipped(q), atol=1e-12,
    )


# ── review round 1: the manifest's overrides are machine-readable ────────────


def test_every_required_override_that_names_a_fraction_is_a_float(
    tmp_path: Path,
) -> None:
    """⚑ ``float(overrides[k])`` is exactly how the rig's own guard test consumes
    the siblings, so a prose value in the same dict is a key nothing can read.
    The reason lives in a ``_note`` key beside it instead.
    """
    out = run_derive(
        game_corpus(tmp_path), tmp_path / "out", "uniform-d9",
        "--value-scheme", derive.VALUE_SCHEME_QZ50,
    )
    overrides = out["required_training_overrides"]
    assert overrides["game_frac"] == 0.0
    for key, value in overrides.items():
        if key.endswith("_note") or key == "search_wdl_frac":
            continue
        assert isinstance(value, float), f"{key} is {type(value).__name__}, not a float"
        float(value)
    assert "REQUIRED" in overrides["game_frac_note"]


# ── review round 1: presence, not value ──────────────────────────────────────


def test_a_qz_knob_passed_at_its_default_value_is_still_refused(
    tmp_path: Path,
) -> None:
    """⚑⚑ THE POINT OF A SENTINEL DEFAULT.  ``--value-scheme search
    --qz-r-boundary 0.27`` passes the knob EXPLICITLY and the run cannot consume
    it; a value-based check ("does it differ from the frozen default?") waves it
    through, which is a knob accepted and then ignored inside the one check
    written to forbid that.
    """
    for flag, value in (
        ("--qz-r-boundary", str(derive.QZ_R_BOUNDARY)),
        ("--qz-u-scale", str(derive.QZ_U_SCALE)),
        ("--qz-lambda-u-scale", str(derive.QZ_U_SCALE)),
    ):
        with pytest.raises(ValueError, match="only affect --value-scheme"):
            derive.main([
                "--corpus", str(game_corpus(tmp_path, name=f"c{flag}")),
                "--out", str(tmp_path / f"out{flag}"),
                "--scheme", "uniform-d9",
                flag, value,
            ])


def test_the_frozen_defaults_are_applied_from_the_sentinel(tmp_path: Path) -> None:
    """The sentinel must not cost the frozen values: a run that passes no qz
    knob still derives at the ledger's 0.27 / 0.05, and the summary says so.
    """
    parser = derive.build_parser()
    args = parser.parse_args(["--corpus", "c", "--out", "o", "--scheme", "uniform-d9"])
    assert args.qz_r_boundary is None
    assert args.qz_u_scale is None
    assert args.qz_no_boundary is None
    out = run_derive(
        game_corpus(tmp_path), tmp_path / "out", "uniform-d9",
        "--value-scheme", derive.VALUE_SCHEME_QZSEGMENT,
    )
    params = out["value_scheme"]["params"]
    assert params["r_boundary"] == derive.QZ_R_BOUNDARY == 0.27
    assert params["u_scale"] == derive.QZ_U_SCALE == 0.05
    assert params["no_boundary"] is False


# ── review round 1: what a truncated game's last row is told ─────────────────


def test_a_game_cut_by_the_limit_hands_its_last_read_row_the_outcome(
    tmp_path: Path,
) -> None:
    """⚑ STATED AND PINNED rather than left to the counter.  ``--limit`` cuts the
    last game wherever the row count ran out, and that row is then treated as
    terminal -- under C it takes ``F = Z``, the game's recorded result, which the
    derivation did not read far enough to witness.  The behaviour is deliberate
    (dropping the cut game's rows would make ``--limit N`` emit fewer than N rows
    as a function of the VALUE scheme, so two arms of a paired round would hold
    different positions); the counter is what keeps it visible.
    """
    plies = [Ply(played=CP_CLEAN) for _ in range(5)]
    corpus_dir = game_corpus(tmp_path, plies)
    out = run_derive(
        corpus_dir, tmp_path / "cut", "uniform-d9",
        "--value-scheme", derive.VALUE_SCHEME_QZSEGMENT, "--limit", "3",
    )
    assert out["realized"]["value_scheme_realized"]["games_cut_by_limit"] == 1
    cut = value_rows(tmp_path / "cut")
    assert sorted(cut) == [(0, 0), (0, 1), (0, 2)]
    q = wdl_of_cp(CP_TOP)
    # Ply 2 is White and the game is a White win: the cut row gets the OUTCOME.
    np.testing.assert_allclose(cut[(0, 2)].search_wdl, 0.5 * q + 0.5 * onehot(0), atol=2e-3)

    # ⚑ And the same row in the UNCUT derivation is not terminal, so the cut
    # genuinely changed what that row was told -- the counter is not decorative.
    run_derive(
        corpus_dir, tmp_path / "whole", "uniform-d9",
        "--value-scheme", derive.VALUE_SCHEME_QZSEGMENT,
    )
    whole = value_rows(tmp_path / "whole")
    np.testing.assert_allclose(
        whole[(0, 2)].search_wdl, 0.5 * q + 0.5 * onehot(0), atol=2e-3,
    )
    # (Both read the outcome here because the whole game is clean -- what the
    # cut changes is WHICH row is terminal, so ply 4 exists only in the uncut
    # run and ply 2's span differs.)
    assert (0, 4) in whole
    assert (0, 4) not in cut


def test_a_writer_that_mangles_the_value_column_is_caught_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The read-back's other branch: the column is present on disk but is not
    the one handed to the writer.  ⚑ The WRITER is what is broken here, not the
    arrays -- ``_flush`` verifies the file against the array it passed down, so
    a serializer that silently altered the column is caught even though every
    in-process reading agreed.
    """
    real_save = derive.save_local_shard_arrays

    def mangling_save(path: Any, *, arrs: Any, meta: Any) -> Any:
        written = dict(arrs)
        values = np.asarray(written["search_wdl"]).copy()
        # ⚑ W and L swapped: a VALID distribution that still sums to 1, so the
        # shard writer's own "active rows have non-positive sum" check cannot
        # see it. That check is a real guard and it covers the degenerate case;
        # the read-back is what covers a wrong-but-well-formed one.
        swapped = values[0][::-1].copy()
        assert not np.array_equal(swapped, values[0]), "pick a row that is not symmetric"
        values[0] = swapped
        written["search_wdl"] = values
        return real_save(path, arrs=written, meta=meta)

    monkeypatch.setattr(derive, "save_local_shard_arrays", mangling_save)
    with pytest.raises(
        derive.CorpusIntegrityError, match="not the one that was written",
    ):
        derive.main([
            "--corpus", str(game_corpus(tmp_path)),
            "--out", str(tmp_path / "out"),
            "--scheme", "uniform-d9",
            "--value-scheme", derive.VALUE_SCHEME_QZ50,
        ])


# ── review round 1: the manifest's game_frac note is now a GATE ──────────────


def test_the_launcher_refuses_game_frac_against_a_baked_corpus(
    tmp_path: Path,
) -> None:
    """⚑⚑ THE NOTE TURNED INTO A GATE, and driven by the RIG'S OWN function
    against shards this tool actually wrote -- the same pattern as
    ``test_the_manifests_overrides_pass_the_rigs_own_guards``.

    Baking the blend created a hazard that did not exist before: ``wdl_target``
    still carries the raw outcome, so ``game_frac > 0`` mixes it in a SECOND
    time on top of the share the scheme chose. A manifest key saying "please
    don't" is not a defence; ``lc0_control_train`` reads the shards' own
    ``derive_value_scheme`` stamp and refuses.
    """
    from scripts.lc0_control_train import baked_value_blend_problems

    corpus_dir = game_corpus(tmp_path)
    control = tmp_path / "v0"
    baked = tmp_path / "a"
    run_derive(corpus_dir, control, "uniform-d9")
    run_derive(
        corpus_dir, baked, "uniform-d9",
        "--value-scheme", derive.VALUE_SCHEME_QZ50,
    )

    # game_frac 0 is fine against anything, baked or not.
    assert baked_value_blend_problems({"game_frac": 0.0}, [baked]) == []
    assert baked_value_blend_problems({"game_frac": 0.3}, [control]) == []
    # ⚑ Absent means unbaked: every pre-value-round corpus (and every
    # lc0_data_to_rows one) carries no such attr and must keep launching.
    assert baked_value_blend_problems({}, [baked]) == []

    problems = baked_value_blend_problems({"game_frac": 0.3}, [baked])
    assert len(problems) == 1
    assert derive.VALUE_SCHEME_QZ50 in problems[0]
    assert "ALREADY blended it in" in problems[0]
    # And the manifest's own override is the number the gate enforces.
    summary = json.loads(
        (baked / derive.SUMMARY_NAME).read_text(encoding="utf-8"),
    )
    assert summary["required_training_overrides"]["game_frac"] == 0.0


def test_the_launcher_gate_reads_the_stamp_and_not_the_directory_name(
    tmp_path: Path,
) -> None:
    """The stamp is per SHARD, so one baked shard in a mixed --shards pool is
    enough to refuse -- the same "coverage, not any()" lesson the sf_wdl gate
    beside it already learned.
    """
    from scripts.lc0_control_train import baked_value_blend_problems

    corpus_dir = game_corpus(tmp_path)
    control = tmp_path / "v0"
    baked = tmp_path / "c"
    run_derive(corpus_dir, control, "uniform-d9")
    run_derive(
        corpus_dir, baked, "uniform-d9",
        "--value-scheme", derive.VALUE_SCHEME_QZSEGMENT,
    )
    problems = baked_value_blend_problems({"game_frac": 0.1}, [control, baked])
    assert len(problems) == 1
    assert derive.VALUE_SCHEME_QZSEGMENT in problems[0]


# ── review round 2 (Codex): evidence, and knobs refused at startup ───────────


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -0.01])
def test_a_non_finite_instability_scale_is_refused_at_startup(bad: float) -> None:
    """⚑ ``nan <= 0.0`` is FALSE, so a bare positivity check ACCEPTS
    ``--qz-u-scale nan`` -- and ``segment_blend_weight`` then propagates NaN
    into every weight, every target and every emitted shard, with the
    take-effect gate noticing only at the END, after a whole corpus has been
    written. Refused at startup, like every other bad knob here.
    """
    with pytest.raises(ValueError, match="positive and finite"):
        derive.QzParams(u_scale=bad)
    with pytest.raises(ValueError, match="positive and finite"):
        derive.segment_blend_weight(0.01, u_scale=bad)


def test_the_ablation_counters_can_move_when_the_emitted_target_does_not() -> None:
    """⚑⚑ WHY THE ABLATION GATE READS TARGETS AND NOT COUNTERS.

    Every move here is a blunder, so every row is a self-boundary and
    ``F == Q``: the blend weight multiplies a difference of ZERO and the
    emitted vector cannot move. ``w_const_differs_from_map`` nonetheless
    increments on all of them, because the weight really is not the map's.
    That counter is a diagnostic about an intermediate choice; it is not
    evidence that anything reached a row.
    """
    plies = [Ply(top9=CP_TOP, played=CP_BLUNDER) for _ in range(3)]
    rows = build_game(plies, result_white=1.0)
    facts = facts_for(rows)
    full, full_readings = derive.game_value_targets(
        facts, value_scheme=derive.VALUE_SCHEME_QZSEGMENT, params=derive.QzParams(),
    )
    ablated, ablated_readings = derive.game_value_targets(
        facts,
        value_scheme=derive.VALUE_SCHEME_QZSEGMENT,
        params=derive.QzParams(w_const=0.9),
    )
    # The counter's premise holds on every row ...
    moved = [
        r for r in ablated_readings
        if r is not None and r.weight != r.weight_from_map
    ]
    assert len(moved) == 3
    # ... and not one emitted target differs.
    for a, b, reading in zip(full, ablated, full_readings):
        assert reading is not None
        assert reading.stop == derive.SEGMENT_BOUNDARY_SELF
        np.testing.assert_allclose(a, b, atol=1e-12)


def test_a_working_ablation_reports_how_many_targets_it_moved(
    tmp_path: Path,
) -> None:
    """The gate's own reading, positive and in the summary: rows whose EMITTED,
    QUANTIZED target differs from the one C-full would have written.
    """
    plies = [Ply(played=CP_CLEAN), Ply(top9=500.0, played=CP_BLUNDER), Ply(played=CP_CLEAN)]
    out = run_derive(
        game_corpus(tmp_path, plies), tmp_path / "out", "uniform-d9",
        "--value-scheme", derive.VALUE_SCHEME_QZSEGMENT, "--qz-no-boundary",
    )
    realized = out["realized"]["value_scheme_realized"]
    assert realized["rows_differing_from_c_full"] > 0
    assert realized["rows_differing_from_c_full"] <= out["realized"]["rows_written"]
    # C-full itself has no reference to compare against, so it reads 0.
    plain = run_derive(
        game_corpus(tmp_path, plies, name="c2"), tmp_path / "full", "uniform-d9",
        "--value-scheme", derive.VALUE_SCHEME_QZSEGMENT,
    )
    assert plain["realized"]["value_scheme_realized"]["rows_differing_from_c_full"] == 0
