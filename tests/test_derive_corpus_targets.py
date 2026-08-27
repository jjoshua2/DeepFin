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
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest
import zarr

from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from chess_anti_engine.moves.leela_index import compact_index_for_move
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

CONFIG_SHA = "0" * 64

#: The staircase the synthetic summary declares: a full-width depth-9 scout and
#: one narrowed depth-11 rung, i.e. the production shape with one rung removed.
STAIRCASE = [{"width": "all", "depth": 9}, {"width": 3, "depth": 11}]


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
) -> dict[str, Any]:
    board = chess.Board(fen)
    played = next(iter(board.legal_moves)).uci()
    return {
        "schema": corpus.ROW_SCHEMA,
        "run": {
            "run_id": "test_corpus",
            "config_sha256": config_sha,
            corpus.KEY_TT_CARRIED: True,
        },
        "fen": fen,
        "dedup_key": " ".join(fen.split(" ")[:5]),
        "worker_id": 0,
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
) -> Path:
    """A corpus directory: the generator's own shard writer plus a summary."""
    out = tmp_path / name
    out.mkdir()
    writer = corpus.ShardWriter(out_dir=out, worker_id=0, shard_rows=1000)
    for row in rows:
        writer.write(row)
    writer.close()
    summary = {
        "schema": corpus.SUMMARY_SCHEMA,
        "row_schema": corpus.ROW_SCHEMA,
        "run_id": "test_corpus",
        "config_sha256": config_sha,
        "config_requested": {"cp_slope": SLOPE, "cp_draw_width": DRAW_WIDTH},
        "staircase_parsed": staircase if staircase is not None else STAIRCASE,
        "shards": [] if drop_shard_from_summary else list(writer.shards),
        "banked_rows_min_piece_count": corpus.MIN_BANKED_PIECES,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


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
