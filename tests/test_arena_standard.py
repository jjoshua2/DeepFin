from __future__ import annotations

import json

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.moves import POLICY_ENCODING_LC0_1858
from scripts.arena_standard import (
    append_result,
    build_result_record,
    game_scores_to_pair_scores,
    pentanomial_counts,
    play_paired_games_matched_sims,
    resolve_search_shape,
    summarize_pentanomial,
)


def test_game_scores_collapse_to_pair_scores():
    # candidate: W,W | W,D | D,D | L,D | L,L | W,L
    games = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0]
    assert game_scores_to_pair_scores(games) == [2.0, 1.5, 1.0, 0.5, 0.0, 1.0]
    assert pentanomial_counts([2.0, 1.5, 1.0, 0.5, 0.0, 1.0]) == (1, 1, 2, 1, 1)


def test_game_scores_validation():
    with pytest.raises(ValueError, match="even number of games"):
        game_scores_to_pair_scores([1.0])  # odd number of games
    with pytest.raises(ValueError, match="per-game score must be"):
        game_scores_to_pair_scores([1.0, 0.3])  # not a chess score
    with pytest.raises(ValueError, match="pair score must be"):
        pentanomial_counts([1.25])  # not a pair score


def test_pentanomial_golden_symmetric():
    # Hand-constructed symmetric vector: zero Elo, known CI from the
    # pentanomial pair variance.
    summary = summarize_pentanomial((10, 20, 40, 20, 10))
    assert summary.pairs == 100
    assert summary.games == 200
    assert summary.score == pytest.approx(0.5)
    assert summary.score_se == pytest.approx(0.0275240941, abs=1e-9)
    assert summary.elo == pytest.approx(0.0, abs=1e-9)
    lo, hi = summary.elo_ci95
    assert lo == pytest.approx(-37.632858, abs=1e-4)
    assert hi == pytest.approx(+37.632858, abs=1e-4)


def test_pentanomial_golden_asymmetric():
    summary = summarize_pentanomial((30, 30, 20, 15, 5))
    assert summary.score == pytest.approx(0.6625)
    assert summary.score_se == pytest.approx(0.0300199849, abs=1e-9)
    assert summary.elo == pytest.approx(117.164842, abs=1e-4)
    lo, hi = summary.elo_ci95
    assert lo == pytest.approx(73.090400, abs=1e-4)
    assert hi == pytest.approx(165.225436, abs=1e-4)


def test_pentanomial_degenerate_score_has_no_elo():
    summary = summarize_pentanomial((4, 0, 0, 0, 0))  # candidate won every pair
    assert summary.score == pytest.approx(1.0)
    assert summary.elo is None
    assert summary.elo_ci95[1] is None


def _tiny_model() -> torch.nn.Module:
    cfg = ModelConfig(
        kind="transformer",
        embed_dim=32,
        num_layers=1,
        num_heads=2,
        use_smolgen=False,
        policy_encoding=POLICY_ENCODING_LC0_1858,
    )
    return build_model(cfg).eval()


def test_smoke_arena_matched_sims_cpu(tmp_path):
    """4-game paired arena with a tiny random-init lc0_1858 model on CPU."""
    torch.manual_seed(0)
    model_a = _tiny_model()
    torch.manual_seed(1)
    model_b = _tiny_model()

    e4 = chess.Board()
    e4.push_uci("e2e4")
    d4 = chess.Board()
    d4.push_uci("d2d4")
    openings = [e4, d4]

    # NOTE: this now hard-requires the compiled gumbel extension. The `play`
    # shape carries vloss_weight=3, and `pick_moves_for_boards` refuses to run
    # the Python reference with a virtual loss it cannot honour rather than
    # silently dropping it. Consistent with the repo's no-skips rule (the test
    # suite hard-requires every extra); it means a missing `.so` fails here
    # loudly instead of quietly measuring a different search.
    side = resolve_search_shape("play")
    pair_scores = play_paired_games_matched_sims(
        model_a, model_b, openings,
        device="cpu", rng=np.random.default_rng(0),
        sims_candidate=2, sims_reference=2,
        max_plies=10, temperature=1.0, gumbel_add_noise=True,
        search_candidate=side, search_reference=side,
    )
    assert len(pair_scores) == 2
    assert all(s in (0.0, 0.5, 1.0, 1.5, 2.0) for s in pair_scores)

    counts = pentanomial_counts(pair_scores)
    assert sum(counts) == 2
    summary = summarize_pentanomial(counts)
    assert summary.games == 4
    assert 0.0 <= summary.score <= 1.0

    record = build_result_record(
        summary,
        mode="matched_sims",
        candidate="cand.pt",
        reference="ref.pt",
        openings_path="book.pgn.zip",
        opening_plies=16,
        sims_candidate=2,
        sims_reference=2,
        ms_per_move=None,
        temperature=1.0,
        gumbel_add_noise=True,
        max_plies=10,
        seed=0,
        device="cpu",
        duration_s=0.0,
    )
    out = tmp_path / "arena_results.jsonl"
    append_result(record, out)

    loaded = json.loads(out.read_text().splitlines()[-1])
    assert loaded["mode"] == "matched_sims"
    assert loaded["candidate"] == "cand.pt"
    assert loaded["reference"] == "ref.pt"
    assert loaded["games"] == 4
    assert loaded["pairs"] == 2
    assert sum(loaded["pentanomial"].values()) == 2
    assert set(loaded["pentanomial"]) == {"WW", "WD_DW", "DD_WL", "LD_DL", "LL"}
    for key in ("ts", "git_sha", "config_hash", "score", "elo", "elo_ci95", "seed"):
        assert key in loaded
