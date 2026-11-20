"""scripts/arena_seat_balanced_pool.py: seat-balanced pooling of two arena runs.

Synthetic arena rows and game logs, shaped like ``arena_standard.py``'s output,
check that a planted candidate-seat bonus cancels exactly, that a planted true
effect is recovered, and that every listed mismatch is refused. A last test
reproduces the banked same-net seat control when its files are present.
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from scripts import arena_seat_balanced_pool as pool_mod

ARM, BASE = "arm.pt", "base.pt"
PENTA = dict(zip(pool_mod.PENTA_KEYS, pool_mod.PAIR_SCORE_BINS, strict=True))


def _search(tag: str) -> dict[str, Any]:
    return {"shape": "training", "gumbel": {"c_scale": 0.1, "tag": tag}}


def _settings(candidate: str, reference: str, pairs: int, over: dict[str, Any]) -> dict[str, Any]:
    """The game-log header settings (arena_standard.arena_game_log_settings)."""
    settings: dict[str, Any] = {
        "mode": "matched_sims", "candidate": candidate, "reference": reference, "games": 2 * pairs,
        "seed": 7, "openings": "eval.pgn", "openings_kind": "pgn", "opening_plies": 16,
        "sims_candidate": 64, "sims_reference": 64, "ms_per_move": None, "max_plies": 300,
        "temperature": 0.1, "gumbel_add_noise": False,
        "search_candidate": _search(candidate), "search_reference": _search(reference),
        "volatility_candidate": None, "uci_args": "", "syzygy": "tb", "syzygy_max_pieces": 6,
        "syzygy_protocol": "p1",
    }
    settings.update(over)
    return settings


def _row(label: str, *, pairs: int, scores: list[float], over: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(scores)
    row: dict[str, Any] = {
        "label": label, "games": 2 * pairs, "pairs": pairs, "games_requested": 2 * pairs, "truncated": False,
        "game_log_fingerprint": f"fp_{label}", "game_log_agrees": True,
        "config_hash": "abc", "config_name": "pbt2_small.yaml", "git_sha": "deadbeef", "compile": "off",
        "eval_hoist": "4096", "eval_max_batch": 4096, "eval_max_batch_realised": 4096,
        "max_concurrent_games": 128, "arena_pool_size": 128, "eval_cuda_graphs": None, "device": "cuda",
        "eval_leaf_cap_bound": False, "buffer_full_flushes": 0,
        "pentanomial": {k: counts.get(b, 0) for k, b in PENTA.items()},
        "elo": None, "elo_ci95": [None, None],
    }
    row.update(over)
    return row


def _write_run(tmp: Path, label: str, *, candidate: str, reference: str, openings: list[str],
               game_scores: list[tuple[float, float]], row_over: dict[str, Any] | None = None,
               settings_over: dict[str, Any] | None = None,
               drop_game: tuple[int, int] | None = None) -> tuple[Path, Path]:
    """One arena run: game_scores[p] = the candidate's two game scores of pair p. Returns (games, results)."""
    log = tmp / f"{label}.games.jsonl"
    header = {"kind": "header", "fingerprint": f"fp_{label}",
              "settings": _settings(candidate, reference, len(openings), settings_over or {})}
    lines = [json.dumps(header)]
    for p, (fen, (s0, s1)) in enumerate(zip(openings, game_scores, strict=True)):
        for half, s in ((0, s0), (1, s1)):
            if (p, half) == drop_game:
                continue
            lines.append(json.dumps({"kind": "game", "pair_id": p, "half": half, "opening_fen": fen,
                                     "opening_uci": ["e2e4"], "score_candidate": s, "a_is_white": half == 0}))
    log.write_text("\n".join(lines) + "\n")
    row = _row(label, pairs=len(openings), scores=[a + b for a, b in game_scores], over=row_over or {})
    res = tmp / f"{label}.results.jsonl"
    res.write_text(json.dumps({"label": "someone_else", "game_log_fingerprint": "fp_x"}) + "\n"
                   + json.dumps(row) + "\n")
    return log, res


def _split(excess: float) -> tuple[float, float]:
    """Two game scores whose pair excess over 0.5 is ``excess`` (a multiple of 0.25)."""
    total = 2 * (excess + 0.5)
    return {0.0: (0.0, 0.0), 0.5: (0.5, 0.0), 1.0: (0.5, 0.5), 1.5: (1.0, 0.5), 2.0: (1.0, 1.0)}[total]


def _pair(tmp: Path, a_excess: list[float], b_excess: list[float], *, openings: list[str] | None = None,
          b_openings: list[str] | None = None, b_over: dict[str, Any] | None = None,
          b_settings: dict[str, Any] | None = None, drop_b: tuple[int, int] | None = None,
          rule: str | None = None, allow_legacy: bool = False) -> dict[str, Any]:
    """Run A (arm candidate) with per-pair candidate excess a_excess, run B (base candidate) with b_excess."""
    openings = openings or [f"fen{i}" for i in range(len(a_excess))]
    a_log, a_res = _write_run(tmp, "A", candidate=ARM, reference=BASE, openings=openings,
                              game_scores=[_split(x) for x in a_excess])
    b_log, b_res = _write_run(tmp, "B", candidate=BASE, reference=ARM, openings=b_openings or openings,
                              game_scores=[_split(x) for x in b_excess], row_over=b_over,
                              settings_over=b_settings, drop_game=drop_b)
    return pool_mod.run(a_log, b_log, a_res, b_res, allow_legacy=allow_legacy, rule=rule)


def test_a_planted_seat_bonus_cancels_exactly(tmp_path: Path) -> None:
    rng = random.Random(3)
    effect = [rng.choice((-0.25, 0.0, 0.25)) for _ in range(60)]
    delta = 0.25
    # candidate excess = effect + delta (arm in A); = -effect + delta (base in B)
    with_bonus = _pair(tmp_path, [e + delta for e in effect], [-e + delta for e in effect])["result"]
    (tmp_path / "nobonus").mkdir()
    without = _pair(tmp_path / "nobonus", effect, [-e for e in effect])["result"]
    mean_effect = sum(effect) / len(effect)
    assert with_bonus["pooled"]["excess"] == pytest.approx(mean_effect, abs=1e-15)
    assert with_bonus["pooled"]["excess"] == without["pooled"]["excess"]
    assert with_bonus["pooled"]["se"] == without["pooled"]["se"]
    assert with_bonus["seat_offset"]["excess"] == pytest.approx(delta, abs=1e-15)
    assert with_bonus["seat_offset"]["interval_status"] == "unavailable_zero_empirical_variance"
    assert without["seat_offset"]["excess"] == pytest.approx(0.0, abs=1e-15)
    # each orientation alone carries the bonus
    assert with_bonus["arm_as_candidate"]["excess"] == pytest.approx(mean_effect + delta)
    assert with_bonus["arm_as_reference"]["excess"] == pytest.approx(mean_effect - delta)


def _sample_game(rng: random.Random, mean_excess: float) -> float:
    p_win = 0.3 + mean_excess                       # P(1) = 0.3 + mu, P(0.5) = 0.4, P(0) = 0.3 - mu
    u = rng.random()
    return 1.0 if u < p_win else (0.5 if u < p_win + 0.4 else 0.0)


def test_a_planted_true_effect_is_recovered(tmp_path: Path) -> None:
    rng = random.Random(11)
    eps, delta, n = 0.05, 0.03, 3000
    openings = [f"fen{i}" for i in range(n)]
    a_games = [(_sample_game(rng, eps + delta), _sample_game(rng, eps + delta)) for _ in range(n)]
    b_games = [(_sample_game(rng, -eps + delta), _sample_game(rng, -eps + delta)) for _ in range(n)]
    a_log, a_res = _write_run(tmp_path, "A", candidate=ARM, reference=BASE, openings=openings, game_scores=a_games)
    b_log, b_res = _write_run(tmp_path, "B", candidate=BASE, reference=ARM, openings=openings, game_scores=b_games)
    r = pool_mod.run(a_log, b_log, a_res, b_res, rule="lb-gt-zero")
    pooled, seat = r["result"]["pooled"], r["result"]["seat_offset"]
    assert abs(pooled["excess"] - eps) < 3 * pooled["se"]
    assert abs(seat["excess"] - delta) < 3 * seat["se"]
    lo, hi = pooled["elo_ci95"]
    assert lo < pool_mod._elo(eps) < hi
    assert r["verdict"] == "success"
    assert r["result"]["n_clusters"] == n
    assert abs(r["result"]["ab_correlation"]) < 0.1     # independent synthetic runs
    # the interval is the arena convention: the ends of x +/- 1.96 SE through the Elo map
    assert lo == pytest.approx(pool_mod._elo(pooled["excess"] - 1.96 * pooled["se"]))


def test_verdict_reads_kill_and_unread() -> None:
    assert pool_mod.verdict({"elo_ci95": [-30.0, -2.0]}, "lb-gt-zero") == "kill"
    assert pool_mod.verdict({"elo_ci95": [-3.0, 9.0]}, "lb-gt-zero") == "unread"
    assert pool_mod.verdict({"elo_ci95": [None, None]}, "lb-gt-zero") == "unread"
    assert pool_mod.verdict({"elo_ci95": [0.5, 9.0]}, "lb-gt-zero") == "success"


def test_repeated_openings_form_one_cluster(tmp_path: Path) -> None:
    openings = ["x", "x", "y", "z"]
    r = _pair(tmp_path, [0.25, -0.25, 0.25, 0.0], [0.0, 0.0, -0.25, 0.0], openings=openings)["result"]
    assert r["n_clusters"] == 3
    assert r["openings_repeated_within_run"] == 1
    # x: e_A = 0, e_B = 0 -> d 0; y: e_A .25, e_B .25 -> .25; z: 0
    assert r["pooled"]["excess"] == pytest.approx(0.25 / 3)


def test_different_openings_are_refused(tmp_path: Path) -> None:
    ex = [0.0, 0.25, -0.25, 0.0]
    with pytest.raises(SystemExit, match="different openings"):
        _pair(tmp_path, ex, ex, b_openings=["fen0", "fen1", "fen2", "other"])


def test_the_same_openings_in_another_order_are_refused(tmp_path: Path) -> None:
    """The index -> opening maps must be identical, not merely the same set."""
    ex = [0.0, 0.25, -0.25, 0.0]
    with pytest.raises(SystemExit, match="index -> opening maps differ at 2 of 4"):
        _pair(tmp_path, ex, ex, b_openings=["fen1", "fen0", "fen2", "fen3"])


@pytest.mark.parametrize(("field", "value"), [("temperature", 0.5), ("seed", 8), ("max_plies", 450),
                                              ("syzygy", "other"), ("openings", "other.pgn")])
def test_mismatched_header_settings_are_refused(tmp_path: Path, field: str, value: Any) -> None:
    ex = [0.0, 0.25, -0.25]
    with pytest.raises(SystemExit, match=f"not a seat-balanced pair.*settings.{field}"):
        _pair(tmp_path, ex, ex, b_settings={field: value})


@pytest.mark.parametrize(("field", "value"), [("config_hash", "zzz"), ("eval_max_batch_realised", 2048),
                                              ("git_sha", "cafe"), ("max_concurrent_games", 64)])
def test_mismatched_row_fields_are_refused(tmp_path: Path, field: str, value: Any) -> None:
    ex = [0.0, 0.25, -0.25]
    with pytest.raises(SystemExit, match=f"not a seat-balanced pair.*{field}"):
        _pair(tmp_path, ex, ex, b_over={field: value})


def test_a_seat_setting_that_does_not_follow_the_arm_is_refused(tmp_path: Path) -> None:
    ex = [0.0, 0.25, -0.25]
    with pytest.raises(SystemExit, match=r"settings\.sims_candidate in A != settings\.sims_reference in B"):
        _pair(tmp_path, ex, ex, b_settings={"sims_reference": 128})


def test_unswapped_seats_are_refused(tmp_path: Path) -> None:
    ex = [0.0, 0.25, -0.25]
    with pytest.raises(SystemExit, match="seats are not swapped"):
        _pair(tmp_path, ex, ex, b_settings={"candidate": ARM, "reference": BASE})


def test_a_volatility_override_is_refused(tmp_path: Path) -> None:
    ex = [0.0, 0.25, -0.25]
    with pytest.raises(SystemExit, match="volatility_candidate"):
        _pair(tmp_path, ex, ex, b_settings={"volatility_candidate": {"anchor": 0.1}})


@pytest.mark.parametrize(("over", "match"), [
    ({"games": 4}, "games 4 != games_requested 6"),
    ({"truncated": True}, "truncated"),
    ({"game_log_agrees": False}, "game_log_agrees"),
    ({"buffer_full_flushes": 3}, "buffer_full_flushes 3"),
    ({"eval_max_batch_realised": None}, "eval_max_batch_realised is null"),
    ({"eval_leaf_cap_bound": True}, "eval_leaf_cap_bound"),
    ({"game_log_fingerprint": "other"}, "expected exactly one result row"),
    ({"pentanomial": {"WW": 3, "WD_DW": 0, "DD_WL": 0, "LD_DL": 0, "LL": 0}}, "pentanomial"),
])
def test_incomplete_or_unverifiable_runs_are_refused(tmp_path: Path, over: dict[str, Any], match: str) -> None:
    ex = [0.0, 0.25, -0.25]
    with pytest.raises(SystemExit, match=match):
        _pair(tmp_path, ex, ex, b_over=over)


def test_a_header_game_count_that_differs_from_the_row_is_refused(tmp_path: Path) -> None:
    ex = [0.0, 0.25, -0.25]
    with pytest.raises(SystemExit, match="log header games 8 != games_requested 6"):
        _pair(tmp_path, ex, ex, b_settings={"games": 8})


def test_a_missing_game_is_refused(tmp_path: Path) -> None:
    ex = [0.0, 0.25, -0.25]
    with pytest.raises(SystemExit, match="missing from the log"):
        _pair(tmp_path, ex, ex, drop_b=(1, 1))


def test_legacy_rows_need_the_explicit_waiver(tmp_path: Path) -> None:
    ex = [0.0, 0.25, -0.25]
    legacy_a, legacy_b = _legacy(tmp_path, ex)
    logs = (tmp_path / "A.games.jsonl", tmp_path / "B.games.jsonl")
    with pytest.raises(SystemExit, match="buffer_full_flushes absent"):
        pool_mod.run(*logs, legacy_a, legacy_b)
    r = pool_mod.run(*logs, legacy_a, legacy_b, allow_legacy=True)
    assert any("waived" in n for n in r["notes"])


def _legacy(tmp: Path, ex: list[float]) -> tuple[Path, Path]:
    _pair(tmp, ex, ex)
    out = []
    for label in ("A", "B"):
        res = tmp / f"{label}.results.jsonl"
        rows = [json.loads(line) for line in res.read_text().splitlines()]
        for r in rows:
            r.pop("buffer_full_flushes", None)
            r.pop("eval_max_batch_realised", None)
        res.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        out.append(res)
    return out[0], out[1]


def test_the_cli_writes_json_and_a_verdict(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _pair(tmp_path, [0.25] * 4 + [0.0], [0.25] * 4 + [0.0])
    out = tmp_path / "pooled.json"
    assert pool_mod.main(["--a-games", str(tmp_path / "A.games.jsonl"),
                          "--b-games", str(tmp_path / "B.games.jsonl"),
                          "--a-result", str(tmp_path / "A.results.jsonl"),
                          "--b-result", str(tmp_path / "B.results.jsonl"),
                          "--verdict-rule", "lb-gt-zero", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["verdict"] in {"success", "kill", "unread"}
    for key in ("excess", "se", "elo", "elo_ci95", "n_clusters"):
        assert key in report["result"]["pooled"]
    assert "ab_correlation" in report["result"]
    assert "seat_offset" in report["result"]
    assert "VERDICT (lb-gt-zero)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Reproduction on the banked same-net seat control (skipped where it is absent)
# ---------------------------------------------------------------------------

BANK = Path(os.environ.get("CAE_HEADSWAP_BANK",
                           Path.home() / "chess-artifacts" / "operations" / "rl-claude-20261119" / "headswap"))


@pytest.mark.skipif(not (BANK / "results" / "S_swap_bb_800.jsonl").is_file(), reason="banked seat control absent")
def test_banked_seat_control_reproduction(tmp_path: Path) -> None:
    """D_hybrid_bb_400 (hybrid base net as candidate) against S_swap_bb_800 (as reference), same seed.

    S played 400 pairs, D 200; S's pairs 0-199 are D's openings, so S is cut to those
    (a different openings set is refused, as it should be). Both seats hold the same
    net, so the pooled effect should be near 0 and the seat offset is D's and S's
    candidate-seat excess averaged over the 200 shared openings.
    """
    d_res, s_res = BANK / "results" / "D_hybrid_bb_400.jsonl", BANK / "results" / "S_swap_bb_800.jsonl"
    d_log = next((BANK / "arena_games").glob("D_hybrid_bb_400.*.games.jsonl"))
    s_log = next((BANK / "arena_games").glob("S_swap_bb_800.*.games.jsonl"))
    with pytest.raises(SystemExit, match="not a seat-balanced pair"):  # uncut: 400 vs 800 games
        pool_mod.run(d_log, s_log, d_res, s_res, allow_legacy=True)
    header, games = pool_mod.load_games(s_log)
    s_row = pool_mod.load_row(s_res, header["fingerprint"], None)
    keep = {k: v for k, v in games.items() if k[0] < 200}
    header = {**header, "settings": {**header["settings"], "games": 400}}
    cut_log = tmp_path / "S_cut.games.jsonl"
    cut_log.write_text("\n".join(json.dumps(x) for x in [header, *keep.values()]) + "\n")
    counts = Counter(keep[(p, 0)]["score_candidate"] + keep[(p, 1)]["score_candidate"] for p in range(200))
    s_row.update(games=400, pairs=200, games_requested=400,
                 pentanomial={k: counts.get(b, 0) for k, b in PENTA.items()})
    cut_res = tmp_path / "S_cut.results.jsonl"
    cut_res.write_text(json.dumps(s_row) + "\n")
    r = pool_mod.run(d_log, cut_log, d_res, cut_res, allow_legacy=True)
    res = r["result"]
    assert res["n_clusters"] == 200
    assert res["openings_repeated_within_run"] == 0
    assert res["arm_as_candidate"]["elo"] == pytest.approx(22.62, abs=0.01)   # D's own row elo
    assert res["pooled"]["excess"] == pytest.approx(PINNED_POOLED, abs=1e-9)
    assert res["seat_offset"]["excess"] == pytest.approx(PINNED_SEAT, abs=1e-9)
    # Same net in both seats: the pooled effect is noise around 0, and the seat offset
    # is the candidate seat's excess over these 200 openings (D +0.0325, S's first 200
    # pairs +0.0125), not the banked 998-pair pool across four samples (+6.4).
    assert res["pooled"]["elo_ci95"][0] < 0.0 < res["pooled"]["elo_ci95"][1]
    assert r["notes"]
    assert all("waived" in n for n in r["notes"])


PINNED_POOLED = 0.01      # (0.0325 + (-0.0125)) / 2: +6.9 Elo [-9.0, +23.0]
PINNED_SEAT = 0.0225      # (0.0325 - (-0.0125)) / 2: +15.6 Elo [-1.5, +32.9]
