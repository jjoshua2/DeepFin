"""Tests for the pooled Ordo fit wrapper.

These exist because the three properties `scripts/ordo_pooled_fit.py` itself
calls load-bearing — the `-z` Elo scale, `-M` convergence, and matchup
stratification — were the three nothing could catch regressing. Each is
asserted on the ACTUAL argv or the ACTUAL resampled games, not on a comment.

Nothing here needs the Ordo binary: `subprocess.run` is intercepted, which is
also what lets the argv assertions be exact.
"""

from __future__ import annotations

import math
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from scripts.ordo_pooled_fit import (
    ORDO_Z_MATCHING_400_SCALE,
    Game,
    Pair,
    block_bootstrap,
    completion_bias_report,
    contrast_samples,
    group_pairs,
    holm_reject,
    matchup_key,
    parse_ordo,
    pct,
    player_samples,
    read_games,
    run_ordo,
)

ORDO = Path("/nonexistent/ordo")


# --------------------------------------------------------------------------
# The Elo scale (-z). A wrong value silently rescales every number reported.
# --------------------------------------------------------------------------

def test_z_constant_equals_our_400_point_scale() -> None:
    # arena_standard's Elo is -400*log10(1/p-1)  => invbeta = 400/ln(10).
    # Ordo's -z is the rating at 76%, i.e. invbeta * -ln(1/0.76-1).
    expected = 400.0 / math.log(10.0) * (-math.log(1.0 / 0.76 - 1.0))
    assert pytest.approx(expected, abs=5e-4) == ORDO_Z_MATCHING_400_SCALE
    # And it is NOT Ordo's default, which is the whole point.
    assert abs(ORDO_Z_MATCHING_400_SCALE - 202.0) > 1.0


def _capture_argv(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: object) -> object:
        seen.append(list(cmd))
        Path(cmd[cmd.index("-o") + 1]).write_text(
            "   # PLAYER    :  RATING  ERROR  POINTS  PLAYED   (%)\n"
            "   1 A         :    10.0    5.0    60.0     100    60\n"
            "   2 B         :     0.0    5.0    40.0     100    40\n"
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return seen


def test_run_ordo_argv_pins_our_elo_scale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    seen = _capture_argv(monkeypatch)
    run_ordo(ORDO, tmp_path / "in.pgn", tmp_path / "out.txt", anchor="B")
    assert len(seen) == 1
    cmd = seen[0]
    assert "-z" in cmd, "Ordo would silently use its own 202 scale"
    assert cmd[cmd.index("-z") + 1] == f"{ORDO_Z_MATCHING_400_SCALE:.4f}"


def test_run_ordo_argv_forces_maximum_likelihood(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # -M is what stops the -D + -s non-convergence hang (1/30 datasets at
    # -s 1000). A hang is deterministic per dataset, so losing this flag does
    # not fail loudly -- it wedges.
    seen = _capture_argv(monkeypatch)
    run_ordo(ORDO, tmp_path / "in.pgn", tmp_path / "out.txt", anchor="B", sims=100)
    cmd = seen[0]
    assert "-M" in cmd
    assert "-D" in cmd
    assert "-W" in cmd
    assert cmd[cmd.index("-s") + 1] == "100"


def test_run_ordo_omits_anchor_when_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    seen = _capture_argv(monkeypatch)
    run_ordo(ORDO, tmp_path / "in.pgn", tmp_path / "out.txt", anchor=None)
    assert "-A" not in seen[0]


def test_run_ordo_returns_none_on_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    def boom(cmd: list[str], **_kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd, 1.0)

    monkeypatch.setattr(subprocess, "run", boom)
    assert run_ordo(ORDO, tmp_path / "i.pgn", tmp_path / "o.txt",
                    anchor="B") is None


# --------------------------------------------------------------------------
# Stratification: resample WITHIN matchup, preserving each matchup's own count.
# --------------------------------------------------------------------------

def _pairs(a: str, b: str, n: int, start: int = 0) -> list[Pair]:
    out: list[Pair] = []
    for i in range(n):
        gs = [
            Game(white=a, black=b, result="1-0", pair_id=str(i), plies=80,
                 duration_s=1.0, order=start + 2 * i),
            Game(white=b, black=a, result="0-1", pair_id=str(i), plies=80,
                 duration_s=1.0, order=start + 2 * i + 1),
        ]
        out.append(Pair(matchup=matchup_key(a, b), games=gs))
    return out


def _half_pairs(
    a: str, b: str, n_complete: int, n_singleton: int, start: int = 0,
) -> list[Pair]:
    """Complete pairs PLUS singleton half-pairs, the partial-run regime.

    Every singleton is deliberately given ``a`` as WHITE. That makes the block
    count exactly recoverable from the resampled PGN: a complete pair
    contributes one a-white game and one b-white game, a singleton contributes
    one a-white game, so ``whites[a] == blocks drawn``. Without that convention
    the two are not separable from a colour-blind game count.
    """
    out = _pairs(a, b, n_complete, start)
    base = start + 2 * n_complete
    for i in range(n_singleton):
        out.append(Pair(matchup=matchup_key(a, b), games=[
            Game(white=a, black=b, result="1-0", pair_id=str(n_complete + i),
                 plies=80, duration_s=1.0, order=base + i),
        ]))
    return out


def _bootstrap_with_recorder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    by_matchup: dict[tuple[str, str], list[Pair]], *, reps: int, unit: str,
) -> list[dict[tuple[str, str], Counter[str]]]:
    """Run the real block_bootstrap and read back what write_pgn ACTUALLY
    wrote each replication, counted per matchup."""
    import random as _random

    seen: list[dict[tuple[str, str], Counter[str]]] = []

    def fake_run(cmd: list[str], **_kwargs: object) -> object:
        pgn = Path(cmd[cmd.index("-p") + 1])
        # Per matchup, a Counter of WHICH PLAYER HELD WHITE. Total games is its
        # sum; pair integrity is whether the two sides are equal.
        counts: dict[tuple[str, str], Counter[str]] = {}
        white = ""
        for line in pgn.read_text().splitlines():
            if line.startswith("[White "):
                white = line.split('"')[1]
            elif line.startswith("[Black "):
                black = line.split('"')[1]
                counts.setdefault(matchup_key(white, black), Counter())[white] += 1
        seen.append(counts)
        Path(cmd[cmd.index("-o") + 1]).write_text(
            "   # PLAYER    :  RATING  POINTS  PLAYED   (%)\n"
            "   1 armA      :    10.0    60.0     100    60\n"
            "   2 armB      :     0.0    40.0     100    40\n"
            "   3 armC      :    -5.0    40.0     100    40\n"
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    block_bootstrap(ORDO, by_matchup, reps=reps, anchor="armC",
                    rng=_random.Random(5), tmp=tmp_path, unit=unit)
    return seen


@pytest.mark.parametrize("unit", ["pair", "game"])
def test_bootstrap_preserves_each_matchup_game_count_exactly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, unit: str,
) -> None:
    by = {
        matchup_key("armA", "armB"): _pairs("armA", "armB", 20, 0),
        matchup_key("armB", "armC"): _pairs("armB", "armC", 3, 100),
        matchup_key("armA", "armC"): _pairs("armA", "armC", 7, 200),
    }
    seen = _bootstrap_with_recorder(monkeypatch, tmp_path, by, reps=25, unit=unit)
    assert len(seen) == 25
    expected = {m: 2 * len(ps) for m, ps in by.items()}
    for rep in seen:
        # Every matchup keeps its OWN count in EVERY replication. Pooling the
        # draw across matchups would let these float, which resamples the
        # design rather than the data.
        assert {m: sum(c.values()) for m, c in rep.items()} == expected


def test_bootstrap_never_mixes_games_across_matchups(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # Deliberately LOPSIDED (30 vs 2): if the draw were pooled across matchups,
    # the small matchup would be flooded by the large one. Asserting only that
    # both matchups are PRESENT does not detect this -- pooled draws still
    # produce games from both -- so the assertion has to be on the exact
    # composition.
    by = {
        matchup_key("armA", "armB"): _pairs("armA", "armB", 30, 0),
        matchup_key("armB", "armC"): _pairs("armB", "armC", 2, 100),
    }
    seen = _bootstrap_with_recorder(monkeypatch, tmp_path, by, reps=20,
                                    unit="pair")
    expected = {m: 2 * len(ps) for m, ps in by.items()}
    for rep in seen:
        assert set(rep) == set(by), "a matchup appeared or vanished"
        assert {m: sum(c.values()) for m, c in rep.items()} == expected


def test_pair_unit_draws_whole_pairs_not_halves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # A pair block must enter the resample as BOTH its games; drawing halves
    # would silently degrade the pair bootstrap into a game bootstrap.
    #
    # Parity is NOT the property to assert -- drawing 6 single halves is also an
    # even count. The property is COLOUR BALANCE: each whole pair contributes
    # exactly one game with armA as white and one with armB as white, so the two
    # must be equal. Half-draws break that balance at random.
    key = matchup_key("armA", "armB")
    by = {key: _pairs("armA", "armB", 6, 0)}
    seen = _bootstrap_with_recorder(monkeypatch, tmp_path, by, reps=15,
                                    unit="pair")
    for rep in seen:
        whites = rep[key]
        assert sum(whites.values()) == 12
        assert whites["armA"] == 6
        assert whites["armB"] == 6


def test_bootstrap_preserves_BLOCK_count_not_game_count_with_half_pairs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The partial-run regime this whole tool exists to serve.

    With only complete pairs, "resample N blocks" and "resample to N*2 games"
    are indistinguishable -- every block is exactly 2 games. Half-pairs
    separate them, and the block count is the correct invariant: a bootstrap
    that topped up to a fixed GAME count would silently over-weight complete
    pairs and under-weight the partial ones.

    ``whites[armA] == blocks drawn`` by the fixture's colour convention, so
    this asserts the block count EXACTLY rather than inferring it.
    """
    key = matchup_key("armA", "armC")
    n_complete, n_singleton = 6, 4
    by = {key: _half_pairs("armA", "armC", n_complete, n_singleton)}
    blocks = n_complete + n_singleton

    seen = _bootstrap_with_recorder(monkeypatch, tmp_path, by, reps=30,
                                    unit="pair")
    assert len(seen) == 30
    totals: set[int] = set()
    for rep in seen:
        whites = rep[key]
        assert whites["armA"] == blocks, (
            f"block count not preserved: drew {whites['armA']} blocks, "
            f"expected {blocks}"
        )
        total = sum(whites.values())
        # blocks <= games <= 2*blocks, and games == blocks + #complete drawn
        assert blocks <= total <= 2 * blocks
        totals.add(total)
    # The GAME count must vary -- if it were constant, the implementation is
    # preserving games rather than blocks and the assertion above is passing
    # for the wrong reason.
    assert len(totals) > 1, (
        f"game count was constant at {totals}; with half-pairs a block "
        f"bootstrap must produce a varying game count"
    )


# --------------------------------------------------------------------------
# Contrast CIs (the number a reader acts on).
# --------------------------------------------------------------------------

def test_contrast_is_paired_within_replication() -> None:
    # Positively correlated players: the players move together, so the CONTRAST
    # has zero spread even though each player's own spread is large.
    reps = [{"A": float(x) + 10.0, "B": float(x)} for x in range(100)]
    assert player_samples(reps, "A")[:3] == [10.0, 11.0, 12.0]
    diffs = contrast_samples(reps, "A", "B")
    assert diffs == [10.0] * 100
    assert pct(diffs, 0.975) - pct(diffs, 0.025) == 0.0
    spread_a = pct(player_samples(reps, "A"), 0.975) - pct(
        player_samples(reps, "A"), 0.025)
    assert spread_a > 90.0


def test_contrast_pairing_is_not_recoverable_from_sorted_marginals() -> None:
    # The case above passes even for a WRONG implementation that sorts the two
    # players' marginals and subtracts, because that data is perfectly rank
    # correlated. NEGATIVELY correlated replications separate them: the true
    # paired difference has large spread, while sorted marginals collapse it to
    # a constant. Same marginals, opposite answer -- which is the whole point of
    # differencing within a replication.
    reps = [{"A": float(x), "B": -float(x)} for x in range(100)]
    diffs = contrast_samples(reps, "A", "B")
    assert diffs[0] == 0.0
    assert diffs[-1] == pytest.approx(198.0)
    assert pct(diffs, 0.975) - pct(diffs, 0.025) > 150.0

    xs = sorted(r["A"] for r in reps)
    ys = sorted(r["B"] for r in reps)
    sorted_marginal = [x - y for x, y in zip(xs, ys)]
    assert len(set(sorted_marginal)) == 1  # a constant: all spread destroyed


def test_contrast_skips_replications_missing_a_player() -> None:
    reps = [{"A": 5.0, "B": 1.0}, {"A": 7.0}, {"A": 9.0, "B": 2.0}]
    assert contrast_samples(reps, "A", "B") == [4.0, 7.0]


# --------------------------------------------------------------------------
# Multiplicity in the missingness guard.
# --------------------------------------------------------------------------

def test_holm_is_stricter_than_uncorrected_alpha() -> None:
    # Six tests, one at p=0.03: significant uncorrected, not after Holm.
    ps = [0.03, 0.2, 0.4, 0.5, 0.6, 0.9]
    assert any(p < 0.05 for p in ps)
    assert not any(holm_reject(ps, alpha=0.05))


def test_holm_still_rejects_a_genuinely_small_p() -> None:
    ps = [0.0005, 0.2, 0.4, 0.5, 0.6, 0.9]
    assert holm_reject(ps, alpha=0.05)[0]


def test_holm_step_down_stops_at_first_failure() -> None:
    # 0.001 <= 0.05/3 rejects; 0.03 > 0.05/2 does not, so 0.04 cannot either.
    assert holm_reject([0.001, 0.03, 0.04], alpha=0.05) == [True, False, False]


def test_holm_handles_empty() -> None:
    assert holm_reject([], alpha=0.05) == []


def test_holm_family_is_global_not_per_matchup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Holm must be applied ONCE over every test in the run.

    Scoping it per matchup passes every other test in this file, because
    ``test_holm_*`` exercise the function in isolation and the firing test uses
    a single matchup -- where per-matchup and global scoping coincide. It is
    not cosmetic: per-matchup scoping measurably raises the false-positive rate
    (0.073 vs 0.013 in the reviewer's run), because each matchup gets its own
    fresh alpha budget.
    """
    import random as _random

    import scripts.ordo_pooled_fit as fit_mod

    calls: list[list[float]] = []
    real = fit_mod.holm_reject

    def spy(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
        calls.append(list(pvalues))
        return real(pvalues, alpha=alpha)

    monkeypatch.setattr(fit_mod, "holm_reject", spy)

    rng = _random.Random(17)
    by: dict[tuple[str, str], list[Pair]] = {}
    for a, b in (("armA", "armB"), ("armB", "armC"), ("armA", "armC")):
        blocks = _pairs(a, b, 12)
        for blk in blocks:
            for g in blk.games:
                g.result = rng.choice(["1-0", "1/2-1/2", "0-1"])
        by[matchup_key(a, b)] = blocks

    fit_mod.completion_bias_report(by, rng=rng, n_perm=200)

    assert len(calls) == 1, (
        f"Holm must be applied once over the whole family; it was called "
        f"{len(calls)} times (per-matchup scoping)"
    )
    assert len(calls[0]) == 6, (
        f"family must be 2 tests x 3 matchups = 6; got {len(calls[0])}"
    )


def test_guard_stays_quiet_on_order_independent_outcomes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import random as _random

    rng = _random.Random(4)
    by: dict[tuple[str, str], list[Pair]] = {}
    for a, b in (("armA", "armB"), ("armB", "armC")):
        blocks = _pairs(a, b, 30)
        for blk in blocks:  # outcomes independent of order
            for g in blk.games:
                g.result = rng.choice(["1-0", "1/2-1/2", "0-1"])
        by[matchup_key(a, b)] = blocks
    assert completion_bias_report(by, rng=rng, n_perm=300) is False
    assert "Holm-corrected" in capsys.readouterr().out


def test_guard_fires_when_decisive_pairs_finish_first(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import random as _random

    by: dict[tuple[str, str], list[Pair]] = {}
    blocks = _pairs("armA", "armB", 40)
    for i, blk in enumerate(blocks):
        # First half decisive (2-0), second half drawn (1-1).
        if i < 20:
            blk.games[0].result = "1-0"
            blk.games[1].result = "0-1"  # armA wins both
        else:
            blk.games[0].result = "1/2-1/2"
            blk.games[1].result = "1/2-1/2"
    by[matchup_key("armA", "armB")] = blocks
    assert completion_bias_report(by, rng=_random.Random(9), n_perm=300) is True
    assert "FLAGGED" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Parsing and grouping.
# --------------------------------------------------------------------------

def test_parse_ordo_is_header_aware() -> None:
    with_err = (
        "   # PLAYER    :  RATING  ERROR  POINTS  PLAYED   (%)\n"
        "   1 ArmA      :  2600.0   ----  1712.0    2400    71\n"
        "   2 ArmB      :  2501.6   13.9  1309.5    2400    55\n"
    )
    no_err = (
        "   # PLAYER    :  RATING  POINTS  PLAYED   (%)\n"
        "   1 A         :    34.0   219.5     400    55\n"
    )
    assert parse_ordo(with_err)["ArmB"] == (2501.6, 13.9)
    assert math.isnan(parse_ordo(with_err)["ArmA"][1])
    # Without -s there is no ERROR column; POINTS must NOT be read as the error.
    rating, err = parse_ordo(no_err)["A"]
    assert rating == 34.0
    assert math.isnan(err)


def test_read_games_drops_unfinished_and_keeps_pair_tags(tmp_path: Path) -> None:
    pgn = tmp_path / "a.pgn"
    pgn.write_text(
        '[White "a"]\n[Black "b"]\n[Result "1-0"]\n[PairId "3"]\n[Plies "77"]\n'
        '[GameDurationSec "12.5"]\n\n1-0\n\n'
        '[White "a"]\n[Black "b"]\n[Result "*"]\n[PairId "4"]\n\n*\n\n'
    )
    games = read_games([pgn])
    assert len(games) == 1  # "*" is DISCARD in Ordo too
    assert games[0].pair_id == "3"
    assert games[0].plies == 77
    assert games[0].duration_s == 12.5


def test_group_pairs_does_not_merge_same_pairid_across_matchups() -> None:
    games = [
        Game("a", "b", "1-0", "0", None, None, 0),
        Game("b", "a", "1-0", "0", None, None, 1),
        Game("c", "d", "1-0", "0", None, None, 2),
        Game("d", "c", "1-0", "0", None, None, 3),
    ]
    by, unpaired = group_pairs(games)
    assert unpaired == 0
    assert len(by) == 2
    assert all(len(ps) == 1 and ps[0].complete for ps in by.values())


def test_group_pairs_treats_missing_pairid_as_singleton_blocks() -> None:
    games = [
        Game("a", "b", "1-0", None, None, None, 0),
        Game("b", "a", "0-1", None, None, None, 1),
    ]
    by, unpaired = group_pairs(games)
    assert unpaired == 2
    assert len(by[matchup_key("a", "b")]) == 2  # NOT collapsed into one pair
