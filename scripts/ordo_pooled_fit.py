#!/usr/bin/env python3
"""Pooled multi-player rating fit (Ordo) with a PAIR-LEVEL block bootstrap.

Why this wrapper exists
-----------------------
Ordo fits every player's rating jointly from one pooled game set, models white
advantage and draw rate, and handles unbalanced/incomplete pairings — none of
which a pairwise arena can do. What it CANNOT do is exploit our mirrored
openings: its entire per-game state is ``{whiteplayer, blackplayer, score}``
(``mytypes.h:90``) and its ``-s`` mode is a PARAMETRIC Monte Carlo that
regenerates each game independently from the fitted model (``sim.c:
simulate_scores``). So its interval is an independent-games interval.

Measured, on synthetic paired-opening data at a known Elo difference: Ordo's
point estimate is unbiased and its ``-s`` interval is only mildly conservative
for a balanced book, but the pairing gain it forgoes grows with the opening
book's colour imbalance (95% width ratio independent/paired: 1.00 at 0 Elo of
per-opening bias, 1.07 at 120, 1.17 at 200, 1.31 at 300, 1.53 at 450). Our
production book is a UHO (unbalanced) book, so that is not a negligible regime.

This wrapper therefore keeps Ordo's joint fit for the POINT estimates and
replaces its interval with a block bootstrap that resamples whole PAIRS.

Two things it is deliberate about
---------------------------------
1. **Stratified by MATCHUP, preserving each matchup's own pair count.** The
   whole reason to pool is that matchups have different completed-pair counts.
   Resampling pairs pooled across matchups would perturb the relative game
   counts between matchups from replication to replication — that resamples the
   DESIGN, not the data, and answers a question nobody asked.
2. **It refuses to be quiet about informative missingness.** A block bootstrap
   is correct only if the pairs you HAVE are representative of the pairs you
   WOULD have. If pairs complete at different rates for reasons correlated with
   the arms (one arm slower per move, one drawing more and playing longer), a
   partial matchup's completed pairs are its FAST pairs, which are
   systematically more decisive — the point estimate is then biased and a
   bootstrap around it returns a tight interval around the wrong number. So the
   completion-order diagnostic runs ALWAYS and prints with the result. A guard
   nobody runs is not a guard.
"""

from __future__ import annotations

import argparse
import random
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ORDO = Path.home() / "local_engines" / "ordo" / "ordo"

# Ordo's default -z 202 sets invbeta = 202 / -ln(1/0.76-1) = 175.2447, while the
# arena's Elo is -400*log10(1/p-1), i.e. invbeta = 400/ln(10) = 173.7178. Ordo's
# number is therefore 0.88% LARGER than ours for the same score. Run it on OUR
# ruler so the two are comparable without a silent rescale.
ORDO_Z_MATCHING_400_SCALE = 200.2409

RESULT_SCORE = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}


def matchup_key(white: str, black: str) -> tuple[str, str]:
    """Order-independent matchup identity (a two-tuple, not a sorted list)."""
    return (white, black) if white <= black else (black, white)


@dataclass
class Game:
    white: str
    black: str
    result: str
    pair_id: str | None
    plies: int | None
    duration_s: float | None
    order: int  # position in the file == completion order (games are appended)


@dataclass
class Pair:
    matchup: tuple[str, str]  # sorted engine names
    games: list[Game] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return len(self.games) == 2

    @property
    def completion_order(self) -> int:
        """When the pair FINISHED = the order of its last game."""
        return max(g.order for g in self.games)

    def score_for(self, player: str) -> float:
        total = 0.0
        for g in self.games:
            s = RESULT_SCORE[g.result]
            total += s if g.white == player else 1.0 - s
        return total


def read_games(paths: list[Path]) -> list[Game]:
    import chess.pgn

    games: list[Game] = []
    order = 0
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            while True:
                g = chess.pgn.read_game(fh)
                if g is None:
                    break
                h = g.headers
                res = h.get("Result", "*")
                if res not in RESULT_SCORE:
                    continue  # Ordo drops "*" too (pgnget.c DISCARD)
                plies = h.get("Plies")
                dur = h.get("GameDurationSec")
                games.append(Game(
                    white=h.get("White", "?"),
                    black=h.get("Black", "?"),
                    result=res,
                    pair_id=h.get("PairId"),
                    plies=int(plies) if plies is not None and plies.isdigit() else None,
                    duration_s=float(dur) if dur is not None else None,
                    order=order,
                ))
                order += 1
    return games


def group_pairs(games: list[Game]) -> tuple[dict[tuple[str, str], list[Pair]], int]:
    """Group into pairs keyed by (matchup, PairId). Returns (pairs by matchup,
    n_unpaired). A game with no PairId becomes its own singleton block, so a
    mixed file degrades to a game-level bootstrap for those rows instead of
    silently grouping unrelated games."""
    buckets: dict[tuple[tuple[str, str], str], Pair] = {}
    singles: list[Pair] = []
    for g in games:
        matchup = matchup_key(g.white, g.black)
        if g.pair_id is None:
            singles.append(Pair(matchup=matchup, games=[g]))
            continue
        buckets.setdefault((matchup, g.pair_id), Pair(matchup=matchup)).games.append(g)
    by_matchup: dict[tuple[str, str], list[Pair]] = defaultdict(list)
    for (matchup, _pid), pair in buckets.items():
        by_matchup[matchup].append(pair)
    for p in singles:
        by_matchup[p.matchup].append(p)
    for lst in by_matchup.values():
        lst.sort(key=lambda p: p.completion_order)
    return dict(by_matchup), len(singles)


def write_pgn(path: Path, games: list[Game]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for i, g in enumerate(games, 1):
            fh.write(
                f'[Event "boot"]\n[Site "-"]\n[Date "????.??.??"]\n[Round "{i}"]\n'
                f'[White "{g.white}"]\n[Black "{g.black}"]\n'
                f'[Result "{g.result}"]\n\n{g.result}\n\n'
            )


def run_ordo(
    ordo: Path, pgn: Path, out: Path, *, anchor: str | None, sims: int = 0,
    timeout: float = 120.0,
) -> dict[str, tuple[float, float]] | None:
    cmd = [
        str(ordo), "-Q", "-p", str(pgn), "-o", str(out), "-a", "0",
        # -M (maximum likelihood) is REQUIRED, not stylistic. With -D (auto draw
        # rate) and -s, Ordo's default estimator can fail to converge on an
        # unlucky SIMULATED replication and spin forever: measured 1/30 datasets
        # hung at -s 1000, 0/30 with -M and 0/30 with a fixed -d. The sim RNG is
        # fixed-seeded (main.c:981), so a hang is deterministic per dataset.
        "-W", "-D", "-M",
        "-z", f"{ORDO_Z_MATCHING_400_SCALE:.4f}", "-n", "1",
    ]
    if anchor is not None:
        cmd += ["-A", anchor]
    if sims:
        cmd += ["-s", str(sims)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    except subprocess.CalledProcessError as exc:
        print(f"[ordo] FAILED: {exc.stderr.strip()[:400]}", file=sys.stderr)
        return None
    return parse_ordo(out.read_text())


def parse_ordo(text: str) -> dict[str, tuple[float, float]]:
    """name -> (rating, error). HEADER-AWARE: the ERROR column exists only when
    ``-s`` was passed, and a fixed-offset parser silently returns POINTS as the
    error without it."""
    out: dict[str, tuple[float, float]] = {}
    err_idx: int | None = None
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "#" and "PLAYER" in parts:
            cols = [p for p in parts if p != ":"]
            err_idx = cols.index("ERROR") + 1 if "ERROR" in cols else None
            continue
        if len(parts) < 4 or not parts[0].rstrip(".").isdigit() or parts[2] != ":":
            continue
        try:
            rating = float(parts[3])
        except ValueError:
            continue
        err = float("nan")
        if err_idx is not None and err_idx < len(parts):
            tok = parts[err_idx]
            err = float("nan") if tok.startswith("-") else float(tok)
        out[parts[1]] = (rating, err)
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return 0.0 if dx == 0 or dy == 0 or n < 3 else num / (dx * dy)


def completion_bias_report(
    by_matchup: dict[tuple[str, str], list[Pair]], *, rng: random.Random,
    n_perm: int = 2000,
) -> bool:
    """Does completion ORDER correlate with pair OUTCOME within a matchup?

    Returns True if any matchup trips the flag. This is the check that a
    bootstrap CANNOT substitute for: resampling the pairs you have cannot tell
    you that the pairs you are MISSING are different.
    """
    print("\n=== informative-missingness diagnostic ===")
    print("  (completion order vs pair decisiveness; a partial matchup's "
          "completed pairs are its FAST pairs)")
    flagged = False
    counts = {m: len(ps) for m, ps in by_matchup.items()}
    if len(set(counts.values())) > 1:
        print(f"  UNEQUAL pair counts across matchups: "
              f"{ {'-'.join(m): n for m, n in counts.items()} }")
        print("  -> the fit is only unbiased if completion is INDEPENDENT of "
              "outcome. Checking that now.")
    for matchup, pairs in sorted(by_matchup.items()):
        complete = [p for p in pairs if p.complete]
        if len(complete) < 8:
            print(f"  {'-'.join(matchup):40s} only {len(complete)} complete "
                  f"pairs — too few to test")
            continue
        a = matchup[0]
        order = [float(p.completion_order) for p in complete]
        # Decisiveness, not signed score: the worry is that early pairs are more
        # DECISIVE, which inflates |Elo| regardless of which side wins.
        decisive = [abs(p.score_for(a) - 1.0) for p in complete]
        signed = [p.score_for(a) for p in complete]
        rho_d = spearman(order, decisive)
        rho_s = spearman(order, signed)
        # Permutation p-value: shuffle outcomes against completion order.
        hits_d = hits_s = 0
        for _ in range(n_perm):
            perm = decisive[:]
            rng.shuffle(perm)
            if abs(spearman(order, perm)) >= abs(rho_d):
                hits_d += 1
            perm2 = signed[:]
            rng.shuffle(perm2)
            if abs(spearman(order, perm2)) >= abs(rho_s):
                hits_s += 1
        p_d = (hits_d + 1) / (n_perm + 1)
        p_s = (hits_s + 1) / (n_perm + 1)
        plies = [p.games[0].plies for p in complete if p.games[0].plies is not None]
        ply_note = ""
        if len(plies) >= 8:
            half = len(plies) // 2
            ply_note = (f"  plies first/last half: "
                        f"{statistics.fmean(plies[:half]):.0f}/"
                        f"{statistics.fmean(plies[half:]):.0f}")
        bad = p_d < 0.05 or p_s < 0.05
        flagged |= bad
        print(f"  {'-'.join(matchup):40s} n={len(complete):4d} "
              f"rho(order,decisive)={rho_d:+.3f} p={p_d:.3f}  "
              f"rho(order,score)={rho_s:+.3f} p={p_s:.3f}"
              f"{'  <-- FLAG' if bad else ''}{ply_note}")
    if flagged:
        print("  ⚑ FLAGGED: completion order predicts outcome. The point "
              "estimate may be biased by WHICH pairs finished, and no bootstrap "
              "corrects that. Do not read a verdict off this fit — finish the "
              "matchup or explain the mechanism.")
    else:
        print("  OK: no detected association between completion order and "
              "outcome (absence of evidence at this n, not proof).")
    return flagged


def block_bootstrap(
    ordo: Path, by_matchup: dict[tuple[str, str], list[Pair]], *,
    reps: int, anchor: str | None, rng: random.Random, tmp: Path,
    unit: str = "pair",
) -> dict[str, list[float]]:
    """Resample WITHIN each matchup to that matchup's own count.

    ``unit='pair'`` resamples whole pairs (correct: keeps the mirrored games
    together). ``unit='game'`` resamples individual games and exists only to
    SHOW the difference the pairing makes.
    """
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(reps):
        drawn: list[Game] = []
        for pairs in by_matchup.values():
            if unit == "pair":
                for _ in range(len(pairs)):
                    drawn.extend(rng.choice(pairs).games)
            else:
                flat = [g for p in pairs for g in p.games]
                drawn.extend(rng.choice(flat) for _ in range(len(flat)))
        write_pgn(tmp / "boot.pgn", drawn)
        est = run_ordo(ordo, tmp / "boot.pgn", tmp / "boot.txt", anchor=anchor)
        if est is None:
            continue
        for name, (rating, _) in est.items():
            samples[name].append(rating)
    return samples


def pct(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return s[idx]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pgn", nargs="+", type=Path, help="arena PGN file(s)")
    ap.add_argument("--ordo", type=Path, default=DEFAULT_ORDO)
    ap.add_argument("--anchor", default=None,
                    help="player whose rating is fixed at 0 (default: pool average)")
    ap.add_argument("--bootstrap", type=int, default=500,
                    help="pair-level block bootstrap replications (default 500)")
    ap.add_argument("--also-game-bootstrap", action="store_true",
                    help="ALSO run a naive per-game bootstrap, to show the "
                         "size of the pairing gain rather than assert it")
    ap.add_argument("--ordo-sims", type=int, default=1000,
                    help="Ordo's own -s replications, for comparison (default 1000)")
    ap.add_argument("--seed", type=int, default=20260813)
    args = ap.parse_args()

    if not args.ordo.exists():
        raise SystemExit(f"ordo binary not found at {args.ordo} (build it first)")

    games = read_games(args.pgn)
    if not games:
        raise SystemExit("no decisive/drawn games found in the PGN(s)")
    by_matchup, n_unpaired = group_pairs(games)
    players = sorted({g.white for g in games} | {g.black for g in games})
    n_complete = sum(1 for ps in by_matchup.values() for p in ps if p.complete)
    n_partial = sum(1 for ps in by_matchup.values() for p in ps if not p.complete)
    print(f"{len(games)} games, {len(players)} players: {', '.join(players)}")
    print(f"{n_complete} complete pairs, {n_partial} half pairs, "
          f"{n_unpaired} games without a PairId")
    for m, ps in sorted(by_matchup.items()):
        print(f"  {'-'.join(m):40s} {len(ps):4d} blocks "
              f"({sum(len(p.games) for p in ps)} games)")

    rng = random.Random(args.seed)
    tmp = Path(tempfile.mkdtemp(prefix="ordofit_"))

    write_pgn(tmp / "all.pgn", games)
    point = run_ordo(args.ordo, tmp / "all.pgn", tmp / "all.txt",
                     anchor=args.anchor, sims=args.ordo_sims)
    if point is None:
        raise SystemExit(
            "Ordo did not return a fit (timeout or error). If it hung, that is "
            "the -D auto-draw-rate non-convergence — this wrapper already passes "
            "-M, so report the input.")

    flagged = completion_bias_report(by_matchup, rng=rng)

    print(f"\n=== ratings (Ordo joint fit; ERROR = its own -s {args.ordo_sims} "
          f"95% margin, independent-games) ===")
    for name in sorted(point, key=lambda n: -point[n][0]):
        r, e = point[name]
        print(f"  {name:40s} {r:+8.1f}  +-{e:6.1f}")

    print(f"\n=== pair-level block bootstrap ({args.bootstrap} reps, stratified "
          f"by matchup, each to its own count) ===")
    boot = block_bootstrap(args.ordo, by_matchup, reps=args.bootstrap,
                           anchor=args.anchor, rng=rng, tmp=tmp, unit="pair")
    game_boot: dict[str, list[float]] = {}
    if args.also_game_bootstrap:
        game_boot = block_bootstrap(args.ordo, by_matchup, reps=args.bootstrap,
                                    anchor=args.anchor, rng=rng, tmp=tmp,
                                    unit="game")

    hdr = f"  {'player':40s} {'rating':>8s} {'pair-boot 95% CI':>24s} {'hw':>7s}"
    if game_boot:
        hdr += f" {'game-boot hw':>13s} {'ratio':>6s}"
    hdr += f" {'ordo -s hw':>11s}"
    print(hdr)
    for name in sorted(point, key=lambda n: -point[n][0]):
        vals = boot.get(name, [])
        lo, hi = pct(vals, 0.025), pct(vals, 0.975)
        hw = (hi - lo) / 2
        line = (f"  {name:40s} {point[name][0]:+8.1f} "
                f"[{lo:+8.1f},{hi:+8.1f}] {hw:7.1f}")
        if game_boot:
            gv = game_boot.get(name, [])
            ghw = (pct(gv, 0.975) - pct(gv, 0.025)) / 2
            ratio = ghw / hw if hw > 0 else float("nan")
            line += f" {ghw:13.1f} {ratio:6.3f}"
        line += f" {point[name][1]:11.1f}"
        print(line)
    if game_boot:
        print("  ratio = naive-per-game width / pair-block width. >1 means the "
              "mirrored pairing is doing real work on THIS book.")

    print("\n⚑ These intervals are correct CONDITIONAL on the pair counts in "
          "hand. Refitting as games arrive is legitimate VISIBILITY; reading a "
          "verdict at whichever refit looks good is optional stopping, and the "
          "bootstrap does not license it. Fix the read point in advance.")
    return 2 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
