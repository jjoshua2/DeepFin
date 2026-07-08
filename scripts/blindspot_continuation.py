"""Continuation classifier for harvested blind-spot seeds — "who ended up being right".

A severe seed is a position the net valued as fine (net_q > 0.2) while the in-loop
Stockfish label said lost (sf_q < -0.5). The in-loop label is a ~700k-node eval,
which is exactly where horizon-effect misevaluations live — so some fraction are
false positives (SF, not the net, was wrong). Rather than re-search statically, we
use data we already have: the SAME game was played out, so we can look forward and
see who was right.

We join each seed to its game in the live replay buffer via ``game_id`` (the seed
comment carries it; ``_stable_game_id`` is byte-identical between the harvest hook
and ``_build_replay_samples``), locate the seed ply by its (net_q, sf_q) fingerprint,
and read:

  * the REALIZED game outcome (``wdl_target`` at the seed ply, side-to-move = net POV),
  * the forward ``sf_q`` trajectory (did SF keep saying lost, or recover).

Buckets:
  CONFIRMED_LOST  net went on to lose            -> real, game-costing blind spot
  RESCUED_LOST    net didn't lose BUT sf_q stayed lost downstream
                                                 -> genuinely lost; handicapped
                                                    opponent let the net escape
                                                    (still a real blind spot)
  REFUTED         net didn't lose AND sf_q recovered under normal play
                                                 -> likely FALSE POSITIVE (SF horizon
                                                    miseval) or an exploitable SF weakness
  UNLOCATED       game found, no (net_q, sf_q) row matched (rounding/eviction of the ply)
  UNRECOVERABLE   game_id not in the replay window (evicted)

CONFIRMED_LOST + RESCUED_LOST = "SF was right" (real blind spot). REFUTED ~= the
false-positive rate the auto-feed gate must filter. The REFUTED (+ optionally all
not-lost) seeds are written out for a deep-SF ground-truth recheck (Tier B).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from chess_anti_engine.replay.shard import load_shard_arrays
from scripts.scan_blindspots import default_replay_dir

_COMMENT_RE = re.compile(
    r"nq=(-?[0-9.]+)\s+sq=(-?[0-9.]+)\s+sev=(\d+)\s+game=(\d+)(?:\s+ply=(-?\d+))?")


@dataclass
class Seed:
    line: str        # full seed line (without comment)
    nq: float
    sq: float
    game_id: int
    src: str         # source file
    ply: int | None = None   # exact game ply if the harvester stamped ply= (newer seeds)


def parse_seeds(paths: list[str]) -> list[Seed]:
    seeds: list[Seed] = []
    seen: set[tuple[int, float, float]] = set()
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.rstrip("\n")
                if not raw.strip() or raw.lstrip().startswith("#") or "#" not in raw:
                    continue
                body, _, comment = raw.partition("#")
                m = _COMMENT_RE.search(comment)
                if m is None:
                    continue
                nq, sq, gid = float(m[1]), float(m[2]), int(m[4])
                ply = int(m[5]) if m[5] is not None else None
                key = (gid, round(nq, 2), round(sq, 2))
                if key in seen:
                    continue
                seen.add(key)
                seeds.append(Seed(line=body.strip(), nq=nq, sq=sq, game_id=gid, src=p, ply=ply))
    return seeds


def _q(wdl: np.ndarray) -> np.ndarray:
    w = np.asarray(wdl, dtype=np.float32)
    return w[..., 0] - w[..., 2]


@dataclass
class GameRow:
    ply: int
    net_q: float
    sf_q: float
    outcome: int   # wdl_target 0/1/2 from side-to-move POV
    selfplay: bool


def load_game_rows(replay_dir: str, wanted: set[int]) -> dict[int, list[GameRow]]:
    """Pull per-ply rows for only the wanted game_ids across all shards."""
    games: dict[int, list[GameRow]] = defaultdict(list)
    shards = sorted(glob.glob(f"{replay_dir}/*.zarr"))
    for sh in shards:
        try:
            a, _ = load_shard_arrays(sh, lazy=True)
        except Exception:
            continue
        if "game_id" not in a or "has_game_id" not in a:
            continue
        gid = np.asarray(a["game_id"])
        has_g = np.asarray(a["has_game_id"]).astype(bool)
        # rows whose game is one we're looking for
        mask = has_g & np.isin(gid, list(wanted))
        if not mask.any():
            continue
        idx = np.nonzero(mask)[0]
        ply = np.asarray(a["ply_index"])[idx]
        sw = _q(np.asarray(a["search_wdl"])[idx])
        sf = _q(np.asarray(a["sf_wdl"])[idx])
        out = np.asarray(a["wdl_target"])[idx].astype(int)
        sp = (np.asarray(a["is_selfplay"])[idx].astype(bool)
              if "is_selfplay" in a else np.zeros(len(idx), dtype=bool))
        has_sw = (np.asarray(a["has_search_wdl"])[idx].astype(bool)
                  if "has_search_wdl" in a else np.ones(len(idx), dtype=bool))
        has_sf = (np.asarray(a["has_sf_wdl"])[idx].astype(bool)
                  if "has_sf_wdl" in a else np.ones(len(idx), dtype=bool))
        for j in range(len(idx)):
            games[int(gid[idx[j]])].append(GameRow(
                ply=int(ply[j]),
                net_q=float(sw[j]) if has_sw[j] else float("nan"),
                sf_q=float(sf[j]) if has_sf[j] else float("nan"),
                outcome=int(out[j]), selfplay=bool(sp[j]),
            ))
    for g in games.values():
        g.sort(key=lambda r: r.ply)
    return games


def _result_to_outcome(result: str, stm_white: bool) -> int:
    """Game result string -> wdl_target (0=W/1=D/2=L) from the side-to-move POV
    (matches the replay-shard wdl_target convention)."""
    if result in ("1-0", "0-1"):
        return 0 if ((result == "1-0") == stm_white) else 2
    return 1  # draw / unknown


def load_game_rows_from_jsonl(
    paths: list[str], wanted: set[int] | None = None,
) -> dict[int, list[GameRow]]:
    """Per-ply rows from the harvester's saved ``<out>.games.pPID.jsonl`` — the
    self-contained record (root_fen + move list + per-ply net_q/sf_q + result), so
    continuation analysis survives the replay window aging out (Codex #125). The
    side-to-move at ply p is root_fen's side flipped by parity; outcome is derived
    from the game result on that POV. (Codex #125 worried an odd-length stripped opening
    flips this — it does NOT: ply IS absolute (root_fen is an opening position with an
    offset — verified maxply 77-91 > 62-79 moves on C-ply games), but root_fen's turn
    field self-encodes the offset parity, so ``root_white == (ply even)`` is the correct
    RELATIVE parity for any offset. VERIFIED 0/2324 outcome mismatches vs the shard
    wdl_target across 117 shared games incl 27 C-ply.) Preferred over the shard join."""
    games: dict[int, list[GameRow]] = defaultdict(list)
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    gid = int(r["game_id"])
                    root_white = str(r["root_fen"]).split()[1] == "w"
                except (json.JSONDecodeError, KeyError, IndexError, ValueError):
                    continue
                if wanted is not None and gid not in wanted:
                    continue
                result, selfplay = str(r.get("result", "")), bool(r.get("selfplay", False))
                for pl in r.get("plies", []):
                    ply = int(pl["ply"])
                    nq, sq = pl.get("nq"), pl.get("sq")
                    games[gid].append(GameRow(
                        ply=ply,
                        net_q=float("nan") if nq is None else float(nq),
                        sf_q=float("nan") if sq is None else float(sq),
                        outcome=_result_to_outcome(result, root_white == (ply % 2 == 0)),
                        selfplay=selfplay,
                    ))
    for g in games.values():
        g.sort(key=lambda r: r.ply)
    return games


@dataclass
class Verdict:
    bucket: str
    outcome: int | None          # net-POV wdl_target at seed ply (terminal, most confounded)
    seed_ply: int | None
    # seed-side sf_q at each look-ahead horizon (plies); the per-horizon value is
    # None where the game ended before it, and the whole dict is empty {} when the
    # seed could not be located / recovered at all.
    profile: dict[int, float | None]
    reached_terminal: bool       # game ended within the deepest horizon


def _seed_side_forward(rows: list[GameRow], t0: int) -> list[tuple[int, float]]:
    """(ply-ahead, seed-side sf_q) for every recorded full ply after the seed.
    In selfplay both colors are recorded so the eval POV flips on odd ply gaps;
    curriculum rows are all the net's color (even gaps) so the flip is a no-op."""
    return [(r.ply - t0, r.sf_q * (-1.0 if (r.ply - t0) % 2 else 1.0))
            for r in rows if r.ply > t0 and not np.isnan(r.sf_q)]


def classify(seed: Seed, rows: list[GameRow] | None, *,
             recover_to: float, still_lost: float, confirm_h: int,
             horizons: list[int], tol: float) -> Verdict:
    if not rows:
        return Verdict("UNRECOVERABLE", None, None, {}, False)
    # Locate the seed ply: exact by ply= if the harvester stamped it (newer
    # seeds), else by the (net_q, sf_q) fingerprint (older seeds pre-ply-stamp).
    best: GameRow | None = None
    if seed.ply is not None:
        best = next((r for r in rows if r.ply == seed.ply), None)
    if best is None:
        best_d = 1e9
        for r in rows:
            if np.isnan(r.net_q) or np.isnan(r.sf_q):
                continue
            d = abs(r.net_q - seed.nq) + abs(r.sf_q - seed.sq)
            if d < best_d:
                best, best_d = r, d
        if best is None or best_d > 2 * tol:
            return Verdict("UNLOCATED", None, None, {}, False)
    t0 = best.ply
    fwd = _seed_side_forward(rows, t0)  # sorted by ply gap (rows are ply-sorted)
    max_gap = fwd[-1][0] if fwd else 0
    # profile[h] = seed-side sf_q at the deepest recorded ply <= t0+h (the eval
    # "as of h plies in"); falls back to the last value when the game ended early.
    profile: dict[int, float | None] = {}
    for h in horizons:
        upto = [q for g, q in fwd if g <= h]
        profile[h] = upto[-1] if upto else None
    reached_terminal = 0 < max_gap <= max(horizons)

    if not fwd:  # no SF-labeled forward rows
        # Distinguish TRUE near-terminal (no recorded rows after the seed at all →
        # the game ended right after, so the terminal result is a SHALLOW, trustable
        # confirmation) from MISSING look-ahead (rows exist after the seed but carry
        # no SF label — e.g. value-only fast plies when record_fast_ply_value is on).
        # In the latter the game continued, so trusting the terminal reintroduces the
        # deep-tail confound the classifier avoids → INCONCLUSIVE (Codex #125).
        if any(r.ply > t0 for r in rows):
            return Verdict("INCONCLUSIVE", best.outcome, t0, profile, reached_terminal)
        term = "CONFIRMED_LOST" if best.outcome == 2 else "INCONCLUSIVE"
        return Verdict(term, best.outcome, t0, profile, reached_terminal)
    # Depth-graded verdict at the confirm horizon (default 8 plies), NOT the
    # terminal result: did SF still read it lost, or had it recovered.
    upto_c = [q for g, q in fwd if g <= confirm_h]
    recovered = any(q >= recover_to for q in upto_c)
    if recovered:
        # A labeled row within the horizon rose to recovery — valid regardless of
        # any missing later labels.
        bucket = "REFUTED"
    else:
        # CONFIRM requires the SF labels to actually REACH the confirm horizon.
        # "did SF STILL read it lost" = the eval AS OF confirm_h (the deepest labeled
        # ply <= confirm_h), NOT min() — [-0.7,-0.3] must not confirm. AND if the last
        # label is before confirm_h while the game continued past it (value-only rows
        # with no SF label), the horizon eval is UNKNOWN → inconclusive, don't confirm
        # off a stale shallow eval (Codex #125).
        last_gap = fwd[-1][0]
        covered = last_gap >= confirm_h or max_gap <= last_gap
        if bool(upto_c) and upto_c[-1] <= still_lost and covered:
            bucket = "CONFIRMED_LOST"
        else:
            bucket = "INCONCLUSIVE"
    return Verdict(bucket, best.outcome, t0, profile, reached_terminal)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--severe-glob", default="data/harvest/blindspot_live.severe.p*.txt")
    ap.add_argument("--replay-dir", default="")
    ap.add_argument("--recover-to", type=float, default=-0.2,
                    help="seed-side sf_q rising to/above this downstream = SF recovered")
    ap.add_argument("--still-lost", type=float, default=-0.5,
                    help="seed-side sf_q staying at/below this downstream = still lost")
    ap.add_argument("--horizons", default="2,4,8,16,32",
                    help="look-ahead depths in plies for the depth profile")
    ap.add_argument("--confirm-h", type=int, default=8,
                    help="horizon (plies) the CONFIRMED/REFUTED bucket verdict uses")
    ap.add_argument("--tol", type=float, default=0.03, help="(net_q,sq) match tolerance")
    ap.add_argument("--dump-refuted", default="",
                    help="write not-confirmed seeds here for a deep-SF recheck")
    ap.add_argument("--dump-all", default="",
                    help="write EVERY located seed with its bucket (deep-SF calibration input)")
    ap.add_argument("--games-glob", default="data/harvest/blindspot_live.games.p*.jsonl",
                    help="saved full-game records (self-contained; preferred over the shard join)")
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    paths = sorted(glob.glob(args.severe_glob))
    seeds = parse_seeds(paths)
    print(f"[continuation] {len(seeds)} unique severe seeds from {len(paths)} file(s)")
    wanted = {s.game_id for s in seeds}
    # Prefer the harvester's self-contained saved games (survive window aging);
    # fall back to the replay-shard join only for games not saved there (Codex #125).
    games = load_game_rows_from_jsonl(sorted(glob.glob(args.games_glob)), wanted)
    from_saved = len(wanted & set(games))
    missing = wanted - set(games)
    if missing:
        replay_dir = args.replay_dir or default_replay_dir()
        print(f"[continuation] {from_saved} game(s) from saved jsonl; "
              f"{len(missing)} from replay shards ({replay_dir})")
        games.update(load_game_rows(replay_dir, missing))
    print(f"[continuation] {len(wanted & set(games))}/{len(wanted)} game_ids recovered "
          f"({from_saved} from saved games.jsonl)")

    buckets: dict[str, list[tuple[Seed, Verdict]]] = defaultdict(list)
    verdicts: list[tuple[Seed, Verdict]] = []
    for s in seeds:
        v = classify(s, games.get(s.game_id), recover_to=args.recover_to,
                     still_lost=args.still_lost, confirm_h=args.confirm_h,
                     horizons=horizons, tol=args.tol)
        buckets[v.bucket].append((s, v))
        verdicts.append((s, v))

    order = ["CONFIRMED_LOST", "REFUTED", "INCONCLUSIVE", "UNLOCATED", "UNRECOVERABLE"]
    n = len(seeds)
    print(f"\n=== continuation verdict (confirm horizon = {args.confirm_h} plies) ===")
    for b in order:
        k = len(buckets.get(b, []))
        print(f"  {b:15s} {k:4d}  ({100*k/n:5.1f}%)")
    real = len(buckets.get("CONFIRMED_LOST", []))
    ref = len(buckets.get("REFUTED", []))
    decided = real + ref
    if decided:
        print(f"\n  of seeds the continuation DECIDED ({decided}):")
        print(f"    real blind spot (SF right) : {real}/{decided} = {100*real/decided:4.1f}%")
        print(f"    REFUTED (SF wrong / FP)    : {ref}/{decided} = {100*ref/decided:4.1f}%")

    # Depth profile: at each look-ahead horizon, how the seed-side sf_q reads,
    # among located seeds that still have game left at that depth. Shows how the
    # verdict firms up (or flips) with depth instead of betting on the terminal.
    print("\n=== depth profile (seed-side sf_q at each look-ahead) ===")
    print(f"  {'horizon':>8}  {'n':>4}  {'still-lost':>10}  {'recovered':>9}  {'middling':>8}")
    for h in horizons:
        vals = [q for _, v in verdicts
                if (q := v.profile.get(h)) is not None]
        if not vals:
            continue
        lost = sum(1 for q in vals if q <= args.still_lost)
        rec = sum(1 for q in vals if q >= args.recover_to)
        mid = len(vals) - lost - rec
        print(f"  {h:>6}pl  {len(vals):>4}  {lost:>4} ({100*lost/len(vals):4.0f}%)  "
              f"{rec:>3} ({100*rec/len(vals):4.0f}%)  {mid:>3} ({100*mid/len(vals):3.0f}%)")

    # Examples per bucket
    for b in ("CONFIRMED_LOST", "REFUTED", "INCONCLUSIVE"):
        ex = buckets.get(b, [])[:4]
        if not ex:
            continue
        print(f"\n--- {b} (examples) ---")
        for s, v in ex:
            prof = " ".join(
                f"+{h}:{'--' if v.profile.get(h) is None else f'{v.profile[h]:+.2f}'}"
                for h in horizons)
            print(f"  game={s.game_id} nq={s.nq:+.2f} sq={s.sq:+.2f} "
                  f"outcome={v.outcome} ply={v.seed_ply}  [{prof}]")

    if args.dump_refuted:
        notconf = (buckets.get("REFUTED", []) + buckets.get("INCONCLUSIVE", [])
                   + buckets.get("UNLOCATED", []))
        with open(args.dump_refuted, "w", encoding="utf-8") as fh:
            for s, v in notconf:
                # SAME tag order as --dump-all so blindspot_deepsf_calibrate.parse_all
                # (bucket..sq..outcome..game) can consume it as a recheck input (Codex #125).
                fh.write(f"{s.line}  # bucket={v.bucket} nq={s.nq:.2f} sq={s.sq:.2f} "
                         f"outcome={v.outcome} game={s.game_id}\n")
        print(f"\n[continuation] wrote {len(notconf)} not-confirmed seeds -> {args.dump_refuted}")

    if args.dump_all:
        with open(args.dump_all, "w", encoding="utf-8") as fh:
            for s, v in verdicts:
                if v.seed_ply is None:
                    continue  # unlocated / unrecoverable: nothing to recheck
                fh.write(f"{s.line}  # bucket={v.bucket} nq={s.nq:.2f} sq={s.sq:.2f} "
                         f"outcome={v.outcome} game={s.game_id}\n")
        print(f"[continuation] wrote all located seeds -> {args.dump_all}")


if __name__ == "__main__":
    main()
