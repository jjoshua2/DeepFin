"""Deep-SF admission gate for harvested blind-spot seeds (batch / manual form).

The raw severe band is ~70% false positives (deep SF agrees with the net, not the
~700k in-loop label), so it must be gated before feeding. This dedups the banked
severe seeds, runs a static unhandicapped deep Stockfish (4M nodes — converged;
4M=8M=16M in calibration — with the 6-man syzygy TB on), and keeps only the
seeds deep SF still calls LOST. Those are the real, deep-confirmed value blind
spots safe to seed selfplay from.

Blame backup (2026-07-09): a confirmed-LOST seed marks the SYMPTOM — by the
time net-Q and label-Q diverge hard, the losing choice was often plies earlier.
For each kept seed we walk the stored history backward two plies at a time
(same side to move as the seed, so the deep verdict's POV convention holds) and
deep-eval each earlier position: the first one deep SF still calls FINE is the
DECISION POINT — the move played from it is the blunder (an opponent reply
cannot make a holdable position lost, so attribution is sound up to eval
noise). The emitted seed starts the game THERE, turning a "this position is
lost" recognition lesson into a "don't enter it" avoidance lesson. If no
stored position is FINE (blunder predates the ~8-ply history window), the
original seed is kept unchanged. Cost: ≤ history/2 extra deep searches per
KEPT seed only.

Outputs:
  --out-list  : vetted seed lines (opening_fen_list_path grammar; comment stripped
                by the loader) — the auto-feed candidate list.
  --out-jsonl : per-seed audit (fen, captured sq/nq, deep_sq, verdict, blame_*).

This is the same gate an automated loop would run; run it on the backlog now,
review, then feed (training-affecting -> ledger + yardstick + one-change/window).
"""
from __future__ import annotations

import argparse
import glob
import json
import time

from chess_anti_engine.selfplay.opening import seed_board_from_line
from chess_anti_engine.stockfish.uci import StockfishUCI
from scripts.blindspot_continuation import parse_seeds
from scripts.blindspot_deepsf_calibrate import deep_verdict, ensure_parent, resolve_syzygy

_DEFAULT_SF = "/home/josh/projects/chess/e2e_server/publish/stockfish"


def _prefix_line(fen_part: str, moves: list[str], drop: int) -> str:
    """Seed line for the position ``drop`` plies before the terminal one."""
    kept = moves[:-drop]
    if not kept:
        return fen_part
    return f"{fen_part} | {' '.join(kept)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--severe-glob", default="data/harvest/blindspot_live.severe.p*.txt")
    ap.add_argument("--stockfish", default=_DEFAULT_SF)
    ap.add_argument("--nodes", type=int, default=4_000_000,
                    help="deep-SF budget (4M = converged per the scaling check)")
    ap.add_argument("--hash-mb", type=int, default=2048)
    ap.add_argument("--syzygy-path", default="data/syzygy_3-4-5-6")
    ap.add_argument("--deep-lost", type=float, default=-0.5)
    ap.add_argument("--deep-fine", type=float, default=-0.2)
    ap.add_argument("--blame-backup", type=int, default=8,
                    help="Max history plies to scan backward for the decision point "
                         "on kept seeds (0 = emit seeds at the divergence, old behavior)")
    ap.add_argument("--nice", type=int, default=15)
    ap.add_argument("--out-list", default="data/harvest/vetted_blindspots.txt")
    ap.add_argument("--out-jsonl", default="scratchpad/harvest_fp/gate_audit.jsonl")
    args = ap.parse_args()
    ensure_parent(args.out_list)
    ensure_parent(args.out_jsonl)
    syzygy = resolve_syzygy(args.syzygy_path)  # fail-fast before the long search

    seeds = parse_seeds(sorted(glob.glob(args.severe_glob)))
    seen: set[str] = set()
    uniq = []
    for s in seeds:
        if s.line in seen:
            continue
        seen.add(s.line)
        uniq.append(s)
    print(f"[gate] {len(uniq)} unique severe seeds; deep SF nodes={args.nodes:,} "
          f"syzygy={args.syzygy_path} blame_backup={args.blame_backup}")

    def make_engine() -> StockfishUCI:
        return StockfishUCI(args.stockfish, nodes=int(args.nodes), hash_mb=int(args.hash_mb),
                            nice=int(args.nice), syzygy_path=syzygy)

    eng = make_engine()

    def search_clean(fen: str) -> tuple[float | None, str | None]:
        """Cold-TT deep eval; returns (dsq, error). Recreates the engine on error
        — a timed-out/raised search leaves SF still calculating, and its stray
        bestmove/info would desync the NEXT search (Codex #125)."""
        nonlocal eng
        try:
            eng.new_game()  # cold TT per verdict — must not warm-start (Codex #125)
            res = eng.search(fen, nodes=int(args.nodes))
            if res.wdl is None:
                return None, None
            return round(float(res.wdl[0] - res.wdl[2]), 3), None
        except Exception as e:
            try:
                eng.close()
            except Exception:
                pass
            eng = make_engine()
            return None, f"ERR:{type(e).__name__}"

    audit = []
    vetted = []
    emitted_placements: set[str] = set()
    t0 = time.time()
    try:
        for i, s in enumerate(uniq):
            try:
                fen = seed_board_from_line(s.line).fen()
            except ValueError as e:
                audit.append({"line": s.line, "fen": "", "nq": s.nq, "sq": s.sq,
                              "deep_sq": None, "deep": f"ERR:{type(e).__name__}",
                              "game": s.game_id})
                continue
            dsq, err = search_clean(fen)
            if err is not None:
                verdict = err
            elif dsq is None:
                verdict = "UNKNOWN"
            else:
                verdict = deep_verdict(dsq, args.deep_lost, args.deep_fine)

            rec = {"line": s.line, "fen": fen, "nq": s.nq, "sq": s.sq,
                   "deep_sq": dsq, "deep": verdict, "game": s.game_id}

            if verdict == "LOST":
                # Blame backup: walk the stored history toward the decision point.
                emit_line = s.line
                fen_part, _, moves_part = s.line.partition("|")
                fen_part = fen_part.strip()
                moves = moves_part.split() if moves_part else []
                if args.blame_backup > 0 and moves:
                    for k in range(2, min(len(moves), int(args.blame_backup)) + 1, 2):
                        cand_line = _prefix_line(fen_part, moves, k)
                        try:
                            cand_fen = seed_board_from_line(cand_line).fen()
                        except ValueError:
                            break  # malformed prefix — keep what we have
                        cand_dsq, cand_err = search_clean(cand_fen)
                        if cand_err is not None or cand_dsq is None:
                            break  # engine trouble — keep the deepest confirmed point
                        cand_verdict = deep_verdict(cand_dsq, args.deep_lost, args.deep_fine)
                        if cand_verdict == "FINE":
                            # First holdable ancestor: the move played FROM here
                            # is the blunder; seed the game at the decision point.
                            emit_line = cand_line
                            rec["blame_k"] = k
                            rec["blame_dsq"] = cand_dsq
                            rec["blame_move"] = moves[len(moves) - k]
                            # Trimmed plies ride the list comment (dropped=)
                            # so resolution/retirement reconstruct + score the
                            # LOST terminal — the decision point itself reads
                            # ~holdable to a correct net, i.e. BLIND forever.
                            rec["dropped"] = ",".join(moves[len(moves) - k:])
                            break
                        # LOST or MID: still inside the lost line — keep walking.
                rec["emitted_line"] = emit_line
                # Dedup on the EMITTED terminal placement: two severe rows can
                # back up to the same decision point (or share a terminal via
                # different histories); the dole plays every list line once
                # per iter, so duplicates would overweight one position.
                emit_key = seed_board_from_line(emit_line).fen().split()[0]
                if emit_key in emitted_placements:
                    rec["deduped_against_emitted"] = True
                    audit.append(rec)
                    continue
                emitted_placements.add(emit_key)
                vetted.append((s, dsq, emit_line, rec))
            audit.append(rec)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(uniq)}  kept={len(vetted)}  ({time.time()-t0:.0f}s)")
    finally:
        eng.close()

    with open(args.out_jsonl, "w", encoding="utf-8") as fh:
        for a in audit:
            fh.write(json.dumps(a) + "\n")
    with open(args.out_list, "w", encoding="utf-8") as fh:
        fh.write("# deep-SF-vetted blind spots (deep-LOST @ 4M nodes + 6-man TB; "
                 "blame-backup seeds start at the decision point)\n")
        for s, dsq, emit_line, rec in vetted:
            blame = ""
            if "blame_k" in rec:
                blame = (f" blame_k={rec['blame_k']} blame_dsq={rec['blame_dsq']}"
                         f" blunder={rec['blame_move']} dropped={rec['dropped']}")
            fh.write(f"{emit_line}  # deep_sq={dsq} sq={s.sq:.2f} nq={s.nq:.2f}"
                     f" game={s.game_id}{blame}\n")

    graded = [a for a in audit if a["deep"] in ("LOST", "MID", "FINE")]
    lost = sum(1 for a in graded if a["deep"] == "LOST")
    fine = sum(1 for a in graded if a["deep"] == "FINE")
    mid = sum(1 for a in graded if a["deep"] == "MID")
    backed = sum(1 for _, _, _, rec in vetted if "blame_k" in rec)
    print(f"\n=== gate result ({len(graded)} graded) ===")
    if graded:
        print(f"  deep-LOST (VETTED, kept): {lost}  ({100*lost/len(graded):.0f}%)")
        print(f"  deep-FINE (false pos)   : {fine}  ({100*fine/len(graded):.0f}%)")
        print(f"  deep-MID                : {mid}")
    else:
        print("  (none graded)")
    print(f"  blame-backed to decision point: {backed}/{len(vetted)}")
    print(f"  vetted list -> {args.out_list} ({len(vetted)} seeds)")
    print(f"  audit       -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
