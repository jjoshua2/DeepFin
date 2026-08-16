#!/usr/bin/env python3
"""Real production-MultiPV-6 vs wide-Stockfish tail calibration (issue #425).

`scripts/tail_censor_screen.py` answers the tail question by TRUNCATING an old
MultiPV-40 label. That is the right first screen and it is not the production
question, for one reason that cannot be argued away: **a node budget is spent
across PVs**, so MultiPV 6 at 150k nodes and MultiPV 40 at 150k nodes do not
give rank 6 the same search. The truncation screen's `r_6` is a rank-6 regret
from a 40-wide search; production's `r_6` comes from a 6-wide one. Whether the
fitted alpha survives that substitution is an EMPIRICAL question, and this panel
is the instrument for it.

⚑⚑ "WHAT IS THE TAIL WORTH" IS TWO QUESTIONS, AND ONE WIDE ARM CANNOT ANSWER BOTH.

  (A) SELF-CONSISTENCY — what would the label's OWN search have said about the
      moves it did not surface? The imputation is a claim about that search, so
      the comparison must hold the node budget fixed and vary only the width.
  (B) TRUTH — what do those moves actually deserve? That needs a stronger
      engine, which also re-scores the SURFACED moves, i.e. it indicts the whole
      label rather than just its tail.

Answering (B) alone and calling it "the tail is mispriced" would be the error
this panel exists to avoid: a deeper arm finds bigger refutations everywhere, so
its tail regrets are larger for reasons that have nothing to do with censoring.
So there are THREE arms on the same positions:

  PROD    — the live labeller's SETTINGS, read off the live yaml rather than
            assumed. ⚑ NOT an exact replica: this arm searches with `fresh=True`
            (cold TT) while production labels through `StockfishPool.submit`,
            which defaults `fresh=False` — a WARM, pooled engine. Cold is the
            right choice HERE (a warm TT would let one arm steer the next, which
            is the agreement being measured) but it is a real difference from
            production and is named rather than papered over. Settings: MultiPV 6, 150,000 nodes (the realized median of
            `sf_label_meta[:,0]`, pinned at `sf_label_nodes_floor`, NOT the 200k
            cap), Threads 1, Hash 17 MB, and `stockfish_syzygy_path` — which is
            the 3-4-5 directory ALONE, not the colon-separated production pair
            that `syzygy_path` names for the playing ENGINES.
  MATCHED — identical to PROD in every respect except width: same nodes, same
            hash, same tablebases, MultiPV at full legal width. This is the
            arm that answers (A), and it is the primary readout.
  DEEP    — MultiPV at full width, a far larger node budget, a real TT and the
            full Syzygy pair. This answers (B), and is deliberately NOT a
            production replica.

⚑ THE PANEL MEASURES ITS OWN CONFOUNDS RATHER THAN ASSUMING THEM AWAY.
Two cross-arm ratios are reported on the moves each pair BOTH surfaced:

  PROD vs MATCHED — SHOULD be ~1.0. These differ only in width at an identical
                    budget, so a ratio away from 1 means widening the search
                    changed the surfaced scores too, and the (A) substitution is
                    not clean. This is a falsifier for the primary readout.
  MATCHED vs DEEP — the size of the depth effect, i.e. exactly how much of the
                    (B) answer is not about censoring at all.

Positions inside tablebase range are flagged and alpha is reported with and
without them, because PROD/MATCHED carry the 3-4-5 set and DEEP carries the pair.

The estimator itself is imported from `tail_censor_screen`, not reimplemented,
so the historical and production numbers are the same statistic on two
populations rather than two statistics.

Usage:

    PYTHONPATH=. python3 scripts/mpv6_tail_panel.py \\
        --replay-dir runs/pbt2_small/replay/<live-trial>/replay_shards \\
        --stockfish /home/josh/projects/chess/e2e_server/publish/stockfish \\
        --checkpoint data/tail_screen_20260814/checkpoint_000218 \\
        --positions 400 --deep-nodes 4000000 --workers 3 \\
        --json-out scratchpad/mpv6_tail_panel.json
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import chess
import numpy as np

from chess_anti_engine.eval.audit import decode_board_from_planes, phase_bucket, position_key
from chess_anti_engine.moves.encode import move_to_index_for_encoding
from chess_anti_engine.replay.shard import load_shard_arrays
from chess_anti_engine.stockfish.uci import StockfishUCI
from chess_anti_engine.stockfish.wdl import mate_to_effective_cp
from chess_anti_engine.utils.git_meta import git_sha
from scripts.diagnostic_replay_utils import record_skipped_shard, select_shards
from scripts.tail_censor_screen import N_BOOT, SF_OWN_REGRET_CAP_CP, Row, analyse

# Live-yaml values. Defaults, not assumptions — every one is overridable, and
# --prod-nodes defaults to the REALIZED median rather than the configured cap.
PROD_MULTIPV = 6
PROD_NODES = 150_000
PROD_HASH_MB = 17
PROD_SYZYGY = "/home/josh/projects/chess/data/syzygy_3-4-5"
DEEP_SYZYGY = ("/home/josh/projects/chess/data/syzygy_3-4-5"
               ":/home/josh/projects/chess/data/syzygy_6")
# python-chess caps out well below this; 64 is a safe ceiling on legal width.
MAX_MULTIPV = 64
# Syzygy tops out at 7 men; at or below this the arms' different TB sets can
# disagree for reasons that have nothing to do with censoring.
TB_PIECE_LIMIT = 7

GRID = ("r_k censor", "midpoint (live)", "alpha_value", "alpha_grad")


def sample_positions(
    shards: list[Path], scan: dict[str, Any], want: int, seed: int,
) -> list[dict[str, Any]]:
    """Draw rows from the LIVE era, keeping the stored planes.

    ⚑ The planes are kept rather than re-encoded from FEN. The audit set has no
    history planes, and re-encoding here would silently reintroduce that defect
    into the prior this panel weights by — the net would be scored on an input it
    never sees in training.
    """
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    # ⚑ PER-SHARD QUOTA, NOT FILL-FROM-THE-FRONT. A shard is one worker's upload
    # batch — a few dozen games over a few minutes — so draining the first shard
    # would make 500 "independent" positions a handful of correlated games, and
    # the row bootstrap would report an interval far tighter than the population
    # earns. Quota per shard, then backfill from whatever had spare rows.
    quota = max(1, -(-want // max(len(shards), 1)))
    for shard in shards:
        if len(out) >= want:
            break
        target = min(want, len(out) + quota)
        try:
            arrs, _ = load_shard_arrays(shard)
        except (OSError, ValueError, KeyError) as exc:
            record_skipped_shard(scan, shard, exc)
            continue
        x = np.asarray(arrs["x"])
        n_shard = int(x.shape[0])
        # ⚑⚑ production only builds sf_p0_regret on SELFPLAY slots (finalize.py:867),
        # so a curriculum row never carries the imputation this panel prices.
        has_sp = np.asarray(arrs.get("has_is_selfplay", np.zeros(n_shard))).astype(bool)
        is_sp = np.asarray(arrs.get("is_selfplay", np.zeros(n_shard))).astype(bool)
        selfplay = has_sp & is_sp
        scan["rows_selfplay"] = scan.get("rows_selfplay", 0) + int(selfplay.sum())
        legal_mask = np.asarray(arrs["legal_mask"]).astype(bool)
        hist = str(np.asarray(arrs["_input_history_encoding"]).reshape(-1)[0])
        pol_enc = str(np.asarray(arrs["_policy_encoding"]).reshape(-1)[0])
        scan["rows_scanned"] += int(x.shape[0])
        for i in rng.permutation(int(x.shape[0])):
            if len(out) >= target:
                break
            if not selfplay[int(i)]:
                scan["skipped_not_selfplay"] = scan.get("skipped_not_selfplay", 0) + 1
                continue
            board = decode_board_from_planes(x[int(i)], input_history_encoding=hist)
            if board is None or board.is_game_over():
                continue
            key = position_key(board)
            if key in seen:
                continue
            n_legal = board.legal_moves.count()
            if n_legal <= PROD_MULTIPV:
                scan["skipped_narrow"] += 1
                continue
            seen.add(key)
            pieces = int(chess.popcount(board.occupied))
            out.append({
                "fen": board.fen(), "board": board, "x": x[int(i)],
                "legal_mask": legal_mask[int(i)], "policy_encoding": pol_enc,
                "n_legal": n_legal, "phase": phase_bucket(round(float(x[int(i)][:12].sum()))),
                "in_tb": pieces <= TB_PIECE_LIMIT, "pieces": pieces,
                "shard": shard.name,
            })
    # ⚑ SECOND PASS = the actual backfill. The first pass caps each shard at its
    # quota, so a SHORT shard's deficit was permanently lost -- the docstring
    # claimed a backfill that did not exist, and the test named for it used one
    # fixture for every shard so it could never create a short shard.
    if len(out) < want:
        for shard in shards:
            if len(out) >= want:
                break
            try:
                arrs, _ = load_shard_arrays(shard)
            except (OSError, ValueError, KeyError):
                continue
            x = np.asarray(arrs["x"])
            n_shard = int(x.shape[0])
            has_sp = np.asarray(arrs.get("has_is_selfplay", np.zeros(n_shard))).astype(bool)
            is_sp = np.asarray(arrs.get("is_selfplay", np.zeros(n_shard))).astype(bool)
            selfplay = has_sp & is_sp
            legal_mask = np.asarray(arrs["legal_mask"]).astype(bool)
            hist = str(np.asarray(arrs["_input_history_encoding"]).reshape(-1)[0])
            pol_enc = str(np.asarray(arrs["_policy_encoding"]).reshape(-1)[0])
            for i in rng.permutation(n_shard):
                if len(out) >= want:
                    break
                if not selfplay[int(i)]:
                    continue
                board = decode_board_from_planes(x[int(i)], input_history_encoding=hist)
                if board is None or board.is_game_over():
                    continue
                key = position_key(board)
                if key in seen or board.legal_moves.count() <= PROD_MULTIPV:
                    continue
                seen.add(key)
                pieces = int(chess.popcount(board.occupied))
                out.append({
                    "fen": board.fen(), "board": board, "x": x[int(i)],
                    "legal_mask": legal_mask[int(i)], "policy_encoding": pol_enc,
                    "n_legal": board.legal_moves.count(),
                    "phase": phase_bucket(round(float(x[int(i)][:12].sum()))),
                    "in_tb": pieces <= TB_PIECE_LIMIT, "pieces": pieces,
                    "shard": shard.name,
                })
    scan["sampled_shards"] = len({o["shard"] for o in out})
    return out


def score_of(pv: Any) -> float | None:
    """Mirror `finalize._sf_move_score`: mate dominates cp, missing cp is no score."""
    if pv.mate is not None and int(pv.mate) != 0:
        return mate_to_effective_cp(int(pv.mate))
    return None if pv.cp is None else float(pv.cp)


def regrets_from(pvs: list[Any], board: chess.Board, pol_enc: str) -> list[tuple[int, float]]:
    """(policy index, raw score) for every scored PV, best first."""
    out: list[tuple[int, float]] = []
    for pv in pvs or []:
        sc = score_of(pv)
        if sc is None:
            continue
        try:
            mv = chess.Move.from_uci(pv.move_uci)
        except (ValueError, TypeError):
            continue
        # ⚑ move_to_index_for_encoding does NOT check legality — it happily encodes
        # a black move from a white-to-move position. So the legality check has to
        # be here: without it, a board/FEN desync would silently produce indices
        # that look fine and belong to a different position.
        if not board.is_legal(mv):
            continue
        try:
            idx = int(move_to_index_for_encoding(mv, board, policy_encoding=pol_enc))
        except (ValueError, KeyError):
            continue
        out.append((idx, sc))
    out.sort(key=lambda t: -t[1])
    return out


def to_regret(scored: list[tuple[int, float]]) -> dict[int, float]:
    best = scored[0][1]
    return {
        mi: min(max(best - sc, 0.0), SF_OWN_REGRET_CAP_CP) / SF_OWN_REGRET_CAP_CP
        for mi, sc in scored
    }


def run_panel(
    positions: list[dict[str, Any]], scan: dict[str, Any], *,
    stockfish: str, prod_nodes: int, deep_nodes: int, deep_hash_mb: int,
    workers: int, nice: int,
) -> list[dict[str, Any]]:
    """Search every position three times. One engine triple per worker thread."""
    local = threading.local()
    done = threading.Lock()
    results: list[dict[str, Any]] = []
    # DEEP can take tens of seconds a position; the default read timeout is sized
    # for the labeller's 150k, not for 4M nodes at full width.
    deep_timeout = max(600.0, deep_nodes / 20_000.0)

    def engines() -> tuple[StockfishUCI, StockfishUCI, StockfishUCI]:
        trio = getattr(local, "trio", None)
        if trio is None:
            prod = StockfishUCI(stockfish, nodes=prod_nodes, multipv=PROD_MULTIPV,
                                hash_mb=PROD_HASH_MB, syzygy_path=PROD_SYZYGY, nice=nice)
            # ⚑ MATCHED differs from PROD in MultiPV and nothing else — same nodes,
            # same hash, same tablebases. Any other difference would reintroduce
            # the confound this arm exists to remove.
            matched = StockfishUCI(stockfish, nodes=prod_nodes, multipv=MAX_MULTIPV,
                                   hash_mb=PROD_HASH_MB, syzygy_path=PROD_SYZYGY, nice=nice)
            deep = StockfishUCI(stockfish, nodes=deep_nodes, multipv=MAX_MULTIPV,
                                hash_mb=deep_hash_mb, syzygy_path=DEEP_SYZYGY, nice=nice,
                                read_timeout_s=deep_timeout)
            trio = (prod, matched, deep)
            local.trio = trio
        return trio

    def one(pos: dict[str, Any]) -> None:
        prod_eng, matched_eng, deep_eng = engines()
        board = pos["board"]
        enc = pos["policy_encoding"]
        # `fresh` on ALL THREE: a warm TT would let one arm steer the next, which
        # is precisely the agreement this panel is trying to measure.
        p_res = prod_eng.search(pos["fen"], nodes=prod_nodes, fresh=True)
        m_res = matched_eng.search(pos["fen"], nodes=prod_nodes, fresh=True)
        d_res = deep_eng.search(pos["fen"], nodes=deep_nodes, fresh=True)
        prod_scored = regrets_from(p_res.pvs or [], board, enc)
        matched_scored = regrets_from(m_res.pvs or [], board, enc)
        deep_scored = regrets_from(d_res.pvs or [], board, enc)
        with done:
            scan["searched"] += 1
            if len(prod_scored) < 2 or len(matched_scored) <= len(prod_scored):
                scan["skipped_no_tail"] += 1
                return
            results.append({
                **{k: v for k, v in pos.items() if k != "board"},
                "prod_scored": prod_scored, "matched_scored": matched_scored,
                "deep_scored": deep_scored,
                "prod_width": len(prod_scored), "matched_width": len(matched_scored),
                "deep_width": len(deep_scored),
                "prod_depth": int(p_res.depth or 0),
                "matched_depth": int(m_res.depth or 0),
                "deep_depth": int(d_res.depth or 0),
            })

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, positions))
    return results


def attach_priors(results: list[dict[str, Any]], checkpoint: str, batch: int) -> str:
    import torch

    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.10)
    model = load_model_from_checkpoint(checkpoint, device=device)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(results), batch):
            chunk = results[start:start + batch]
            xb = torch.from_numpy(np.stack([c["x"] for c in chunk])).float().to(device)
            out = model(xb)
            logits = out["policy"] if "policy" in out else out["policy_own"]
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            for row, pr in zip(chunk, probs):
                legal = np.flatnonzero(row["legal_mask"])
                sub = pr[legal]
                total = float(sub.sum())
                row["legal_idx"] = legal
                row["prior"] = sub / total if total > 0 else None
    return device


def rebase_offset(prod: list[tuple[int, float]], truth: list[tuple[int, float]]) -> float | None:
    """Per-row level offset between two arms, on the moves BOTH scored.

    ⚑⚑ THIS IS THE CONFOUND `cross_arm_check` STRUCTURALLY CANNOT SEE. `to_regret`
    measures each arm against ITS OWN best, so a uniform level shift between the
    arms cancels out of every regret difference — the ratio reads exactly 1.000
    while the shift lands entirely on the substituted tail, which is the one
    quantity being measured. Measured impact: a 20 cp offset moves alpha from
    0.1986 to 0.245 (+23%) with the falsifier passing at 1.000.

    So do not merely check the levels agree — REMOVE the dependence. Scores are
    mapped into the PROD arm's frame before any regret is taken.
    """
    ps = dict(prod)
    common = [(ps[m], sc) for m, sc in truth if m in ps]
    if len(common) < 2:
        return None
    return float(np.mean([a - b for a, b in common]))


def build_rows(
    results: list[dict[str, Any]], truth_arm: str, *, substitute: bool, rebase: bool = True,
) -> list[Row]:
    """Price the tail with `truth_arm` as the withheld truth.

    ``substitute=True`` keeps the PROD arm's own regrets for the moves production
    actually surfaced, so only the CENSORING is varied. ``substitute=False`` takes
    every regret from `truth_arm`, which also re-scores the surfaced moves and
    therefore answers a different question — see the module docstring.

    ``rebase`` maps the truth arm's raw scores into the PROD arm's level before
    taking regrets, so the substituted tail carries no inter-arm offset.
    """
    rows: list[Row] = []
    for r in results:
        prior = r.get("prior")
        if prior is None:
            continue
        legal = r["legal_idx"]
        truth_scored = r[truth_arm]
        if substitute and rebase:
            delta = rebase_offset(r["prod_scored"], truth_scored)
            if delta is None:
                continue
            truth_scored = [(m, sc + delta) for m, sc in truth_scored]
            r["rebase_delta"] = delta
        prod_reg = to_regret(r["prod_scored"])
        truth_reg = to_regret(truth_scored)
        surfaced = [mi for mi, _ in r["prod_scored"]]
        hidden = [int(m) for m in legal if int(m) not in set(surfaced) and int(m) in truth_reg]
        if not hidden:
            continue
        if substitute:
            # ⚑ In PROD's frame: regret is measured off PROD's own best, and the
            # hidden moves' scores have been rebased into that same frame above.
            best = r["prod_scored"][0][1]
            regret = {m: prod_reg[m] for m in surfaced}
            for m in hidden:
                sc = dict(truth_scored)[m]
                regret[m] = min(max(best - sc, 0.0), SF_OWN_REGRET_CAP_CP) / SF_OWN_REGRET_CAP_CP
            r_k = max(prod_reg[m] for m in surfaced)
        else:
            if any(m not in truth_reg for m in surfaced):
                continue
            regret = dict(truth_reg)
            r_k = max(truth_reg[m] for m in surfaced)
        row = Row(legal=legal, regret=regret, surfaced=surfaced, hidden=hidden, r_k=r_k)
        row.prior = prior
        rows.append(row)
    return rows


def cross_arm_check(
    results: list[dict[str, Any]], arm_a: str, arm_b: str,
) -> dict[str, float]:
    """⚑ The confound guard: do two arms agree where they BOTH looked?

    PROD vs MATCHED should read ~1.0 — same budget, only the width differs — and
    a departure falsifies the clean-substitution premise. MATCHED vs DEEP is the
    size of the depth effect, which is NOT a censoring effect.
    """
    a_sum = b_sum = 0.0
    n = agree_best = rows = 0
    for r in results:
        reg_a = to_regret(r[arm_a])
        reg_b = to_regret(r[arm_b])
        both = [m for m in reg_a if m in reg_b]
        if len(both) < 2:
            continue
        a_sum += sum(reg_a[m] for m in both)
        b_sum += sum(reg_b[m] for m in both)
        n += len(both)
        rows += 1
        agree_best += int(r[arm_a][0][0] == r[arm_b][0][0])
    return {
        "n_moves": n, "n_rows": rows,
        "a_mean_cp": SF_OWN_REGRET_CAP_CP * a_sum / max(n, 1),
        "b_mean_cp": SF_OWN_REGRET_CAP_CP * b_sum / max(n, 1),
        "ratio": b_sum / a_sum if a_sum > 0 else float("nan"),
        "bestmove_agreement": agree_best / max(rows, 1),
    }


def report(label: str, res: dict[str, Any]) -> None:
    print(f"\n{label}  (n={res['n_rows']})")
    print(f"    TRUE (withheld wide-SF)   {res['true_cp']:6.1f} cp")
    print(f"    r_6 censor    (alpha=0)   {res['r_k_cp']:6.1f} cp"
          f"   bias {res['r_k_cp'] - res['true_cp']:+7.1f}")
    print(f"    midpoint LIVE (alpha=1)   {res['midpoint_cp']:6.1f} cp"
          f"   bias {res['midpoint_cp'] - res['true_cp']:+7.1f}"
          f"   = {res['midpoint_cp'] / max(res['true_cp'], 1e-9):.2f}x")
    vlo, vhi = res["alpha_value_ci"]
    glo, ghi = res["alpha_grad_ci"]
    print(f"    alpha_value {res['alpha_value']:.4f} [{vlo:.4f}, {vhi:.4f}]"
          f"    alpha_grad {res['alpha_grad_raw']:.4f} [{glo:.4f}, {ghi:.4f}]")
    # The historical MultiPV-40 truncation screen fitted 0.1759 [banked
    # 2026-08-14]. Whether that survives the real MultiPV-6 substitution is the
    # question this panel exists to answer, so say it rather than leave it to
    # the reader's memory.
    verdict = "CONSISTENT with" if vlo <= 0.1759 <= vhi else "INCONSISTENT with"
    print(f"      -> {verdict} the historical truncation screen's alpha = 0.1759"
          f"  ({N_BOOT} row bootstrap draws)")
    for name in GRID:
        v = res["variants"][name]
        print(f"      {name:17s} cos {v['cos']:.4f}  relL2 {v['rel_l2_pooled']:.4f}"
              f"  tail pressure {v['tail_pressure'] / max(res['true_pressure'], 1e-12):.2f}x")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--replay-dir", type=Path, required=True, action="append")
    ap.add_argument("--stockfish", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="the loss weights by the NET'S prior, so this is not optional here")
    ap.add_argument("--positions", type=int, default=400)
    ap.add_argument("--prod-nodes", type=int, default=PROD_NODES,
                    help="REALIZED production label nodes (median of sf_label_meta[:,0]), "
                         "not the configured cap")
    ap.add_argument("--deep-nodes", type=int, default=4_000_000,
                    help="MEASURED floor: at MultiPV 64 the DEEP arm needs ~4M nodes to "
                         "median depth 16 against the PROD arm's 12. At 1M it reaches 13 "
                         "and at 200k only 10 -- i.e. SHALLOWER than what it grades.")
    ap.add_argument("--deep-hash-mb", type=int, default=512)
    ap.add_argument("--workers", type=int, default=3,
                    help="engine TRIPLES; selfplay already holds ~18 of 32 cores")
    ap.add_argument("--nice", type=int, default=19)
    ap.add_argument("--max-shards", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    shards: list[Path] = []
    for d in args.replay_dir:
        shards.extend(select_shards(Path(d), args.max_shards))
    scan: dict[str, Any] = {
        "shard_names": [p.name for p in shards], "rows_scanned": 0, "searched": 0,
        "skipped_narrow": 0, "skipped_no_tail": 0,
        "skipped_shards": [], "skipped_shards_omitted": 0,
        "git_sha": git_sha(),
    }
    positions = sample_positions(shards, scan, args.positions, args.seed)
    print(f"sampled {len(positions)} positions from {scan.get('sampled_shards', 0)}"
          f" of {len(shards)} shards"
          f"   ({scan['skipped_narrow']} skipped as <= {PROD_MULTIPV} legal)")
    if scan.get("sampled_shards", 0) < min(len(shards), 2):
        print("  ⚑ every position came from ONE shard — a few dozen correlated games."
              " The bootstrap CI below is NOT earned; widen --max-shards.")
    print(f"PROD    arm: MultiPV {PROD_MULTIPV:>2}, {args.prod_nodes:>8} nodes, "
          f"Hash {PROD_HASH_MB}MB, syzygy {PROD_SYZYGY}")
    print(f"MATCHED arm: MultiPV {MAX_MULTIPV:>2}, {args.prod_nodes:>8} nodes, "
          f"Hash {PROD_HASH_MB}MB, syzygy {PROD_SYZYGY}   <- differs from PROD in WIDTH ONLY")
    print(f"DEEP    arm: MultiPV {MAX_MULTIPV:>2}, {args.deep_nodes:>8} nodes, "
          f"Hash {args.deep_hash_mb}MB, syzygy {DEEP_SYZYGY}")

    results = run_panel(
        positions, scan, stockfish=args.stockfish, prod_nodes=args.prod_nodes,
        deep_nodes=args.deep_nodes, deep_hash_mb=args.deep_hash_mb,
        workers=args.workers, nice=args.nice,
    )
    print(f"\nsearched {scan['searched']}   usable {len(results)}"
          f"   ({scan['skipped_no_tail']} had no hidden tail)")
    if not results:
        raise SystemExit("no usable positions")

    # ⚑ MultiPV is capped at MAX_MULTIPV, so in a wide-open position the wide arm
    # does not reach every legal move. Uncovered moves are DROPPED from the tail
    # rather than imputed, which biases the measured tail toward the moves SF
    # ranked highest — report the coverage instead of leaving it implicit.
    for arm in ("matched", "deep"):
        cov = np.array([r[f"{arm}_width"] / r["n_legal"] for r in results])
        scan[f"{arm}_coverage_mean"] = float(cov.mean())
        print(f"  {arm.upper():>7} arm covers {cov.mean():.4f} of legal moves"
              f"   (min {cov.min():.3f}, {(cov < 0.999).mean():.3f} of rows incomplete)")
        if cov.mean() < 0.95:
            print("  ⚑ coverage below 0.95: the deepest tail is UNDER-SAMPLED and the")
            print("    fitted alpha is biased LOW. Raise MAX_MULTIPV before reading it.")

    widths = np.array([r["prod_width"] for r in results])
    print(f"  realized PROD MultiPV width: mean {widths.mean():.2f}  "
          f"frac == {PROD_MULTIPV}: {(widths == PROD_MULTIPV).mean():.3f}")
    print("  depth (medians): "
          + "  ".join(f"{a} {np.median([r[f'{a}_depth'] for r in results]):.0f}"
                      for a in ("prod", "matched", "deep")))

    print("\n⚑ CROSS-ARM CALIBRATION — the moves each PAIR both surfaced")
    xa: dict[str, dict[str, float]] = {}
    for a, b, why in (
        ("prod_scored", "matched_scored",
         "SHOULD be ~1.0 (same budget, width only) — this is the primary's falsifier"),
        ("matched_scored", "deep_scored",
         "the DEPTH effect — not a censoring effect"),
    ):
        c = cross_arm_check(results, a, b)
        xa[f"{a}_vs_{b}"] = c
        an, bn = a.split("_")[0].upper(), b.split("_")[0].upper()
        print(f"  {an:>7} vs {bn:<7} n={c['n_moves']:>5}   "
              f"{c['a_mean_cp']:6.1f} -> {c['b_mean_cp']:6.1f} cp   "
              f"ratio {c['ratio']:.3f}   bestmove agree {c['bestmove_agreement']:.3f}")
        print(f"          {why}")
    if not 0.9 <= xa["prod_scored_vs_matched_scored"]["ratio"] <= 1.111:
        print("  ⚑⚑ PROD vs MATCHED IS NOT ~1.0. Widening the search at a FIXED budget")
        print("     changed the surfaced scores, so the primary readout's substitution is")
        print("     NOT clean and its alpha carries a width effect on top of the censoring.")

    device = attach_priors(results, args.checkpoint, args.batch)
    print(f"  prior: {args.checkpoint} on {device}")

    out: dict[str, Any] = {"scan": scan, "cross_arm": xa, "modes": {},
                           "prod_nodes": args.prod_nodes, "deep_nodes": args.deep_nodes}
    panels = (
        ("matched_substituted", "matched_scored", True,
         "(A) SELF-CONSISTENCY — PRIMARY. Same budget, tail from full width"),
        ("deep_substituted", "deep_scored", True,
         "(B) TRUTH — tail from the deep ruler, surfaced still production's"),
        ("deep_all", "deep_scored", False,
         "(B') ALL-DEEP — every regret from the ruler; indicts the whole label"),
    )
    for key, arm, sub, label in panels:
        rows = build_rows(results, arm, substitute=sub)
        if not rows:
            continue
        res = analyse(rows, GRID)
        report(label, res)
        out["modes"][key] = res

    non_tb = [r for r in results if not r["in_tb"]]
    if non_tb and len(non_tb) < len(results):
        rows = build_rows(non_tb, "matched_scored", substitute=True)
        if rows:
            res = analyse(rows, GRID)
            report(f"(A) TABLEBASE-FREE ONLY (> {TB_PIECE_LIMIT} pieces)", res)
            out["modes"]["matched_substituted_non_tb"] = res
    print(f"\n  ({sum(r['in_tb'] for r in results)} of {len(results)} positions are"
          f" within tablebase range, where the arms' TB sets differ)")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
        print(f"\nbanked -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
