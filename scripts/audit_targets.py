#!/usr/bin/env python3
"""Score training-target candidates directly against the frozen audit set.

For every deep-labeled audit position (scripts/build_audit_set.py) this
computes candidate POLICY distributions:

  a) net raw policy — single batched forward of --checkpoint
  b) net + Gumbel search at production sims (--sims, default 256)
  c) the SF MultiPV soft target (--sf-soft-nodes / --sf-soft-multipv,
     production 50k/40), built with the production sf_policy_temp /
     label-smoothing / cp-logistic params from --config
  d) the production training target: (b) retempered with the production
     move-selection temperature (policy_t IS the visit distribution at that
     temperature — see CLAUDE.md head table)

and scores each as expected deep-SF regret (cp) of a move sampled from the
distribution, plus top-1 regret — reported per phase and per source.

For VALUE it scores, against the deep-SF native WDL (and separately against
full-strength game outcomes on the positions that have them):

  i)   cp->logistic transform of the shallow SF eval (production slope/width)
  ii)  shallow SF native WDL
  iii) the production blend (sf_wdl_frac / search_wdl_frac from --config;
       the game-outcome component only contributes on outcome-labeled rows)
  iv)  search root WDL (root Q from (b) mapped to W/D/L like the loss does)

as Brier score and expected calibration error.

Shallow SF results are cached to <audit>.shallow_sf.jsonl (append-only,
resumable) so reruns against new checkpoints don't repay the CPU bill.
GPU use is the batched forwards + search only; --max-positions and
--batch-size bound the run (5k positions / 256 sims fits in <1h on a 5090).

Output: runs/target_audit_<git-sha>.md.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import numpy as np

from chess_anti_engine.eval.audit import (
    AuditPosition,
    PHASE_NAMES,
    SOURCE_NAMES,
    expected_and_top1_regret,
    legal_full_indices,
    load_audit_set,
    move_regrets,
    wdl_brier,
    wdl_ece,
)
from chess_anti_engine.moves import COMPACT_TO_FULL_POLICY, POLICY_SIZE, policy_batch_to_full_if_needed
from chess_anti_engine.moves.encode import uci_to_policy_index
from chess_anti_engine.utils.git_meta import git_sha
from chess_anti_engine.selfplay.stockfish_turn import (
    _build_sf_policy_target,
    _pv_wdl_score,
)
from chess_anti_engine.selfplay.temperature import apply_policy_temperature
from chess_anti_engine.stockfish.uci import StockfishUCI
from chess_anti_engine.stockfish.wdl import cp_to_wdl
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file

_CANDIDATE_NAMES = {
    "raw": "a) net raw policy",
    "search": "b) net + Gumbel search",
    "sf_soft": "c) SF MultiPV soft target",
    "train": "d) production training target",
}
_VALUE_NAMES = {
    "cp_logistic": "i) cp-logistic of shallow SF eval",
    "sf_native": "ii) shallow SF native WDL",
    "blend": "iii) production WDL blend",
    "search_root": "iv) search root WDL",
}


def _q_to_wdl(q: float) -> np.ndarray:
    """Root Q in [-1, 1] -> (W, D, L), mirroring losses._q_to_wdl_probs."""
    qc = max(-1.0, min(1.0, float(q)))
    win = max(0.0, qc)
    loss = max(0.0, -qc)
    draw = max(0.0, 1.0 - win - loss)
    return np.array([win, draw, loss], dtype=np.float64)


# ---------------------------------------------------------------------------
# Candidate computation
# ---------------------------------------------------------------------------


def _net_candidates(
    boards: list[chess.Board],
    *,
    checkpoint: str,
    device: str,
    batch_size: int,
    sims: int,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """(raw-policy probs over legal, search visit probs over legal, root Q)
    per position. Probs are aligned with _legal_full_indices order."""
    import torch

    from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard
    from chess_anti_engine.inference import LocalModelEvaluator
    from chess_anti_engine.mcts.gumbel import GumbelConfig
    from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    model = load_model_from_checkpoint(checkpoint, device=device)
    model.eval()
    hist = str(getattr(model, "input_history_encoding", "legacy"))
    extra = str(getattr(model, "input_extra_features", "v1"))
    pol_enc = str(getattr(model, "policy_encoding", "az_4672"))
    use_rel = bool(getattr(model, "use_dynamic_relations", False))
    evaluator = LocalModelEvaluator(model, device=device)
    rng = np.random.default_rng(seed)
    cfg = GumbelConfig(
        simulations=int(sims), add_noise=False, temperature=0.0,
        input_history_encoding=hist, input_extra_features=extra,
        policy_encoding=pol_enc, compute_relations=use_rel,
    )

    raw_out: list[np.ndarray] = []
    search_out: list[np.ndarray] = []
    root_q: list[float] = []
    for start in range(0, len(boards), batch_size):
        chunk = boards[start:start + batch_size]
        cbs = [CBoard.from_board(b) for b in chunk]
        xs = np.stack([
            encode_cboard(cb, input_history_encoding=hist, input_extra_features=extra)
            for cb in cbs
        ])
        rels = (
            np.stack([cb.compute_relations() for cb in cbs]) if use_rel else None
        )
        with torch.no_grad():
            if rels is None:
                pol_logits, _wdl = evaluator.evaluate_encoded(xs)
            else:
                pol_logits, _wdl = evaluator.evaluate_encoded(xs, relations=rels)
        pol_logits = np.asarray(pol_logits, dtype=np.float32)
        if pol_logits.shape[1] != POLICY_SIZE:
            pol_logits = policy_batch_to_full_if_needed(pol_logits, policy_encoding=pol_enc, fill_value=-1e9)

        probs_b, _actions, values, _masks, _tree, _ids = run_gumbel_root_many_c(
            model=None, boards=list(chunk), device=device, rng=rng, cfg=cfg,
            evaluator=evaluator,
        )
        for j, board in enumerate(chunk):
            _, idxs = legal_full_indices(board)
            logits = pol_logits[j, idxs].astype(np.float64)
            logits -= logits.max()
            e = np.exp(logits)
            raw_out.append(e / e.sum())
            visit = np.asarray(probs_b[j], dtype=np.float64)
            if visit.shape[0] != POLICY_SIZE:
                full = np.zeros(POLICY_SIZE, dtype=np.float64)
                full[COMPACT_TO_FULL_POLICY] = visit
                visit = full
            search_out.append(visit[idxs])
            root_q.append(float(values[j]))
        done = min(start + batch_size, len(boards))
        print(f"[net] {done}/{len(boards)} positions")
    return raw_out, search_out, root_q


def _shallow_sf_records(
    positions: list[AuditPosition],
    *,
    cache_path: Path,
    stockfish: str | None,
    nodes: int,
    multipv: int,
    workers: int,
    nice: int,
) -> dict[str, dict]:
    """Shallow (production-strength) SF search per position, JSONL-cached."""
    cache: dict[str, dict] = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    if int(d.get("nodes_requested", 0)) == nodes and int(d.get("multipv", 0)) == multipv:
                        cache[str(d["key"])] = d
    todo = [p for p in positions if p.key not in cache]
    if todo and stockfish is None:
        raise SystemExit(
            f"{len(todo)} positions lack shallow-SF cache entries; pass --stockfish"
        )
    if not todo:
        return cache

    print(f"[sf-soft] labeling {len(todo)} positions at {nodes} nodes, multipv {multipv}")
    engines = [
        StockfishUCI(str(stockfish), nodes=nodes, multipv=multipv, nice=nice)
        for _ in range(max(1, workers))
    ]
    lock = threading.Lock()
    work = iter(todo)
    t0 = time.time()
    n_done = 0
    try:
        with open(cache_path, "a", encoding="utf-8") as f:
            def run_worker(wi: int) -> None:
                nonlocal n_done
                eng = engines[wi]
                while True:
                    with lock:
                        pos = next(work, None)
                    if pos is None:
                        return
                    res = eng.search(pos.fen, nodes=nodes)
                    rec = {
                        "key": pos.key,
                        "nodes_requested": nodes,
                        "multipv": multipv,
                        "cp": None if res.cp is None else int(res.cp),
                        "mate": res.mate,
                        "wdl": None if res.wdl is None else [float(v) for v in res.wdl],
                        "pvs": [
                            {"move": pv.move_uci,
                             "cp": None if pv.cp is None else int(pv.cp),
                             "mate": pv.mate,
                             "wdl": None if pv.wdl is None else [float(v) for v in pv.wdl]}
                            for pv in (res.pvs or [])
                        ],
                    }
                    with lock:
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        cache[pos.key] = rec
                        n_done += 1
                        if n_done % 50 == 0:
                            rate = n_done / max(1e-9, time.time() - t0)
                            print(f"[sf-soft] {n_done}/{len(todo)} ({rate:.2f} pos/s)")

            with ThreadPoolExecutor(max_workers=len(engines)) as pool:
                for fut in [pool.submit(run_worker, wi) for wi in range(len(engines))]:
                    fut.result()
    finally:
        for eng in engines:
            eng.close()
    return cache


@dataclasses.dataclass(frozen=True)
class _SfSoftParams:
    sf_policy_temp: float
    sf_policy_label_smooth: float
    sf_wdl_use_cp_logistic: bool
    sf_wdl_cp_slope: float
    sf_wdl_cp_draw_width: float


class _PvLike:
    """Adapter so cached shallow-SF rows feed the live _pv_wdl_score."""

    def __init__(self, d: dict) -> None:
        self.move_uci = str(d["move"])
        self.cp = d.get("cp")
        self.mate = d.get("mate")
        self.wdl = None if d.get("wdl") is None else np.asarray(d["wdl"], dtype=np.float32)


def _sf_soft_distribution(
    rec: dict, legal_idxs: np.ndarray, *, params: _SfSoftParams,
) -> np.ndarray:
    legal_set = {int(i) for i in legal_idxs}
    cand_idxs: list[int] = []
    cand_scores: list[float] = []
    for d in rec.get("pvs", []):
        pv = _PvLike(d)
        a = uci_to_policy_index(pv.move_uci, True)
        if a < 0 or a not in legal_set:
            continue
        score = _pv_wdl_score(
            pv,
            sf_wdl_use_cp_logistic=params.sf_wdl_use_cp_logistic,
            sf_wdl_cp_slope=params.sf_wdl_cp_slope,
            sf_wdl_cp_draw_width=params.sf_wdl_cp_draw_width,
        )
        if score is None:
            continue
        cand_idxs.append(a)
        cand_scores.append(float(score))
    if not cand_idxs:
        cand_idxs = [int(legal_idxs[0])]
        cand_scores = [0.0]
    full = _build_sf_policy_target(
        cand_idxs, cand_scores, legal_indices=legal_idxs,
        sf_policy_temp=params.sf_policy_temp,
        sf_policy_label_smooth=params.sf_policy_label_smooth,
    )
    return full[legal_idxs].astype(np.float64)


# ---------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------


def _aggregate(
    rows: list[dict], key: str,
) -> dict[tuple[str, str], tuple[float, float, int]]:
    """(group, candidate) -> (mean expected regret, mean top1 regret, n)."""
    groups: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for row in rows:
        for grp in ("overall", PHASE_NAMES[row["phase"]], SOURCE_NAMES[row["source"]]):
            groups.setdefault((grp, row[key]), []).append((row["expected"], row["top1"]))
    return {
        k: (float(np.mean([v[0] for v in vals])),
            float(np.mean([v[1] for v in vals])), len(vals))
        for k, vals in groups.items()
    }


def _policy_table(agg: dict, group_names: list[str]) -> str:
    lines = ["| candidate | " + " | ".join(f"{g} E[regret] / top-1 (n)" for g in group_names) + " |"]
    lines.append("|" + "---|" * (len(group_names) + 1))
    for cand, label in _CANDIDATE_NAMES.items():
        cells = []
        for g in group_names:
            v = agg.get((g, cand))
            cells.append("—" if v is None else f"{v[0]:.1f} / {v[1]:.1f} ({v[2]})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--audit-set", type=Path, default=Path("data/audit_set_v1.jsonl"))
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/pbt2_small.yaml"),
                    help="production config for target-construction params")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--sims", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stockfish", type=str, default=None,
                    help="needed only when the shallow-SF cache is incomplete")
    ap.add_argument("--sf-soft-nodes", type=int, default=50_000)
    ap.add_argument("--sf-soft-multipv", type=int, default=40)
    ap.add_argument("--sf-workers", type=int, default=4)
    ap.add_argument("--nice", type=int, default=15)
    ap.add_argument("--max-positions", type=int, default=0,
                    help=">0 limits positions (smoke runs)")
    ap.add_argument("--out-dir", type=Path, default=Path("runs"))
    args = ap.parse_args()

    flat = flatten_run_config_defaults(load_yaml_file(args.config))
    sf_params = _SfSoftParams(
        sf_policy_temp=float(flat.get("sf_policy_temp", 0.25)),
        sf_policy_label_smooth=float(flat.get("sf_policy_label_smooth", 0.05)),
        sf_wdl_use_cp_logistic=bool(flat.get("sf_wdl_use_cp_logistic", False)),
        sf_wdl_cp_slope=float(flat.get("sf_wdl_cp_slope", 0.010)),
        sf_wdl_cp_draw_width=float(flat.get("sf_wdl_cp_draw_width", 60.0)),
    )
    train_temp = float(flat.get("temperature", 1.0))
    sf_wdl_frac = float(flat.get("sf_wdl_frac", 0.0))
    search_wdl_frac = float(flat.get("search_wdl_frac", 0.0))

    positions = load_audit_set(args.audit_set)
    if args.max_positions > 0:
        positions = positions[: args.max_positions]
    boards = [chess.Board(p.fen) for p in positions]
    print(f"[audit] {len(positions)} positions from {args.audit_set}")

    shallow = _shallow_sf_records(
        positions,
        cache_path=args.audit_set.with_suffix(args.audit_set.suffix + ".shallow_sf.jsonl"),
        stockfish=args.stockfish, nodes=int(args.sf_soft_nodes),
        multipv=int(args.sf_soft_multipv), workers=int(args.sf_workers),
        nice=int(args.nice),
    )

    raw_probs, search_probs, root_q = _net_candidates(
        boards, checkpoint=args.checkpoint, device=args.device,
        batch_size=int(args.batch_size), sims=int(args.sims), seed=int(args.seed),
    )

    policy_rows: list[dict] = []
    value_rows: dict[str, list[np.ndarray]] = {k: [] for k in _VALUE_NAMES}
    deep_wdls: list[np.ndarray] = []
    # Rows can be skipped (no encodable legal moves); every per-row list below
    # must stay aligned with kept_positions, NOT with the input order.
    kept_positions: list[AuditPosition] = []
    outcome_idx: list[int] = []
    for i, (pos, board) in enumerate(zip(positions, boards, strict=True)):
        legal_ucis, legal_idxs = legal_full_indices(board)
        if not legal_ucis:
            continue
        regrets = move_regrets(pos, legal_ucis)
        cands = {
            "raw": raw_probs[i],
            "search": search_probs[i],
            # policy_t is the visit distribution at the move-selection
            # temperature; production temperature 0.0 (and 1.0) store the
            # raw visit distribution -- the temperature then only shapes
            # action SAMPLING, not the stored target.
            "train": (
                search_probs[i]
                if train_temp <= 0.0 or train_temp == 1.0
                else apply_policy_temperature(
                    search_probs[i].astype(np.float32), train_temp,
                ).astype(np.float64)
            ),
            "sf_soft": _sf_soft_distribution(
                shallow[pos.key], legal_idxs, params=sf_params,
            ),
        }
        for cand, probs in cands.items():
            exp_r, top1_r = expected_and_top1_regret(probs, regrets)
            policy_rows.append({
                "cand": cand, "phase": pos.phase, "source": pos.source,
                "expected": exp_r, "top1": top1_r,
            })

        rec = shallow[pos.key]
        sf_native = (
            np.asarray(rec["wdl"], dtype=np.float64) if rec.get("wdl") else
            np.array([333.0, 334.0, 333.0])
        )
        sf_native = np.clip(sf_native, 0.0, None)
        sf_native = sf_native / max(1e-9, sf_native.sum())
        if rec.get("cp") is not None or rec.get("mate"):
            cp_log = cp_to_wdl(
                rec.get("cp"), rec.get("mate"),
                slope=sf_params.sf_wdl_cp_slope,
                draw_width_cp=sf_params.sf_wdl_cp_draw_width,
            ).astype(np.float64)
        else:
            cp_log = sf_native
        search_root = _q_to_wdl(root_q[i])
        # Production blend: outcome component only exists on outcome-labeled
        # rows; elsewhere the sf/search fractions are renormalized (this is
        # the same fallback shape the loss uses when a component is absent).
        w_sf, w_search = sf_wdl_frac, search_wdl_frac
        game_frac = max(0.0, 1.0 - w_sf - w_search)
        if pos.outcome is not None:
            onehot = np.zeros(3)
            onehot[int(pos.outcome)] = 1.0
            blend = game_frac * onehot + w_sf * sf_native + w_search * search_root
        else:
            denom = max(1e-9, w_sf + w_search)
            blend = (w_sf * sf_native + w_search * search_root) / denom
        value_rows["cp_logistic"].append(cp_log)
        value_rows["sf_native"].append(sf_native)
        value_rows["blend"].append(blend / max(1e-9, blend.sum()))
        value_rows["search_root"].append(search_root)
        deep_wdls.append(np.asarray(pos.deep_wdl, dtype=np.float64))
        kept_positions.append(pos)
        if pos.outcome is not None:
            outcome_idx.append(len(deep_wdls) - 1)

    agg = _aggregate(policy_rows, "cand")
    group_names = ["overall", *PHASE_NAMES, *SOURCE_NAMES]
    deep = np.stack(deep_wdls)

    value_lines = [
        "| candidate | Brier vs deep WDL | ECE vs deep WDL | Brier vs outcome (n) |",
        "|---|---|---|---|",
    ]
    for key, label in _VALUE_NAMES.items():
        preds = np.stack(value_rows[key])
        brier = float(np.mean([wdl_brier(p, t) for p, t in zip(preds, deep, strict=True)]))
        ece = wdl_ece(preds, deep)
        if outcome_idx:
            oc = [
                wdl_brier(preds[i], np.eye(3)[kept_positions[i].outcome])  # type: ignore[index]
                for i in outcome_idx
            ]
            oc_cell = f"{float(np.mean(oc)):.4f} ({len(outcome_idx)})"
        else:
            oc_cell = "— (0)"
        value_lines.append(f"| {label} | {brier:.4f} | {ece:.4f} | {oc_cell} |")

    sha = git_sha(short=True)
    out_path = args.out_dir / f"target_audit_{sha}.md"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    headline_search = agg.get(("overall", "search"))
    headline_sf = agg.get(("overall", "sf_soft"))
    report = (
        f"# Target audit @ {sha}\n\n"
        f"- audit set: {args.audit_set} ({len(deep_wdls)} scored positions)\n"
        f"- checkpoint: {args.checkpoint}\n"
        f"- search: {args.sims} sims; shallow SF: {args.sf_soft_nodes} nodes "
        f"MultiPV {args.sf_soft_multipv}; config: {args.config}\n\n"
        f"## Headline\n\n"
        f"- search-policy expected regret (overall): "
        f"{'—' if headline_search is None else f'{headline_search[0]:.1f} cp'} vs "
        f"SF-soft-target {'—' if headline_sf is None else f'{headline_sf[0]:.1f} cp'} — "
        f"prices whether {args.sf_soft_nodes}-node MultiPV-{args.sf_soft_multipv} "
        f"labeling is still worth its CPU bill (per-phase split below).\n"
        f"- production WDL blend calibration vs its best single component: "
        f"see the value table.\n\n"
        f"## Policy: expected / top-1 deep-SF regret (cp)\n\n"
        f"Unlisted legal moves carry the worst-listed-line regret as a "
        f"floor (lower bound; MultiPV >= 10 at >=1M nodes).\n\n"
        f"{_policy_table(agg, group_names)}\n\n"
        f"## Value: calibration against deep-SF WDL\n\n"
        + "\n".join(value_lines)
        + "\n\nOutcome column counts only positions whose game continued at "
        "full strength; the v1 audit set has none (handicapped curriculum), "
        "so the column awaits full-strength continuations.\n"
    )
    out_path.write_text(report, encoding="utf-8")
    print(f"[audit] report written to {out_path}")
    print(report)


if __name__ == "__main__":
    main()
