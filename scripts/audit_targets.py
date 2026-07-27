#!/usr/bin/env python3
"""Score training-target candidates directly against the frozen audit set.

For every deep-labeled audit position (scripts/build_audit_set.py) this
computes candidate POLICY distributions:

  a) net raw policy — single batched forward of --checkpoint
  b) net + Gumbel search at PLAY (UCI/TCEC) search settings, --sims
  c) the SF MultiPV soft target (--sf-soft-nodes / --sf-soft-multipv,
     low=500k / high=2M via --sf-effort, matching the 500k production
     teacher; default 500k), built with the production sf_policy_temp /
     label-smoothing / cp-logistic params from --config
  d) the production TRAINING target — the RL selfplay search from --config at
     full sims, retempered with the production move-selection temperature
     (policy_t IS the visit distribution at that temperature — see CLAUDE.md
     head table). This is the WHOLE stored policy corpus.
  e) the same search at the playout-capped fast sims, for reference only.
     Playout-capped plies carry NO policy target: finalize.py drops them, and
     with record_fast_ply_value they become value-only rows whose MAIN policy
     head is masked. Never average (e) into (d) — that invents a mixture the
     pipeline does not store.

(b) and (d)/(e) are DIFFERENT SEARCHES and must not be substituted for one
another. RL selfplay keeps `gumbel_c_scale` 0.1 with the legacy LINEAR root
value-transform; UCI/TCEC play uses c_scale 0.025 with the LOG root
(c_scale_root 7.0). Both are deliberate and separately tuned — at the 256-sim
selfplay budget 0.1 measured 0.688 puzzle accuracy against 0.598 for 0.025 —
so one config cannot stand in for the other. Before 2026-07-25 this script
built ONE search from the PLAY defaults and labelled it "production training
target", which put a play-path number next to the SF soft target in the
headline that prices SF's MultiPV CPU bill.

and scores each as expected deep-SF regret (cp) of a move sampled from the
distribution, plus top-1 regret — reported per phase and per source.

For VALUE it scores, against the deep-SF native WDL (and separately against
full-strength game outcomes on the positions that have them):

  i)   cp->logistic transform of the shallow SF eval (production slope/width)
  ii)  shallow SF native WDL
  iii) the production blend (sf_wdl_frac / search_wdl_frac from --config;
       the game-outcome component only contributes on outcome-labeled rows)
  iv)  search root WDL — the RL search's root Q from (d), reconstructed the
       way selfplay stores it: the root network's OWN draw mass is preserved
       and only the remaining mass is split around Q (see
       _search_wdl_like_selfplay). Not `1 - |Q|`, which is a different
       distribution and a different target.

  !! THIS VALUE TABLE IS A CALIBRATION RULER, NOT A TARGET-QUALITY RULER. It
  ranks candidates by AGREEMENT WITH DEEP SF, and (ii) shallow SF native WDL
  will normally win it for a reason that has nothing to do with being a good
  teacher: it is the SAME KIND OF OBJECT as the reference. Both are Stockfish,
  so wherever SF is decisive they go one-hot together and the ECE collapses.
  Measured 2026-07-27 on 2000 audit positions: (ii) Brier 0.0348 / ECE 0.0069
  vs the production blend (iii) 0.0484 / 0.0868 — a 12x ECE gap that reads as
  "switch to native WDL" and is NOT that.

  Production deliberately does the opposite (`sf_wdl_use_cp_logistic: true`)
  because SF's UCI_ShowWDL is **~72% one-hot**, and a one-hot value target
  teaches over-confidence — the failure actually observed in play (2026-06-28
  loss: the net evaluated +557 while the position was lost by ~300, an ~860cp
  sign error). The cp-logistic's high ECE **against a deep-SF ruler IS the
  deliberate softness**, not a defect; see CLAUDE.md ("the cp-logistic label is
  deliberately soft; don't chase value sharpness against a deep-SF ruler") and
  the WDL blend section of docs/model_heads.md.

  So: use this table to detect a candidate that has DRIFTED or BROKEN, never to
  pick the value target. Reading it as a target ranking was attempted and
  retracted on 2026-07-27.

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
    criticality_gap,
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
    "search": "b) net + Gumbel search (PLAY settings)",
    "sf_soft": "c) SF MultiPV soft target",
    "train": "d) production training target (full sims)",
    "train_fast": "e) fast-ply search — NOT a policy target in production",
}


@dataclasses.dataclass(frozen=True)
class _SearchProfile:
    """One search shape to score, named for the pipeline stage it belongs to.

    Keeping these separate is the point: the value-transform knobs below are
    tuned per sim-budget and the RL and play budgets disagree, so scoring a
    training target with play settings (or vice versa) reports a number no
    stage of the pipeline actually produces.
    """

    label: str
    sims: int
    topk: int
    c_scale: float
    c_visit: float
    c_visit_root: float
    c_scale_root: float
    q_visit_exp_root: float
  # Descent knobs. PLAY and training disagree on all four, and they act on
  # tree descent, so omitting them leaves a hybrid search that neither path
  # actually runs.
    c_puct: float
    cpuct_factor: float
    cpuct_base: float
    fpu_reduction: float
  # Volatility-aware search. Both default OFF; when either is non-zero the
  # mechanism exists only on the PYTHON search path, and selfplay drops to it.
    volatility_q_scale: float = 0.0
    volatility_fpu: float = 0.0
    volatility_anchor: float | None = None


def build_search_profiles(
    flat: dict[str, object], *, play_sims: int, play_topk: int | None,
) -> dict[str, _SearchProfile]:
    """The search shapes to score: one PLAY, two TRAINING.

    `flat` is the flattened run config, so the training profiles follow the
    live yaml rather than a constant that goes stale the moment a search knob
    is tuned. GumbelConfig's own defaults ARE the RL shape (c_scale 0.1 and the
    legacy LINEAR root via the c_visit_root/c_scale_root/q_visit_exp_root
    sentinels below), which is the deliberate "training/RL stays bit-identical"
    choice from PR #84 — the sentinels are not placeholders.
    """
    from chess_anti_engine.mcts.gumbel import PLAY_SEARCH_DEFAULTS, GumbelConfig

    rl = GumbelConfig()  # the RL/training shape, by construction
    rl_c_scale = float(flat.get("gumbel_c_scale", rl.c_scale))  # pyright: ignore[reportArgumentType]
    rl_topk = int(flat.get("gumbel_topk", rl.topk))  # pyright: ignore[reportArgumentType]
    rl_sims = int(flat.get("mcts_simulations", 256))  # pyright: ignore[reportArgumentType]
    rl_fast_sims = int(flat.get("fast_simulations", 32))  # pyright: ignore[reportArgumentType]

    def _rl(label: str, sims: int) -> _SearchProfile:
        return _SearchProfile(
            label=label, sims=sims, topk=rl_topk, c_scale=rl_c_scale,
            c_visit=rl.c_visit, c_visit_root=rl.c_visit_root,
            c_scale_root=rl.c_scale_root, q_visit_exp_root=rl.q_visit_exp_root,
            c_puct=rl.c_puct, cpuct_factor=rl.cpuct_factor,
            cpuct_base=rl.cpuct_base, fpu_reduction=rl.fpu_reduction,
          # Volatility search is an open, default-off flag family that the
          # audit-first rule still has to be able to judge. Carrying the
          # values means enabling them in the yaml changes the audited
          # target, instead of the audit quietly scoring the baseline.
            volatility_q_scale=float(flat.get("volatility_q_scale", 0.0)),  # pyright: ignore[reportArgumentType]
            volatility_fpu=float(flat.get("volatility_fpu", 0.0)),  # pyright: ignore[reportArgumentType]
            volatility_anchor=(
                float(flat["volatility_anchor"])  # pyright: ignore[reportArgumentType]
                if flat.get("volatility_anchor") is not None else None
            ),
        )

    return {
        "search": _SearchProfile(
            label="PLAY (UCI/TCEC)", sims=int(play_sims),
          # The PLAY row must be the WHOLE play shape. topk, c_puct,
          # cpuct_factor and fpu_reduction all differ from the training
          # defaults and all act on descent; taking only the root-transform
          # subset left a hybrid neither path runs.
            topk=int(play_topk if play_topk is not None else PLAY_SEARCH_DEFAULTS["topk"]),
            c_scale=float(PLAY_SEARCH_DEFAULTS["c_scale"]),
            c_visit=float(PLAY_SEARCH_DEFAULTS["c_visit"]),
            c_visit_root=float(PLAY_SEARCH_DEFAULTS["c_visit_root"]),
            c_scale_root=float(PLAY_SEARCH_DEFAULTS["c_scale_root"]),
            q_visit_exp_root=float(PLAY_SEARCH_DEFAULTS["q_visit_exp_root"]),
            c_puct=float(PLAY_SEARCH_DEFAULTS["c_puct"]),
            cpuct_factor=float(PLAY_SEARCH_DEFAULTS["cpuct_factor"]),
            cpuct_base=float(PLAY_SEARCH_DEFAULTS["cpuct_base"]),
            fpu_reduction=float(PLAY_SEARCH_DEFAULTS["fpu_reduction"]),
        ),
        "train": _rl("RL selfplay, full sims", rl_sims),
        "train_fast": _rl("RL selfplay, playout-capped fast sims", rl_fast_sims),
    }


_VALUE_NAMES = {
    "cp_logistic": "i) cp-logistic of shallow SF eval",
    "sf_native": "ii) shallow SF native WDL",
    "blend": "iii) production WDL blend",
    "search_root": "iv) search root WDL",
}


def _wdl_softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax of raw WDL logits, as `network_turn.py` does.

    `LocalModelEvaluator.evaluate_encoded` returns the model's RAW ``out["wdl"]``
    logits, not probabilities. Selfplay softmaxes them before reading the draw
    component (`network_turn.py:365-371`); feeding the logits straight into
    `_search_wdl_like_selfplay` would treat an arbitrary real number as a draw
    probability and produce negative or >1 entries that are still finite, so the
    non-finite fallback there would not catch them.
    """
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=-1, keepdims=True)
    return z


def _search_wdl_like_selfplay(q: float, net_wdl: np.ndarray) -> np.ndarray:
    """The search WDL exactly as `network_turn.py` stores it.

    ``net_wdl`` must be PROBABILITIES (see `_wdl_softmax`), not logits.

    Selfplay KEEPS the root network's own draw mass and splits only the
    remaining mass around the searched Q::

        d_raw = net_wdl[1]; rem = 1 - d_raw
        q     = clip(q, -rem, +rem)
        W     = 0.5 * (rem + q);  D = d_raw;  L = rem - W

    `losses._q_to_wdl_probs` -- the game-outcome regret correction, a DIFFERENT
    target -- instead invents ``D = 1 - |q|``, which is a different
    distribution whenever the net predicts a draw mass other than ``1 - |q|``
    -- i.e. almost always. Scoring the production WDL blend with the wrong
    draw mass makes candidates (iii) and (iv) describe a target the pipeline
    never writes, which is the same mislabeling this script was just fixed for
    on the policy side.
    """
    d_raw = float(net_wdl[1])
    rem = max(0.0, 1.0 - d_raw)
    qc = float(max(-rem, min(rem, float(q))))
    win = 0.5 * (rem + qc)
    out = np.array([win, d_raw, rem - win], dtype=np.float64)
    if not np.all(np.isfinite(out)):
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# Candidate computation
# ---------------------------------------------------------------------------


def _net_candidates(
    boards: list[chess.Board],
    *,
    checkpoint: str,
    device: str,
    batch_size: int,
    seed: int,
    profiles: dict[str, _SearchProfile],
    policy_temp: float = 1.0,
    syzygy_path: str | None = None,
    target_batch: int = 0,
) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]], dict[str, list[float]], list[np.ndarray]]:
    """(raw-policy probs, {profile: search visit probs}, {profile: root Q}).

    Every profile is run over the same batches against the same evaluator, so
    the raw forward and the model load are paid once no matter how many search
    shapes are being priced. Probs are aligned with _legal_full_indices order."""
    import torch

    from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard
    from chess_anti_engine.inference import LocalModelEvaluator
    from chess_anti_engine.mcts.gumbel import (
        GumbelConfig,
        run_gumbel_root_many,
        volatility_search_enabled,
        warn_volatility_python_path,
    )
    from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    model = load_model_from_checkpoint(checkpoint, device=device)
    model.eval()
    hist = str(getattr(model, "input_history_encoding", "legacy"))
    extra = str(getattr(model, "input_extra_features", "v1"))
    pol_enc = str(getattr(model, "policy_encoding", "lc0_1858"))
    use_rel = bool(getattr(model, "use_dynamic_relations", False))
    evaluator = LocalModelEvaluator(model, device=device)
    rng = np.random.default_rng(seed)
  # add_noise=False on every profile: root Gumbel noise (`gumbel_scale` 0.75
  # selfplay / 0.25 curriculum) DOES perturb the stored visit distribution, so
  # the training-target rows measure the noise-free shape of the target rather
  # than a single noisy draw of it. That is a deliberate, stated deviation --
  # the alternative is a non-deterministic ruler -- and it is the ONE axis on
  # which the train profiles still differ from live selfplay.
    def _build(p: _SearchProfile) -> GumbelConfig:
        kw = {}
        if p.volatility_anchor is not None:
            kw["volatility_anchor"] = p.volatility_anchor
        return GumbelConfig(
            simulations=int(p.sims), add_noise=False, temperature=0.0,
            input_history_encoding=hist, input_extra_features=extra,
            policy_encoding=pol_enc, compute_relations=use_rel,
            policy_temp=float(policy_temp), topk=int(p.topk),
            c_scale=p.c_scale, c_visit=p.c_visit,
            c_visit_root=p.c_visit_root, c_scale_root=p.c_scale_root,
            q_visit_exp_root=p.q_visit_exp_root,
            c_puct=p.c_puct, cpuct_factor=p.cpuct_factor,
            cpuct_base=p.cpuct_base, fpu_reduction=p.fpu_reduction,
            volatility_q_scale=p.volatility_q_scale,
            volatility_fpu=p.volatility_fpu,
            **kw,
        )

    cfgs = {name: _build(p) for name, p in profiles.items()}
  # Volatility-aware search exists ONLY on the Python path; selfplay drops to
  # it when either flag is set. Always calling the C path would silently score
  # the baseline search and report it as the configured training target -- the
  # audit-first gate would then be structurally unable to judge the one flag
  # family it was asked about.
  # Production runs `syzygy_in_search: true`, and selfplay hands the probe to
  # the C search so TB-eligible roots and leaves get their WDL overridden
  # (`network_turn.py:762`). Without it the endgame bucket -- a third of the
  # audit set, and the bucket TB probing exists for -- scores a pure-network
  # search rather than the target production stores.
    tb_probe = None
    if syzygy_path:
        from chess_anti_engine.tablebase import SyzygyProbe
        tb_probe = SyzygyProbe(syzygy_path)
        print(f"[audit] syzygy_in_search: probing {syzygy_path}", flush=True)

    runners = {}
    tb_kwargs: dict[str, dict[str, object]] = {}
    for name, cfg in cfgs.items():
        if volatility_search_enabled(cfg):
            warn_volatility_python_path()
            print(
                f"[audit] {_CANDIDATE_NAMES[name]}: volatility search on "
                f"(q_scale={cfg.volatility_q_scale}, fpu={cfg.volatility_fpu}) "
                f"— using the Python search path, as selfplay does",
                flush=True,
            )
            if tb_probe is not None:
              # The Python path takes no probe, so this profile is scored
              # WITHOUT the TB overrides production applies. Say so rather
              # than reporting it as the production target.
                print(
                    f"[audit] WARNING {_CANDIDATE_NAMES[name]}: the Python "
                    "volatility path cannot take a syzygy probe — endgame "
                    "numbers for this row are NOT the production target",
                    flush=True,
                )
            runners[name] = run_gumbel_root_many
            tb_kwargs[name] = {}
        else:
            runners[name] = run_gumbel_root_many_c
            tb_kwargs[name] = {"tb_probe": tb_probe} if tb_probe is not None else {}
            # C17 separating test: production accumulates leaves across halving
            # reps to fill GSS_GPU_BATCH, and with vloss_weight=0 a later rep
            # re-walks an UNCHANGED tree and re-evaluates the SAME leaf --
            # 29-76% duplicates at 256 sims, -34% tree nodes. Those duplicate
            # visits still increment N, which inflates max_visit, which sets the
            # root q_scale that sharpens the improved-policy TRAINING TARGET.
            # `--target-batch 1` flushes per rep, removing the duplication, so
            # running this audit at 0 vs 1 separates "C17 wastes compute" from
            # "C17 corrupts the target". The Python reference path takes no such
            # argument, hence C-runner only.
            if target_batch > 0:
                tb_kwargs[name]["target_batch"] = int(target_batch)

    raw_out: list[np.ndarray] = []
    search_out: dict[str, list[np.ndarray]] = {name: [] for name in profiles}
    root_q: dict[str, list[float]] = {name: [] for name in profiles}
  # The ROOT NETWORK's WDL, needed to rebuild the search WDL the way selfplay
  # does (see _search_wdl_like_selfplay).
    root_wdl_out: list[np.ndarray] = []
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
                pol_logits, net_wdl = evaluator.evaluate_encoded(xs)
            else:
                pol_logits, net_wdl = evaluator.evaluate_encoded(xs, relations=rels)
        net_wdl = _wdl_softmax(net_wdl)
        pol_logits = np.asarray(pol_logits, dtype=np.float32)
        if pol_logits.shape[1] != POLICY_SIZE:
            pol_logits = policy_batch_to_full_if_needed(pol_logits, policy_encoding=pol_enc, fill_value=-1e9)

        searched = {
            name: runners[name](
                model=None, boards=list(chunk), device=device, rng=rng,
                cfg=cfgs[name], evaluator=evaluator, **tb_kwargs[name],
            )
            for name in cfgs
        }
        for j, board in enumerate(chunk):
            _, idxs = legal_full_indices(board)
            logits = pol_logits[j, idxs].astype(np.float64)
            logits -= logits.max()
            e = np.exp(logits)
            raw_out.append(e / e.sum())
          # The C runner returns 6 elements and the Python one 4; only the
          # leading (probs, actions, values, masks) are common, so index in
          # rather than destructuring a length this code does not control.
            for name, result in searched.items():
                probs_b, values = result[0], result[2]
                visit = np.asarray(probs_b[j], dtype=np.float64)
                if visit.shape[0] != POLICY_SIZE:
                    full = np.zeros(POLICY_SIZE, dtype=np.float64)
                    full[COMPACT_TO_FULL_POLICY] = visit
                    visit = full
                search_out[name].append(visit[idxs])
                root_q[name].append(float(values[j]))
            root_wdl_out.append(net_wdl[j].copy())
        done = min(start + batch_size, len(boards))
        print(f"[net] {done}/{len(boards)} positions")
        # Release the batch's reserved CUDA blocks so the allocator's pool
        # doesn't creep across batches and collide with a concurrent trainer
        # (same fragmentation issue fixed in eval/puzzles.py). Matters most at
        # high sims (256) where per-batch trees are largest.
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return raw_out, search_out, root_q, root_wdl_out


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
    other_node_counts: set[int] = set()
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    if int(d.get("nodes_requested", 0)) == nodes and int(d.get("multipv", 0)) == multipv:
                        cache[str(d["key"])] = d
                    elif int(d.get("multipv", 0)) == multipv:
                        other_node_counts.add(int(d.get("nodes_requested", 0)))
    todo = [p for p in positions if p.key not in cache]
    if todo and stockfish is None:
        hint = "pass --stockfish to populate the cache"
        if other_node_counts and not cache:
  # The cache is fully populated but at a different node budget — the audit
  # default is now --sf-effort=low (500k) to match production, so an older 50k
  # cache no longer matches. Point the user at the mismatch instead of a bare
  # "pass --stockfish".
            have = ",".join(f"{n:_}" for n in sorted(other_node_counts))
            hint = (
                f"the cache has entries only at {have} nodes, but this run wants "
                f"{nodes:_} (default --sf-effort=low=500k). Re-run with "
                f"--sf-soft-nodes {sorted(other_node_counts)[0]} (or matching "
                "--sf-effort) to reuse it, or pass --stockfish to regenerate"
            )
        raise SystemExit(f"{len(todo)} positions lack shallow-SF cache entries; {hint}")
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
    ap.add_argument("--policy-temp", type=float, default=1.0,
                    help="prior temperature on policy logits before gumbel search "
                         "(>1 softens prior, <1 sharpens, 1.0=no-op). Measures search-prior "
                         "calibration on the REAL audit-set distribution (vs puzzle bias).")
    ap.add_argument("--gumbel-topk", type=int, default=None,
                    help="Override the PLAY row's Gumbel root candidate count. "
                         "Default None = the PLAY default (32). The TRAINING rows "
                         "always take selfplay's value from --config (16) and are "
                         "NOT affected by this flag -- overriding the target's own "
                         "topk would score a search selfplay never runs. At 256 "
                         "sims, ~30 legal moves means topk=32 ≈ all-legal.")
    ap.add_argument("--gpu-mem-fraction", type=float, default=None,
                    help="cap this process to a fraction of GPU memory "
                         "(set_per_process_memory_fraction) so a high-sim audit run "
                         "CONCURRENT with a live trainer fails-fast on its own OOM instead "
                         "of faulting the shared GPU/broker. e.g. 0.4 on a 32GB card.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stockfish", type=str, default=None,
                    help="needed only when the shallow-SF cache is incomplete")
    ap.add_argument("--sf-soft-nodes", type=int, default=None,
                    help="explicit shallow-SF node count; overrides --sf-effort when set")
    ap.add_argument("--sf-effort", choices=("low", "high"), default="low",
                    help="shallow-SF strength tier when --sf-soft-nodes is unset: "
                         "low=500k (matches the production teacher), high=2M (deeper reference). "
                         "The 50k default was retired once production moved to 500k nodes. NOTE: "
                         "the cache is keyed by node count, so switching tiers needs --stockfish to "
                         "(re)label at the new count.")
    ap.add_argument("--sf-soft-multipv", type=int, default=40)
    ap.add_argument("--sf-workers", type=int, default=4)
    ap.add_argument("--nice", type=int, default=15)
    ap.add_argument("--target-batch", type=int, default=0,
                    help="C-search leaf-accumulation batch. 0 = production (accumulate across "
                         "halving reps to fill GSS_GPU_BATCH). 1 = flush per rep, which removes "
                         "C17's duplicate leaves (29-76%% at 256 sims, -34%% tree nodes). Run the "
                         "audit at 0 and at 1 to separate 'C17 wastes compute' from 'C17 corrupts "
                         "the training target': duplicate visits still increment N, inflating "
                         "max_visit and hence the root q_scale that sharpens the improved-policy "
                         "target. C-runner only; the Python reference path takes no such argument.")
    ap.add_argument("--max-positions", type=int, default=0,
                    help=">0 limits positions (smoke runs)")
    ap.add_argument("--dump-per-position", type=Path, default=None,
                    help="if set, write one JSONL record per scored position "
                         "(phase, source, criticality gap, per-candidate "
                         "expected/top1 regret) for offline slicing")
    ap.add_argument("--out-dir", type=Path, default=Path("runs"))
    args = ap.parse_args()

    if args.sf_soft_nodes is None:
        args.sf_soft_nodes = {"low": 500_000, "high": 2_000_000}[args.sf_effort]

    if args.gpu_mem_fraction is not None and str(args.device).startswith("cuda"):
        import torch
        torch.cuda.set_per_process_memory_fraction(
            float(args.gpu_mem_fraction), torch.device(args.device).index or 0)
        print(f"[audit] GPU memory capped at fraction {args.gpu_mem_fraction}")

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

    from chess_anti_engine.mcts.gumbel import PLAY_SEARCH_DEFAULTS

    full_share = float(flat.get("playout_cap_fraction", 1.0))
    profiles = build_search_profiles(
        flat, play_sims=int(args.sims), play_topk=(int(args.gumbel_topk) if args.gumbel_topk is not None else None),
    )
    rl_c_scale = profiles["train"].c_scale
    rl_sims = profiles["train"].sims
    rl_fast_sims = profiles["train_fast"].sims
    for name, prof in profiles.items():
        print(
            f"[audit] {_CANDIDATE_NAMES[name]}: {prof.label} — "
            f"sims={prof.sims} topk={prof.topk} c_scale={prof.c_scale} "
            f"root={'log' if prof.q_visit_exp_root < 0 else 'linear'}",
            flush=True,
        )

  # Production probes tablebases inside the search; the audited target has to
  # as well or the endgame bucket describes a search production never runs.
    sz_path = str(flat.get("syzygy_path") or "") if flat.get("syzygy_in_search") else ""

    raw_probs, search_by_profile, root_q_by_profile, root_wdl = _net_candidates(
        boards, checkpoint=args.checkpoint, device=args.device,
        batch_size=int(args.batch_size), seed=int(args.seed),
        profiles=profiles, policy_temp=float(args.policy_temp),
        syzygy_path=sz_path or None,
        target_batch=int(args.target_batch),
    )
    search_probs = search_by_profile["search"]
  # The production WDL blend's search component comes from the RL search, so
  # value candidate (iv) must read the RL root Q, not the play-path one.
    root_q = root_q_by_profile["train"]

    policy_rows: list[dict] = []
    per_pos_dump: list[dict] = []
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
        def _as_stored(probs: np.ndarray) -> np.ndarray:
            # policy_t is the visit distribution at the move-selection
            # temperature; production temperature 0.0 (and 1.0) store the
            # raw visit distribution -- the temperature then only shapes
            # action SAMPLING, not the stored target.
            if train_temp <= 0.0 or train_temp == 1.0:
                return probs
            return apply_policy_temperature(
                probs.astype(np.float32), train_temp,
            ).astype(np.float64)

        cands = {
            "raw": raw_probs[i],
            "search": search_probs[i],
            "train": _as_stored(search_by_profile["train"][i]),
            "train_fast": _as_stored(search_by_profile["train_fast"][i]),
            "sf_soft": _sf_soft_distribution(
                shallow[pos.key], legal_idxs, params=sf_params,
            ),
        }
        per_cand: dict[str, dict] = {}
        for cand, probs in cands.items():
            exp_r, top1_r = expected_and_top1_regret(probs, regrets)
            policy_rows.append({
                "cand": cand, "phase": pos.phase, "source": pos.source,
                "expected": exp_r, "top1": top1_r,
            })
            top_i = int(np.argmax(probs))
            pv = np.asarray(probs, dtype=np.float64)
            pv = pv / max(1e-12, pv.sum())
            entropy = float(-(pv[pv > 0] * np.log(pv[pv > 0])).sum())
            per_cand[cand] = {
                "exp": exp_r, "top1": top1_r,
                "move": legal_ucis[top_i], "p": float(probs[top_i]),
                "entropy": entropy,
            }
        if args.dump_per_position is not None:
            # Criticality = deep-SF gap between the best and 2nd-best listed line
            # (cp). Small gap = quiet position where SF's "best" is near-arbitrary
            # among near-equal moves; large gap = decision-critical. Shared with
            # bt4_audit / audit_compare_buckets so the joined comparison agrees.
            gap = criticality_gap(pos.move_cp)
            per_pos_dump.append({
                "key": pos.key, "phase": pos.phase, "source": pos.source,
                # null (not inf -> non-standard JSON "Infinity") for <2-move positions
                "gap_cp": float(gap) if np.isfinite(gap) else None,
                "n_legal": len(legal_ucis),
                "n_listed": len(pos.move_cp), "best_cp": float(pos.best_cp),
                "cand": per_cand,
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
        search_root = _search_wdl_like_selfplay(root_q[i], root_wdl[i])
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

    if args.dump_per_position is not None:
        args.dump_per_position.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_per_position.open("w") as fh:
            for rec in per_pos_dump:
                fh.write(json.dumps(rec) + "\n")
        print(f"[audit] per-position dump → {args.dump_per_position} "
              f"({len(per_pos_dump)} rows)")

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
                wdl_brier(preds[i], np.eye(3)[kept_positions[i].outcome])
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
  # The stored POLICY corpus is full-sim rows ONLY -- it is NOT a playout-cap
  # mixture. `finalize.py` drops playout-capped rows outright by default, and
  # with `record_fast_ply_value` they become value-only rows whose MAIN policy
  # head is masked ("Fast plies never get SF label queries either way"). That
  # is KataGo's playout-cap design working as intended: cheap plies buy game
  # length and value coverage, never policy supervision. So the headline is the
  # full-sim row alone -- weighting it by playout_cap_fraction would invent a
  # mixture nothing stores and understate the target by ~9cp.
    headline_full = agg.get(("overall", "train"))
    headline_fast = agg.get(("overall", "train_fast"))
    train_note = "—" if headline_full is None else f"{headline_full[0]:.1f} cp"
    fast_note = (
        "—" if headline_fast is None
        else f"{headline_fast[0]:.1f} cp at {rl_fast_sims} sims"
    )
    report = (
        f"# Target audit @ {sha}\n\n"
        f"- audit set: {args.audit_set} ({len(deep_wdls)} scored positions)\n"
        f"- checkpoint: {args.checkpoint}\n"
        f"- search: PLAY {args.sims} sims / RL train {rl_sims} full + {rl_fast_sims} fast "
        f"(playout_cap_fraction {full_share}); shallow SF: {args.sf_soft_nodes} nodes "
        f"MultiPV {args.sf_soft_multipv}; config: {args.config}\n\n"
        f"## Headline\n\n"
        f"- **production TRAINING target** expected regret (overall): {train_note} vs "
        f"SF-soft-target {'—' if headline_sf is None else f'{headline_sf[0]:.1f} cp'} — "
        f"this is the pair that prices whether {args.sf_soft_nodes}-node "
        f"MultiPV-{args.sf_soft_multipv} labeling is still worth its CPU bill, "
        f"because both sides are targets training actually stores "
        f"(per-phase split below).\n"
        f"- fast-ply (playout-capped) search: {fast_note} — reported for "
        f"reference only. Playout-capped plies carry NO policy target: "
        f"finalize.py drops them, and with record_fast_ply_value they become "
        f"value-only rows with the MAIN policy head masked. Do not average "
        f"this into the training-target number.\n"
        f"- PLAY-path search regret (overall): "
        f"{'—' if headline_search is None else f'{headline_search[0]:.1f} cp'} — "
        f"the UCI/TCEC number. NOT comparable to the SF soft target for the "
        f"labeling decision: it is a different search (c_scale "
        f"{PLAY_SEARCH_DEFAULTS['c_scale']} + log root vs RL's {rl_c_scale} + "
        f"linear root) and no training row is ever built from it.\n"
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
