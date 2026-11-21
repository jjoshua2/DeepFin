"""Offline scorer for the RL run 1 plan: forward nets on audit rows, score them
against ruler labels, choose a cell's dose, and sample the A2 audit set.

See docs/experiments/2026-11-20-rl-run1-multi-generation-plan.md (Stage 0b, the
dose choice, P5). CPU by default; nothing here trains or plays.

Audit rows (``--rows``) are an npz with ``x`` (N, 175, 8, 8), ``legal``
(N, 1858) bool and ``game_id`` (N,) -- the rl-audit ``positions.npz`` schema;
``a2-sample`` writes the same schema. A ruler is ``NAME=FILE.npz:KEY`` whose
array is either (N, 1858) (a policy / visit distribution or logits: its move
is the legal argmax) or (N,) int (a move index in the 1858 compact space).

Subcommands:
  forward    NAME=CKPT ... -> priors npz (pol_NAME, wdl_NAME, sha_NAME)
  score      per net and ruler: top-1, calibration gap, entropy, each with a
             game-clustered SE; paired top-1 difference vs --reference; value
             SHAPE columns from the saved WDL logits; optional deep-SF regret
  validate   the plan's three ruler-validation conditions for one ruler file
  dose       the plan's dose rule over one cell's snapshots + final model; plays
             the FINAL model unless --ruler-validation passes for that exact ruler
  a2-openings  the a2 block of the EVAL split as a collect's openings dir
  a2-sample  about K rows per game from a sealed collect, with replayed FENs and
             a position-key check against training collections

A2, end to end (SL base, 64 sims; the collect is GPU work, the rest CPU):
  rl_audit_score.py a2-openings --eval-jsonl $R/openings/eval.jsonl --lines 800:1800 --out-dir $R/a2/openings
  simple_rl_loop.py collect --run-dir $R/a2/games --generation 1 --openings-dir $R/a2/openings \
      --init-checkpoint $R/sl_base.pt --games 1000 --opening-sampling cycle --book-mix 1.0 --seed 2026112500 \
      --no-eval <the Stage 1 collect's search flags>
  rl_audit_score.py a2-sample --collect-dir $R/a2/games/gen_0001/collect --per-game 3 \
      --exclude-collect $R/s1/collect/gen_0001/collect --out $R/a2/a2_rows.npz

Every SE is clustered by game: rows of one game are not independent.

Every output records the rows file and its sha256, the mask (spec, sha256,
polarity) and the rows kept. ``dose`` refuses rows without a mask unless
``--no-mask`` is given (A1 holds rows that reappear in training games).

Deep-SF regret (``--regret NAME=FILE.npz``: ``moves`` (N, K) compact move
indices, -1 padded, and ``regret_cp`` (N, K)) is a REPORTED column only: it
tracks raw-prior strength (it agrees with the temperature-0 1-sim matches and
not with 64-sim play), so it never selects a dose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- statistics


def clustered_mean_se(values: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    """Mean of ``values`` and its cluster-robust SE (clusters = ``groups``)."""
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    if n == 0:
        return float("nan"), float("nan")
    mean = float(v.mean())
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    g = int(inv.max()) + 1
    if g < 2:
        return mean, float("nan")
    sums = np.bincount(inv, weights=v - mean, minlength=g)
    return mean, float(np.sqrt(g / (g - 1) * float((sums ** 2).sum())) / n)


def log_softmax_legal(logits: np.ndarray, legal: np.ndarray) -> np.ndarray:
    z = np.where(legal, logits.astype(np.float64), -np.inf)
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def legal_argmax(arr: np.ndarray, legal: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr.astype(np.int64)
    return np.argmax(np.where(legal, arr.astype(np.float64), -np.inf), axis=1)


def policy_stats(logits: np.ndarray, legal: np.ndarray, ruler_move: np.ndarray) -> dict[str, np.ndarray]:
    """Per-row agreement with the ruler, top-1 probability and entropy."""
    lp = log_softmax_legal(logits, legal)
    p = np.exp(lp)
    move = np.argmax(np.where(legal, lp, -np.inf), axis=1)
    return {"move": move, "agree": (move == ruler_move).astype(np.float64),
            "p_top1": p.max(axis=1), "entropy": -np.where(legal, p * np.where(legal, lp, 0.0), 0.0).sum(axis=1)}


def score_one(logits: np.ndarray, legal: np.ndarray, ruler_move: np.ndarray, game: np.ndarray) -> dict[str, Any]:
    st = policy_stats(logits, legal, ruler_move)
    top1, top1_se = clustered_mean_se(st["agree"], game)
    gap, gap_se = clustered_mean_se(st["p_top1"] - st["agree"], game)
    ent, ent_se = clustered_mean_se(st["entropy"], game)
    return {"rows": int(legal.shape[0]), "games": int(np.unique(game).size), "top1": top1, "top1_se": top1_se,
            "calibration_gap": gap, "calibration_gap_se": gap_se, "entropy": ent, "entropy_se": ent_se}


def paired_top1(logits_a: np.ndarray, logits_b: np.ndarray, legal: np.ndarray, ruler_move: np.ndarray,
                game: np.ndarray) -> dict[str, float]:
    """top-1(a) - top-1(b) on the same rows, game-clustered (the paired SE)."""
    a = policy_stats(logits_a, legal, ruler_move)["agree"]
    b = policy_stats(logits_b, legal, ruler_move)["agree"]
    diff, se = clustered_mean_se(a - b, game)
    return {"diff": diff, "se": se, "lb95": diff - 1.96 * se, "ub95": diff + 1.96 * se}


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# |W - L| buckets of the REFERENCE net, as search-compose/wdl_shape.json splits them.
WL_BUCKETS = ((0.0, 0.1, "|WL_ref|<0.1"), (0.1, 0.3, "0.1-0.3"), (0.3, 1.01, ">0.3"))


def value_shape(wdl_logits: np.ndarray, game: np.ndarray, ref_wl: np.ndarray | None = None,
                ref_logits: np.ndarray | None = None) -> dict[str, Any]:
    """WDL SHAPE columns (game-clustered): P(draw), WDL entropy, |W - L|, and
    P(draw) by the reference net's |W - L| bucket; paired differences vs the
    reference when given. These separated the damaged value heads where the
    error metrics did not (search-compose/wdl_shape.json)."""
    p = softmax_rows(wdl_logits)
    per = {"p_draw": p[:, 1], "entropy": -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1),
           "abs_wl": np.abs(p[:, 0] - p[:, 2])}
    out: dict[str, Any] = {}
    for k, v in per.items():
        out[k], out[f"{k}_se"] = clustered_mean_se(v, game)
    if ref_wl is not None:
        out["p_draw_by_ref_bucket"] = {}
        for lo, hi, name in WL_BUCKETS:
            sel = (ref_wl >= lo) & (ref_wl < hi)
            m, se = clustered_mean_se(per["p_draw"][sel], game[sel])
            out["p_draw_by_ref_bucket"][name] = {"n": int(sel.sum()), "mean": m, "se": se}
    if ref_logits is not None:
        pr = softmax_rows(ref_logits)
        ref = {"p_draw": pr[:, 1], "entropy": -(pr * np.log(np.clip(pr, 1e-12, 1.0))).sum(axis=1),
               "abs_wl": np.abs(pr[:, 0] - pr[:, 2])}
        out["vs_reference"] = {}
        for k in per:
            d, se = clustered_mean_se(per[k] - ref[k], game)
            out["vs_reference"][k] = {"diff": d, "se": se, "lb95": d - 1.96 * se, "ub95": d + 1.96 * se}
    return out


REGRET_CAVEAT = ("deep-SF regret tracks RAW-prior strength (it agrees with the temperature-0 1-sim matches, "
                 "and not with 64-sim play); reported only, never a dose selector")


def _regret_of(move: np.ndarray, table: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(regret per row, covered per row): an uncovered row has no regret (it is DROPPED, never zero-filled)."""
    hit = table["moves"] == move[:, None]
    covered = hit.any(axis=1)
    return np.where(hit, table["regret_cp"], 0.0).sum(axis=1), covered


def regret_column(move: np.ndarray, table: dict[str, np.ndarray], game: np.ndarray) -> dict[str, Any]:
    """Mean deep-SF regret (cp) of each row's chosen move, over the rows whose
    move the label table covers; the coverage count is reported."""
    val, covered = _regret_of(move, table)
    m, se = clustered_mean_se(val[covered], game[covered])
    return {"regret_cp": m, "regret_cp_se": se, "covered": int(covered.sum()), "rows": int(move.size),
            "caveat": REGRET_CAVEAT}


def paired_regret(move_a: np.ndarray, move_b: np.ndarray, table: dict[str, np.ndarray],
                  game: np.ndarray) -> dict[str, Any]:
    """Paired regret difference a - b (cp) on the rows whose moves the table covers for BOTH nets."""
    va, ca = _regret_of(move_a, table)
    vb, cb = _regret_of(move_b, table)
    both = ca & cb
    d, se = clustered_mean_se(va[both] - vb[both], game[both])
    return {"diff_cp": d, "se": se, "lb95": d - 1.96 * se, "ub95": d + 1.96 * se, "rows_both_covered": int(both.sum())}


# ---------------------------------------------------------------- loading


def load_rows(path: Path, mask_spec: str | None = None, *, invert: bool = False) -> dict[str, Any]:
    """The audit rows plus ``_keep`` (the mask applied) and ``_record`` (what was
    read and kept: rows sha, mask spec / sha / polarity, rows kept and dropped)."""
    with np.load(path, allow_pickle=False) as z:
        rows: dict[str, Any] = {k: z[k] for k in z.files}
    for k in ("x", "legal", "game_id"):
        if k not in rows:
            raise SystemExit(f"{path}: audit rows need {k!r}")
    n = int(rows["legal"].shape[0])
    keep = np.ones(n, dtype=bool)
    mask_rec = None
    if mask_spec:
        m = load_array(mask_spec).astype(bool)
        if m.shape != (n,):
            raise SystemExit(f"mask {mask_spec}: shape {m.shape}, rows {n}")
        keep = ~m if invert else m
        mpath = mask_spec.rpartition(":")[0]
        mask_rec = {"spec": mask_spec, "sha256": sha256_file(Path(mpath)), "invert": bool(invert),
                    "true_in_mask": int(m.sum())}
    rows["_keep"] = keep
    rows["_record"] = {"rows_file": str(path), "rows_sha256": sha256_file(Path(path)), "mask": mask_rec,
                       "rows_total": n, "rows_kept": int(keep.sum()), "rows_dropped": int(n - keep.sum())}
    return rows


def load_array(spec: str) -> np.ndarray:
    path, _, key = spec.rpartition(":")
    if not path:
        raise SystemExit(f"want FILE.npz:KEY, got {spec!r}")
    with np.load(path, allow_pickle=False) as z:
        if key not in z.files:
            raise SystemExit(f"{path} has no array {key!r}")
        return z[key]


def named(specs: list[str]) -> list[tuple[str, str]]:
    out = []
    for s in specs:
        name, sep, rest = s.partition("=")
        if not sep or not name:
            raise SystemExit(f"want NAME=VALUE, got {s!r}")
        out.append((name, rest))
    return out


def forward_logits(ckpt: Path, x: np.ndarray, *, device: str = "cpu", batch: int = 64) -> tuple[
        np.ndarray, np.ndarray]:
    import torch

    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    model = load_model_from_checkpoint(Path(ckpt), device=device).eval()
    pol, wdl = [], []
    with torch.no_grad():
        for s in range(0, len(x), batch):
            out = model(torch.from_numpy(np.ascontiguousarray(x[s:s + batch], dtype=np.float32)).to(device))
            pol.append(out["policy_own"].float().cpu().numpy())
            wdl.append(out["wdl"].float().cpu().numpy())
    return np.concatenate(pol), np.concatenate(wdl)


# ---------------------------------------------------------------- commands


def cmd_forward(a: argparse.Namespace) -> int:
    rows = load_rows(Path(a.rows))
    out = Path(a.out)
    rows_sha = rows["_record"]["rows_sha256"]
    have = dict(np.load(out, allow_pickle=False)) if out.exists() else {}
    if have and str(have.get("rows_sha256", "")) != rows_sha:
        raise SystemExit(f"{out} holds priors of other rows (sha {str(have.get('rows_sha256', ''))[:12] or '?'}); "
                         "use a new --out")
    have["rows_sha256"] = np.array(rows_sha)
    for name, ck in named(a.model):
        sha = sha256_file(Path(ck))
        if f"pol_{name}" in have and str(have.get(f"sha_{name}")) == sha:
            continue
        pol, wdl = forward_logits(Path(ck), rows["x"], device=a.device)
        have.update({f"pol_{name}": pol, f"wdl_{name}": wdl, f"sha_{name}": np.array(sha)})
        np.savez_compressed(out, **have)
        print(f"forward {name}: {pol.shape[0]} rows, sha {sha[:12]}", flush=True)
    return 0


def load_priors(path: Path, rows: dict[str, Any]) -> dict[str, np.ndarray]:
    """The priors npz, refused unless it was forwarded on these exact rows."""
    with np.load(path, allow_pickle=False) as z:
        pri = {k: z[k] for k in z.files}
    got = str(pri.get("rows_sha256", ""))
    want = rows["_record"]["rows_sha256"]
    if got != want:
        raise SystemExit(f"{path}: forwarded on rows sha {got[:12] or '<unrecorded>'}, not these rows "
                         f"({want[:12]}); re-run forward")
    return pri


def _ruler_record(spec: str) -> dict[str, Any]:
    name, _, rest = spec.partition("=")
    path, _, key = rest.rpartition(":")
    return {"name": name, "file": path, "key": key, "sha256": sha256_file(Path(path))}


def _rulers(specs: list[str], legal: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for name, spec in named(specs):
        arr = load_array(spec)
        if arr.shape[0] != legal.shape[0]:
            raise SystemExit(f"ruler {name}: {arr.shape[0]} rows, audit set has {legal.shape[0]}")
        move = legal_argmax(arr, legal)
        bad = ~legal[np.arange(legal.shape[0]), move]
        if bad.any():
            raise SystemExit(f"ruler {name}: {int(bad.sum())} rows pick an illegal move")
        out[name] = move
    return out


def cmd_score(a: argparse.Namespace) -> int:
    rows = load_rows(Path(a.rows), a.mask, invert=a.mask_invert)
    keep = rows["_keep"]
    legal, game = rows["legal"][keep], rows["game_id"][keep]
    rulers = {k: v[keep] for k, v in _rulers(a.ruler or [], rows["legal"]).items()}
    pri = load_priors(Path(a.priors), rows)
    pols = {k[4:]: v[keep] for k, v in pri.items() if k.startswith("pol_")}
    wdls = {k[4:]: v[keep] for k, v in pri.items() if k.startswith("wdl_")}
    regrets = {}
    for name, path in named(a.regret or []):
        with np.load(path, allow_pickle=False) as z:
            if z["moves"].shape[0] != rows["legal"].shape[0]:
                raise SystemExit(f"regret {name}: {z['moves'].shape[0]} rows, audit set has {rows['legal'].shape[0]}")
            regrets[name] = {"moves": z["moves"][keep], "regret_cp": z["regret_cp"][keep],
                             "_record": {"file": path, "sha256": sha256_file(Path(path))}}
    if a.nets:
        missing = [n for n in a.nets if n not in pols]
        if missing:
            raise SystemExit(f"priors have no {missing}")
        pols = {n: pols[n] for n in a.nets}
    if a.reference and a.reference not in pols:
        raise SystemExit(f"reference {a.reference!r} not in the priors")
    table: dict[str, Any] = {"rows": int(keep.sum()), "games": int(np.unique(game).size), "nets": {},
                             "input": rows["_record"], "rulers": [_ruler_record(r) for r in a.ruler or []],
                             "rulers_validated": False,
                             "ruler_note": "top-1 columns are descriptive; no ruler here is validated (see validate)",
                             "regret": {k: v["_record"] for k, v in regrets.items()}, "value": {}}
    ref_wl = None
    if a.reference and a.reference in wdls:
        pr = softmax_rows(wdls[a.reference])
        ref_wl = np.abs(pr[:, 0] - pr[:, 2])
    for net, logits in pols.items():
        table["nets"][net] = {}
        for rname, rmove in rulers.items():
            entry = score_one(logits, legal, rmove, game)
            if a.reference and net != a.reference:
                entry["vs_reference"] = paired_top1(logits, pols[a.reference], legal, rmove, game)
            table["nets"][net][rname] = entry
        move = np.argmax(np.where(legal, logits.astype(np.float64), -np.inf), axis=1)
        for gname, tab in regrets.items():
            col = regret_column(move, tab, game)
            if a.reference and net != a.reference:
                ref_move = np.argmax(np.where(legal, pols[a.reference].astype(np.float64), -np.inf), axis=1)
                col["vs_reference"] = paired_regret(move, ref_move, tab, game)
            table["nets"][net][f"regret:{gname}"] = col
        if net in wdls:
            ref_logits = wdls[a.reference] if a.reference and net != a.reference and a.reference in wdls else None
            table["value"][net] = value_shape(wdls[net], game, ref_wl, ref_logits)
    Path(a.out).write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    for net, by in table["nets"].items():
        print(net, " ".join(f"{r}: top1 {v['top1']:.4f}±{v['top1_se']:.4f} gap {v['calibration_gap']:+.3f}"
                            for r, v in by.items() if "top1" in v))
    return 0


def snapshot_candidates(train_dir: Path) -> list[dict[str, Any]]:
    """The cell's snapshots in row order, then its final model."""
    cands = []
    for d in sorted(Path(train_dir, "snapshots").glob("step_*"), key=lambda p: int(p.name.split("_")[1])):
        m = json.loads((d / "metrics.json").read_text())
        cands.append({"name": d.name, "model": str(d / "model.pt"), "update": int(m["update"]),
                      "rows_presented": int(m["rows_presented"])})
    final = Path(train_dir) / "model.pt"
    if not final.is_file():
        raise SystemExit(f"{train_dir} has no final model.pt (train stage not sealed?)")
    summary = json.loads((Path(train_dir) / "metrics.json").read_text()).get("summary", {})
    cands.append({"name": "final", "model": str(final), "update": int(summary.get("updates", -1)),
                  "rows_presented": None})
    return cands


def choose_dose(scores: list[dict[str, Any]], paired: dict[tuple[int, int], dict[str, float]],
                band: tuple[float, float]) -> tuple[int, str]:
    """The plan's rule. ``scores[i]`` has top1 and calibration_gap, in row order
    (final last); ``paired[(i, j)]`` is top1(i) - top1(j). Returns (index, why)."""
    lo, hi = band
    inside = [i for i, s in enumerate(scores) if lo <= s["calibration_gap"] <= hi]
    if not inside:
        dist = [max(lo - s["calibration_gap"], s["calibration_gap"] - hi, 0.0) for s in scores]
        i = int(np.argmin(dist))
        return i, f"no snapshot inside the band [{lo:.4f}, {hi:.4f}]; nearest to it"
    best = max(inside, key=lambda i: scores[i]["top1"])
    tied = [i for i in inside if i == best or abs(paired[(best, i)]["diff"]) <= paired[(best, i)]["se"]]
    pick = min(tied)
    return pick, (f"highest A1 top-1 inside the band is {best}; {len(tied)} within one paired game-clustered SE; "
                  f"earliest is {pick}")


def validate_ruler(ruler_move: np.ndarray, pols: dict[str, np.ndarray], legal: np.ndarray, game: np.ndarray, *,
                   base: str, others: list[str], search_move: np.ndarray, independent: bool) -> dict[str, Any]:
    """The plan's three conditions (Stage 0b). (1) not derived from the SL base's
    net or search (declared); (2) negative control: the base ranks above every
    other net and the last of ``others`` ranks last, a pair within one paired
    game-clustered SE being a tie, not an error; (3) positive control: the base's
    search move agrees with the ruler more than its raw argmax, paired lower
    bound > 0."""
    cond1 = {"pass": bool(independent), "declared_independent_of_base": bool(independent)}
    errors = []
    pairs = {}
    for o in others:
        d = paired_top1(pols[base], pols[o], legal, ruler_move, game)
        pairs[f"{base}-{o}"] = d
        if d["diff"] < 0 and abs(d["diff"]) > d["se"]:
            errors.append(f"{o} ranks above {base}")
    last = others[-1]
    for o in [base, *others[:-1]]:
        d = paired_top1(pols[o], pols[last], legal, ruler_move, game)
        pairs[f"{o}-{last}"] = d
        if d["diff"] < 0 and abs(d["diff"]) > d["se"]:
            errors.append(f"{last} does not rank last ({o} below it)")
    cond2 = {"pass": not errors, "errors": errors, "pairs": pairs}
    raw = policy_stats(pols[base], legal, ruler_move)["agree"]
    srch = (search_move == ruler_move).astype(np.float64)
    diff, se = clustered_mean_se(srch - raw, game)
    cond3 = {"pass": bool(diff - 1.96 * se > 0), "diff": diff, "se": se, "lb95": diff - 1.96 * se,
             "ub95": diff + 1.96 * se}
    return {"conditions": {"1_independent": cond1, "2_negative_control": cond2, "3_positive_control": cond3},
            "passed": bool(cond1["pass"] and cond2["pass"] and cond3["pass"])}


def cmd_validate(a: argparse.Namespace) -> int:
    rows = load_rows(Path(a.rows), a.mask, invert=a.mask_invert)
    keep = rows["_keep"]
    legal, game = rows["legal"][keep], rows["game_id"][keep]
    (rname, rmove), = _rulers([a.ruler], rows["legal"]).items()
    pri = load_priors(Path(a.priors), rows)
    names = [a.base, *a.others]
    missing = [n for n in names if f"pol_{n}" not in pri]
    if missing:
        raise SystemExit(f"priors have no {missing}")
    pols = {n: pri[f"pol_{n}"][keep] for n in names}
    search = legal_argmax(load_array(a.search_move), rows["legal"])[keep]
    res = validate_ruler(rmove[keep], pols, legal, game, base=a.base, others=list(a.others), search_move=search,
                         independent=a.provenance == "independent")
    rec = {"ruler": _ruler_record(a.ruler), "input": rows["_record"], "base": a.base, "others": list(a.others),
           "search_move": {"spec": a.search_move, "sha256": sha256_file(Path(a.search_move.rpartition(":")[0]))},
           "provenance": a.provenance, **res}
    Path(a.out).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(f"validate {rname}: {'PASS' if res['passed'] else 'FAIL'} "
          f"({', '.join(k for k, v in res['conditions'].items() if not v['pass']) or 'all conditions'})")
    return 0


def ruler_validation_status(spec: str, validation: str | None, rows_record: dict[str, Any] | None = None,
                            ) -> tuple[bool, str, dict[str, Any] | None]:
    """Whether ``validation`` records a PASS for this exact ruler file (name, key, sha)
    on the same rows and mask as the dose (``rows_record``: ``load_rows``'s ``_record``)."""
    if not validation:
        return False, "no --ruler-validation given", None
    vpath = Path(validation)
    rec = json.loads(vpath.read_text())
    mine = _ruler_record(spec)
    theirs = rec.get("ruler", {})
    ref = {"file": str(vpath), "sha256": sha256_file(vpath)}
    if any(theirs.get(k) != mine[k] for k in ("name", "key", "sha256")):
        other = f"{theirs.get('name')}, sha {str(theirs.get('sha256'))[:12]}"
        return False, f"the validation record is for another ruler file ({other})", ref
    if rows_record is not None:
        vin = rec.get("input") or {}
        if vin.get("rows_sha256") != rows_record["rows_sha256"]:
            return False, (f"the validation ran on other rows (sha {str(vin.get('rows_sha256'))[:12]}, "
                           f"dose rows {rows_record['rows_sha256'][:12]})"), ref
        vmask, dmask = vin.get("mask"), rows_record["mask"]
        same_mask = (vmask is None and dmask is None) or (
            vmask is not None and dmask is not None
            and all(vmask.get(k) == dmask.get(k) for k in ("sha256", "invert", "spec")))
        if not same_mask:
            return False, f"the validation used another mask ({vmask} vs the dose's {dmask})", ref
    if not rec.get("passed"):
        failed = [k for k, v in rec.get("conditions", {}).items() if not v.get("pass")]
        return False, f"the ruler failed validation ({', '.join(failed)})", ref
    return True, "validated", ref


def cmd_dose(a: argparse.Namespace) -> int:
    cands = snapshot_candidates(Path(a.train_dir))
    out_model = Path(a.out_model)
    record: dict[str, Any] = {"train_dir": str(a.train_dir), "candidates": cands, "band_hi": a.band_hi,
                              "ruler_spec": a.ruler, "ruler_validated": False}
    final_i = len(cands) - 1
    if not a.mask and not a.no_mask:
        raise SystemExit("dose: give --mask (A1 holds rows that reappear in training games; the leak mask) or "
                         "--no-mask explicitly")
    labels_ok = Path(a.ruler.partition("=")[2].rpartition(":")[0]).is_file()
    rows = load_rows(Path(a.rows), a.mask, invert=a.mask_invert) if labels_ok else None
    validated, vwhy, vref = (ruler_validation_status(a.ruler, a.ruler_validation, rows["_record"])
                             if rows is not None else (False, "labels absent", None))
    record.update({"ruler_validated": validated, "ruler_validation": vref, "ruler_validation_reason": vwhy})
    if rows is not None:
        record["input"] = rows["_record"]
    if not labels_ok:
        pick, why = final_i, "the dose ruler's labels have not landed; every cell plays its final model"
    elif not validated:
        record["ruler"] = _ruler_record(a.ruler)
        pick, why = final_i, f"ruler not validated ({vwhy}); final model"
    else:
        assert rows is not None
        record["input"] = rows["_record"]
        keep = rows["_keep"]
        legal, game = rows["legal"][keep], rows["game_id"][keep]
        (rname, rmove), = _rulers([a.ruler], rows["legal"]).items()
        rmove = rmove[keep]
        if a.band_reference:
            (bname, bspec), = named([a.band_reference])
            lo = score_one(load_array(bspec)[keep], legal, rmove, game)["calibration_gap"]
            record["band_reference"] = {"name": bname, "gap": lo}
        elif a.band_lo is not None:
            lo = float(a.band_lo)
        else:
            raise SystemExit("give --band-lo or --band-reference (g_BT4: BT4's raw gap on this ruler)")
        logits = [forward_logits(Path(c["model"]), rows["x"][keep], device=a.device)[0] for c in cands]
        scores = [score_one(lg, legal, rmove, game) for lg in logits]
        paired = {(i, j): paired_top1(logits[i], logits[j], legal, rmove, game)
                  for i in range(len(cands)) for j in range(len(cands)) if i != j}
        pick, why = choose_dose(scores, paired, (lo, float(a.band_hi)))
        record.update({"ruler": {**_ruler_record(a.ruler), "parsed_name": rname}, "band_lo": lo, "scores": scores})
    chosen = cands[pick]
    out_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(chosen["model"], out_model)
    record.update({"played": chosen, "why": why, "played_model": str(out_model),
                   "played_model_sha256": sha256_file(out_model)})
    Path(a.out).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(f"dose: {chosen['name']} ({why})")
    return 0


# ---------------------------------------------------------------- deep-SF labels -> regret table
# The search-compose/scripts/sf_parse.py convention: per (depth, multipv rank) the FIRST
# complete line (bounds skipped); the deepest depth at which every rank is present;
# scores cp from the side to move, mate m -> sign(m) * (10000 - 10 * |m|); a "b"
# searchmoves pass fills moves the MultiPV pass missed; regret = best - move clipped to
# [0, cap].
SF_REGRET_CAP = 1000


def _sf_score(t: list[str]) -> int:
    i = t.index("score")
    kind, v = t[i + 1], int(t[i + 2])
    return v if kind == "cp" else int(np.sign(v) * (10000 - 10 * abs(v)))


def _sf_first_by_depth_rank(lines: list[str]) -> dict[tuple[int, int], tuple[str, int]]:
    d: dict[tuple[int, int], tuple[str, int]] = {}
    for line in lines:
        t = line.split()
        if "lowerbound" in t or "upperbound" in t or "pv" not in t or "multipv" not in t:
            continue
        key = (int(t[t.index("depth") + 1]), int(t[t.index("multipv") + 1]))
        if key not in d:
            d[key] = (t[t.index("pv") + 1], _sf_score(t))
    return d


def _sf_final(d: dict[tuple[int, int], tuple[str, int]], nranks: int) -> dict[int, tuple[str, int]]:
    depths = sorted({k[0] for k in d}, reverse=True)
    for dep in depths:
        rs = {r: d[(dep, r)] for r in range(1, nranks + 1) if (dep, r) in d}
        if len(rs) == nranks:
            return rs
    return {r: v for (dd, r), v in d.items() if dd == depths[0]} if depths else {}


def sf_record_scores(rec: dict[str, Any]) -> tuple[dict[str, int], int]:
    """(uci -> cp from the side to move, best cp) for one sf_label.py record."""
    rs = _sf_final(_sf_first_by_depth_rank(rec["a"]["lines"]), int(rec["multipv_a"]))
    if not rs:
        raise ValueError(f"row {rec.get('row')}: no complete MultiPV line")
    scores = dict(rs.values())
    best = scores[rs[min(rs)][0]]
    if "b" in rec:
        rb = _sf_final(_sf_first_by_depth_rank(rec["b"]["lines"]), len(rec["b"]["searchmoves"]))
        for mv, cp in rb.values():
            scores.setdefault(mv, cp)
    return scores, best


def _row_board(root_fen: str, history: str) -> Any:
    import chess

    board = chess.Board(str(root_fen))
    for mv in str(history).split():
        board.push_uci(mv)
    return board


def cmd_sf_regret_table(a: argparse.Namespace) -> int:
    from chess_anti_engine.moves.encode import FULL_TO_COMPACT_POLICY, move_to_index

    with np.load(a.rows, allow_pickle=False) as z:
        if "root_fen" not in z.files or "history" not in z.files:
            raise SystemExit(f"{a.rows}: the converter needs root_fen and history to rebuild each row's board")
        root_fen, history = z["root_fen"], z["history"]
    by_fen: dict[str, dict[str, Any]] = {}
    dup = 0
    with open(a.sf_jsonl) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["fen"] in by_fen:
                dup += 1
            by_fen[rec["fen"]] = rec
    n = int(root_fen.shape[0])
    k = int(a.max_moves)
    moves = np.full((n, k), -1, dtype=np.int64)
    regret = np.zeros((n, k), dtype=np.float32)
    labelled = truncated = 0
    for i in range(n):
        board = _row_board(root_fen[i], history[i])
        rec = by_fen.get(board.fen())
        if rec is None:
            continue
        scores, best = sf_record_scores(rec)
        legal = {m.uci(): m for m in board.legal_moves}
        items = sorted(((best - cp, mv) for mv, cp in scores.items() if mv in legal))
        truncated += max(0, len(items) - k)
        for j, (r, mv) in enumerate(items[:k]):
            moves[i, j] = int(FULL_TO_COMPACT_POLICY[move_to_index(legal[mv], board)])
            regret[i, j] = float(min(max(r, 0), int(a.cap)))
        labelled += 1
    np.savez_compressed(a.out, moves=moves, regret_cp=regret)
    meta = {"rows_file": str(a.rows), "rows_sha256": sha256_file(Path(a.rows)), "sf_jsonl": str(a.sf_jsonl),
            "sf_jsonl_sha256": sha256_file(Path(a.sf_jsonl)), "rows": n, "rows_labelled": labelled,
            "duplicate_fens_in_labels": dup, "moves_truncated": truncated, "cap_cp": int(a.cap),
            "convention": "search-compose sf_parse.py: first line per (depth, rank), deepest complete depth, "
                          "mate m -> sign(m)*(10000-10|m|), searchmoves pass fills, regret clipped to [0, cap]"}
    Path(str(a.out) + ".json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(f"sf-regret-table: {labelled}/{n} rows labelled -> {a.out}")
    return 0


def _fill_forced(board: Any, ply: int) -> bool:
    """Push single-legal-move plies (which store no row) up to ``ply``; False when
    a gap is not forced (its move is then unknown)."""
    while len(board.move_stack) < ply:
        moves = list(board.legal_moves)
        if len(moves) != 1:
            return False
        board.push(moves[0])
    return len(board.move_stack) == ply


def _position_key(board: Any) -> str:
    """Board, side to move, castling and en passant (the plan's P2 key)."""
    return " ".join(board.fen().split(" ")[:4])


def cmd_a2_sample(a: argparse.Namespace) -> int:
    import chess

    from chess_anti_engine.moves.encode import index_to_move_for_encoding, legal_move_mask_for_encoding
    from chess_anti_engine.simple_rl.collect import load_generation_rows
    from chess_anti_engine.simple_rl.common import resolve_collect_dir

    collect = Path(a.collect_dir)
    manifest = json.loads((collect / "manifest.json").read_text())
    train_jsonl = Path(a.openings) if a.openings else Path(manifest["inputs"]["train_openings"]["path"])
    line_of = {json.loads(ln)["id"]: json.loads(ln)["line"] for ln in train_jsonl.read_text().splitlines() if ln}
    games = {int(g["game_id"]): g for g in map(json.loads, (collect / "games.jsonl").read_text().splitlines())}
    arrays, meta = load_generation_rows(collect)
    order = np.lexsort((meta["ply_index"], meta["game_id"]))
    exclude: set[str] = set()
    unkeyed = 0
    excluded_sources = []
    for d in a.exclude_collect or []:
        keys, gaps = _collect_position_keys(Path(d))
        exclude |= keys
        unkeyed += gaps
        excluded_sources.append({"collect": str(d), "manifest_sha256": sha256_file(
            resolve_collect_dir(Path(d)) / "manifest.json"), "keys": len(keys)})
    for f in a.exclude_rows or []:
        keys = rows_position_keys(Path(f))
        exclude |= keys
        excluded_sources.append({"rows": str(f), "sha256": sha256_file(Path(f)), "keys": len(keys)})
    rng = np.random.default_rng(int(a.seed))
    picked: list[int] = []
    fens, roots, hists, keys = [], [], [], []
    dropped_excluded = 0
    truncated = 0
    gid_all = meta["game_id"][order]
    for gid in np.unique(gid_all):
        idx = order[gid_all == gid]
        g = games[int(gid)]
        fen, _, moves = line_of[g["opening_id"]].partition("|")
        board = chess.Board(fen.strip())
        history = moves.split()
        for mv in history:
            board.push_uci(mv)
        states = []
        for i in idx:
            if not _fill_forced(board, int(meta["ply_index"][i])):
                # A ply stored no row and was not forced: its move is unknown, so stop this game here.
                truncated += 1
                break
            legal = legal_move_mask_for_encoding(board).astype(bool)
            if not np.array_equal(legal, arrays["legal_mask"][i].astype(bool)):
                raise SystemExit(f"game {gid} ply {len(board.move_stack)}: replayed legal moves differ from the row's")
            states.append((int(i), board.fen(), [m.uci() for m in board.move_stack], _position_key(board)))
            if not arrays["has_played_move"][i]:
                break
            board.push(index_to_move_for_encoding(int(arrays["played_move_index"][i]), board))
        ok = [s for s in states if s[3] not in exclude]
        dropped_excluded += len(states) - len(ok)
        for j in sorted(rng.choice(len(ok), size=min(int(a.per_game), len(ok)), replace=False).tolist()) if ok else []:
            i, f, h, k = ok[j]
            picked.append(i)
            fens.append(f)
            roots.append(chess.Board(line_of[g["opening_id"]].partition("|")[0].strip()).fen())
            hists.append(" ".join(h))
            keys.append(k)
    sel = np.asarray(picked, dtype=np.int64)
    np.savez_compressed(Path(a.out), x=arrays["x"][sel].astype(np.float16), legal=arrays["legal_mask"][sel].astype(bool),
                        game_id=meta["game_id"][sel].astype(np.int64), ply=meta["ply_index"][sel].astype(np.int64),
                        fen=np.asarray(fens), root_fen=np.asarray(roots), history=np.asarray(hists),
                        position_key=np.asarray(keys), source_row=sel)
    summary = {"collect": str(collect), "collect_manifest_sha256": sha256_file(collect / "manifest.json"),
               "rows": int(sel.size), "games": int(np.unique(meta["game_id"][sel]).size), "per_game": int(a.per_game),
               "seed": int(a.seed), "excluded_collects": list(a.exclude_collect or []),
               "excluded_sources": excluded_sources,
               "rows_dropped_as_training_positions": dropped_excluded,
               "games_truncated_at_a_row_gap": truncated,
               "training_positions_unkeyed_after_a_row_gap": unkeyed,
               "out_sha256": sha256_file(Path(a.out))}
    Path(str(a.out) + ".json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary))
    return 0


def rows_position_keys(path: Path) -> set[str]:
    """Position keys of an audit-rows file that carries ``root_fen`` + ``history``
    (A1 or an earlier A2), replayed: a transposition maps to the same key."""
    import chess

    with np.load(path, allow_pickle=False) as z:
        if "root_fen" not in z.files or "history" not in z.files:
            raise SystemExit(f"{path}: --exclude-rows needs root_fen and history")
        roots, hists = z["root_fen"], z["history"]
    keys = set()
    for root, hist in zip(roots, hists, strict=True):
        b = chess.Board(str(root))
        for mv in str(hist).split():
            b.push_uci(mv)
        keys.add(_position_key(b))
    return keys


def _collect_position_keys(collect: Path) -> tuple[set[str], int]:
    """Every position key a training collect's rows replay to, and the number of
    rows after a row gap (their positions cannot be keyed; reported)."""
    import chess

    from chess_anti_engine.moves.encode import index_to_move_for_encoding
    from chess_anti_engine.simple_rl.collect import load_generation_rows
    from chess_anti_engine.simple_rl.common import resolve_collect_dir

    collect = resolve_collect_dir(collect)
    manifest = json.loads((collect / "manifest.json").read_text())
    train_jsonl = Path(manifest["inputs"]["train_openings"]["path"])
    line_of = {json.loads(ln)["id"]: json.loads(ln)["line"] for ln in train_jsonl.read_text().splitlines() if ln}
    games = {int(g["game_id"]): g for g in map(json.loads, (collect / "games.jsonl").read_text().splitlines())}
    arrays, meta = load_generation_rows(collect, keys=("x", "policy_target", "played_move_index", "has_played_move"))
    order = np.lexsort((meta["ply_index"], meta["game_id"]))
    keys: set[str] = set()
    gaps = 0
    gid_all = meta["game_id"][order]
    for gid in np.unique(gid_all):
        fen, _, moves = line_of[games[int(gid)]["opening_id"]].partition("|")
        board = chess.Board(fen.strip())
        for mv in moves.split():
            board.push_uci(mv)
        idx = order[gid_all == gid]
        for n, i in enumerate(idx):
            if not _fill_forced(board, int(meta["ply_index"][i])):
                gaps += len(idx) - n
                break
            keys.add(_position_key(board))
            if not arrays["has_played_move"][i]:
                break
            board.push(index_to_move_for_encoding(int(arrays["played_move_index"][i]), board))
    return keys, gaps


def cmd_a2_openings(a: argparse.Namespace) -> int:
    lines = [ln for ln in Path(a.eval_jsonl).read_text().splitlines() if ln.strip()]
    lo, _, hi = str(a.lines).partition(":")
    block = lines[int(lo):int(hi)]
    if len(block) != int(hi) - int(lo):
        raise SystemExit(f"{a.eval_jsonl} has {len(lines)} lines; the block {a.lines} does not fit")
    out = Path(a.out_dir)
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.mkdir(parents=True)
    rows = [json.loads(ln) for ln in block]
    # One book: every a2 line is played once (book 1, --book-mix 1.0, cycle sampling).
    (out / "train.jsonl").write_text("".join(json.dumps({**r, "book": 1}, sort_keys=True) + "\n" for r in rows))
    rec = {"eval_jsonl": str(a.eval_jsonl), "eval_sha256": sha256_file(Path(a.eval_jsonl)), "lines": a.lines,
           "count": len(rows), "train_sha256": sha256_file(out / "train.jsonl")}
    (out / "a2_block.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(json.dumps(rec))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threads", type=int, default=2)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("forward")
    f.add_argument("--rows", required=True)
    f.add_argument("--model", action="append", required=True, metavar="NAME=CKPT")
    f.add_argument("--out", required=True)
    f.add_argument("--device", default="cpu")
    s = sub.add_parser("score")
    s.add_argument("--rows", required=True)
    s.add_argument("--priors", required=True)
    s.add_argument("--ruler", action="append", default=[], metavar="NAME=FILE.npz:KEY")
    s.add_argument("--regret", action="append", default=[], metavar="NAME=FILE.npz",
                   help="deep-SF regret table (moves, regret_cp); reported, never a selector")
    s.add_argument("--nets", nargs="*")
    s.add_argument("--reference", default=None)
    s.add_argument("--mask", default=None, metavar="FILE.npz:KEY", help="keep rows where true")
    s.add_argument("--mask-invert", action="store_true", help="keep rows where the mask is false (e.g. leak)")
    s.add_argument("--out", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--rows", required=True, help="A1 (and A2) rows")
    v.add_argument("--priors", required=True)
    v.add_argument("--ruler", required=True, metavar="NAME=FILE.npz:KEY")
    v.add_argument("--provenance", required=True, choices=("independent", "base-derived"),
                   help="condition 1: is the ruler free of the SL base's own net and search?")
    v.add_argument("--base", required=True, help="the SL base's name in the priors")
    v.add_argument("--others", nargs="+", required=True,
                   help="nets the base must outrank; the LAST must rank last (plan: P1c12 F12c12 Q50c12)")
    v.add_argument("--search-move", required=True, metavar="FILE.npz:KEY", help="the base's 64-sim search move")
    v.add_argument("--mask", default=None)
    v.add_argument("--mask-invert", action="store_true")
    v.add_argument("--out", required=True)
    d = sub.add_parser("dose")
    d.add_argument("--train-dir", required=True, help="<cell run>/gen_0001/train")
    d.add_argument("--rows", required=True, help="A1 rows")
    d.add_argument("--ruler", required=True, metavar="NAME=FILE.npz:KEY")
    d.add_argument("--band-lo", type=float, default=None)
    d.add_argument("--band-reference", default=None, metavar="NAME=FILE.npz:KEY",
                   help="a policy whose gap on this ruler is the band's lower edge (BT4 raw: g_BT4)")
    d.add_argument("--band-hi", type=float, default=0.02)
    d.add_argument("--mask", default=None)
    d.add_argument("--mask-invert", action="store_true")
    d.add_argument("--no-mask", action="store_true", help="score every row (refused without this or --mask)")
    d.add_argument("--ruler-validation", default=None, metavar="VALIDATION.json",
                   help="a validate record; without a PASS for this exact ruler file the final model plays")
    d.add_argument("--device", default="cpu")
    d.add_argument("--out", required=True, help="dose.json")
    d.add_argument("--out-model", required=True, help="the played model (a copy, sha recorded)")
    t = sub.add_parser("sf-regret-table", help="deep-SF label jsonl -> the --regret table for these rows")
    t.add_argument("--rows", required=True, help="audit rows npz with root_fen and history")
    t.add_argument("--sf-jsonl", required=True, help="sf_label.py output (e.g. search-compose/sf_deep8M.jsonl)")
    t.add_argument("--cap", type=int, default=SF_REGRET_CAP)
    t.add_argument("--max-moves", type=int, default=64, help="K: moves kept per row, lowest regret first")
    t.add_argument("--out", required=True)
    ao = sub.add_parser("a2-openings")
    ao.add_argument("--eval-jsonl", required=True)
    ao.add_argument("--lines", default="800:1800", help="half-open line range of the block (plan: a2 = 800:1800)")
    ao.add_argument("--out-dir", required=True)
    a2 = sub.add_parser("a2-sample")
    a2.add_argument("--collect-dir", required=True, help="sealed collect of SL-base games from the a2 block")
    a2.add_argument("--openings", default=None, help="the openings jsonl it played (default: its manifest's)")
    a2.add_argument("--per-game", type=int, default=3)
    a2.add_argument("--seed", type=int, default=2026112500)
    a2.add_argument("--exclude-collect", action="append", default=[],
                    help="a training collect dir; positions it played are dropped (P2)")
    a2.add_argument("--exclude-rows", action="append", default=[],
                    help="an audit-rows npz with root_fen + history (A1, an earlier A2); its positions are dropped")
    a2.add_argument("--out", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    import torch

    torch.set_num_threads(max(1, int(a.threads)))
    return {"forward": cmd_forward, "score": cmd_score, "validate": cmd_validate, "dose": cmd_dose,
            "a2-openings": cmd_a2_openings, "a2-sample": cmd_a2_sample,
            "sf-regret-table": cmd_sf_regret_table}[a.command](a)


if __name__ == "__main__":
    raise SystemExit(main())
