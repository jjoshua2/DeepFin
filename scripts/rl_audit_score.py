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
             game-clustered SE; paired top-1 difference vs --reference
  dose       the plan's dose rule over one cell's snapshots + final model
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


# ---------------------------------------------------------------- loading


def load_rows(path: Path, mask_spec: str | None = None, *, invert: bool = False) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        rows = {k: z[k] for k in z.files}
    for k in ("x", "legal", "game_id"):
        if k not in rows:
            raise SystemExit(f"{path}: audit rows need {k!r}")
    keep = np.ones(rows["legal"].shape[0], dtype=bool)
    if mask_spec:
        m = load_array(mask_spec).astype(bool)
        keep = ~m if invert else m
    rows["_keep"] = keep
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
    have = dict(np.load(out, allow_pickle=False)) if out.exists() else {}
    for name, ck in named(a.model):
        sha = sha256_file(Path(ck))
        if f"pol_{name}" in have and str(have.get(f"sha_{name}")) == sha:
            continue
        pol, wdl = forward_logits(Path(ck), rows["x"], device=a.device)
        have.update({f"pol_{name}": pol, f"wdl_{name}": wdl, f"sha_{name}": np.array(sha)})
        np.savez_compressed(out, **have)
        print(f"forward {name}: {pol.shape[0]} rows, sha {sha[:12]}", flush=True)
    return 0


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
    rulers = {k: v[keep] for k, v in _rulers(a.ruler, rows["legal"]).items()}
    with np.load(a.priors, allow_pickle=False) as z:
        pols = {k[4:]: z[k][keep] for k in z.files if k.startswith("pol_")}
    if a.nets:
        missing = [n for n in a.nets if n not in pols]
        if missing:
            raise SystemExit(f"priors have no {missing}")
        pols = {n: pols[n] for n in a.nets}
    if a.reference and a.reference not in pols:
        raise SystemExit(f"reference {a.reference!r} not in the priors")
    table: dict[str, Any] = {"rows": int(keep.sum()), "games": int(np.unique(game).size), "nets": {}}
    for net, logits in pols.items():
        table["nets"][net] = {}
        for rname, rmove in rulers.items():
            entry = score_one(logits, legal, rmove, game)
            if a.reference and net != a.reference:
                entry["vs_reference"] = paired_top1(logits, pols[a.reference], legal, rmove, game)
            table["nets"][net][rname] = entry
    Path(a.out).write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    for net, by in table["nets"].items():
        print(net, " ".join(f"{r}: top1 {v['top1']:.4f}±{v['top1_se']:.4f} gap {v['calibration_gap']:+.3f}"
                            for r, v in by.items()))
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


def cmd_dose(a: argparse.Namespace) -> int:
    cands = snapshot_candidates(Path(a.train_dir))
    out_model = Path(a.out_model)
    record: dict[str, Any] = {"train_dir": str(a.train_dir), "candidates": cands, "band_hi": a.band_hi}
    final_i = len(cands) - 1
    labels_ok = all(Path(spec.partition("=")[2].rpartition(":")[0]).is_file() for spec in [a.ruler])
    if not labels_ok:
        pick, why = final_i, "the dose ruler's labels have not landed; every cell plays its final model"
    else:
        rows = load_rows(Path(a.rows), a.mask, invert=a.mask_invert)
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
        record.update({"ruler": rname, "band_lo": lo, "scores": scores})
    chosen = cands[pick]
    out_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(chosen["model"], out_model)
    record.update({"played": chosen, "why": why, "played_model": str(out_model),
                   "played_model_sha256": sha256_file(out_model)})
    Path(a.out).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(f"dose: {chosen['name']} ({why})")
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

    collect = Path(a.collect_dir)
    manifest = json.loads((collect / "manifest.json").read_text())
    train_jsonl = Path(a.openings) if a.openings else Path(manifest["inputs"]["train_openings"]["path"])
    line_of = {json.loads(ln)["id"]: json.loads(ln)["line"] for ln in train_jsonl.read_text().splitlines() if ln}
    games = {int(g["game_id"]): g for g in map(json.loads, (collect / "games.jsonl").read_text().splitlines())}
    arrays, meta = load_generation_rows(collect)
    order = np.lexsort((meta["ply_index"], meta["game_id"]))
    exclude: set[str] = set()
    unkeyed = 0
    for d in a.exclude_collect or []:
        keys, gaps = _collect_position_keys(Path(d))
        exclude |= keys
        unkeyed += gaps
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
               "rows_dropped_as_training_positions": dropped_excluded,
               "games_truncated_at_a_row_gap": truncated,
               "training_positions_unkeyed_after_a_row_gap": unkeyed,
               "out_sha256": sha256_file(Path(a.out))}
    Path(str(a.out) + ".json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary))
    return 0


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
    s.add_argument("--ruler", action="append", required=True, metavar="NAME=FILE.npz:KEY")
    s.add_argument("--nets", nargs="*")
    s.add_argument("--reference", default=None)
    s.add_argument("--mask", default=None, metavar="FILE.npz:KEY", help="keep rows where true")
    s.add_argument("--mask-invert", action="store_true", help="keep rows where the mask is false (e.g. leak)")
    s.add_argument("--out", required=True)
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
    d.add_argument("--device", default="cpu")
    d.add_argument("--out", required=True, help="dose.json")
    d.add_argument("--out-model", required=True, help="the played model (a copy, sha recorded)")
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
    a2.add_argument("--out", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    import torch

    torch.set_num_threads(max(1, int(a.threads)))
    return {"forward": cmd_forward, "score": cmd_score, "dose": cmd_dose, "a2-openings": cmd_a2_openings,
            "a2-sample": cmd_a2_sample}[a.command](a)


if __name__ == "__main__":
    raise SystemExit(main())
