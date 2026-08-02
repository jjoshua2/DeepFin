#!/usr/bin/env python3
"""Offline policy-absorption screen from the banked iter-478 state.

Held-out ruler: a FULL DETERMINISTIC PASS (production's `_iter_full_pass_batches`)
over a game-disjoint, desync-exactly-zero eval set, dumping PER-ROW policy CE and
target entropy so arm-vs-arm differences get a game-clustered CI.

Both held-out properties are ENFORCED here rather than assumed of the manifest:
`EvalSet` aborts on any desync row (and on an empty label denominator), and
`assert_pools_game_disjoint` intersects the train pool's game ids with each eval
set's. The shard-linking and row-count reconciliation exist for the same reason --
see `link_name` and `assert_pool_loaded`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.model import build_model
from chess_anti_engine.replay.buffer import ArrayReplayBuffer
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.replay.shard import (
    LOCAL_SHARD_SUFFIX,
    iter_shard_paths,
    load_shard_arrays,
    shard_index,
)
from chess_anti_engine.train.losses import (
    align_policy_target,
    apply_policy_mask_to_logits,
    normalize_distribution,
    soft_cross_entropy,
)
from chess_anti_engine.train.trainer import Trainer, trainer_kwargs_from_config
from chess_anti_engine.tune.trial_config import TrialConfig
from chess_anti_engine.uci.model_loader import model_config_from_arch
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file

_EPS = 1e-12


def _entropy(p: torch.Tensor) -> torch.Tensor:
    q = p.clamp_min(0.0)
    lg = torch.where(q > 0, torch.log(q.clamp_min(_EPS)), torch.zeros_like(q))
    return -(q * lg).sum(dim=-1)


def _flag(arrs: dict[str, Any], key: str, n: int) -> np.ndarray:
    """A shard's boolean has-flag, with an absent column meaning all-False."""
    v = arrs.get(key)
    return np.zeros(n, bool) if v is None else np.asarray(v).astype(bool)


class EvalSet:
    """Fixed held-out rows + their game ids, in the buffer's own row order.

    The desync gate lives in the constructor on purpose: a contaminated eval set
    must be impossible to *hold*, not merely impossible to pass to the one call
    site that remembered to check it. Removing the check therefore means editing
    the class the tests cover, not deleting a line from ``main``.
    """

    def __init__(self, name: str, paths: list[str], planes: int):
        self.name = name
        buf = ArrayReplayBuffer(10**9, rng=np.random.default_rng(0))
        gids: list[np.ndarray] = []
        lab = des = 0
        for p in paths:
            arrs, _meta = load_shard_arrays(p, lazy=False)
            n = int(np.asarray(arrs["x"]).shape[0])
            if n == 0:
                continue
            stored = int(np.asarray(arrs["x"]).shape[1])
            if stored != planes:
                raise SystemExit(f"{p}: {stored} planes, model wants {planes}")
            # An ABSENT has-flag is not an error and is not "clean": the shard
            # writer omits the column when no row carries the field, so a
            # pre-MultiPV shard has no `has_sf_multipv_raw` at all and every one
            # of its labelled rows is desynced. Reading absence as all-False
            # states that; `arrs[...]` raised a bare KeyError instead.
            hw = _flag(arrs, "has_sf_wdl", n)
            hm = _flag(arrs, "has_sf_multipv_raw", n)
            lab += int(hw.sum())
            des += int((hw & ~hm).sum())
            gids.append(np.asarray(arrs["game_id"], dtype=np.int64))
            buf.add_many_arrays(arrs)
        self.buf = buf
        self.game_id = np.concatenate(gids) if gids else np.zeros(0, np.int64)
        self.labelled = lab
        self.desync_rows = des
        # NaN, not 0.0, when nothing is labelled: a fraction with an empty
        # denominator is undefined, and 0.0 would read as "clean" on the one
        # input the gate exists to reject.
        self.desync_frac = des / lab if lab else float("nan")
        if len(buf) != self.game_id.size:
            raise SystemExit(f"{name}: buffer rows {len(buf)} != game ids {self.game_id.size}")
        self.gate()

    def report(self) -> str:
        return (f"[eval:{self.name}] rows={len(self.buf)} games={np.unique(self.game_id).size} "
                f"labelled={self.labelled} desync_rows={self.desync_rows} "
                f"DESYNC_FRAC={self.desync_frac:.6f}")

    def gate(self) -> None:
        """Print the desync reading, then abort unless it is exactly zero.

        Zero labelled rows ABORTS. ``des / lab if lab else 0.0`` used to make an
        eval set with no SF labels sail through the gate whose whole purpose is
        to reject contaminated eval sets -- the denominator being empty is the
        loudest possible sign that the reading means nothing.
        """
        print(self.report(), flush=True)
        if self.labelled == 0:
            raise SystemExit(
                f"eval set {self.name}: 0 labelled rows -- the desync gate has no denominator "
                f"and cannot certify anything")
        if self.desync_rows != 0:
            raise SystemExit(
                f"eval set {self.name}: {self.desync_rows} desync rows "
                f"(frac {self.desync_frac:.6f}) is not exactly 0")


def full_pass_rows(trainer: Trainer, ev: EvalSet, batch_size: int) -> dict[str, np.ndarray]:
    """Per-row policy CE / target entropy / argmax agreement over EVERY eval row."""
    model = trainer.model
    was = model.training
    model.eval()
    ce_l: list[np.ndarray] = []
    fl_l: list[np.ndarray] = []
    ag_l: list[np.ndarray] = []
    base_l: list[np.ndarray] = []
    temps = [0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    tsum = dict.fromkeys(temps, 0.0)
    tden = 0.0
    seen = 0
    with torch.no_grad():
        for b in trainer._iter_full_pass_batches(cast(Any, ev.buf), batch_size=batch_size):
            out = model(b["x"], relations=b["relations"]) if b.get("relations") is not None else model(b["x"])
            lg = out["policy"] if "policy" in out else out["policy_own"]
            w = int(lg.shape[-1])
            nm = b.get("is_network_turn")
            nm = torch.ones(lg.shape[0], device=lg.device) if nm is None else nm.to(torch.float32)
            hp = b.get("has_policy")
            hp = torch.ones_like(nm) if hp is None else hp.to(torch.float32)
            base = nm * hp
            tgt = align_policy_target(b["policy_t"], w)
            ml = apply_policy_mask_to_logits(lg, b, "legal_mask", "has_legal_mask")
            tn = normalize_distribution(tgt).float()
            ce_l.append(soft_cross_entropy(ml, tgt).float().cpu().numpy())
            fl_l.append(_entropy(tn).float().cpu().numpy())
            ag_l.append((ml.float().argmax(-1) == tn.argmax(-1)).float().cpu().numpy())
            base_l.append(base.float().cpu().numpy())
            seen += int(lg.shape[0])
            bf = base.float()
            tden += float(bf.sum())
            for t in temps:
                tsum[t] += float((soft_cross_entropy(ml / t, tgt).float() * bf).sum())
    if was:
        model.train()
    if seen != len(ev.buf):
        raise SystemExit(f"full pass covered {seen} of {len(ev.buf)} rows")
    return {"ce": np.concatenate(ce_l), "floor": np.concatenate(fl_l),
            "agree": np.concatenate(ag_l), "base": np.concatenate(base_l),
            "game_id": ev.game_id,
            "temps": np.asarray(temps, dtype=np.float64),
            "temp_ce": np.asarray([tsum[t] / max(tden, 1.0) for t in temps], dtype=np.float64)}


def summarize(r: dict[str, np.ndarray]) -> dict[str, Any]:
    m = r["base"] > 0
    ce, fl, ag = r["ce"][m], r["floor"][m], r["agree"][m]
    out: dict[str, Any] = {"rows": int(m.sum()), "ce": float(ce.mean()), "floor": float(fl.mean()),
           "excess": float((ce - fl).mean()), "agree": float(ag.mean())}
    if "temp_ce" in r:
        tc = np.asarray(r["temp_ce"])
        tt = np.asarray(r["temps"])
        out["temp_ce"] = {float(t): float(v) for t, v in zip(tt, tc)}
        out["best_temp"] = float(tt[int(np.argmin(tc))])
        out["best_temp_ce"] = float(tc.min())
    return out


class ShuffleTargets:
    """Negative control: permute policy_target across rows inside every batch."""

    def __init__(self, inner: Any, rng: np.random.Generator):
        self._inner = inner
        self._rng = rng
        self.rng = inner.rng

    def sample_batch_arrays(self, batch_size: int, *, wdl_balance: bool = True) -> dict[str, np.ndarray]:
        a = self._inner.sample_batch_arrays(batch_size, wdl_balance=wdl_balance)
        out = dict(a)
        pt = np.asarray(out["policy_target"])
        perm = self._rng.permutation(pt.shape[0])
        while pt.shape[0] > 1 and np.any(perm == np.arange(pt.shape[0])):
            perm = self._rng.permutation(pt.shape[0])
        out["policy_target"] = pt[perm]
        return out

    def __getattr__(self, k: str) -> Any:
        return getattr(self._inner, k)


def link_name(src: Path) -> str:
    """A link name unique to the SOURCE path that the buffer still discovers.

    Shard indices are REUSED across salvage bundles with different contents --
    ``shard_033118.zarr`` exists in ``post_quarantine_20260801`` with 2000 rows
    and in ``pre_sf_reallocation_ck148_20260722`` with 155 -- so the basename is
    not a key. Linking by basename dropped 707 of A1_BIG's 5,244 shards
    (1,331,642 rows, 13.6% of the designed pool) while the screen went on
    printing the manifest's count.

    Two constraints fix the shape of the name. ``iter_shard_paths`` globs
    ``shard_*.zarr``, so the ``shard_`` prefix is mandatory or the shard is not
    found at all; and ``shard_index`` reads ``stem.split("_")[1]``, so the index
    must stay in the second field. The index is zero-padded to a fixed width so
    the glob's lexicographic sort stays index-monotonic (recency order), and the
    digest of the resolved source disambiguates reused indices.
    """
    idx = shard_index(src)
    if idx < 0:
        raise SystemExit(f"{src}: name does not parse as shard_<index>{LOCAL_SHARD_SUFFIX}")
    h = hashlib.blake2b(str(src).encode(), digest_size=5).hexdigest()
    return f"shard_{idx:09d}_{h}{LOCAL_SHARD_SUFFIX}"


def link_dir(paths: list[str], dst: Path) -> tuple[Path, dict[str, Path]]:
    """Symlink every manifest shard into *dst* under a source-unique name.

    Returns the directory and the ``{link name: resolved source}`` map, which the
    caller reconciles against what the buffer actually loaded. Repeated identical
    paths collapse to one link (they are one shard); two DIFFERENT sources landing
    on one name aborts.
    """
    dst.mkdir(parents=True, exist_ok=True)
    linked: dict[str, Path] = {}
    for p in paths:
        src = Path(p).resolve()
        name = link_name(src)
        prev = linked.get(name)
        if prev is not None:
            if prev != src:
                raise SystemExit(f"link name {name} claimed by both {prev} and {src}")
            continue
        linked[name] = src
        link = dst / name
        if link.is_symlink():
            if Path(os.readlink(link)).resolve() != src:
                raise SystemExit(f"stale link {link} -> {os.readlink(link)}, expected {src}")
            continue
        if link.exists():
            raise SystemExit(f"{link} already exists and is not a link to {src}")
        os.symlink(str(src), str(link), target_is_directory=True)
    return dst, linked


def read_pool_game_ids(paths: list[str]) -> tuple[np.ndarray, dict[Path, int]]:
    """Every train-pool game id, plus per-source row counts.

    The row counts come free with the game-id read and are what the buffer's own
    row total is checked against, so the manifest's arithmetic never gets
    reported in place of the buffer's.
    """
    gids: list[np.ndarray] = []
    rows: dict[Path, int] = {}
    for p in paths:
        src = Path(p).resolve()
        if src in rows:
            continue
        arrs, _meta = load_shard_arrays(src, lazy=True)
        if "game_id" not in arrs:
            raise SystemExit(f"{src}: no game_id column -- the train/eval leak gate cannot be enforced")
        g = np.asarray(arrs["game_id"][:], dtype=np.int64)
        rows[src] = int(g.size)
        gids.append(g)
    return (np.concatenate(gids) if gids else np.zeros(0, np.int64)), rows


def assert_pools_game_disjoint(train_game_id: np.ndarray, evs: list[EvalSet]) -> None:
    """Abort if any eval game also appears in the train pool.

    Game disjointness is the property the whole ruler rests on -- a per-row split
    leaks ~23 sibling positions per game -- and nothing in this script used to
    check it. It was a property of a manifest builder that never shipped, so a
    hand-built manifest could leak freely behind a green desync line.
    """
    tr = np.unique(np.asarray(train_game_id, dtype=np.int64))
    for e in evs:
        ev = np.unique(e.game_id)
        shared = np.intersect1d(tr, ev, assume_unique=True)
        print(f"[screen] game-leak {e.name}: train_games={tr.size} eval_games={ev.size} "
              f"SHARED={shared.size}", flush=True)
        if shared.size:
            raise SystemExit(
                f"train pool shares {shared.size} game(s) with eval set {e.name} "
                f"(e.g. {shared[:10].tolist()}) -- the held-out ruler is leaking")


def prepare_train_pool(
    paths: list[str], evs: list[EvalSet], dst: Path,
) -> tuple[Path, dict[str, Path], dict[Path, int]]:
    """Vet the train pool against the eval sets, THEN link it.

    One function, for the same reason the desync gate lives in ``EvalSet``'s
    constructor: ``main`` has no other way to obtain a link directory, so the
    leak gate cannot be dropped from the call site while the pool still gets
    built. The order matters too -- a leaking pool aborts before anything is
    linked, so a failed run leaves no half-built directory to be reused.
    """
    tr_gid, rows_by_src = read_pool_game_ids(paths)
    assert_pools_game_disjoint(tr_gid, evs)
    tdir, linked = link_dir(paths, dst)
    return tdir, linked, rows_by_src


def assert_pool_loaded(
    linked: dict[str, Path], rows_by_src: dict[Path, int],
    found_links: list[Path], buf_rows: int,
) -> int:
    """Reconcile what the manifest asked for against what the buffer holds.

    Returns the reconciled row count. Every count this script prints or banks
    must come from here, never from ``len(manifest_paths)``.
    """
    if len(found_links) != len(linked):
        raise SystemExit(
            f"linked {len(linked)} shards but the buffer's glob found {len(found_links)}")
    expect = sum(rows_by_src[s] for s in linked.values())
    if buf_rows != expect:
        raise SystemExit(
            f"buffer holds {buf_rows} rows but the linked shards carry {expect} "
            f"({expect - buf_rows} rows never entered the pool)")
    return expect


def run_training(
    trainer: Trainer, sampler: Any, *,
    steps: int, chunk_steps: int, batch_size: int, label: str,
) -> Any:
    """Run *steps* optimizer steps in *chunk_steps* chunks, and PROVE they ran.

    ``trainer.step`` is the optimizer's own counter. If the training call were a
    no-op every arm would report exactly zero degradation and nothing else in
    this script would notice -- the screen's central number would be produced by
    not training. Requiring the counter to advance by exactly *steps* makes that
    failure loud instead of publishable.
    """
    before = int(trainer.step)
    tm = None
    left = int(steps)
    while left > 0:
        c = min(int(chunk_steps), left)
        tm = trainer.train_steps(sampler, batch_size=int(batch_size), steps=c)
        left -= c
    advanced = int(trainer.step) - before
    if advanced != int(steps):
        raise SystemExit(
            f"{label}: trainer.step advanced {advanced}, expected {steps} -- training did not happen")
    return tm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/pbt2_small.yaml"))
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--train-pools", default="train_recent", help="comma-separated manifest keys")
    ap.add_argument("--eval-sets", default="evalA,evalB")
    ap.add_argument("--label", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--probe-every", type=int, default=500)
    ap.add_argument("--chunk-steps", type=int, default=88,
                    help="production's per-iteration step budget; sets the sqrt_release LR cycle")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--warm-draws", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--shuffle-targets", action="store_true")
    ap.add_argument("--verify-production-metric", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cfg = flatten_run_config_defaults(load_yaml_file(args.config))
    for ov in args.override:
        k, v = ov.split("=", 1)
        k, v = k.strip(), v.strip()
        if k not in cfg:
            raise SystemExit(f"override key {k!r} not in flat config -- refusing a silently-ignored knob")
        low = v.lower()
        cfg[k] = True if low == "true" else False if low == "false" else float(v)
        print(f"[screen] OVERRIDE {k} -> {cfg[k]!r}", flush=True)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg = model_config_from_arch(ckpt["arch"])
    planes = input_plane_count(model_cfg.input_extra_features)
    model = build_model(model_cfg)
    kwargs = trainer_kwargs_from_config(cfg)
    kwargs["device"] = args.device
    trainer = Trainer(model, model_config=model_cfg, **kwargs)
    trainer.load(Path(args.checkpoint))
    print(f"[screen] loaded step={trainer.step} planes={planes} "
          f"lrs={[g['lr'] for g in trainer.opt.param_groups][:4]}", flush=True)

    man = json.loads(args.manifest.read_text())
    # EvalSet's constructor prints its desync reading and aborts unless it is
    # exactly zero, so a contaminated set cannot reach the code below.
    evs = [EvalSet(n, man[n], planes) for n in args.eval_sets.split(",")]

    tr_paths: list[str] = []
    for k in args.train_pools.split(","):
        tr_paths += man[k]
    tdir, linked, rows_by_src = prepare_train_pool(tr_paths, evs, args.out / "train_links")
    tc = TrialConfig.from_dict(cfg)
    tr_buf = DiskReplayBuffer(
        10**9, shard_dir=tdir, rng=np.random.default_rng(args.seed), read_only=True,
        input_planes=planes, upgrade_v1_planes=tc.replay_upgrade_v1_planes,
        shuffle_cap=tc.shuffle_buffer_size, shard_size=tc.shard_size,
        refresh_interval=tc.shuffle_refresh_interval, refresh_shards=tc.shuffle_refresh_shards,
        draw_cap_frac=tc.shuffle_draw_cap_frac, wl_max_ratio=tc.shuffle_wl_max_ratio,
        sf_gap_priority_weight=tc.replay_sf_gap_priority_weight,
        sf_gap_priority_signed=tc.replay_sf_gap_priority_signed,
        fast_low_surprise_priority=tc.replay_fast_low_surprise_priority,
        diff_focus_pol_scale=tc.diff_focus_pol_scale, diff_focus_q_weight=tc.diff_focus_q_weight,
    )
    pool_rows = assert_pool_loaded(linked, rows_by_src, iter_shard_paths(tdir), len(tr_buf))
    print(f"[screen] train shards={len(linked)} (manifest paths={len(tr_paths)}) rows={pool_rows} "
          f"views/row@{args.steps}steps={args.steps*args.batch_size/max(pool_rows,1):.3f}", flush=True)

    sampler: Any = tr_buf
    for _ in range(int(args.warm_draws)):
        sampler.sample_batch_arrays(int(args.batch_size))
    if args.shuffle_targets:
        sampler = ShuffleTargets(tr_buf, np.random.default_rng(args.seed + 991))
        print("[screen] NEGATIVE CONTROL: policy_target permuted within every batch", flush=True)

    if args.verify_production_metric:
        # The fp32 and bf16 probes are BOTH reported, because the residual
        # against production is a PRECISION difference, not a pooling one.
        # `Trainer._compute_metrics` runs its forward inside `_amp_context()`
        # (bf16, `use_amp` defaults True) and this rig does not. Pooling
        # contributes exactly zero: `pol_base` is identically 1 on these rows,
        # so production's row-weighted mean IS the row-exact mean. Printing only
        # the fp32 number invited -- and produced -- the wrong explanation of the
        # 0.00025 gap. Measured on the full evalB: fp32 1.1002154,
        # bf16 1.1004793, production 1.1004793 (equal to 7 dp).
        m = trainer.eval_full_pass(evs[0].buf, batch_size=args.eval_batch_size)
        prod = float(getattr(m, "policy_loss", float("nan")))
        r32 = summarize(full_pass_rows(trainer, evs[0], args.eval_batch_size))
        with trainer._amp_context():
            r16 = summarize(full_pass_rows(trainer, evs[0], args.eval_batch_size))
        print(f"[screen] VERIFY production eval_full_pass().policy_loss={prod:.7f} | "
              f"probe fp32 ce={r32['ce']:.7f} (delta {prod - r32['ce']:+.7f}) | "
              f"probe bf16 ce={r16['ce']:.7f} (delta {prod - r16['ce']:+.7f}) -- "
              f"the bf16 delta is the one that should be ~0", flush=True)
        (args.out / "verify.json").write_text(json.dumps(
            {"production_policy_loss": prod,
             "probe_ce_fp32": r32["ce"], "probe_excess_fp32": r32["excess"],
             "probe_ce_bf16": r16["ce"], "probe_excess_bf16": r16["excess"],
             "delta_vs_fp32": prod - r32["ce"], "delta_vs_bf16": prod - r16["ce"],
             "note": "probes and arms run fp32; production's own holdout runs bf16"},
            indent=2))

    hist: list[dict[str, Any]] = []

    def probe(step: int) -> None:
        rec: dict[str, Any] = {"step": step, "t": time.time()}
        for e in evs:
            rows = full_pass_rows(trainer, e, args.eval_batch_size)
            np.savez_compressed(
                str(args.out / f"rows_{e.name}_step{step}.npz"),
                ce=rows["ce"], floor=rows["floor"], agree=rows["agree"],
                base=rows["base"], game_id=rows["game_id"],
                temps=rows["temps"], temp_ce=rows["temp_ce"],
            )
            s = summarize(rows)
            rec[e.name] = s
            print(f"[screen] {args.label} step {step:6d} {e.name}: ce {s['ce']:.5f} "
                  f"floor {s['floor']:.5f} excess {s['excess']:.5f} agree {s['agree']:.5f} "
                  f"rows {s['rows']}", flush=True)
        hist.append(rec)
        (args.out / "history.json").write_text(json.dumps(hist, indent=2))

    probe(0)
    done = 0
    while done < int(args.steps):
        n = min(int(args.probe_every), int(args.steps) - done)
        t0 = time.time()
        tm = run_training(trainer, sampler, steps=n, chunk_steps=int(args.chunk_steps),
                          batch_size=int(args.batch_size), label=args.label)
        done += n
        pl = float(getattr(tm, "policy_loss", float("nan"))) if tm is not None else float("nan")
        lr = [g["lr"] for g in trainer.opt.param_groups]
        print(f"[screen] {args.label} trained {n} steps in {time.time()-t0:.0f}s "
              f"train_policy_loss={pl:.5f} lr={lr[0]:.3e}/{lr[2]:.3e}", flush=True)
        probe(done)

    (args.out / "meta.json").write_text(json.dumps({
        "label": args.label, "steps": args.steps, "batch_size": args.batch_size,
        "seed": args.seed, "chunk_steps": args.chunk_steps, "train_pools": args.train_pools,
        # Counts as LOADED, reconciled against the buffer -- not the manifest's
        # arithmetic, which over-reported A1_BIG by 707 shards / 1.33M rows.
        "train_shards": len(linked), "train_manifest_paths": len(tr_paths),
        "train_rows": pool_rows, "overrides": args.override,
        "shuffle_targets": bool(args.shuffle_targets),
        "views_per_row": args.steps * args.batch_size / max(pool_rows, 1),
        "eval": {e.name: {"rows": len(e.buf), "games": int(np.unique(e.game_id).size),
                          "desync_frac": e.desync_frac} for e in evs},
    }, indent=2))
    tr_buf.close()


if __name__ == "__main__":
    main()
