#!/usr/bin/env python3
"""Offline policy-absorption screen from the banked iter-478 state.

Held-out ruler: a FULL DETERMINISTIC PASS (production's `_iter_full_pass_batches`)
over a game-disjoint, desync-exactly-zero eval set, dumping PER-ROW policy CE and
target entropy so arm-vs-arm differences get a game-clustered CI.
"""
from __future__ import annotations

import argparse
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
from chess_anti_engine.replay.shard import load_shard_arrays
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


class EvalSet:
    """Fixed held-out rows + their game ids, in the buffer's own row order."""

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
            hw = np.asarray(arrs["has_sf_wdl"]).astype(bool)
            hm = np.asarray(arrs["has_sf_multipv_raw"]).astype(bool)
            lab += int(hw.sum())
            des += int((hw & ~hm).sum())
            gids.append(np.asarray(arrs["game_id"], dtype=np.int64))
            buf.add_many_arrays(arrs)
        self.buf = buf
        self.game_id = np.concatenate(gids) if gids else np.zeros(0, np.int64)
        self.labelled = lab
        self.desync_rows = des
        self.desync_frac = des / lab if lab else 0.0
        if len(buf) != self.game_id.size:
            raise SystemExit(f"{name}: buffer rows {len(buf)} != game ids {self.game_id.size}")

    def report(self) -> str:
        return (f"[eval:{self.name}] rows={len(self.buf)} games={np.unique(self.game_id).size} "
                f"labelled={self.labelled} desync_rows={self.desync_rows} "
                f"DESYNC_FRAC={self.desync_frac:.6f}")


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


def link_dir(paths: list[str], dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for p in paths:
        src = Path(p).resolve()
        link = dst / src.name
        if link.exists() or link.is_symlink():
            continue
        os.symlink(str(src), str(link), target_is_directory=True)
    return dst


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
    evs = [EvalSet(n, man[n], planes) for n in args.eval_sets.split(",")]
    for e in evs:
        print(e.report(), flush=True)
        if e.desync_frac != 0.0:
            raise SystemExit(f"eval set {e.name} desync fraction {e.desync_frac} is not exactly 0")

    tr_paths: list[str] = []
    for k in args.train_pools.split(","):
        tr_paths += man[k]
    tdir = link_dir(tr_paths, args.out / "train_links")
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
    print(f"[screen] train shards={len(tr_paths)} rows={len(tr_buf)} "
          f"views/row@{args.steps}steps={args.steps*args.batch_size/max(len(tr_buf),1):.3f}", flush=True)

    sampler: Any = tr_buf
    for _ in range(int(args.warm_draws)):
        sampler.sample_batch_arrays(int(args.batch_size))
    if args.shuffle_targets:
        sampler = ShuffleTargets(tr_buf, np.random.default_rng(args.seed + 991))
        print("[screen] NEGATIVE CONTROL: policy_target permuted within every batch", flush=True)

    if args.verify_production_metric:
        m = trainer.eval_full_pass(evs[0].buf, batch_size=args.eval_batch_size)
        r = summarize(full_pass_rows(trainer, evs[0], args.eval_batch_size))
        print(f"[screen] VERIFY production evaluate().policy_loss={getattr(m,'policy_loss',None)} "
              f"vs probe ce={r['ce']:.6f}", flush=True)
        (args.out / "verify.json").write_text(json.dumps(
            {"production_policy_loss": float(getattr(m, "policy_loss", float("nan"))),
             "probe_ce": r["ce"], "probe_excess": r["excess"]}, indent=2))

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
        left = n
        tm = None
        while left > 0:
            c = min(int(args.chunk_steps), left)
            tm = trainer.train_steps(sampler, batch_size=int(args.batch_size), steps=c)
            left -= c
        done += n
        pl = float(getattr(tm, "policy_loss", float("nan"))) if tm is not None else float("nan")
        lr = [g["lr"] for g in trainer.opt.param_groups]
        print(f"[screen] {args.label} trained {n} steps in {time.time()-t0:.0f}s "
              f"train_policy_loss={pl:.5f} lr={lr[0]:.3e}/{lr[2]:.3e}", flush=True)
        probe(done)

    (args.out / "meta.json").write_text(json.dumps({
        "label": args.label, "steps": args.steps, "batch_size": args.batch_size,
        "seed": args.seed, "chunk_steps": args.chunk_steps, "train_pools": args.train_pools, "train_shards": len(tr_paths),
        "train_rows": len(tr_buf), "overrides": args.override,
        "shuffle_targets": bool(args.shuffle_targets),
        "views_per_row": args.steps * args.batch_size / max(len(tr_buf), 1),
        "eval": {e.name: {"rows": len(e.buf), "games": int(np.unique(e.game_id).size),
                          "desync_frac": e.desync_frac} for e in evs},
    }, indent=2))
    tr_buf.close()


if __name__ == "__main__":
    main()
