#!/usr/bin/env python3
"""Offline input-plane usefulness audit + candidate-feature pre-screen.

No training. Runs on existing replay shards + a frozen checkpoint. Three modes:

  ablation   Mean-ablate each EXISTING plane family (replace with its batch mean
             so the net stays roughly in-distribution) and measure the rise in
             policy/value loss. Ranks the planes the net actually leans on.

  redundancy Linear-probe the frozen trunk to predict a CANDIDATE feature block.
             High held-out R^2 => the net already computes it internally =>
             adding it as an input plane is redundant (low expected value).

  relevance  Correlate a candidate feature with the net's per-position SF-regret
             (CE of the net's value/policy vs the stored Stockfish labels). High
             => the feature fires where the net is wrong => likely to help.

Candidate feature for the predictors: the NEW planes a candidate extra-features
version adds beyond the base it extends, recomputed offline from stored planes
(encoding/plane_decode.py). Selectable via ``--candidate-version`` (default
v2_threats; also v3_checks / v3_xray). Calibrate against v2_threats first — that
sidecar is a KNOWN win, so a trustworthy pre-screen must score it high-relevance
/ low-redundancy. Then point it at new v3 bundles.

  PYTHONPATH=. python3 scripts/feature_prescreen.py --checkpoint X.pt \
      --shard-dir runs/pbt2_small/replay_shards --mode ablation -n 4096
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from chess_anti_engine.encoding.plane_decode import recompute_extra_planes
from chess_anti_engine.replay.shard import iter_shard_paths, load_shard_arrays
from chess_anti_engine.replay.threat_upgrade import upgrade_arrays_to_planes
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

# Absolute plane-index ranges per family (see encoding/features.py).
FAMILIES_LC0 = {
    "lc0_history": (0, 104),
    "lc0_aux": (104, 112),
}
FAMILIES_V1 = {
    "v1_kingzone": (112, 122),
    "v1_pins": (122, 128),
    "v1_pawnstruct": (128, 136),
    "v1_mobility": (136, 142),
    "v1_outposts": (142, 146),
}
FAMILIES_V2 = {
    "v2_attacks_all": (146, 148),
    "v2_attacks_typed": (148, 160),
    "v2_attacker_counts": (160, 162),
    "v2_hanging": (162, 164),
    "v2_cheaper_threat": (164, 166),
    "v2_safe_checks": (166, 170),
    "v2_control": (170, 171),
    "v2_pawn_tension": (171, 173),
    "v2_pawn_storm": (173, 175),
}
# v3 families: appended after the v2 block. v3_checks and v3_xray are SEPARATE
# versions that both START at absolute plane 175 (extra-block idx 63) — keyed
# off the net's total plane count (179 = v3_checks, 181 = v3_xray).
FAMILIES_V3_CHECKS = {
    "v3_opp_safe_checks": (175, 179),
}
FAMILIES_V3_XRAY = {
    "v3_xray_attacks": (175, 181),
}

# One-shot guard so the policy-width-mismatch warning prints once, not per call.
_WARNED_POLICY_WIDTH = {"done": False}


def load_sample(
    shard_dir: str,
    n: int,
    rng: np.random.Generator,
    target_planes: int,
    history_encoding: str | None,
):
    """Concatenate shards until >= n rows, then subsample n. Returns (arrs, meta).

    Shards under one dir can mix plane widths (146 v1 / 175 v2 / 179 v3_checks /
    181 v3_xray …) and the random shard order makes which widths land together
    nondeterministic, so each chunk is normalized to the model's ``target_planes``
    width BEFORE the x-concat — otherwise numpy raises a dim mismatch on the
    first cross-width pair. Per-chunk normalization mirrors the replay-buffer
    load path (recompute the extra block from the 112 LC0 base, never zero-pad
    across versions).
    """
    paths = list(iter_shard_paths(Path(shard_dir)))
    if not paths:
        raise SystemExit(f"no shards under {shard_dir}")
    paths = [paths[i] for i in rng.permutation(len(paths))]
    chunks: list[dict[str, np.ndarray]] = []
    meta = None
    got = 0
    for p in paths:
        arrs, meta = load_shard_arrays(p)
        arrs = _normalize_arrs_to_model_planes(arrs, target_planes, history_encoding)
        chunks.append(arrs)
        got += int(arrs["x"].shape[0])
        if got >= n:
            break
    keys = set(chunks[0])
    for c in chunks[1:]:
        keys &= set(c)
    # Keep only per-row arrays (ndim>=1, leading dim == that chunk's row count);
    # drops scalar/metadata fields that can't be row-concatenated.
    row_keys = [
        k for k in keys
        if all(getattr(c[k], "ndim", 0) >= 1 and c[k].shape[0] == c["x"].shape[0] for c in chunks)
    ]
    cat = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in row_keys}
    total = int(cat["x"].shape[0])
    idx = rng.choice(total, size=min(n, total), replace=False)
    return {k: v[idx] for k, v in cat.items()}, meta


def to_tensors(arrs: dict[str, np.ndarray], device: str) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in arrs.items():
        if v.dtype == np.float16:
            v = v.astype(np.float32)
        try:
            out[k] = torch.from_numpy(np.ascontiguousarray(v)).to(device)
        except TypeError:
            pass
    return out


def soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample cross-entropy -sum(target * log_softmax(logits))."""
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1)


@torch.no_grad()
def eval_losses(model, x: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Per-sample policy + value losses vs the net's own targets, and SF-regret."""
    out = model(x)
    res: dict[str, torch.Tensor] = {}
    pol = out["policy_own"]
    if "legal_mask" in batch and batch["legal_mask"].shape[-1] == pol.shape[-1]:
        pol = pol.masked_fill(batch["legal_mask"] <= 0, -1e9)
    if "policy_target" in batch:
        if batch["policy_target"].shape[-1] == pol.shape[-1]:
            res["policy_ce"] = soft_ce(pol, batch["policy_target"])
        elif not _WARNED_POLICY_WIDTH["done"]:
            _WARNED_POLICY_WIDTH["done"] = True
            print(
                f"WARNING: skipping policy_ce — stored policy_target width "
                f"{int(batch['policy_target'].shape[-1])} != model policy width "
                f"{int(pol.shape[-1])} (1858 vs 4672 encoding mismatch); "
                f"policy losses/ablation deltas will be value-only."
            )
    if "wdl_target" in batch:
        res["wdl_ce"] = F.cross_entropy(out["wdl"], batch["wdl_target"].long(), reduction="none")
    # SF-regret (relevance target): net vs stored Stockfish labels.
    if "sf_wdl" in batch:
        res["sf_wdl_regret"] = soft_ce(out["wdl"], batch["sf_wdl"])
    if "sf_policy_target" in batch and batch["sf_policy_target"].shape[-1] == pol.shape[-1]:
        res["sf_pol_regret"] = soft_ce(pol, batch["sf_policy_target"])
    return res


def run_ablation(model, x, batch, families):
    base = eval_losses(model, x, batch)
    metric_keys = [k for k in ("policy_ce", "wdl_ce") if k in base]
    base_means = {k: base[k].mean().item() for k in metric_keys}
    rows = []
    for name, (s, e) in families.items():
        x_ab = x.clone()
        x_ab[:, s:e] = x[:, s:e].mean(dim=(0, 2, 3), keepdim=True)
        ab = eval_losses(model, x_ab, batch)
        rows.append((name, e - s, {k: ab[k].mean().item() - base_means[k] for k in metric_keys}))
    rows.sort(key=lambda r: -sum(r[2].values()))
    print(f"\n=== ABLATION (mean-replace) — baseline {base_means} ===")
    hdr = "family".ljust(20) + "n".rjust(4) + "".join(f"  Δ{k}".rjust(14) for k in metric_keys)
    print(hdr)
    for name, npl, d in rows:
        print(name.ljust(20) + str(npl).rjust(4) + "".join(f"{d[k]:+.4f}".rjust(14) for k in metric_keys))
    print("(higher Δ = net relies on that family more)")


def _capture_trunk(model):
    """Monkeypatch _encode_tokens to stash the trunk tensor (B,64,D)."""
    orig = model._encode_tokens  # noqa: SLF001
    box: dict[str, torch.Tensor] = {}

    def patched(x, relations=None):
        t, rel = orig(x, relations)
        box["t"] = t.detach()
        return t, rel

    model._encode_tokens = patched  # noqa: SLF001
    return box, (lambda: setattr(model, "_encode_tokens", orig))


def _ridge_r2(feats: np.ndarray, target: np.ndarray, lam: float = 1.0) -> float:
    """Held-out R^2 of a ridge fit feats->target (50/50 split, multi-output)."""
    n = feats.shape[0]
    half = n // 2
    xtr, xte = feats[:half], feats[half:]
    ytr, yte = target[:half], target[half:]
    xtr = np.concatenate([xtr, np.ones((xtr.shape[0], 1))], axis=1)
    xte = np.concatenate([xte, np.ones((xte.shape[0], 1))], axis=1)
    d = xtr.shape[1]
    w = np.linalg.solve(xtr.T @ xtr + lam * np.eye(d), xtr.T @ ytr)
    pred = xte @ w
    ss_res = float(((yte - pred) ** 2).sum())
    ss_tot = float(((yte - yte.mean(axis=0)) ** 2).sum()) + 1e-9
    return 1.0 - ss_res / ss_tot


# Candidate version -> (families dict for the NEW planes beyond the base
# version, base extra-plane count). The candidate recompute returns the full
# extra block for ``version``; we score only the planes ADDED on top of the
# base version it extends (v2 extends v1; v3_* extend v2).
_CANDIDATE_BLOCKS = {
    "v2_threats": (FAMILIES_V2, 34, "v1"),
    "v3_checks": (FAMILIES_V3_CHECKS, 63, "v2_threats"),
    "v3_xray": (FAMILIES_V3_XRAY, 63, "v2_threats"),
}


@torch.no_grad()
def candidate_feature_block(
    arrs: dict[str, np.ndarray],
    history_encoding: str | None,
    version: str,
) -> tuple[np.ndarray, list[str]]:
    """Recompute ``version``'s extra block and summarize the NEW planes per family.

    Returns ``(cols, family_names)`` where ``cols`` is ``(N, n_families)`` — each
    column the spatial+plane mean of one family of planes ADDED by ``version``
    beyond the base version it extends (e.g. v3_checks vs v2 base => the 4 planes
    [175:179]). So redundancy/relevance screen the ACTUAL candidate, not always
    v2_threats.
    """
    if version not in _CANDIDATE_BLOCKS:
        raise SystemExit(
            f"--candidate-version {version!r} not screenable; "
            f"expected one of {sorted(_CANDIDATE_BLOCKS)}"
        )
    families, base_extra, _base_version = _CANDIDATE_BLOCKS[version]
    extra = recompute_extra_planes(arrs["x"].astype(np.float32), history_encoding, version=version)
    block = extra[:, base_extra:]  # planes beyond the base version's extra block
    # Family ranges are absolute plane indices; the recomputed extra block starts
    # at absolute 112, and ``block`` starts at absolute 112 + base_extra.
    base_abs = 112 + base_extra
    cols = []
    for _name, (s, e) in families.items():
        ls, le = s - base_abs, e - base_abs
        cols.append(block[:, ls:le].mean(axis=(1, 2, 3)))
    return np.stack(cols, axis=1), list(families.keys())


def run_redundancy(model, x, arrs, history_encoding, candidate_version):
    box, restore = _capture_trunk(model)
    trunks = []
    try:
        with torch.no_grad():
            for i in range(0, x.shape[0], 1024):
                _ = model(x[i:i + 1024])
                trunks.append(box["t"].mean(dim=1).float().cpu())
    finally:
        restore()
    trunk = torch.cat(trunks, dim=0).numpy()  # (N, D)
    feat, _names = candidate_feature_block(arrs, history_encoding, candidate_version)  # (N, n_families)
    r2 = _ridge_r2(trunk, feat)
    print(f"\n=== REDUNDANCY — trunk -> {candidate_version}(family-summary) held-out R^2 = {r2:.3f} ===")
    print("high R^2 (>~0.7) => net already encodes it => redundant; low => novel")


def run_relevance(model, x, batch, arrs, history_encoding, candidate_version):
    losses = eval_losses(model, x, batch)
    regret_key = "sf_wdl_regret" if "sf_wdl_regret" in losses else ("policy_ce" if "policy_ce" in losses else "wdl_ce")
    regret = losses[regret_key].float().cpu().numpy()  # (N,)
    feat, names = candidate_feature_block(arrs, history_encoding, candidate_version)  # (N, n_families)
    print(f"\n=== RELEVANCE — corr({candidate_version} family activation, net regret[{regret_key}]) ===")
    for name, col in zip(names, feat.T):
        if col.std() < 1e-9:
            r = float("nan")
        else:
            r = float(np.corrcoef(col, regret)[0, 1])
        print(f"  {name.ljust(20)} r={r:+.3f}")
    r2_all = _ridge_r2(feat, regret.reshape(-1, 1))
    print(f"  [all {candidate_version} families -> regret] held-out R^2 = {r2_all:.3f}  (higher = more relevant to errors)")


def _model_input_planes(model) -> int:
    """Input-plane count the model expects (from its TransformerConfig)."""
    cfg = getattr(model, "cfg", None)
    if cfg is not None and getattr(cfg, "in_planes", None) is not None:
        return int(cfg.in_planes)
    raise SystemExit("could not determine model input-plane count (no cfg.in_planes)")


def _normalize_arrs_to_model_planes(
    arrs: dict[str, np.ndarray],
    target_planes: int,
    history_encoding: str | None,
) -> dict[str, np.ndarray]:
    """Recompute stored ``x`` to the model's expected input-plane width.

    Mirrors the replay-buffer load path: stored width == target passes through;
    stored < target recomputes the extra block from the 112 LC0 base (never
    zero-pads across versions); stored > target is a hard error. Keeps
    ``model(x)`` from crashing on a channel mismatch (e.g. 146-plane shards into
    a 175/179/181-plane net).
    """
    stored = int(np.asarray(arrs["x"]).shape[1])
    target = int(target_planes)
    if stored == target:
        return arrs
    if stored > target:
        raise SystemExit(
            f"shards store {stored} input planes but the model expects {target}; "
            f"refusing to truncate encoded inputs"
        )
    out, _stats = upgrade_arrays_to_planes(arrs, target, history_encoding=history_encoding)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--shard-dir", default="runs/pbt2_small/replay_shards")
    p.add_argument("--mode", choices=["ablation", "redundancy", "relevance", "all"], default="ablation")
    p.add_argument(
        "--candidate-version",
        choices=sorted(_CANDIDATE_BLOCKS),
        default="v2_threats",
        help="extra-features version whose NEW planes the redundancy/relevance "
             "screens score (default v2_threats — a known win, for calibration).",
    )
    p.add_argument("-n", "--n-positions", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--history-encoding",
        default=None,
        help="LC0 history encoding for plane decode (legacy / lc0_root / "
             "lc0_root_legacy_meta). Default: inferred from the loaded "
             "checkpoint's input_history_encoding (current nets use "
             "lc0_root_legacy_meta) so EP planes decode from the right slot.",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"[prescreen] loading {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    net_planes = _model_input_planes(model)

    # Resolve history encoding: an explicit flag wins; otherwise infer from the
    # loaded net (its input_history_encoding attr) so the decode reads EP from
    # the slot the checkpoint actually used, falling back to the legacy default.
    history_encoding = args.history_encoding
    if history_encoding is None:
        history_encoding = str(getattr(model, "input_history_encoding", "legacy"))
        print(f"[prescreen] history-encoding inferred from checkpoint: {history_encoding}")

    print(f"[prescreen] sampling {args.n_positions} positions from {args.shard_dir}")
    # load_sample normalizes each chunk to net_planes before concat (shards may
    # mix widths), so arrs["x"] is already at the model width here.
    arrs, _meta = load_sample(args.shard_dir, args.n_positions, rng, net_planes, history_encoding)
    n_planes = int(arrs["x"].shape[1])
    print(f"[prescreen] {arrs['x'].shape[0]} positions, x has {n_planes} planes (model width {net_planes})")

    batch = to_tensors({k: v for k, v in arrs.items() if k != "x"}, args.device)
    x = torch.from_numpy(np.ascontiguousarray(arrs["x"].astype(np.float32))).to(args.device)

    # Ablation families key off the NET's plane count (== x width after
    # normalization), not the stored shard width.
    families = dict(FAMILIES_LC0)
    families.update(FAMILIES_V1)
    if n_planes >= 175:
        families.update(FAMILIES_V2)
    if n_planes == 179:
        families.update(FAMILIES_V3_CHECKS)
    elif n_planes == 181:
        families.update(FAMILIES_V3_XRAY)

    if args.mode in ("ablation", "all"):
        run_ablation(model, x, batch, families)
    if args.mode in ("redundancy", "all"):
        run_redundancy(model, x, arrs, history_encoding, args.candidate_version)
    if args.mode in ("relevance", "all"):
        run_relevance(model, x, batch, arrs, history_encoding, args.candidate_version)


if __name__ == "__main__":
    main()
