from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from chess_anti_engine.model import ModelConfig, build_model, load_state_dict_tolerant
from chess_anti_engine.selfplay.match import play_match_batch


def _load_state_dict(path: Path) -> dict:
    ckpt = torch.load(str(path), map_location="cpu")
    # We sometimes save {"model": state_dict}.
    sd = ckpt.get("model", ckpt)
    if not isinstance(sd, dict):
        raise ValueError(f"Unexpected checkpoint format in {path}")
    return sd


def _resolve_from_manifest(manifest_path: Path) -> tuple[dict, Path, Path]:
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    mc = m.get("model_config") or {}

    pub_dir = manifest_path.parent

    latest = None
    if "model" in m and isinstance(m["model"], dict):
        latest = m["model"].get("filename")
    latest_path = pub_dir / str(latest or "latest_model.pt")

    best = None
    if "best_model" in m and isinstance(m["best_model"], dict):
        best = m["best_model"].get("filename")
    best_path = pub_dir / str(best or "best_model.pt")

    return mc, latest_path, best_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Arena match: latest vs best")

    ap.add_argument("--manifest", type=str, default=None, help="Optional path to publish/manifest.json to infer model config and weights")
    ap.add_argument("--latest-model", type=str, default=None)
    ap.add_argument("--best-model", type=str, default=None)

    ap.add_argument("--device", type=str, default=None)

    # Model config (used if --manifest is not provided)
    ap.add_argument("--model", type=str, default="transformer", choices=["tiny", "transformer"])
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--ffn-mult", type=float, default=2)
    ap.add_argument("--use-smolgen", action="store_true")
    ap.add_argument("--use-nla", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")

    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--max-plies", type=int, default=200)

    ap.add_argument("--mcts", type=str, default="puct", choices=["puct", "gumbel"])
    ap.add_argument("--mcts-simulations", type=int, default=200)
    ap.add_argument("--c-puct", type=float, default=2.5)

    ap.add_argument("--seed", type=int, default=0)

    sides = ap.add_mutually_exclusive_group()
    sides.add_argument("--swap-sides", dest="swap_sides", action="store_true")
    sides.add_argument("--no-swap-sides", dest="swap_sides", action="store_false")
    ap.set_defaults(swap_sides=True)

    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(int(args.seed))

    mc = None
    latest_path = Path(args.latest_model) if args.latest_model else None
    best_path = Path(args.best_model) if args.best_model else None

    if args.manifest:
        mc, mf_latest, mf_best = _resolve_from_manifest(Path(args.manifest))
        if latest_path is None:
            latest_path = mf_latest
        if best_path is None:
            best_path = mf_best

    if latest_path is None or best_path is None:
        raise SystemExit("Must provide --latest-model and --best-model (or provide --manifest)")

    if mc is None:
        mc = {
            "kind": str(args.model),
            "embed_dim": int(args.embed_dim),
            "num_layers": int(args.num_layers),
            "num_heads": int(args.num_heads),
            "ffn_mult": float(args.ffn_mult),
            "use_smolgen": bool(args.use_smolgen),
            "use_nla": bool(args.use_nla),
            "gradient_checkpointing": bool(args.gradient_checkpointing),
        }

    model_cfg = ModelConfig(
        kind=str(mc.get("kind", "transformer")),
        embed_dim=int(mc.get("embed_dim", 256)),
        num_layers=int(mc.get("num_layers", 6)),
        num_heads=int(mc.get("num_heads", 8)),
        ffn_mult=float(mc.get("ffn_mult", 2)),
        use_smolgen=bool(mc.get("use_smolgen", True)),
        use_nla=bool(mc.get("use_nla", False)),
        use_gradient_checkpointing=bool(mc.get("gradient_checkpointing", False)),
    )

    latest = build_model(model_cfg)
    load_state_dict_tolerant(latest, _load_state_dict(latest_path), label="arena-latest")
    latest.to(device)
    latest.eval()

    best = build_model(model_cfg)
    load_state_dict_tolerant(best, _load_state_dict(best_path), label="arena-best")
    best.to(device)
    best.eval()

    g = int(args.games)
    if bool(args.swap_sides):
        # Alternate colors to remove first-move bias.
        a_plays_white = [bool(i % 2 == 0) for i in range(g)]
    else:
        a_plays_white = [True] * g

    stats = play_match_batch(
        latest,
        best,
        device=str(device),
        rng=rng,
        games=g,
        max_plies=int(args.max_plies),
        a_plays_white=a_plays_white,
        mcts_type=str(args.mcts),
        mcts_simulations=int(args.mcts_simulations),
        temperature=0.0,
        c_puct=float(args.c_puct),
    )

    winrate = (float(stats.a_win) + 0.5 * float(stats.a_draw)) / float(max(1, stats.games))
    print(
        "arena latest vs best:\n"
        f"  games={stats.games} max_plies={stats.max_plies} latest_as_white={stats.a_as_white} latest_as_black={stats.a_as_black}\n"
        f"  latest W/D/L = {stats.a_win}/{stats.a_draw}/{stats.a_loss}  winrate={winrate:.3f}"
    )


if __name__ == "__main__":
    main()
