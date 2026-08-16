#!/usr/bin/env python3
"""Top-1 agreement with lc0's visit-argmax, on frozen rows, paired (McNemar).

The lc0 positive control's yardstick, exactly as
`scratchpad/lc0_positive_control/PREREG_DRAFT.md` defines it and no other
metric. Two subcommands:

  score    --config <yaml> --frozen <frozen.json> --shards <dirs>
           [--checkpoint <ckpt.pt>] [--seed N] [--shuffle-targets]
           --out <scores.npz>
           Per-row hit/miss against `argmax(policy_target)`. Omit
           --checkpoint to score an untrained net — that is prereg guard 2,
           the RANDOM-INIT FLOOR, and it must land at the chance level
           `scripts/lc0_control_heldout.py chance` printed. `--shuffle-targets`
           is prereg guard 1, the NEGATIVE CONTROL: it permutes the targets
           across rows, so any trained net must collapse to the same floor.

  compare  --a <scores_A.npz> --b <scores_B.npz>
           Paired difference on the rows BOTH files scored, with the McNemar
           discordant counts, a Wald CI on the paired difference, and the
           exact binomial p. Refuses to compare files whose row sets differ,
           and refuses to report a slope whose halfwidth exceeds the
           pre-committed bar.

⚑ THE RANDOM-INIT FLOOR IS SEEDED, AND THE SEED IS IN THE OUTPUT. Four
unseeded draws of the identical command on the identical 100,000 frozen rows
read 0.058770 / 0.062050 / 0.071630 / 0.073500 / 0.080590 — a 2.18 pp spread,
5.6x the 0.392 pp material bar, straddling chance 0.064577 in both directions.
A guard whose own value moves further than the effect it gates on is not a
guard. `--seed` names the draw; the spread across seeds is what the prereg's
band must be derived from, NOT one invocation.

⚑ AND A RANDOM-INIT NET IS NOT EXPECTED TO SIT AT CHANCE. It is a fixed
arbitrary function over a highly non-uniform argmax distribution, so it beats
uniform guessing on the frequent moves. "Above chance" is not evidence the
evaluator manufactures agreement; only the seeded band can say that.

⚑ WHY PAIRED, AND WHY IT REFUSES TO PAD. Two independent top-1 rates at
n=100,000 resolve ~0.30 pp; the same rows scored by both checkpoints resolve
~0.20 pp because the row-to-row difficulty variance cancels. That is the whole
reason the row set is frozen. If the two score files do not cover the same
rows the pairing is a fiction, so `compare` exits rather than intersecting and
quietly reporting a smaller n against the prereg's 0.392 pp bar — which was
derived AT n=100,000.

⚑ THIS SCRIPT MEASURES POLICY AGREEMENT AND NOTHING ELSE. It says nothing
about Elo, and per the prereg a held-out slope alone is not a verdict: it is
read jointly with the same statistic on a frozen sample of ALREADY-TRAINED
rows. Both use this script; only the `--frozen`/`--shards` inputs differ.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from chess_anti_engine.eval.lc0_control_rows import iter_shard_arrays, load_frozen, row_ids
from chess_anti_engine.replay.dataset import collate_arrays
from chess_anti_engine.train.losses import apply_policy_mask_to_logits
from chess_anti_engine.train.trainer import Trainer, trainer_kwargs_from_config
from chess_anti_engine.model import build_model, model_config_from_flat_config
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file


def _policy_logits(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """The base policy head, chosen exactly as ``compute_loss`` chooses it."""
    logits = outputs["policy"] if "policy" in outputs else outputs.get("policy_own")
    if logits is None:
        raise KeyError("model outputs carry neither 'policy' nor 'policy_own'")
    return logits


def _load_trainer(
    config_path: Path, checkpoint: Path | None, device: str, *, seed: int,
) -> Trainer:
    cfg = flatten_run_config_defaults(load_yaml_file(str(config_path)))
    kwargs = trainer_kwargs_from_config(cfg)
    kwargs["device"] = device
  # ⚑ Compilation is off for scoring regardless of the config. It changes
  # nothing about the arithmetic and costs minutes per invocation, and the
  # ruler is run far more often than the trainer.
    kwargs["use_compile"] = False
    model_cfg = model_config_from_flat_config(cfg)
  # ⚑ SEED BEFORE build_model, not after. Without this the random-init floor —
  # a preregistered GATE — is a different net on every invocation and cannot be
  # reproduced from the score metadata. Measured spread 2.18 pp.
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    model = build_model(model_cfg)
  # See lc0_control_train.py: without model_config the trainer cannot read
  # LC0-root history rows at all.
    trainer = Trainer(model, model_config=model_cfg, **kwargs)
    if checkpoint is not None:
  # ⚑ NOT Trainer.load. That calls load_state_dict_tolerant with the default
  # require_complete=False, so a config whose architecture drifted from the
  # checkpoint's scores a HYBRID of trained and freshly-initialised tensors
  # under the checkpoint's name — and the catastrophic-load detector only
  # fires below 50%. Top-1 is this arm's only yardstick; a partial load has to
  # be an error, not a silent discount. Optimizer/scheduler state is
  # deliberately not restored: nothing here steps.
        from chess_anti_engine.model import load_state_dict_tolerant

        payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
        load_state_dict_tolerant(
            trainer.model, payload["model"],
            label=f"lc0-control-score {Path(checkpoint).name}",
            require_complete=True,
        )
    return trainer


def _predict_rows(
    trainer: Trainer,
    shard_dirs: list[Path],
    wanted: list[str],
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """``(predicted_move_id, target_move_id, tied_rows)`` aligned to ``wanted``.

    ⚑ The target is ``argmax(policy_target)``, i.e. lc0's own visit-argmax over
    the compact 1858 encoding. `tied` counts rows where the top visit count is
    shared by more than one move: on those, argmax breaks the tie by index and
    "agreement" is partly arbitrary. It is reported rather than dropped —
    dropping rows would change the population the prereg's resolution was
    computed for.

    ⚑ MOVE IDS, not hits. The negative control needs to re-pair predictions
    with somebody else's target, which a hit vector has already thrown away —
    and permuting a hit vector preserves the hit RATE exactly, i.e. produces a
    negative control that cannot fail.
    """
    want = set(wanted)
    predicted_by_id: dict[str, int] = {}
    target_by_id: dict[str, int] = {}
    tied = 0
    rng = np.random.default_rng(0)
    device = trainer.device
    trainer.model.eval()

    for _path, arrs in iter_shard_arrays(shard_dirs):
        ids = row_ids(arrs)
        take = [
            i for i, row_id in enumerate(ids)
            if row_id in want and row_id not in predicted_by_id
        ]
        if not take:
            continue
        n_rows = len(ids)
        row_keys = [
            key for key, value in arrs.items()
            if value.ndim >= 1 and value.shape[0] == n_rows
        ]
  # ⚑ The 0-d metadata scalars (`_input_history_encoding`, `_policy_encoding`,
  # `_policy_size`, `_history_rep_fix`) must be CARRIED, not indexed. Dropping
  # them does not fail quietly: `select_input_history_arrays` then reads the
  # subset as legacy-history rows and refuses the whole batch, because
  # synthesizing LC0-root planes from legacy cannot recover side-to-move
  # (rl_loop_audit M12). Slicing a shard is not the same as sampling it.
        scalars = {key: value for key, value in arrs.items() if value.ndim == 0}
        for start in range(0, len(take), batch_size):
            index = np.array(take[start:start + batch_size], dtype=np.int64)
            rows = {key: np.ascontiguousarray(arrs[key][index]) for key in row_keys}
            rows.update(scalars)
            prepared = trainer._prepare_host_arrays(
                rows, rng=rng, mirror_prob=0.0,
            )
            batch = collate_arrays(prepared, device=device)
            with torch.no_grad(), trainer._amp_context():
                outputs = trainer.model(batch["x"])
                logits = apply_policy_mask_to_logits(
                    _policy_logits(outputs), batch, "legal_mask", "has_legal_mask",
                )
                predicted = logits.float().argmax(dim=-1).detach().cpu().numpy()
            target = batch["policy_t"].float().detach().cpu().numpy()
            best = target.argmax(axis=-1)
            top = target.max(axis=-1, keepdims=True)
            tied += int((((target >= top) & (top > 0.0)).sum(axis=-1) > 1).sum())
            for offset, row_index in enumerate(index):
                predicted_by_id[ids[int(row_index)]] = int(predicted[offset])
                target_by_id[ids[int(row_index)]] = int(best[offset])

    missing = [row_id for row_id in wanted if row_id not in predicted_by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(wanted)} frozen rows were not found in the "
            "given shards. Scoring a subset would silently shrink n below the "
            "value the prereg's resolution was computed at.",
        )
    return (
        np.array([predicted_by_id[row_id] for row_id in wanted], dtype=np.int64),
        np.array([target_by_id[row_id] for row_id in wanted], dtype=np.int64),
        tied,
    )


def _score_rows(
    trainer: Trainer,
    shard_dirs: list[Path],
    wanted: list[str],
    *,
    batch_size: int,
    shuffle_targets_seed: int | None = None,
) -> tuple[np.ndarray, int]:
    """Per-row hit vector aligned to ``wanted``, plus the tied-argmax count.

    ⚑ ``shuffle_targets_seed`` is PREREG GUARD 1, the NEGATIVE CONTROL: each
    row is scored against another row's lc0 target. Every row keeps a real
    target, just not its own, so a net that learned the position-to-move
    mapping must collapse to the chance floor. A net that does not has been
    reading something in the RIG rather than in the position.
    """
    predicted, target, tied = _predict_rows(
        trainer, shard_dirs, wanted, batch_size=batch_size,
    )
    if shuffle_targets_seed is not None:
        target = target[np.random.default_rng(shuffle_targets_seed).permutation(target.size)]
    return (predicted == target).astype(np.uint8), tied


def cmd_score(args: argparse.Namespace) -> int:
    payload = load_frozen(Path(args.frozen))
    wanted = list(payload["row_ids"])
    trainer = _load_trainer(
        Path(args.config), args.checkpoint, args.device, seed=int(args.seed),
    )
    shuffle_seed = int(args.shuffle_seed) if args.shuffle_targets else None
    hits, tied = _score_rows(
        trainer, args.shards, wanted, batch_size=int(args.batch_size),
        shuffle_targets_seed=shuffle_seed,
    )
    rate = float(hits.mean())
    np.savez_compressed(
        Path(args.out),
        row_ids=np.array(wanted, dtype="U32"),
        hit=hits,
        meta=np.array([json.dumps({
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "config": str(args.config),
            "frozen": str(args.frozen),
            "frozen_sha_source": payload.get("row_id_version"),
  # ⚑ The seed is part of the RESULT, not of the invocation. Without it in the
  # file the random-init floor cannot be reproduced from its own metadata.
            "init_seed": int(args.seed),
            "shuffled_targets_seed": shuffle_seed,
            "rows": int(hits.size),
            "top1_agreement": rate,
            "tied_argmax_rows": int(tied),
        })], dtype=object),
        allow_pickle=True,
    )
    label = args.checkpoint or "NONE (random init — the floor guard)"
    print(f"checkpoint        {label}")
    print(f"init seed         {int(args.seed)}")
    if shuffle_seed is not None:
        print(f"⚑ SHUFFLED TARGETS (negative control), seed {shuffle_seed}")
    print(f"rows scored       {hits.size}")
    print(f"top-1 agreement   {rate:.6f}  ({rate * 100:.4f}%)")
    print(f"tied-argmax rows  {tied}")
    print(f"written           {args.out}")
    return 0


def _load_scores(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = np.load(path, allow_pickle=True)
    meta = json.loads(str(data["meta"][0]))
    return data["row_ids"], data["hit"].astype(np.int64), meta


def cmd_compare(args: argparse.Namespace) -> int:
    ids_a, hit_a, meta_a = _load_scores(Path(args.a))
    ids_b, hit_b, meta_b = _load_scores(Path(args.b))
    if list(ids_a) != list(ids_b):
        print(
            "FAIL: the two score files do not cover the same rows in the same "
            "order, so they cannot be paired. Re-score both against the SAME "
            "frozen row set instead of intersecting them here.",
            file=sys.stderr,
        )
        return 1
    n = int(hit_a.size)
  # McNemar's discordant cells. b = A right / B wrong, c = A wrong / B right.
    b = int(((hit_a == 1) & (hit_b == 0)).sum())
    c = int(((hit_a == 0) & (hit_b == 1)).sum())
    rate_a, rate_b = hit_a.mean(), hit_b.mean()
    delta = (c - b) / n
  # Wald SE for the difference of CORRELATED proportions. The concordant cells
  # contribute nothing, which is exactly why pairing buys resolution.
    var = (b + c - (c - b) ** 2 / n) / n**2
    se = float(np.sqrt(max(var, 0.0)))
    half = 1.959963985 * se
    p_exact = _exact_mcnemar_p(b, c)
    print(f"A  {meta_a.get('checkpoint')}   top-1 {rate_a:.6f}")
    print(f"B  {meta_b.get('checkpoint')}   top-1 {rate_b:.6f}")
    print(f"paired rows          {n}")
    print(f"discordant  b(A only)={b}  c(B only)={c}  "
          f"discordance={(b + c) / n:.4f}")
    print(f"delta (B - A)        {delta * 100:+.4f} pp")
    print(f"95% CI               [{(delta - half) * 100:+.4f}, "
          f"{(delta + half) * 100:+.4f}] pp   (halfwidth {half * 100:.4f} pp)")
    print(f"exact McNemar p      {p_exact:.6g}")
  # ⚑ THE PRE-COMMITTED BAR IS A PROPERTY OF n, AND NOTHING WAS ENFORCING n.
  # The prereg's 0.392 pp material bar is derived AT n=100,000. The reviewer's
  # end-to-end demonstration paired 5,000 train-side rows and got a halfwidth
  # of 1.0721 pp — 2.7x the bar it would then be judged against — and the rig
  # printed it without comment. A slope whose own CI is wider than the effect
  # size it is compared to cannot decide anything, so this refuses rather than
  # letting prose carry the gate.
    half_pp = half * 100.0
    bar_pp = float(args.max_halfwidth_pp)
    if half_pp > bar_pp:
        message = (
            f"halfwidth {half_pp:.4f} pp exceeds the pre-committed bar "
            f"{bar_pp:.4f} pp at n={n}. The prereg derives that bar at "
            "n=100,000; at this n the comparison cannot resolve the effect it "
            "is being judged against. Score more rows, or pass "
            "--max-halfwidth-pp explicitly with the bar you are re-deriving."
        )
        if not args.allow_underpowered:
            print(f"FAIL: {message}", file=sys.stderr)
            return 1
        print(f"⚑⚑ --allow-underpowered: {message}")
    return 0


def _exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact binomial p on the discordant pairs."""
    total = b + c
    if total == 0:
        return 1.0
    from math import comb

    smaller = min(b, c)
    tail = sum(comb(total, k) for k in range(smaller + 1)) / 2**total
    return float(min(1.0, 2.0 * tail))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score")
    score.add_argument("--config", type=Path, default=Path("configs/lc0_positive_control.yaml"))
    score.add_argument("--frozen", type=Path, required=True)
    score.add_argument("--shards", type=Path, nargs="+", required=True)
    score.add_argument("--checkpoint", type=Path, default=None)
    score.add_argument("--out", type=Path, required=True)
    score.add_argument("--batch-size", type=int, default=512)
    score.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    score.add_argument(
        "--seed", type=int, default=0,
        help="⚑ seeds build_model. The random-init floor is a GATE; unseeded it "
             "spans 2.18 pp, 5.6x the 0.392 pp material bar.",
    )
    score.add_argument(
        "--shuffle-targets", action="store_true",
        help="prereg guard 1, the NEGATIVE CONTROL: score each row against "
             "another row's lc0 target. Any net must collapse to chance.",
    )
    score.add_argument("--shuffle-seed", type=int, default=0)
    score.set_defaults(handler=cmd_score)

    compare = sub.add_parser("compare")
    compare.add_argument("--a", type=Path, required=True)
    compare.add_argument("--b", type=Path, required=True)
    compare.add_argument(
        "--max-halfwidth-pp", type=float, default=0.392,
        help="the prereg's material bar. A slope whose 95%% halfwidth exceeds "
             "it is refused rather than reported.",
    )
    compare.add_argument("--allow-underpowered", action="store_true")
    compare.set_defaults(handler=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
