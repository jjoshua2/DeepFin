#!/usr/bin/env python3
"""Freeze, audit and floor the lc0 positive control's held-out split.

Three subcommands, in the order they must be run:

  freeze  --shards <the 6 newest hourly dirs> --out <frozen.json>
          Builds the explicit row-id list and prints its sha256.

  purity  --frozen <frozen.json> --train-shards <train dirs> [--receipt <f>]
          Asserts ZERO EXPOSURE — no held-out INPUT (`x` alone) occurs
          anywhere in train — and reports the record-id intersection
          alongside it. Exit 1 on any overlap, and exit 1 when the train side
          turns out to hold no rows at all. `--receipt` banks WHICH
          directories were scanned and the frozen artifact's sha256, so the
          training driver can refuse a corpus this check never saw.

  chance  --frozen <frozen.json> --shards <the same 6 dirs>
          Computes E[1/n_legal] on exactly the frozen rows — the negative
          control's floor. Prints 1/E[n_legal] alongside it purely so the
          wrong one is visibly wrong.

⚑ `freeze` must run BEFORE the first training step, and the sha256 it prints
goes in the ledger. A held-out set chosen after seeing a number is not a
held-out set. See docs/lc0_positive_control_prereg.md.

Why the id is a content digest rather than the tar name, and why the purity
check is by id: chess_anti_engine/eval/lc0_control_rows.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from chess_anti_engine.eval.lc0_control_rows import (
    EmptyTrainCorpus,
    chance_level,
    frozen_row_set,
    legal_counts_for_ids,
    load_frozen,
    purity_against_train,
    write_frozen,
)


def cmd_freeze(args: argparse.Namespace) -> int:
    payload = frozen_row_set(args.shards, sample=int(args.sample), seed=int(args.seed))
    print(f"sources ({len(payload['sources'])}):")
    for name, rows in zip(payload["sources"], payload["source_rows"], strict=True):
        print(f"  {name}: {rows} rows")
    print(f"pool rows              {payload['pool_rows']}")
    print(f"pool unique ids        {payload['pool_unique_ids']}")
    print(f"pool duplicate ids     {payload['pool_duplicate_ids']}  "
          "(non-zero = the RECORD id is not separating distinct records)")
    print(f"pool unique inputs     {payload['pool_unique_inputs']}")
    print(f"pool duplicate inputs  {payload['pool_duplicate_inputs']}  "
          "(repeated POSITIONS; always >= duplicate ids, by ~1000x on real data)")
    print(f"frozen rows            {payload['frozen_rows']} "
          f"(unique ids {payload['frozen_unique_ids']}, unique inputs "
          f"{payload['frozen_unique_inputs']}, seed {payload['sample_seed']})")
    print(f"row id version         {payload['row_id_version']}")

  # ⚑ Both refusals happen BEFORE the artifact is written. Codex #1: `freeze
  # --sample 100000` on a 48,360-row pool used to write 48,360 rows, print a
  # sha256 and exit 0 — and every downstream threshold is calibrated at
  # n=100,000, so the artifact itself has to refuse to exist at a smaller n.
    if payload["frozen_rows"] != payload["sample_requested"]:
        print(
            f"FAIL: asked for {payload['sample_requested']} rows and the pool "
            f"could only supply {payload['frozen_rows']}. The prereg's 0.392 pp "
            "bar and the ~0.2 pp paired resolution are both derived AT "
            "n=100,000; an artifact at a smaller n would silently apply them to "
            "an underpowered experiment. Convert more hours, or re-derive the "
            "bar at the n you actually have and pass --sample explicitly.",
            file=sys.stderr,
        )
        return 1
    if payload["frozen_rows"] != payload["frozen_unique_ids"]:
        print("FAIL: the frozen set contains duplicate row ids", file=sys.stderr)
        return 1

    digest = write_frozen(payload, Path(args.out))
    print(f"written                {args.out}")
    print(f"sha256                 {digest}")
    return 0


def _write_purity_receipt(
    path: Path, *, frozen: Path, train_shards: list[Path], result: object, pure: bool,
) -> str:
    """Bank WHAT was checked, not just that it passed.

    ⚑ REVIEW F5. `purity --train-shards ...` and `lc0_control_train --shards
    ...` were two free-floating CLI arguments with nothing tying them together,
    so the arm's headline held-out slope had no machine-checkable evidence that
    the corpus it TRAINED on is the corpus purity CLEARED. That is the same
    shape as the `game_id` purity check `lc0_control_rows.py` rejects: an
    assertion that can only fail if someone types the wrong thing twice — and
    here not even that. The receipt names the directories and the frozen
    artifact's sha256; `lc0_control_train --purity-receipt` refuses to launch
    on a corpus the receipt does not cover.
    """
    receipt: dict[str, object] = {
        "frozen": str(Path(frozen).resolve()),
        "frozen_sha256": hashlib.sha256(Path(frozen).read_bytes()).hexdigest(),
        "train_shards": sorted(str(Path(d).resolve()) for d in train_shards),
        "train_rows": int(getattr(result, "train_rows", 0)),
        "frozen_rows": int(getattr(result, "frozen_rows", 0)),
        "exposed_inputs": int(getattr(result, "exposed_inputs", 0)),
        "intersecting_ids": int(getattr(result, "intersecting_ids", 0)),
        "pure": bool(pure),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return str(receipt["frozen_sha256"])


def cmd_purity(args: argparse.Namespace) -> int:
    payload = load_frozen(Path(args.frozen))
    try:
        result = purity_against_train(
            payload["row_ids"], args.train_shards,
            frozen_input_ids=payload["input_ids"],
        )
    except EmptyTrainCorpus as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"frozen held-out rows {result.frozen_rows}")
    print(f"frozen distinct x    {result.frozen_inputs}")
    print(f"train rows scanned   {result.train_rows} "
          f"(unique ids {result.train_unique_ids}, "
          f"unique inputs {result.train_unique_inputs})")
    print(f"intersecting ids     {result.intersecting_ids}   "
          "<-- RECORD level; under-reports exposure")
    print(f"EXPOSED inputs       {result.exposed_inputs}   "
          f"({result.exposed_input_frac * 100:.4f}% of held-out x)   <-- THE GATE")
    if args.receipt is not None:
  # ⚑ Written on FAIL too, and it records `pure: false`. A receipt that only
  # exists when the news is good is an artifact you can produce by re-running
  # until it appears.
        digest = _write_purity_receipt(
            Path(args.receipt), frozen=Path(args.frozen),
            train_shards=list(args.train_shards), result=result,
            pure=result.is_pure,
        )
        print(f"receipt              {args.receipt}  (frozen sha256 {digest})")
    if result.is_pure:
        print("PURE: zero exposed inputs and zero record-id intersection")
        return 0
    print(f"examples             {list(result.examples)}")
    print(
        "FAIL: held-out inputs also occur in train. This set measures exposure "
        "recency, not generalisation — see "
        "memory/exposure_recency_dominates_heldout_ce.md. Rebuild the split.",
        file=sys.stderr,
    )
    return 1


def cmd_chance(args: argparse.Namespace) -> int:
    payload = load_frozen(Path(args.frozen))
    counts = legal_counts_for_ids(args.shards, payload["row_ids"])
    level = chance_level(counts)
    print(f"frozen rows          {level.rows}")
    print(f"mean n_legal         {level.mean_legal:.4f}")
    print(f"E[1/n_legal]         {level.expected_inverse:.6f}   <-- CHANCE TOP-1")
    print(f"1/E[n_legal]         {level.inverse_of_mean:.6f}   "
          "<-- WRONG (Jensen); shown only so it cannot be quoted by accident")
    print(f"ratio                {level.jensen_ratio:.4f}x  "
          "(how far the wrong one sits below the real floor)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze")
    freeze.add_argument("--shards", type=Path, nargs="+", required=True)
    freeze.add_argument("--out", type=Path, required=True)
    freeze.add_argument(
        "--sample", type=int, default=100_000,
        help="prereg resolution point: 100k paired resolves ~0.2 pp",
    )
    freeze.add_argument("--seed", type=int, default=0)
    freeze.set_defaults(handler=cmd_freeze)

    purity = sub.add_parser("purity")
    purity.add_argument("--frozen", type=Path, required=True)
    purity.add_argument("--train-shards", type=Path, nargs="+", required=True)
    purity.add_argument(
        "--receipt", type=Path, default=None,
        help="write a JSON receipt naming the frozen sha256 and the train "
             "directories actually scanned. `lc0_control_train "
             "--purity-receipt` refuses to launch on a corpus it does not "
             "cover — without one, nothing ties the trained rows to this check.",
    )
    purity.set_defaults(handler=cmd_purity)

    chance = sub.add_parser("chance")
    chance.add_argument("--frozen", type=Path, required=True)
    chance.add_argument("--shards", type=Path, nargs="+", required=True)
    chance.set_defaults(handler=cmd_chance)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
