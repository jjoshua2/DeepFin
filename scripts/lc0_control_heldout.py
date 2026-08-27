#!/usr/bin/env python3
"""Freeze, audit and floor the lc0 positive control's held-out split.

Three subcommands, in the order they must be run:

  freeze  --shards <the 6 newest hourly dirs> --out <frozen.json>
          Builds the explicit row-id list and prints its sha256. ⚑ EXACTLY the
          preregistered six sources, or `--allow-source-selection` and the
          artifact is stamped as a non-preregistered population — the prereg
          names "the LAST 6 hourly tars by wall-clock", and until 2026-08-17 the
          ROW COUNT was gated while the POPULATION was not.

  purity  --frozen <frozen.json> --train-shards <train dirs> [--receipt <f>]
          [--exposed-out <f>] [--cache-dir <d>] [--workers N]
          Asserts ZERO EXPOSURE — no held-out INPUT (`x` alone) occurs
          anywhere in train — and reports the record-id intersection
          alongside it. Exit 1 on any overlap, and exit 1 when the train side
          turns out to hold no rows at all. `--receipt` banks WHICH
          directories were scanned and the frozen artifact's sha256, so the
          training driver can refuse a corpus this check never saw.
          `--exposed-out` banks the exposed ids THEMSELVES — see `subtract`.
          `--cache-dir` banks the train id index, keyed by a fingerprint of
          every file in the corpus, so a re-check costs seconds.

  subtract --frozen <frozen.json> --exposed <exposed.json> --out <clean.json>
          Rebuilds the frozen set as the rows whose INPUT is not exposed,
          preserving order and schema. The banked per-row score files are
          keyed by `row_id`, so restricting them to the clean set is masking,
          not re-scoring: the same nets, no GPU, strictly more comparable.

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
    PREREG_HELDOUT_SOURCES,
    EmptyTrainCorpus,
    chance_level,
    duplicate_resolved_dirs,
    exposed_rows,
    frozen_minus_exposed,
    frozen_row_set,
    legal_counts_for_ids,
    load_frozen,
    purity_against_train,
    source_selection_problems,
    write_frozen,
)


def cmd_freeze(args: argparse.Namespace) -> int:
  # ⚑ FIRST, before a single shard is read: a repeated directory is a property of
  # the ARGUMENTS, and reading the corpus to discover it costs minutes on the real
  # hours. Refused unconditionally, exactly as on the training side —
  # `--allow-source-selection` declares a different NUMBER of hours, which is a
  # legitimate smoke choice, while naming one hour twice is not a population
  # anybody chose, so no flag reaches it.
  #
  # ⚑⚑ AND IT MUST PRECEDE THE ROW-ID GATE. `freeze --shards h h h h h h` was
  # stopped by `FAIL: the frozen set contains duplicate row ids` — a gate about the
  # ID FUNCTION, firing on an input that is wrong for an entirely different reason,
  # so the operator is told the wrong thing and the population gate reads clean.
    repeated = duplicate_resolved_dirs(args.shards)
    if repeated:
        print(
            "FAIL: these --shards directories are named more than once "
            "(resolved): " + ", ".join(repeated)
            + ". One hour named N times is not N hours: it is a single net "
            "generation's worth of a correlated stream, which is the population "
            "the prereg's SIX exists to avoid. Name each hour once.",
            file=sys.stderr,
        )
        return 1
    payload = frozen_row_set(args.shards, sample=int(args.sample), seed=int(args.seed))
  # ⚑ RESOLVED paths, not basenames: six distinct hours all called `h` printed
  # (and banked) as `h` six times, which is unauditable against the prereg's
  # actual claim — the LAST 6 hourly tars BY WALL-CLOCK.
    print(f"sources ({len(payload['source_paths'])}):")
    for path, rows in zip(payload["source_paths"], payload["source_rows"],
                          strict=True):
        print(f"  {path}: {rows} rows")
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

  # ⚑ REQUIRE THE PREREGISTERED SIX, OR STAMP THE ARTIFACT. Refusing by default
  # is the half that matters: an unstamped artifact from a non-preregistered
  # source selection could not exist, so nothing downstream has to guess. The
  # stamp is not decoration either — `lc0_control_eval score` banks it and
  # `compare` refuses it unless `--allow-non-prereg-heldout` says the operator
  # meant it. Same shape as `--allow-arch-drift`: the deviation is possible,
  # declared, and carried in every artifact derived from it.
  #
  # ⚑⚑ COUNTED AS DISTINCT RESOLVED PATHS. The first version counted the LIST, so
  # `freeze --shards h h h h h h` — one hour named six times, the exact input the
  # SIX exists to rule out — passed this gate and was stopped only by the
  # unrelated duplicate-row-id refusal below. `source_selection_problems` now
  # reads `source_paths` (resolved) and de-duplicates with
  # `duplicate_resolved_dirs`, the same function the training driver uses for the
  # same hazard on `--shards`; that function existed in the same commit as the
  # defect and was not reused. Independent review of #438.
    selection_problems = source_selection_problems(payload)
    if selection_problems and not args.allow_source_selection:
        print(
            "FAIL: " + " | ".join(selection_problems)
            + " Point --shards at the preregistered six hours, or pass "
            "--allow-source-selection to stamp this artifact as a "
            "non-preregistered population (every score and comparison derived "
            "from it then carries that stamp).",
            file=sys.stderr,
        )
        return 1
    payload["source_selection_problems"] = selection_problems
    payload["preregistered_source_selection"] = not selection_problems
    if selection_problems:
        print("⚑⚑ --allow-source-selection: NOT THE PREREGISTERED HELD-OUT "
              "POPULATION — " + " | ".join(selection_problems))

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
        "train_index_cached": bool(getattr(result, "train_index_cached", False)),
        "pure": bool(pure),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return str(receipt["frozen_sha256"])


def _write_exposed_dump(
    path: Path, *, frozen: Path, payload: dict[str, object],
    train_shards: list[Path], result: object,
) -> int:
    """Bank the exposed ids THEMSELVES, and the held-out rows carrying them.

    ⚑ THE BLOCKER THIS CLOSES. The 2026-08-16 scan read 42.2 GB over 2h40m,
    found 5,065 exposed inputs, and banked the NUMBER 5065 plus five example
    hashes. Repairing the split is then a subtraction whose operand no longer
    exists, so the only way to rebuild it was to repeat the scan. A count is a
    lossy summary of a set; the gate's job is not finished when it has decided
    PASS/FAIL, because the FAIL branch has a next step.
    """
    exposed_ids = list(getattr(result, "exposed_input_ids", ()))
    rows = exposed_rows(payload, exposed_ids)
    dump = {
        "frozen": str(Path(frozen).resolve()),
        "frozen_sha256": hashlib.sha256(Path(frozen).read_bytes()).hexdigest(),
        "row_id_version": payload.get("row_id_version"),
        "train_shards": sorted(str(Path(d).resolve()) for d in train_shards),
        "exposed_inputs": int(getattr(result, "exposed_inputs", 0)),
        "intersecting_ids": int(getattr(result, "intersecting_ids", 0)),
        "exposed_rows": len(rows),
        "exposed_input_ids": sorted(exposed_ids),
        "intersecting_row_ids": sorted(getattr(result, "intersecting_row_ids", ())),
        "rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dump, indent=2, sort_keys=True), encoding="utf-8")
    return len(rows)


def _progress(done: int, total: int, rows: int) -> None:
    if done % 200 == 0 or done == total:
        print(f"  scanned {done}/{total} shards, {rows} rows", flush=True)


def cmd_purity(args: argparse.Namespace) -> int:
    payload = load_frozen(Path(args.frozen))
    try:
        result = purity_against_train(
            payload["row_ids"], args.train_shards,
            frozen_input_ids=payload["input_ids"],
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
            workers=int(args.workers),
            progress=_progress,
        )
    except EmptyTrainCorpus as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"train index          "
          f"{'CACHED' if result.train_index_cached else 'rebuilt by scanning'}")
    print(f"frozen held-out rows {result.frozen_rows}")
    print(f"frozen distinct x    {result.frozen_inputs}")
    print(f"train rows scanned   {result.train_rows} "
          f"(unique ids {result.train_unique_ids}, "
          f"unique inputs {result.train_unique_inputs})")
    print(f"intersecting ids     {result.intersecting_ids}   "
          "<-- RECORD level; under-reports exposure")
    print(f"EXPOSED inputs       {result.exposed_inputs}   "
          f"({result.exposed_input_frac * 100:.4f}% of held-out x)   <-- THE GATE")
    if args.exposed_out is not None:
  # ⚑ Written on PASS too, where the lists are empty. A dump that only
  # exists on failure makes "no file" ambiguous between "the set was clean"
  # and "the flag was forgotten".
        rows = _write_exposed_dump(
            Path(args.exposed_out), frozen=Path(args.frozen), payload=payload,
            train_shards=list(args.train_shards), result=result,
        )
        print(f"exposed dump         {args.exposed_out}  "
              f"({result.exposed_inputs} inputs on {rows} held-out rows)")
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


def cmd_subtract(args: argparse.Namespace) -> int:
    """Rebuild the frozen set as the rows whose INPUT is not exposed."""
    payload = load_frozen(Path(args.frozen))
    dump = json.loads(Path(args.exposed).read_text(encoding="utf-8"))
    if dump.get("row_id_version") != payload.get("row_id_version"):
        print(
            f"FAIL: the exposed dump was built under row id version "
            f"{dump.get('row_id_version')!r} and the frozen set under "
            f"{payload.get('row_id_version')!r}. Subtracting one from the other "
            "would remove nothing and report a clean set.",
            file=sys.stderr,
        )
        return 1
    digest = hashlib.sha256(Path(args.frozen).read_bytes()).hexdigest()
    if dump.get("frozen_sha256") != digest:
  # ⚑ The dump names the artifact it was computed against. Subtracting a
  # different run's exposure list is the same class of error the purity
  # receipt exists to stop: two free-floating CLI paths with nothing tying
  # them together. A MISSING sha is refused rather than waved through — a
  # check that an absent field satisfies is a check that cannot fail.
        print(
            f"FAIL: the exposed dump was computed against frozen sha256 "
            f"{dump.get('frozen_sha256')}, but --frozen has sha256 {digest}.",
            file=sys.stderr,
        )
        return 1
    exposed = list(dump.get("exposed_input_ids", ()))
    predicted_inputs = int(payload["frozen_unique_inputs"]) - len(exposed)
    trimmed = frozen_minus_exposed(payload, exposed)
    print(f"source rows          {payload['frozen_rows']} "
          f"(distinct x {payload['frozen_unique_inputs']})")
    print(f"exposed inputs       {len(exposed)}")
    print(f"predicted distinct x {predicted_inputs}   <-- stated BEFORE the build")
    print(f"surviving rows       {trimmed['frozen_rows']} "
          f"(distinct x {trimmed['frozen_unique_inputs']})")
    print(f"removed rows         {trimmed['removed_rows']}")
    if trimmed["frozen_unique_inputs"] != predicted_inputs:
  # Not a formality. The distinct-input count is exactly determined by the
  # subtraction, so a mismatch means the dump and the frozen set are not
  # describing the same population and nothing downstream is trustworthy.
        print(
            f"FAIL: expected {predicted_inputs} distinct inputs to survive "
            f"({payload['frozen_unique_inputs']} - {len(exposed)}) but got "
            f"{trimmed['frozen_unique_inputs']}. Some exposed input is not in "
            "this frozen set, or the two artifacts disagree.",
            file=sys.stderr,
        )
        return 1
    if trimmed["frozen_rows"] != trimmed["frozen_unique_ids"]:
        print("FAIL: the trimmed set contains duplicate row ids", file=sys.stderr)
        return 1
    if trimmed["frozen_rows"] == 0:
        print("FAIL: every row was exposed; there is no held-out set left",
              file=sys.stderr)
        return 1
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"FAIL: {out} already exists. The frozen set a result is recorded "
              "against must never be overwritten in place; pass a new path.",
              file=sys.stderr)
        return 1
    trimmed["derived_from"] = str(Path(args.frozen).resolve())
    trimmed["derived_from_sha256"] = digest
    trimmed["exposed_dump"] = str(Path(args.exposed).resolve())
    written = write_frozen(trimmed, out)
    print(f"written              {out}")
    print(f"sha256               {written}")
    return 0


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
    freeze.add_argument(
        "--allow-source-selection", action="store_true",
        help=f"⚑⚑ build the frozen set from a number of source directories "
             f"other than the preregistered {PREREG_HELDOUT_SOURCES} hourly "
             "tars. The artifact is STAMPED, `lc0_control_eval score` banks the "
             "stamp, and `compare` refuses it unless "
             "--allow-non-prereg-heldout. Without this flag such a set is "
             "refused and no file is written.",
    )
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
    purity.add_argument(
        "--exposed-out", type=Path, default=None,
        help="write the exposed INPUT ids and the held-out row_ids carrying "
             "them. Without it the gate banks a count and the repair needs a "
             "second full scan to recover the operand it already computed.",
    )
    purity.add_argument(
        "--cache-dir", type=Path, default=None,
        help="reuse/bank the train id index here, keyed by a fingerprint of "
             "every file in the corpus. A mismatch rescans; there is no "
             "'probably fine' branch.",
    )
    purity.add_argument(
        "--workers", type=int, default=1,
        help="parallel shard scanners (spawned). The result is a pair of sets "
             "and a row count, so it does not depend on completion order.",
    )
    purity.set_defaults(handler=cmd_purity)

    subtract = sub.add_parser("subtract")
    subtract.add_argument("--frozen", type=Path, required=True)
    subtract.add_argument("--exposed", type=Path, required=True)
    subtract.add_argument("--out", type=Path, required=True)
    subtract.add_argument(
        "--force", action="store_true",
        help="allow --out to overwrite an existing file",
    )
    subtract.set_defaults(handler=cmd_subtract)

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
