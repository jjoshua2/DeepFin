"""Row identity and frozen row sets for the lc0 positive control.

⚑⚑ WHY THE ROW ID IS A CONTENT DIGEST AND NOT A NAME.

The held-out split for this arm is "the last 6 hourly tars by wall-clock". The
obvious purity check is "the train tar list and the held-out tar list are
disjoint", and it is worthless: it can only fail if someone types the same
filename twice. A gate that cannot fail is not a gate, and
`memory/exposure_recency_dominates_heldout_ce.md` records what a held-out set
that overlaps its train window actually costs — it measures forgetting and
recency, and reads as generalisation.

So the id is derived from THE ROW'S OWN CONTENT — its 175-plane input and its
1858-wide lc0 policy target — and never from the tar it arrived in, the shard
index, or the split it was assigned to. Two rows built from the same lc0 v6
record produce the same id no matter which conversion run produced them, so an
overlap is detectable; two rows from different records do not, so the check is
not trivially satisfied.

⚑ Two candidate ids were considered and one was rejected. `game_id` +
`ply_index` are stored in the shard and are cheap, but `game_id` is
`enumerate()`'s index over the games of ONE conversion invocation. Converting
train and held-out separately restarts it at 0, so those ids are disjoint BY
CONSTRUCTION and the assertion would pass on a set that is 100% contaminated.
That is the failure this module exists to avoid, not a cheaper way to reach it.

The RECORD id covers `x` and `policy_target` — the input and the label, i.e.
everything the yardstick reads. `x` alone would merge a transposition reached
with identical history; the policy floats come from a specific lc0 search and
break that tie. The claim that the pair discriminates is not left as an
argument: `frozen_row_set` reports `duplicate_ids`, the number of ids that
repeat WITHIN one split, and a non-zero count there is the id failing to
separate distinct records.

⚑⚑ BUT PURITY IS NOT A RECORD-IDENTITY QUESTION, IT IS AN EXPOSURE QUESTION,
AND THE RECORD ID IS THE WRONG INSTRUMENT FOR IT. "Did the net already see
this position" is answered by `x` ALONE. The same position reached in two
games gets two different lc0 visit distributions — search noise, a different
net generation — so the pair digest calls them distinct while the model
trained on exactly that input. Measured on the shipped smoke split:

    frozen held-out rows                          100,000
    INTERSECT by (x, policy_target) record id           0   <- reads PURE
    INTERSECT by x-only INPUT id                      450   <- actually exposed
    duplicate RECORD ids inside the held-out set         3
    duplicate INPUT ids inside the held-out set      3,153

450 exposed inputs at a train side of 48,360 rows — 0.06% of the full corpus.
The arm's entire readout is held-out generalisation, so an exposure gate that
under-reports biases the headline in the flattering direction.

So there are TWO ids and they answer different questions. `input_ids` is the
GATE (`PurityResult.is_pure`); `row_ids` remains the record identity used to
address a specific row when scoring. Both are reported, always, because the
gap between them is itself the diagnostic.

Cost, measured: ~31k rows/s single-threaded (blake2b-128 over 22.5 KB/row), so
~0.8 h for the full 87M-position corpus — against ~105 h for the conversion
that produces it.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from chess_anti_engine.replay.shard import iter_shard_paths, load_shard_arrays

# 128 bits. Over the full corpus (~8.7e7 rows) the birthday collision
# probability is ~1.4e-23, i.e. far below every other error source here.
ROW_ID_BYTES = 16

# Fields the digest covers, in this order. Changing this list changes every id
# and INVALIDATES every frozen set built with the old one, which is why the
# version below is stamped into the artifact and checked on load.
ROW_ID_FIELDS = ("x", "policy_target")
# ⚑ The EXPOSURE id. `x` alone, because "did the net already train on this
# input" does not depend on which label came with it.
INPUT_ID_FIELDS = ("x",)
ROW_ID_VERSION = "lc0_control_row_id_v2_x_policy_plus_x_only_blake2b128"


def _content_ids(arrs: dict[str, np.ndarray], fields: Sequence[str]) -> list[str]:
    columns = [np.ascontiguousarray(arrs[name]) for name in fields]
    n = columns[0].shape[0]
    for name, col in zip(fields, columns, strict=True):
        if col.shape[0] != n:
            raise ValueError(
                f"shard column {name!r} has {col.shape[0]} rows, expected {n}",
            )
    return [
        hashlib.blake2b(
            b"".join(col[i].tobytes() for col in columns), digest_size=ROW_ID_BYTES,
        ).hexdigest()
        for i in range(n)
    ]


def row_ids(arrs: dict[str, np.ndarray]) -> list[str]:
    """RECORD ids — ``(x, policy_target)``. Addresses a row; not the gate."""
    return _content_ids(arrs, ROW_ID_FIELDS)


def input_ids(arrs: dict[str, np.ndarray]) -> list[str]:
    """EXPOSURE ids — ``x`` alone. This is what the purity gate counts.

    ⚑ Deliberately merges transpositions and re-searches of the same position.
    That merge is the point: they are the same input, and the model that
    trained on one has seen the other.
    """
    return _content_ids(arrs, INPUT_ID_FIELDS)


def iter_shard_arrays(shard_dirs: Sequence[Path]) -> Iterator[tuple[Path, dict[str, np.ndarray]]]:
    """``(path, arrays)`` for every shard under every directory, in order.

    Directories are visited in the order given and shards within a directory in
    sorted order, so the traversal is deterministic and two runs over the same
    inputs produce the same id sequence.
    """
    for shard_dir in shard_dirs:
        for path in sorted(iter_shard_paths(Path(shard_dir))):
            arrs, _meta = load_shard_arrays(path)
            yield path, arrs


@dataclass
class SplitIds:
    """Every row id in a split, plus what it took to say that."""

    ids: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    rows: int = 0

    @property
    def unique_ids(self) -> int:
        return len(set(self.ids))

    @property
    def unique_inputs(self) -> int:
        return len(set(self.inputs))

    @property
    def duplicate_ids(self) -> int:
        """Rows whose RECORD id repeats within this split.

        ⚑ Report this, always. It is the id's own discrimination check: a
        content digest that collapses distinct records would inflate it, and
        the purity assertion downstream would then be comparing the wrong
        things while still reading "0 intersection".
        """
        return self.rows - self.unique_ids

    @property
    def duplicate_inputs(self) -> int:
        """Rows whose INPUT repeats within this split.

        ⚑ Always >= ``duplicate_ids``, and on real lc0 data by three orders of
        magnitude (3 vs 3,153 on the shipped smoke split). Reporting only the
        first invites reading "the id discriminates" as "the split has no
        repeated positions", which is a different and false claim.
        """
        return self.rows - self.unique_inputs


def collect_split_ids(shard_dirs: Sequence[Path]) -> SplitIds:
    """Record ids AND exposure (input) ids for every row under ``shard_dirs``."""
    split = SplitIds()
    for path, arrs in iter_shard_arrays(shard_dirs):
        ids = row_ids(arrs)
        split.ids.extend(ids)
        split.inputs.extend(input_ids(arrs))
        split.sources.append(str(path.name))
        split.rows += len(ids)
    return split


def _stratified_sample(
    per_source: list[list[tuple[str, str]]], *, sample: int, seed: int,
) -> list[tuple[str, str]]:
    """Take ``sample`` ids spread proportionally across sources.

    ⚑ A flat random draw over the pooled ids would still be *time-disjoint*
    from train, so it would not break purity — but it would let one hour
    dominate the frozen set by luck, and the prereg's whole reason for taking
    SIX hours is that a single hour is one net generation's worth of a
    correlated stream. Proportional-by-source keeps every hour represented.
    """
    total = sum(len(chunk) for chunk in per_source)
    if sample >= total:
        return [row for chunk in per_source for row in chunk]
    rng = np.random.default_rng(seed)
    picked: list[tuple[str, str]] = []
    remaining = sample
    for index, chunk in enumerate(per_source):
        sources_left = len(per_source) - index
        want = remaining // sources_left if sources_left > 1 else remaining
        want = min(want, len(chunk))
        take = rng.choice(len(chunk), size=want, replace=False)
        picked.extend(chunk[int(i)] for i in sorted(take))
        remaining -= want
    return picked


def frozen_row_set(
    shard_dirs: Sequence[Path], *, sample: int, seed: int,
) -> dict[str, Any]:
    """Build the frozen held-out artifact (ids + everything needed to audit it).

    The returned mapping is what gets written to disk and sha256'd. It carries
    the id VERSION so a later run cannot silently compare ids built under a
    different digest definition, and the per-source row counts so a reader can
    see the six hours are all present without re-reading the shards.
    """
    per_source: list[list[tuple[str, str]]] = []
    source_names: list[str] = []
    source_rows: list[int] = []
    all_ids: list[str] = []
    all_inputs: list[str] = []
    for shard_dir in shard_dirs:
        split = collect_split_ids([shard_dir])
        per_source.append(list(zip(split.ids, split.inputs, strict=True)))
        source_names.append(Path(shard_dir).name)
        source_rows.append(split.rows)
        all_ids.extend(split.ids)
        all_inputs.extend(split.inputs)
    pool = SplitIds(ids=all_ids, inputs=all_inputs, rows=len(all_ids))
    frozen = _stratified_sample(per_source, sample=sample, seed=seed)
    frozen_ids = [row_id for row_id, _ in frozen]
    frozen_inputs = [input_id for _, input_id in frozen]
    return {
        "row_id_version": ROW_ID_VERSION,
        "row_id_fields": list(ROW_ID_FIELDS),
        "input_id_fields": list(INPUT_ID_FIELDS),
        "sample_seed": seed,
        "sample_requested": sample,
        "pool_rows": pool.rows,
        "pool_unique_ids": pool.unique_ids,
        "pool_duplicate_ids": pool.duplicate_ids,
        "pool_unique_inputs": pool.unique_inputs,
        "pool_duplicate_inputs": pool.duplicate_inputs,
        "sources": source_names,
        "source_rows": source_rows,
        "frozen_rows": len(frozen),
        "frozen_unique_ids": len(set(frozen_ids)),
        "frozen_unique_inputs": len(set(frozen_inputs)),
        "row_ids": frozen_ids,
        "input_ids": frozen_inputs,
    }


def artifact_sha256(payload: dict[str, Any]) -> str:
    """sha256 of the artifact's canonical serialisation.

    Canonical = the exact bytes `write_frozen` puts on disk, so the recorded
    digest is a digest of the FILE and can be re-checked with `sha256sum`.
    """
    return hashlib.sha256(serialize_frozen(payload)).hexdigest()


def serialize_frozen(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def write_frozen(payload: dict[str, Any], path: Path) -> str:
    """Write the frozen set and return its sha256."""
    blob = serialize_frozen(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return hashlib.sha256(blob).hexdigest()


def load_frozen(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = payload.get("row_id_version")
    if version != ROW_ID_VERSION:
        raise ValueError(
            f"frozen row set {path} was built with row id version {version!r}, "
            f"but this code builds {ROW_ID_VERSION!r}. The ids are not "
            "comparable; rebuild the frozen set or check out the code that "
            "made it. Silently proceeding would compare two different "
            "identities and report a clean purity check over nothing.",
        )
    if len(payload.get("input_ids", ())) != len(payload.get("row_ids", ())):
        raise ValueError(
            f"frozen row set {path} carries {len(payload.get('input_ids', ()))} "
            f"exposure (input) ids for {len(payload.get('row_ids', ()))} rows. "
            "The purity GATE is the x-only exposure count; a set without one "
            "can only be checked by record id, which under-reports exposure "
            "(measured 0 vs 450 on the shipped smoke split). Re-freeze.",
        )
    return payload


class EmptyTrainCorpus(ValueError):
    """The purity check found no training rows to compare against.

    ⚑ A "0 intersections" result over 0 scanned rows is not a clean split, it
    is an unperformed check — and it exits 0, prints PURE, and licenses a
    contaminated held-out set (Codex #6, reproduced verbatim by the reviewer).
    """


@dataclass(frozen=True)
class PurityResult:
    """Overlap between a frozen held-out set and a train corpus.

    ⚑ TWO instruments, deliberately. ``exposed_inputs`` is the GATE — the
    x-only count of held-out inputs the model would already have trained on.
    ``intersecting_ids`` is the record-level count and is reported for
    continuity; it under-reports exposure by construction, because the same
    position carrying a different lc0 policy target is a different record.
    """

    frozen_rows: int
    frozen_inputs: int
    train_rows: int
    train_unique_ids: int
    train_unique_inputs: int
    intersecting_ids: int
    exposed_inputs: int
    examples: tuple[str, ...]

    @property
    def exposed_input_frac(self) -> float:
        if self.frozen_inputs <= 0:
            return 0.0
        return self.exposed_inputs / self.frozen_inputs

    @property
    def is_pure(self) -> bool:
        """Zero EXPOSURE. Record-id agreement is not sufficient."""
        return self.exposed_inputs == 0 and self.intersecting_ids == 0


def purity_against_train(
    frozen_ids: Sequence[str],
    train_shard_dirs: Sequence[Path],
    *,
    frozen_input_ids: Sequence[str],
) -> PurityResult:
    """Count frozen rows the train corpus also contains, by record AND by input.

    Raises ``EmptyTrainCorpus`` when the train side turns out to hold no rows —
    a check that scanned nothing must not be able to report PURE.
    """
    frozen = set(frozen_ids)
  # ⚑ No default. An omitted exposure list would make ``exposed_inputs`` 0 for
  # every input and put the gate straight back where F5 found it.
    frozen_inputs = set(frozen_input_ids)
    seen: set[str] = set()
    seen_inputs: set[str] = set()
    hits: set[str] = set()
    exposed: set[str] = set()
    rows = 0
    for _path, arrs in iter_shard_arrays(train_shard_dirs):
        ids = row_ids(arrs)
        rows += len(ids)
        seen.update(ids)
        hits.update(frozen.intersection(ids))
        train_inputs = input_ids(arrs)
        seen_inputs.update(train_inputs)
        exposed.update(frozen_inputs.intersection(train_inputs))
    if rows == 0:
        raise EmptyTrainCorpus(
            f"purity scanned {len(list(train_shard_dirs))} train directory(ies) "
            "and found ZERO rows. That is not a pure split, it is an "
            "unperformed check: with no train rows the intersection is 0 for "
            "any held-out set, including a 100% contaminated one. Check the "
            "--train-shards paths.",
        )
    return PurityResult(
        frozen_rows=len(frozen),
        frozen_inputs=len(frozen_inputs),
        train_rows=rows,
        train_unique_ids=len(seen),
        train_unique_inputs=len(seen_inputs),
        intersecting_ids=len(hits),
        exposed_inputs=len(exposed),
        examples=tuple(sorted(hits | exposed)[:5]),
    )


@dataclass(frozen=True)
class ChanceLevel:
    """Chance top-1 accuracy over a set of variable-size legal-move sets."""

    rows: int
    mean_legal: float
    expected_inverse: float
    inverse_of_mean: float

    @property
    def jensen_ratio(self) -> float:
        """How far ``1/E[n]`` sits BELOW the true floor. Always >= 1."""
        if self.inverse_of_mean <= 0.0:
            return float("nan")
        return self.expected_inverse / self.inverse_of_mean


def chance_level(legal_counts: np.ndarray) -> ChanceLevel:
    """``E[1/n_legal]`` — the chance top-1 rate. NOT ``1/E[n_legal]``.

    ⚑ Jensen. A uniform-random guesser on a position with ``n`` legal moves is
    right with probability ``1/n``; its overall rate is the AVERAGE of those
    per-position probabilities, ``E[1/n]``. ``1/E[n]`` is the reciprocal of the
    average branching factor and is a strictly SMALLER number whenever ``n``
    varies at all — so using it puts the negative control's floor UNDER the
    real floor and a shuffled-label run can clear a gate it should fail. Both
    are returned, with their ratio, so the wrong one cannot be quoted by
    accident and the size of the error is on the record.
    """
    counts = np.asarray(legal_counts, dtype=np.float64).reshape(-1)
    positive = counts[counts > 0.0]
    if positive.size == 0:
        return ChanceLevel(rows=0, mean_legal=float("nan"),
                           expected_inverse=float("nan"), inverse_of_mean=float("nan"))
    mean_legal = float(positive.mean())
    return ChanceLevel(
        rows=int(positive.size),
        mean_legal=mean_legal,
        expected_inverse=float((1.0 / positive).mean()),
        inverse_of_mean=1.0 / mean_legal,
    )


def legal_counts_for_ids(
    shard_dirs: Sequence[Path], wanted: Sequence[str],
) -> np.ndarray:
    """Legal-move counts for exactly the rows named by ``wanted``.

    ⚑ Scoped to the FROZEN rows, not to the shards they live in. The chance
    floor has to describe the population the gate is read on; computing it over
    the whole held-out pool would be a different set with a different branching
    distribution, and the difference is invisible in the number itself.
    """
    want = set(wanted)
    found: dict[str, int] = {}
    for _path, arrs in iter_shard_arrays(shard_dirs):
        ids = row_ids(arrs)
        legal = np.asarray(arrs["legal_mask"]).astype(np.int64).sum(axis=1)
        for index, row_id in enumerate(ids):
            if row_id in want and row_id not in found:
                found[row_id] = int(legal[index])
    missing = len(want) - len(found)
    if missing:
        raise ValueError(
            f"{missing} of {len(want)} frozen row ids were not found in the "
            "given shard dirs. The frozen set and the shards do not match — "
            "recompute one of them rather than scoring a partial set.",
        )
    return np.array([found[row_id] for row_id in wanted], dtype=np.int64)
