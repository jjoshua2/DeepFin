#!/usr/bin/env python3
"""Paired comparison of two per-position yardstick dumps, with bootstrap CIs.

Every kill/hold decision in docs/experiment_ledger.md compares two checkpoint
reads on the same frozen positions. Comparing the two MEANS throws away the
pairing; this tool joins the dumps position-by-position and reports the paired
mean delta with a bootstrap confidence interval — typically several times
tighter than the naive two-means comparison, and it makes the ledger's cp
thresholds statistically meaningful.

Inputs: two JSONL per-position dumps. Supported sources:

  scripts/value_regret.py --dump-per-position   (defaults: join on ``fen``,
    compare the ``value`` field)
  scripts/audit_targets.py --dump-per-position  (join on ``key``; pick the
    metric with a dotted --field path), e.g.:
      --join-key key --field cand.search.exp   # net+search E[regret]
      --join-key key --field cand.raw.top1     # raw net top-1 regret
    (the raw net candidate is deterministic per checkpoint; the search
    candidate re-runs Gumbel search each audit, so its paired delta still
    carries search-seed noise on top of position pairing)

Rows missing from either side, and rows whose metric is null / missing /
non-finite, are dropped. The report accounts for both PER SIDE — read the
header as ``rows = unusable + indexed`` and ``indexed = paired + unmatched``,
and check it against the input files before trusting a verdict. Duplicate join
keys are refused outright — see ``load_dump``. ``phase`` (int index or string)
groups the per-phase breakdown.

⚑ THE TWO DUMPS MUST COME FROM THE SAME RULER. Both sides' ruler stamps
(``input_encoding``, ``batch_size``) are compared before anything is joined and
a disagreement is REFUSED, because a paired delta across two rulers measures
the ruler rather than the checkpoints. An unstamped dump counts as
``input_encoding=fen_only`` — every dump predating the audit-v2 flag is, by
construction — so legacy-vs-``stored`` is refused too; only ``batch_size`` is
warn-only when absent. The inferred stamp adopts the counterpart's SHAPE, since
``audit_targets`` stamps one encoding per candidate where ``value_regret``
stamps a scalar. See ``require_same_ruler`` and ``_match_stamp_shape``.

Sign convention: delta = A - B per position. For regret-style metrics (lower
is better), a NEGATIVE mean delta means A is better.
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Any, NamedTuple

import numpy as np

# ⚑ The LEAF, not `eval.audit_cache`: that package's __init__ imports
# `.puzzles` -> torch, which costs ~4.0 s and ~750 MB for one string
# constant. This module is deliberately stdlib+numpy and
# `scripts/monitor_fen.sh` runs it against the live training box.
from chess_anti_engine.utils.audit_cache_format import (
    AUDIT_CACHE_FORMAT,
    CORE_STAMP_KEYS,
    ROW_COUNT_KEY,
    STAMP_FORMAT_KEY,
    STAMP_NON_IDENTITY_KEYS,
    is_stamp_record,
)

PHASE_NAMES = ("endgame", "middlegame", "opening")


def paired_bootstrap_ci(
    deltas: np.ndarray, *, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired deltas."""
    rng = np.random.default_rng(seed)
    n = deltas.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    means = deltas[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def get_field(rec: dict, path: str) -> object | None:
    """Resolve a dotted path (``cand.search.exp``) inside a dump record."""
    cur: object = rec
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def phase_label(p: object) -> str:
    if isinstance(p, int) and not isinstance(p, bool) and 0 <= p < len(PHASE_NAMES):
        return PHASE_NAMES[p]
    return str(p)


class Dump(NamedTuple):
    """One per-position dump, indexed for the join.

    ``unusable`` counts the rows that were read but could not enter the index
    — no join key, or a null / missing / non-finite metric. It rides along
    because ``report`` cannot recover it from ``rows`` and used to print a
    ``dropped`` figure that silently excluded it.
    """

    rows: dict[str, tuple[float, str]]
    unusable: int
    # RULER PROVENANCE: field -> the distinct values seen across the dump's
    # rows. Empty when the dump predates provenance stamping.
    provenance: dict[str, set[str]]
    # The dump's provenance HEADER, verbatim. Empty when the dump predates
    # stamping. Compared across the pair by `require_same_stamp`.
    stamp: dict[str, Any]


def load_dump(
    path: str, *, join_key: str = "fen", field: str = "value",
) -> Dump:
    """Index one per-position dump by its join key.

    **A metric must be finite, not merely numeric.** ``isinstance(v, (int,
    float))`` alone admits NaN, and one NaN row poisons the entire comparison:
    numpy's ``mean`` and ``percentile`` both propagate it, so the delta and
    both CI bounds print as ``nan``, and since ``nan < 0`` and ``nan > 0`` are
    both False the verdict falls through to "NOT significant". Demonstrated on
    a 50-row pair: a single NaN turned a clean −5.0 delta into "NOT
    significant" — silently converting a KILL into a HOLD, which is the worst
    direction for a tool every ledger verdict is read off. Non-finite rows are
    now dropped and counted like nulls.

    **Duplicate join keys are refused.** A dump is one deterministic read of
    one checkpoint over a frozen position set, so a repeated key means the file
    is not what the join assumes — two runs concatenated, a re-run appended, or
    the wrong ``--join-key``. Before this check the dict build made duplicates
    last-win and the losers were invisible: not in ``common``, not in the
    reported ``dropped`` either, so the caller read a clean join over a
    silently smaller and silently biased sample (audit invariant L14). There is
    no principled winner between two rows claiming the same position, so the
    tool stops instead of guessing. Rows dropped as unusable never enter the
    index and so cannot trip it.
    """
    rows: dict[str, tuple[float, str]] = {}
    duplicates: list[str] = []
    unusable = 0
    stamp: dict[str, Any] = {}
    provenance: dict[str, set[str]] = {f: set() for f in RULER_FIELDS}
    n_data_lines = 0
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            r = json.loads(line)
            # ⚑ The provenance HEADER is not a data row. `audit_targets
            # --dump-per-position` writes its dump through
            # `chess_anti_engine.eval.audit_cache`, whose stamp occupies line 1
            # and legitimately carries keys that also exist on data rows —
            # `input_encoding` among them, as a SCALAR where the rows hold a
            # per-candidate DICT. Counting it here made a single-ruler dump look
            # like two, and `require_same_ruler` refused it with a diagnosis
            # ("this dump mixes two rulers within itself") that named the wrong
            # culprit entirely. It also inflated `unusable` by one, breaking the
            # `rows = unusable + indexed` arithmetic this module's own docstring
            # tells operators to check before trusting a verdict.
            #
            # Skipping on the SENTINEL rather than special-casing
            # `input_encoding` is deliberate: it stays correct for every stamp
            # key added later, and no data row can carry the sentinel.
            if is_stamp_record(r):
                # ⚑ CAPTURED, not merely skipped. Skipping was right for the
                # ROW loop — the stamp's `input_encoding` is a scalar where
                # rows hold a per-candidate dict, so folding it into
                # `provenance` is what made a single-ruler dump look like two.
                # But "not a data row" is not "not evidence": until this was
                # captured, `paired_compare` RECOGNISED the stamp and then
                # discarded it, so two dumps declaring DIFFERENT
                # `audit_ruler_version` joined to exit 0 and printed a verdict
                # under a banner that reads as a provenance certificate.
                # Reviewer-confirmed by execution on PR #423.
                if stamp:
                    # ⚑ A SECOND header, not an overwrite. `stamp = dict(r)`
                    # was last-wins, so a file made of two caches concatenated
                    # kept only the LAST header and silently discarded the
                    # first — including a disagreeing `audit_ruler_version`.
                    # `read_audit_cache` already refuses this exact shape; the
                    # comparison tool has no business being laxer than the
                    # reader whose files it consumes.
                    raise SystemExit(
                        f"{path}: line {lineno} is a SECOND provenance header. "
                        "This looks like two dumps concatenated, which the "
                        "line-1 stamp cannot describe — and the first header's "
                        "provenance would be silently discarded. Split them, or "
                        "re-dump."
                    )
                if n_data_lines:
                    # ⚑ A HEADER THAT FOLLOWS ROWS DOES NOT DESCRIBE THEM.
                    # Without this, an unstamped dump with a stamped one
                    # appended reads as a single stamped file: the sole header
                    # is accepted as line-1 provenance, and if its declared
                    # `rows` happens to cover the whole body the count guard
                    # passes too — certifying rows written before the stamp
                    # existed. `write_audit_cache` always emits the header
                    # first and `read_audit_cache_stamp` reads only line 1, so
                    # a later header is not a shape any writer produces.
                    raise SystemExit(
                        f"{path}: the provenance header is on line {lineno}, "
                        f"after {n_data_lines} data rows. A stamp certifies the "
                        "body that FOLLOWS it — rows above it were written "
                        "before it existed and are not covered by it. This "
                        "looks like an unstamped dump with a stamped one "
                        "appended. Split them, or re-dump."
                    )
                stamp = dict(r)
                declared_format = r.get(STAMP_FORMAT_KEY)
                if declared_format != AUDIT_CACHE_FORMAT:
                    # ⚑ The value, not merely the key. `if STAMP_FORMAT_KEY in r`
                    # is a PRESENCE test, and presence was the whole check: two
                    # dumps at formats 1 and 99 compared clean and exited 0
                    # (measured). A format this reader does not know is a stamp
                    # it cannot interpret, so every downstream identity check
                    # over it is unsound.
                    raise SystemExit(
                        f"{path}: provenance stamp declares "
                        f"{STAMP_FORMAT_KEY}={declared_format!r}, but this tool "
                        f"understands {AUDIT_CACHE_FORMAT}. The stamp's layout "
                        "is not the one these checks assume, so its identity "
                        "fields cannot be compared. Update the tool, or re-dump."
                    )
                continue
            n_data_lines += 1
            for pf in RULER_FIELDS:
                if pf in r:
                    # json.dumps so a dict-valued stamp (audit_targets records
                    # one encoding per candidate) compares by value, not by id.
                    provenance[pf].add(json.dumps(r[pf], sort_keys=True))
            k = r.get(join_key)
            v = get_field(r, field)
            if k is None or not isinstance(v, (int, float)) or not math.isfinite(v):
                unusable += 1
                continue
            key = str(k)
            if key in rows:
                duplicates.append(key)
                continue
            rows[key] = (float(v), phase_label(r.get("phase", "?")))
    if stamp:
        # ⚑ THE STAMP BINDS TO LINE 1 ONLY, so without this a header lifted
        # from a good dump certifies a TRUNCATED body — `read_audit_cache`
        # enforces the count for exactly that reason and `paired_compare` did
        # not, so a stamp declaring 9999 rows over an 8-row file exited 0 with
        # a verdict (measured). A short dump is not a small sample here: it is
        # a file that is not what its provenance says it is.
        declared_rows = stamp.get(ROW_COUNT_KEY)
        if not isinstance(declared_rows, int) or isinstance(declared_rows, bool):
            raise SystemExit(
                f"{path}: provenance stamp carries no integer "
                f"'{ROW_COUNT_KEY}' count (found {declared_rows!r}), so the "
                "header cannot vouch for the body it is stamped onto."
            )
        if declared_rows != n_data_lines:
            raise SystemExit(
                f"{path}: stamp declares {declared_rows} rows but the file "
                f"holds {n_data_lines} — TRUNCATED, appended to, or carrying a "
                "stamp lifted from another dump. Refusing to read a verdict off "
                "a body its own provenance does not describe."
            )
    if duplicates:
        unique_dupes = sorted(set(duplicates))
        raise SystemExit(
            f"{path}: {len(duplicates)} duplicate rows across "
            f"{len(unique_dupes)} repeated '{join_key}' values, e.g. "
            f"{unique_dupes[:3]}. A paired comparison cannot join an ambiguous "
            f"key — de-duplicate the dump (or pass the right --join-key) and "
            f"re-run. Refusing rather than silently dropping them."
        )
    return Dump(rows, unusable, provenance, stamp)


# Stamps that identify WHICH RULER produced a dump. A change to any of them
# invalidates the dump's records, so two dumps that disagree cannot be joined.
# `input_encoding` distinguishes the audit-v2 stored encoding from the FEN-only
# one (93 planes of difference); `batch_size` matters because both the value
# and raw-policy regret rulers are batch-size dependent (0.66 cp between 128
# and 256 for value, ~0.8 cp between 64 and 256 for policy) and a paired delta
# of that size is one this tool is routinely asked to adjudicate.
# `search_shape` is the training rows' COMPLETE realized search shape
# (audit_targets `train_shape_stamp_fields()` — every GumbelConfig field bar
# the checkpoint-derived ones, plus the two runner arguments that are not
# GumbelConfig fields). It is complete rather than "the three that were wrong"
# because a three-field stamp only catches the ruler change that already
# happened: move `topk` or the sim budget after both dumps use the current
# code and each run passes its own live-config check while emitting an
# identical stamp. Rows (d)/(e) MOVED on 2026-08-16: until then the
# audit built its "production training target" without
# `gumbel_policy_temp`/`gumbel_target_max_visit_cap`/
# `gumbel_target_untempered_prior`, and with the last two at their defaults
# `mcts/gumbel.py` takes the `imp_store = imp_all` branch — so those rows were
# the PLAY distribution, not the stored target. A pre-fix dump joins cleanly
# against a post-fix one and reports a tight-CI delta that is entirely the
# ruler. That is what this entry stops.
#
# ⚑ `search_shape` IS KEYED BY TRAINING ROW (`{"train": {...}, "train_fast":
# {...}}`), because the two rows are separately realized. It banked only the
# full-sims row until 2026-08-16, so a change to `fast_simulations` alone left
# the stamp byte-identical while `cand.train_fast.*` came from a different
# search budget — the same failure this field exists to stop, on the row that
# was not being stamped.
#
# `target_config` is the second half of the ruler, and it closes a hole
# `config_authority` structurally cannot: that flag is a SAME-RUN verdict, so
# two audits made weeks apart under different `sf_policy_temp` each stamp
# themselves authoritative and their row-(c) numbers still are not comparable.
# Banking the audit's realized `AUDIT_DIRECT_CONFIG_KEYS` VALUES (never their
# names — the names are identical in both dumps by construction) is what makes
# that visible. `config_authority` itself is deliberately NOT a ruler field: it
# carries an absolute reference PATH and free-text reason, so joining on it
# would refuse legitimate comparisons for reasons that are not ruler changes.
RULER_FIELDS: tuple[str, ...] = (
    "input_encoding", "batch_size", "search_shape", "target_config",
)

# ⚑ ABSENCE IS INFORMATIVE FOR `input_encoding`, AND ONLY FOR IT.
# Every dump written before the audit-v2 flag existed is `fen_only` BY
# CONSTRUCTION, because `stored` did not exist to produce one. Treating an
# unstamped dump as "unknown" would let the single join this gate exists to
# stop — a legacy fen_only dump against a new stored one — take the warn path
# and exit 0, which is the join an operator is most likely to attempt and least
# likely to notice. So an unstamped dump DECLARES fen_only.
#
# `batch_size` gets no such default: old dumps genuinely varied (the ledger's
# standing VALUE yardstick pins --batch-size 128 while the CLI default is 256),
# so inferring one would be a guess rather than a deduction, and a wrong guess
# here refuses a legitimate comparison.
#
# ⚑ `search_shape` GETS NO INFERENCE, AND THAT IS A CORRECTION.
# A previous revision inferred `{policy_temp: 1.0, target_max_visit_cap: 0,
# target_untempered_prior: False}` for an unstamped dump and called all three a
# DEDUCTION. Two of the three are. `policy_temp` is not: pre-fix,
# `_net_candidates(policy_temp=...)` fed the operator-settable `--policy-temp`
# to EVERY profile including the training rows, so a legacy dump made with
# `--policy-temp 2.2` was inferred as 1.0. That both accepts a legacy-2.2 vs
# current-1.0 join (attributing the ruler change to the checkpoints, the exact
# failure this gate exists to stop) and refuses a legitimate legacy-2.2 vs
# current-2.2 one. Same shape as the `batch_size` argument above, which is why
# `batch_size` was correctly refused an inference.
#
# So an unstamped dump declares UNSTAMPED_LEGACY — a value of its own, not a
# guess at what it held. It compares EQUAL to another unstamped dump (legacy vs
# legacy still joins, as before) and UNEQUAL to any real stamp (legacy vs
# post-fix is still REFUSED, which is the join this entry exists to stop). The
# refusal is kept without the guess; nothing that used to work stops working.
UNSTAMPED_LEGACY = json.dumps("<unstamped: pre-2026-08-16 audit_targets build>")

INFERRED_WHEN_ABSENT: dict[str, str] = {
    "input_encoding": json.dumps("fen_only"),
    "search_shape": UNSTAMPED_LEGACY,
}


def _declared(dump: Dump, field: str) -> tuple[set[str], bool]:
    """(values the dump declares for `field`, whether they were INFERRED)."""
    values = dump.provenance.get(field, set())
    if values:
        return values, False
    inferred = INFERRED_WHEN_ABSENT.get(field)
    return ({inferred}, True) if inferred is not None else (set(), False)


def _match_stamp_shape(inferred: set[str], other: set[str]) -> set[str]:
    """Re-express an INFERRED scalar stamp in the counterpart's shape.

    The two producers stamp different shapes: ``value_regret.py`` writes one
    scalar encoding per record, while ``audit_targets.py`` writes one encoding
    PER CANDIDATE (a dict), because its ``--input-encoding`` moves only row (a)
    and the search rows are always ``fen_only``. `INFERRED_WHEN_ABSENT` can only
    name the scalar, and a scalar never equals a dict — so without this an
    unstamped ``audit_targets`` dump would be refused against a fresh
    DEFAULT-encoding one, i.e. the same ruler on both sides.

    That is not hypothetical: 103 banked unstamped ``audit_targets`` dumps exist
    under ``scratchpad/``, and several documented ledger readouts join exactly
    that pair. An unstamped dump predates ``--input-encoding`` entirely, so
    every one of its candidates was ``fen_only``; expanding to the counterpart's
    key set says precisely that. Candidates the counterpart records as ``null``
    (``sf_soft``, which has no net input) stay ``null``, so they need no special
    case and cannot manufacture a disagreement.

    A ``stored`` counterpart still differs on ``raw`` and is still refused.
    """
    if len(inferred) != 1 or len(other) != 1:
        return inferred
    inf = json.loads(next(iter(inferred)))
    oth = json.loads(next(iter(other)))
    if not isinstance(oth, dict) or isinstance(inf, dict):
        return inferred
    return {
        json.dumps(
            {k: (None if v is None else inf) for k, v in oth.items()},
            sort_keys=True,
        )
    }


def require_same_stamp(a: Dump, b: Dump, *, label_a: str, label_b: str) -> None:
    """Refuse to join two dumps whose provenance HEADERS disagree.

    ⚑ `require_same_ruler` compares what the ROWS carry. This compares what the
    two dumps DECLARE, and until PR #423's review nothing did: `load_dump`
    recognised the stamp by its sentinel and dropped it, so two dumps built by
    different rulers joined to exit 0 and printed a verdict under a banner that
    reads as a provenance certificate. Reviewer-confirmed by execution with two
    different `audit_ruler_version` values.

    **EXCLUDE, not include.** Every stamp key is ruler identity unless
    `STAMP_NON_IDENTITY_KEYS` says otherwise. An include list would have to be
    edited in lockstep with the writer and would fail silently when it was not —
    which is the defect class the stamp exists to prevent, one level up.

    ⚑ WHAT THAT DOES AND DOES NOT BUY — the earlier wording ("a version field
    added to the stamp later is guarded the day it appears") OVERCLAIMED, and
    the day it appears is exactly the day it is NOT guarded. A key present on
    one side and absent on the other takes the WARNING + `continue` path below
    and exits 0; only a key present on BOTH sides with different values is
    refused. So the exclude set buys automatic coverage from the day BOTH
    writers emit the field, not from the day the first one does.

    ⚑⚑ THE RATIONALE THAT USED TO SIT HERE NAMED A TOOL THAT DOES NOT GO
    THROUGH THIS FUNCTION. It said one-sided keys must be permitted because
    "`scripts/audit_compare_buckets.py` exists to join exactly that
    cross-producer pair". `audit_compare_buckets.py` never calls
    `require_same_stamp` — it validates through `read_audit_cache_stamp` +
    `require_same_audit_set`, which this could not affect. Outside `tests/`
    there is exactly ONE call site, `main` below. A justification for a hole
    that names a comparison which does not pass through the hole is the same
    defect as a stamp that is read and then ignored, one level up
    (#442 review B3). The hole is now half closed and the remaining half has a
    reason that survives checking:

      - a one-sided **CORE** key (`CORE_STAMP_KEYS`: the format, the policy-map
        version, the ruler version) is REFUSED. `audit_cache_stamp` writes all
        three into every stamped cache any writer has ever produced, so their
        absence on one side cannot be writer skew — it is a hand-made or
        mangled stamp, and the identity comparison over it is not sound.
      - a one-sided **extra** key still warns. This is the real case, and it is
        a real `paired_compare` invocation: `scripts/monitor_fen.sh` joins a
        BANKED baseline against a fresh dump every deep cycle, and a banked
        dump is by definition written by an older build than the fresh one. Add
        a stamp field today (`foreign_net_audit`'s input contract, say) and
        every banked baseline becomes one-sided on it. Refusing there would
        invalidate every baseline on the day a field is added, to catch a case
        the warning names — and per the #442 review B2 that warning is now
        distinguishable on the monitor line (`PARTIAL`), which it was not when
        this argument was first made.

    An ABSENT stamp is warned about, not refused: dumps predating stamping are
    legitimately unstamped, and `require_same_ruler` already refuses the
    encoding mismatch that actually invalidates a join. A PRESENT-but-different
    value is refused.
    """
    if not a.stamp or not b.stamp:
        missing = label_a if not a.stamp else label_b
        print(
            f"[paired-compare] WARNING: {missing} carries no provenance stamp "
            "— cannot verify both dumps came from the same ruler build. "
            "Re-dump with a current scorer to make this checkable."
        )
        return
    for key in sorted((set(a.stamp) | set(b.stamp)) - STAMP_NON_IDENTITY_KEYS):
        va = json.dumps(a.stamp.get(key), sort_keys=True)
        vb = json.dumps(b.stamp.get(key), sort_keys=True)
        if va == vb:
            continue
        if key not in a.stamp or key not in b.stamp:
            absent = label_a if key not in a.stamp else label_b
            if key in CORE_STAMP_KEYS:
                raise SystemExit(
                    f"{absent} declares no '{key}' in its provenance stamp, but "
                    f"the other side does. Every stamp `audit_cache_stamp` has "
                    "ever written carries it, so this is not two writer builds "
                    "disagreeing — it is a stamp that was not produced by the "
                    "writer, and its remaining identity fields cannot be "
                    "trusted to mean what they say. Re-dump."
                )
            print(
                f"[paired-compare] WARNING: stamp key '{key}' is declared by "
                f"only one side ({absent} lacks it) — the two dumps were "
                "written by different scorer builds."
            )
            continue
        raise SystemExit(
            f"{label_a} and {label_b} disagree on stamp key '{key}': "
            f"{va} vs {vb}. A paired delta across two rulers measures the "
            "ruler, not the checkpoints. Re-dump both sides with one scorer."
        )


def ruler_fields_for(metric: str | None) -> tuple[str, ...]:
    """Which ruler stamps actually govern a comparison of ``metric``.

    ⚑ `search_shape` describes the TRAINING rows' search and nothing else. The
    PR that introduced it says so in as many words — "rows (b) and (c) are
    unaffected" — and then checked the stamp globally, so comparing
    `cand.raw.exp` or `cand.sf_soft.exp` across a legacy and a current dump was
    refused over a training-row stamp that cannot touch either number. A gate
    that refuses comparisons it does not govern trains operators to route
    around it, which is the failure mode after "a gate that cannot fail".

    Unknown/None metric: every stamp applies. Defaulting the other way would
    make a typo'd `--field` silently skip the ruler check.

    ⚑ `target_config` is scoped BY PRODUCER, not by row, and the distinction is
    load-bearing. Within an `audit_targets` dump it governs every candidate:
    its keys span row (c) (`sf_policy_*`), rows (d)/(e) (`temperature`,
    `playout_cap_fraction`, the sim budgets) and value row (iii) (`sf_wdl_*`,
    `search_wdl_frac`), and inventing an approximate per-row partition would be
    a gate that looks scoped and is wrong at the edges. But `value_regret.py`
    does not build its rows from those keys and never stamps the field, so
    checking it on a `--field value` comparison can only ever produce the
    "not declared by either side" warning — which is #442 review B2 verbatim: a
    warning about something the run is not verifying made `prov:ok` UNREACHABLE
    on `monitor_fen.sh`'s line for months, and it is exactly what a fresh
    unscoped ruler field would have done again.
    `tests/test_paired_compare_gate_is_wired.py` is what caught it.
    """
    if metric is None:
        return RULER_FIELDS
    text = str(metric)
    governs_training_rows = text.startswith(("cand.train", "train"))
    is_audit_targets_metric = text.startswith("cand.")
    skip = set()
    if not governs_training_rows:
        skip.add("search_shape")
    if not is_audit_targets_metric:
        skip.add("target_config")
    return tuple(f for f in RULER_FIELDS if f not in skip)


def require_same_ruler(
    a: Dump, b: Dump, *, label_a: str, label_b: str, metric: str | None = None,
) -> None:
    """Refuse to join two dumps made with different rulers.

    Carrying the ruler on every record is only half the rule — a stamp nothing
    reads is a value accepted and then ignored. This is the half that can fail.

    For `input_encoding`, a dump with no stamp is read as `fen_only` (see
    INFERRED_WHEN_ABSENT), so legacy-vs-legacy still compares and
    legacy-vs-`stored` is REFUSED. For `search_shape`, an unstamped dump
    declares `UNSTAMPED_LEGACY`, so legacy-vs-legacy compares and
    legacy-vs-post-fix is REFUSED without guessing what the legacy shape was.
    For `batch_size`, an unstamped dump is genuinely unknown and is warned
    about rather than refused.

    ``metric`` is the value being compared; stamps that cannot govern it are
    skipped (``ruler_fields_for``). Defaulting to ``None`` — check everything —
    keeps every existing caller strict.
    """
    for field in ruler_fields_for(metric):
        for name, dump in ((label_a, a), (label_b, b)):
            if len(dump.provenance.get(field, set())) > 1:
                raise SystemExit(
                    f"{name}: rows disagree on '{field}' "
                    f"({sorted(dump.provenance[field])}) — this dump mixes two "
                    "rulers within itself and cannot be compared to anything."
                )
        va, a_inferred = _declared(a, field)
        vb, b_inferred = _declared(b, field)
        # An inferred stamp has to be compared in the SHAPE the other side
        # actually writes, or the two producers can never agree. ⚑ Scoped to
        # `input_encoding`, which is what it was written for and the only field
        # with two producers stamping different shapes. Left unscoped it would
        # also fire on the `search_shape` UNSTAMPED_LEGACY sentinel, expanding a
        # deliberate "we do not know" into a per-key dict of that sentinel — the
        # refusal survives either way, but the message would describe a stamp
        # nothing ever wrote.
        if field == "input_encoding":
            if a_inferred:
                va = _match_stamp_shape(va, vb)
            if b_inferred:
                vb = _match_stamp_shape(vb, va)
        if not va or not vb:
            print(
                f"[paired-compare] WARNING: '{field}' not declared by "
                f"{label_a if not va else label_b} — cannot verify both sides "
                "used the same ruler. Re-dump with a current scorer to make "
                "this checkable."
            )
            continue
        if va != vb:
            how = {
                label_a: " (INFERRED from an unstamped dump)" if a_inferred else "",
                label_b: " (INFERRED from an unstamped dump)" if b_inferred else "",
            }
            raise SystemExit(
                f"REFUSING TO JOIN: {label_a} has {field}={sorted(va)[0]}"
                f"{how[label_a]} and {label_b} has {field}={sorted(vb)[0]}"
                f"{how[label_b]}. These are DIFFERENT RULERS of the same "
                "positions; a ruler change invalidates its records, so the "
                "paired delta between them measures the ruler, not the "
                "checkpoints. Re-run one side to match the other."
            )
        source = (
            " (inferred: neither dump is stamped)" if a_inferred and b_inferred
            else " (one side inferred from an unstamped dump)"
            if a_inferred or b_inferred else " (both sides)"
        )
        print(f"[paired-compare] ruler {field}={sorted(va)[0]}{source}")


def report(a: Dump, b: Dump, *, label_a: str, label_b: str, n_boot: int) -> None:
    common = sorted(set(a.rows) & set(b.rows))
    if not common:
  # Report `unusable` here too, in the same per-side shape as the success
  # path. Total scorer failure on one side is the EXTREME of the defect the
  # rest of this function fixes: "A has 50, B has 0" alone is indistinguishable
  # from an empty file or a wrong --field, so the operator goes hunting for a
  # schema bug when in fact every position scored and every score was
  # null/NaN. The count is already sitting in `Dump.unusable`; withholding it
  # from the one message the operator sees is how a data failure gets read as
  # a config typo.
  # Name the side that indexed nothing rather than saying "a side": the
  # accounting printed immediately before it already labels A and B, so an
  # unlabelled "A side indexed nothing" reads as the literal side A and
  # contradicts the numbers on the same line.
        empty = [name for name, d in (("A", a), ("B", b)) if not d.rows]
        raise SystemExit(
            "no joinable rows — "
            f"A: {len(a.rows) + a.unusable} rows, {a.unusable} unusable, "
            f"{len(a.rows)} indexed"
            f"   B: {len(b.rows) + b.unusable} rows, {b.unusable} unusable, "
            f"{len(b.rows)} indexed. "
            + (
                "Both sides indexed rows but share no key — check "
                "--join-key/--field against the dump schema."
                if not empty
                else f"{' and '.join(empty)} indexed nothing: with rows read but "
                "unusable the scorer failed on them (non-finite or null "
                "--field); with no rows at all the dump is empty. Otherwise "
                "check --join-key/--field against the dump schema."
            ),
        )
    va = np.array([a.rows[k][0] for k in common])
    vb = np.array([b.rows[k][0] for k in common])
    ph = np.array([a.rows[k][1] for k in common])
    d = va - vb

    lo, hi = paired_bootstrap_ci(d, n_boot=n_boot)
    frac_a = float((d < 0).mean())
    frac_b = float((d > 0).mean())
  # Per side, and never as one summed `dropped`. The old single number was
  # computed from the two INDEXES, so a row the scorer failed on -- absent from
  # both -- was invisible: three null rows per side printed "dropped 0". Summing
  # the two categories instead would misreport the opposite way, since a row
  # unusable on B is also unmatched from A and would be counted twice for one
  # lost position. Per side, every figure is checkable against the input files
  # and `rows = unusable + indexed`, `indexed = paired + unmatched`.
    print(f"paired positions: {len(common)}")
    print(f"  A: {len(a.rows) + a.unusable} rows, {a.unusable} unusable, "
          f"{len(a.rows) - len(common)} unmatched"
          f"   B: {len(b.rows) + b.unusable} rows, {b.unusable} unusable, "
          f"{len(b.rows) - len(common)} unmatched")
    print(f"A = {label_a}: mean {va.mean():.2f}")
    print(f"B = {label_b}: mean {vb.mean():.2f}")
    print(f"paired delta (A-B): {d.mean():+.2f}  [95% CI {lo:+.2f} .. {hi:+.2f}]")
    verdict = "A better" if hi < 0 else ("B better" if lo > 0 else "NOT significant")
    print(f"verdict at 95%: {verdict}   "
          f"(A better {frac_a:.1%} / B better {frac_b:.1%} / tied {1 - frac_a - frac_b:.1%})")
    for name in sorted(set(ph)):
        m = ph == name
        if m.sum() < 30:
            continue
        plo, phi = paired_bootstrap_ci(d[m], n_boot=n_boot)
        print(f"  {name:11s} n={int(m.sum()):5d} delta {d[m].mean():+.2f} "
              f"[{plo:+.2f} .. {phi:+.2f}]")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("dump_a", help="per-position JSONL for checkpoint/candidate A")
    ap.add_argument("dump_b", help="per-position JSONL for checkpoint/candidate B")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--join-key", default="fen",
                    help="record field to join on (audit_targets dumps: 'key')")
    ap.add_argument("--field", default="value",
                    help="dotted path to the compared metric "
                         "(audit_targets dumps: e.g. 'cand.search.exp')")
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args()
    dump_a = load_dump(args.dump_a, join_key=args.join_key, field=args.field)
    dump_b = load_dump(args.dump_b, join_key=args.join_key, field=args.field)
    label_a = args.label_a or args.dump_a
    label_b = args.label_b or args.dump_b
    require_same_ruler(
        dump_a, dump_b, label_a=label_a, label_b=label_b, metric=args.field,
    )
    require_same_stamp(dump_a, dump_b, label_a=label_a, label_b=label_b)
    report(dump_a, dump_b, label_a=label_a, label_b=label_b, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
