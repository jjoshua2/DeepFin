from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.paired_compare import paired_bootstrap_ci


def test_bootstrap_ci_covers_known_shift() -> None:
    rng = np.random.default_rng(7)
    deltas = rng.normal(loc=-2.0, scale=5.0, size=2000)
    lo, hi = paired_bootstrap_ci(deltas, n_boot=4000, seed=1)
    assert lo < -2.0 < hi or (lo < deltas.mean() < hi)
    # width ~ 2*1.96*5/sqrt(2000) ≈ 0.44
    assert 0.2 < (hi - lo) < 1.0
    assert hi < 0  # a real 2cp shift at n=2000 is clearly significant


def test_bootstrap_ci_null_is_not_significant() -> None:
    rng = np.random.default_rng(3)
    deltas = rng.normal(loc=0.0, scale=5.0, size=2000)
    lo, hi = paired_bootstrap_ci(deltas, n_boot=4000, seed=2)
    assert lo < 0 < hi


def test_load_dump_audit_shape(tmp_path) -> None:
    from scripts.paired_compare import load_dump

    p = tmp_path / "audit.jsonl"
    rows = [
        {"key": "k1", "phase": "middlegame",
         "cand": {"search": {"exp": 12.5, "top1": 30.0}, "net": {"exp": 20.0}}},
        {"key": "k2", "phase": 1,
         "cand": {"search": {"exp": 0.0, "top1": 0.0}}},
        {"key": "k3", "cand": {"search": {"exp": None}}},  # null metric -> dropped
        {"key": "k4", "cand": {}},                          # missing path -> dropped
    ]
    with p.open("w") as f:
        for r in rows:
            f.write(__import__("json").dumps(r) + "\n")

    d = load_dump(str(p), join_key="key", field="cand.search.exp")
    assert set(d.rows) == {"k1", "k2"}
    assert d.rows["k1"] == (12.5, "middlegame")
    assert d.rows["k2"] == (0.0, "middlegame")  # int phase index mapped to name
    assert d.unusable == 2  # k3 null, k4 missing path

    top1 = load_dump(str(p), join_key="key", field="cand.search.top1")
    assert top1.rows["k1"] == (30.0, "middlegame")


def _write_jsonl(path, rows: list[dict]) -> str:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


# ---------------------------------------------------------------------------
# Two ways this tool used to print a confident answer that was not the answer.
# Both matter more than an ordinary bug because every kill/hold decision in
# docs/experiment_ledger.md is read off this output: a comparison that launders
# a KILL into a HOLD is the worst defect class here.
# ---------------------------------------------------------------------------

def test_report_counts_null_rows_that_never_entered_the_join(tmp_path, capsys) -> None:
    """The reported drop count used to exclude nulls entirely.

    ``dropped`` was computed from the two INDEXES, so a row the scorer failed
    on -- absent from both -- was invisible. Demonstrated on main: eight rows
    per side, three of them null on both sides, printed "dropped 0". A dump the
    scorer had largely failed on read as a complete join.
    """
    from scripts.paired_compare import load_dump, report

    rows_a: list[dict[str, object]] = [{"fen": f"p{i}", "value": 10.0} for i in range(5)]
    rows_b: list[dict[str, object]] = [{"fen": f"p{i}", "value": 12.0} for i in range(5)]
    for i in range(3):  # measured on neither side -> in neither index
        rows_a.append({"fen": f"n{i}", "value": None})
        rows_b.append({"fen": f"n{i}", "value": None})

    a = load_dump(_write_jsonl(tmp_path / "a.jsonl", rows_a))
    b = load_dump(_write_jsonl(tmp_path / "b.jsonl", rows_b))
    assert (a.unusable, b.unusable) == (3, 3)

    report(a, b, label_a="A", label_b="B", n_boot=200)

    out = capsys.readouterr().out
    assert "paired positions: 5\n" in out
    assert "A: 8 rows, 3 unusable, 0 unmatched   B: 8 rows, 3 unusable, 0 unmatched" in out


def test_report_counts_unmatched_and_unusable_separately(tmp_path, capsys) -> None:
    """The two losses are different problems and must not be summed away."""
    from scripts.paired_compare import load_dump, report

    a = load_dump(_write_jsonl(tmp_path / "a.jsonl", [
        {"fen": "p1", "value": 10.0}, {"fen": "p2", "value": 20.0},
        {"fen": "only_a", "value": 5.0}, {"fen": "bad_a", "value": None},
    ]))
    b = load_dump(_write_jsonl(tmp_path / "b.jsonl", [
        {"fen": "p1", "value": 13.0}, {"fen": "p2", "value": 23.0},
    ]))

    report(a, b, label_a="A", label_b="B", n_boot=200)

    out = capsys.readouterr().out
    assert "paired positions: 2\n" in out
    assert "A: 4 rows, 1 unusable, 1 unmatched   B: 2 rows, 0 unusable, 0 unmatched" in out
    assert "paired delta (A-B): -3.00" in out


def test_one_nan_row_cannot_launder_a_kill_into_a_hold(tmp_path, capsys) -> None:
    """``isinstance(v, (int, float))`` admits NaN, and one NaN poisons everything.

    numpy's ``mean`` and ``percentile`` propagate NaN, so the delta and both CI
    bounds print as ``nan``; ``nan < 0`` and ``nan > 0`` are both False, so the
    verdict falls through to "NOT significant". Measured on main with exactly
    this input: a clean -5.0 delta over 50 rows became "verdict at 95%: NOT
    significant" -- a KILL silently reported as a HOLD.
    """
    from scripts.paired_compare import load_dump, report

    rows_a = [{"fen": f"q{i}", "value": 100.0} for i in range(50)]
    rows_b = [{"fen": f"q{i}", "value": 105.0} for i in range(50)]
    rows_b[7]["value"] = float("nan")
    assert json.loads(json.dumps(rows_b[7]))["value"] != rows_b[7]["value"], \
        "NaN must survive the JSON round-trip or this test proves nothing"

    a = load_dump(_write_jsonl(tmp_path / "a.jsonl", rows_a))
    b = load_dump(_write_jsonl(tmp_path / "b.jsonl", rows_b))
    assert b.unusable == 1

    report(a, b, label_a="A", label_b="B", n_boot=500)

    out = capsys.readouterr().out
    assert "nan" not in out.lower()
    assert "paired positions: 49\n" in out
  # The one lost position is visible on BOTH sides and counted once on each:
  # B could not measure it, so A has it unpartnered. Summing the two into a
  # single "dropped 2" would claim two positions were lost.
    assert "A: 50 rows, 0 unusable, 1 unmatched   B: 50 rows, 1 unusable, 0 unmatched" in out
    assert "paired delta (A-B): -5.00" in out
    assert "verdict at 95%: A better" in out


def test_infinite_metric_is_dropped_like_a_null(tmp_path) -> None:
    """``inf`` is numeric and finite-looking to ``isinstance`` too."""
    from scripts.paired_compare import load_dump

    d = load_dump(_write_jsonl(tmp_path / "inf.jsonl", [
        {"fen": "a", "value": 1.0},
        {"fen": "b", "value": float("inf")},
        {"fen": "c", "value": float("-inf")},
    ]))

    assert set(d.rows) == {"a"}
    assert d.unusable == 2


def test_total_scorer_failure_is_not_reported_as_a_schema_mismatch(tmp_path) -> None:
    """The empty-join message used to hide ``unusable`` entirely.

    Total scorer failure on one side is the extreme of the same defect the rest
    of this file pins: 50 rows on A, all 50 non-finite on B printed only "no
    joinable rows (A has 50, B has 0) -- check --join-key/--field against the
    dump schema". That is indistinguishable from an empty file or a wrong
    ``--field``, so it sends the operator after a config typo when in fact
    every position was scored and every score came back NaN.
    """
    from scripts.paired_compare import load_dump, report

    a = load_dump(_write_jsonl(tmp_path / "a.jsonl", [
        {"fen": f"p{i}", "value": float(i)} for i in range(50)
    ]))
    b = load_dump(_write_jsonl(tmp_path / "b.jsonl", [
        {"fen": f"p{i}", "value": float("nan")} for i in range(50)
    ]))
    assert (len(a.rows), a.unusable) == (50, 0)
    assert (len(b.rows), b.unusable) == (0, 50)

    with pytest.raises(SystemExit) as exc:
        report(a, b, label_a="A", label_b="B", n_boot=200)

    message = str(exc.value)
    assert "no joinable rows" in message
    # The number the operator needs, in the same per-side shape as the success
    # path. Without it the message reads as a schema problem.
    assert "A: 50 rows, 0 unusable, 50 indexed" in message
    assert "B: 50 rows, 50 unusable, 0 indexed" in message
    assert "scorer failed" in message
    # Name the side that indexed nothing. It is B here, and A indexed all 50 --
    # a message reading "A ... indexed nothing" would contradict the accounting
    # printed immediately before it on the same line.
    assert "B indexed nothing" in message
    assert "A indexed nothing" not in message


def test_empty_join_over_two_healthy_dumps_still_blames_the_join_key(tmp_path) -> None:
    """Both sides indexed fine and simply share no key -- that IS a schema/key
    problem, and the message must keep saying so rather than blaming the scorer.
    """
    from scripts.paired_compare import load_dump, report

    a = load_dump(_write_jsonl(tmp_path / "a.jsonl", [
        {"fen": "a1", "value": 1.0}, {"fen": "a2", "value": 2.0},
    ]))
    b = load_dump(_write_jsonl(tmp_path / "b.jsonl", [
        {"fen": "b1", "value": 1.0}, {"fen": "b2", "value": 2.0},
    ]))

    with pytest.raises(SystemExit) as exc:
        report(a, b, label_a="A", label_b="B", n_boot=200)

    message = str(exc.value)
    assert "A: 2 rows, 0 unusable, 2 indexed" in message
    assert "B: 2 rows, 0 unusable, 2 indexed" in message
    assert "--join-key/--field" in message
    assert "scorer failed" not in message
    assert "indexed nothing" not in message


def test_both_sides_all_unusable_names_both_sides(tmp_path) -> None:
    """When neither side indexed anything, say so for both rather than one."""
    from scripts.paired_compare import load_dump, report

    a = load_dump(_write_jsonl(tmp_path / "a.jsonl", [
        {"fen": "p1", "value": None}, {"fen": "p2", "value": None},
    ]))
    b = load_dump(_write_jsonl(tmp_path / "b.jsonl", [
        {"fen": "p1", "value": float("nan")}, {"fen": "p2", "value": float("nan")},
    ]))

    with pytest.raises(SystemExit) as exc:
        report(a, b, label_a="A", label_b="B", n_boot=200)

    message = str(exc.value)
    assert "A: 2 rows, 2 unusable, 0 indexed" in message
    assert "B: 2 rows, 2 unusable, 0 indexed" in message
    assert "A and B indexed nothing" in message


def test_load_dump_refuses_duplicate_join_keys(tmp_path) -> None:
    """Audit L14: duplicates used to last-win and stay out of ``dropped``.

    The caller then read a clean-looking join over a silently smaller and
    silently biased sample. Refusing is the only honest answer -- there is no
    principled winner between two rows claiming the same position.
    """
    from scripts.paired_compare import load_dump

    p = tmp_path / "dupes.jsonl"
    _write_jsonl(p, [
        {"fen": "a", "value": 1.0},
        {"fen": "b", "value": 2.0},
        {"fen": "a", "value": 99.0},
    ])

    with pytest.raises(SystemExit) as exc:
        load_dump(str(p))

    message = str(exc.value)
    assert "duplicate" in message
    assert "'fen'" in message
    assert "'a'" in message


def test_duplicate_detection_ignores_rows_that_never_entered_the_index(tmp_path) -> None:
    """A repeated key on unusable rows leaves the join unambiguous.

    Null/non-finite metrics are skipped before indexing, so two unusable rows
    sharing a key must not trip the refusal -- otherwise a dump with a couple of
    failed positions becomes unreadable.
    """
    from scripts.paired_compare import load_dump

    d = load_dump(_write_jsonl(tmp_path / "nulls.jsonl", [
        {"fen": "a", "value": 1.0},
        {"fen": "b", "value": None},
        {"fen": "b"},
    ]))

    assert set(d.rows) == {"a"}
    assert d.unusable == 2


# ---------------------------------------------------------------------------
# Ruler provenance: the two dumps must have been made by the SAME ruler
# ---------------------------------------------------------------------------
#
# `--input-encoding fen_only` and `stored` are different rulers of the same
# positions (93 planes of difference), and both regret rulers are batch-size
# dependent. Stamping every dumped record with its ruler is only half the rule;
# a stamp nothing reads is a value accepted and then ignored. These pin the
# half that can fail.


def _stamped(fen: str, value: float, **stamps) -> dict:
    return {"fen": fen, "value": value, "phase": 1, **stamps}


def test_joining_two_encodings_is_refused(tmp_path) -> None:
    from scripts.paired_compare import load_dump, require_same_ruler

    a = load_dump(_write_jsonl(tmp_path / "a.jsonl", [
        _stamped("p", 10.0, input_encoding="fen_only", batch_size=256),
    ]))
    b = load_dump(_write_jsonl(tmp_path / "b.jsonl", [
        _stamped("p", 20.0, input_encoding="stored", batch_size=256),
    ]))

    with pytest.raises(SystemExit) as exc:
        require_same_ruler(a, b, label_a="A", label_b="B")
    assert "REFUSING TO JOIN" in str(exc.value)
    assert "input_encoding" in str(exc.value)


def test_joining_two_batch_sizes_is_refused(tmp_path) -> None:
    """0.66 cp of ruler drift is larger than deltas this tool adjudicates."""
    from scripts.paired_compare import load_dump, require_same_ruler

    a = load_dump(_write_jsonl(tmp_path / "a.jsonl", [
        _stamped("p", 10.0, input_encoding="fen_only", batch_size=128),
    ]))
    b = load_dump(_write_jsonl(tmp_path / "b.jsonl", [
        _stamped("p", 20.0, input_encoding="fen_only", batch_size=256),
    ]))

    with pytest.raises(SystemExit, match="batch_size"):
        require_same_ruler(a, b, label_a="A", label_b="B")


def test_matching_rulers_pass_and_are_printed(tmp_path, capsys) -> None:
    from scripts.paired_compare import load_dump, require_same_ruler

    rows = [_stamped("p", 1.0, input_encoding="stored", batch_size=256)]
    a = load_dump(_write_jsonl(tmp_path / "a.jsonl", rows))
    b = load_dump(_write_jsonl(tmp_path / "b.jsonl", rows))

    require_same_ruler(a, b, label_a="A", label_b="B")
    out = capsys.readouterr().out
    assert "input_encoding=\"stored\"" in out
    assert "batch_size=256" in out


def test_a_dump_that_mixes_rulers_within_itself_is_refused(tmp_path) -> None:
    from scripts.paired_compare import load_dump, require_same_ruler

    a = load_dump(_write_jsonl(tmp_path / "a.jsonl", [
        _stamped("p", 1.0, input_encoding="fen_only", batch_size=256),
        _stamped("q", 2.0, input_encoding="stored", batch_size=256),
    ]))
    b = load_dump(_write_jsonl(tmp_path / "b.jsonl", [
        _stamped("p", 1.0, input_encoding="fen_only", batch_size=256),
    ]))

    with pytest.raises(SystemExit, match="mixes two rulers within itself"):
        require_same_ruler(a, b, label_a="A", label_b="B")


def test_two_unstamped_dumps_still_compare(tmp_path, capsys) -> None:
    """Dumps predating provenance must still compare — both are fen_only."""
    from scripts.paired_compare import load_dump, require_same_ruler

    a = load_dump(_write_jsonl(tmp_path / "a.jsonl", [{"fen": "p", "value": 1.0}]))
    b = load_dump(_write_jsonl(tmp_path / "b.jsonl", [{"fen": "p", "value": 2.0}]))

    require_same_ruler(a, b, label_a="A", label_b="B")
    out = capsys.readouterr().out
    assert "inferred" in out
    # batch_size is genuinely unknown on an old dump and stays warn-only.
    assert "WARNING" in out
    assert "batch_size" in out


def test_legacy_unstamped_vs_stored_is_refused(tmp_path) -> None:
    """THE join this gate exists to stop.

    Every dump written before audit-v2 is fen_only by construction, so an
    unstamped dump is not "unknown" — it is a declaration. Treating it as
    unknown let old-fen_only vs new-stored proceed with exit 0, which is the
    comparison an operator is most likely to attempt and least likely to
    notice.
    """
    from scripts.paired_compare import load_dump, require_same_ruler

    legacy = load_dump(_write_jsonl(tmp_path / "a.jsonl", [
        {"fen": "p", "value": 1.0},
    ]))
    stored = load_dump(_write_jsonl(tmp_path / "b.jsonl", [
        _stamped("p", 2.0, input_encoding="stored", batch_size=256),
    ]))

    with pytest.raises(SystemExit) as exc:
        require_same_ruler(legacy, stored, label_a="LEGACY", label_b="NEW")
    assert "REFUSING TO JOIN" in str(exc.value)
    assert "INFERRED" in str(exc.value)


def test_legacy_unstamped_vs_new_fen_only_still_compares(tmp_path, capsys) -> None:
    """The inference must not break the comparison it is allowed to make."""
    from scripts.paired_compare import load_dump, require_same_ruler

    legacy = load_dump(_write_jsonl(tmp_path / "a.jsonl", [
        {"fen": "p", "value": 1.0},
    ]))
    new = load_dump(_write_jsonl(tmp_path / "b.jsonl", [
        _stamped("p", 2.0, input_encoding="fen_only", batch_size=256),
    ]))

    require_same_ruler(legacy, new, label_a="LEGACY", label_b="NEW")
    assert "input_encoding" in capsys.readouterr().out


def test_batch_size_is_never_inferred() -> None:
    """Old dumps really did vary on batch size, so a guess would be wrong.

    The ledger's standing VALUE yardstick pins --batch-size 128 while the CLI
    default is 256; inferring either would refuse a legitimate comparison.
    """
    from scripts.paired_compare import INFERRED_WHEN_ABSENT

    assert "batch_size" not in INFERRED_WHEN_ABSENT
    assert INFERRED_WHEN_ABSENT["input_encoding"] == json.dumps("fen_only")


def test_every_ruler_field_is_written_by_some_scorer() -> None:
    """The stamp the gate reads is one a scorer actually writes.

    A gate reading a field NO scorer emits would pass everything forever, so
    the field NAMES are pinned against the producers rather than assumed.

    ⚑ At least one producer, not every producer. This asserted "every field
    appears in value_regret.py" until `search_shape` was added, and that is a
    stronger claim than the anti-vacuity property it was after — `value_regret`
    runs NO search, so it has no search shape to declare. Stamping one there
    anyway would have been worse than useless: an old value_regret dump
    (`search_shape` absent, hence INFERRED to the pre-fix shape) against a fresh
    one (declaring `null`) would be REFUSED, breaking the standing VALUE
    yardstick's own paired comparison. Cross-producer joins are not a real
    workflow to begin with — `value_regret` dumps carry `value` and
    `audit_targets` dumps carry `cand`, so no `--field` names both.
    """
    import scripts.audit_targets as at
    import scripts.paired_compare as pc
    import scripts.value_regret as vr

    producers = {
        "value_regret.py": Path(vr.__file__).read_text(encoding="utf-8"),
        "audit_targets.py": Path(at.__file__).read_text(encoding="utf-8"),
    }
    for field in pc.RULER_FIELDS:
        writers = [n for n, src in producers.items() if f'"{field}": ' in src]
        assert writers, f"{field} is read by the gate and written by nobody"
    # And the split is the one documented above, pinned so a field silently
    # migrating between producers is visible.
    vr_src = producers["value_regret.py"]
    assert '"input_encoding": ' in vr_src
    assert '"batch_size": ' in vr_src
    assert '"search_shape": ' not in vr_src
    assert '"search_shape": ' in producers["audit_targets.py"]


# ---------------------------------------------------------------------------
# The gate has TWO producers, and they stamp DIFFERENT SHAPES
# ---------------------------------------------------------------------------
#
# `value_regret.py` writes one scalar encoding per record. `audit_targets.py`
# writes one encoding PER CANDIDATE (a dict), because its --input-encoding moves
# only row (a) while the search rows are always fen_only. The first version of
# the unstamped-dump inference only ever compared the scalar shape, so it
# FALSE-REFUSED an unstamped audit_targets dump against a fresh default-encoding
# one -- same ruler on both sides, no override flag -- which is the join behind
# several banked ledger readouts.
#
# The gap was a PRODUCER the gate's tests never fed it, which is the same shape
# as the defect the stamping exists to prevent. So these build the stamp from
# audit_targets' OWN candidate list rather than hardcoding it: add or rename a
# candidate and the tests follow the producer.


def _audit_targets_stamp(encoding: str) -> dict[str, str | None]:
    """The `input_encoding` dict `audit_targets.py --dump-per-position` writes.

    Mirrors the producer at `audit_targets.py` (row (a) takes the flag, the
    searches encode internally and are always fen_only, `sf_soft` has no net
    input at all), keyed off `_CANDIDATE_NAMES` so it cannot drift from it.
    """
    import scripts.audit_targets as at

    return {
        c: (encoding if c == "raw" else None if c == "sf_soft" else "fen_only")
        for c in at._CANDIDATE_NAMES
    }


def _audit_rows(encoding: str | None) -> list[dict]:
    import scripts.audit_targets as at

    rows: list[dict] = []
    for i in range(5):
        row: dict = {
            "key": f"k{i}", "phase": 1, "source": 0,
            "cand": {c: {"exp": float(i)} for c in at._CANDIDATE_NAMES},
        }
        if encoding is not None:
            row["input_encoding"] = _audit_targets_stamp(encoding)
            row["batch_size"] = 256
        rows.append(row)
    return rows


def test_audit_targets_stamp_is_a_dict_not_a_scalar() -> None:
    """Pin the shape difference the gate has to cope with.

    If this ever becomes a scalar the inference below is dead weight; if the
    scalar producer becomes a dict, `INFERRED_WHEN_ABSENT` needs revisiting.
    """
    import scripts.value_regret as vr

    stamp = _audit_targets_stamp("fen_only")
    assert isinstance(stamp, dict)
    assert stamp["raw"] == "fen_only"
    assert stamp["sf_soft"] is None
    # ...and the other producer really does write a bare scalar.
    src = Path(vr.__file__).read_text(encoding="utf-8")
    assert '"input_encoding": encoding,' in src


def test_unstamped_audit_targets_dump_compares_to_a_default_run(tmp_path, capsys) -> None:
    """THE regression: same ruler, dict vs inferred scalar, must NOT refuse.

    103 banked unstamped audit_targets dumps exist under scratchpad/ and several
    documented ledger readouts join exactly this pair. Refusing it would make
    those readouts unreproducible without re-running the old checkpoint on GPU.
    """
    from scripts.paired_compare import load_dump, require_same_ruler

    legacy = load_dump(
        _write_jsonl(tmp_path / "a.jsonl", _audit_rows(None)),
        join_key="key", field="cand.search.exp",
    )
    fresh = load_dump(
        _write_jsonl(tmp_path / "b.jsonl", _audit_rows("fen_only")),
        join_key="key", field="cand.search.exp",
    )

    require_same_ruler(legacy, fresh, label_a="LEGACY", label_b="NEW")
    out = capsys.readouterr().out
    assert "REFUSING" not in out
    assert "inferred" in out


def test_unstamped_audit_targets_dump_is_still_refused_against_stored(
    tmp_path,
) -> None:
    """The inference must not buy the false-accept back.

    Expanding the inferred scalar to the counterpart's shape has to keep row
    (a) discriminating -- that is the only row --input-encoding moves.
    """
    from scripts.paired_compare import load_dump, require_same_ruler

    legacy = load_dump(
        _write_jsonl(tmp_path / "a.jsonl", _audit_rows(None)),
        join_key="key", field="cand.raw.exp",
    )
    stored = load_dump(
        _write_jsonl(tmp_path / "b.jsonl", _audit_rows("stored")),
        join_key="key", field="cand.raw.exp",
    )

    with pytest.raises(SystemExit) as exc:
        require_same_ruler(legacy, stored, label_a="LEGACY", label_b="NEW")
    assert "REFUSING TO JOIN" in str(exc.value)
    assert "stored" in str(exc.value)


def test_match_stamp_shape_leaves_a_scalar_counterpart_alone() -> None:
    """value_regret-vs-value_regret must keep comparing scalars."""
    from scripts.paired_compare import _match_stamp_shape

    scalar = {json.dumps("fen_only")}
    assert _match_stamp_shape(scalar, scalar) == scalar
    # Ambiguous input (a dump that mixes shapes) is left untouched rather than
    # guessed at; require_same_ruler has already refused that case upstream.
    assert _match_stamp_shape(scalar, {json.dumps("a"), json.dumps("b")}) == scalar


def test_match_stamp_shape_preserves_null_candidates() -> None:
    """`sf_soft` has no net input on either side and must stay null.

    If the expansion filled it with "fen_only" the two sides would disagree on
    a candidate that has no encoding at all.
    """
    from scripts.paired_compare import _match_stamp_shape

    other = {json.dumps(_audit_targets_stamp("fen_only"), sort_keys=True)}
    got = json.loads(next(iter(_match_stamp_shape({json.dumps("fen_only")}, other))))
    assert got["sf_soft"] is None
    assert got["raw"] == "fen_only"
    assert got == _audit_targets_stamp("fen_only")


# ---------------------------------------------------------------------------
# `search_shape`: rows (d)/(e) MOVED on 2026-08-16
# ---------------------------------------------------------------------------
#
# Until then `audit_targets.py` built its "production training target" without
# `gumbel_policy_temp` / `gumbel_target_max_visit_cap` /
# `gumbel_target_untempered_prior`, and with the last two at their defaults
# `mcts/gumbel.py` takes the `imp_store = imp_all` branch — so those rows were
# the PLAY distribution. A pre-fix dump and a post-fix one join cleanly on key
# and report a tight-CI delta that is entirely the ruler change. The eval
# protocol note that records this cannot stop a tool; these can.

_POST_FIX_SHAPE = {
    "policy_temp": 1.5, "target_max_visit_cap": 5, "target_untempered_prior": True,
}
_PRE_FIX_SHAPE = {
    "policy_temp": 1.0, "target_max_visit_cap": 0, "target_untempered_prior": False,
}


def test_pre_fix_dump_against_post_fix_dump_is_refused(tmp_path) -> None:
    """The join this entry exists to stop: unstamped vs stamped."""
    from scripts.paired_compare import load_dump, require_same_ruler

    legacy = load_dump(_write_jsonl(tmp_path / "legacy.jsonl", [
        _stamped("p", 10.0, input_encoding="fen_only", batch_size=256),
    ]))
    fresh = load_dump(_write_jsonl(tmp_path / "fresh.jsonl", [
        _stamped("p", 20.0, input_encoding="fen_only", batch_size=256,
                 search_shape=_POST_FIX_SHAPE),
    ]))

    with pytest.raises(SystemExit, match="search_shape"):
        require_same_ruler(legacy, fresh, label_a="LEGACY", label_b="NEW")


def test_two_pre_fix_dumps_still_compare(tmp_path) -> None:
    """Legacy-vs-legacy is the same ruler and must NOT be refused.

    103 banked unstamped dumps exist under `scratchpad/`; an inference that
    refused them against each other would break every historical readout to
    stop a join that is not happening.
    """
    from scripts.paired_compare import load_dump, require_same_ruler

    a = load_dump(_write_jsonl(tmp_path / "a.jsonl", [
        _stamped("p", 10.0, input_encoding="fen_only", batch_size=256),
    ]))
    b = load_dump(_write_jsonl(tmp_path / "b.jsonl", [
        _stamped("p", 11.0, input_encoding="fen_only", batch_size=256),
    ]))

    require_same_ruler(a, b, label_a="A", label_b="B")


def test_an_absent_stamp_is_UNKNOWN_not_a_guessed_shape(tmp_path) -> None:
    """MUTANT (F5 / Codex P1): re-infer a concrete legacy shape from absence.

    An earlier revision inferred ``{policy_temp: 1.0, target_max_visit_cap: 0,
    target_untempered_prior: False}`` and called all three a DEDUCTION. Two are;
    ``policy_temp`` is not — pre-fix, the operator-settable ``--policy-temp``
    fed every profile including the training rows, so a legacy dump made at 2.2
    was inferred as 1.0. That accepts a legacy-2.2 vs current-1.0 join (the
    exact failure this gate exists to stop) and refuses a legitimate 2.2-vs-2.2
    one.

    So absence is its own value. It cannot equal ANY concrete stamp — including
    one that happens to hold the GumbelConfig defaults, because such a dump was
    written by a build that DID stamp and therefore is not the legacy case.
    """
    from scripts.paired_compare import load_dump, require_same_ruler

    absent = load_dump(_write_jsonl(tmp_path / "absent.jsonl", [
        _stamped("p", 10.0, input_encoding="fen_only", batch_size=256),
    ]))
    explicit = load_dump(_write_jsonl(tmp_path / "explicit.jsonl", [
        _stamped("p", 11.0, input_encoding="fen_only", batch_size=256,
                 search_shape=_PRE_FIX_SHAPE),
    ]))

    with pytest.raises(SystemExit, match="search_shape"):
        require_same_ruler(absent, explicit, label_a="ABSENT", label_b="EXPLICIT")


def test_search_shape_is_skipped_for_a_metric_it_cannot_govern(tmp_path) -> None:
    """MUTANT (Codex P2): check the training-row ruler on a non-training metric.

    `search_shape` describes rows (d)/(e) only — the change that introduced it
    says "rows (b) and (c) are unaffected" — so refusing a `cand.raw.exp` or
    `cand.sf_soft.exp` comparison over it refuses a join it does not govern.
    BOTH branches: skipped for `cand.raw.exp`, still enforced for
    `cand.train.exp`, and still enforced when no metric is named.
    """
    from scripts.paired_compare import load_dump, require_same_ruler, ruler_fields_for

    assert "search_shape" not in ruler_fields_for("cand.raw.exp")
    assert "search_shape" not in ruler_fields_for("cand.sf_soft.exp")
    assert "search_shape" in ruler_fields_for("cand.train.exp")
    assert "search_shape" in ruler_fields_for("cand.train_fast.exp")
    assert "search_shape" in ruler_fields_for(None)
    # ...and `input_encoding` is never skipped: it governs every row.
    assert "input_encoding" in ruler_fields_for("cand.raw.exp")

    legacy = load_dump(_write_jsonl(tmp_path / "legacy.jsonl", [
        _stamped("p", 10.0, input_encoding="fen_only", batch_size=256),
    ]))
    fresh = load_dump(_write_jsonl(tmp_path / "fresh.jsonl", [
        _stamped("p", 20.0, input_encoding="fen_only", batch_size=256,
                 search_shape=_POST_FIX_SHAPE),
    ]))
    require_same_ruler(
        legacy, fresh, label_a="LEGACY", label_b="NEW", metric="cand.raw.exp",
    )
    with pytest.raises(SystemExit, match="search_shape"):
        require_same_ruler(
            legacy, fresh, label_a="LEGACY", label_b="NEW",
            metric="cand.train.exp",
        )


def test_two_post_fix_dumps_compare(tmp_path) -> None:
    """And the gate must be capable of PASSING on the current ruler."""
    from scripts.paired_compare import load_dump, require_same_ruler

    rows_a = [_stamped("p", 1.0, input_encoding="fen_only", batch_size=256,
                       search_shape=_POST_FIX_SHAPE)]
    rows_b = [_stamped("p", 2.0, input_encoding="fen_only", batch_size=256,
                       search_shape=_POST_FIX_SHAPE)]
    a = load_dump(_write_jsonl(tmp_path / "a.jsonl", rows_a))
    b = load_dump(_write_jsonl(tmp_path / "b.jsonl", rows_b))

    require_same_ruler(a, b, label_a="A", label_b="B")


def test_a_dump_that_mixes_two_search_shapes_is_refused(tmp_path) -> None:
    """Within-dump disagreement means the file is not one reading."""
    from scripts.paired_compare import load_dump, require_same_ruler

    mixed = load_dump(_write_jsonl(tmp_path / "mixed.jsonl", [
        _stamped("p", 1.0, input_encoding="fen_only", batch_size=256,
                 search_shape=_POST_FIX_SHAPE),
        _stamped("q", 2.0, input_encoding="fen_only", batch_size=256,
                 search_shape=_PRE_FIX_SHAPE),
    ]))
    ok = load_dump(_write_jsonl(tmp_path / "ok.jsonl", [
        _stamped("p", 1.0, input_encoding="fen_only", batch_size=256,
                 search_shape=_POST_FIX_SHAPE),
    ]))

    with pytest.raises(SystemExit, match="mixes two rulers"):
        require_same_ruler(mixed, ok, label_a="MIXED", label_b="OK")

def _stamped_header(path: Path, *, ruler: str, n: int = 6, off: float = 0.0,
             extra: dict | None = None) -> str:
    """A dump with a provenance HEADER on line 1, like `audit_cache` writes.

    ⚑ It must emit EVERY field `audit_cache_stamp` emits, `policy_map_version`
    included. While that key was missing here, adding it to
    `STAMP_NON_IDENTITY_KEYS` — i.e. silently retiring a real identity check —
    broke no test: `audit_ruler_version` and `audit_set_digest` each have a
    refuse-test, and this one had none. A fixture that under-declares turns the
    exclude set into an unpinned constant.
    """
    head = {"audit_cache_format": 1, "rows": n,
            "audit_set": "data/audit_set_v1.jsonl",
            "audit_set_digest": "deadbeef", "audit_ruler_version": ruler,
            "policy_map_version": "PMV-000",
            **(extra or {})}
    lines = [json.dumps(head)]
    lines += [json.dumps({"fen": f"p{i}", "value": 10.0 + off + i, "phase": 1})
              for i in range(n)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_the_stamp_is_captured_not_merely_skipped(tmp_path) -> None:
    """⚑ The header is not a data row, but "not a data row" is not "not evidence".

    Before PR #423's review `load_dump` recognised the stamp by its sentinel and
    DROPPED it, so nothing downstream could check what the dump declared.
    """
    from scripts.paired_compare import load_dump

    d = load_dump(_stamped_header(tmp_path / "a.jsonl", ruler="RULER-AAA"))
    assert d.stamp["audit_ruler_version"] == "RULER-AAA"
    # And it still is not counted as a row, which is what the skip was for.
    assert len(d.rows) == 6
    assert d.unusable == 0


def test_two_dumps_from_different_rulers_are_refused(tmp_path) -> None:
    """The reviewer's scenario, reproduced: two dumps declaring different
    `audit_ruler_version` used to join to exit 0 and print a verdict under a
    banner that reads as a provenance certificate."""
    from scripts.paired_compare import load_dump, require_same_stamp

    a = load_dump(_stamped_header(tmp_path / "a.jsonl", ruler="RULER-AAA"))
    b = load_dump(_stamped_header(tmp_path / "b.jsonl", ruler="RULER-BBB", off=-5.0))
    with pytest.raises(SystemExit, match="audit_ruler_version"):
        require_same_stamp(a, b, label_a="A", label_b="B")


def test_a_matching_pair_is_not_refused(tmp_path) -> None:
    """⚑ The control that stops this being a gate that cannot pass. Two dumps
    from ONE ruler must join even though every other stamp field is identical
    too — a guard that refuses everything is not a guard."""
    from scripts.paired_compare import load_dump, require_same_stamp

    a = load_dump(_stamped_header(tmp_path / "a.jsonl", ruler="RULER-AAA"))
    b = load_dump(_stamped_header(tmp_path / "b.jsonl", ruler="RULER-AAA", off=-5.0))
    require_same_stamp(a, b, label_a="A", label_b="B")


def test_a_differing_non_identity_key_is_not_refused(tmp_path) -> None:
    """`rows` and the human-readable `audit_set` PATH legitimately differ; the
    DIGEST is the provenance value and is deliberately not excluded."""
    from scripts.paired_compare import load_dump, require_same_stamp

    a = load_dump(_stamped_header(tmp_path / "a.jsonl", ruler="RULER-AAA", n=6))
    b = load_dump(_stamped_header(tmp_path / "b.jsonl", ruler="RULER-AAA", n=4, off=-5.0))
    require_same_stamp(a, b, label_a="A", label_b="B")


def test_a_differing_audit_set_digest_is_refused(tmp_path) -> None:
    """Two checkpoints scored against DIFFERENT position sets cannot be paired,
    and the digest is what says so — a path string cannot."""
    from scripts.paired_compare import load_dump, require_same_stamp

    a = load_dump(_stamped_header(tmp_path / "a.jsonl", ruler="R"))
    b = load_dump(_stamped_header(tmp_path / "b.jsonl", ruler="R", off=-5.0,
                           extra={"audit_set_digest": "cafebabe"}))
    with pytest.raises(SystemExit, match="audit_set_digest"):
        require_same_stamp(a, b, label_a="A", label_b="B")


def test_a_stamp_key_added_later_is_guarded_without_editing_this_file(
    tmp_path,
) -> None:
    """⚑ Why the rule is an EXCLUDE set. A version field added to the stamp
    later must be compared the day it appears; an include list would have to be
    edited in lockstep with the writer and would fail SILENTLY when it was not."""
    from scripts.paired_compare import load_dump, require_same_stamp

    a = load_dump(_stamped_header(tmp_path / "a.jsonl", ruler="R",
                           extra={"future_field_nobody_has_written_yet": "v1"}))
    b = load_dump(_stamped_header(tmp_path / "b.jsonl", ruler="R", off=-5.0,
                           extra={"future_field_nobody_has_written_yet": "v2"}))
    with pytest.raises(SystemExit, match="future_field_nobody_has_written_yet"):
        require_same_stamp(a, b, label_a="A", label_b="B")


def test_unstamped_dumps_warn_rather_than_refuse(tmp_path, capsys) -> None:
    """Dumps predating stamping are legitimately unstamped, and
    `require_same_ruler` already refuses the encoding mismatch that actually
    invalidates a join."""
    from scripts.paired_compare import load_dump, require_same_stamp

    a = load_dump(_write_jsonl(tmp_path / "a.jsonl",
                               [{"fen": "p0", "value": 1.0, "phase": 1}]))
    b = load_dump(_write_jsonl(tmp_path / "b.jsonl",
                               [{"fen": "p0", "value": 2.0, "phase": 1}]))
    require_same_stamp(a, b, label_a="A", label_b="B")
    assert "no provenance stamp" in capsys.readouterr().out


def test_the_stamp_gate_is_WIRED_into_the_command_line_path(
    tmp_path, monkeypatch,
) -> None:
    """⚑ THE MUTANT THAT SURVIVED THE FIRST BATTERY. Every other test in this
    group calls `require_same_stamp` DIRECTLY, so deleting its call from `main`
    left them all green — a guard that is correct and never invoked, which is
    this codebase's signature defect wearing the fix's own clothes.

    This one goes through the real entry point.
    """
    from scripts.paired_compare import main

    a = _stamped_header(tmp_path / "a.jsonl", ruler="RULER-AAA")
    b = _stamped_header(tmp_path / "b.jsonl", ruler="RULER-BBB", off=-5.0)
    monkeypatch.setattr(sys, "argv", ["paired_compare.py", a, b])
    with pytest.raises(SystemExit) as exc:
        main()
    assert "audit_ruler_version" in str(exc.value)


def test_the_command_line_path_still_joins_a_matching_pair(
    tmp_path, capsys, monkeypatch,
) -> None:
    """The wiring test's control: `main` must still produce a verdict when the
    stamps agree, or the gate above would be indistinguishable from a crash."""
    from scripts.paired_compare import main

    a = _stamped_header(tmp_path / "a.jsonl", ruler="RULER-AAA")
    b = _stamped_header(tmp_path / "b.jsonl", ruler="RULER-AAA", off=-5.0)
    monkeypatch.setattr(sys, "argv", ["paired_compare.py", a, b])
    main()
    assert "paired delta" in capsys.readouterr().out


def test_a_stamp_key_only_one_side_declares_is_reported(tmp_path, capsys) -> None:
    """⚑ Killed mutant M5 (union -> intersection over the two key sets).

    A key present on only one side means the two dumps came from different
    scorer BUILDS. It is a warning rather than a refusal — the older build
    simply did not write the field, and refusing would make every stamp
    addition retroactively unjoinable — but it must be SAID. With an
    intersection the key is never examined and the operator hears nothing,
    which is silence standing in for a finding.
    """
    from scripts.paired_compare import load_dump, require_same_stamp

    a = load_dump(_stamped_header(tmp_path / "a.jsonl", ruler="R",
                                  extra={"only_on_a": "v1"}))
    b = load_dump(_stamped_header(tmp_path / "b.jsonl", ruler="R", off=-5.0))
    require_same_stamp(a, b, label_a="A", label_b="B")
    out = capsys.readouterr().out
    assert "only_on_a" in out
    assert "only one side" in out


# ---------------------------------------------------------------------------
# #442 review — the stamp is now READ, not merely present
# ---------------------------------------------------------------------------


def test_a_differing_policy_map_version_is_refused(tmp_path) -> None:
    """⚑ F5: this check existed but was not BEHAVIOURALLY PINNED.

    `audit_cache_stamp` emits three identity fields. Two of them
    (`audit_ruler_version`, `audit_set_digest`) had a refuse-test; the third
    did not, and the test fixture did not even emit it — so moving
    `policy_map_version` into `STAMP_NON_IDENTITY_KEYS` and thereby retiring a
    real guard would have passed the whole suite. A guard with no test that
    fails when it is removed is a guard nobody is holding.
    """
    from scripts.paired_compare import load_dump, require_same_stamp

    a = load_dump(_stamped_header(tmp_path / "a.jsonl", ruler="R"))
    b = load_dump(_stamped_header(tmp_path / "b.jsonl", ruler="R", off=-5.0,
                                  extra={"policy_map_version": "PMV-999"}))
    with pytest.raises(SystemExit, match="policy_map_version"):
        require_same_stamp(a, b, label_a="A", label_b="B")


def test_a_stamp_format_this_tool_does_not_understand_is_refused(tmp_path) -> None:
    """⚑ F2: `if STAMP_FORMAT_KEY in r` is a PRESENCE test.

    `STAMP_FORMAT_KEY` was excluded from the identity comparison on the grounds
    that it is "equal by construction on any pair a reader accepts" — true of
    `read_audit_cache_stamp`, which RAISES on a mismatch, and FALSE of
    `load_dump`, which never read the value. MEASURED before the fix: format 1
    vs format 99, everything else identical, exit 0 with a verdict printed.
    """
    from scripts.paired_compare import load_dump

    path = _stamped_header(tmp_path / "b.jsonl", ruler="R",
                           extra={"audit_cache_format": 99})
    with pytest.raises(SystemExit, match="audit_cache_format"):
        load_dump(path)


def test_the_format_key_is_not_excluded_from_identity() -> None:
    """Belt and braces: the exclude set must no longer carry the format key.

    `load_dump`'s range check and `require_same_stamp`'s comparison are
    independent — the first catches a format neither side understands, the
    second catches a disagreeing pair — and a rationale that was false for one
    reader must not be left standing for the next one.
    """
    from chess_anti_engine.utils.audit_cache_format import (
        STAMP_FORMAT_KEY,
        STAMP_NON_IDENTITY_KEYS,
    )

    assert STAMP_FORMAT_KEY not in STAMP_NON_IDENTITY_KEYS


def test_a_stamp_declaring_the_wrong_row_count_is_refused(tmp_path) -> None:
    """⚑ F4: the stamp binds to line 1 only.

    `audit_cache_format.ROW_COUNT_KEY`'s own docstring says it exists to stop
    "a truncated file, or two caches concatenated", and `read_audit_cache`
    enforces it. `paired_compare` did not: MEASURED, a stamp declaring 9999
    rows over an 8-row body exited 0 with a verdict.
    """
    from scripts.paired_compare import load_dump

    path = _stamped_header(tmp_path / "a.jsonl", ruler="R", n=6)
    Path(path).write_text(
        Path(path).read_text(encoding="utf-8").replace('"rows": 6', '"rows": 9999', 1),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="9999 rows but the file holds 6"):
        load_dump(path)


def test_a_truncated_body_is_refused(tmp_path) -> None:
    """The same guard from the other direction: rows removed, stamp untouched."""
    from scripts.paired_compare import load_dump

    path = _stamped_header(tmp_path / "a.jsonl", ruler="R", n=6)
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    Path(path).write_text("\n".join(lines[:4]) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="6 rows but the file holds 3"):
        load_dump(path)


def test_a_stamp_with_no_row_count_is_refused(tmp_path) -> None:
    """A header that cannot vouch for its body is not provenance."""
    from scripts.paired_compare import load_dump

    path = _stamped_header(tmp_path / "a.jsonl", ruler="R", n=6)
    Path(path).write_text(
        Path(path).read_text(encoding="utf-8").replace('"rows": 6, ', "", 1),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="no integer 'rows' count"):
        load_dump(path)


def test_a_correct_row_count_still_loads(tmp_path) -> None:
    """The control. A guard that refuses every file is not a guard."""
    from scripts.paired_compare import load_dump

    d = load_dump(_stamped_header(tmp_path / "a.jsonl", ruler="R", n=6))
    assert len(d.rows) == 6


def test_a_second_provenance_header_is_refused(tmp_path) -> None:
    """⚑ F4: `stamp = dict(r)` was LAST-WINS.

    Two dumps concatenated kept only the final header and silently discarded
    the first — including a disagreeing `audit_ruler_version`. MEASURED before
    the fix: a file whose FIRST header declared `audit_ruler_version: R_EVIL`
    exited 0, the evil header having been overwritten by the second.
    `read_audit_cache` already refuses this exact shape.
    """
    from scripts.paired_compare import load_dump

    a = Path(_stamped_header(tmp_path / "a.jsonl", ruler="R_EVIL", n=3))
    b = Path(_stamped_header(tmp_path / "b.jsonl", ruler="R", n=3, off=1.0))
    merged = tmp_path / "merged.jsonl"
    merged.write_text(
        a.read_text(encoding="utf-8") + b.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="SECOND provenance header"):
        load_dump(str(merged))


def test_the_row_count_guard_is_wired_into_the_command_line_path(
    tmp_path, monkeypatch,
) -> None:
    """The lesson of this file's own surviving mutant: reach it through `main`.

    Every assertion above calls `load_dump` directly, so a guard that is
    correct and unreachable from the entry point would leave them all green.
    """
    from scripts.paired_compare import main

    a = _stamped_header(tmp_path / "a.jsonl", ruler="R", n=6)
    b = Path(_stamped_header(tmp_path / "b.jsonl", ruler="R", n=6, off=-5.0))
    b.write_text(
        b.read_text(encoding="utf-8").replace('"rows": 6', '"rows": 9999', 1),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["paired_compare.py", a, str(b)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert "9999 rows" in str(exc.value)


# ---------------------------------------------------------------------------
# #442 independent review — B3, and the Codex inline findings on this file
# ---------------------------------------------------------------------------


def test_a_provenance_header_that_follows_data_rows_is_refused(tmp_path) -> None:
    """A stamp certifies the body BELOW it, so a late header certifies nothing.

    Codex inline finding on `load_dump`: an unstamped dump with a stamped one
    appended presents a single header, which the reader accepted as line-1
    provenance. MEASURED before the fix — the pre-header rows were counted into
    `n_data_lines`, so a declared `rows` covering the whole file satisfied the
    count guard too, and the verdict printed under a banner certifying rows
    written before the stamp existed.
    """
    from scripts.paired_compare import load_dump

    stamped = Path(_stamped_header(tmp_path / "s.jsonl", ruler="R", n=3))
    late = tmp_path / "late.jsonl"
    legacy = "\n".join(
        json.dumps({"fen": f"q{i}", "value": 1.0 + i, "phase": 1}) for i in range(3)
    )
    body = stamped.read_text(encoding="utf-8")
    late.write_text(
        legacy + "\n" + body.replace('"rows": 3', '"rows": 6', 1), encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="after 3 data rows"):
        load_dump(str(late))


def test_a_core_stamp_key_missing_from_one_side_is_refused(tmp_path) -> None:
    """⚑ B3: half the one-sided-key hole, closed with a criterion that checks out.

    `audit_cache_stamp` writes `policy_map_version` and `audit_ruler_version`
    into EVERY stamped cache any writer has produced, so one side lacking one
    cannot be explained by writer skew — the warn path's whole justification.
    It is a stamp that did not come from the writer.
    """
    from scripts.paired_compare import load_dump, require_same_stamp

    a = load_dump(_stamped_header(tmp_path / "a.jsonl", ruler="R"))
    b_path = Path(_stamped_header(tmp_path / "b.jsonl", ruler="R", off=-5.0))
    kept = []
    for line in b_path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        rec.pop("policy_map_version", None)
        kept.append(json.dumps(rec))
    b_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    b = load_dump(str(b_path))
    assert "policy_map_version" not in b.stamp, "fixture did not remove the key"
    with pytest.raises(SystemExit, match="declares no 'policy_map_version'"):
        require_same_stamp(a, b, label_a="A", label_b="B")


def test_a_one_sided_EXTRA_key_still_only_warns(tmp_path, capsys) -> None:
    """The other half stays a warning, and B3's replacement rationale needs it.

    `scripts/monitor_fen.sh` joins a BANKED baseline against a fresh dump every
    deep cycle. A banked dump is written by an older build by definition, so the
    day a stamp field is added every baseline goes one-sided on it. Refusing
    there invalidates every baseline to catch a case the warning names.
    """
    from scripts.paired_compare import load_dump, require_same_stamp

    a = load_dump(_stamped_header(tmp_path / "a.jsonl", ruler="R"))
    b = load_dump(_stamped_header(tmp_path / "b.jsonl", ruler="R", off=-5.0,
                                  extra={"history_fill": "repeat"}))
    require_same_stamp(a, b, label_a="A", label_b="B")   # must NOT raise
    assert "history_fill" in capsys.readouterr().out


def test_the_compared_NET_is_not_treated_as_ruler_identity(tmp_path, capsys) -> None:
    """Codex inline finding on `foreign_net_audit`: `net` names the SUBJECT.

    `foreign_net_audit.py` stamps the net it scored. Two such caches are exactly
    what `paired_compare --join-key key --field exp_regret` is for, and their
    `net` values necessarily differ — so treating it as ruler identity refuses
    the comparison on the one field guaranteed to disagree. The subject of a
    measurement is not its ruler.
    """
    from scripts.paired_compare import load_dump, require_same_stamp

    a = load_dump(_stamped_header(tmp_path / "a.jsonl", ruler="R",
                                  extra={"net": "bt4.onnx", "topk": 5}))
    b = load_dump(_stamped_header(tmp_path / "b.jsonl", ruler="R", off=-5.0,
                                  extra={"net": "ours_ckpt443.pt", "topk": 5}))
    require_same_stamp(a, b, label_a="A", label_b="B")   # must NOT raise
    assert "net" not in capsys.readouterr().out.replace("net a", "")


def test_topk_beside_net_is_still_ruler_identity(tmp_path) -> None:
    """The negative control for the exclusion above: only `net` was excluded.

    `topk` is a foreign-net SCORING parameter — how many policy moves were
    cached — so two caches that disagree on it are not comparable. If the fix
    for "the gate cannot pass" had been to exclude foreign-net keys wholesale,
    this test is what would have caught it.
    """
    from scripts.paired_compare import load_dump, require_same_stamp

    a = load_dump(_stamped_header(tmp_path / "a.jsonl", ruler="R",
                                  extra={"net": "bt4.onnx", "topk": 5}))
    b = load_dump(_stamped_header(tmp_path / "b.jsonl", ruler="R", off=-5.0,
                                  extra={"net": "ours.pt", "topk": 20}))
    with pytest.raises(SystemExit, match="disagree on stamp key 'topk'"):
        require_same_stamp(a, b, label_a="A", label_b="B")
