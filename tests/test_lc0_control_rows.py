"""Row identity, held-out purity and the chance floor for the lc0 control.

Every test here is written to be KILLABLE by the mistake it guards against:

* the purity check must fail on an injected overlap, and must not be satisfiable
  by keeping two tar lists disjoint,
* the row id must be a function of CONTENT, so it survives the shard index and
  the per-conversion ``game_id`` restarting at 0,
* ``chance_level`` must return ``E[1/n]`` and must differ from ``1/E[n]``
  whenever the legal-move count varies.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from chess_anti_engine.eval.lc0_control_rows import (
    ROW_ID_VERSION,
    EmptyTrainCorpus,
    _digest_pair,
    build_train_id_index,
    chance_level,
    collect_split_ids,
    corpus_fingerprint,
    exposed_rows,
    frozen_minus_exposed,
    frozen_row_set,
    input_ids,
    legal_counts_for_ids,
    load_frozen,
    purity_against_train,
    row_ids,
    write_frozen,
)
from chess_anti_engine.moves import COMPACT_POLICY_SIZE
from chess_anti_engine.replay.sample import ReplaySample
from chess_anti_engine.replay.shard import (
    ShardMeta,
    load_shard_arrays,
    samples_to_arrays,
    save_local_shard_arrays,
)

PLANES = 175


def _sample(seed: int, *, game_id: int = 0, ply: int = 0) -> ReplaySample:
    """A row whose CONTENT is a pure function of ``seed``.

    ``game_id``/``ply`` are varied independently so a test can assert the id
    ignores them — the exact property that makes the purity check meaningful
    across two separate conversion runs.
    """
    rng = np.random.default_rng(seed)
    policy = rng.random(COMPACT_POLICY_SIZE).astype(np.float32)
    policy /= policy.sum()
    legal = np.zeros(COMPACT_POLICY_SIZE, dtype=np.uint8)
    legal[rng.choice(COMPACT_POLICY_SIZE, size=20 + seed % 15, replace=False)] = 1
    return ReplaySample(
        x=rng.random((PLANES, 8, 8)).astype(np.float32),
        policy_target=policy,
        wdl_target=int(seed % 3),
        legal_mask=legal,
        search_wdl=np.array([0.5, 0.3, 0.2], dtype=np.float32),
        moves_left=0.1,
        has_policy=True,
        is_selfplay=True,
        is_network_turn=True,
        game_id=game_id,
        ply_index=ply,
    )


def _write(shard_dir: Path, seeds: list[int], *, game_offset: int = 0) -> Path:
    shard_dir.mkdir(parents=True, exist_ok=True)
    samples = [_sample(s, game_id=game_offset + i, ply=i) for i, s in enumerate(seeds)]
    save_local_shard_arrays(
        shard_dir / "shard_000000.zarr",
        arrs=samples_to_arrays(samples),
        meta=ShardMeta(positions=len(samples)),
    )
    return shard_dir


def test_row_id_is_content_not_provenance(tmp_path: Path) -> None:
    """⚑ The same lc0 record converted twice gets the SAME id.

    Both dirs hold shard_000000.zarr and both restart game_id at their own
    offset — the two facts that make a name- or index-based id disjoint by
    construction and therefore useless as a purity check.
    """
    left = _write(tmp_path / "run_a", [1, 2, 3], game_offset=0)
    right = _write(tmp_path / "run_b", [1, 2, 3], game_offset=900)
    assert collect_split_ids([left]).ids == collect_split_ids([right]).ids


def test_row_id_separates_different_rows(tmp_path: Path) -> None:
    a = _write(tmp_path / "a", [1, 2, 3])
    b = _write(tmp_path / "b", [4, 5, 6])
    assert not set(collect_split_ids([a]).ids) & set(collect_split_ids([b]).ids)


def test_row_id_is_stable_across_calls(tmp_path: Path) -> None:
    shard = _write(tmp_path / "a", [7, 8, 9])
    assert collect_split_ids([shard]).ids == collect_split_ids([shard]).ids


def test_purity_passes_on_disjoint_content(tmp_path: Path) -> None:
    heldout = _write(tmp_path / "held", [10, 11, 12])
    train = _write(tmp_path / "train", [20, 21, 22])
    frozen = frozen_row_set([heldout], sample=100, seed=0)
    result = purity_against_train(
        frozen["row_ids"], [train], frozen_input_ids=frozen["input_ids"],
    )
    assert result.is_pure
    assert result.intersecting_ids == 0
    assert result.exposed_inputs == 0


def test_purity_fails_on_an_injected_overlap(tmp_path: Path) -> None:
    """⚑ THE MUTATION. One shared row must be enough to fail the check."""
    heldout = _write(tmp_path / "held", [10, 11, 12])
    train = _write(tmp_path / "train", [20, 11, 22])  # seed 11 is in both
    frozen = frozen_row_set([heldout], sample=100, seed=0)
    result = purity_against_train(
        frozen["row_ids"], [train], frozen_input_ids=frozen["input_ids"],
    )
    assert not result.is_pure
    assert result.intersecting_ids == 1
    assert result.exposed_inputs == 1


def test_purity_is_not_satisfied_by_renaming(tmp_path: Path) -> None:
    """Identical content under a DIFFERENT directory name still fails.

    The name-based check this replaces would pass here, which is the whole
    reason the id is content-derived.
    """
    heldout = _write(tmp_path / "hour_23", [30, 31])
    train = _write(tmp_path / "totally_different_hour", [30, 31])
    frozen = frozen_row_set([heldout], sample=100, seed=0)
    result = purity_against_train(
        frozen["row_ids"], [train], frozen_input_ids=frozen["input_ids"],
    )
    assert result.intersecting_ids == 2


def test_frozen_set_round_trips_and_pins_its_id_version(tmp_path: Path) -> None:
    heldout = _write(tmp_path / "held", list(range(40, 60)))
    payload = frozen_row_set([heldout], sample=8, seed=3)
    path = tmp_path / "frozen.json"
    digest = write_frozen(payload, path)
    assert len(digest) == 64
    reloaded = load_frozen(path)
    assert reloaded["row_ids"] == payload["row_ids"]
    assert reloaded["row_id_version"] == ROW_ID_VERSION


def test_frozen_set_refuses_a_foreign_id_version(tmp_path: Path) -> None:
    """A frozen set built under a different digest must not be silently used."""
    heldout = _write(tmp_path / "held", [1, 2, 3])
    payload = frozen_row_set([heldout], sample=3, seed=0)
    payload["row_id_version"] = "something_else"
    path = tmp_path / "frozen.json"
    write_frozen(payload, path)
    with pytest.raises(ValueError, match="row id version"):
        load_frozen(path)


def test_frozen_sample_covers_every_source(tmp_path: Path) -> None:
    """A stratified draw must not let one hour dominate the frozen set."""
    dirs = [_write(tmp_path / f"h{i}", list(range(i * 50, i * 50 + 30))) for i in range(6)]
    payload = frozen_row_set(dirs, sample=12, seed=1)
    assert payload["frozen_rows"] == 12
    per_source = [set(collect_split_ids([d]).ids) for d in dirs]
    frozen = set(payload["row_ids"])
    assert all(source & frozen for source in per_source), "a source contributed nothing"


def test_chance_level_is_expected_inverse_not_inverse_of_expected() -> None:
    """⚑ Jensen, on a set whose branching factor varies as chess does."""
    counts = np.array([1, 2, 5, 20, 40, 60], dtype=np.int64)
    level = chance_level(counts)
    assert level.expected_inverse == pytest.approx(float((1.0 / counts).mean()))
    assert level.inverse_of_mean == pytest.approx(1.0 / counts.mean())
  # The wrong statistic is strictly SMALLER, i.e. it puts the floor of a
  # negative control BELOW the real floor.
    assert level.inverse_of_mean < level.expected_inverse
    assert level.jensen_ratio > 1.0


def test_chance_level_agrees_only_when_n_is_constant() -> None:
    level = chance_level(np.full(50, 20, dtype=np.int64))
    assert level.expected_inverse == pytest.approx(level.inverse_of_mean)
    assert level.jensen_ratio == pytest.approx(1.0)


def test_legal_counts_are_taken_for_the_frozen_rows_only(tmp_path: Path) -> None:
    """The floor must describe the rows the gate is read on, not the pool."""
    pool = _write(tmp_path / "held", list(range(100, 130)))
    payload = frozen_row_set([pool], sample=7, seed=2)
    counts = legal_counts_for_ids([pool], payload["row_ids"])
    assert counts.size == 7
    arrays_ids = collect_split_ids([pool])
    assert arrays_ids.rows == 30
    assert counts.min() > 0


def test_legal_counts_refuse_a_frozen_set_the_shards_do_not_contain(tmp_path: Path) -> None:
    pool = _write(tmp_path / "held", [1, 2, 3])
    other = _write(tmp_path / "other", [90, 91, 92])
    payload = frozen_row_set([pool], sample=3, seed=0)
    with pytest.raises(ValueError, match="not found"):
        legal_counts_for_ids([other], payload["row_ids"])


def test_duplicate_ids_within_a_split_are_reported(tmp_path: Path) -> None:
    """The id's own discrimination check must not be silent."""
    shard = _write(tmp_path / "dup", [5, 5, 6])
    split = collect_split_ids([shard])
    assert split.rows == 3
    assert split.duplicate_ids == 1


def test_row_ids_reject_a_ragged_shard() -> None:
    arrs = {
        "x": np.zeros((3, PLANES, 8, 8), dtype=np.float16),
        "policy_target": np.zeros((2, COMPACT_POLICY_SIZE), dtype=np.float16),
    }
    with pytest.raises(ValueError, match="rows, expected"):
        row_ids(arrs)


# ── exposure vs record identity (review F5 / Codex #5) ────────────────────────
#
# ⚑ "0 intersecting ids" is not "0 exposure". On the shipped smoke split the
# record-id gate reported 0 while 450 held-out INPUTS were already in train,
# because the same position carrying a different lc0 policy target is a
# different record. The arm's whole readout is held-out generalisation, so the
# gate has to count inputs.


def _sample_with_target(seed: int, *, target_seed: int) -> ReplaySample:
    """The SAME `x` as ``_sample(seed)``, with a DIFFERENT policy target.

    This is the real failure mode in one object: lc0 searching the same
    position twice (search noise, a newer net) writes two different visit
    distributions over one input.
    """
    sample = _sample(seed)
    rng = np.random.default_rng(10_000 + target_seed)
    policy = rng.random(COMPACT_POLICY_SIZE).astype(np.float32)
    policy /= policy.sum()
    sample.policy_target = policy
    return sample


def _write_samples(shard_dir: Path, samples: list[ReplaySample]) -> Path:
    shard_dir.mkdir(parents=True, exist_ok=True)
    save_local_shard_arrays(
        shard_dir / "shard_000000.zarr",
        arrs=samples_to_arrays(samples),
        meta=ShardMeta(positions=len(samples)),
    )
    return shard_dir


def test_the_record_id_reports_pure_on_a_split_that_is_exposed(tmp_path: Path) -> None:
    """⚑⚑ THE FINDING, reproduced in miniature: 0 by record id, 3 by input."""
    heldout = _write_samples(
        tmp_path / "held", [_sample_with_target(s, target_seed=1) for s in (1, 2, 3)],
    )
    train = _write_samples(
        tmp_path / "train", [_sample_with_target(s, target_seed=2) for s in (1, 2, 3)],
    )
    frozen = frozen_row_set([heldout], sample=3, seed=0)
    result = purity_against_train(
        frozen["row_ids"], [train], frozen_input_ids=frozen["input_ids"],
    )
    assert result.intersecting_ids == 0, "the record ids must genuinely differ"
    assert result.exposed_inputs == 3, "every input was already trained on"
    assert not result.is_pure, (
        "the gate passed a split in which the model has seen every held-out "
        "input — this is exactly the state the reviewer measured at 450 rows"
    )


def test_duplicate_inputs_are_reported_alongside_duplicate_ids(tmp_path: Path) -> None:
    """`pool_duplicate_ids` said 3 where input duplication was 3,153."""
    shard = _write_samples(
        tmp_path / "dup",
        [_sample_with_target(7, target_seed=i) for i in range(4)] + [_sample(8)],
    )
    split = collect_split_ids([shard])
    assert split.rows == 5
    assert split.duplicate_ids == 0, "four distinct targets are four records"
    assert split.duplicate_inputs == 3, "but they are ONE position, seen 4 times"


def test_purity_refuses_a_train_corpus_with_no_rows(tmp_path: Path) -> None:
    """⚑ Codex #6. A check that scanned nothing must not be able to say PURE."""
    heldout = _write(tmp_path / "held", [1, 2, 3])
    empty = tmp_path / "empty_train"
    empty.mkdir()
    frozen = frozen_row_set([heldout], sample=3, seed=0)
    with pytest.raises(EmptyTrainCorpus, match="ZERO rows"):
        purity_against_train(
            frozen["row_ids"], [empty], frozen_input_ids=frozen["input_ids"],
        )


def test_a_frozen_set_without_exposure_ids_is_refused(tmp_path: Path) -> None:
    """An artifact from before the gate moved to inputs cannot be re-used."""
    heldout = _write(tmp_path / "held", [1, 2, 3])
    payload = frozen_row_set([heldout], sample=3, seed=0)
    payload.pop("input_ids")
    path = tmp_path / "frozen.json"
    write_frozen(payload, path)
    with pytest.raises(ValueError, match="exposure"):
        load_frozen(path)


def test_frozen_payload_carries_one_exposure_id_per_row(tmp_path: Path) -> None:
    dirs = [_write(tmp_path / f"h{i}", list(range(i * 50, i * 50 + 30))) for i in range(3)]
    payload = frozen_row_set(dirs, sample=9, seed=1)
    assert len(payload["input_ids"]) == len(payload["row_ids"]) == 9
    assert payload["pool_unique_inputs"] <= payload["pool_rows"]


# ── the two-id fast path ──────────────────────────────────────────────────────


def test_digest_pair_matches_the_generic_content_id(tmp_path: Path) -> None:
    """⚑ The scan's speed must not come from a DIFFERENT id.

    ``_digest_pair`` hashes ``x`` once and forks the blake2b state, so it is
    only equal to hashing ``x`` and ``x || policy_target`` separately while the
    input fields are a prefix of the record fields. Equality is asserted here
    against the generic implementation rather than argued in a comment.
    """
    shard = _write(tmp_path / "a", [1, 2, 3, 4])
    arrs, _meta = load_shard_arrays(shard / "shard_000000.zarr")
    record, inputs = _digest_pair(arrs)
    assert [value.decode("ascii") for value in record.tolist()] == row_ids(arrs)
    assert [value.decode("ascii") for value in inputs.tolist()] == input_ids(arrs)


def test_digest_pair_rejects_a_ragged_shard() -> None:
    arrs = {
        "x": np.zeros((3, PLANES, 8, 8), dtype=np.float16),
        "policy_target": np.zeros((2, COMPACT_POLICY_SIZE), dtype=np.float16),
    }
    with pytest.raises(ValueError, match="rows, expected"):
        _digest_pair(arrs)


# ── the train id cache: the honest re-check has to be cheap ───────────────────
#
# ⚑ The 2026-08-16 scan read 42.2 GB over 2h40m to rebuild a set that is a pure
# function of a fixed corpus. A gate that costs 2h40m to re-run is the gate that
# gets skipped under time pressure. But a cache that always hits is a gate that
# CANNOT fail, so every test below is paired: one that it is reused, one that it
# is REJECTED the moment the corpus is not byte-identical.


def test_train_id_index_counts_match_the_scan(tmp_path: Path) -> None:
    train = _write(tmp_path / "train", [1, 2, 3])
    index = build_train_id_index([train])
    assert index.rows == 3
    assert index.record_ids.size == 3
    assert index.input_ids.size == 3
    assert not index.from_cache


def test_train_id_index_is_identical_in_parallel(tmp_path: Path) -> None:
    """The parallel scan must not be a different measurement."""
    dirs = [_write(tmp_path / f"d{i}", list(range(i * 5, i * 5 + 5))) for i in range(4)]
    serial = build_train_id_index(dirs, workers=1)
    parallel = build_train_id_index(dirs, workers=3)
    assert serial.rows == parallel.rows
    assert np.array_equal(serial.record_ids, parallel.record_ids)
    assert np.array_equal(serial.input_ids, parallel.input_ids)


def test_train_id_cache_is_reused_on_an_unchanged_corpus(tmp_path: Path) -> None:
    train = _write(tmp_path / "train", [1, 2, 3])
    cache = tmp_path / "cache"
    first = build_train_id_index([train], cache_dir=cache)
    assert not first.from_cache
    second = build_train_id_index([train], cache_dir=cache)
    assert second.from_cache, "the second call re-scanned a byte-identical corpus"
    assert second.rows == first.rows
    assert np.array_equal(second.record_ids, first.record_ids)
    assert np.array_equal(second.input_ids, first.input_ids)


def test_train_id_cache_is_rejected_when_a_shard_is_added(tmp_path: Path) -> None:
    """⚑⚑ THE MUTATION-CRITICAL ONE. A cache that always hits cannot fail.

    Growing the corpus must produce a fresh scan AND a bigger id set. A key
    that ignored file contents — directory names only, say — would return the
    3-row index here and clear a corpus it never read.
    """
    train = _write(tmp_path / "train", [1, 2, 3])
    cache = tmp_path / "cache"
    first = build_train_id_index([train], cache_dir=cache)
    _write(tmp_path / "train2", [4, 5, 6])
    grown = build_train_id_index([train, tmp_path / "train2"], cache_dir=cache)
    assert not grown.from_cache
    assert grown.rows == 6
    assert grown.record_ids.size == first.record_ids.size + 3


def test_train_id_cache_is_rejected_when_a_shard_is_rewritten(tmp_path: Path) -> None:
    """Same directory list, same shard count, DIFFERENT rows."""
    train = tmp_path / "train"
    _write(train, [1, 2, 3])
    cache = tmp_path / "cache"
    before = build_train_id_index([train], cache_dir=cache)
    shutil.rmtree(train)
    _write(train, [7, 8, 9])
    after = build_train_id_index([train], cache_dir=cache)
    assert not after.from_cache, "a rewritten corpus reused its predecessor's ids"
    assert not np.array_equal(before.record_ids, after.record_ids)


def test_corpus_fingerprint_changes_when_one_byte_changes(tmp_path: Path) -> None:
    """The key must be content-sensitive, not name-sensitive."""
    train = _write(tmp_path / "train", [1, 2, 3])
    before = corpus_fingerprint([train])
    chunk = next(
        path for path in sorted((train / "shard_000000.zarr" / "x").iterdir())
        if path.is_file() and not path.name.startswith(".")
    )
    chunk.write_bytes(chunk.read_bytes() + b"\0")
    after = corpus_fingerprint([train])
    assert before.key != after.key
    assert before.files == after.files, "the file LIST is unchanged; the size is not"


def test_corpus_fingerprint_is_stable_across_calls(tmp_path: Path) -> None:
    train = _write(tmp_path / "train", [1, 2, 3])
    assert corpus_fingerprint([train]).key == corpus_fingerprint([train]).key


def test_a_cache_whose_stored_key_disagrees_with_its_name_is_ignored(
    tmp_path: Path,
) -> None:
    """The file NAME is a hint; the key INSIDE it is the check.

    Without this the name-vs-content branch in ``load_train_id_index`` could
    not be reached by any test, i.e. it would be a guard nobody had ever seen
    fire — which is how a cache silently starts answering for another corpus
    after someone copies a cache file.
    """
    train = _write(tmp_path / "train", [1, 2, 3])
    cache = tmp_path / "cache"
    build_train_id_index([train], cache_dir=cache)
    banked = next(iter(cache.glob("train_ids_*.npz")))
    with np.load(banked, allow_pickle=False) as blob:
        payload = {name: blob[name] for name in blob.files}
    payload["key"] = np.array("f" * 64)
    np.savez(banked.with_suffix(""), **payload)
    assert not build_train_id_index([train], cache_dir=cache).from_cache


def test_a_cache_from_another_corpus_is_not_loaded(tmp_path: Path) -> None:
    """Two corpora sharing one cache dir must not read each other's index."""
    left = _write(tmp_path / "left", [1, 2, 3])
    right = _write(tmp_path / "right", [4, 5, 6, 7])
    cache = tmp_path / "cache"
    build_train_id_index([left], cache_dir=cache)
    other = build_train_id_index([right], cache_dir=cache)
    assert not other.from_cache
    assert other.rows == 4


# ── the exposed set is an OPERAND, not a count ────────────────────────────────


def test_purity_banks_the_exposed_ids_themselves(tmp_path: Path) -> None:
    """⚑ The blocker of 2026-08-16: 5,065 was banked, the 5,065 ids were not."""
    heldout = _write_samples(
        tmp_path / "held",
        [_sample_with_target(s, target_seed=1) for s in (1, 2, 3)] + [_sample(9)],
    )
    train = _write_samples(
        tmp_path / "train", [_sample_with_target(s, target_seed=2) for s in (1, 2)],
    )
    frozen = frozen_row_set([heldout], sample=4, seed=0)
    result = purity_against_train(
        frozen["row_ids"], [train], frozen_input_ids=frozen["input_ids"],
    )
    assert result.exposed_inputs == 2
    assert len(result.exposed_input_ids) == result.exposed_inputs
    exposed_side = {
        input_id
        for row_id, input_id in zip(frozen["row_ids"], frozen["input_ids"], strict=True)
        if row_id in set(collect_split_ids([train]).ids) or input_id in set(
            collect_split_ids([train]).inputs,
        )
    }
    assert set(result.exposed_input_ids) == exposed_side


def test_exposed_rows_round_trip_the_ids_that_build_the_clean_set(
    tmp_path: Path,
) -> None:
    """⚑ The dump must name ROWS, not only inputs.

    Rows share inputs (100,000 frozen rows over 96,853 distinct positions), so
    an input-hash list alone does not say what to drop. This asserts the two
    views agree: the rows the dump names are exactly the rows the subtraction
    removes.
    """
    duplicated = [_sample_with_target(1, target_seed=i) for i in range(3)]
    heldout = _write_samples(
        tmp_path / "held", [*duplicated, _sample(9), _sample(10)],
    )
    train = _write_samples(tmp_path / "train", [_sample_with_target(1, target_seed=99)])
    frozen = frozen_row_set([heldout], sample=5, seed=0)
    result = purity_against_train(
        frozen["row_ids"], [train], frozen_input_ids=frozen["input_ids"],
    )
    assert result.exposed_inputs == 1, "one POSITION, carried by three rows"
    named = exposed_rows(frozen, result.exposed_input_ids)
    assert len(named) == 3
    trimmed = frozen_minus_exposed(frozen, result.exposed_input_ids)
    assert {row["row_id"] for row in named} == (
        set(frozen["row_ids"]) - set(trimmed["row_ids"])
    )


def test_frozen_minus_exposed_preserves_order_and_recounts(tmp_path: Path) -> None:
    heldout = _write(tmp_path / "held", list(range(20, 32)))
    frozen = frozen_row_set([heldout], sample=12, seed=0)
    drop = {frozen["input_ids"][3], frozen["input_ids"][7]}
    trimmed = frozen_minus_exposed(frozen, sorted(drop))
    assert trimmed["frozen_rows"] == 10
    assert trimmed["frozen_unique_inputs"] == 10
    assert trimmed["removed_rows"] == 2
    assert trimmed["row_ids"] == [
        row_id
        for row_id, input_id in zip(frozen["row_ids"], frozen["input_ids"], strict=True)
        if input_id not in drop
    ]
  # The surviving rows keep their POSITION relative to one another, which is
  # what lets the banked per-row score arrays be masked rather than re-scored.
    assert trimmed["row_ids"] == [
        row_id for row_id in frozen["row_ids"] if row_id in set(trimmed["row_ids"])
    ]


def test_the_trimmed_set_is_pure_against_the_corpus_that_failed(
    tmp_path: Path,
) -> None:
    """The repair, end to end: FAIL, subtract, PASS — on the same train dirs.

    ⚑ Clean BY CONSTRUCTION is exactly why it is verified rather than assumed.
    """
    heldout = _write_samples(
        tmp_path / "held",
        [_sample_with_target(s, target_seed=1) for s in (1, 2, 3)]
        + [_sample(40), _sample(41)],
    )
    train = _write_samples(
        tmp_path / "train",
        [_sample_with_target(s, target_seed=2) for s in (1, 2, 3)] + [_sample(77)],
    )
    frozen = frozen_row_set([heldout], sample=5, seed=0)
    failed = purity_against_train(
        frozen["row_ids"], [train], frozen_input_ids=frozen["input_ids"],
    )
    assert not failed.is_pure
    trimmed = frozen_minus_exposed(frozen, failed.exposed_input_ids)
    assert trimmed["frozen_rows"] == 2
    passed = purity_against_train(
        trimmed["row_ids"], [train], frozen_input_ids=trimmed["input_ids"],
    )
    assert passed.is_pure
    assert passed.exposed_inputs == 0
    assert passed.intersecting_ids == 0
