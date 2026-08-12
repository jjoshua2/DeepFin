"""The server compactor must carry EVERY ``ShardMeta`` field, not 48 of 56.

Why this file exists (audit wave 3, K1 / G3-1): the compactor rebuilt
``ShardMeta`` from a hand-written kwarg list. A hand-written list cannot fail
when a field is ABSENT from it, so eight fields silently took their dataclass
defaults on every compacted shard — and compaction is unconditional, so that is
100% of production shards. Two of the eight (``opponent_wdl_regret_limit``,
``sf_nodes``) are the promotion gate's ONLY instrument for the confound its own
design document names as dominant; ``gate_sample_confound_elo`` was NaN on 109
of 109 live rows and the gate's pre-registered KILL leg evaluated an empty set.

The 110-green gate suite missed it because
``test_ingest_splits_anchored_counts_by_publishing_model`` hand-writes shards
into the inbox with the field already set: it tests the CONSUMER against an
input the PRODUCER never emits. So the tests here:

1. enumerate ``dataclasses.fields(ShardMeta)`` — never a hand-list, because a
   diff-based completeness check is blind to exactly the ABSENT key that caused
   this — and drive a synthetic upload in which every field carries a
   distinguishable non-default value through the real accumulate -> flush path;
2. walk the real chain worker producer -> HTTP upload -> server compactor ->
   trainer ingest and require the two gate-critical fields to arrive finite and
   numerically equal to what the worker stamped.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from chess_anti_engine.moves.encode import normalize_policy_encoding
from chess_anti_engine.replay import ArrayReplayBuffer, ReplaySample
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.replay.shard import (
    LOCAL_SHARD_SUFFIX,
    SHARD_VERSION,
    ShardMeta,
    load_shard_arrays,
    pack_shard_for_upload,
)
from chess_anti_engine.server.app import (
    _AGGREGATE_COUNTER_FIELDS,
    _AGGREGATE_FLOAT_FIELDS,
    _BufferedUploadAccumulator,
    _COMPACTION_IDENTITY_FIELDS,
    _SHARD_META_FIELD_KINDS,
    _flush_buffered_upload_to_inbox,
    _upload_identity_acc_key,
)
from chess_anti_engine.tune.distributed_runtime import _ingest_distributed_selfplay
from chess_anti_engine.tune.trainable_phases import _arm_mean_regret
from chess_anti_engine.worker_buffer import (
    _buffer_add_completed_game,
    _BufferedUpload,
    _flush_upload_buffer_to_pending,
)

# The difficulty the promotion gate's confound leg needs. Real values off a
# live shard, kept exact: the point of the fix is that the number the worker
# stamped is the number ingest reads, not something rounded on the way.
LIVE_REGRET_LIMIT = 0.08165037588615515
LIVE_SF_NODES = 698289

TRIAL_ID = "13a9f_00000"
MODEL_SHA = "a1b2c3d4e5f60718"

# Per-shard identity values. Every one of these is part of the accumulator key,
# so a merge group is uniform in all of them; the compacted shard carries them
# verbatim rather than picking an arbitrary member's value.
_IDENTITY_VALUES: dict[str, Any] = {
    "model_sha256": MODEL_SHA,
    "input_history_encoding": "lc0_root",
    "history_rep_fix": True,
    # Declared as an ALIAS on the way in: the shard writer canonicalizes the
    # pair from the policy arrays, so the value on disk is the normalized name,
    # not the string the upload declared.
    "policy_encoding": "lc0_4672",
    "policy_size": 4672,
    "opponent_wdl_regret_limit": LIVE_REGRET_LIMIT,
    "sf_nodes": LIVE_SF_NODES,
}


def _sample(i: int = 0) -> ReplaySample:
    pol = np.zeros(4672, dtype=np.float32)
    pol[i % 4672] = 1.0
    return ReplaySample(
        x=np.zeros((146, 8, 8), dtype=np.float32),
        policy_target=pol,
        wdl_target=1,
    )


def _synthetic_upload_meta(*, variant: int) -> dict[str, Any]:
    """An upload meta with every ShardMeta field set, distinguishably.

    Values are derived from the classification table rather than typed out, so
    a field added to ``ShardMeta`` is covered here the moment it is classified
    — and if it is NOT classified, the module refuses to import.
    """
    meta: dict[str, Any] = {}
    for i, name in enumerate(_AGGREGATE_COUNTER_FIELDS):
        meta[name] = 1000 * variant + i + 1
    for i, name in enumerate(_AGGREGATE_FLOAT_FIELDS):
        meta[name] = 1000.0 * variant + i + 0.5
    meta.update(_IDENTITY_VALUES)
    meta["diff_focus_priority_min"] = 1.0 + variant
    meta["diff_focus_priority_max"] = 5.0 + 50.0 * variant
    meta["outcome_stats"] = {"shared_stat": variant + 1, f"only_v{variant}": 7}
    meta["model_step"] = 100 + variant
    # Writer-owned fields, set to values the compacted shard must NOT inherit.
    # Asserting the writer's own value below is a positive check; skipping the
    # field would reopen the hole this file exists to close.
    meta["version"] = 999
    meta["username"] = f"worker_{variant}"
    meta["generated_at_unix"] = 111 + variant
    meta["positions"] = 999_999
    meta["run_id"] = TRIAL_ID
    return meta


def _expected_compacted_meta(a: dict[str, Any], b: dict[str, Any], *, now_unix: float,
                             total_positions: int) -> dict[str, Any]:
    """What every ShardMeta field must be after compacting ``a`` and ``b``."""
    expected: dict[str, Any] = {}
    for name in _AGGREGATE_COUNTER_FIELDS:
        expected[name] = int(a[name]) + int(b[name])
    for name in _AGGREGATE_FLOAT_FIELDS:
        expected[name] = float(a[name]) + float(b[name])
    for name in (*_COMPACTION_IDENTITY_FIELDS, "model_sha256", "input_history_encoding",
                 "history_rep_fix"):
        expected[name] = a[name]
    expected["policy_encoding"] = normalize_policy_encoding(str(a["policy_encoding"]))
    expected["diff_focus_priority_min"] = min(a["diff_focus_priority_min"],
                                              b["diff_focus_priority_min"])
    expected["diff_focus_priority_max"] = max(a["diff_focus_priority_max"],
                                              b["diff_focus_priority_max"])
    expected["outcome_stats"] = {"shared_stat": 3, "only_v0": 7, "only_v1": 7}
    expected["model_step"] = b["model_step"]
    expected["version"] = SHARD_VERSION
    expected["username"] = "server_compactor"
    expected["generated_at_unix"] = int(now_unix)
    expected["positions"] = int(total_positions)
    expected["run_id"] = TRIAL_ID
    # writer_owned_provenance: the compactor assembles this from the
    # AUTHENTICATED uploader of each contributing upload, in row order. Row
    # ranges, not just a name set, so a ban can excise one contributor's rows
    # instead of discarding a shard several volunteers contributed to.
    expected["contributors"] = [
        {"username": "alice", "start": 0, "count": 2},
        {"username": "bob", "start": 2, "count": 2},
    ]
    return expected


def test_shard_meta_field_kinds_cover_every_declared_field() -> None:
    """MUTATION: add a field to ShardMeta without classifying it.

    ``chess_anti_engine.server.app`` raises at IMPORT when the table and the
    dataclass disagree, so this assertion is really a second witness. It is
    written as a set comparison in both directions because the failure that
    produced K1 was an ABSENT key: a check that only walks the keys it already
    knows about cannot see the one nobody wrote down.
    """
    declared = {f.name for f in dataclasses.fields(ShardMeta)}
    classified = set(_SHARD_META_FIELD_KINDS)
    assert classified == declared, (
        f"unclassified={sorted(declared - classified)} "
        f"stale={sorted(classified - declared)}"
    )
    assert declared, "ShardMeta has no fields — enumeration is not running"


def test_compactor_carries_every_shard_meta_field(tmp_path: Path) -> None:
    """THE recurrence guard. MUTATION: drop any field from the flush build.

    Every ``ShardMeta`` field is enumerated from the dataclass and given a
    distinguishable non-default value on the way in; the compacted shard must
    show the aggregation its classification promises. A field the compactor
    forgets shows up here as its dataclass default (usually ``None``), which is
    exactly how ``opponent_wdl_regret_limit`` reached production dead.
    """
    acc = _BufferedUploadAccumulator(
        trial_id=TRIAL_ID,
        model_sha256=MODEL_SHA,
        created_at_unix=100.0,
        last_update_unix=100.0,
    )
    meta_a = _synthetic_upload_meta(variant=0)
    meta_b = _synthetic_upload_meta(variant=1)
    # Distinct usernames so `contributors` has to carry BOTH -- a compactor that
    # kept only the first or last uploader would still tile the rows correctly.
    acc.add_upload(
        samples=[_sample(0), _sample(1)], meta=meta_a, now_unix=100.0, username="alice",
    )
    acc.add_upload(
        samples=[_sample(2), _sample(3)], meta=meta_b, now_unix=101.0, username="bob",
    )

    now_unix = 12345.0
    out = _flush_buffered_upload_to_inbox(
        inbox_root=tmp_path / "inbox",
        acc=acc,
        now_unix=now_unix,
        flush_token="0123456789abcdef",
    )
    assert out is not None
    _arrs, meta = load_shard_arrays(out)

    expected = _expected_compacted_meta(meta_a, meta_b, now_unix=now_unix, total_positions=4)
    # Enumerated from the dataclass, so a NEW field cannot slip past by simply
    # not being mentioned in this file.
    missing_expectation = sorted(
        f.name for f in dataclasses.fields(ShardMeta) if f.name not in expected
    )
    assert not missing_expectation, (
        f"no expected value written for {missing_expectation}; classify the field "
        "and extend _expected_compacted_meta"
    )

    defaults = {f.name: f.default for f in dataclasses.fields(ShardMeta)}
    for name, want in expected.items():
        got = meta.get(name, "<ABSENT>")
        if isinstance(want, float):
            assert got == pytest.approx(want), f"{name}: {got!r} != {want!r}"
        else:
            assert got == want, f"{name}: {got!r} != {want!r}"
        # ...and it did not merely happen to equal the default it would have
        # taken had the compactor dropped it. ``version`` is the one field the
        # writer legitimately owns at its default value.
        if name != "version":
            assert got != defaults[name], f"{name} took its ShardMeta default"


def test_compactor_splits_accumulators_by_difficulty() -> None:
    """MUTATION: drop the difficulty pair from ``_upload_identity_acc_key``.

    The worker flushes its own upload buffer when difficulty changes, so a raw
    shard's recorded difficulty is EXACT rather than an average. The server has
    to match that: merging two difficulties under one model sha would write a
    number that is true of neither half of the games, and nothing downstream
    can detect it. The key is what makes that unreachable — the accumulator's
    uniformity raise is only the backstop, so both are asserted here.
    """
    base = _synthetic_upload_meta(variant=0)
    assert _upload_identity_acc_key(base) == _upload_identity_acc_key(
        _synthetic_upload_meta(variant=1)
    ), "uploads that agree on every identity field must share an accumulator"
    for name, other in (
        ("opponent_wdl_regret_limit", LIVE_REGRET_LIMIT + 1e-6),
        ("sf_nodes", LIVE_SF_NODES + 1),
        ("policy_encoding", "lc0_1858"),
        ("policy_size", 1858),
        ("input_history_encoding", "legacy"),
        ("history_rep_fix", False),
    ):
        changed = dict(base)
        changed[name] = other
        assert _upload_identity_acc_key(changed) != _upload_identity_acc_key(base), (
            f"{name} is not part of the compaction merge key"
        )
    # Absent is its own identity: a legacy shard with no difficulty recorded
    # must not be merged into a shard that has one (0.0 regret would then be
    # invented for games played against a handicapped opponent, or vice versa).
    absent = dict(base)
    absent.pop("opponent_wdl_regret_limit")
    assert _upload_identity_acc_key(absent) != _upload_identity_acc_key(base)

    acc = _BufferedUploadAccumulator(
        trial_id=TRIAL_ID,
        model_sha256=MODEL_SHA,
        created_at_unix=0.0,
        last_update_unix=0.0,
    )
    base = _synthetic_upload_meta(variant=0)
    acc.add_upload(samples=[_sample(0)], meta=base, now_unix=0.0)
    mixed = dict(base)
    mixed["opponent_wdl_regret_limit"] = LIVE_REGRET_LIMIT + 0.01
    with pytest.raises(ValueError, match="mixed opponent_wdl_regret_limit"):
        acc.add_upload(samples=[_sample(1)], meta=mixed, now_unix=1.0)

    mixed_nodes = dict(base)
    mixed_nodes["sf_nodes"] = LIVE_SF_NODES + 1
    with pytest.raises(ValueError, match="mixed sf_nodes"):
        acc.add_upload(samples=[_sample(2)], meta=mixed_nodes, now_unix=2.0)


def test_absent_difficulty_stays_absent_not_zero(tmp_path: Path) -> None:
    """A shard predating the field must compact to ``None``, never to ``0.0``.

    ``0.0`` regret means UNHANDICAPPED Stockfish — the opposite end of the
    range from "no data" — and every consumer distinguishes the two. A
    ``float(x or 0.0)`` style carry-through would silently invent the hardest
    possible opponent for legacy shards.
    """
    acc = _BufferedUploadAccumulator(
        trial_id=TRIAL_ID,
        model_sha256=MODEL_SHA,
        created_at_unix=0.0,
        last_update_unix=0.0,
    )
    legacy = {"games": 1, "wins": 1, "model_step": 3}
    acc.add_upload(samples=[_sample(0)], meta=legacy, now_unix=0.0)
    out = _flush_buffered_upload_to_inbox(
        inbox_root=tmp_path / "inbox", acc=acc, now_unix=5.0, flush_token="fedcba9876543210",
    )
    assert out is not None
    _arrs, meta = load_shard_arrays(out)
    assert meta.get("opponent_wdl_regret_limit") is None
    assert meta.get("sf_nodes") is None
    assert meta.get("sf_d6_sum") == 0.0
    assert meta.get("sf_d6_n") == 0


# --------------------------------------------------------------------------
# producer -> consumer: the path no previous test walked end to end
# --------------------------------------------------------------------------
def _seed_user(server_root: Path, username: str = "u", password: str = "p") -> None:
    from chess_anti_engine.server.auth import UserRecord, hash_password, save_users

    salt, hsh, iters = hash_password(password)
    users = {username: UserRecord(username=username, salt_b64=salt, hash_b64=hsh,
                                  iterations=iters)}
    save_users(server_root / "users.json", users)


def _worker_written_shard(tmp_path: Path, *, games: int, w: int, d: int, l: int) -> Path:
    """A raw shard produced by the REAL worker producer, not hand-written meta."""
    buf = _BufferedUpload()
    batch = SimpleNamespace(
        samples=[_sample(i) for i in range(4)],
        positions=4,
        games=games,
        w=w, d=d, l=l,
        total_game_plies=120,
        input_history_encoding="lc0_root",
        history_rep_fix=True,
        sf_d6_sum=2.5,
        sf_d6_n=4,
    )
    _buffer_add_completed_game(
        buf=buf,
        game_batch=batch,
        now_s=1000.0,
        model_sha=MODEL_SHA,
        model_step=42,
        opponent_wdl_regret_limit=LIVE_REGRET_LIMIT,
        sf_nodes=LIVE_SF_NODES,
    )
    pending = tmp_path / "worker_pending"
    pending.mkdir(parents=True, exist_ok=True)
    shard, _elapsed = _flush_upload_buffer_to_pending(
        pending_dir=pending,
        username="u",
        buf=buf,
        now_s=1000.0,
        trial_id=TRIAL_ID,
    )
    assert shard is not None
    _arrs, raw_meta = load_shard_arrays(shard)
    # Precondition, not the assertion under test: the producer stamps the pair.
    assert raw_meta["opponent_wdl_regret_limit"] == pytest.approx(LIVE_REGRET_LIMIT)
    assert int(raw_meta["sf_nodes"]) == LIVE_SF_NODES
    return shard


def test_worker_upload_reaches_trainer_ingest_with_difficulty_intact(tmp_path: Path) -> None:
    """The whole chain: worker producer -> HTTP upload -> compactor -> ingest.

    MUTATION: drop ``opponent_wdl_regret_limit`` from the compactor again. The
    gate arm's regret denominator returns to 0 and ``_arm_mean_regret`` to NaN,
    which is precisely the live state this fix ends.
    """
    from fastapi.testclient import TestClient

    from chess_anti_engine.server.app import create_app

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    shard = _worker_written_shard(tmp_path, games=6, w=3, d=2, l=1)
    _name, tar = pack_shard_for_upload(shard)

    # Threshold below the shard's 4 positions so the upload compacts immediately.
    client = TestClient(create_app(server_root=str(server_root), users_db="users.json",
                                   upload_compact_shard_size=1))
    r = client.post(
        f"/v1/trials/{TRIAL_ID}/upload_shard",
        auth=("u", "p"),
        files={"file": (f"shard{LOCAL_SHARD_SUFFIX}.tar", tar.getvalue(), "application/x-tar")},
        headers={"X-CAE-Worker-Version": "0.0.0", "X-CAE-Protocol-Version": "1"},
    )
    assert r.status_code == 200, r.text
    assert r.json().get("stored") is True, r.text

    inbox = server_root / "trials" / TRIAL_ID / "inbox"
    compacted = sorted((inbox / "_compacted").glob(f"*{LOCAL_SHARD_SUFFIX}"))
    assert len(compacted) == 1, f"expected one compacted shard, found {compacted}"

    _arrs, meta = load_shard_arrays(compacted[0])
    regret = meta.get("opponent_wdl_regret_limit")
    nodes = meta.get("sf_nodes")
    assert regret is not None
    assert math.isfinite(float(regret))
    assert nodes is not None
    assert math.isfinite(float(nodes))
    assert float(regret) == pytest.approx(LIVE_REGRET_LIMIT)
    assert int(nodes) == LIVE_SF_NODES
    # The other two fields the compactor used to drop, which made
    # ``sf_eval_delta6`` read 0.0 on every live row.
    assert float(meta["sf_d6_sum"]) == pytest.approx(2.5)
    assert int(meta["sf_d6_n"]) == 4

    # ...and the trainer's own ingest turns them into a finite arm mean.
    summary = _ingest_distributed_selfplay(
        buf=DiskReplayBuffer(
            256, shard_dir=tmp_path / "replay", rng=np.random.default_rng(0),
            shuffle_cap=64, shard_size=8, read_only=False,
        ),
        holdout_buf=ArrayReplayBuffer(32, rng=np.random.default_rng(1)),
        holdout_frac=0.0, holdout_frozen=False,
        inbox_dir=inbox, processed_dir=tmp_path / "processed",
        target_games=1,
        accepted_model_shas={MODEL_SHA},
        prev_model_sha=None,
        prev_model_max_fraction=1.0,
        wait_timeout_s=0.5, poll_seconds=0.01,
        rng=np.random.default_rng(2), on_poll=None, min_games_fraction=0.0,
    )
    assert summary["gate_cur_regret_games"] == 6
    assert summary["gate_cur_regret_weighted"] == pytest.approx(LIVE_REGRET_LIMIT * 6)
    arm = _arm_mean_regret(summary, "cur")
    assert math.isfinite(arm), "the gate's confound instrument is still NaN"
    assert arm == pytest.approx(LIVE_REGRET_LIMIT)
    assert float(summary["matching_sf_d6_sum"]) == pytest.approx(2.5)
    assert int(summary["matching_sf_d6_n"]) == 4
