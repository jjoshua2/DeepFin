"""Resume cursor contracts that a checkpoint reinstall must not move.

Two production-reachable defects, both on the path that opens the replay
buffer after a Ray checkpoint or salvage load:

* ``DiskReplayBuffer`` seeds its prefetch generator with one
  ``rng.integers`` draw. A fresh trial draws that once before any
  checkpoint, so the sidecar already excludes it. Resume installs the
  sidecar first; drawing again shifts every later mirror decision and
  sample index. ``preserve_sampling_rng`` rolls the parent back, and the
  learner sets it only when a checkpointed generator was actually installed.

* ``__init__`` enforces the window before the caller can raise capacity.
  A resume whose ``trial_meta.json`` has no ``current_window`` used to
  construct at ``replay_window_start`` and delete the durable window.
  ``resume_open_capacity`` opens that case at the configured max.

The learner wiring lives in ``tune/trainable_init.py``, which imports the
native encoder. These tests pin the buffer behaviour by execution and the
call site by AST, so they run without that extension.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from chess_anti_engine.replay.disk_buffer import (
    DiskReplayBuffer,
    resume_open_capacity,
)
from chess_anti_engine.replay.shard import (
    iter_shard_paths,
    local_shard_path,
    save_local_shard_arrays,
)

_REPO = Path(__file__).resolve().parents[1]
_ROWS = 4


def _open(tmp: Path, rng: np.random.Generator, *, preserve: bool) -> DiskReplayBuffer:
    return DiskReplayBuffer(
        10,
        shard_dir=tmp,
        rng=rng,
        read_only=False,
        refresh_interval=0,
        refresh_shards=0,
        preserve_sampling_rng=preserve,
    )


def test_preserve_sampling_rng_keeps_the_parent_and_the_prefetch_seed(tmp_path: Path) -> None:
    """The resumed generator's next draw is the one the sidecar promised.

    The prefetch generator is still seeded from the integer that draw
    would have been, so the shuffle-pool seed does not move.
    """
    parent = np.random.default_rng(7)
    seed_source = np.random.default_rng(7)
    expected_seed = int(seed_source.integers(0, 2**32 - 1))
    untouched = np.random.default_rng(7)
    expected_prefetch = int(np.random.default_rng(expected_seed).integers(0, 2**32 - 1))

    buf = _open(tmp_path / "replay", parent, preserve=True)
    try:
        assert parent.bit_generator.state == untouched.bit_generator.state
        assert int(buf._prefetch_rng.integers(0, 2**32 - 1)) == expected_prefetch
        assert parent.bit_generator.state == untouched.bit_generator.state
    finally:
        buf.close()


def test_default_construction_still_consumes_one_integers_draw(tmp_path: Path) -> None:
    """Fresh starts and cross-trial forks leave the flag false.

    Their stream has no checkpoint behind it, so the one prefetch-seed
    draw stays part of construction. A regression that preserved the
    parent unconditionally would move every fresh trial's samples.
    """
    parent = np.random.default_rng(7)
    advanced = np.random.default_rng(7)
    drawn = int(advanced.integers(0, 2**32 - 1))
    expected_prefetch = int(np.random.default_rng(drawn).integers(0, 2**32 - 1))

    buf = _open(tmp_path / "replay", parent, preserve=False)
    try:
        assert parent.bit_generator.state == advanced.bit_generator.state
        assert int(buf._prefetch_rng.integers(0, 2**32 - 1)) == expected_prefetch
    finally:
        buf.close()


def test_resume_open_capacity_preserves_saved_window_and_caps_missing_window() -> None:
    """A saved window and a fresh start construct at the caller's window."""
    assert resume_open_capacity(
        current_window=400_000,
        replay_window_max=1_500_000,
        restored_window=900_000,
        durable_resume=True,
    ) == 400_000
    assert resume_open_capacity(
        current_window=400_000,
        replay_window_max=1_500_000,
        restored_window=0,
        durable_resume=False,
    ) == 400_000
    assert resume_open_capacity(
        current_window=400_000,
        replay_window_max=1_500_000,
        restored_window=0,
        durable_resume=True,
    ) == 1_500_000
    # A transient exploit pre-bump must not exceed the configured ceiling.
    assert resume_open_capacity(
        current_window=2_000_000,
        replay_window_max=1_500_000,
        restored_window=0,
        durable_resume=True,
    ) == 1_500_000


def _write_two_shards(directory: Path) -> None:
    policy = np.zeros((_ROWS, 4672), dtype=np.float32)
    policy[:, 0] = 1.0
    arrays = {
        "x": np.zeros((_ROWS, 146, 8, 8), dtype=np.float32),
        "policy_target": policy,
        "wdl_target": np.zeros((_ROWS,), dtype=np.int8),
        "priority": np.ones((_ROWS,), dtype=np.float32),
        "has_policy": np.ones((_ROWS,), dtype=np.uint8),
    }
    for index in range(2):
        save_local_shard_arrays(local_shard_path(directory, index), arrs=arrays)


def test_missing_window_open_capacity_keeps_shards_a_start_cap_deletes(
    tmp_path: Path,
) -> None:
    """The constructor still deletes, and the resume capacity does not ask it to.

    Two synthetic shards, eight positions. Opening at ``replay_window_start``
    (four positions) unlinks one shard. Opening at the capacity a missing
    ``current_window`` resume now requests keeps both.
    """
    kept = tmp_path / "kept"
    trimmed = tmp_path / "trimmed"
    _write_two_shards(kept)
    _write_two_shards(trimmed)
    open_at = resume_open_capacity(
        current_window=_ROWS,
        replay_window_max=100,
        restored_window=0,
        durable_resume=True,
    )
    assert open_at == 100

    kept_buf = DiskReplayBuffer(
        open_at,
        shard_dir=kept,
        rng=np.random.default_rng(0),
        read_only=False,
        refresh_interval=0,
        refresh_shards=0,
        shard_size=_ROWS,
    )
    trimmed_buf = DiskReplayBuffer(
        _ROWS,
        shard_dir=trimmed,
        rng=np.random.default_rng(0),
        read_only=False,
        refresh_interval=0,
        refresh_shards=0,
        shard_size=_ROWS,
    )
    try:
        assert len(iter_shard_paths(kept)) == 2
        assert len(kept_buf) == 2 * _ROWS
        assert len(iter_shard_paths(trimmed)) == 1
        assert len(trimmed_buf) == _ROWS
    finally:
        kept_buf.close()
        trimmed_buf.close()


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _assigned_true(fn: ast.FunctionDef, attribute: str) -> list[ast.Assign]:
    return [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == attribute
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and node.value.value is True
    ]


def test_learner_wires_the_flag_only_after_a_successful_rng_install() -> None:
    """The buffer default is false. The learner has to opt in, and only then."""
    tree = ast.parse(
        (_REPO / "chess_anti_engine/tune/trainable_init.py").read_text(encoding="utf-8"),
    )
    restore = _function(tree, "_restore_checkpoint_or_salvage")
    installs = _assigned_true(restore, "sampling_rng_restored")
    assert len(installs) == 1
    install = installs[0]
    try_node = next(
        node for node in ast.walk(restore)
        if isinstance(node, ast.Try) and install in node.body
    )
    state_assigns = [
        node for node in try_node.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "state"
            for target in node.targets
        )
    ]
    assert len(state_assigns) == 1
    assert try_node.body.index(state_assigns[0]) < try_node.body.index(install)
    guard = next(
        node for node in ast.walk(restore)
        if isinstance(node, ast.If) and try_node in node.body
    )
    guard_src = ast.dump(guard.test)
    assert "cross_trial_restore" in guard_src
    assert "restored_rng_state" in guard_src

    init = _function(tree, "_init_replay_buffers")
    buffer_calls = [
        node for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "DiskReplayBuffer")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "DiskReplayBuffer")
        )
    ]
    assert len(buffer_calls) == 1
    call = buffer_calls[0]
    assert isinstance(call.args[0], ast.Name)
    assert call.args[0].id == "open_capacity"
    flag = next(kw for kw in call.keywords if kw.arg == "preserve_sampling_rng")
    assert isinstance(flag.value, ast.Attribute)
    assert flag.value.attr == "sampling_rng_restored"
    assert isinstance(flag.value.value, ast.Name)
    assert flag.value.value.id == "restore"

    openings = [
        node for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "resume_open_capacity")
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "resume_open_capacity"
            )
        )
    ]
    assert len(openings) == 1
    keywords = {kw.arg: kw.value for kw in openings[0].keywords}
    assert set(keywords) >= {
        "current_window", "replay_window_max", "restored_window", "durable_resume",
    }
    durable = ast.dump(keywords["durable_resume"])
    assert "ckpt" in durable
    assert "seed_warmstart_used" in durable
    assert not (
        isinstance(keywords["durable_resume"], ast.Constant)
        and keywords["durable_resume"].value is False
    )


def test_sampling_rng_restored_defaults_false_at_the_end_of_restore_result() -> None:
    """Existing ``RestoreResult(...)`` constructors must keep their meaning."""
    tree = ast.parse(
        (_REPO / "chess_anti_engine/tune/trial_config.py").read_text(encoding="utf-8"),
    )
    cls = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "RestoreResult"
    )
    fields = [
        node.target.id
        for node in cls.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]
    assert fields[-1] == "sampling_rng_restored"
    assign = next(
        node for node in cls.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "sampling_rng_restored"
    )
    assert isinstance(assign.value, ast.Constant)
    assert assign.value.value is False
