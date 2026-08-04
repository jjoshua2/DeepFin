"""Every config key the DRIVER reads must carry a classification.

Audit angle C (tranche 12), findings C2-C5. The three sets in
`trainable_config_ops` that answer "will a live yaml edit take effect" are all
derived by running the trial-actor reloader, so they structurally cannot see a
key whose only consumer lives in the driver process: `run.py` builds one
`base_config`, `harness.run_tune` reads it once to construct the Tuner and to
spawn uvicorn and the inference broker, and the trial actor never sees those
reads. The default answer for "not in `restart_required_config_keys()`" is
"live-reloadable", and for those keys it is wrong in a way that tells an
operator their edit is in effect when it cannot be.

⚑ THE POINT OF THIS MODULE IS THE RE-DERIVATION, NOT THE LIST. It walks the
driver's AST for `base_config.get("...")` and friends and demands a
classification for every literal it finds, so a key added to `harness.py`
without one FAILS here. Restating the membership would only assert that two
hand-written lists match. Same self-invalidating shape as
`tests/test_reco_coverage.py`, which is what has kept the worker's published
key set honest.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from chess_anti_engine.tune.trainable_config_ops import (
    construction_only_config_keys,
    driver_derived_config_keys,
    driver_dual_clock_config_keys,
    driver_launch_fixed_config_keys,
    restart_required_config_keys,
)

_ROOT = Path(__file__).resolve().parents[1] / "chess_anti_engine"
_HARNESS = _ROOT / "tune" / "harness.py"
_RUN = _ROOT / "run.py"

# The dict names that hold the driver's copy of the flattened yaml. `cfg` is
# `_prepare_distributed_worker_auth`'s parameter name for the same dict; it is
# in here because that function is where the three first-provisioning-only auth
# keys are read, which is finding C3.
_DRIVER_CONFIG_NAMES = frozenset({"base_config", "base", "cfg"})


def _config_keys_read_by(path: Path) -> dict[str, int]:
    """Literal keys read from a driver config dict, key -> first line number."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, int] = {}

    def _record(name: str, lineno: int) -> None:
        found.setdefault(name, lineno)

    for node in ast.walk(tree):
        # base_config.get("key"[, default])
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in _DRIVER_CONFIG_NAMES
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            _record(node.args[0].value, node.lineno)
        # base_config["key"], READ only. A Store is the driver computing a
        # derived value into its own dict (run.py:44-56) -- that is not a yaml
        # key being consumed, and demanding a classification for it would make
        # this test fire on code that has no config-provenance question at all.
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id in _DRIVER_CONFIG_NAMES
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            _record(node.slice.value, node.lineno)
    return found


def _classification_of(key: str) -> str | None:
    for label, keys in (
        ("driver-launch-fixed", driver_launch_fixed_config_keys()),
        ("driver-dual-clock", driver_dual_clock_config_keys()),
        ("driver-derived", driver_derived_config_keys()),
        ("restart-required", restart_required_config_keys()),
        ("construction-only", construction_only_config_keys()),
    ):
        if key in keys:
            return label
    return None


@pytest.mark.parametrize("path", [_HARNESS, _RUN], ids=["harness", "run"])
def test_every_driver_config_read_is_classified(path: Path) -> None:
    """A key the driver reads and nobody classified is the C2 defect returning.

    The failure message names the file and line so the fix is "classify it",
    not "go find it".
    """
    unclassified = [
        f"{path.name}:{lineno} {key}"
        for key, lineno in sorted(_config_keys_read_by(path).items())
        if key.startswith("_") is False and _classification_of(key) is None
    ]
    assert not unclassified, (
        "these driver-side config keys are in no classification set, so a "
        "provenance report will call them live-reloadable when the driver reads "
        "them once at launch:\n  " + "\n  ".join(unclassified) + "\n"
        "Add each to driver_launch_fixed_config_keys(), driver_dual_clock_"
        "config_keys() or driver_derived_config_keys() in trainable_config_ops.py."
    )


def test_the_walk_actually_finds_the_driver_reads() -> None:
    """NEGATIVE CONTROL for the instrument above.

    A walk that silently matched nothing would make the coverage test pass by
    finding no keys at all -- the "gate that cannot fail" shape. Pin a floor and
    three keys that must be in the result, one per read style this file cares
    about.
    """
    harness_keys = _config_keys_read_by(_HARNESS)
    assert len(harness_keys) >= 30, f"only found {len(harness_keys)} keys in harness.py"
    for key in ("distributed_server_port", "cpus_per_trial", "distributed_worker_password"):
        assert key in harness_keys, f"{key} not found by the AST walk"
    assert "distributed_workers_per_trial" in _config_keys_read_by(_RUN)


def test_an_unclassified_driver_key_fails_the_check() -> None:
    """And the gate can fail: an invented key is not classified."""
    assert _classification_of("a_key_nobody_declared") is None
    assert _classification_of("distributed_server_port") == "driver-launch-fixed"


def test_the_classification_sets_do_not_overlap() -> None:
    """A key in two sets has two answers, which is no answer.

    `restart_required_config_keys()` is deliberately excluded from the
    driver-side sets: its contract is "the live reloader refuses to apply this
    to a running trial", and these keys never reach the reloader at all.
    """
    launch = driver_launch_fixed_config_keys()
    dual = driver_dual_clock_config_keys()
    derived = driver_derived_config_keys()
    assert not (launch & dual), sorted(launch & dual)
    assert not (launch & derived), sorted(launch & derived)
    assert not (dual & derived), sorted(dual & derived)
    assert not (launch & restart_required_config_keys()), sorted(
        launch & restart_required_config_keys()
    )


# ---------------------------------------------------------------------------
# C4/C5 — the two-clock keys, pinned on BOTH clocks.
# ---------------------------------------------------------------------------


def _reads_key(path: Path, key: str) -> list[int]:
    return [
        lineno
        for k, lineno in _config_keys_read_by(path).items()
        if k == key
    ]


def test_tune_num_to_keep_is_read_on_both_clocks() -> None:
    """C4. Ray's checkpoint retention is launch-fixed; the trial's pruner is live.

    A live edit moves `_prune_trial_checkpoints` and leaves `RunConfig` on the
    launch value, so the two silently disagree about how many checkpoints to
    keep. Pinned on both sides so that unifying them later is a deliberate
    change with a failing test, not a silent one.
    """
    assert _reads_key(_HARNESS, "tune_num_to_keep"), "driver-side read vanished"
    phases = (_ROOT / "tune" / "trainable_phases.py").read_text(encoding="utf-8")
    assert "keep_last=tc.tune_num_to_keep" in phases, "trial-side read vanished"
    assert "tune_num_to_keep" in driver_dual_clock_config_keys()


def test_shard_size_is_read_on_both_clocks() -> None:
    """C5, and it crosses a process boundary.

    `distributed_upload_compact_shard_size` is unset in production, so
    `shard_size` is the server's compaction target AND the trainer's replay
    writer size. Editing it live re-aims the writer while the already-spawned
    uvicorn keeps compacting at the launch value.
    """
    harness = _HARNESS.read_text(encoding="utf-8")
    assert 'base_config.get("shard_size", 2000)' in harness, "driver-side read vanished"
    init = (_ROOT / "tune" / "trainable_init.py").read_text(encoding="utf-8")
    assert "shard_size=tc.shard_size" in init, "trial-side read vanished"
    assert "shard_size" in driver_dual_clock_config_keys()


def test_the_auth_keys_are_first_provisioning_only() -> None:
    """C3. Both writes are gated on `not user_existed`, so against an existing
    server_root the yaml value reaches no consumer at any value.

    Not `dead_config_keys()`: it IS consumed, exactly once, on first
    provisioning — and that set makes the next restart REFUSE to start, which
    would turn a documented no-op into a crash.
    """
    harness = _HARNESS.read_text(encoding="utf-8")
    assert harness.count("if not user_existed") >= 1
    assert "if not user_existed and not password_file.exists():" in harness
    for key in (
        "distributed_worker_username",
        "distributed_worker_password",
        "distributed_worker_password_env",
    ):
        assert key in driver_launch_fixed_config_keys()


# ---------------------------------------------------------------------------
# C1 — the one-way keys, and the observation that they were declined.
# ---------------------------------------------------------------------------


def test_a_declined_off_is_audible_once_per_transition(caplog) -> None:
    """C1. `true -> false` on a one-way key must produce exactly one warning.

    ⚑ THE COUNT IS THE ASSERTION, not merely that a line appears. A warning
    emitted every iteration is one an operator learns to scroll past, and a
    warning emitted never is the state this fixes. Three consecutive
    still-false iterations must produce one line.
    """
    import logging

    from chess_anti_engine.tune import trainable as trainable_mod

    trainable_mod._DECLINED_OFF_WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.iter"):
        for iteration in range(3):
            trainable_mod._warn_declined_off(
                "distributed_prefetch_shards", False, object(), iteration
            )
    lines = [r for r in caplog.records if "distributed_prefetch_shards" in r.getMessage()]
    assert len(lines) == 1, [r.getMessage() for r in lines]
    assert "ACCEPTED AND IGNORED" in lines[0].getMessage()


def test_a_healthy_key_is_silent_and_a_flip_flop_is_reported(caplog) -> None:
    """The negative control: silence when the key is on, or when the object
    does not exist yet, and a SECOND warning after the key goes back on."""
    import logging

    from chess_anti_engine.tune import trainable as trainable_mod

    trainable_mod._DECLINED_OFF_WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.iter"):
        trainable_mod._warn_declined_off("distributed_async_test_eval", True, object(), 0)
        trainable_mod._warn_declined_off("distributed_async_test_eval", False, None, 1)
    assert not [r for r in caplog.records if "distributed_async_test_eval" in r.getMessage()]

    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.iter"):
        trainable_mod._warn_declined_off("distributed_async_test_eval", False, object(), 2)
        trainable_mod._warn_declined_off("distributed_async_test_eval", True, object(), 3)
        trainable_mod._warn_declined_off("distributed_async_test_eval", False, object(), 4)
    assert (
        len([r for r in caplog.records if "distributed_async_test_eval" in r.getMessage()]) == 2
    ), "a key toggled off, on, off must warn both times"


def test_the_lazy_helper_calls_the_declined_off_warning() -> None:
    """Pins the WIRING, not just the helper.

    A correct `_warn_declined_off` that nothing calls is this codebase's
    signature defect wearing a fix's clothes, and the helper's own tests
    cannot see it.
    """
    src = (_ROOT / "tune" / "trainable.py").read_text(encoding="utf-8")
    body = src[src.index("def _lazy_construct_iter_helpers(") : src.index("def _log_iter_phase_split(")]
    assert body.count("_warn_declined_off(") == 2, body.count("_warn_declined_off(")
    for key in ("distributed_prefetch_shards", "distributed_async_test_eval"):
        assert f'"{key}", tc.{key}' in body, f"{key} not passed to the warning"
