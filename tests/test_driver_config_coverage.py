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

# The dict names that hold the driver's copy of the flattened yaml, and -- for
# `cfg` -- the ONE function that name is allowed to mean it in.
#
# ⚑ `cfg` IS SCOPED. It is `_prepare_distributed_worker_auth`'s parameter name
# for the driver config (that function is where the auth keys of finding C3 are
# read), but `cfg` is also a perfectly ordinary local elsewhere in these files
# -- `harness.py`'s resume overlay uses `cfg[key]` for a TRIAL dict. Matching it
# unscoped made this test demand a driver classification for trial keys, which
# is a false positive that trains people to add keys to the wrong set.
_DRIVER_CONFIG_NAMES = frozenset({"base_config", "base"})
_SCOPED_CONFIG_NAMES: dict[str, frozenset[str]] = {
    "cfg": frozenset({"_prepare_distributed_worker_auth"}),
}

# Reads through a non-literal key, by `file:line`, each with the reason it can
# stay dynamic. ⚑ THE WALK CANNOT RESOLVE THESE, so without this allowlist a new
# `base_config.get(some_var)` would be silently invisible to the coverage test
# -- the reviewer demonstrated exactly that with `_k = "unclassified"`. Listing
# them by line makes a new dynamic read FAIL until someone accounts for it.
_NON_LITERAL_READS: dict[str, str] = {
    "harness.py:493": (
        "the opening-book loop: `for cfg_key, flag in (...)` over "
        "opening_book_path / opening_book_path_2, both classified "
        "driver-launch-fixed"
    ),
}


def _config_keys_read_by(path: Path) -> dict[str, int]:
    """Literal keys read from a driver config dict, key -> first line number."""
    return _walk_driver_reads(path)[0]


def _enclosing_functions(tree: ast.AST) -> dict[int, str]:
    """Map every line inside a function body to that function's name."""
    owner: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno) or node.lineno
            for line in range(node.lineno, end + 1):
                owner[line] = node.name
    return owner


def _walk_driver_reads(path: Path) -> tuple[dict[str, int], dict[str, str]]:
    """``(literal key -> first line, "file:line" -> source text)`` for reads of a
    driver config dict. The second element is the NON-LITERAL reads."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = _enclosing_functions(tree)
    found: dict[str, int] = {}
    dynamic: dict[str, str] = {}

    def _record(name: str, lineno: int) -> None:
        found.setdefault(name, lineno)

    def _is_driver_dict(node: ast.expr, lineno: int) -> bool:
        if not isinstance(node, ast.Name):
            return False
        if node.id in _DRIVER_CONFIG_NAMES:
            return True
        allowed = _SCOPED_CONFIG_NAMES.get(node.id)
        return allowed is not None and owner.get(lineno, "") in allowed

    for node in ast.walk(tree):
        # base_config.get("key"[, default])
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and _is_driver_dict(node.func.value, node.lineno)
            and node.args
        ):
            if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                _record(node.args[0].value, node.lineno)
            else:
                dynamic[f"{path.name}:{node.lineno}"] = ast.unparse(node)
        # base_config["key"], READ only. A Store is the driver computing a
        # derived value into its own dict (run.py:44-56) -- that is not a yaml
        # key being consumed, and demanding a classification for it would make
        # this test fire on code that has no config-provenance question at all.
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Load)
            and _is_driver_dict(node.value, node.lineno)
        ):
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                _record(node.slice.value, node.lineno)
            else:
                dynamic[f"{path.name}:{node.lineno}"] = ast.unparse(node)
    return found, dynamic


def _classifications_of(key: str) -> list[str]:
    """EVERY set the key is in, never the first match.

    First-match resolution is how round 1 shipped `iterations` in three sets and
    reported one of them: a masking lookup makes the overlap test the only thing
    standing between a double-booked key and a confident wrong answer, and that
    test was not pairwise-complete either.
    """
    return [
        label
        for label, keys in (
            ("driver-launch-fixed", driver_launch_fixed_config_keys()),
            ("driver-dual-clock", driver_dual_clock_config_keys()),
            ("driver-derived", driver_derived_config_keys()),
            ("restart-required", restart_required_config_keys()),
            ("construction-only", construction_only_config_keys()),
        )
        if key in keys
    ]


@pytest.mark.parametrize("path", [_HARNESS, _RUN], ids=["harness", "run"])
def test_every_driver_config_read_is_classified(path: Path) -> None:
    """A key the driver reads and nobody classified is the C2 defect returning.

    The failure message names the file and line so the fix is "classify it",
    not "go find it".

    ⚑ THE TWO LEGS ARE NOT EQUAL COVERAGE, and "walks harness.py/run.py" reads
    as more than the `run` leg provides. `harness.py` contributes ~40 literal
    reads; `run.py` contributes exactly ONE (`distributed_workers_per_trial`),
    because its other config touches are `Store`s -- the driver computing
    derived values into its own dict -- which this walk deliberately ignores.
    The `run` leg can genuinely fail (verified by injecting an unclassified
    literal read into it), but it is a denominator of one, so treat it as a
    tripwire on that file rather than as coverage of it.
    """
    unclassified = [
        f"{path.name}:{lineno} {key}"
        for key, lineno in sorted(_config_keys_read_by(path).items())
        if key.startswith("_") is False and not _classifications_of(key)
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
    for key in ("distributed_server_port", "cpus_per_trial", "distributed_worker_username"):
  # `distributed_worker_password` was the third probe until the credential
  # moved to $CAE_WORKER_PASSWORD. It is deliberately no longer read from the
  # config AT ALL, so probing for it would pin the defect back into place.
  # `distributed_worker_username` is the right replacement: same call site,
  # same function, and still a genuine config read.
        assert key in harness_keys, f"{key} not found by the AST walk"
    assert "distributed_workers_per_trial" in _config_keys_read_by(_RUN)


def test_an_unclassified_driver_key_fails_the_check() -> None:
    """And the gate can fail: an invented key is not classified."""
    assert _classifications_of("a_key_nobody_declared") == []
    assert _classifications_of("distributed_server_port") == ["driver-launch-fixed"]


def test_every_non_literal_driver_read_is_accounted_for() -> None:
    """The walk cannot resolve `base_config.get(some_var)`, so it must COUNT them.

    ⚑ WITHOUT THIS THE COVERAGE TEST HAS A HOLE IT CANNOT SEE: a dynamic read is
    silently skipped, so `_k = "unclassified"; base_config.get(_k)` passes. The
    reviewer demonstrated it. Accounting by `file:line` means a NEW dynamic read
    fails here until someone writes down why it may stay dynamic — which is the
    same self-invalidating contract as the literal half.

    The one entry today is real and is a genuine C4/C5 key: `harness.py:491`
    reads the two opening-book paths through a loop variable and bakes them into
    the uvicorn command line, so `/v1/opening_book` serves the launch file for
    the life of the server.
    """
    seen: dict[str, str] = {}
    for path in (_HARNESS, _RUN):
        seen.update(_walk_driver_reads(path)[1])
    unaccounted = set(seen) - set(_NON_LITERAL_READS)
    vanished = set(_NON_LITERAL_READS) - set(seen)
    assert not unaccounted | vanished, (
        f"non-literal driver config reads changed.\n"
        f"  new/unaccounted: {sorted(unaccounted)}\n"
        f"  gone (update the allowlist): {sorted(vanished)}\n"
        f"A dynamic read is INVISIBLE to test_every_driver_config_read_is_"
        f"classified, so each one must be named here with the reason it stays."
    )


def test_the_opening_book_paths_are_classified_on_the_driver_axis() -> None:
    """The dynamic read of H4 points at two keys, and they must land somewhere.

    Launch-fixed on the DRIVER axis and restart-required on the TRIAL axis is
    not a contradiction — two independent reasons the same edit does nothing.
    """
    for key in ("opening_book_path", "opening_book_path_2"):
        assert key in driver_launch_fixed_config_keys()
        assert key in restart_required_config_keys()


def test_the_classification_sets_do_not_overlap() -> None:
    """A key in two sets has two answers, which is no answer.

    ⚑ PAIRWISE-COMPLETE OVER ALL FIVE SETS, with the two legitimate overlaps
    named as exemptions rather than left out of the loop. Round 1 checked only
    `launch & restart_required` and therefore could not see `iterations` sitting
    in dual-clock AND restart-required AND construction-only.

    The two exemptions, and why each is not a contradiction:

    * `construction_only ⊆ restart_required` — asserted as a subset, because
      that is what `construction_only_config_keys()`'s own docstring promises.
    * `driver_launch_fixed × {restart_required, construction_only}` — orthogonal
      AXES, not competing answers: the driver reads the key once at launch, and
      the trial separately refuses a live edit. The opening-book paths are
      exactly this.

    `driver_dual_clock × restart_required` gets NO exemption: dual-clock asserts
    the trial-side consumer moves on a live edit and restart-required asserts it
    does not. That pair is the `iterations` bug, so it must stay red.
    """
    sets = {
        "driver_launch_fixed": driver_launch_fixed_config_keys(),
        "driver_dual_clock": driver_dual_clock_config_keys(),
        "driver_derived": driver_derived_config_keys(),
        "restart_required": restart_required_config_keys(),
        "construction_only": construction_only_config_keys(),
    }
    exempt = {
        ("construction_only", "restart_required"),
        ("driver_launch_fixed", "restart_required"),
    }
  # ⚑ AN EXEMPTION MUST BE EARNING ITS KEEP. `construction_only x
  # driver_launch_fixed` was in this set and its intersection is EMPTY, so it
  # excused nothing while reading as a documented, considered overlap -- and it
  # would have gone on silently excusing the pair the day one did appear. A
  # vacuous exemption is a hole with a comment in front of it, so every pair
  # named here has to actually overlap today.
    for a, b in sorted(exempt):
        assert sets[a] & sets[b], (
            f"the ({a}, {b}) exemption is VACUOUS -- those sets do not "
            f"intersect, so it excuses nothing and would silently excuse a "
            f"future overlap nobody re-justified. Delete it."
        )
    names = sorted(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if (a, b) in exempt:
                continue
            assert not (sets[a] & sets[b]), (
                f"{a} & {b} = {sorted(sets[a] & sets[b])} — a key in both has "
                f"two answers. If the overlap is legitimate, add the pair to "
                f"`exempt` WITH the reason in this docstring."
            )
    assert sets["construction_only"] <= sets["restart_required"], sorted(
        sets["construction_only"] - sets["restart_required"]
    )


def test_iterations_is_restart_required_not_dual_clock() -> None:
    """Pins the specific wrong answer round 1 shipped.

    Dual-clock means "a live edit moves the trial-side consumer". The reloader
    REFUSES `iterations`, so neither clock moves and the sets cannot disagree —
    `restart_required` is the whole story.
    """
    assert "iterations" not in driver_dual_clock_config_keys()
    assert "iterations" in restart_required_config_keys()


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
    for key in ("distributed_worker_password", "distributed_worker_password_env"):
        assert key in driver_launch_fixed_config_keys()


def test_the_username_is_not_inert_and_is_not_classified_as_inert() -> None:
    """H1. Round 1 called `distributed_worker_username` first-provisioning-only
    and wrote "changes nothing, forever" into the production yaml. False, and
    expensively so: the username is in `_WORKER_LAUNCH_CONFIG_KEYS`, which
    `_ensure_distributed_workers` hashes every iteration, so a live edit
    RELAUNCHES THE WHOLE FLEET — with a username the server never provisioned,
    because `_prepare_distributed_worker_auth` only ever provisions on the first
    run against a server root. Live edit = fleet-wide 401.

    Pinned from the worker-launch tuple itself, so this goes red if someone
    removes the username from it and makes the round-1 story true again.
    """
    from chess_anti_engine.tune import distributed_runtime

    assert "distributed_worker_username" in distributed_runtime._WORKER_LAUNCH_CONFIG_KEYS
    assert "distributed_worker_username" in driver_dual_clock_config_keys()
    assert "distributed_worker_username" not in driver_launch_fixed_config_keys()
    # ...and the yaml must not carry the round-1 claim next to it.
  # Sliced AROUND the username line rather than between it and the password
  # line: the password line is gone, so the old end-marker no longer exists and
  # `.index` raised instead of asserting.
    yaml_text = (_ROOT.parent / "configs" / "pbt2_small.yaml").read_text(encoding="utf-8")
    at = yaml_text.index("distributed_worker_username")
    block = yaml_text[max(0, at - 1400): at + 400]
    assert "changes nothing, forever" not in block


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


def test_the_declined_off_warning_fires_through_the_real_helper() -> None:
    """L2. Drives `_lazy_construct_iter_helpers` end to end.

    ⚑ THE WIRING TEST ABOVE COUNTS SOURCE STRINGS, so it cannot see reachability:
    a call MOVED INSIDE the `if ... is None:` construction branch would keep the
    count at 2, keep every other test green, and never fire in the declined case
    — which is the only case that matters. This drives the real helper with a
    real `TrialConfig` instead.
    """
    import logging

    from chess_anti_engine.tune import trainable as trainable_mod
    from chess_anti_engine.tune.trial_config import TrialConfig

    off = TrialConfig.from_dict(
        {"distributed_prefetch_shards": False, "distributed_async_test_eval": False}
    )
    running_prefetcher, running_eval = object(), object()

    trainable_mod._DECLINED_OFF_WARNED.clear()
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # pyright: ignore[reportAttributeAccessIssue]
    log = logging.getLogger("chess_anti_engine.iter")
    log.addHandler(handler)
    try:
        out: tuple[object, object] = (running_prefetcher, running_eval)
        for iteration in range(3):
            out = trainable_mod._lazy_construct_iter_helpers(
                shard_prefetcher=out[0],
                async_test_eval=out[1],
                tc=off,
                distributed_dirs={},
                iteration_idx=iteration,
            )
        # The edit really is ignored: the same objects come back out.
        assert out == (running_prefetcher, running_eval)
        warned = [r for r in records if r.levelno >= logging.WARNING]
        assert len(warned) == 2, [r.getMessage() for r in warned]

        # Negative control on the same path: nothing constructed yet -> silent.
        records.clear()
        trainable_mod._DECLINED_OFF_WARNED.clear()
        for iteration in range(3):
            still_none = trainable_mod._lazy_construct_iter_helpers(
                shard_prefetcher=None, async_test_eval=None, tc=off,
                distributed_dirs={}, iteration_idx=iteration,
            )
            assert still_none == (None, None)
        assert not [r for r in records if r.levelno >= logging.WARNING]
    finally:
        log.removeHandler(handler)


def test_a_driver_key_reports_launch_fixed_not_wait_for_the_reload() -> None:
    """H5. The sets must change what an OPERATOR sees, not just what a test does.

    Before wiring, `classify_config_provenance` reported an edited
    `distributed_server_port` as `RELOAD-NOT-APPLIED-UNRESOLVED ... re-run after
    the next iteration` — "wait, it will land". It never lands. That report being
    wrong is the entire justification the new sets give for existing.
    """
    import importlib.util
    import sys

    root = _ROOT.parent
    spec = importlib.util.spec_from_file_location(
        "_arc_probe", root / "scripts" / "audit_realized_config.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_arc_probe"] = module
    spec.loader.exec_module(module)

    report, findings = module.classify_config_provenance(
        {},
        {"distributed_server_port": 45999, "cpus_per_trial": 4},
        {"distributed_server_port": 45453, "cpus_per_trial": 1},
        restart_keys=restart_required_config_keys(),
        construction_only_keys=construction_only_config_keys(),
        driver_launch_fixed_keys=driver_launch_fixed_config_keys(),
        driver_dual_clock_keys=driver_dual_clock_config_keys(),
    )
    text = "\n".join(report)
    assert "DRIVER-LAUNCH-FIXED distributed_server_port" in text, text
    assert "DRIVER-LAUNCH-FIXED cpus_per_trial" in text, text
    assert "PENDING-RESTART" not in text, "restart is the wrong instruction here"
    assert len(findings) == 2
    assert all("re-run run.py" in f for f in findings), findings


def test_the_operator_entry_point_reports_a_driver_key_as_launch_fixed(
    tmp_path: Path, capsys
) -> None:
    """The H5 wiring pin, at the level an OPERATOR actually invokes.

    ⚑ THE EARLIER PIN WAS ONE LAYER TOO DEEP. It drove
    `classify_config_provenance` and passed the two sets in itself, so it proved
    the function honours them — and proved nothing about whether anything passes
    them. The reviewer showed exactly that: deleting `driver_launch_fixed_keys=`
    and `driver_dual_clock_keys=` from `audit_config_provenance`'s internal call
    left all five test files green. A kwarg nothing supplies is this codebase's
    signature defect, and the test written to prevent it could not see it.

    So this drives `audit_config_provenance(rows, yaml, params)` — the function
    the script's `main` calls — and asserts the operator-visible line. Unwire
    either kwarg and this goes red.

    BOTH kwargs are exercised, and separately. An earlier revision asserted only
    on launch-fixed keys, so deleting `driver_dual_clock_keys=` alone left the
    suite green -- one pin covering two independent wires is one pin short.
    `tune_num_to_keep` is here to make the second wire load-bearing on its own.

    The row timestamp is deliberately AHEAD of the yaml mtime so the assertions
    land on the CLEAN-PATH label -- `DRIVER-LAUNCH-FIXED` / `DRIVER-DUAL-CLOCK`,
    the strings an operator reads when they audit a running trial whose yaml has
    already been picked up. An earlier version of this docstring justified it by
    claiming the `yaml_is_newer` branch would pass vacuously; that was WRONG on
    both counts, measured by the reviewer: that branch is labelled
    `DRIVER-LAUNCH-FIXED-UNRESOLVED`, so it still carries the classification and
    would discriminate perfectly well if asserted on. The choice is about which
    label matters, not about one of them being untestable.
    """
    import importlib.util
    import sys
    import time

    root = _ROOT.parent
    spec = importlib.util.spec_from_file_location(
        "_arc_entry_probe", root / "scripts" / "audit_realized_config.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_arc_entry_probe"] = module
    spec.loader.exec_module(module)

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "tune:\n"
        "  distributed_server_port: 45999\n"
        "  cpus_per_trial: 4\n"
  # Dual-clock, so `driver_dual_clock_keys=` is load-bearing on its own.
        "  tune_num_to_keep: 9\n",
        encoding="utf-8",
    )
    rows = [
        {
            "training_iteration": 200,
            # Ahead of the yaml mtime, so the real branch runs -- see docstring.
            "timestamp": time.time() + 3600.0,
            "config": {
                "distributed_server_port": 45453,
                "cpus_per_trial": 1,
                "tune_num_to_keep": 6,
                "_yaml_config_path": str(yaml_path),
            },
        }
    ]

    findings = module.audit_config_provenance(rows, yaml_path, None)
    printed = capsys.readouterr().out

    assert "DRIVER-LAUNCH-FIXED distributed_server_port" in printed, printed
    assert "DRIVER-LAUNCH-FIXED cpus_per_trial" in printed, printed
  # The second wire. Without this line, deleting `driver_dual_clock_keys=` from
  # the entry point leaves the whole suite green.
    assert "DRIVER-DUAL-CLOCK tune_num_to_keep" in printed, printed
    assert "PENDING-RESTART" not in printed, (
        "a driver key must not be reported as restart-required: restarting the "
        "TRIAL does not apply it, only re-running run.py does"
    )
    assert len(findings) == 3, findings
    assert any("re-run run.py" in f for f in findings), findings
    assert any("TWO clocks" in f for f in findings), findings
