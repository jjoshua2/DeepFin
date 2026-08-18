"""A yaml key that only a CONSTRUCTOR reads must not read back as healthy.

`shuffle_buffer_size`, `shuffle_refresh_interval`, `shuffle_refresh_shards`,
`shuffle_draw_cap_frac` and `shuffle_wl_max_ratio` are arguments to the
``DiskReplayBuffer`` built once in ``trainable_init._init_replay_buffers``.
Until 2026-08-01 they were in the yaml allowlist and in NONE of the
restart-required sets, so a mid-run yaml edit was overlaid into the trial's
``config`` dict, echoed back in every result row, and consumed by nobody. The
tool whose job is to catch that -- ``scripts/audit_realized_config.py`` -- then
compared the yaml against the overlaid dict and printed

    STALE-IN-PARAMS-JSON  ... (the running value is correct):
      shuffle_buffer_size: params.json=25000 running=100000
    FINDINGS: []

which is an affirmative statement of correctness about a cap the buffer was
not running, and it could not have printed anything else.

The two halves pinned here:

  * the CLASSIFICATION -- every ``tc.<key>`` handed to a replay-buffer
    constructor is either restart-required or has a named, verified live-push
    site, so a constructor argument added tomorrow cannot silently join the
    broken class;
  * the REPORT -- the provenance classifier cannot reach its
    "params.json holds the LAUNCH value" verdict for a construction-only key.

Both are asserted on the SPECIFIC signal (this key, this line), not on an exit
code or a findings count: an aggregate assertion passes on the disjunction and
would survive the very substitution it exists to forbid.
"""
from __future__ import annotations

import ast
import logging
import re
from pathlib import Path

import pytest

from chess_anti_engine.tune.trainable_config_ops import (
    _reload_yaml_into_config,
    _STARTUP_ONLY_TRIAL_KEYS,
    construction_only_config_keys,
    restart_required_config_keys,
)
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file
from chess_anti_engine.utils.config_yaml import _FLAT_ALLOWLIST
from scripts.audit_realized_config import classify_config_provenance

_REPO = Path(__file__).resolve().parents[1]

# Replay-buffer constructor arguments that are NOT construction-bound, because
# some per-iteration consumer re-reads them. Each entry names the file and the
# exact source text that makes the claim true, and the test greps for it: an
# unverified exemption is how a knob rejoins the broken class wearing an
# "it's live" comment.
_LIVE_PUSHED_BUFFER_KEYS: dict[str, tuple[str, str]] = {
    "replay_sf_gap_priority_weight": (
        "chess_anti_engine/tune/trainable.py",
        "buf.sf_gap_priority_weight = tc.replay_sf_gap_priority_weight",
    ),
    "replay_fast_low_surprise_priority": (
        "chess_anti_engine/tune/trainable.py",
        "buf.fast_low_surprise_priority = tc.replay_fast_low_surprise_priority",
    ),
    "diff_focus_pol_scale": (
        "chess_anti_engine/tune/trainable.py",
        "buf.diff_focus_pol_scale = tc.diff_focus_pol_scale",
    ),
    "diff_focus_q_weight": (
        "chess_anti_engine/tune/trainable.py",
        "buf.diff_focus_q_weight = tc.diff_focus_q_weight",
    ),
    # The buffer's own `_shard_size` IS frozen at construction, but the key has
    # a second consumer: the per-iteration exploit-replay share ingest re-reads
    # it off the freshly reloaded TrialConfig, so freezing the key would stop a
    # live edit that does take effect there.
    "shard_size": (
        "chess_anti_engine/tune/trainable_phases.py",
        "shard_size=tc.shard_size",
    ),
}

# Files allowed to mention a construction-only key: declaring it, parsing it
# out of the config, allowlisting it in the yaml schema, and the ONE
# construction site. Any other module means a live consumer exists and the key
# does not belong in the set.
_ALLOWED_CONSUMER_FILES = {
    "chess_anti_engine/tune/trial_config.py",
    "chess_anti_engine/tune/trainable_init.py",
    "chess_anti_engine/tune/trainable_config_ops.py",
    "chess_anti_engine/utils/config_yaml.py",
}


# Where each startup-only key is actually READ, one entry per key. The bare-word
# scan above cannot be used for these: "iterations" appears in prose in a dozen
# modules. This is the enumeration the scan enforces, so adding a reader without
# updating it fails the test.
_STARTUP_ONLY_READER_FILES: dict[str, set[str]] = {
    # `iterations = tc.iterations` (the while bound, read once into a local) and
    # the DRIVER-side `base_config.get("iterations")` that sizes ASHA's max_t at
    # experiment creation. Neither runs per iteration.
    "iterations": {
        "chess_anti_engine/tune/trainable.py",
        "chess_anti_engine/tune/harness.py",
    },
    "puzzle_epd": {"chess_anti_engine/tune/trainable.py"},
    # `run.py` resolves its default (`base["eval_sf_nodes"] = args.sf_nodes`)
    # while building the base config, once, before any trial exists — a CLI-time
    # write, not a per-iteration read.
    "eval_sf_nodes": {
        "chess_anti_engine/tune/trainable.py",
        "chess_anti_engine/run.py",
    },
    "sf_pid_enabled": {"chess_anti_engine/tune/trainable.py"},
    # Read ONCE into `Trainer._loss_kwargs` at construction, so a live yaml edit
    # cannot reach the loss without a restart — declared startup-only for that
    # reason. It reshapes the TRAINING TARGET, so a mid-run change would also
    # split a readout window across two different targets with nothing in the
    # metrics able to say where the split fell.
    "policy_target_temp": {"chess_anti_engine/train/trainer.py"},
    # `_resolve_pause_marker_paths`, which the startup block calls once.
    "pause_file": {"chess_anti_engine/tune/trainable_config_ops.py"},
    # SF-policy-floor SHAPE. TWO readers, and only one of them is a consumer:
    # `trainer.py` folds all three into one frozen `SfPolicyFloorParams` at
    # construction (so a live edit provably cannot reach the loss -- hence
    # startup-only), while `trial_config.py` reads them EVERY iteration purely to
    # re-run the range check and DISCARDS the result. A validator is not a
    # consumer: it turns a bad live value into a loud death instead of silent
    # wrongness, and it deliberately stores nothing that could disagree with the
    # object the loss holds. See `TrialConfig.from_dict`.
    **{
        key: {
            "chess_anti_engine/train/trainer.py",
            "chess_anti_engine/tune/trial_config.py",
        }
        for key in (
            "sf_policy_floor_delta_cp",
            "sf_policy_floor_tau",
            "sf_policy_floor_tau_top1",
            "sf_policy_floor_tau_played",
        )
    },
    # SF-shape teacher temperature. Same two readers and the same split: the
    # Trainer folds it into a frozen `SfShapeParams` at construction, and
    # `TrialConfig.from_dict` re-reads it every iteration purely to re-run the
    # range check and discards the result.
    "sf_shape_temp_cp": {
        "chess_anti_engine/train/trainer.py",
        "chess_anti_engine/tune/trial_config.py",
    },
}

# A config READ, as opposed to a mention: `tc.key`, `config["key"]`,
# `config.get("key"`, and the `base_config`/`cfg` spellings of the same. Prose,
# argparse `--flag` strings and yaml allowlist tuples do not match.
def _config_read_pattern(key: str) -> re.Pattern[str]:
    k = re.escape(key)
    return re.compile(
        rf"""(?:\b(?:tc|trial_config)\.{k}\b)"""
        rf"""|(?:\.get\(\s*["']{k}["'])"""
        rf"""|(?:\[\s*["']{k}["']\s*\])""",
    )


def _config_read_offenders(
    key: str, *, sources: dict[str, str], allowed: set[str],
) -> list[str]:
    """Files that READ ``key`` and are neither a declared reader nor a parser."""
    pattern = _config_read_pattern(key)
    return sorted(
        rel for rel, src in sources.items()
        if rel not in _ALLOWED_CONSUMER_FILES
        and rel not in allowed
        and pattern.search(src)
    )


def _write_one_key_yaml(path: Path, key: str, value: object) -> None:
    """Write a yaml the live validator ACCEPTS, carrying only ``key``.

    The section is discovered rather than hard-coded: the validator is
    all-or-nothing, so a key written into the wrong section rejects the whole
    file, every value stays put, and a test asserting "the value did not move"
    passes for exactly the wrong reason.
    """
    import yaml as _yaml

    for doc in (
        {key: value},
        *({section: {key: value}} for section in
          ("tune", "train", "selfplay", "model", "stockfish")),
    ):
        path.write_text(_yaml.safe_dump(doc), encoding="utf-8")
        try:
            flat = flatten_run_config_defaults(load_yaml_file(str(path)))
        except (ValueError, KeyError):
            continue
        if key in flat:
            return
    raise AssertionError(f"no yaml placement of {key!r} survives the validator")


def _tc_attrs_in_call(call: ast.Call) -> set[str]:
    """Every ``tc.<attr>`` appearing anywhere inside a call's arguments."""
    found: set[str] = set()
    for node in ast.walk(call):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "tc"
        ):
            found.add(node.attr)
    return found


def _ctor_name(func: ast.expr) -> str | None:
    """The constructor's name whether called bare or through a module.

    ``ast.Name`` alone misses ``replay.DiskReplayBuffer(...)``, which is the
    same construction wearing an import style.
    """
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _replay_buffer_constructor_keys() -> dict[str, set[str]]:
    """AST over the WHOLE package: {constructor -> keys handed off TrialConfig}.

    ⚑ This scanned only ``trainable_init.py`` until 2026-08-01. An independent
    review killed that version by adding a live ``DiskReplayBuffer(...)`` call
    in ``trainable_phases.py`` and watching the suite pass anyway -- a second
    construction site was invisible, so the guard could not fail for the very
    shape it exists to catch. Scoping a guard to the one file where the defect
    was first found is how it stops generalising.
    """
    out: dict[str, set[str]] = {}
    for path in sorted((_REPO / "chess_anti_engine").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken file fails elsewhere
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _ctor_name(node.func)
            if name not in ("DiskReplayBuffer", "ArrayReplayBuffer"):
                continue
            out.setdefault(name, set()).update(_tc_attrs_in_call(node))
    return out


def test_every_replay_buffer_constructor_argument_is_classified() -> None:
    """MUTATION: delete ``"shuffle_buffer_size"`` from
    ``_CONSTRUCTION_ONLY_REPLAY_KEYS`` (or add a new ``tc.*`` argument to the
    ``DiskReplayBuffer(...)`` call without classifying it).

    This is the guard that generalises past today's list. The defect was never
    "these five keys are in the wrong set" -- it was that nothing anywhere
    connected "argument to a once-per-process constructor" to "cannot be
    live-reloaded", so the next argument added would have joined them in
    silence. Reading the call site with the AST means the source of the answer
    is the production call itself.
    """
    by_ctor = _replay_buffer_constructor_keys()
    assert set(by_ctor) == {"DiskReplayBuffer", "ArrayReplayBuffer"}, by_ctor
    keys = by_ctor["DiskReplayBuffer"] | by_ctor["ArrayReplayBuffer"]
    # Sanity: the reproduced key must actually be found by the parse, or a
    # refactor that moves the call would make this test vacuous.
    assert "shuffle_buffer_size" in keys, sorted(keys)
    assert "holdout_capacity" in by_ctor["ArrayReplayBuffer"], by_ctor

    restart = restart_required_config_keys()
    construction_only = construction_only_config_keys()
    for key in sorted(keys):
        if key not in _FLAT_ALLOWLIST:
            # Not settable from yaml at all, so no live edit can reach it.
            continue
        if key in _LIVE_PUSHED_BUFFER_KEYS:
            rel, needle = _LIVE_PUSHED_BUFFER_KEYS[key]
            src = (_REPO / rel).read_text(encoding="utf-8")
            assert needle in src, (
                f"{key} is exempted as live-pushed via {rel}, but {needle!r} is "
                "no longer there; the exemption is now false and the key is "
                "construction-bound"
            )
            assert key not in construction_only, key
            continue
        assert key in restart, (
            f"{key} is handed to a replay-buffer constructor that runs once per "
            "trial process, but a live yaml reload will overlay it into config "
            "anyway. Either classify it in _CONSTRUCTION_ONLY_REPLAY_KEYS or add "
            "it to _LIVE_PUSHED_BUFFER_KEYS with the per-iteration push site"
        )


def test_construction_only_keys_have_no_live_consumer() -> None:
    """MUTATION: add ``"replay_window_max"`` to ``_CONSTRUCTION_ONLY_REPLAY_KEYS``.

    The opposite error, and the more damaging one: freezing a key that DOES
    have a live consumer stops a working knob from reloading. A key qualifies
    only if the whole runtime package mentions it in the declaring, parsing,
    allowlisting and constructing files and nowhere else.

    ``_STARTUP_ONLY_TRIAL_KEYS`` (audit T2) is scanned too, by
    ``_config_read_offenders`` below rather than by the bare-word regex: those
    keys live in ``trainable.py``/``harness.py`` alongside prose that mentions
    them, so word matching is unusable, while the *reader* files still have to
    be enumerated. The first version of this PR exempted them from this test
    entirely and hand-classified ``eval_games`` as startup-only -- and this test
    is precisely the one that would have named ``trainable_phases.py`` and
    stopped it (review B1). See
    ``test_the_startup_only_scan_names_the_key_that_slipped_through``.
    """
    pkg = _REPO / "chess_anti_engine"
    sources = {
        p.relative_to(_REPO).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted(pkg.rglob("*.py"))
    }
    assert len(sources) > 50, "package scan found almost nothing; the glob is wrong"
    scanned = construction_only_config_keys() - _STARTUP_ONLY_TRIAL_KEYS
    assert "shuffle_buffer_size" in scanned, sorted(scanned)
    for key in sorted(scanned):
        pattern = re.compile(rf"\b{re.escape(key)}\b")
        offenders = sorted(
            rel for rel, src in sources.items()
            if rel not in _ALLOWED_CONSUMER_FILES and pattern.search(src)
        )
        assert not offenders, (
            f"{key} is declared construction-only but is also referenced in "
            f"{offenders} — if any of those reads it after startup, a live edit "
            "DOES take effect there and freezing the key breaks it"
        )

    for key in sorted(_STARTUP_ONLY_TRIAL_KEYS):
        allowed = _STARTUP_ONLY_READER_FILES.get(key)
        assert allowed is not None, (
            f"{key} is declared startup-only but names no reader file; add it to "
            "_STARTUP_ONLY_READER_FILES with the ONE place that reads it"
        )
        offenders = _config_read_offenders(key, sources=sources, allowed=allowed)
        assert not offenders, (
            f"{key} is declared startup-only but is READ (not merely mentioned) "
            f"in {offenders}, which is not among its declared readers "
            f"{sorted(allowed)} — if any of those runs per iteration, a live "
            "edit DOES take effect there and freezing the key breaks it"
        )


def test_the_startup_only_scan_names_the_key_that_slipped_through() -> None:
    """NEGATIVE CONTROL on the guard above, and the B1 regression test.

    An instrument is only worth having if it FAILS on the known-bad input. PR F
    as first written froze ``eval_games`` as startup-only; the scan above,
    handed that key with the same reader set the other startup-only keys get,
    must name ``trainable_phases.py`` — the per-iteration ``games=tc.eval_games``
    that made the freeze wrong. If this ever comes back empty, the guard has
    stopped guarding and the hand-classification is unchecked again.
    """
    pkg = _REPO / "chess_anti_engine"
    sources = {
        p.relative_to(_REPO).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted(pkg.rglob("*.py"))
    }
    offenders = _config_read_offenders(
        "eval_games",
        sources=sources,
        allowed={"chess_anti_engine/tune/trainable.py"},
    )
    assert "chess_anti_engine/tune/trainable_phases.py" in offenders, offenders
    assert "eval_games" not in _STARTUP_ONLY_TRIAL_KEYS


@pytest.mark.parametrize("key", sorted(construction_only_config_keys()))
def test_live_reload_refuses_and_warns_for_each_construction_only_key(
    key: str, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """MUTATION: drop ``_CONSTRUCTION_ONLY_REPLAY_KEYS`` from the union that
    builds ``_LIVE_RELOAD_SKIPPED_KEYS``.

    Parametrised per key rather than asserted over the set: a loop inside one
    test that asserted "some key warned" would pass while four of the five
    silently overlaid. Both halves are required -- the value must NOT move
    (an applied value is a knob that cannot act) AND the operator must be told,
    since the whole difference between "restart required" and "quietly ignored"
    is the log line.
    """

    def _restart_warning_for(key: str) -> list[str]:
        """Warnings that name THIS key AND say restart. One record, not two.

        Matching "key somewhere in caplog.text" and "requires restart somewhere
        in caplog.text" separately would be satisfied by two unrelated records
        -- some other key's warning plus an incidental mention.
        """
        return [
            r.getMessage() for r in caplog.records
            if key in r.getMessage() and "requires restart" in r.getMessage()
        ]

    yaml_path = tmp_path / "live.yaml"
    _write_one_key_yaml(yaml_path, key, 4321)

    # (a) the live CHANGE case.
    config: dict[str, object] = {key: 1234}
    with caplog.at_level(logging.WARNING):
        _reload_yaml_into_config(config, str(yaml_path), live_reload=True)
    assert config[key] == 1234, (
        f"{key} reached the running config, but the object that consumes it was "
        "constructed at startup and will never re-read it"
    )
    assert _restart_warning_for(key), caplog.text

    # (b) the live ADD case -- shuffle_draw_cap_frac / shuffle_wl_max_ratio are
    # absent from configs/pbt2_small.yaml, so an ADD is the only way an operator
    # ever sets them, and it is the case the old `k in config` guard missed.
    caplog.clear()
    config_add: dict[str, object] = {}
    with caplog.at_level(logging.WARNING):
        _reload_yaml_into_config(config_add, str(yaml_path), live_reload=True)
    assert key not in config_add, caplog.text
    assert _restart_warning_for(key), caplog.text


@pytest.mark.parametrize("key", sorted(construction_only_config_keys()))
def test_startup_reload_still_applies_each_construction_only_key(
    key: str, tmp_path: Path,
) -> None:
    """MUTATION: move the construction-only keys into ``_TOPOLOGY_KEYS``
    instead of ``_LIVE_RELOAD_SKIPPED_KEYS``.

    The two sets are branched on differently: the topology branch is NOT gated
    on ``live_reload``, so it would also refuse the value at startup/resume and
    a restart could never pick the yaml up -- turning "your edit needs a
    restart" into "your edit is unreachable", which is worse than the bug being
    fixed. Restart is the ONLY way these keys change, so this is the assertion
    that keeps the fix from eating the feature.
    """
    yaml_path = tmp_path / "live.yaml"
    _write_one_key_yaml(yaml_path, key, 4321)
    config: dict[str, object] = {key: 1234}
    _reload_yaml_into_config(config, str(yaml_path), live_reload=False)
    assert config[key] == 4321, (
        f"{key} must still be applied at startup/resume; a restart is the only "
        "way it can ever change"
    )


@pytest.mark.parametrize("key", sorted(construction_only_config_keys()))
def test_provenance_never_calls_a_construction_only_key_correct(key: str) -> None:
    """MUTATION: in ``classify_config_provenance``, drop ``construction_only``
    from the branch condition, or restore the header wording
    "(the running value is correct)".

    This is the reproduced output: `params.json` differs from the row, and the
    old classifier listed exactly that under a header asserting the running
    value was correct -- with an EMPTY findings list, so the audit exited 0.

    What it asserts is a LINE, for THIS key, not ``findings != []`` -- a
    findings-count assertion is satisfied by any other key diverging, which is
    the disjunction, not the term.
    """
    params = {key: 25000}
    flat_yaml = {key: 100000}
    realized = {key: 100000}  # the row: yaml value, from the startup reload

    report, findings = classify_config_provenance(
        params, flat_yaml, realized,
        # Deliberately WITHOUT the key in restart_keys: that models the exact
        # pre-fix classification, and proves the second gate stands on its own.
        restart_keys=(),
        construction_only_keys=construction_only_config_keys(),
    )
    joined = "\n".join(report)
    assert f"{key}: params.json=25000 running=100000" not in joined, joined
    assert "the running value is correct" not in joined, joined
    # And it must not swing to the opposite error either: params.json is the
    # trial's ORIGINAL creation config, so "the object runs params.json's value"
    # is the J5 mistake, not the fix. The honest line says which source the row
    # is, and raises no finding.
    assert any(
        line.startswith(f"  note(ctor)  {key}:")
        and "ORIGINAL" in line and "does not describe this process" in line
        for line in report
    ), joined
    assert not findings, findings


@pytest.mark.parametrize("key", sorted(construction_only_config_keys()))
def test_provenance_reports_pending_restart_for_a_live_edit(key: str) -> None:
    """The post-fix state: the reloader no longer overlays, so the row holds the
    launch value and the yaml holds the operator's edit.

    MUTATION: drop ``or key in construction_only`` from the branch condition in
    ``classify_config_provenance``, or delete the "constructor argument"
    explanation from the finding.

    ``restart_keys`` is empty here on purpose. The whole defect was a checker
    that inherited one set's answer to a different question, so the guarantee
    worth pinning is that the construction-only gate produces PENDING-RESTART
    on its own: with the ``or`` gone this key falls through to
    RELOAD-NOT-APPLIED, which tells the operator to look for a rejected reload
    that never happened. Reverting the classification is caught by
    ``test_live_reload_refuses_and_warns_for_each_construction_only_key`` and
    ``test_every_replay_buffer_constructor_argument_is_classified``, which is
    where that mutation belongs.
    """
    report, findings = classify_config_provenance(
        {key: 25000}, {key: 100000}, {key: 25000},
        restart_keys=(),
        construction_only_keys=construction_only_config_keys(),
    )
    assert any(line.startswith(f"  PENDING-RESTART {key}:") for line in report), report
    assert any(
        f.startswith(f"{key}: ") and "constructor argument" in f for f in findings
    ), findings


def test_the_stale_header_never_asserts_the_running_value_is_correct() -> None:
    """MUTATION: restore "(the running value is correct)" to the
    STALE-IN-PARAMS-JSON header in ``classify_config_provenance``.

    ⚑ ``test_provenance_never_calls_a_construction_only_key_correct`` looks like
    it already covers this and does NOT: construction-only keys are routed to
    the ``note(ctor)`` branch and never reach this header at all, so its
    ``"the running value is correct" not in joined`` assertion is vacuously
    true for every key it is parametrised over. An independent review proved
    that by restoring the wording alone and watching all 34 tests pass.

    This drives an ORDINARY live-reloadable key -- one that genuinely is stale
    in params.json and genuinely was overlaid -- so the header is actually
    emitted and its wording is on the hook. The claim being forbidden is the
    strong one: the reloader applying a value does not establish that the
    consumer re-read it.
    """
    key = "replay_sf_gap_priority_weight"  # live-pushed every iteration
    assert key not in construction_only_config_keys(), (
        f"{key} must stay OUT of the construction-only set, or this test stops "
        "exercising the stale-header branch and goes vacuous like the one above"
    )

    report, findings = classify_config_provenance(
        {key: 0.0}, {key: 9.5}, {key: 9.5},
        restart_keys=(),
        construction_only_keys=construction_only_config_keys(),
    )
    joined = "\n".join(report)

    assert "STALE-IN-PARAMS-JSON" in joined, joined
    assert f"{key}: params.json=0.0 running=9.5" in joined, joined
    assert "the running value is correct" not in joined, (
        "the header asserts the running value is correct; that is the claim the "
        f"audit cannot support -- it knows the reloader applied {key}, not that "
        f"any consumer re-read it\n{joined}"
    )
    assert "That the RELOADER applied it is all this shows" in joined, joined
    assert findings == [], findings
