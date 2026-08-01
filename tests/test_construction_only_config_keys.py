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
    construction_only_config_keys,
    restart_required_config_keys,
)
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file
from chess_anti_engine.utils.config_yaml import _FLAT_ALLOWLIST
from scripts.audit_realized_config import classify_config_provenance

_REPO = Path(__file__).resolve().parents[1]
_INIT_SRC = _REPO / "chess_anti_engine" / "tune" / "trainable_init.py"

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


def _replay_buffer_constructor_keys() -> dict[str, set[str]]:
    """AST: {constructor name -> config keys it is handed off the TrialConfig}."""
    tree = ast.parse(_INIT_SRC.read_text(encoding="utf-8"))
    out: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in ("DiskReplayBuffer", "ArrayReplayBuffer"):
            continue
        out.setdefault(node.func.id, set()).update(_tc_attrs_in_call(node))
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
    """
    pkg = _REPO / "chess_anti_engine"
    sources = {
        p.relative_to(_REPO).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted(pkg.rglob("*.py"))
    }
    assert len(sources) > 50, "package scan found almost nothing; the glob is wrong"
    for key in sorted(construction_only_config_keys()):
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

    This is the reproduced output. `params.json` holds the launch value the
    constructor got, the row echoes the yaml because the reloader overlaid it,
    and the old classifier listed that under a header asserting the running
    value was correct -- with an EMPTY findings list, so the audit exited 0.

    Asserted on the specific line for THIS key, not on ``findings != []``:
    a findings-count assertion is satisfied by any other key diverging, which
    is the disjunction, not the term.
    """
    params = {key: 25000}
    flat_yaml = {key: 100000}
    realized = {key: 100000}  # the overlay -- what the row reports

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
    assert any(line.startswith(f"  INERT-OVERLAY {key}:") for line in report), joined
    assert any(
        f.startswith(f"{key}: ") and "only ever read by a constructor" in f
        for f in findings
    ), findings


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
