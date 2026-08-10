"""Pins that the C17 batching knobs actually REACH the selfplay search call.

This project's recurring defect is not a wrong value, it is a value that never
arrives: `diff_focus` never reached the worker, `soft_policy_temp` was never
published, the sf_p0 teacher was dead for 15.8 days. A knob that is accepted and
silently ignored is worse than no knob, because it produces a confident verdict
on an experiment that never ran.

`selfplay/network_turn.py` is the ONLY production selfplay search call site
(`selfplay/match.py` is the arena/match path). Before this plumbing it passed
neither `target_batch` nor `vloss_weight`, so both took their 0 defaults and
production ran the 56%-duplicate arm with no way to change it from config.
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path

import pytest
import yaml

from chess_anti_engine.selfplay import network_turn
from chess_anti_engine.selfplay.config import SearchConfig
from chess_anti_engine.tune.trainable_config_ops import _play_batch_kwargs
from chess_anti_engine.tune.trial_config import TrialConfig
from chess_anti_engine.utils.config_yaml import (
    SELFPLAY_CONFIG_KEYS,
    flatten_run_config_defaults,
)
from chess_anti_engine.worker import WorkerSession
from tests.test_reco_coverage import _bare_session, _reco_from


def test_the_knobs_default_to_todays_production_behaviour() -> None:
    """0/0 must stay byte-identical to the behaviour before the fields existed.

    The whole point of a default-off research knob is that merging it changes
    nothing until someone opts in.
    """
    cfg = SearchConfig()
    assert cfg.gumbel_target_batch == 0
    assert cfg.gumbel_vloss_weight == 0


def _gumbel_c_call_kwargs() -> set[str]:
    """Keyword names passed to the C gumbel runner in run_network_turn."""
    # `run_network_turn` is module-level, so its source is already at column 0.
    # Do NOT cleandoc it -- that strips the body's indentation and the parse
    # fails on the docstring.
    src = inspect.getsource(network_turn.run_network_turn)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id == "gumbel_c_fn":
            return {kw.arg for kw in node.keywords if kw.arg is not None}
    raise AssertionError(
        "no gumbel_c_fn(...) call found in run_network_turn -- the selfplay "
        "search call site moved; this test must be re-pointed, not deleted",
    )


def test_both_knobs_are_passed_to_the_c_gumbel_runner() -> None:
    """The plumbing test proper: the kwargs are present at the call site.

    Asserted on the AST rather than by running a search, because reaching this
    call needs a model, a broker and a live game state. An AST check cannot
    prove the VALUE is right, so `test_the_call_reads_them_from_search_config`
    below pins where the value comes from.
    """
    kwargs = _gumbel_c_call_kwargs()
    missing = {"target_batch", "vloss_weight"} - kwargs
    assert not missing, f"selfplay drops {sorted(missing)} on the floor again"


def test_the_call_reads_them_from_search_config() -> None:
    """Guards the failure where a knob is passed but wired to a constant.

    Passing `target_batch=0` literally would satisfy the test above while making
    the config field dead -- which is exactly how `matrix_weight_decay` became
    decorative.
    """
    src = inspect.getsource(network_turn.run_network_turn)
    for name in ("gumbel_target_batch", "gumbel_vloss_weight"):
        assert f"search.{name}" in src, (
            f"{name} is not read from the search config at the call site"
        )


def test_virtual_mean_is_not_reachable_from_selfplay_config() -> None:
    """VIRTUAL_MEAN is deliberately NOT offered to selfplay.

    It is measured-dominated: it removes duplicates but collapses the batch to
    ~26 rows/call, the same 8x round-trip cost as `target_batch=1`, and returns
    nothing for it. The pessimism in LEGACY is what spreads walkers onto
    distinct leaves; VIRTUAL_MEAN's whole selling point -- leaving Q untouched --
    is why it cannot fill a batch. Keeping it out of SearchConfig means nobody
    can select it here by reading the C enum and assuming higher is better.
    """
    assert not hasattr(SearchConfig(), "gumbel_vloss_mode")


_REPO = Path(__file__).resolve().parents[1]
_PRODUCTION_YAML = _REPO / "configs" / "pbt2_small.yaml"


# ---------------------------------------------------------------------------
# Where SearchConfig is built, found deterministically
# ---------------------------------------------------------------------------


def _search_config_call_sites(rel_path: str) -> list[ast.Call]:
    """Every ``SearchConfig(...)`` construction in *rel_path*, in file order.

    Deliberately NOT ``next(...)`` over ``ast.walk``: ``ast.walk`` is
    breadth-first, so "the first match" is the SHALLOWEST call, not the first in
    the file. Appending a module-level ``SearchConfig(simulations=1)`` to the end
    of ``worker.py`` used to make this helper return that call instead of the one
    inside ``_build_selfplay_configs``. Collect them all and sort by position, so
    a second construction site is a visible fact rather than a silent shadow.
    """
    tree = ast.parse((_REPO / rel_path).read_text(encoding="utf-8"))
    sites = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SearchConfig"
    ]
    sites.sort(key=lambda n: (n.lineno, n.col_offset))
    assert sites, (
        f"no SearchConfig(...) construction found in {rel_path} -- the config "
        f"builder moved; re-point this test, do not delete it"
    )
    return sites


def _sole_call_site(rel_path: str) -> ast.Call:
    """The one construction site in *rel_path*; fail if a second appears.

    A second site is how ``trainable_config_ops._play_batch_kwargs`` came to
    build a SearchConfig that omitted both C17 knobs while every by-name test
    passed. Each site scanned here gets its own coverage table below, so a NEW
    one must be classified rather than inherited.
    """
    sites = _search_config_call_sites(rel_path)
    assert len(sites) == 1, (
        f"{rel_path} now has {len(sites)} SearchConfig(...) sites at lines "
        f"{[n.lineno for n in sites]}. Every site is a place a knob can go dead; "
        f"give the new one its own coverage table in this file."
    )
    return sites[0]


def _kwarg_names(call: ast.Call) -> set[str]:
    return {kw.arg for kw in call.keywords if kw.arg is not None}


def _worker_reco_keys() -> dict[str, str | None]:
    """Map the worker's SearchConfig kwarg -> the reco key it resolves from.

    ``None`` means the kwarg is set from something other than a literal
    ``self._resolve_reco(reco, "key", ...)``. That is not automatically wrong,
    but it IS the `matrix_weight_decay`-decorative shape (a literal that looks
    like plumbing), so every such field must be named in
    ``_WORKER_NOT_A_RECO_KEY`` with a reason -- it may not be silently skipped.
    """
    out: dict[str, str | None] = {}
    for kw in _sole_call_site("chess_anti_engine/worker.py").keywords:
        if kw.arg is None:
            continue
        key: str | None = None
        for sub in ast.walk(kw.value):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "_resolve_reco"
                and len(sub.args) >= 2
                and isinstance(sub.args[1], ast.Constant)
                and isinstance(sub.args[1].value, str)
            ):
                key = sub.args[1].value
                break
        out[kw.arg] = key
    return out


# ---------------------------------------------------------------------------
# The exemption tables. Entry here is a claim, and each is checked.
# ---------------------------------------------------------------------------

# SearchConfig fields the distributed worker deliberately leaves on their
# dataclass default. Anything NOT listed must be reachable end to end, so this
# table is self-invalidating: deleting a field's plumbing forces someone to
# write the reason down here rather than let the knob quietly go dead. It is
# also invalidated from the other side -- an exempt field that reappears in the
# published reco fails `test_exempt_fields_are_not_published`.
_INTENTIONALLY_NOT_CONFIGURABLE: dict[str, str] = {
    # PUCT-only. network_turn.py builds a GumbelConfig for the production search
    # and never forwards either one, so publishing them would create a knob that
    # is reachable and still unread -- strictly worse than a documented gap.
    #
    # This reason holds ONLY while the production config runs `mcts: gumbel`:
    # network_turn.py takes the PUCT branch for a non-gumbel mcts_type at full
    # sim counts, and that branch DOES read both fields -- at which point they
    # would be read from the dataclass default and a yaml edit would be
    # accepted and ignored, the canonical defect. `mcts` is itself a live-
    # editable allowlisted key, so the dependency is pinned by
    # `test_the_fpu_exemption_depends_on_a_pinned_mcts_gumbel`.
    #
    # Note the reason is about the DISTRIBUTED path only: _play_batch_kwargs
    # does set both from TrialConfig for gate/eval matches.
    "fpu_reduction": (
        "PUCT path only; the Gumbel production search never reads it "
        "(valid only while configs/pbt2_small.yaml pins mcts: gumbel)"
    ),
    "fpu_at_root": (
        "PUCT path only; Gumbel uses gumbel_scale root noise instead "
        "(valid only while configs/pbt2_small.yaml pins mcts: gumbel)"
    ),
    # NOT merely unread -- FATAL. A non-zero volatility_q_scale/volatility_fpu
    # makes volatility_search_enabled() true, which drops network_turn.py off
    # the C path onto mcts.gumbel.run_gumbel_root_many, which raises ValueError
    # unless the evaluator exposes `evaluate_encoded_with_volatility`. Every
    # evaluator a distributed worker can hold lacks it -- MultiSlotInferenceClient,
    # ThreadedDispatcher, SlotInferenceClient, AOTEvaluator (only
    # LocalModelEvaluator and DirectGPUEvaluator implement it, and both are
    # in-trainer). Nothing catches the ValueError, so the worker process exits;
    # all workers read the same manifest, so they exit together. Publishing
    # these would convert an inert yaml key into a fleet crash switch, so the
    # honest state is not "configurable" but "cannot work here at all", enforced
    # by config_yaml._check_volatility_search_unsupported.
    "volatility_q_scale": (
        "no distributed-worker evaluator implements "
        "evaluate_encoded_with_volatility; a non-zero value kills every worker"
    ),
    "volatility_fpu": (
        "no distributed-worker evaluator implements "
        "evaluate_encoded_with_volatility; a non-zero value kills every worker"
    ),
    "volatility_anchor": (
        "inert without volatility_q_scale/volatility_fpu, which cannot run on "
        "the distributed path at all"
    ),
}

# Worker SearchConfig kwargs that are NOT a plain `_resolve_reco(reco, "k", ...)`.
# Empty on purpose: an unexplained entry here is the decorative-literal defect.
_WORKER_NOT_A_RECO_KEY: dict[str, str] = {}

# SearchConfig fields _play_batch_kwargs (gate/eval matches) does not set.
_GATE_EVAL_NOT_SET: dict[str, str] = {
    "simulations": (
        "set per phase by the caller via dataclasses.replace "
        "(gate_mcts_sims / eval_mcts_simulations), not from TrialConfig"
    ),
}

# yaml key -> a sentinel value distinct from the SearchConfig default, used to
# drive the publisher and the worker resolver for real. The table must cover
# exactly the non-exempt fields (asserted), so adding a field forces adding a
# sentinel rather than widening a skip.
_SENTINELS: dict[str, tuple[str, object]] = {
    "simulations": ("mcts_simulations", 137),
    "mcts_type": ("mcts", "gumbel"),
    "playout_cap_fraction": ("playout_cap_fraction", 0.37),
    "full_ply_pair_fraction": ("full_ply_pair_fraction", 0.41),
    "fast_simulations": ("fast_simulations", 13),
    "gumbel_topk": ("gumbel_topk", 7),
    "gumbel_policy_temp": ("gumbel_policy_temp", 1.63),
    "gumbel_target_batch": ("gumbel_target_batch", 3),
    "gumbel_vloss_weight": ("gumbel_vloss_weight", 2),
    "gumbel_target_max_visit_cap": ("gumbel_target_max_visit_cap", 9),
    "gumbel_c_scale": ("gumbel_c_scale", 0.077),
    "gumbel_scale": ("gumbel_scale", 0.31),
    "gumbel_scale_after": ("gumbel_scale_after", 0.29),
    "gumbel_scale_decay_start_move": ("gumbel_scale_decay_start_move", 11),
    "gumbel_scale_decay_moves": ("gumbel_scale_decay_moves", 12),
    "curriculum_gumbel_scale": ("curriculum_gumbel_scale", 0.23),
    "curriculum_gumbel_scale_after": ("curriculum_gumbel_scale_after", 0.19),
    "curriculum_gumbel_scale_decay_start_move": (
        "curriculum_gumbel_scale_decay_start_move", 14,
    ),
    "curriculum_gumbel_scale_decay_moves": (
        "curriculum_gumbel_scale_decay_moves", 15,
    ),
}


def _realized_search(config: dict[str, object]) -> SearchConfig:
    """The SearchConfig a worker ACTUALLY builds for *config*.

    Runs the real publisher and the real worker resolver back to back, so it
    fails on any break anywhere in between: a publisher that reads the wrong
    config key, a publish line that survives only in a comment, a key emitted
    into an unrelated dict, a resolver replaced by a literal, or a
    publisher/worker default disagreement.
    """
    cfgs, _sf_args = _bare_session()._build_selfplay_configs(_reco_from(config))
    search = cfgs["search"]
    assert isinstance(search, SearchConfig)
    return search


# ---------------------------------------------------------------------------
# The functional plumbing tests
# ---------------------------------------------------------------------------


def test_the_sentinel_table_covers_exactly_the_configurable_fields() -> None:
    """Neither table may drift from SearchConfig; together they must partition it."""
    declared = {f.name for f in dataclasses.fields(SearchConfig)}
    covered = set(_SENTINELS) | set(_INTENTIONALLY_NOT_CONFIGURABLE)
    assert covered == declared, (
        f"SearchConfig fields with no sentinel and no exemption: "
        f"{sorted(declared - covered)}; stale table entries: "
        f"{sorted(covered - declared)}"
    )
    assert not (set(_SENTINELS) & set(_INTENTIONALLY_NOT_CONFIGURABLE))


def test_a_yaml_value_actually_reaches_the_worker_search_config() -> None:
    """The general form of the C17 defect, end to end and by execution.

    `gumbel_target_batch` shipped as a field the search call site duly forwarded
    and that NOTHING could ever set -- absent from the publisher, the resolver
    and the yaml allowlist, pinned at 0 forever while looking configurable in
    three places. Pinning knobs by name cannot catch a knob nobody named.

    Asking it the other way round -- for every field the production search
    reads, does a yaml value arrive intact? -- catches that, and also catches
    the four ways a SOURCE-TEXT check is fooled: a publisher reading
    `config.get("gumbel_target_batchXX")` while emitting the right name, a
    publish line deleted down to a comment, the key emitted into some unrelated
    dict in the same file, and a resolver swapped for a bare literal.
    """
    for field, (yaml_key, sentinel) in sorted(_SENTINELS.items()):
        realized = getattr(_realized_search({yaml_key: sentinel}), field)
        assert realized == sentinel, (
            f"{yaml_key}={sentinel!r} does not survive publisher -> reco -> "
            f"worker: SearchConfig.{field} realized as {realized!r}. The knob "
            f"is accepted and ignored."
        )


def test_publisher_and_worker_agree_on_every_default() -> None:
    """An unset yaml key must realize the dataclass default, not a third number.

    The publisher has its own default per key and the worker's `_resolve_reco`
    has another; nothing forces them equal, and a mismatch means the realized
    value differs from the documented `SearchConfig` default with no yaml
    involved. This is the check the PR's "no behaviour change" claim rested on,
    made automatic.
    """
    realized = _realized_search({})
    default = SearchConfig()
    for field in sorted(_SENTINELS):
        if field == "simulations":
            # publisher parameter, not a config key -- `_reco_from` supplies 32.
            continue
        assert getattr(realized, field) == getattr(default, field), (
            f"with nothing set, SearchConfig.{field} realizes as "
            f"{getattr(realized, field)!r} but the dataclass default is "
            f"{getattr(default, field)!r} -- publisher and worker disagree."
        )
        assert type(getattr(realized, field)) is type(getattr(default, field))


def test_every_worker_reco_key_is_in_the_selfplay_allowlist() -> None:
    """Membership, not substring.

    The live-yaml validator is ALL-OR-NOTHING per section: a selfplay knob that
    lands in the wrong tuple (`_TRAIN_KEYS`, say) makes `_check_unknown` reject
    the WHOLE reload and take every other live experiment down with it. A
    substring search over the whole of `config_yaml.py` cannot see that -- it
    passes for `w_volatility`, which lives in `_TRAIN_KEYS` and which
    `flatten_run_config_defaults` rejects outright under `selfplay:`. Assert
    real membership in the exported selfplay tuple instead.
    """
    for field, key in sorted(_worker_reco_keys().items()):
        if key is None or field in _INTENTIONALLY_NOT_CONFIGURABLE:
            continue
        if key == "mcts_simulations":
            continue  # publisher parameter; `mcts_simulations` is allowlisted anyway
        assert key in SELFPLAY_CONFIG_KEYS, (
            f"{field} <- {key!r} is not in SELFPLAY_CONFIG_KEYS, so setting it "
            f"under `selfplay:` rejects the ENTIRE live reload"
        )


def test_every_search_config_key_forces_a_worker_restart() -> None:
    """A SearchConfig knob that does not force a restart is silently frozen.

    `_build_selfplay_configs` runs ONCE at session start, so a running
    `SelfplayState` keeps the `SearchConfig` it was built with. A mid-flight
    change to any key feeding it is therefore accepted, published, seen by the
    worker -- and ignored, while the ledger records a verdict for an experiment
    that never ran. Membership in `_RECO_RESTART_KEYS` is the ONLY thing that
    turns such a change into a restart.

    This is the exact bypass the rest of this file could not see. Verified by
    negative control 2026-07-28: deleting `"gumbel_target_batch"` from
    `_RECO_RESTART_KEYS` left all 16 other tests GREEN -- it was the sole
    escape of seven mutations, the other six each being caught by
    `test_a_yaml_value_actually_reaches_the_worker_search_config` (publisher
    drops the key / reads a wrong key / line commented out / emitted under
    another name / resolver replaced by a literal) or by
    `test_publisher_and_worker_agree_on_every_default` (default disagreement).

    Note the asymmetry with the allowlist test above: `SELFPLAY_CONFIG_KEYS`
    governs whether the yaml is ACCEPTED, `_RECO_RESTART_KEYS` governs whether
    it TAKES EFFECT. Passing the first and failing the second is precisely the
    "accepted and then silently ignored" shape.
    """
    restart_keys = set(WorkerSession._RECO_RESTART_KEYS)
    live_keys = set(WorkerSession._RECO_LIVE_KEYS)
    offenders: list[str] = []
    for field, key in sorted(_worker_reco_keys().items()):
        if key is None or field in _INTENTIONALLY_NOT_CONFIGURABLE:
            continue
        if key == "mcts_simulations":
            continue  # publisher parameter, not a per-worker SearchConfig knob
        if key in restart_keys:
            continue
        where = "_RECO_LIVE_KEYS" if key in live_keys else "NEITHER tuple"
        offenders.append(f"{field} <- {key!r} (in {where})")
    assert not offenders, (
        "SearchConfig keys that do not force a worker restart, so a live-yaml "
        "change to them is accepted and then silently ignored by every running "
        f"SelfplayState: {offenders}. Add them to _RECO_RESTART_KEYS, or make "
        "SearchConfig genuinely rebuildable mid-session and say so here."
    )


def test_no_worker_kwarg_is_silently_not_a_reco_key() -> None:
    """The decorative-literal escape must be an explicit list, not a skip.

    A kwarg whose value is not a literal `_resolve_reco(reco, "k", ...)` used to
    map to None and be skipped by the publish check while still counting as
    "reachable" -- so `gumbel_target_batch=0` passed both tests. Every such
    field now needs a written reason.
    """
    odd = {
        field for field, key in _worker_reco_keys().items()
        if key is None and field not in _WORKER_NOT_A_RECO_KEY
    }
    assert not odd, (
        f"worker SearchConfig kwargs not read from a named reco key: "
        f"{sorted(odd)}. A literal here is the matrix_weight_decay shape -- "
        f"wire it, or record why in _WORKER_NOT_A_RECO_KEY."
    )


def test_exempt_fields_are_not_published() -> None:
    """The exemption table is invalidated from BOTH sides.

    An exempt field that reappears in the reco means the reason is stale --
    which for the volatility knobs is not a documentation problem: publishing
    them makes a live yaml edit crash every selfplay worker.
    """
    reco = _reco_from({})
    leaked = sorted(k for k in _INTENTIONALLY_NOT_CONFIGURABLE if k in reco)
    assert not leaked, (
        f"published despite being exempt: {leaked}. Either delete the exemption "
        f"and give the field a sentinel, or stop publishing it."
    )


def test_the_gate_and_eval_matches_search_like_production() -> None:
    """The second construction site, which no by-name test could see.

    `trainable_config_ops._play_batch_kwargs` builds the SearchConfig for gate
    matches and eval games and used to omit BOTH C17 knobs, so those matches
    would have run the 56%-duplicate `vloss_weight=0` arm while the live yaml
    asked for 1 -- realized != configured, in the exact shape this file exists
    to prevent. Dormant today only because `gate_games: 0` / `eval_games: 0`,
    which is a config value and not a guarantee.
    """
    kwargs = _kwarg_names(_sole_call_site("chess_anti_engine/tune/trainable_config_ops.py"))
    declared = {f.name for f in dataclasses.fields(SearchConfig)}
    missing = declared - kwargs - set(_GATE_EVAL_NOT_SET)
    assert not missing, (
        f"gate/eval SearchConfig drops {sorted(missing)}, so those matches "
        f"search differently from the training path. Set them from TrialConfig "
        f"or record why in _GATE_EVAL_NOT_SET."
    )
    assert not (set(_GATE_EVAL_NOT_SET) & kwargs), "stale _GATE_EVAL_NOT_SET entry"


def test_every_configurable_value_survives_into_the_gate_eval_search_config() -> None:
    """...and the VALUES arrive, not just the kwarg names.

    ⚑ The by-name sibling above reads kwarg names off the AST, and **a constant
    satisfies a name**. That is not hypothetical here: this test used to cover
    only the two C17 knobs, so when `gumbel_policy_temp=tc.gumbel_policy_temp`
    was added to `_play_batch_kwargs`, pinning it to a literal `1.0` kept all
    233 tests green. The gate and eval matches would then have searched at a
    temperature production does not run, while the AST test reported the field
    as wired --- the same "accepted and then silently ignored" shape this file
    exists to catch, one construction site over.

    So drive it off `_SENTINELS`, which
    `test_the_sentinel_table_covers_exactly_the_configurable_fields` already
    forces to equal the non-exempt SearchConfig fields exactly. A field added
    later cannot slip through: it has no sentinel, so the coverage test fails
    first, and once it has one this test demands its value arrive. Narrowing
    this back to a hand-listed pair is how the gap reappeared.

    Each sentinel must also DIFFER from the dataclass default, or a field
    hard-wired to its own default would satisfy the check --- which is exactly
    what `gumbel_policy_temp=1.0` was.
    """
    skip = set(_GATE_EVAL_NOT_SET)
    covered = {f: v for f, v in _SENTINELS.items() if f not in skip}
    assert covered, "every field got skipped -- this test is asserting nothing"

    defaults = SearchConfig()
    degenerate = [
        field for field, (_key, sentinel) in covered.items()
        if getattr(defaults, field) == sentinel
    ]
    assert not degenerate, (
        f"sentinels equal to the SearchConfig default: {sorted(degenerate)}. "
        f"A field wired to a constant equal to its default would pass this test, "
        f"so pick a distinct sentinel rather than leaving a hole."
    )

    tc = TrialConfig.from_dict(dict(covered.values()))
    search = _play_batch_kwargs(tc)["search"]
    assert isinstance(search, SearchConfig)
    wrong = {
        field: (getattr(search, field), sentinel)
        for field, (_key, sentinel) in sorted(covered.items())
        if getattr(search, field) != sentinel
    }
    assert not wrong, (
        f"gate/eval SearchConfig does not carry the configured value for "
        f"{sorted(wrong)} (got vs want: {wrong}). The kwarg is present at the "
        f"call site, so the by-name test above is green -- the value is wired "
        f"to a constant, and those matches search differently from training."
    )


# ---------------------------------------------------------------------------
# The two exemption reasons that depend on facts outside this file
# ---------------------------------------------------------------------------


def test_the_fpu_exemption_depends_on_a_pinned_mcts_gumbel() -> None:
    """`fpu_*` are exempt because the Gumbel search never reads them.

    That is true only while the production config runs Gumbel: at full sim
    counts a non-gumbel `mcts` takes network_turn.py's PUCT branch, which DOES
    read `search.fpu_reduction` / `search.fpu_at_root` -- and the worker never
    resolves either from the reco, so a yaml edit would be accepted and ignored.
    Pin the dependency rather than leave the exemption inherited.
    """
    raw = yaml.safe_load(_PRODUCTION_YAML.read_text(encoding="utf-8"))
    assert raw["selfplay"]["mcts"] == "gumbel", (
        "production mcts is no longer gumbel -- the fpu_reduction/fpu_at_root "
        "exemption in _INTENTIONALLY_NOT_CONFIGURABLE no longer holds; the PUCT "
        "branch reads both and nothing can set them."
    )
    src = Path(inspect.getfile(network_turn)).read_text(encoding="utf-8")
    assert "search.fpu_reduction" in src, (
        "no PUCT reader of fpu_reduction left; re-derive the exemption reason"
    )


def test_volatility_search_is_rejected_at_config_load() -> None:
    """Loud-and-early beats both silent-ignore and fleet-crash.

    The knob is not publishable (see `_INTENTIONALLY_NOT_CONFIGURABLE`), so
    without this it would be back to accepted-and-ignored. `flatten_run_config_defaults`
    is what BOTH `run.py` at startup and `trainable_config_ops._reload_yaml_into_config`
    on every live reload call, so a bad live edit is rejected there instead of
    reaching a worker.
    """
    ok = flatten_run_config_defaults(
        {"selfplay": {"volatility_q_scale": 0.0, "volatility_fpu": 0.0,
                      "volatility_anchor": 0.05}},
    )
    assert ok["volatility_anchor"] == 0.05

    for key in ("volatility_q_scale", "volatility_fpu"):
        with pytest.raises(ValueError, match="volatility-aware search") as exc:
            flatten_run_config_defaults({"selfplay": {key: 0.5}})
        msg = str(exc.value)
        assert key in msg
        # The message must name the missing CAPABILITY and the classes lacking
        # it, or the next operator just re-adds the publish line.
        assert "evaluate_encoded_with_volatility" in msg
        for cls in ("MultiSlotInferenceClient", "ThreadedDispatcher",
                    "SlotInferenceClient", "AOTEvaluator"):
            assert cls in msg


def test_the_production_config_still_loads() -> None:
    """The new guard must not reject the config that is running right now.

    A validator that rejects the live yaml is not a guard, it is an outage: the
    all-or-nothing reload would drop every other live experiment on the floor.
    """
    raw = yaml.safe_load(_PRODUCTION_YAML.read_text(encoding="utf-8"))
    flat = flatten_run_config_defaults(raw)
    assert float(flat.get("volatility_q_scale", 0.0)) == 0.0
    assert float(flat.get("volatility_fpu", 0.0)) == 0.0


def test_the_recurring_defect_is_documented_where_someone_will_look() -> None:
    """The field comment must carry the measurement, not just the mechanism.

    A bare `gumbel_vloss_weight: int = 0` invites someone to 'clean up' an
    apparently unused knob. The measured duplicate rate quoted WITH its board
    count is what makes the field's existence self-justifying.
    """
    src = Path(inspect.getfile(SearchConfig)).read_text()
    idx = src.index("gumbel_target_batch")
    preamble = src[max(0, idx - 1800):idx]
    assert "C17" in preamble
    # The rate is meaningless without the board count it was measured at.
    assert "1024/boards-per-call" in preamble
