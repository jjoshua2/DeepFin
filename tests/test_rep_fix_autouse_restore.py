"""The conftest guard on the process-global ``history_rep_fix`` flag works.

``tests/conftest.py``'s ``pytest_runtest_protocol`` hookwrapper is implicit, so
nothing in the suite requests it and nothing would notice if it stopped doing
anything — the classic shape of a guard that is accepted and then silently
inert. These tests run a REAL nested pytest session (``pytest.Pytester``,
in-process, so it shares this interpreter's globals) whose inner tests change the
flag, and then assert on the flag THIS process is left holding.

⚑ The inner conftest registers the REAL ``tests/conftest.py`` as a plugin rather
than restating any of it. A copied guard would keep passing while the real one
rotted — and naming the mechanism instead of the file would leave these tests
unable to judge a change of mechanism, which is exactly the change that produced
this file's newest test.

⚑ Irony guard: the guard under test also wraps these tests. Each one therefore
snapshots the enclosing state itself, restores it in a ``finally``, and asserts
the final state explicitly — so a broken guard cannot be what makes this file
look clean, and this file cannot leak the state it is testing.
"""
from __future__ import annotations

import pytest

from chess_anti_engine.encoding import rep_fix

pytest_plugins = ["pytester"]


# ⚑ Register the REAL conftest as a plugin rather than importing one name out of
# it. Two reasons, and the second is the load-bearing one: a copied guard would
# keep passing while the real one rotted, and naming the mechanism (`the hook`,
# `the autouse fixture`) would make these tests unable to judge a change of
# mechanism — the exact change this file exists to pin. As a plugin, whatever
# `tests/conftest.py` uses to guard the flag is what the nested session runs.
_INNER_CONFTEST = '''\
"""Run the nested session under the REAL tests/conftest.py, whatever shape it has."""
pytest_plugins = ["tests.conftest"]
'''

# The Codex review scenario, verbatim in shape: a MODULE-scoped fixture that
# changes the flag during the FIRST test's setup — i.e. before any
# function-scoped fixture exists to snapshot it. `tests/test_param_count.py`'s
# `production_model` is the live instance (it builds the production config,
# whose `history_rep_fix` is true).
_INNER_MODULE_SCOPED_FLIPPER = '''\
from pathlib import Path

import pytest

from chess_anti_engine.encoding import rep_fix


@pytest.fixture(scope="module")
def flipping_module_fixture():
    rep_fix.apply(True, boards_discarded=True)
    return "built"


def test_a_first_test_of_the_module(flipping_module_fixture):
    assert rep_fix.current() is True


def test_b_second_test_of_the_module(flipping_module_fixture):
    # The fixture VALUE is still cached; only the process flag was rewound.
    assert flipping_module_fixture == "built"
    Path(__file__).with_name("second_saw.txt").write_text(repr(rep_fix.current()))
'''

_INNER_OBSERVER = '''\
from pathlib import Path

from chess_anti_engine.encoding import rep_fix


def test_the_next_file_sees_the_baseline():
    Path(__file__).with_name("next_file_saw.txt").write_text(repr(rep_fix.current()))
'''

_INNER_FLIPPING_TEST = '''\
from pathlib import Path

from chess_anti_engine.encoding import rep_fix


def test_inner_flips_the_process_global_flag():
    flipped = not (rep_fix.current() or False)
    rep_fix.apply(flipped, boards_discarded=True)
    assert rep_fix.current() is flipped
    # Reported outward so the enclosing test can prove the flip really happened
    # instead of passing on an inner test that quietly did nothing.
    Path(__file__).with_name("flipped.txt").write_text(repr(flipped))
'''

_INNER_PASSIVE_TEST = '''\
from pathlib import Path

from chess_anti_engine.encoding import rep_fix


def test_inner_reads_the_flag_without_changing_it():
    Path(__file__).with_name("observed.txt").write_text(repr(rep_fix.current()))
'''


def _set_flag_state(value: bool | None) -> None:
    """Force the process flag to ``value``, ``None`` included.

    ``apply`` takes a ``bool`` and so cannot express "never set"; the two-step
    below is the same one the hook uses to restore that snapshot, and the
    same one ``tests/test_c_only_gumbel_knobs.py`` spells out inline. Kept here
    as a local helper rather than imported from the conftest on purpose: the
    restore path is what is under test, and a test must not steer itself with
    the code it is judging.
    """
    rep_fix.apply(bool(value), boards_discarded=True)
    if value is None:
        rep_fix._current = None


def test_a_flipped_bool_flag_is_restored_after_the_inner_test(
    pytester: pytest.Pytester,
) -> None:
    """The ordinary case: a test flips the flag, the next file must not inherit it.

    ``boards_discarded=True`` on the calls this file makes is true by
    construction — nothing here builds a ``CBoard``, so no board can straddle a
    flip (audit E3).
    """
    pytester.makeconftest(_INNER_CONFTEST)
    pytester.makepyfile(test_inner_flip=_INNER_FLIPPING_TEST)

    outer = rep_fix.current()
    # A KNOWN bool baseline, and deliberately ``True``: a restore that always
    # writes ``False`` has to be distinguishable from a correct one, and it
    # would be invisible if the baseline were ``False`` to begin with.
    _set_flag_state(True)
    try:
        result = pytester.runpytest_inprocess()
        result.assert_outcomes(passed=1)
        assert (pytester.path / "flipped.txt").read_text() == "False"
        assert rep_fix.current() is True
    finally:
        _set_flag_state(outer)
    assert rep_fix.current() is outer


def test_the_never_set_sentinel_is_restored_after_the_inner_test(
    pytester: pytest.Pytester,
) -> None:
    """The branch ``apply`` cannot express.

    Restoring a never-set snapshot as ``False`` would be wrong in a way nothing
    later crashes on: ``current()`` would report that somebody chose "off" when
    in fact nobody has chosen, and the next reader of that value — a test
    asserting on the default, or the next snapshot/restore pair — inherits the
    invention.
    """
    pytester.makeconftest(_INNER_CONFTEST)
    pytester.makepyfile(test_inner_flip=_INNER_FLIPPING_TEST)

    outer = rep_fix.current()
    _set_flag_state(None)
    try:
        assert rep_fix.current() is None
        result = pytester.runpytest_inprocess()
        result.assert_outcomes(passed=1)
        # never-set reads as falsey, so the inner flip lands on True.
        assert (pytester.path / "flipped.txt").read_text() == "True"
        assert rep_fix.current() is None
    finally:
        _set_flag_state(outer)
    assert rep_fix.current() is outer


def test_a_module_scoped_fixtures_flip_does_not_reach_the_next_file(
    pytester: pytest.Pytester,
) -> None:
    """The Codex finding: a broader-scoped fixture must not evade the snapshot.

    A module-scoped fixture is set up during the FIRST test's setup phase, which
    is BEFORE any function-scoped fixture — so the function-scoped autouse
    fixture this guard used to be snapshotted the ALREADY-MUTATED value and
    restored it faithfully, forever, into every later file. Reproduced here with
    two inner files: the first flips the flag from a module-scoped fixture, the
    second must still see the baseline.

    Also pins the deliberate consequence: the SECOND test of the flipping module
    sees the baseline too, because the flag is rewound after every protocol even
    though the fixture VALUE stays cached. No test in this suite relies on such
    persistence — checked against every higher-scoped fixture that builds a
    model; see the hook's docstring for the enumeration.
    """
    pytester.makeconftest(_INNER_CONFTEST)
    pytester.makepyfile(
        test_a_module_flipper=_INNER_MODULE_SCOPED_FLIPPER,
        test_z_observer=_INNER_OBSERVER,
    )

    outer = rep_fix.current()
    # `False`, so an inherited `True` is unmistakably the module fixture's.
    _set_flag_state(False)
    try:
        result = pytester.runpytest_inprocess()
        result.assert_outcomes(passed=3)
        assert (pytester.path / "next_file_saw.txt").read_text() == "False"
        assert (pytester.path / "second_saw.txt").read_text() == "False"
        assert rep_fix.current() is False
    finally:
        _set_flag_state(outer)
    assert rep_fix.current() is outer


def test_the_guard_makes_no_calls_when_the_test_left_the_flag_alone(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The common path must stay free.

    Every test in the suite pays this hook's teardown half, so "restores the
    flag" is only half the contract — the other half is that a test which never
    touched the flag causes no ``apply`` call at all, and therefore no writes
    into the compiled encoders' globals.
    """
    pytester.makeconftest(_INNER_CONFTEST)
    pytester.makepyfile(test_inner_passive=_INNER_PASSIVE_TEST)

    calls: list[bool] = []
    real_apply = rep_fix.apply

    def _recording_apply(enabled: bool, *, boards_discarded: bool = False) -> None:
        calls.append(bool(enabled))
        real_apply(enabled, boards_discarded=boards_discarded)

    outer = rep_fix.current()
    monkeypatch.setattr(rep_fix, "apply", _recording_apply)
    try:
        result = pytester.runpytest_inprocess()
        result.assert_outcomes(passed=1)
        # In-process, so the nested session sees this process's own flag.
        assert (pytester.path / "observed.txt").read_text() == repr(outer)
        assert calls == []
    finally:
        monkeypatch.undo()
        _set_flag_state(outer)
    assert rep_fix.current() is outer
