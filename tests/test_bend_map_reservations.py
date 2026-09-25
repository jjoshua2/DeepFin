"""Reserved IDs are provisional; failed caller preparation must not publish them."""
import pytest

from native.bend_engine.u64_map_probe import registry as r
from native.bend_engine.u64_map_probe.run_probe import MASK


def test_abort_leaves_no_binding_and_reuses_candidate_for_a_different_key() -> None:
    case = r.Case('retry', 2, 7, [('a', 1), ('g', 1), ('c', 2), ('g', 1), ('g', 2)])
    assert r.expected(case) == (
        'begin 2 7\na 0 1 reserved 7 aborted size 0 next 7\ng 0 1 missing size 0 next 7\n'
        'c 0 2 reserved 7 assigned 7 size 1 next 8\ng 0 1 missing size 1 next 8\n'
        'g 0 2 value 7 size 1 next 8\nend size 1 next 8\n')


def test_existing_key_never_reserves_or_consumes_an_id() -> None:
    case = r.Case('known', 1, 0, [('r', 1), ('a', 1), ('c', 1)])
    assert r.expected(case) == (
        'begin 1 0\nr 0 1 assigned 0 size 1 next 1\na 0 1 known 0 size 1 next 1\n'
        'c 0 1 known 0 size 1 next 1\nend size 1 next 1\n')


def test_full_table_never_hands_out_a_reservation() -> None:
    case = r.Case('full', 1, 0, [('c', 0), ('c', 1), ('a', 1), ('g', 0)])
    assert r.expected(case) == (
        'begin 1 0\nc 0 0 reserved 0 assigned 0 size 1 next 1\nc 0 1 full size 1 next 1\n'
        'a 0 1 full size 1 next 1\ng 0 0 value 0 size 1 next 1\nend size 1 next 1\n')


def test_abort_final_u32_id_does_not_exhaust_it() -> None:
    case = r.Case('last', 3, MASK, [('a', 1), ('g', 1), ('c', 2), ('a', 3), ('c', 2)])
    assert r.expected(case) == (
        f'begin 3 {MASK}\na 0 1 reserved {MASK} aborted size 0 next {MASK}\n'
        f'g 0 1 missing size 0 next {MASK}\nc 0 2 reserved {MASK} assigned {MASK} size 1 next none\n'
        f'a 0 3 exhausted size 1 next none\nc 0 2 known {MASK} size 1 next none\nend size 1 next none\n')


def test_abort_does_not_disturb_prior_commits() -> None:
    case = r.Case('keep', 3, 10, [('c', 0), ('a', 1), ('a', 1), ('g', 0), ('r', 2)])
    assert r.expected(case).endswith(
        'a 0 1 reserved 11 aborted size 1 next 11\ng 0 0 value 10 size 1 next 11\n'
        'r 0 2 assigned 11 size 2 next 12\nend size 2 next 12\n')


@pytest.mark.parametrize('first', [0, 1, MASK - 1, MASK])
def test_immediate_and_committed_interning_have_identical_final_state(first: int) -> None:
    keys = [1 << 63, 0, 1 << 63, MASK, 0, (1 << 64) - 1]
    for bits in (1, 3):
        immediate = r.Case('instant', bits, first, [('r', k) for k in keys] + [('g', k) for k in keys])
        staged = r.Case('staged', bits, first, [('c', k) for k in keys] + [('g', k) for k in keys])
        def readbacks(case: r.Case) -> list[str]:
            return [s for s in r.expected(case).splitlines() if s.startswith(('g ', 'end '))]
        assert readbacks(immediate) == readbacks(staged)


@pytest.mark.parametrize('keys', [[], [0] * 521, [-1], [1 << 64], [True]])
def test_invalid_reservation_replay(keys: list[int]) -> None:
    with pytest.raises(ValueError, match=r'budget|operation'):
        r.reservation_case('bad', keys)


def test_replay_aborts_before_retry_and_checks_first_seen_bindings() -> None:
    keys = [1 << 63, 0, 1 << 63]
    case = r.reservation_case('walk', keys)
    assert case.ops[:4] == [('a', keys[0]), ('g', keys[0]), ('c', keys[0]), ('g', keys[0])]
    text = r.expected(case)
    assert 'a 2147483648 0 known 0 size 2 next 2\n' in text
    assert text.endswith('g 2147483648 0 value 0 size 2 next 2\ng 0 0 value 1 size 2 next 2\nend size 2 next 2\n')


@pytest.mark.parametrize(('old', 'new'), [('reserved 7', 'reserved 8'), ('aborted', 'assigned 7'),
                                        ('size 0 next 7', 'size 1 next 7'), ('size 0 next 7', 'size 0 next 8'),
                                        ('c 0 2', 'c 0 1'), ('assigned 7', 'assigned 8')])
def test_corrupt_reservation_results_rejected(old: str, new: str) -> None:
    case = r.Case('corrupt', 2, 7, [('a', 1), ('c', 2), ('g', 2)])
    text = r.expected(case)
    assert old in text
    with pytest.raises(ValueError, match='trace differs'):
        r.verify(text.replace(old, new, 1), case)


def test_reservation_fixtures_are_separate_and_reproducible() -> None:
    cases = r.reservation_fixtures()
    assert cases == r.reservation_fixtures()
    assert len(cases) == len({c.name for c in cases}) == 11
    assert len(r.fixtures()) == 35  # Previous cases remain intact.
    assert all(len(r.encode(c)) <= 100_000 for c in cases)
