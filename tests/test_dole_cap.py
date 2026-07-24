"""Dole cap: bound seeded games/iter so the seed pool cannot crowd out
normal-opening selfplay (2026-07-24 — the uncapped dole reached 100%)."""

from __future__ import annotations

from chess_anti_engine.selfplay.opening import cap_dole_batch, rotate_slice


def _seeds(n: int) -> list[str]:
    return [f"fen{i}" for i in range(n)]


def test_rotate_slice_covers_every_item_over_a_cycle() -> None:
    items = _seeds(10)
    seen: set[str] = set()
    for it in range(5):  # k=2 -> 5 iterations to cover 10
        seen.update(rotate_slice(items, 2, it))
    assert seen == set(items), "rotation must not permanently starve the tail"


def test_rotate_slice_degenerate_cases() -> None:
    assert rotate_slice([], 5, 0) == []
    assert rotate_slice(_seeds(3), 0, 0) == []
    assert rotate_slice(_seeds(3), -1, 0) == []
    # k >= n returns everything, without duplicating.
    assert rotate_slice(_seeds(3), 3, 7) == _seeds(3)
    assert rotate_slice(_seeds(3), 99, 7) == _seeds(3)


def test_cap_disabled_is_identity() -> None:
    q, sf = _seeds(300), _seeds(200)
    out_q, out_sf = cap_dole_batch(q, sf, max_games=0, training_iteration=1)
    assert out_q is q
    assert out_sf is sf


def test_cap_not_binding_is_identity() -> None:
    q, sf = _seeds(30), _seeds(20)
    out_q, out_sf = cap_dole_batch(q, sf, max_games=100, training_iteration=1)
    assert out_q is q
    assert out_sf is sf


def test_cap_bounds_total_and_splits_proportionally() -> None:
    # 300 + 200 = 500 seeded games, capped to 220 (0.5 * 440 games_per_iter).
    q, sf = _seeds(300), _seeds(200)
    out_q, out_sf = cap_dole_batch(q, sf, max_games=220, training_iteration=0)
    assert len(out_q) + len(out_sf) == 220
    # 200/500 = 40% of the budget to the refute channel.
    assert len(out_sf) == 88
    assert len(out_q) == 132


def test_cap_rotates_across_iterations() -> None:
    q, sf = _seeds(300), _seeds(200)
    a_q, _ = cap_dole_batch(q, sf, max_games=220, training_iteration=0)
    b_q, _ = cap_dole_batch(q, sf, max_games=220, training_iteration=1)
    assert a_q != b_q, "consecutive iterations must serve different seeds"
    # Over enough iterations every seed is served at least once.
    seen: set[str] = set()
    for it in range(4):
        cq, csf = cap_dole_batch(q, sf, max_games=220, training_iteration=it)
        seen.update(cq)
        seen.update(csf)
    assert seen == set(q)


def test_cap_handles_one_empty_channel() -> None:
    q = _seeds(300)
    out_q, out_sf = cap_dole_batch(q, [], max_games=100, training_iteration=3)
    assert out_sf == []
    assert len(out_q) == 100


def test_cap_never_exceeds_budget_for_any_split() -> None:
    for nq in (0, 1, 17, 300):
        for nsf in (0, 1, 17, 200):
            for cap in (1, 7, 220):
                out_q, out_sf = cap_dole_batch(
                    _seeds(nq), _seeds(nsf), max_games=cap, training_iteration=2,
                )
                assert len(out_q) + len(out_sf) <= max(cap, 0) or nq + nsf <= cap
