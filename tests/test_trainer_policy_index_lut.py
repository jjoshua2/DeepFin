"""The trainer must use torch_maps' device cache, not a private copy of it.

Play-path code audit 2026-08-03, F11. ``train/trainer.py`` defined
``_policy_index_lut`` / ``_policy_index_lut_for``: a second, module-private
``lru_cache`` over ``COMPACT_TO_FULL_POLICY`` / ``FULL_TO_COMPACT_POLICY``,
which is precisely what ``moves/torch_maps.py`` exists to prevent (CLAUDE.md:
"Use the shared device-cached lookups in ``moves/torch_maps.py`` — don't add
per-module ``lru_cache`` copies"). It was also strictly worse: it keyed on
``target.device.index`` raw, without ``torch_maps._device_key``'s cuda-index
normalisation, so ``torch.device("cuda")`` and ``torch.device("cuda", 0)``
allocated two separate copies of both tables.

This touches the train path, so the swap has to be VALUE-IDENTICAL, not merely
equivalent in intent. The first test reproduces the old helper's exact body and
compares tensors element-by-element; the second is the normalisation bug the
swap fixes.
"""
from __future__ import annotations

import inspect

import pytest
import torch

from chess_anti_engine.moves import torch_maps
from chess_anti_engine.moves.encode import (
    COMPACT_POLICY_SIZE,
    COMPACT_TO_FULL_POLICY,
    FULL_TO_COMPACT_POLICY,
    POLICY_SIZE,
)
from chess_anti_engine.train import trainer as trainer_mod


def _old_helper(values, device_type: str, device_index: int | None) -> torch.Tensor:
    """Verbatim body of the deleted ``_policy_index_lut`` (pre-fix `main`)."""
    device = (
        torch.device(device_type) if device_index is None
        else torch.device(device_type, device_index)
    )
    return torch.as_tensor(values, dtype=torch.long, device=device)


def test_the_shared_tables_are_value_identical_to_the_deleted_private_ones() -> None:
    device = torch.device("cpu")

    f2c_old = _old_helper(FULL_TO_COMPACT_POLICY, "cpu", None)
    f2c_new = torch_maps.full_to_compact_index(device)
    assert f2c_new.dtype == f2c_old.dtype == torch.long
    assert f2c_new.device == f2c_old.device
    assert f2c_new.shape == f2c_old.shape
    assert torch.equal(f2c_new, f2c_old)

    c2f_old = _old_helper(COMPACT_TO_FULL_POLICY, "cpu", None)
    c2f_new = torch_maps.compact_to_full_index(device)
    assert c2f_new.dtype == c2f_old.dtype == torch.long
    assert c2f_new.device == c2f_old.device
    assert c2f_new.shape == c2f_old.shape
    assert torch.equal(c2f_new, c2f_old)


def test_the_two_shared_tables_round_trip_on_every_real_move() -> None:
    """Guards against the two SHARED TABLES having been swapped.

    Deliberately narrow, and named for what it actually covers. It does NOT
    see which table the trainer calls for which width pair — a reviewer swapped
    the two branch bodies in `_align_index` (both symbol names still present)
    and every test in this file passed. That hole is closed by construction
    now: the width dispatch lives once, in
    ``torch_maps.policy_index_remap_table``, and the test below asserts it.
    """
    f2c = torch_maps.full_to_compact_index(torch.device("cpu"))
    c2f = torch_maps.compact_to_full_index(torch.device("cpu"))

    compact = torch.arange(c2f.numel(), dtype=torch.long)
    assert torch.equal(f2c.index_select(0, c2f.index_select(0, compact)), compact)

    full = torch.arange(f2c.numel(), dtype=torch.long)
    mapped = f2c.index_select(0, full)
    real = mapped >= 0
    assert torch.equal(c2f.index_select(0, mapped[real]), full[real])


def test_the_width_pair_selects_the_table_that_actually_maps_it() -> None:
    """The binding the round-trip test cannot see, asserted semantically.

    `_align_index` no longer carries its own width dispatch — it calls
    `policy_index_remap_table`, so there is no longer a pair of branch bodies
    to transpose. This pins that single dispatch by BEHAVIOUR, not by which
    symbol it names: map a known full-4672 index through the table the
    (full -> compact) pair returns and require the compact index that the
    encoding says it is, and conversely. Swapping the two returns inside
    `policy_index_remap_table` fails this.
    """
    cpu = torch.device("cpu")
    f2c_ref = torch_maps.full_to_compact_index(cpu)
    c2f_ref = torch_maps.compact_to_full_index(cpu)

    to_compact = torch_maps.policy_index_remap_table(POLICY_SIZE, COMPACT_POLICY_SIZE, cpu)
    to_full = torch_maps.policy_index_remap_table(COMPACT_POLICY_SIZE, POLICY_SIZE, cpu)
    assert to_compact is not None
    assert to_full is not None

    # Widths alone give it away: only the full->compact table is 4672 long.
    assert to_compact.numel() == POLICY_SIZE == f2c_ref.numel()
    assert to_full.numel() == COMPACT_POLICY_SIZE == c2f_ref.numel()

    # ...and the VALUES map the way the encoding says, on every real move.
    full_ids = torch.arange(POLICY_SIZE, dtype=torch.long)
    got_compact = to_compact.index_select(0, full_ids)
    real = got_compact >= 0
    assert real.any()
    assert torch.equal(to_full.index_select(0, got_compact[real]), full_ids[real]), (
        "the (full -> compact) width pair does not return the table that maps "
        "full indices to compact ones"
    )

    compact_ids = torch.arange(COMPACT_POLICY_SIZE, dtype=torch.long)
    assert torch.equal(
        to_compact.index_select(0, to_full.index_select(0, compact_ids)), compact_ids,
    )

    # Equal widths mean "no remap", not "identity table".
    assert torch_maps.policy_index_remap_table(POLICY_SIZE, POLICY_SIZE, cpu) is None
    with pytest.raises(ValueError, match="incompatible"):
        torch_maps.policy_index_remap_table(POLICY_SIZE, 7, cpu)


def test_align_index_is_value_identical_to_the_two_branch_version() -> None:
    """The refactor that closed the hole must not have moved the numbers.

    `_align_index` is a closure inside `Trainer._policy_accuracy_stats`, so it
    cannot be imported. This reconstructs BOTH the pre-refactor two-branch body
    and the post-refactor single-dispatch body and requires them to agree
    element-wise on the mapped indices AND the validity mask, over in-range
    ids, negatives, and out-of-range ids (the `clamp` path) in both directions.
    """
    cpu = torch.device("cpu")

    def _old(target: torch.Tensor, *, source_width: int, dst_width: int):
        if source_width == dst_width:
            return target, torch.ones_like(target, dtype=torch.bool)
        if source_width == POLICY_SIZE and dst_width == COMPACT_POLICY_SIZE:
            lut = torch_maps.full_to_compact_index(target.device)
            valid = (target >= 0) & (target < POLICY_SIZE)
            safe = target.clamp(0, POLICY_SIZE - 1).to(torch.long)
            mapped = lut.index_select(0, safe)
            return mapped, valid & (mapped >= 0)
        if source_width == COMPACT_POLICY_SIZE and dst_width == POLICY_SIZE:
            lut = torch_maps.compact_to_full_index(target.device)
            valid = (target >= 0) & (target < COMPACT_POLICY_SIZE)
            safe = target.clamp(0, COMPACT_POLICY_SIZE - 1).to(torch.long)
            mapped = lut.index_select(0, safe)
            return mapped, valid
        raise ValueError("width mismatch")

    def _new(target: torch.Tensor, *, source_width: int, dst_width: int):
        lut = torch_maps.policy_index_remap_table(source_width, dst_width, target.device)
        if lut is None:
            return target, torch.ones_like(target, dtype=torch.bool)
        valid = (target >= 0) & (target < source_width)
        safe = target.clamp(0, source_width - 1).to(torch.long)
        mapped = lut.index_select(0, safe)
        return mapped, valid & (mapped >= 0)

    cases = [
        (POLICY_SIZE, COMPACT_POLICY_SIZE),
        (COMPACT_POLICY_SIZE, POLICY_SIZE),
        (POLICY_SIZE, POLICY_SIZE),
        (COMPACT_POLICY_SIZE, COMPACT_POLICY_SIZE),
    ]
    for source_width, dst_width in cases:
        target = torch.cat([
            torch.arange(source_width, dtype=torch.long),       # every in-range id
            torch.tensor([-1, -7, source_width, source_width + 99], dtype=torch.long),
        ]).to(cpu)
        old_mapped, old_valid = _old(target, source_width=source_width, dst_width=dst_width)
        new_mapped, new_valid = _new(target, source_width=source_width, dst_width=dst_width)
        assert torch.equal(old_mapped, new_mapped), (source_width, dst_width)
        assert torch.equal(old_valid, new_valid), (source_width, dst_width)
        assert old_mapped.dtype == new_mapped.dtype

    # The `valid & (mapped >= 0)` term that the compact->full branch did not
    # carry is a no-op there, which is WHY unifying them was safe. Assert the
    # premise rather than trusting it.
    assert int(torch_maps.compact_to_full_index(cpu).min()) >= 0
    assert int(torch_maps.full_to_compact_index(cpu).min()) < 0


def test_the_trainer_no_longer_carries_its_own_width_dispatch() -> None:
    """The structural half of F-1: no second pair of branches to transpose."""
    src = inspect.getsource(trainer_mod)
    code = "\n".join(
        line for line in src.splitlines() if not line.strip().startswith("#")
    )
    assert "policy_index_remap_table" in code
    assert "torch_maps.full_to_compact_index" not in code
    assert "torch_maps.compact_to_full_index" not in code


def test_an_unindexed_cuda_device_shares_the_indexed_one_s_cache_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The double-cache this swap fixes — asserted on CPU CI, not skipped there.

    The first version of this test put the real claim behind
    ``if torch.cuda.is_available()`` and left one unconditional assertion
    (``_device_key(cuda)[0] == "cuda"``) that is trivially true. That is a gate
    that cannot fail on the CI which actually runs it — this repo's named
    anti-pattern. ``_device_key`` only consults ``torch.cuda.is_available`` and
    ``torch.cuda.current_device``, so faking both makes the real claim testable
    with no GPU.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    unindexed = torch_maps._device_key(torch.device("cuda"))
    indexed = torch_maps._device_key(torch.device("cuda", 0))
    assert unindexed == indexed == ("cuda", 0), (
        f"unindexed cuda keys to {unindexed}, indexed to {indexed}: two cache "
        "entries for one physical device"
    )

    # ...and that is exactly what the deleted private helper did NOT do: its key
    # was `(device.type, device.index)` raw, i.e. ("cuda", None) vs ("cuda", 0).
    def _old_key(device: torch.device) -> tuple[str, int | None]:
        return device.type, device.index

    assert _old_key(torch.device("cuda")) != _old_key(torch.device("cuda", 0))

    # A second device index must still key separately — the normalisation must
    # not collapse genuinely different GPUs.
    assert torch_maps._device_key(torch.device("cuda", 1)) == ("cuda", 1)


def test_the_private_lut_is_gone_from_the_trainer() -> None:
    """Fails on pre-fix `main`, where both helpers existed."""
    assert not hasattr(trainer_mod, "_policy_index_lut")
    assert not hasattr(trainer_mod, "_policy_index_lut_for")
    src = inspect.getsource(trainer_mod)
    code = "\n".join(
        line for line in src.splitlines() if not line.strip().startswith("#")
    )
    assert "_policy_index_lut" not in code
    assert "lru_cache" not in code, "no private device cache may come back"
    assert "torch_maps.policy_index_remap_table" in src
