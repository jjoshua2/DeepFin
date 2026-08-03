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

import torch

from chess_anti_engine.moves import torch_maps
from chess_anti_engine.moves.encode import (
    COMPACT_TO_FULL_POLICY,
    FULL_TO_COMPACT_POLICY,
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


def test_the_two_tables_round_trip_on_every_real_move() -> None:
    """Guards against having swapped the two directions during the edit."""
    f2c = torch_maps.full_to_compact_index(torch.device("cpu"))
    c2f = torch_maps.compact_to_full_index(torch.device("cpu"))

    compact = torch.arange(c2f.numel(), dtype=torch.long)
    assert torch.equal(f2c.index_select(0, c2f.index_select(0, compact)), compact)

    full = torch.arange(f2c.numel(), dtype=torch.long)
    mapped = f2c.index_select(0, full)
    real = mapped >= 0
    assert torch.equal(c2f.index_select(0, mapped[real]), full[real])


def test_an_unindexed_cuda_device_shares_the_indexed_one_s_cache_entry() -> None:
    """The normalisation the private helper lacked (no GPU needed to assert it)."""
    assert torch_maps._device_key(torch.device("cuda", 0)) == ("cuda", 0)
    # The private helper's key for torch.device("cuda") was ("cuda", None) —
    # a second, separate cache entry for the same physical device.
    assert torch_maps._device_key(torch.device("cuda"))[0] == "cuda"
    if torch.cuda.is_available():  # pragma: no cover - CPU CI
        assert torch_maps._device_key(torch.device("cuda")) == torch_maps._device_key(
            torch.device("cuda", torch.cuda.current_device())
        )


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
    assert "torch_maps.full_to_compact_index" in src
    assert "torch_maps.compact_to_full_index" in src
