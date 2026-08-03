"""A mixed holdout must FAIL, not agree with itself.

`MODEL_OPT_AUDIT.md` M4-3. `ArrayReplayBuffer._gather_rows` merged chunks by
allocating each output field from whichever chunk happened to be the prototype
and scattering the rest in. For the scalar encoding-identity markers that had
two consequences, both silent:

* a `<U4` prototype (`"true"`) TRUNCATED the other chunk's `"false"` to
  `"fals"`, which `history_rep_fix_from_arrays` reads as False -- so a chunk
  that had the fix ON came back with it off;
* the sidecar round trip then promoted row 0's value to the whole set, so a
  holdout holding both encodings saved and reloaded as uniformly one of them.

The set this happens on is the best-model ruler's own. `history_rep_fix`
changes the repetition planes under an UNCHANGED encoding name, so nothing else
can say the ruler is scoring two encodings at once, and `samples_to_arrays`
already refuses a mixed batch "so the buffer's scalar-metadata merge can
hard-fail on mixed chunks" -- the merge just never let it.

Reachability, measured: `checkpoint_000476/477/478`'s live `holdout.npz` all
carry `_history_rep_fix` uniformly `"true"` and `_input_history_encoding`
uniformly `"lc0_root_legacy_meta"`, so the guard cannot fire on a resume from
today's state. It is armed for the next encoding flip, which is exactly when
the old behaviour would have mis-measured the ruler.
"""

from __future__ import annotations

import numpy as np
import pytest

from chess_anti_engine.replay import buffer as buffer_mod
from chess_anti_engine.replay.buffer import ArrayReplayBuffer
from chess_anti_engine.replay.shard import history_rep_fix_from_arrays

PLANES = 8
POLICY = 1858


def _arrays(n: int, *, first_row: int = 0, **markers: str) -> dict:
    rows = np.arange(first_row, first_row + n, dtype=np.int64)
    pol = np.zeros((n, POLICY), dtype=np.float32)
    pol[np.arange(n), rows % POLICY] = 1.0
    x = np.zeros((n, PLANES, 8, 8), dtype=np.float32)
    x[:, 0, 0, 0] = rows.astype(np.float32)
    out: dict = {
        "x": x,
        "policy_target": pol,
        "wdl_target": (rows % 3).astype(np.int8),
        "priority": rows.astype(np.float32) + 1.0,
        "has_policy": np.ones((n,), dtype=np.uint8),
    }
    for key, value in markers.items():
        out[f"_{key}"] = np.asarray(value)
    return out


def _buffer(*chunks: dict) -> ArrayReplayBuffer:
    buf = ArrayReplayBuffer(1024, rng=np.random.default_rng(0))
    for chunk in chunks:
        buf.add_many_arrays(chunk)
    return buf


# --- the negative control: mixed markers must refuse ------------------------


def test_a_mixed_rep_fix_holdout_refuses_to_export() -> None:
    buf = _buffer(
        _arrays(4, history_rep_fix="true"),
        _arrays(4, first_row=4, history_rep_fix="false"),
    )

    with pytest.raises(ValueError, match=r"mixed replay metadata '?_history_rep_fix"):
        buf.export_arrays()


def test_a_mixed_input_history_encoding_refuses_to_export() -> None:
    buf = _buffer(
        _arrays(4, input_history_encoding="lc0_root"),
        _arrays(4, first_row=4, input_history_encoding="legacy"),
    )

    with pytest.raises(ValueError, match=r"mixed replay metadata '?_input_history_encoding"):
        buf.export_arrays()


def test_a_mixed_policy_encoding_refuses_at_the_guard() -> None:
    # Exercised on the guard directly, not through `add_many_arrays`: two
    # different `_policy_encoding` names imply two different policy WIDTHS, and
    # ingest validation rejects the mismatched shard before a buffer can hold
    # both. The marker is covered anyway so the three identity markers are one
    # list rather than two rules with a carve-out nobody re-derives.
    chunks = [
        ({"_policy_encoding": np.asarray("lc0_1858")}, np.zeros(1, dtype=bool)),
        ({"_policy_encoding": np.asarray("az_4672")}, np.zeros(1, dtype=bool)),
    ]

    with pytest.raises(ValueError, match=r"mixed replay metadata '?_policy_encoding"):
        buffer_mod._refuse_mixed_identity_markers(chunks)


def test_an_absent_rep_fix_marker_counts_as_off() -> None:
    # shard.py materializes the marker on every write path and documents that a
    # missing one "provably means the flag was off", so absent-beside-true is a
    # MIXED set. Treating absence as unknown would leave the exact case the
    # 2026-06-17 flip produced (old chunks written before the marker existed)
    # unguarded.
    buf = _buffer(_arrays(4, history_rep_fix="true"), _arrays(4, first_row=4))

    with pytest.raises(ValueError, match=r"mixed replay metadata '?_history_rep_fix"):
        buf.export_arrays()


# --- and must not fire on anything the reader calls identical ---------------


def test_a_uniform_holdout_still_exports() -> None:
    buf = _buffer(
        _arrays(4, history_rep_fix="true", input_history_encoding="lc0_root"),
        _arrays(4, first_row=4, history_rep_fix="true", input_history_encoding="lc0_root"),
    )

    out = buf.export_arrays()

    assert [int(v) for v in np.asarray(out["x"])[:, 0, 0, 0]] == list(range(8))
    assert history_rep_fix_from_arrays(out) is True


def test_the_guard_uses_the_readers_own_predicate_for_spelling() -> None:
    # `history_rep_fix_from_arrays` strips and lowercases before comparing, so
    # "TRUE" and "true" are the same set to every consumer. A guard that fired
    # on that difference would be a guard on formatting, not on identity.
    #
    # This is also where the dtype widening earns its keep, and why it is not
    # made redundant by the guard above. The guard deliberately lets these two
    # chunks through, and the FIRST one's `<U4` used to size the output, so the
    # second's `" TRUE "` was stored as `" TRU"` -- a value no reader resolves
    # to True. `history_rep_fix_from_arrays` looks only at element 0, so
    # nothing downstream would ever have reported it: assert the whole column.
    buf = _buffer(
        _arrays(4, history_rep_fix="true"),
        _arrays(4, first_row=4, history_rep_fix=" TRUE "),
    )

    out = buf.export_arrays()

    marker = np.asarray(out["_history_rep_fix"]).reshape(-1)
    assert marker.shape == (8,)
    assert {str(v).strip().lower() for v in marker} == {"true"}
    assert history_rep_fix_from_arrays(out) is True


def test_a_single_chunk_is_never_mixed() -> None:
    buf = _buffer(_arrays(6, history_rep_fix="false"))

    out = buf.export_arrays()

    assert history_rep_fix_from_arrays(out) is False


# --- the truncation half of M4-3 --------------------------------------------


def test_numeric_field_dtypes_are_unchanged_by_the_widening() -> None:
    # `np.result_type` must be a no-op for everything the shard schema fixes,
    # or the merge would silently upcast payload arrays and change memory use
    # on the training path.
    buf = _buffer(_arrays(4), _arrays(4, first_row=4))

    out = buf.export_arrays()

    assert out["x"].dtype == np.float32
    assert out["wdl_target"].dtype == np.int8
    assert out["priority"].dtype == np.float32
    assert out["has_policy"].dtype == np.uint8
