"""Process-wide toggle for the lc0-root repetition-plane fix (gated candidate).

The C encoders normally reconstruct per-history-slot repetition planes from the
``hash_stack``, which is cleared on every irreversible move and therefore
under-reports repetitions older than the kept window. The fix records a
per-slot repetition flag at push time (full look-back valid then) and reads it
at encode time, making the C encoding bit-identical to python-chess.

The flag lives as a global in each compiled extension (``_lc0_ext`` for the
single-board encode path, ``_mcts_tree`` for the batch encoders), so it must be
set in every loaded module. ``apply`` does that idempotently; selfplay calls it
from ``play_batch`` based on ``GameConfig.history_rep_fix`` (default off), and
``build_model`` applies the checkpoint's value for eval/loading paths.

Ordering contract: apply BEFORE constructing or pushing boards. With the flag
on, per-slot repetition flags are recorded at push/construction time; that
recording is skipped while the flag is off (zero cost on the default path) and
is not retroactively recomputed, so a board built before a flip would encode
all-clear history repetition planes. Every current call site (batch start,
model build) precedes board construction.

That contract used to be prose with nothing enforcing it. A board pushed across
a flip encodes repetition planes matching **neither** regime — measured, with
the flag alternated per ply on one live ``CBoard``: slots ``[1,1,1,1,1,1,1,0]``
under either clean regime, ``[1,0,1,0,1,0,1,0]`` across the flip (audit E3,
``scratchpad/code_audit_20260803/enc_repfix_midgame_flip.py``). ``CBoard`` is a
C type with no ``__dict__`` and no weakref support, so the flag cannot be
stamped per board from Python; the guard therefore sits on the flip itself.

:func:`apply` REFUSES a genuine flip (a change of value after the flag has been
set) unless the caller passes ``boards_discarded=True``, which asserts that no
``CBoard`` pushed under the previous value survives the call. Setting the flag
for the first time, and re-applying the value already in force, are unguarded.
The three production callers that legitimately flip pass the keyword and record
why it holds; anything new fails loudly instead of silently mis-encoding.
"""
from __future__ import annotations

import logging

_LOG = logging.getLogger(__name__)
_current: bool | None = None


class RepFixFlipError(RuntimeError):
    """A live-board-unsafe flip of the process-global ``history_rep_fix`` flag."""


def current() -> bool | None:
    """The flag value in force, or ``None`` if it has never been set."""
    return _current


def apply(enabled: bool, *, boards_discarded: bool = False) -> None:
    """Set the repetition-fix flag in every encoder extension that exposes it.

    Idempotent and cheap; safe to call once per batch. Missing setters (e.g. an
    extension built before this change) are skipped with a one-time warning so a
    stale build fails visibly rather than silently encoding the wrong planes.

    ``boards_discarded=True`` certifies that no ``CBoard`` constructed or pushed
    under the previous value is still alive — required for any call that CHANGES
    the flag, because per-slot repetition flags are recorded at push time and
    never recomputed. Raises :class:`RepFixFlipError` otherwise.
    """
    global _current
    enabled = bool(enabled)
    if _current == enabled:
        return
    if _current is not None and not boards_discarded:
        raise RepFixFlipError(
            f"history_rep_fix flip {_current} -> {enabled} without "
            "boards_discarded=True: per-slot repetition flags are recorded at "
            "push time and are not recomputed, so any CBoard that survives this "
            "flip encodes repetition planes matching NEITHER regime (audit E3). "
            "Apply the flag before constructing boards, or pass "
            "boards_discarded=True if no board built under the old value is "
            "still alive."
        )
    applied = 0
    for mod_name in ("chess_anti_engine.encoding._lc0_ext", "chess_anti_engine.mcts._mcts_tree"):
        try:
            mod = __import__(mod_name, fromlist=["set_history_rep_fix"])
        except Exception:
            continue
        setter = getattr(mod, "set_history_rep_fix", None)
        if setter is None:
            _LOG.warning(
                "%s lacks set_history_rep_fix; rebuild the C extensions to use "
                "history_rep_fix (encoding may diverge from the configured value)",
                mod_name,
            )
            continue
        setter(enabled)
        applied += 1
    _current = enabled
    if enabled and applied:
        _LOG.info("history_rep_fix enabled in %d encoder module(s)", applied)
