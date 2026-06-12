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
"""
from __future__ import annotations

import logging

_LOG = logging.getLogger(__name__)
_current: bool | None = None


def apply(enabled: bool) -> None:
    """Set the repetition-fix flag in every encoder extension that exposes it.

    Idempotent and cheap; safe to call once per batch. Missing setters (e.g. an
    extension built before this change) are skipped with a one-time warning so a
    stale build fails visibly rather than silently encoding the wrong planes.
    """
    global _current
    enabled = bool(enabled)
    if _current == enabled:
        return
    applied = 0
    for mod_name in ("chess_anti_engine.encoding._lc0_ext", "chess_anti_engine.mcts._mcts_tree"):
        try:
            mod = __import__(mod_name, fromlist=["set_history_rep_fix"])
        except Exception:  # noqa: BLE001 - extension may be unavailable in some installs
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
