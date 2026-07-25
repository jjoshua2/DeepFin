"""Canonical accessors for per-iteration counter metrics in ``result.json``.

Two counters were renamed on 2026-07-24 because their old names claimed to be
totals when they only ever counted CURRENT-MODEL ("matching") games/positions:

    games_generated  -> matching_games
    positions_added  -> matching_positions

That gap is not cosmetic. Stale-model shards ARE ingested (``_process_shard``
calls ``_ingest_train_arrays`` before the ``model_sha`` check), and they
outnumber matching ones 4.5-6.5x, so a budget keyed off ``positions_added``
under-counts real ingest by that factor. `train_views_per_position` did exactly
that and delivered 0.46 true views while the config read 2.5 — over half of all
data never trained on once. The names invited the bug; these ones state the
denominator.

Readers go through here rather than indexing rows directly, because
``result.json`` files predating the rename are still on disk and still read:
``replay_exchange`` sizes cross-trial sharing from OTHER trials' historical
rows, and the monitoring scripts scan months of history. A blind rename would
make those silently read 0 — a silent degradation, which is the failure mode
this whole rename exists to remove.

Delete the legacy fallbacks once no ``result.json`` older than 2026-07-24 is
still consumed (all live trials rotated, and no archived run is being
re-analyzed). The tests pin the fallback behavior, so removing it is a
deliberate edit, not a silent drift.
"""

from __future__ import annotations

# new canonical key -> the pre-2026-07-24 name it replaced
_LEGACY_ALIASES: dict[str, str] = {
    "matching_games": "games_generated",
    "matching_positions": "positions_added",
}


def row_counter(row: dict, key: str, default: int = 0) -> int:
    """Read a counter metric, falling back to its pre-rename name.

    ``key`` must be a current canonical name. Rows written before the rename
    only carry the legacy name; rows written after carry only the new one.
    """
    val = row.get(key)
    if val is None:
        legacy = _LEGACY_ALIASES.get(key)
        if legacy is not None:
            val = row.get(legacy)
    if val is None:
        return int(default)
    try:
        return int(val)
    except (TypeError, ValueError):
        return int(default)


def row_counter_opt(row: dict, key: str) -> int | None:
    """Like :func:`row_counter` but distinguishes "absent" from a real zero.

    Callers that treat a missing metric as "cannot check" (rather than as 0)
    need this — reporting a missing metric as agreement is how the views bug
    stayed invisible.
    """
    val = row.get(key)
    if val is None:
        legacy = _LEGACY_ALIASES.get(key)
        if legacy is not None:
            val = row.get(legacy)
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None
