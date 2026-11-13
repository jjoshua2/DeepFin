"""Raw/stored target joins for one independently verified finite training game."""
from __future__ import annotations

from typing import Any

from chess_anti_engine.verified_outcome import VerifiedOutcome, require_verified


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def capture_raw_targets(samples: list[Any], audit: Any, row_ids: list[str], *, proof: VerifiedOutcome,
                        game: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Capture real float32 targets before shard quantization; never upcast F16."""
    import numpy as np
    from chess_anti_engine.selfplay.finalize import _result_to_wdl

    verified = require_verified(proof, audit=audit)
    n = len(samples)
    require(n == len(row_ids) == audit.retained_samples == verified['retained_rows'], 'raw row count differs')
    require(row_ids == [verified['physical_game_id'] + ':' + str(i) for i in range(n)], 'raw physical row order')
    require(not samples or verified['resolved'] is True, 'censored game cannot supply rows')
    records = {r.ply_index: r for r in audit.records}
    raw: dict[str, Any] = {
        'physical_row_id': np.asarray(row_ids, dtype=np.str_),
        'policy_target_raw': np.zeros((n, 1858), dtype=np.float32),
        'outcome_wdl': np.zeros(n, dtype=np.int8),
        'pov_white': np.zeros(n, dtype=np.uint8),
        'search_wdl_raw': np.zeros((n, 3), dtype=np.float32),
        'has_search_raw': np.zeros(n, dtype=np.uint8),
        'categorical_raw': np.zeros((n, game.categorical_bins), dtype=np.float32),
        'has_categorical_raw': np.zeros(n, dtype=np.uint8),
    }
    for i, sample in enumerate(samples):
        require(sample.game_id == audit.game_id and sample.ply_index in records, 'raw record/sample join')
        record = records[sample.ply_index]
        require(record.has_policy and not record.is_sf_refute_opp and sample.has_policy and
                sample.is_network_turn is True and sample.is_selfplay is True, 'raw row not eligible neural selfplay')
        outcome = _result_to_wdl(verified['result'], pov_white=bool(record.pov_color))
        require(sample.wdl_target == outcome and outcome in (0, 1, 2), 'raw verified outcome/POV differs')
        raw['outcome_wdl'][i] = outcome
        raw['pov_white'][i] = record.pov_color
        for source, target, width, mask in (
            (sample.policy_target, 'policy_target_raw', 1858, None),
            (sample.search_wdl, 'search_wdl_raw', 3, 'has_search_raw'),
            (sample.categorical_target, 'categorical_raw', game.categorical_bins, 'has_categorical_raw'),
        ):
            if source is None:
                require(mask is not None, 'main policy absent')
                continue
            value = np.asarray(source)
            require(value.dtype == np.dtype('float32') and value.shape == (width,) and
                    bool(np.isfinite(value).all()) and bool((value >= 0).all()) and
                    abs(float(value.sum()) - 1) < 1e-5, 'raw target not original float32 simplex: ' + target)
            raw[target][i] = value
            if mask is not None:
                raw[mask][i] = 1
    metadata = {'schema': 'CORE_RAW_TARGETS_V1', 'rows': n, 'raw_targets_verified': bool(n),
                'verified_outcome': verified, 'verified_outcome_sha256': proof.proof_sha256,
                'policy_source': 'actual stored Gumbel improved policy or verified TB policy override',
                'outcome_source': 'fresh strict natural/TB result in recorded player-to-move POV',
                'search_and_categorical': 'diagnostic original float32 fields; no baseline auxiliary objective implied'}
    return raw, metadata


def validate_stored_targets(raw: dict[str, Any], arrays: dict[str, Any], *, proof: VerifiedOutcome,
                            metadata: dict[str, Any], game_id: int, plies: list[int]) -> None:
    import numpy as np

    require_verified(proof, metadata=metadata)
    n = len(raw['physical_row_id'])
    require(n == len(plies) == int(arrays['x'].shape[0]), 'raw/storage rows differ')
    require(np.array_equal(arrays['wdl_target'], raw['outcome_wdl']), 'stored hard outcome differs')
    for key, expected in (('game_id', game_id), ('has_game_id', 1), ('has_ply_index', 1),
                          ('has_policy', 1), ('is_network_turn', 1), ('has_is_network_turn', 1),
                          ('is_selfplay', 1), ('has_is_selfplay', 1), ('has_legal_mask', 1)):
        require(key in arrays and np.asarray(arrays[key]).shape == (n,) and
                bool(np.all(np.asarray(arrays[key]) == expected)), 'stored identity/availability differs: ' + key)
    require(np.array_equal(arrays['ply_index'], np.asarray(plies)), 'stored physical ply differs')
    if 'is_sf_refute_opp' in arrays:
        require(not np.asarray(arrays['is_sf_refute_opp']).any(), 'stored SF-refute row')
    for source, stored, availability, stored_mask in (
        ('policy_target_raw', 'policy_target', None, None),
        ('search_wdl_raw', 'search_wdl', 'has_search_raw', 'has_search_wdl'),
        ('categorical_raw', 'categorical_target', 'has_categorical_raw', 'has_categorical'),
    ):
        if availability is not None and not np.asarray(raw[availability]).any() and stored not in arrays:
            require(stored_mask in arrays and np.array_equal(arrays[stored_mask], raw[availability]),
                    'absent stored target has nonzero/missing availability: ' + stored)
            continue
        require(stored in arrays, 'stored target absent: ' + stored)
        expected = raw[source].astype(np.float16)
        actual = np.asarray(arrays[stored])
        require(actual.dtype == expected.dtype and actual.shape == expected.shape and
                actual.tobytes() == expected.tobytes(), 'stored target quantization differs: ' + stored)
        if availability is not None:
            require(stored_mask in arrays and np.array_equal(arrays[stored_mask], raw[availability]),
                    'stored target availability differs: ' + stored)
