"""Service-sample integrity, warmup exclusion and offline equal-work plans."""
from __future__ import annotations

import copy
import json

import pytest

from native.bend_engine.service_profile.profile import (
    cells, object_json, percentile95, validate_native, wave_plan,
)


def record(batch: int = 4, offset: int = 0) -> tuple[dict, dict]:
    identity = {'batch': batch, 'channels': 175, 'package_sha256': 'a' * 64,
                'checkpoint_sha256': 'b' * 64, 'input_sha256': 'c' * 64, 'reference_sha256': 'd' * 64}
    samples = [{'sweep': s, 'real_rows': 1 + (j + s + offset) % batch,
                'warmup': s < 1, 'service_ns': 10000 if s < 1 else 100 * (1 + (j + s + offset) % batch)}
               for s in range(4) for j in range(batch)]
    return {'schema': 'deepfin.native-service-samples.v1', 'status': 'passed', **identity,
            'device': 'cpu', 'dtype': 'float32', 'output_width': 1861, 'torch_threads': 2, 'interop_threads': 1,
            'warmups_per_size': 1, 'samples_per_size': 3, 'order_offset': offset,
            'model_open_ns': 1, 'panel_ns': sum(s['service_ns'] for s in samples) + 1,
            'max_logit_error': 0.0, 'atol': 2e-6, 'rtol': 2e-5, 'accepted_neural_rows': None,
            'useful_eps': None, 'samples': samples}, identity


def test_nominal_all_batches() -> None:
    for batch in (1, 2, 4, 8, 16):
        for offset in range(batch):
            r, identity = record(batch, offset)
            validate_native(r, identity, 1, 3, offset)


@pytest.mark.parametrize(('key', 'value'), [
    ('status', 'failed'), ('batch', True), ('batch', 8), ('channels', 146),
    ('output_width', 1858), ('torch_threads', 3), ('interop_threads', True),
    ('warmups_per_size', 0), ('samples_per_size', 4), ('order_offset', 1),
    ('device', 'cuda'), ('dtype', 'bfloat16'), ('accepted_neural_rows', 0), ('useful_eps', 1.0),
    ('model_open_ns', 0), ('panel_ns', 1), ('atol', 2e-5), ('rtol', 2e-4),
    ('max_logit_error', float('nan')), ('max_logit_error', -1), ('max_logit_error', True),
    ('checkpoint_sha256', 'f' * 64), ('input_sha256', 'bad'), ('reference_sha256', 'f' * 64),
    ('package_sha256', 'f' * 64),
])
def test_wrong_metadata_rejected(key: str, value: object) -> None:
    r, identity = record()
    r[key] = value
    with pytest.raises(ValueError, match=r'profile|configuration|identity|SHA256|search work|tolerance|error|integer|panel|matrix|order|CPU-F32'):
        validate_native(r, identity, 1, 3, 0)


@pytest.mark.parametrize('key', ['accepted_neural_rows', 'useful_eps', 'samples', 'package_sha256'])
def test_required_fields(key: str) -> None:
    r, identity = record()
    del r[key]
    with pytest.raises(ValueError, match=r'profile|configuration|identity|SHA256|search work|tolerance|error|integer|panel|matrix|order|CPU-F32'):
        validate_native(r, identity, 1, 3, 0)


@pytest.mark.parametrize(('key', 'value'), [('warmup', 1), ('warmup', False), ('sweep', True),
                                      ('sweep', 1), ('real_rows', True), ('real_rows', 2),
                                      ('service_ns', 0), ('service_ns', -1), ('service_ns', 3.5)])
def test_bad_sample(key: str, value: object) -> None:
    r, identity = record()
    r['samples'][0][key] = value
    with pytest.raises(ValueError, match=r'profile|configuration|identity|SHA256|search work|tolerance|error|integer|panel|matrix|order|CPU-F32'):
        validate_native(r, identity, 1, 3, 0)


def test_missing_duplicate_or_reordered_samples() -> None:
    for method in ('drop', 'duplicate', 'reorder'):
        r, identity = record()
        if method == 'drop':
            r['samples'].pop()
        elif method == 'duplicate':
            r['samples'].append(r['samples'][-1])
        else:
            r['samples'][0], r['samples'][1] = r['samples'][1], r['samples'][0]
        with pytest.raises(ValueError, match=r'matrix|order'):
            validate_native(r, identity, 1, 3, 0)


def test_warmup_not_counted_and_padding_not_useful() -> None:
    runs = []
    for offset in (0, 1):
        r, identity = record(offset=offset)
        runs.append({'identity': identity, 'native': r, 'warmups_per_size': 1, 'samples_per_size': 3})
    rows = cells(runs)
    one = rows[0]
    assert one['median_ns'] == one['p95_ns'] == 100
    assert one['forward_calls'] == one['executed_real_rows'] == 6
    assert one['physical_rows'] == 24
    assert one['padded_rows'] == 18
    assert one['executed_rows_per_service_second'] == 10000000
    corrupted = copy.deepcopy(runs)
    corrupted[-1]['identity']['checkpoint_sha256'] = 'e' * 64
    with pytest.raises(ValueError, match='incompatible'):
        cells(corrupted)


def target(batch: int, costs: list[int]) -> dict:
    return {'batch': batch, 'cells': [{'physical_batch': batch, 'real_rows': r, 'p95_ns': cost}
                                    for r, cost in enumerate(costs, 1)]}


def test_equal_work_choice_not_always_biggest_batch() -> None:
    targets = [target(1, [100]), target(4, [300, 310, 320, 330])]
    assert wave_plan(targets, 1)['lowest_estimated_cost_batch'] == 1
    r = wave_plan(targets, 4)
    assert r['lowest_estimated_cost_batch'] == 4
    assert r['alternatives'][0]['real_rows_per_call'] == [4]
    for ready in range(1, 65):
        for alternative in wave_plan(targets, ready)['alternatives']:
            assert sum(alternative['real_rows_per_call']) == ready
            assert alternative['executed_real_rows'] == ready
            assert alternative['padded_rows'] == alternative['batch'] * alternative['forward_calls'] - ready
    assert r['applied_to_runtime'] is False
    assert r['deadline_guarantee'] is False


def test_nonmonotonic_service_cost_can_split_within_same_package() -> None:
    result = wave_plan([target(4, [100, 120, 500, 600])], 4)
    assert result['alternatives'][0]['real_rows_per_call'] == [2, 2]
    assert result['alternatives'][0]['sum_sample_p95_ns'] == 240


@pytest.mark.parametrize('bad', [0, 65, True, 1.5, '4'])
def test_invalid_work_size(bad: int) -> None:
    with pytest.raises(ValueError, match='bounded integer'):
        wave_plan([target(1, [10])], bad)


@pytest.mark.parametrize('profiles', [
    [], [target(4, [1, 2])], [target(3, [1, 2, 3])],
    [target(1, [1]), target(1, [2])], [target(1, [0])],
    [target(1, [True])], [target(4, [1, 2, 3, -1])],
])
def test_missing_or_incompatible_cost_cells(profiles: list[dict]) -> None:
    with pytest.raises(ValueError, match=r'profiles|batch|occupancy|integer'):
        wave_plan(profiles, 3)


@pytest.mark.parametrize('text', ['[]', 'null', '{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_malformed_json(text: str) -> None:
    with pytest.raises(ValueError, match=r'JSON|object'):
        object_json(text)


def test_sample_percentile_is_nearest_rank() -> None:
    assert percentile95(list(range(1, 21))) == 19
    assert percentile95([5]) == 5
    with pytest.raises(ValueError, match='no service'):
        percentile95([])
    r, _ = record()
    assert object_json(json.dumps(r)) == r
