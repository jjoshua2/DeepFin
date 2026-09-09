"""Bounded descriptive analysis; no engine or model imports."""
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BANK = HERE.parent / 'soft_sf_qualified_sample_v1/bank'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    complete = json.loads((BANK / 'complete.json').read_text())
    observations_path = BANK / 'observations.json'
    assert sha(observations_path) == complete['outputs']['observations.json']['sha256']
    observations = json.loads(observations_path.read_text())
    assert len(observations) == 4096
    grouped = collections.defaultdict(list)
    for row in observations:
        grouped[row['derived_shard']].append(row)
    results = []
    pins = {}
    for shard, rows in grouped.items():
        path = BANK / (shard + '.npz')
        pins[path.name] = sha(path)
        assert pins[path.name] == complete['outputs'][path.name]['sha256']
        with np.load(path, allow_pickle=False) as bank:
            positions = {int(v): i for i, v in enumerate(bank['selected_rows'])}
            for row in rows:
                i = positions[row['derived_row']]
                assert int(bank['game_id'][i]) == row['game_id']
                assert int(bank['ply_index'][i]) == row['ply']
                legal = bank['legal_mask'][i].astype(bool)
                raw = bank['BT4_policy'][i, legal].astype(float)
                cp = bank['effective_cp'][i, legal].astype(float)
                assert np.isfinite(cp).all() and np.isfinite(raw).all()
                assert (raw >= 0).all() and raw.sum() > 0
                b = np.square(raw / raw.max())
                b /= b.sum()
                delta = cp.max() - cp
                for gap in (100, 200):
                    weights = np.maximum(.1, np.exp(-np.maximum(0, delta-gap)/100))
                    if row['mate_present']:
                        weights[:] = 1
                    q = b * weights
                    q /= q.sum()
                    assert np.isfinite(q).all() and abs(q.sum()-1) < 1e-12
                    assert np.array_equal(q > 0, b > 0)
                    unchanged = (weights == 1) & (b > 0)
                    assert np.ptp(q[unchanged] / b[unchanged]) < 1e-12 if unchanged.any() else True
                    if row['mate_present']:
                        assert np.allclose(q, b, rtol=0, atol=1e-15)
                    entropy = lambda p: float(-np.sum(p[p > 0] * np.log(p[p > 0])))
                    results.append({
                        'gap_cp': gap, 'weight': row['weight'],
                        'source_dir': row['source_dir'], 'raw_shard': row['raw_shard'],
                        'physical_row': row['physical_row'], 'game_id': row['game_id'],
                        'mate_excluded': bool(row['mate_present']),
                        'exposed_mass': float(b[weights < 1].sum()),
                        'tv': float(np.abs(q-b).sum()/2),
                        'top_changed': bool(np.argmax(q) != np.argmax(b)),
                        'entropy_change': entropy(q)-entropy(b),
                    })
    summary = {}
    for gap in (100, 200):
        rows = [r for r in results if r['gap_cp'] == gap]
        weights = np.array([r['weight'] for r in rows])
        summary[gap] = {key: float(np.average([r[key] for r in rows], weights=weights))
                        for key in ('mate_excluded', 'exposed_mass', 'tv', 'top_changed', 'entropy_change')}
    report = {'status': 'DESCRIPTIVE_ONLY', 'summary': summary, 'rows': results,
              'input_pins': pins, 'observations_sha256': sha(observations_path),
              'plan_sha256': sha(HERE/'PLAN.md'), 'script_sha256': sha(Path(__file__))}
    with (HERE/'readout.json').open('x') as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
