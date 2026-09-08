"""Independent arithmetic and identity readback of the completed sample bank."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
BANK = ROOT / 'bank'
started = time.monotonic()
complete = json.loads((BANK / 'complete.json').read_text())
plan = json.loads((ROOT / 'plan.json').read_text())
records = json.loads((BANK / 'observations.json').read_text())
assert complete['status'] == 'COMPLETE_QUALIFIED_TRAINING_SAMPLE'
assert len(records) == 4096
for name, proof in complete['outputs'].items():
    payload = (BANK / name).read_bytes()
    assert len(payload) == proof['bytes']
    assert hashlib.sha256(payload).hexdigest() == proof['sha256']
by_row = {(r['derived_shard'], r['derived_row']): r for r in records}
assert len(by_row) == 4096
totals = {}
weight_sum = 0.0
checked = 0
for choice in plan['selection']:
    with np.load(BANK / (choice['shard'] + '.npz'), allow_pickle=False) as bank:
        assert bank['selected_rows'].tolist() == choice['rows']
        masks = bank['legal_mask'].astype(bool)
        cp_rows = bank['effective_cp']
        sf = bank['policy_target']
        c = bank['C_policy']
        for i, row in enumerate(choice['rows']):
            rec = by_row[(choice['shard'], row)]
            assert (rec['game_id'], rec['ply']) == (int(bank['game_id'][i]), int(bank['ply_index'][i]))
            assert rec['weight'] == choice['inverse_probability_weight']
            cp = cp_rows[i, masks[i]]
            assert np.isfinite(cp).all()
            targets = {'SF': sf[i, masks[i]], 'C': c[i, masks[i]]}
            for temp in (10, 20, 40, 80):
                q = np.exp((cp - cp.max()) / temp)
                q /= q.sum()
                targets[f'cp{temp}_ideal'] = q
                targets[f'cp{temp}_stored'] = q.astype(np.float32).astype(np.float16)
            for label, q in targets.items():
                q = np.asarray(q, dtype=np.float64)
                q /= q.sum()
                positive = q > 0
                entropy = float(-(q[positive] * np.log(q[positive])).sum())
                observed = rec['metrics'][label]
                assert abs(entropy - observed['entropy']) < 1e-12
                assert int(positive.sum()) == observed['support']
                assert abs(float(q.max()) - observed['top1']) < 1e-12
                assert abs(float(q[cp == cp.max()].sum()) - observed['raw_cp_maxima_mass']) < 1e-12
                totals[label] = totals.get(label, 0.0) + entropy * rec['weight']
            weight_sum += rec['weight']
            checked += 1
means = {k: v / weight_sum for k, v in totals.items()}
assert checked == 4096
for key, value in means.items():
    assert abs(value - complete['weighted_mean_entropy_self_normalized'][key]) < 1e-12
chosen = min((10, 20, 40, 80), key=lambda t: (abs(means[f'cp{t}_stored'] - means['C']), t))
assert chosen == complete['entropy_matching_temperature_cp'] == 10
receipt = {
    'status': 'PASS_INDEPENDENT_COMPLETED_BANK_ARITHMETIC_AND_HASH_REVIEW',
    'rows': checked, 'shards': 64, 'weighted_entropy': means,
    'temperature_cp': chosen, 'wall_seconds': time.monotonic() - started,
    'complete_sha256': hashlib.sha256((BANK / 'complete.json').read_bytes()).hexdigest(),
    'reviewer_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'scope': 'All output hashes, exact selected rows and identities, every banked entropy/support/top1/raw-best mass and weighted choice independently recomputed. Source-history/SF/C joins were reviewed in the producing code and passing execution; original corpus was not rescanned. This is target geometry, not strength or full-corpus qualification.',
}
with (ROOT / 'root_completed_review.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
print(json.dumps(receipt))
