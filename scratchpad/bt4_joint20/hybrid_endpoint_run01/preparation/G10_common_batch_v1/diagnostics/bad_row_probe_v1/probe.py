"""Bounded direct-row replay after failed lane0's last completed spill."""
import dataclasses
import hashlib
import json
import time
import traceback
from pathlib import Path

from numcodecs import blosc
from scripts import derive_corpus_targets as derive
from scripts import corpus_row_provenance as provenance
from scripts import gen_sf_rooted_corpus as corpus

OUT = Path(__file__).parent
BASE = OUT.parents[1]
START = 6100
COUNT = 2048
RAW = Path('/home/josh/projects/chess/data/nnue_bootstrap/run06_g10/w00-00001.jsonl.zst')
manifest = json.loads((RAW.parent / 'manifest.json').read_text())
slope, width = derive.cp_map_params(manifest)
options = derive.DeriveOptions(
    scheme=dataclasses.replace(derive.parse_scheme('uniform-d9'), policy_observation='phase0', value_observation='latest-phase'),
    temp=0.0005, cp_slope=slope, cp_draw_width=width, limit=265528, seed=0,
    rows_per_shard=8192, max_envelope_misses=0, floor=0, value_scheme='search', row_provenance=True,
)
blosc.set_nthreads(2)
corpus.apply_history_rep_fix()
engine = derive.TargetDeriver(options)
started = time.time()
before = RAW.stat()
results = []
failure = None
with (OUT / 'selected_raw_rows.jsonl').open('x') as raw_bank:
    for index, row in enumerate(derive.iter_corpus_rows(RAW)):
        if index < START:
            continue
        if index >= START + COUNT:
            break
        if time.time() - started > 570:
            raise TimeoutError('bounded probe wall limit')
        raw_bank.write(json.dumps(row, separators=(',', ':')) + '\n')
        identity = {'source_dir': str(RAW.parent), 'source_shard': RAW.name, 'source_row': index,
                    'worker_id': row.get('worker_id'), 'game_id': row.get('game_id'), 'ply': row.get('ply'),
                    'input_key': row.get('input_key')}
        stage = 'row_identity'
        try:
            engine.stats.rows_read += 1
            derive._check_row_identity(row, str(manifest['config_sha256']))
            stage = 'derive_row'
            derived = engine.derive_row(row)
            if derived is not None:
                stage = 'row_provenance'
                provenance.reference(row, RAW, index, str(manifest['config_sha256']), derived.sample.x)
                stage = 'apply_value_scheme'
                derive.apply_value_scheme([derived], options=options, stats=engine.stats, banked_tail_ply=None)
            results.append({**identity, 'status': 'derived' if derived is not None else 'dropped_no_result'})
        except Exception as exc:
            failure = {**identity, 'status': 'exception', 'stage': stage,
                       'exception_type': f'{type(exc).__module__}.{type(exc).__qualname__}',
                       'message': str(exc), 'traceback': traceback.format_exc()}
            results.append(failure)
            (OUT / 'failed_raw_row.json').write_text(json.dumps(row, indent=2) + '\n')
            break

after = RAW.stat()
fields = ['st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns']
assert all(getattr(before, key) == getattr(after, key) for key in fields)
(OUT / 'per_row_results.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in results))
report = {'schema': 1, 'status': 'RECOVERED_ROW_EXCEPTION' if failure else 'NO_EXCEPTION_IN_BOUNDED_ROWS',
          'lane': 'w000', 'reason_lane1_not_replayed': 'parent narrowed scope after saved boundary; healthy lane interrupted by STOP',
          'raw': str(RAW), 'physical_start': START, 'max_rows': COUNT, 'rows_replayed': len(results),
          'elapsed_seconds': time.time() - started, 'failure': failure,
          'options': dataclasses.asdict(options), 'raw_stable_metadata': {key: getattr(after, key) for key in fields},
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'deriver_sha256': hashlib.sha256(Path(derive.__file__).read_bytes()).hexdigest(),
          'source_manifest_sha256': hashlib.sha256((RAW.parent / 'manifest.json').read_bytes()).hexdigest(),
          'limits': ['direct imported frozen module, no process pool or wrapper', 'no spill or output corpus written',
                     'following lane0 rows only; preceding raw bytes decoded solely to seek physical row',
                     'no tolerance or runtime changes; original corpus/stopped batch preserved']}
(OUT / 'readout.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': report['status'], 'rows_replayed': len(results), 'failure': failure, 'elapsed_seconds': report['elapsed_seconds']}))
