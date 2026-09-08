"""One-shot selected training-row bank; no inference or corpus writing.

The metadata-only plan fixes every derived row before this program reads targets.
Original source-qualified history, raw d9 lines, stored SF/C and BT4 are retained.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def write(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def main(plan_path):
    start = time.monotonic()
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    assert plan['status'] == 'PREPARED_NOT_EXECUTED'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.sched_getaffinity(0) == {6, 7}
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19
    out = Path(plan['output_dir'])
    out.mkdir(exist_ok=False)
    pins = dict(plan['pins'])
    for p in [plan_path, Path(__file__), plan_path.parent/'preregistration.md']:
        pins[str(p)] = {'sha256': digest(p)}
    read_files = {}
    raw_files = {}
    def terminated(signum, frame):
        raise TimeoutError(f'outer supervisor signal {signum}; partial bank preserved')
    signal.signal(signal.SIGTERM, terminated)
    try:
        def guard():
            if (plan_path.parent/'STOP').exists():
                raise RuntimeError('STOP requested; preserve partial bank')
            if time.monotonic()-start >= 560:
                raise TimeoutError('collector work budget exhausted before outer timeout')
            if sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > plan['limits']['max_output_bytes']:
                raise RuntimeError('bank output cap exceeded')
        def check_pins():
            for path, proof in pins.items():
                assert digest(path) == proof['sha256'], f'changed pin: {path}'
        check_pins()
        sys.path.insert(0, plan['runtime_checkout'])
        import numpy as np
        import zarr
        from numcodecs import blosc
        from scripts import derive_corpus_targets as derive
        from scripts import bt4_policy_mix as mix
        from scripts.sf_d9_rank_sidecar import d9_lines, rank_observation
        from chess_anti_engine.stockfish.wdl import SF_CP_CLAMP_CP
        blosc.set_nthreads(2)
        for module in list(sys.modules.values()):
            filename = getattr(module, '__file__', None)
            if filename and Path(filename).is_file() and Path(filename).is_relative_to(plan['runtime_checkout']):
                if Path(filename).suffix in {'.py', '.so'}:
                    pins.setdefault(str(Path(filename)), {'sha256': digest(filename)})

        class ObservedStore(zarr.storage.DirectoryStore):
            def __getitem__(self, key):
                path = str(Path(self.path)/key)
                try:
                    before = identity(path)
                except FileNotFoundError as exc:
                    raise KeyError(key) from exc
                value = super().__getitem__(key)
                assert identity(path) == before, f'changed during read: {path}'
                proof = {'stat': before, 'sha256': hashlib.sha256(value).hexdigest()}
                assert path not in read_files or read_files[path] == proof, f'changed reread: {path}'
                read_files[path] = proof
                return value
        def group(root, shard):
            return zarr.open_group(ObservedStore(str(Path(root)/shard)), mode='r')
        wanted = {}
        selected = []
        # Read only selected stored identifiers first; raw file scope is then fixed.
        for choice in plan['selection']:
            guard()
            sf = group(plan['sf_dir'], choice['shard'])
            rr = np.asarray(choice['rows'])
            games = np.asarray(sf['game_id'].oindex[rr])
            plies = np.asarray(sf['ply_index'].oindex[rr])
            for row, game, ply in zip(rr, games, plies, strict=True):
                key = (int(game), int(ply))
                assert key not in wanted, f'ambiguous sampled identity: {key}'
                wanted[key] = (choice['shard'], int(row))
                selected.append({'source_dir': plan['source_dir'], 'derived_shard': choice['shard'],
                                 'derived_row': int(row), 'game_id': key[0], 'ply': key[1],
                                 'stratum': choice['stratum'], 'weight': choice['inverse_probability_weight']})
        assert len(selected) == plan['rows'] == 4096
        summary = json.loads((Path(plan['source_dir'])/'summary.json').read_text())
        games = {key[0] for key in wanted}
        entries = [s for s in summary['shards'] if games.intersection(s['games'])]
        paths = [Path(plan['source_dir'])/s['path'] for s in entries]
        assert len(paths) == len(set(paths)) <= plan['limits']['max_raw_shards']
        assert sum(p.stat().st_size for p in paths) <= plan['limits']['max_raw_compressed_bytes']
        write(out/'selection.json', {'selected': selected, 'raw_scope': [str(p) for p in paths],
                                    'raw_rows_inventory': sum(s['rows'] for s in entries),
                                    'raw_compressed_bytes': sum(p.stat().st_size for p in paths)})
        found = {}
        for path, entry in zip(paths, entries, strict=True):
            guard()
            before = identity(path)
            count = 0
            for physical, raw in enumerate(derive.iter_corpus_rows(path)):
                count += 1
                if count % 1024 == 0: guard()
                key = (int(raw['game_id']), int(raw['ply']))
                if key in wanted:
                    assert key not in found, f'duplicate raw identity: {key}'
                    assert raw.get('result') is not None, 'selected source row has no result'
                    found[key] = (str(path), physical, raw)
            assert count == entry['rows'], f'raw inventory drift: {path}'
            # Hash only these already bounded compressed raw files, never the corpus.
            raw_files[str(path)] = {'stat': before, 'sha256': digest(path), 'rows': count}
            assert identity(path) == before, f'changed raw source: {path}'
        assert set(found) == set(wanted), 'missing sampled raw join; no replacement permitted'
        options = derive.DeriveOptions(scheme=derive.parse_scheme('uniform-d9'), temp=.0005,
                    cp_slope=.006, cp_draw_width=120, limit=4096, seed=0,
                    rows_per_shard=8192, max_envelope_misses=0)
        deriver = derive.TargetDeriver(options)
        records = []
        with (out/'raw_rows.jsonl').open('x') as rawout:
            for choice in plan['selection']:
                guard()
                shard = choice['shard']; rr = np.asarray(choice['rows'])
                sf = group(plan['sf_dir'], shard); c = group(plan['c_dir'], shard)
                bt = group(plan['bt4_dir'], shard)
                for g in [sf, c]:
                    assert g.attrs['input_history_encoding'] == 'lc0_root_legacy_meta'
                    assert g.attrs['history_rep_fix'] is True and g.attrs['positions'] == choice['shard_rows']
                    assert g.attrs['policy_encoding'] == 'lc0_1858'
                assert bt.attrs['source_dir'] == plan['sf_dir'] and bt.attrs['source_shard'] == shard
                names = sorted(sf.array_keys())
                assert set(names) == set(c.array_keys()) and len(names) == 17
                bank = {k: np.asarray(sf[k].oindex[rr]) for k in names}
                for name in names:
                    if name != 'policy_target':
                        assert np.array_equal(bank[name], np.asarray(c[name].oindex[rr])), f'C nonpolicy mismatch: {shard}/{name}'
                cp_target = np.asarray(c['policy_target'].oindex[rr])
                bt_target = np.asarray(bt['bt4_policy'].oindex[rr])
                assert np.array_equal(np.asarray(bt['source_key'].oindex[rr]), mix._source_keys(bank['x'], 'lc0_root_legacy_meta'))
                legal = bank['legal_mask'].astype(bool)
                assert all(np.all(bank[k] == 1) for k in ['has_policy','has_legal_mask','has_game_id','has_ply_index'])
                effective = np.full((len(rr),1858), np.nan, dtype=np.float64)
                for i, row_index in enumerate(rr):
                    guard()
                    key = (int(bank['game_id'][i]), int(bank['ply_index'][i]))
                    path, physical, raw = found[key]
                    board = deriver._board_for(raw)
                    derived = deriver.derive_row(raw)
                    assert derived is not None
                    assert np.array_equal(derived.sample.x.astype(np.float16), bank['x'][i]), 'history/quantization mismatch'
                    lines = d9_lines(raw)
                    obs = rank_observation(raw, top_k=len(lines))
                    assert len(lines) == board.legal_moves.count()
                    assert set(obs.indices.tolist()) == set(np.flatnonzero(legal[i]).tolist())
                    for index, line in zip(obs.indices, sorted(lines,key=lambda x:int(x[0])), strict=True):
                        effective[i, int(index)] = float(line[2])
                    rebuilt = derived.sample.policy_target.astype(np.float16)
                    assert np.array_equal(rebuilt, bank['policy_target'][i]), 'stored SF reconstruction differs'
                    ranks = np.full((1,3),65535,dtype=np.uint16); gaps = np.full((1,3),np.inf,dtype=np.float32)
                    k = min(3,obs.count); ranks[0,:k] = obs.indices[:k]; gaps[0,:k] = obs.gaps_cp[:k]
                    expected_c = mix.mix_policy_targets(bank['policy_target'][i:i+1],bt_target[i:i+1],legal[i:i+1],
                        alpha=1.,scope='sf-cp-window',bt4_temperature=.5,sf_rank_indices=ranks,
                        sf_rank_gaps_cp=gaps,sf_rank_cap=3,sf_cp_window=20.).astype(np.float16)
                    assert np.array_equal(expected_c[0],cp_target[i]), 'actual C recipe reconstruction differs'
                    rec = {'source_dir':plan['source_dir'],'raw_shard':Path(path).name,'physical_row':physical,
                           'derived_shard':shard,'derived_row':int(row_index),'game_id':key[0],'ply':key[1],
                           'worker_id':raw['worker_id'],'input_key':raw['input_key'],
                           'stratum':choice['stratum'],'weight':choice['inverse_probability_weight']}
                    rawout.write(json.dumps({**rec,'raw':raw},allow_nan=False)+'\n')
                    cp = effective[i,legal[i]]
                    targets = {'SF':bank['policy_target'][i,legal[i]],'C':cp_target[i,legal[i]]}
                    for temp in plan['temperatures_cp']:
                        p = derive.softmax_at_temp(cp,temp=temp)
                        targets[f'cp{temp}_ideal'] = p
                        targets[f'cp{temp}_stored'] = derive.shard_stored(p)
                    metrics = {}
                    for label,p in targets.items():
                        q = np.asarray(p,dtype=np.float64); q = q/q.sum(); pos = q>0
                        metrics[label] = {'entropy':float(-np.sum(q[pos]*np.log(q[pos]))),
                            'top1':float(q.max()),'support':int(pos.sum()),
                            'raw_cp_maxima_mass':float(q[cp == cp.max()].sum())}
                    records.append({**rec,'mate_present':bool(np.any(np.abs(cp)>SF_CP_CLAMP_CP)),
                                    'best_effective_cp':float(cp.max()),'metrics':metrics})
                with (out/f'{shard}.npz').open('xb') as f:
                    np.savez_compressed(f,**bank,selected_rows=rr,C_policy=cp_target,BT4_policy=bt_target,effective_cp=effective)
                print(json.dumps({'banked_shards':choice['stratum']+1,'rows':len(records),'elapsed_s':time.monotonic()-start}),flush=True)
        write(out/'observations.json',records)
        for path, proof in {**read_files,**raw_files}.items():
            assert identity(path) == proof['stat'], f'changed consumed input: {path}'
        check_pins(); guard()
        write(out/'consumed_inputs.json',{'metadata_and_code_pins':pins,'stored_chunks':read_files,'raw_files':raw_files})
        weights = np.asarray([r['weight'] for r in records]); means = {}
        for label in records[0]['metrics']:
            means[label] = float(np.average([r['metrics'][label]['entropy'] for r in records],weights=weights))
        closest = min(plan['temperatures_cp'],key=lambda t:(abs(means[f'cp{t}_stored']-means['C']),t))
        write(out/'complete.json',{'status':'COMPLETE_QUALIFIED_TRAINING_SAMPLE','rows':len(records),
              'derived_shards':64,'raw_shards':len(paths),'unique_source_games':len({(r['source_dir'],r['worker_id'],r['game_id']) for r in records}),
              'weighted_mean_entropy_self_normalized':means,'entropy_matching_temperature_cp':closest,
              'elapsed_s':time.monotonic()-start,'cpu_seconds':time.process_time(),
              'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'outputs':{p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()},
              'limits':'Stratified cluster training sample, not iid/heldout or gameplay evidence. No new GPU selection. Raw mate effective-cp unchanged. All selected joins must pass; no dropped/replacement rows. Full corpus Soft-SF derivation remains unqualified.'})
    except BaseException as exc:
        write(out/'failed.json',{'status':'FAILED_NO_TEMPERATURE_CHOICE','error':repr(exc),
                                'elapsed_s':time.monotonic()-start,'partial_bank_preserved':True})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    main(parser.parse_args().plan)
