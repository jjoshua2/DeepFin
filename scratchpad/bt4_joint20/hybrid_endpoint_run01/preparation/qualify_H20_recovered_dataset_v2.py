"""Qualify recovered H20 publication using explicit process and rounding proofs.

This fixed operation never decompresses policy/input payloads, builds schedules,
launches other processes, or modifies the source or published corpus.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import stat
import time

PREP = Path(__file__).resolve().parent
STATE = PREP.parent
ROOT = STATE.parents[2]
PLAN_SHA = 'a43ca32437855afecb296df9b026d8714904b638be90d4bc0c714785294250a2'
REVIEW_SHA = '2927d829dccb0128494720759e64c71707d2f9078d676add87afc6c21306603a'
SOURCE = ROOT / 'data/nnue_derived/armB/qtemp_0.0005_hist_20m'
CORPUS = SOURCE.with_name(SOURCE.name + '_bt4_hybrid_H20T05')
PARENT = SOURCE.with_name(SOURCE.name + '_bt4_sfclose_C20T05')
SOURCE_SHA = '391837e49773465edced77bfd13f4084edc60feeff0484078280873d942e50ef'
C_DERIVE = 'fc6a33b4f75b154ae2945240d4169b40685618ff0082cc759914a0c4f5b6e8a5'
C_MIX = '5bf8502a12af0b9ce938a39ddfd9d95df4bfb2b1a0f80292d80d04109cce7100'
RECOVERY = PREP / 'H20_recovery_v1'
RECOVERY_PLAN_SHA = '77c2c3c60ec0f5e482030f417043dc0ba89cc88fa33d12d193959a68a7ed8b2e'
RUNTIME_PIN_ALIASES = {'/tmp/deepfin-h20-recovery-tools/chess_anti_engine/encoding/_features_ext.cpython-313-x86_64-linux-gnu.so': '/tmp/deepfin-bt4-hybrid-tools/chess_anti_engine/encoding/_features_ext.cpython-313-x86_64-linux-gnu.so', '/tmp/deepfin-h20-recovery-tools/chess_anti_engine/encoding/_lc0_ext.cpython-313-x86_64-linux-gnu.so': '/tmp/deepfin-bt4-hybrid-tools/chess_anti_engine/encoding/_lc0_ext.cpython-313-x86_64-linux-gnu.so', '/tmp/deepfin-h20-recovery-tools/chess_anti_engine/mcts/_mcts_tree.cpython-313-x86_64-linux-gnu.so': '/tmp/deepfin-bt4-hybrid-tools/chess_anti_engine/mcts/_mcts_tree.cpython-313-x86_64-linux-gnu.so', '/tmp/deepfin-h20-recovery-tools/chess_anti_engine/nnue/_nnue_ext.cpython-313-x86_64-linux-gnu.so': '/tmp/deepfin-bt4-sf-close-followup/chess_anti_engine/nnue/_nnue_ext.cpython-313-x86_64-linux-gnu.so'}
ORIGINAL_FAILED_SHA = 'ae847094c421a26cc62205c4aff7016b6b1e13a46e4e240c633eb249b31599e4'
OUT = PREP / 'H20.dataset_qualification.json'
METADATA = PREP / 'H20.publication_metadata.jsonl'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    s = path.lstat()
    require(stat.S_ISREG(s.st_mode), f'not a regular file: {path}')
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def equivalent(left, right):
    # Derive summaries contain historical NaNs. Compare their serialized values.
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'run with CUDA hidden')
    require(not OUT.exists() and not METADATA.exists(), 'qualification output exists; preserve it')
    seen = {}
    hashes = {}
    alias_identities = {}

    def read(path, expected=None, *, parse=True):
        path = Path(path)
        if str(path) in RUNTIME_PIN_ALIASES:
            require(not parse and path.is_symlink(), f'expected qualified binary link: {path}')
            target = Path(RUNTIME_PIN_ALIASES[str(path)])
            require(path.resolve(strict=True) == target, f'native link target changed: {path}')
            link_stat = path.lstat()
            alias_identities[path] = (link_stat.st_dev, link_stat.st_ino, link_stat.st_size,
                                      link_stat.st_mtime_ns, link_stat.st_ctime_ns)
            path = target
        before = identity(path)
        data = path.read_bytes()
        require(identity(path) == before, f'file changed while reading: {path}')
        digest = hashlib.sha256(data).hexdigest()
        if expected is not None:
            require(digest == expected, f'pin mismatch: {path}')
        seen[path] = before
        hashes[str(path)] = digest
        return json.loads(data) if parse else digest

    plan = read(STATE / 'preparation_plan.json', PLAN_SHA)
    review = read(PREP / 'cpu_preparation_independent_review.json', REVIEW_SHA)
    require(review['status'] == 'PASS_CPU_PREPARATION_LAUNCH', 'CPU preparation was not qualified')
    for path, digest in plan['pins'].items():
        read(path, digest, parse=False)
    require(plan['output'] == str(CORPUS), 'wrong fixed destination')
    require(CORPUS.is_dir() and not CORPUS.is_symlink(), 'published corpus absent or aliased')
    require(not CORPUS.with_name(CORPUS.name + '.writing').exists(), 'partial still exists')
    require(not (STATE / 'STOP').exists() and not (RECOVERY / 'STOP').exists(),
            'STOP present; inspect before qualification')
    for stage in ('audit', 'mix'):
        status = read(STATE / f'{stage}.status.json')
        if stage == 'audit':
            require(status['stage'] == stage and status['status'] == 'COMPLETE'
                    and status['returncode'] == 0 and not status.get('stop_reason'), 'audit incomplete')
        else:
            require(hashes[str(STATE / 'mix.status.json')] == ORIGINAL_FAILED_SHA
                    and status['status'] == 'FAILED_OR_STOPPED' and status['returncode'] == 124,
                    'original failure receipt changed')
        require(status['plan_sha256'] == PLAN_SHA and status['argv'] == plan[f'{stage}_argv']
                and status['supervisor_sha256'] == plan['pins'][str(STATE / 'run_preparation.py')],
                f'{stage} execution identity differs')
        require(status['completed_unix'] >= status['started_unix'], f'{stage} timing invalid')
    recovery_plan = read(RECOVERY / 'launch.json', RECOVERY_PLAN_SHA)
    for path, digest in recovery_plan['pins'].items():
        read(path, digest, parse=False)
    recovery_status = read(RECOVERY / 'status.json')
    require(recovery_status['status'] == 'COMPLETE' and recovery_status['returncode'] == 0
            and not recovery_status.get('stop_reason') and recovery_status['stage'] == 'recovery'
            and recovery_status['argv'] == recovery_plan['argv']
            and recovery_status['plan_sha256'] == RECOVERY_PLAN_SHA
            and recovery_status['supervisor_sha256'] == recovery_plan['supervisor_sha256'],
            'recovery did not complete under the pinned plan')
    require(0 <= recovery_status['completed_unix'] - recovery_status['started_unix'] <= 1800,
            'recovery timing exceeded registered budget')
    audit_status = read(STATE / 'audit.status.json')
    audit = read(STATE / 'audit_H20.json', audit_status['audit_sha256'])
    require(audit['gate']['training_permitted'] is True
            and audit['gate']['treatment_invariants_passed'] is True
            and audit['admission']['mode'] == 'descriptive', 'descriptive audit fidelity failed')
    source = read(SOURCE / 'derive_targets_summary.json', SOURCE_SHA)
    c_derive = read(PARENT / 'derive_targets_summary.json', C_DERIVE)
    c_mix = read(PARENT / 'bt4_policy_mix_summary.json', C_MIX)
    require(equivalent(c_derive.pop('policy_target_postprocess'), c_mix)
            and equivalent(c_derive, source), 'C transitive derive lineage differs')
    derive_path = CORPUS / 'derive_targets_summary.json'
    mix_path = CORPUS / 'bt4_policy_mix_summary.json'
    derive = read(derive_path)
    mix = read(mix_path)
    require(equivalent(derive.pop('policy_target_postprocess'), mix)
            and equivalent(derive, source), 'H20 transitive derive lineage differs')
    expected = {'schema': 1, 'kind': 'c20-global', 'algorithm': 'stored-c20t05-then-global-bt4-v1',
                'alpha': .2, 'bt4_temperature': .5, 'near_max_ratio': None,
                'sf_rank_cap': 3, 'sf_cp_window': 20.0, 'rows': 18910484, 'shards': 2309,
                'expected_rows': 18910484, 'expected_shards': 2309, 'source_dir': str(SOURCE),
                'source_derive_summary_sha256': SOURCE_SHA, 'mutated_arrays': ['policy_target'],
                'value_columns_unchanged': ['wdl_target', 'search_wdl'],
                'formula': '0.8*legal_normalize(actual stored C20T05)+0.2*legal_normalize(BT4^2)'}
    require(all(mix.get(k) == v for k, v in expected.items()), 'final H20 recipe/counts differ')
    require(mix['admission'] == audit['admission']
            and mix['audit_receipt'] == {'path': str(STATE / 'audit_H20.json'), 'sha256': audit_status['audit_sha256']},
            'final audit/admission provenance differs')
    parent = mix['c20_parent']
    require(parent['source_dir'] == str(PARENT)
            and parent['derive_summary'] == {'path': str(PARENT / 'derive_targets_summary.json'), 'sha256': C_DERIVE}
            and parent['mix_summary'] == {'path': str(PARENT / 'bt4_policy_mix_summary.json'), 'sha256': C_MIX}
            and equivalent(parent['policy_target_postprocess'], c_mix), 'actual C parent differs')
    require(mix['parent_recipe'] == audit['treatment']['parent_recipe'], 'parent recipe audit differs')
    for key in ('sidecar_summary', 'sf_rank_sidecar_summary'):
        item = mix[key]
        require(plan['pins'].get(item['path']) == item['sha256'], f'{key} lineage differs')
    recovered = mix['recovery']
    require(recovered['method'] == 'completed-prefix-process-proof-v1'
            and recovered['receipt'] == recovery_plan['recovery_receipt']
            and recovered['original_status']['sha256'] == ORIGINAL_FAILED_SHA
            and recovered['completed_prefix_shards'] == 2292
            and recovered['completed_prefix_rows'] == 18776064,
            'recovery prefix/provenance differs')
    suffix = recovered['suffix_statistics']
    require(suffix['rows'] == 134420 and suffix['shards'] == 17
            and 0 < suffix['changed_rows'] <= suffix['rows']
            and math.isfinite(suffix['mean_l1_from_source']) and suffix['mean_l1_from_source'] > 0,
            'suffix does not establish non-inert completed treatment')
    suffix_drift = suffix['selected_mass_abs_drift']
    require(suffix_drift['reference'] == 'normalized_total_legal_mass'
            and all(math.isfinite(suffix_drift[k]) for k in ('mean', 'max'))
            and 0 <= suffix_drift['mean'] <= suffix_drift['mean_bound'] == 2**-10
            and 0 <= suffix_drift['max'] <= suffix_drift['row_bound'] == 2**-10,
            'suffix measured legal mass bounds failed')
    certificate = recovered['mass_bound_certificate']
    delta = 1e-9
    e32 = delta + 2**-24 * (1 + delta) + 1858 * 2**-150
    e16 = e32 + 2**-11 * (1 + e32) + 1858 * 2**-25
    expected_certificate = {
        'method': 'normalized-f64-f32-f16-rounding-v1', 'max_policy_width': 1858,
        'float64_normalization_error_allowance': delta, 'derived_absolute_bound': e16,
        'certified_row_abs_bound': .000545, 'certified_mean_abs_bound': .000545,
        'observed_mean': None, 'observed_max': None}
    require(certificate == expected_certificate and e16 < .000545 < 2**-10,
            'analytic legal mass certificate differs')
    drift = mix['selected_mass_abs_drift']
    require(drift['reference'] == 'normalized_total_legal_mass'
            and drift['mean'] is None and drift['max'] is None
            and drift['mean_bound'] == drift['row_bound'] == 2**-10,
            'unmeasured aggregate mass statistics must remain unavailable')
    unavailable = ('changed_rows', 'changed_fraction', 'source_top_tied_rows',
                   'source_top_tied_fraction', 'source_candidate_multi_rows',
                   'source_candidate_multi_fraction', 'candidate_set_wider_rows',
                   'candidate_set_wider_fraction', 'changed_unique_max_rows',
                   'source_bt4_top1_agreement', 'mixed_source_top1_agreement',
                   'mixed_bt4_top1_agreement', 'mean_entropy_nats',
                   'top1_ge_0_99_fraction', 'mean_l1_from_source')
    require(all(key in mix and mix[key] is None for key in unavailable),
            'full-corpus descriptive statistics were not recovered')
    names = sorted(p.name for p in SOURCE.glob('shard_*.zarr') if p.is_dir())
    require(len(names) == 2309 and names == sorted(p.name for p in CORPUS.glob('shard_*.zarr') if p.is_dir()),
            'published shard membership differs')
    rows = 0
    metadata = []
    expected_attrs = {'policy_target_mix_schema': 1, 'policy_target_mix_kind': 'c20-global',
                      'policy_target_mix_algorithm': expected['algorithm'], 'policy_target_mix_alpha': .2,
                      'policy_target_mix_bt4_temperature': .5, 'policy_target_mix_sf_rank_cap': 3,
                      'policy_target_mix_sf_cp_window': 20.0, 'policy_target_mix_value_columns_unchanged': True,
                      'policy_target_mix_c20_parent_derive_sha256': C_DERIVE,
                      'policy_target_mix_c20_parent_mix_sha256': C_MIX,
                      'policy_target_mix_sidecar': mix['sidecar_dir'],
                      'policy_target_mix_sf_rank_sidecar': mix['sf_rank_sidecar_dir']}
    for name in names:
        s, d = SOURCE / name, CORPUS / name
        require(not s.is_symlink() and not d.is_symlink(), f'aliased shard: {name}')
        source_arrays = sorted(p.name for p in s.iterdir() if p.is_dir())
        target_arrays = sorted(p.name for p in d.iterdir() if p.is_dir())
        require(source_arrays == target_arrays and len(source_arrays) == 17
                and all((d / field / '.zarray').is_file() and not (d / field).is_symlink()
                        for field in target_arrays), f'array inventory differs: {name}')
        sa, da = read(s / '.zattrs'), read(d / '.zattrs')
        require(all(k in da and equivalent(v, da[k]) for k, v in sa.items()), f'original attrs changed: {name}')
        require(all(da.get(k) == v for k, v in expected_attrs.items()), f'incomplete/wrong shard recipe: {name}')
        for k in ('source_key', 'source_policy', 'sf_rank_payload', 'c20_parent_policy'):
            value = da[f'policy_target_mix_{k}_sha256']
            require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
                    f'missing payload validation digest: {name}:{k}')
        sp, dp = read(s / 'policy_target/.zarray'), read(d / 'policy_target/.zarray')
        sx, dx = read(s / 'x/.zarray'), read(d / 'x/.zarray')
        require(sp == dp and sx == dx and dp['dtype'] == '<f2'
                and dp['shape'][1:] == [1858] and sx['shape'][0] == dp['shape'][0], f'layout differs: {name}')
        rows += dp['shape'][0]
        metadata.append({'shard': name, 'rows': dp['shape'][0], 'files': {
            str(p.relative_to(ROOT)): hashes[str(p)]
            for p in (s / '.zattrs', d / '.zattrs', s / 'policy_target/.zarray', d / 'policy_target/.zarray',
                      s / 'x/.zarray', d / 'x/.zarray')}})
    require(rows == 18910484, 'summed published row count differs')
    for path, previous in seen.items():
        require(identity(path) == previous, f'metadata/input changed during qualification: {path}')
    for path, previous in alias_identities.items():
        now = path.lstat()
        require(path.is_symlink() and path.resolve(strict=True) == Path(RUNTIME_PIN_ALIASES[str(path)])
                and (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns, now.st_ctime_ns) == previous,
                f'qualified native link changed during qualification: {path}')
    content = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in metadata).encode()
    result = {'schema': 1, 'status': 'PASS_REGISTERED_CORPUS_QUALIFICATION', 'profile': 'H20',
              'corpus': str(CORPUS), 'rows': rows, 'shards': len(names),
              'source': {'path': str(SOURCE), 'derive_sha256': SOURCE_SHA},
              'derive_summary': {'path': str(derive_path), 'sha256': hashes[str(derive_path)]},
              'mix_summary': {'path': str(mix_path), 'sha256': hashes[str(mix_path)]},
              'completed_unix': time.time(), 'qualifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'preparation_plan_sha256': PLAN_SHA, 'cpu_preparation_independent_review_sha256': REVIEW_SHA,
              'completion_proofs': {'audit': hashes[str(STATE / 'audit.status.json')],
                                    'original_failed_mix': ORIGINAL_FAILED_SHA,
                                    'completed_recovery': hashes[str(RECOVERY / 'status.json')]},
              'recovery_plan_sha256': RECOVERY_PLAN_SHA, 'recovery': recovered,
              'qualified_native_aliases': RUNTIME_PIN_ALIASES,
              'audit_sha256': audit_status['audit_sha256'], 'c20_parent': parent,
              'metadata_manifest': {'path': str(METADATA), 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)},
              'observed_checks': 'Completed audit; pinned original timeout and completed recovery; analytic mass bound, suffix non-inert witness, final recipe, transitive summaries, all shard validation stamps, source attrs, layouts, counts and stable metadata.',
              'materializer_proof_reused': 'Original pinned mixer completed all chunk checks before stamping 2292 shards; the separately reviewed recovery completes 17 remaining shards with the original validations. These executed checks and per-shard digests are reused. The recovered summary certifies normalization analytically; full empirical aggregates were lost and are not claimed.',
              'limits': ['No second prefix payload scan. Completed-prefix process proof and conservative IEEE rounding bound replace missing original finalization; this is not an independent full payload checksum or measured full-corpus descriptive statistics.',
                         'Metadata stability during this check does not establish immutable payload bytes forever; preserve this finalized corpus.',
                         'Prospective canonical H20/C schedule proof remains a separate mandatory launcher input.',
                         'No training, inference, GPU probe, strength judgment or automatic launch.']}
    with METADATA.open('xb') as f:
        f.write(content)
    with OUT.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'rows', 'shards', 'derive_summary', 'mix_summary')}, indent=2))


if __name__ == '__main__':
    main()
