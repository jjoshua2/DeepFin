#!/usr/bin/env python3
"""One registered seed-zero epoch and its fixed comparison sequence; plan by default.

E0T05 retains schema1; schema2 explicitly selects H20, B100 or G50, never a grid.
Schema3 also admits the original-source SoftSF10 rewrite; arenas are separate.

Existing published targets only. No mixing, resume, retries or automatic promotion.
The prospective and realized schedule checks use the frozen seed-zero verifier.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import subprocess
import sys

# Direct CLI execution must import the sibling from this launcher checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import bt4_direct_screen as arena

ROOT = arena.ROOT
CORPUS = ROOT / 'data/nnue_derived/armB/qtemp_0.0005_hist_20m_bt4_toptie_t050'
SOURCE = ROOT / 'data/nnue_derived/armB/qtemp_0.0005_hist_20m'
VERIFIER = ROOT / 'scratchpad/bt4_joint20/global_run02/verify_matched_schedule.py'
CANONICAL = 'dc687fc333295dee565d19bb4f20da5aa95479dba3aacc5499c22a4004acc64f'
C_ROLE, C_CHECKPOINT, C_SHA = arena.CHECKPOINTS['candidate']
INPUT_PINS = {
    str(SOURCE / 'derive_targets_summary.json'): '391837e49773465edced77bfd13f4084edc60feeff0484078280873d942e50ef',
    str(CORPUS / 'bt4_policy_mix_summary.json'): 'a85ba1403c2477018bdc59b622311094027ca17cf045dc78090f6c0ee9f5463d',
    str(CORPUS / 'derive_targets_summary.json'): '4354bafe8e1435bb030faaa161810bbfff667a2b83c647940ad371eb2ab1caf1',
    str(VERIFIER): '4e8e27e861021dd4c75a1a21404ff9aed0d55775e0212def585bc21b1d8cde18',
    str(C_CHECKPOINT.parent / 'summary.json'): 'cacc3652db3deb2bcd12cf69b0493d3f48d7ef2555c3ea4dc7502fedff3313fd',
    str(ROOT / 'scratchpad/bt4_joint20/sf_close_run02/optional_sharpened_ties_data_qualification.json'):
        '48897206e4af6d37bb4fa351cebe1e1d4b493e07b63126fd8a65332bd85af8c1',
    str(arena.RUNTIME / 'scripts/lc0_control_train.py'): '52d1132689c1cd53a23b63c9274226b467bd9aafdc34548a18db121a03bf9337',
    str(arena.RUNTIME / 'configs/lc0_positive_control.yaml'): '413dbea9dcde2774eafc2fde706e639fef9e944e301717b938b39b4729633de2',
    str(arena.RUNTIME / 'chess_anti_engine/replay/game_epoch.py'): '621e5d0764e62cee492688e63e4099ff8cbc0d39ea094b252c3cae31cd74fde3',
}


CORPORA = {
    'H20': SOURCE.with_name(SOURCE.name + '_bt4_hybrid_H20T05'),
    'B100': SOURCE.with_name(SOURCE.name + '_bt4_global_B100T05'),
    'G50': SOURCE.with_name(SOURCE.name + '_bt4_global_G50T05'),
    'SoftSF10': SOURCE.with_name(SOURCE.name + '_softsf_cp10'),
    'B100V10': SOURCE.with_name(SOURCE.name + '_bt4_global_B100T05_value10'),
}
TOTAL_CAPS = {'H20': 32400, 'B100': 27000, 'G50': 21600}
C_CORPUS = SOURCE.with_name(SOURCE.name + '_bt4_sfclose_C20T05')
C_CORPUS_PINS = {
    str(C_CORPUS / 'derive_targets_summary.json'): 'fc6a33b4f75b154ae2945240d4169b40685618ff0082cc759914a0c4f5b6e8a5',
    str(C_CORPUS / 'bt4_policy_mix_summary.json'): '5bf8502a12af0b9ce938a39ddfd9d95df4bfb2b1a0f80292d80d04109cce7100',
}
COMMON_PINS = {path: digest for path, digest in INPUT_PINS.items()
               if path not in {str(CORPUS / 'bt4_policy_mix_summary.json'), str(CORPUS / 'derive_targets_summary.json'),
                               str(ROOT / 'scratchpad/bt4_joint20/sf_close_run02/optional_sharpened_ties_data_qualification.json')}}


# The completed legacy raw-source producer reviewed for the cp10 control.
# These are producing-code identities, not a requirement to run that CPU runtime
# during training. Training still uses the unchanged historical CUDA runtime.
SOFTSF_PRODUCER_PINS = {
    'scripts/sf_policy_rewrite.py': '85152a4e70e85f2f9ddf8d799ff726f3b4e6ccbc385c1693079cb97edb606a3b',
    'scripts/derive_corpus_targets.py': '248574582a904a563217ce6cb86b0dc1fb000b3fc4bb6c6c232d3dbc9770380f',
    'scripts/sf_d9_rank_sidecar.py': '53c2d1d1638c9e0646e68430c56a413c2b0b78a5ca815996eda770eecd8d3def',
}
SOFTSF_RAW = ROOT / 'data/nnue_bootstrap/run03_s3'
SOFTSF_RAW_SUMMARY_SHA = '55a9cf043b9b90a005bd1adf1dc6d810cb282341bcc11a4eccf127c07c09d6af'


VALUE_PRODUCER_PINS = {'scripts/bt4_value_rewrite.py': 'eedcee1030211503173bf6936d890ab64cadacf35f1b5b4e557cec85d173c428', 'scripts/bt4_derived_wdl_sidecar.py': '6fdaf63038f58e71d6ee6ab8fc6bbc04cca422b1469ecd0f65eabf890085a370', 'scripts/bt4_raw_corpus_sidecar.py': '1c63aa4147ef8717855d226f30094cbc03431e556cd7106d0619adee7945363a', 'chess_anti_engine/encoding/lc0.py': 'a20b56a7c0666e134791855c0f124a66504013983f8f97948539a015dac9c3ee', 'scripts/sf_policy_rewrite.py': '85152a4e70e85f2f9ddf8d799ff726f3b4e6ccbc385c1693079cb97edb606a3b'}
B100_PARENT_PINS = {'derive_targets_summary.json': '47e0e0cca578a89278383d1faef70c5f1f8c45fbc5a256cb91400243315dbb43', 'bt4_policy_mix_summary.json': '221a8296608ee5c698a4d8bf59145208c409de43a2fc1427824c5ad5d453fd38'}

def recipe_summary_name(m):
    if role_for(m) == 'B100V10':
        return 'bt4_value_rewrite_summary.json'
    return 'sf_policy_rewrite_summary.json' if role_for(m) == 'SoftSF10' else 'bt4_policy_mix_summary.json'


def verify_softsf_recipe(m, rewritten, derived):
    """Admit the genuine cp10 producer proof; never synthesize a BT4 mix receipt."""
    corpus = corpus_for(m)
    arena.require(corpus.is_dir() and not corpus.is_symlink()
                  and not corpus.with_name(corpus.name + '.writing').exists()
                  and not corpus.with_name(corpus.name + '.writing').is_symlink()
                  and not (corpus / 'failed.json').exists(), 'SoftSF10 output is incomplete')
    source = arena.read(SOURCE / 'derive_targets_summary.json')
    rest = {k: v for k, v in derived.items() if k != 'policy_target_postprocess'}
    # Historical diagnostics contain NaNs. Compare their serialized source bytes
    # semantically without interpreting NaN as a valid scientific measurement.
    arena.require(json.dumps(rest, sort_keys=True) == json.dumps(source, sort_keys=True),
                  'SoftSF10 changed source selectors/value/history metadata')
    expected = {'schema': 1, 'status': 'COMPLETE', 'kind': 'sf_policy_score_rewrite',
                'score_space': 'effective-cp', 'temperature': 10.0,
                'source_dir': str(SOURCE), 'raw_dir': str(SOFTSF_RAW),
                'raw_limit': 20000000, 'rows': 18910484, 'shards': 2309,
                'rows_dropped_no_result': 1089516, 'mutated_arrays': ['policy_target'],
                'nonpolicy_arrays_copied': 16, 'raw_manifest_present': False,
                'source_derive_summary_sha256': COMMON_PINS[str(SOURCE / 'derive_targets_summary.json')]}
    arena.require(all(rewritten.get(k) == v for k, v in expected.items()),
                  'SoftSF10 score/source/nonpolicy recipe differs')
    count, error = rewritten.get('changed_rows'), rewritten.get('stored_mass_error_max')
    arena.require(type(count) is int and 0 < count <= 18910484
                  and type(error) in (float, int) and math.isfinite(error) and 0 <= error <= 2**-10,
                  'SoftSF10 is inert or has invalid stored mass')
    producers = rewritten.get('producer_sha256', {})
    arena.require(len(producers) == len(SOFTSF_PRODUCER_PINS), 'SoftSF10 producer identities differ')
    for suffix, digest in SOFTSF_PRODUCER_PINS.items():
        matches = [v for k, v in producers.items() if Path(k).is_absolute() and k.endswith('/' + suffix)]
        arena.require(matches == [digest], 'SoftSF10 producer identities differ')
    metadata = rewritten.get('metadata_sha256', {})
    arena.require(metadata.get(str(SOFTSF_RAW / 'summary.json')) == SOFTSF_RAW_SUMMARY_SHA
                  and metadata.get(str(SOURCE / 'derive_targets_summary.json'))
                  == expected['source_derive_summary_sha256'], 'SoftSF10 source metadata pins differ')
    outputs = rewritten.get('outputs', [])
    arena.require(len(outputs) == 2309 and [(o['path'], o['rows']) for o in outputs]
                  == [(o['path'], o['rows']) for o in source['shards']],
                  'SoftSF10 completed shard inventory differs')



def verify_value_recipe(m, rewritten, derived):
    """B100 policy is retained; only the baked search_wdl target may change."""
    corpus = corpus_for(m)
    arena.require(corpus.is_dir() and not corpus.is_symlink()
                  and not corpus.with_name(corpus.name + '.writing').exists()
                  and not (corpus / 'failed.json').exists(), 'value output is incomplete')
    parent = CORPORA['B100']
    parent_summary = parent / 'derive_targets_summary.json'
    parent_policy = parent / 'bt4_policy_mix_summary.json'
    arena.require(rewritten['source_derive_summary_sha256'] == B100_PARENT_PINS['derive_targets_summary.json']
                  and rewritten['source_policy_summary_sha256'] == B100_PARENT_PINS['bt4_policy_mix_summary.json'],
                  'value parent is not the qualified B100 corpus')
    arena.pin(parent_summary, rewritten['source_derive_summary_sha256'])
    arena.pin(parent_policy, rewritten['source_policy_summary_sha256'])
    original = arena.read(parent_summary)
    policy = arena.read(parent_policy)
    sf = arena.read(SOURCE / 'derive_targets_summary.json')
    def require_json(a, b):
        return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    arena.require(require_json({k: v for k, v in original.items() if k != 'policy_target_postprocess'}, sf)
                  and original.get('policy_target_postprocess') == policy,
                  'value parent changed original SF metadata')
    expected_policy = {'kind': 'global', 'algorithm': 'legal-normalized-global-arithmetic-v1',
                       'alpha': 1.0, 'bt4_temperature': .5, 'rows': 18910484,
                       'expected_shards': 2309, 'source_dir': str(SOURCE),
                       'source_derive_summary_sha256': COMMON_PINS[str(SOURCE / 'derive_targets_summary.json')],
                       'mutated_arrays': ['policy_target']}
    arena.require(all(policy.get(k) == v for k, v in expected_policy.items()), 'value parent is not B100')
    arena.require(arena.sha(corpus / 'bt4_policy_mix_summary.json') == rewritten['source_policy_summary_sha256'],
                  'copied B100 recipe differs')
    expected = {'schema': 1, 'status': 'COMPLETE', 'kind': 'bt4_value_rewrite',
                'algorithm': 'normalized-wdl-arithmetic-90-10-float16-v1',
                'sf_weight': .9, 'bt4_weight': .1, 'wdl_order': 'WDL', 'wdl_pov': 'side_to_move',
                'wdl_kind': 'probabilities', 'wdl_output': '/output/wdl',
                'onnx_sha256': '1d3c0bd28ebfb42b015d18f67831cb1d6d15ad5d358b25b8a8cf500786262fc0',
                'rows': 18910484, 'shards': 2309, 'source_dir': str(parent), 'sf_source_dir': str(SOURCE),
                'sf_derive_summary_sha256': COMMON_PINS[str(SOURCE / 'derive_targets_summary.json')],
                'mutated_arrays': ['search_wdl'], 'value_scheme': 'sf90-bt4-native10',
                'value_source': 'stored-sf-search-and-derived-bt4-wdl;onnx='
                    '1d3c0bd28ebfb42b015d18f67831cb1d6d15ad5d358b25b8a8cf500786262fc0;output=/output/wdl',
                'unchanged_arrays': sorted(['x', 'policy_target', 'legal_mask', 'game_id', 'ply_index',
                    'wdl_target', 'priority', 'is_selfplay', 'is_network_turn', 'has_game_id', 'has_ply_index',
                    'has_policy', 'has_legal_mask', 'has_search_wdl', 'has_is_selfplay', 'has_is_network_turn'])}
    arena.require(all(rewritten.get(k) == v for k, v in expected.items()), 'registered value recipe differs')
    count, error = rewritten.get('changed_rows'), rewritten.get('stored_mass_error_max')
    arena.require(type(count) is int and 0 < count <= 18910484 and type(error) in (int, float)
                  and math.isfinite(error) and 0 <= error <= 2**-10, 'inert or invalid value rewrite')
    wanted = dict(original)
    wanted['value_scheme'] = {'name': expected['value_scheme'], 'source': expected['value_source']}
    wanted['value_target_postprocess'] = {k: v for k, v in rewritten.items() if k != 'outputs'}
    arena.require(require_json(wanted, derived), 'value rewrite changed policy/history or source lineage')
    producers = rewritten.get('producer_sha256', {})
    arena.require(len(producers) == len(VALUE_PRODUCER_PINS), 'value producer identities differ')
    for suffix, digest in VALUE_PRODUCER_PINS.items():
        matches = [v for k, v in producers.items() if Path(k).is_absolute() and k.endswith('/' + suffix)]
        arena.require(matches == [digest], 'value producer identities differ')
    outputs = rewritten.get('outputs', [])
    arena.require(len(outputs) == 2309 and [(o['path'], o['rows']) for o in outputs]
                  == [(o['path'], o['rows']) for o in sf['shards']], 'value coverage differs')


def role_for(m):
    return m.get('profile', 'E0T05')


def corpus_for(m):
    return CORPORA[role_for(m)] if 'profile' in m else CORPUS


def input_pins(m):
    return m['input_pins'] if 'profile' in m else INPUT_PINS


def training_only(m):
    return m.get('schema') == 3


def comparisons(m):
    if training_only(m):
        return ()
    return arena.REGISTERED_COMPARISONS[role_for(m)] if 'profile' in m else (('C20T05', 100, 1000),)


def validate(m):
    keys = {'schema', 'state', 'run', 'training_seconds', 'arena_seconds', 'total_seconds',
            'runtime_manifest', 'preregistration', 'prospective_schedule',
            'launcher_sha256', 'arena_launcher_sha256', 'input_pins'}
    registered = 'profile' in m
    only = training_only(m)
    if only:
        keys -= {'arena_seconds', 'total_seconds', 'arena_launcher_sha256'}
        keys |= {'mode', 'stage_helper_sha256'}
        arena.require(m.get('mode') == 'training_only' and registered, 'schema3 requires training_only mode and profile')
    if registered:
        keys |= {'profile', 'data_qualification'}
        arena.require(m['schema'] in (2, 3) and role_for(m) in CORPORA, 'unsupported registered profile')
        arena.require(role_for(m) not in {'SoftSF10', 'B100V10'} or only, f'{role_for(m)} requires schema3 training_only')
        if not only:
            keys |= {'reader', 'comparisons'}
            arena.require(m['comparisons'] == [list(cell) for cell in comparisons(m)], 'registered comparison order differs')
            arena.require(m['reader']['path'] == str(arena.EXTENDED_READER) and set(m['reader']) == {'path', 'sha256'}, 'wrong registered reader')
        corpus = corpus_for(m)
        expected_keys = set(COMMON_PINS) | {str(corpus / recipe_summary_name(m)), str(corpus / 'derive_targets_summary.json')}
        arena.require(set(m['input_pins']) == expected_keys and all(m['input_pins'][k] == v for k, v in COMMON_PINS.items()),
                      'registered source/runtime input pins differ')
        arena.require(all(isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v)
                          for v in m['input_pins'].values()), 'final published input hashes required')
    else:
        arena.require(m['schema'] == 1 and m['input_pins'] == INPUT_PINS, 'published corpus/source/config identity differs')
    arena.require(set(m) == keys, 'one-epoch manifest keys differ')
    for key in (('training_seconds',) if only else ('training_seconds', 'arena_seconds', 'total_seconds')):
        arena.require(type(m[key]) in (int, float) and math.isfinite(m[key]) and m[key] > 30, f'invalid {key}')
    if only:
        arena.require(m['training_seconds'] == 16200, 'registered training budget differs')
    else:
        expected_total = TOTAL_CAPS[role_for(m)] if registered else 21600
        arena.require(m['training_seconds'] == 16200 and m['arena_seconds'] == 5400
                      and m['total_seconds'] == expected_total, 'registered training/arena/total budget differs')
    outputs = [Path(m[k]) for k in ('state', 'run')]
    inputs = [corpus_for(m), SOURCE, arena.RUNTIME, C_CHECKPOINT, Path(__file__).resolve(), Path(arena.__file__).resolve()]
    evidence = ('runtime_manifest', 'preregistration', 'prospective_schedule')
    if registered:
        evidence += ('data_qualification',) if only else ('reader', 'data_qualification')
    for key in evidence:
        arena.require(set(m[key]) == {'path', 'sha256'} and Path(m[key]['path']).is_absolute(), f'invalid {key}')
        inputs.append(Path(m[key]['path']).resolve())
    for i, out in enumerate(outputs):
        arena.require(out.is_absolute() and out == out.resolve() and out.parent.is_dir(), 'canonical output with existing parent required')
        arena.require(not out.exists() and not out.is_symlink(), f'new output required: {out}')
        for other in inputs + outputs[i+1:]:
            arena.require(out != other and out not in other.parents and other not in out.parents, 'overlapping input/output paths')


def verify_data_qualification(m):
    corpus = corpus_for(m)
    receipt = arena.read(m['data_qualification']['path'])
    expected = {
        'schema': 1, 'status': 'PASS_REGISTERED_CORPUS_QUALIFICATION', 'profile': role_for(m),
        'corpus': str(corpus), 'rows': 18910484, 'shards': 2309,
        'source': {'path': str(SOURCE), 'derive_sha256': COMMON_PINS[str(SOURCE / 'derive_targets_summary.json')]},
        'derive_summary': {'path': str(corpus / 'derive_targets_summary.json'),
                           'sha256': input_pins(m)[str(corpus / 'derive_targets_summary.json')]},
        ('rewrite_summary' if role_for(m) in {'SoftSF10', 'B100V10'} else 'mix_summary'): {
            'path': str(corpus / recipe_summary_name(m)),
            'sha256': input_pins(m)[str(corpus / recipe_summary_name(m))]},
    }
    arena.require(all(receipt.get(key) == value for key, value in expected.items()),
                  'data qualification failed or profile/corpus/final identities differ')


def verify_window_cadence(summary):
    windows = summary['train_window_metrics']
    batches, rows = summary['sampling']['batches_realized'], summary['sampling']['rows_realized']
    arena.require(summary['train_windows'] == len(windows) == (batches + 87) // 88, 'training window count differs')
    for index, window in enumerate(windows, 1):
        steps = min(88, batches - (index - 1) * 88)
        arena.require(window['window_index'] == index and window['steps_requested'] == steps
                      and window['train_steps_done'] == steps and window['steps_cumulative'] == min(index * 88, batches)
                      and type(window['train_samples_seen']) is int and 0 < window['train_samples_seen'] <= steps * 512,
                      'training window cadence/cumulative counts differ')
    arena.require(sum(window['train_samples_seen'] for window in windows) == rows, 'training window sample total differs')


def check_pins(m):
    arena.pin(__file__, m['launcher_sha256'])
    arena.pin(arena.__file__, m['stage_helper_sha256' if training_only(m) else 'arena_launcher_sha256'])
    for path, digest in input_pins(m).items():
        arena.pin(path, digest)
    if not training_only(m):
        arena.pin(C_CHECKPOINT, C_SHA)
        arena.pin(arena.BOOK, arena.BOOK_SHA)
    if 'profile' in m:
        if not training_only(m):
            arena.pin(m['reader']['path'], m['reader']['sha256'])
        arena.pin(m['data_qualification']['path'], m['data_qualification']['sha256'])
        verify_data_qualification(m)
        for reference in {cell[0] for cell in comparisons(m)} - {'C20T05'}:
            control = next(item for item in arena.CHECKPOINTS.values() if item[0] == reference)
            arena.pin(control[1], control[2])
    else:
        arena.pin(arena.READER, arena.READER_SHA)
    for key in ('preregistration', 'prospective_schedule'):
        arena.pin(m[key]['path'], m[key]['sha256'])
    corpus = corpus_for(m)
    mix = arena.read(corpus / recipe_summary_name(m))
    derived = arena.read(corpus / 'derive_targets_summary.json')
    expected_postprocess = {k: v for k, v in mix.items() if k != 'outputs'} if role_for(m) == 'SoftSF10' else mix
    if role_for(m) == 'B100V10':
        verify_value_recipe(m, mix, derived)
    else:
        arena.require(derived['policy_target_postprocess'] == expected_postprocess, 'published recipe lineage differs')
    if role_for(m) == 'B100V10':
        pass
    elif role_for(m) == 'SoftSF10':
        verify_softsf_recipe(m, mix, derived)
    elif 'profile' in m:
        kind, algorithm, alpha = ('c20-global', 'stored-c20t05-then-global-bt4-v1', .2) if role_for(m) == 'H20' else (
            'global', 'legal-normalized-global-arithmetic-v1', 1. if role_for(m) == 'B100' else .5)
        arena.require(mix['kind'] == kind and mix['algorithm'] == algorithm and mix['alpha'] == alpha
                      and mix['bt4_temperature'] == .5 and mix['rows'] == 18910484
                      and mix['expected_shards'] == 2309 and mix['source_dir'] == str(SOURCE)
                      and mix['source_derive_summary_sha256'] == COMMON_PINS[str(SOURCE / 'derive_targets_summary.json')]
                      and mix['mutated_arrays'] == ['policy_target'], 'registered teacher recipe differs')
        if role_for(m) == 'H20':
            parent = mix['c20_parent']
            for path, digest in C_CORPUS_PINS.items():
                arena.pin(path, digest)
            arena.require(parent['source_dir'] == str(C_CORPUS)
                          and parent['derive_summary'] == {'path': str(C_CORPUS / 'derive_targets_summary.json'),
                              'sha256': C_CORPUS_PINS[str(C_CORPUS / 'derive_targets_summary.json')]}
                          and parent['mix_summary'] == {'path': str(C_CORPUS / 'bt4_policy_mix_summary.json'),
                              'sha256': C_CORPUS_PINS[str(C_CORPUS / 'bt4_policy_mix_summary.json')]}
                          and parent['policy_target_postprocess'] == arena.read(C_CORPUS / 'bt4_policy_mix_summary.json'),
                          'H20 parent is not the qualified stored C corpus')
    return arena.runtime_identity(m['runtime_manifest'])


def training_runtime_probe(rt):
    """Verify the same training environment without loading arena-only evidence."""
    code = f"""import contextlib,json,sys
with contextlib.redirect_stdout(sys.stderr):
    import importlib,torch,numpy
    modules={list(rt['native_extensions'])!r}
    actual=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,
        cuda=torch.version.cuda,numpy=numpy.__version__,
        native_extensions={{m:importlib.import_module(m).__file__ for m in modules}})
print(json.dumps(actual))
"""
    actual = json.loads(subprocess.check_output([rt['executable'], '-c', code], cwd=arena.RUNTIME,
                                               env=arena.environment(), text=True, timeout=60))
    arena.require(actual == {k: v for k, v in rt.items() if k != 'native_extension_sha256'}, 'actual training runtime differs')
    return actual


def verify_schedule(report, *, prospective, m=None):
    m = {} if m is None else m
    role, corpus = role_for(m), corpus_for(m)
    arena.require(report['verifier_sha256'] == INPUT_PINS[str(VERIFIER)] and report['seed'] == 0
                  and report['batch_size'] == 512 and report['runtime']['numpy'] == '1.26.2', 'schedule verifier/runtime differs')
    plan = report['source_plan']
    arena.require(plan['plan_sha256'] == CANONICAL and plan['rows_planned'] == 18910484
                  and plan['batches_planned'] == 36935, 'source canonical epoch differs')
    required = {role, 'C'} if prospective else {role}
    arena.require(set(report['arms']) == required, 'schedule arms differ')
    for arm_role, arm in report['arms'].items():
        expected = corpus if arm_role == role else C_CHECKPOINT.parent
        if arm_role == 'C':
            arena.require(arm['summary_sha256'] == INPUT_PINS[str(expected / 'summary.json')]
                          and arm['training_completion_verified'] is True, 'C completed schedule differs')
        else:
            arena.require(arm['corpus'] == str(corpus), 'candidate schedule corpus differs')
            arena.require(arm['staging'] == ('prospective only' if prospective else 'verified actual'), 'schedule stage differs')
            if not prospective:
                arena.require(arm['training_completion_verified'] is True, 'candidate realized schedule incomplete')
        arena.require(arm['metadata_matches_source'] is True and arm['canonical_plan_sha256'] == CANONICAL,
                      f'{role}: ordered source/game schedule differs')


def train_command(m):
    python = arena.read(m['runtime_manifest']['path'])['runtime']['executable']
    return [python, 'scripts/lc0_control_train.py', '--config', 'configs/lc0_positive_control.yaml',
            '--shards', str(corpus_for(m)), '--out-dir', m['run'], '--steps', '0', '--batch-size', '512',
            '--sampling-mode', 'game_epoch', '--epoch-plan-workers', '16', '--epoch-load-workers', '16',
            '--seed', '0', '--device', 'cuda', '--train-window-steps', '88', '--allow-invalid-control']


def completed_training(m, schedule_path):
    run = Path(m['run'])
    summary = arena.read(run / 'summary.json')
    report = arena.read(schedule_path)
    verify_schedule(report, prospective=False, m=m)
    arm = report['arms'][role_for(m)]
    arena.require(arm['summary_sha256'] == arena.sha(run / 'summary.json'), 'verified training summary changed')
    sampling = summary['sampling']
    expected = {'mode': 'game_epoch', 'complete': True, 'seed': 0, 'batch_size': 512,
                'rows_planned': 18910484, 'rows_realized': 18910484, 'batches_planned': 36935,
                'batches_realized': 36935, 'shards': 2309, 'games': 97968, 'plan_workers': 16,
                'load_workers': 16, 'same_game_repeats_max': 0, 'decoded_rows_resident': 0}
    arena.require(all(sampling.get(k) == v for k, v in expected.items()), 'incomplete/mismatched exact epoch')
    arena.require(sampling['plan_sha256'] == sampling['realized_sha256'] == arm['physical_plan_sha256'],
                  'actual staging and realized training schedule differ')
    arena.require(summary['seed'] == 0 and summary['batch_size'] == 512 and summary['train_window_steps'] == 88
                  and summary['steps_realized'] == summary['compute_loss_calls'] == 36935
                  and summary['corpus']['shard_dirs'] == [str(corpus_for(m))], 'training settings differ')
    windows = summary['train_window_metrics']
    arena.require(len(windows) == 420 and all(w['grad_nonfinite_skip_rate'] == 0
                  and w['transient_cuda_retry_batches'] == 0 and math.isfinite(w['loss'])
                  and math.isfinite(w['grad_norm_mean']) for w in windows), 'training skipped/retried or nonfinite windows')
    if 'profile' in m:
        verify_window_cadence(summary)
    checkpoint = {'role': role_for(m), 'path': str(run / 'checkpoint.pt'), 'sha256': arena.sha(run / 'checkpoint.pt')}
    arena.require(any(c['role'] == 'last' and c['path'] == checkpoint['path'] and c['sha256'] == checkpoint['sha256']
                      for c in summary['checkpoints']), 'completed checkpoint differs')
    return {'complete': True, 'role': role_for(m), 'run': str(run), 'checkpoint': checkpoint,
            'summary_sha256': arm['summary_sha256'], 'canonical_plan_sha256': CANONICAL,
            'physical_plan_sha256': arm['physical_plan_sha256'],
            'schedule': {'path': str(schedule_path), 'sha256': arena.sha(schedule_path)},
            'historical_valid_control': summary['valid_control'], 'historical_validity_problems': summary['validity_problems']}


def execute(m):
    validate(m)
    rt = check_pins(m)
    actual = training_runtime_probe(rt) if training_only(m) else arena.runtime_probe(rt)
    verify_schedule(arena.read(m['prospective_schedule']['path']), prospective=True, m=m)
    state, run = Path(m['state']), Path(m['run'])
    arena.disk_guard(state)
    arena.disk_guard(run)
    state.mkdir(exist_ok=False)
    arena.write(state / 'manifest.json', m)
    try:
        with (ROOT / 'scratchpad/gpu0_experiment.lock').open('a') as lease:
            arena.acquire_gpu_lease(lease)
            check_pins(m)
            arena.require(not run.exists() and not (state / 'STOP').exists(), 'existing run or stop requested')
            arena.disk_guard(state)
            arena.disk_guard(run)
            arena.require(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                                       text=True, timeout=10).strip(), 'competing GPU process')
            charge = arena.run_owned_stage(train_command(m), state / 'training', m['training_seconds'], lease.fileno(),
                'training', {'runtime': actual, 'launcher_sha256': m['launcher_sha256'], 'input_pins': input_pins(m)},
                manifest=m, stop_paths=(state / 'STOP',))
        arena.require(0 < charge['gpu_seconds'] <= m['training_seconds'], 'training charge exceeds cap')
        check_pins(m)
        # No GPU lease during metadata-only schedule reconstruction. Raw BT4 may run.
        schedule = state / 'realized_schedule.json'
        command = ['/usr/bin/nice', '-n', '19', '/usr/bin/ionice', '-c', '3', '/usr/bin/taskset', '-c', '0,1',
                   rt['executable'], str(VERIFIER), '--run', f'{role_for(m)}={run}', '--output', str(schedule)]
        arena.run_owned_stage(command, state / 'schedule', 1800, None, 'schedule',
                              {'gpu_stage': False}, manifest=m, stop_paths=(state / 'STOP',))
        check_pins(m)
        completion = completed_training(m, schedule)
        completion.update(training_charge_seconds=charge['gpu_seconds'], input_pins=input_pins(m))
        arena.write(state / 'training.complete.json', completion)
        total = charge['gpu_seconds']
        if training_only(m):
            arena.write(state / 'complete.json', {
                'scope': 'training_only', 'training_only_complete': True,
                'profile': role_for(m), 'gpu_seconds': total,
                'training_receipt_sha256': arena.sha(state / 'training.complete.json'),
                'promotion': 'NONE; no arena executed',
            })
            return
        arena_receipts = []
        opening_anchor = None
        controls = {item[0]: item for item in arena.CHECKPOINTS.values()}
        for reference, sims, games in comparisons(m):
            arena.require(total + m['arena_seconds'] <= m['total_seconds'], 'remaining registered GPU allowance is insufficient')
            arena.require(not (state / 'STOP').exists(), 'stop requested before arena')
            ref_role, checkpoint, checkpoint_sha = controls[reference]
            label = f'{reference}.s{sims}' if 'profile' in m else 'arena'
            out = state / label
            direct = {'schema': 2 if 'profile' in m else 1, 'output': str(out), 'sims': sims,
                      'hard_seconds': m['arena_seconds'], 'candidate': completion['checkpoint'],
                      'reference': {'role': ref_role, 'path': str(checkpoint), 'sha256': checkpoint_sha},
                      'candidate_training': {'path': str(state / 'training.complete.json'), 'sha256': arena.sha(state / 'training.complete.json')},
                      'book': {'path': str(arena.BOOK), 'sha256': arena.BOOK_SHA}, 'runtime_manifest': m['runtime_manifest'],
                      'preregistration': m['preregistration'], 'launcher_sha256': m['arena_launcher_sha256']}
            if 'profile' in m:
                direct.update(games=games, reader=m['reader'])
                if games == 500:
                    arena.require(opening_anchor is not None, 'C100 opening anchor required before probe')
                    direct['opening_anchor'] = opening_anchor
            arena.write(state / (f'{label}.manifest.json' if 'profile' in m else 'arena_manifest.json'), direct)
            arena.execute(direct, stop_paths=(state / 'STOP',))
            arena_done = arena.read(out / 'complete.json')
            arena.require(arena_done['complete'] is True and 0 < arena_done['gpu_seconds'] <= m['arena_seconds'], 'arena incomplete or charge exceeds cap')
            total += arena_done['gpu_seconds']
            arena.require(total <= m['total_seconds'], 'total charge exceeded')
            arena_receipts.append({'path': str(out / 'complete.json'), 'sha256': arena.sha(out / 'complete.json')})
            if reference == 'C20T05' and sims == 100:
                opening_anchor = {'bank': {'path': str(out / 'arena.games.jsonl'), 'sha256': arena_done['games_sha256']},
                                  'completion': arena_receipts[-1]}
        final = {'complete': True, 'gpu_seconds': total, 'training_receipt_sha256': arena.sha(state / 'training.complete.json'),
                 'promotion': 'NONE; apply preregistered rule'}
        if 'profile' in m:
            final.update(profile=role_for(m), arena_receipts=arena_receipts)
        else:
            final['arena_receipt_sha256'] = arena_receipts[0]['sha256']
        arena.write(state / 'complete.json', final)
    except BaseException as error:
        arena.write(state / 'failed.json', {'complete': False, 'error': str(error), 'partial_artifacts_preserved': True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    m = arena.read(args.manifest)
    validate(m)
    if not args.execute:
        plan = {'execute': False, 'training_command': train_command(m),
                'comparisons': comparisons(m), 'profile': role_for(m)}
        if training_only(m):
            plan.update(scope='training_only', training_seconds=m['training_seconds'], schedule_seconds=1800)
        else:
            plan['total_seconds'] = m['total_seconds']
        print(json.dumps(plan, indent=2))
        return
    def interrupted(signum, _frame):
        raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    execute(m)


if __name__ == '__main__':
    main()
