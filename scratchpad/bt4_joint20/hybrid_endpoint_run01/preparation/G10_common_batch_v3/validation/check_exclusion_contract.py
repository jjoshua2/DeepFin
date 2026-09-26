from pathlib import Path
import copy
import hashlib
import importlib.util
import json

state = Path('/home/josh/projects/chess/scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/G10_common_batch_v3')
fixture = state / 'validation/actual_small_fixture_final'
spec = importlib.util.spec_from_file_location('common_contract_fixture', state / 'run_common.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
summary = json.loads((fixture / 'serial/derive_targets_summary.json').read_text())
def pin(name):
    path = fixture / name
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
source = {'source_dir': summary['realized']['policy_support_exclusions'][0]['source_dir'],
          'raw_rows': 8, 'max_policy_support_misses': 2, 'expected_no_result_rows': 1,
          'expected_emitted_rows': 5, 'expected_policy_support_exclusions': pin('expected_support.jsonl'),
          'expected_no_result_exclusions': pin('expected_no_result.jsonl'),
          'selection': [{'source_shard': 'w00-00000.jsonl.zst', 'raw_rows': 4},
                        {'source_shard': 'w00-00001.jsonl.zst', 'raw_rows': 4}]}
checks = []
for kind in ['serial', 'spawn']:
    actual = json.loads((fixture / kind / 'derive_targets_summary.json').read_text())
    assert len(module.expected_exclusions(source, actual)) == 3
    checks.append(kind + '_actual_completed_summary_accepted')
for name, change in [
    ('different_raw_ref', lambda value: value['realized']['policy_support_exclusions'][0].update(source_row=999)),
    ('unverified_history', lambda value: value['realized']['policy_support_exclusions'][0].update(full_history_input_key_verified=False)),
    ('unexpected_envelope', lambda value: value['realized'].update(rows_dropped_envelope=1)),
    ('wrong_no_result_count', lambda value: value['realized'].update(rows_dropped_no_result=2)),
]:
    bad = copy.deepcopy(summary)
    change(bad)
    try:
        module.expected_exclusions(source, bad)
    except ValueError:
        checks.append(name + '_refused')
    else:
        raise AssertionError(name)
bad_source = copy.deepcopy(source)
bad_source['selection'] = [{'source_shard': 'different.jsonl.zst', 'raw_rows': 8}]
try:
    module.expected_exclusions(bad_source, summary)
except ValueError:
    checks.append('excluded_refs_outside_source_universe_refused')
else:
    raise AssertionError('out-of-universe')
receipt = {'status': 'UNIT_CONTRACT_CHECKS_PASS_NOT_OPERATIONAL_QUALIFICATION',
           'checks': checks, 'runner_sha256': hashlib.sha256((state / 'run_common.py').read_bytes()).hexdigest(),
           'scope': 'Reused real final synthetic serial/spawn outputs; no derivation or corpus qualification rerun.'}
with (fixture / 'root_contract_checks_final.json').open('x') as handle:
    json.dump(receipt, handle, indent=2)
    handle.write('\n')
print(json.dumps(receipt))
