"""UNRUN actual normal-import/caller controls; no model/session execution or stubs."""
import hashlib
import importlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np
_FORBIDDEN_PREFIXES = ('torch', 'scripts.gen_sf_rooted_corpus', 'scripts.gen_random_selfplay_shards',
                       'scripts.bt4_raw_corpus_sidecar', 'scripts.derive_corpus_targets')


def _forbidden_modules():
    return frozenset(name for name in sys.modules
                     if any(name == prefix or name.startswith(prefix + '.') for prefix in _FORBIDDEN_PREFIXES))


def _load_test_subjects():
    # Warm CI may already contain Torch/corpus through conftest or earlier files.
    # Measure only this exact ordinary target-import sequence, at collection time.
    before = _forbidden_modules()
    rep = importlib.import_module('chess_anti_engine.encoding.rep_fix')
    features = importlib.import_module('chess_anti_engine.encoding.features')
    lc0 = importlib.import_module('chess_anti_engine.encoding.lc0')
    actual_evaluator = importlib.import_module('scripts.bt4_generation_evaluator')
    actual_contracts = importlib.import_module('scripts.bt4_root_contracts')
    actual_worker = importlib.import_module('scripts.bt4_root_policy_worker')
    actual_owner = importlib.import_module('scripts.shared_teacher_owner')
    after = _forbidden_modules()  # Capture immediately, before unrelated collection.
    return (rep, features.EXTRA_FEATURES_V2_THREATS, lc0.LC0_HISTORY_ROOT_LEGACY_META,
            actual_evaluator, actual_contracts, actual_worker, actual_owner, after - before)


(rep_fix, EXTRA_FEATURES_V2_THREATS, LC0_HISTORY_ROOT_LEGACY_META,
 evaluator, contracts, worker, shared_teacher_owner, _INTRODUCED_FORBIDDEN_MODULES) = _load_test_subjects()

class MetadataSession:
    def __init__(self, outputs):
        self.outputs = outputs
    def get_outputs(self):
        return self.outputs
    def run(self, *_args, **_kwargs):
        raise AssertionError('no inference in utility controls')

def output(name, shape, dtype='tensor(float)'):
    return SimpleNamespace(name=name, shape=shape, type=dtype)

class UtilityBoundaryControls(unittest.TestCase):
    def test_normal_selected_worker_imports_do_not_load_torch_or_corpus(self):
        assert callable(shared_teacher_owner.run_owned_roster)
        assert callable(worker.run_pooled_units)
        # This public test proves no introduced forbidden prefix; warm CI cannot
        # prove global absence. The isolated controller separately enforces it.
        assert not _INTRODUCED_FORBIDDEN_MODULES
    def test_authoritative_export_identity_and_existing_class_origin(self):
        assert evaluator.input_tensor_key is contracts.input_tensor_key
        assert evaluator.resolve_wdl_output is contracts.resolve_wdl_output
        assert evaluator.BT4RootOutput is contracts.BT4RootOutput
        assert evaluator.BT4RootOutput.__module__ == 'scripts.bt4_root_contracts'
    def test_worker_constant_public_exports_preserved(self):
        assert worker.INPUT_HISTORY_ENCODING == LC0_HISTORY_ROOT_LEGACY_META
        assert worker.INPUT_EXTRA_FEATURES == EXTRA_FEATURES_V2_THREATS
        assert (worker.INPUT_HISTORY_ENCODING, worker.INPUT_EXTRA_FEATURES) == ('lc0_root_legacy_meta', 'v2_threats')
    def test_actual_input_key_caller_preserves_float32_c_order(self):
        base = np.arange(64, dtype=np.float64).reshape(8, 8)
        for row in (base, base.T, base.astype(np.float32)):
            expected = hashlib.blake2b(np.ascontiguousarray(row, dtype=np.float32).tobytes(), digest_size=16).hexdigest()
            assert evaluator.input_tensor_key(row) == expected
    def test_resolver_explicit_positive_types_and_none(self):
        assert evaluator.resolve_wdl_output(MetadataSession([]), None, 'policy') is None
        for kind, dtype in (('tensor(float16)', 'float16'), ('tensor(float)', 'float32'), ('tensor(double)', 'float64')):
            session = MetadataSession([output('wdl', ['N', 3], kind)])
            actual = evaluator.resolve_wdl_output(session, {'output': 'wdl', 'kind': 'probabilities'}, 'policy')
            assert actual == {'output': 'wdl', 'kind': 'probabilities', 'dtype': dtype}
    def test_resolver_refuses_missing_duplicate_policy_alias_and_bad_layout(self):
        for outputs, policy in (([], 'policy'), ([output('wdl', [1, 3])] * 2, 'policy'),
                                ([output('wdl', [1, 3])], 'wdl'), ([output('wdl', [1, 2])], 'policy'),
                                ([output('wdl', [1, 3], 'tensor(int64)')], 'policy')):
            try:
                evaluator.resolve_wdl_output(MetadataSession(outputs), {'output': 'wdl', 'kind': 'logits'}, policy)
            except ValueError:
                pass
            else:
                raise AssertionError('expected ValueError for invalid explicit WDL contract')
    def construct(self, outputs, *, wdl='wdl'):
        prior = rep_fix.current()
        primary = None
        try:
            rep_fix.apply(True, boards_discarded=True)
            return evaluator.BT4OnnxEvaluator(MetadataSession(outputs), input_name='input', input_dtype=np.dtype('float32'),
                policy_output='policy', wdl_output=wdl, wdl_kind='probabilities',
                input_history_encoding=worker.INPUT_HISTORY_ENCODING, input_extra_features=worker.INPUT_EXTRA_FEATURES,
                model_sha256='a' * 64, history_rep_fix=True)
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                rep_fix.apply(bool(prior), boards_discarded=True)
                if prior is None:
                    # Same never-set sentinel restoration as canonical conftest.
                    setattr(rep_fix, '_current', None)
            except BaseException as cleanup_error:
                if primary is not None:
                    # Retain the original type/object and expose unsafe cleanup.
                    setattr(primary, 'rep_fix_cleanup_error', cleanup_error)
                    raise primary from cleanup_error
                raise
    def test_actual_evaluator_constructor_uses_named_resolver_without_inference(self):
        actor = self.construct([output('policy', ['N', 1858]), output('wdl', ['N', 3])])
        assert actor.wdl_contract == {'output': 'wdl', 'kind': 'probabilities', 'dtype': 'float32'}
        assert (actor.root_calls, actor.root_rows, actor.leaf_calls) == (0, 0, 0)
    def test_actual_evaluator_constructor_keeps_missing_wdl_failure(self):
        try:
            self.construct([output('policy', ['N', 1858])])
        except ValueError:
            pass
        else:
            raise AssertionError('expected ValueError for missing named WDL output')
