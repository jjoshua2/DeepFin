"""Synthetic stdlib controls only; no operational artifacts or array imports."""
import copy
import json
import math
import pathlib
import tempfile
import unittest
from typing import Any
import pytest
from scripts import bt4_fp16_conversion_contract_v1 as p

class PrecisionControls(unittest.TestCase):
    def fixture(self, root: pathlib.Path) -> tuple[dict[str, Any], dict[str, Any], pathlib.Path, str]:
        evidence = {}
        for kind in p.KINDS:
            path = root / (kind + '.json')
            path.write_bytes(b'{}')
            evidence[kind] = {'path': str(path), 'sha256': p.digest_bytes(b'{}')}
        entry = {'bank': 'synthetic', 'unit': 'synthetic-unit', 'game': 0,
                 'model_sha256': p.MODEL, 'checkpoint_binding_sha256': 'b'*64,
                 'receipt': {'path': str(root/'raw.npz'), 'sha256': 'c'*64, 'rows': 1}}
        doc = {'schema': p.SCHEMA, 'status': 'INDEPENDENTLY_VERIFIED_EXACT809_RAW_PROVENANCE',
               'roster_sha256': 'a'*64, 'model_sha256': p.MODEL, 'arithmetic_precision': 'fp16',
               'storage_dtype': 'float32', 'evidence': evidence, 'games': [entry]}
        path = root/'manifest.json'
        raw = json.dumps(doc).encode()
        path.write_bytes(raw)
        return entry, doc, path, p.digest_bytes(raw)

    def test_authenticated_positive_and_ordinary_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry, _, path, sha = self.fixture(pathlib.Path(tmp))
            doc = p.load_manifest(path, sha, 'a'*64)
            assert p.mass_tolerance(entry) == 2e-6
            row = [.5, .25, .250244140625]
            with pytest.raises(ValueError, match='WDL mass'):
                p.check_mass_rows([row], p.mass_tolerance(entry))
            entry['_exact809_context'] = p.authenticated_context(doc, entry)
            assert p.mass_tolerance(entry) == 2**-10
            p.check_mass_rows([row, [.5, .25, .2503662109375]], p.mass_tolerance(entry))

    def test_forged_json_context_and_wrong_raw_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry, doc, path, sha = self.fixture(pathlib.Path(tmp))
            doc = p.load_manifest(path, sha, 'a'*64)
            context = p.authenticated_context(doc, entry)
            entry['_exact809_context'] = dict(context)
            with pytest.raises(ValueError, match='authenticated context mismatch'):
                p.mass_tolerance(entry)
            entry['_exact809_context'] = context
            entry['receipt']['sha256'] = 'd'*64
            with pytest.raises(ValueError, match='authenticated context mismatch'):
                p.mass_tolerance(entry)
            with pytest.raises(ValueError, match='raw game absent from reviewed provenance'):
                p.authenticated_context(doc, {**entry, 'game': 1})

    def test_stale_manifest_evidence_and_roster(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, doc, path, sha = self.fixture(pathlib.Path(tmp))
            with pytest.raises(ValueError, match='provenance hash changed'):
                p.load_manifest(path, '0'*64, 'a'*64)
            with pytest.raises(ValueError, match='exact roster/model/precision'):
                p.load_manifest(path, sha, 'd'*64)
            pathlib.Path(doc['evidence']['descriptor']['path']).write_bytes(b'{"changed":true}')
            with pytest.raises(ValueError, match='provenance evidence changed'):
                p.load_manifest(path, sha, 'a'*64)

    def test_closed_schema_precision_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, doc, _, _ = self.fixture(pathlib.Path(tmp))
            for mutate in (lambda d:d.update(extra=True), lambda d:d.update(arithmetic_precision='fp32'),
                           lambda d:d.update(model_sha256='0'*64), lambda d:d['games'].append(copy.deepcopy(d['games'][0])),
                           lambda d:d['evidence'].pop('checkpoint')):
                bad = copy.deepcopy(doc)
                mutate(bad)
                with pytest.raises(ValueError, match=r'closed provenance schema|exact roster/model/precision|duplicate provenance game|closed evidence kinds'):
                    p.validate_manifest(bad, 'a'*64)

    def test_numerical_negatives(self):
        for row in ([.5,.25,.252], [-.01,.5,.51], [1.01,0,0], [float('nan'),0,1],
                    [float('inf'),0,0], [0,0,0], [.5,.5]):
            with pytest.raises(ValueError, match=r'WDL (range/finite/shape|mass)'):
                p.check_mass_rows([row], 2**-10)
        p.check_mass_rows([[.5,.25,.25]], 2e-6)

    def test_exact_mass_boundary(self):
        p.check_mass_rows([[.5, .25, .25 + 2**-10]], 2**-10)
        above = math.nextafter(1.0 + 2**-10, math.inf) - .75
        with pytest.raises(ValueError, match='WDL mass'):
            p.check_mass_rows([[.5, .25, above]], 2**-10)

    def test_authenticated_context_cannot_transfer_to_other_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry, doc, path, sha = self.fixture(pathlib.Path(tmp))
            doc = p.load_manifest(path, sha, 'a'*64)
            entry['_exact809_context'] = p.authenticated_context(doc, entry)
            entry['model_sha256'] = '0' * 64
            with pytest.raises(ValueError, match='authenticated context mismatch'):
                p.mass_tolerance(entry)

    def test_distinct_raw_alias_and_physical_game_duplicates_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, doc, _, _ = self.fixture(pathlib.Path(tmp))
            duplicate_uid = copy.deepcopy(doc['games'][0])
            duplicate_uid['receipt']['sha256'] = 'd' * 64
            doc['games'].append(duplicate_uid)
            with pytest.raises(ValueError, match='duplicate physical game'):
                p.validate_manifest(doc, 'a' * 64)
            doc['games'][1]['game'] = 1
            doc['games'][1]['receipt']['sha256'] = 'c' * 64
            with pytest.raises(ValueError, match='duplicate raw content'):
                p.validate_manifest(doc, 'a' * 64)

    def test_raw_dictionary_cannot_bypass_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry, doc, _, _ = self.fixture(pathlib.Path(tmp))
            with pytest.raises(ValueError, match='manifest was not loaded through trusted evidence gate'):
                p.authenticated_context(doc, entry)
            with pytest.raises(ValueError, match='manifest loader token'):
                p.LoadedManifest(doc, object())

if __name__ == '__main__':
    unittest.main()
