"""Exact successful-audit exclusions, separate from ordinary no-result drops."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError('baseline exclusions: ' + message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(ref: dict[str, str]) -> dict[str, Any]:
    p = Path(ref['path'])
    require(p.is_absolute() and p == p.resolve(), 'noncanonical pin')
    require(sha(p) == ref['sha256'], 'changed pin')
    return json.loads(p.read_text())


@dataclass(frozen=True)
class Exclusions:
    proof: dict[str, Any]
    selection: dict[str, Any]
    rows: tuple[dict[str, Any], ...]

    def bind(self, selection: dict[str, Any] | None) -> None:
        require(selection is not None, 'source-shards selection required')
        if selection is None:
            raise ValueError('selection missing')
        require(all(selection.get(k) == v for k, v in self.selection.items()), 'selected roster differs')
        for ref in self.proof['pins']:
            require(sha(Path(ref['path'])) == ref['sha256'], 'changed exclusion evidence')

    def pending(self, shards: list[Path]) -> dict[tuple[str, int], dict[str, Any]]:
        names = {p.name for p in shards}
        return {(r['source_shard'], r['source_row']): r for r in self.rows if r['source_shard'] in names}


def consume(pending: dict[tuple[str, int], dict[str, Any]], path: Path, index: int,
            row: dict[str, Any]) -> bool:
    ref = pending.pop((path.name, index), None)
    if ref is None:
        return False
    require(row.get('result') is not None, 'explicit exclusion became no-result')
    require(all(type(row.get(k)) is int and row[k] == ref[k] for k in ('game_id', 'ply')), 'row identity mismatch')
    require(type(row.get('worker_id')) is int and row['worker_id'] == ref['worker'], 'worker identity mismatch')
    return True


def load(path: Path) -> Exclusions:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    require(manifest['schema'] == 1, 'manifest schema')
    audit = read(manifest['audit'])
    require(audit['status'] == 'PASS_BASELINE_ELIGIBILITY_NOT_DERIVATION', 'audit incomplete')
    selection = read(manifest['selection'])
    shards = audit['shards']
    admission = audit['manifest']
    require(('collection' in admission) != ('receipt_selection' in admission), 'one receipt admission route required')
    saved = 'receipt_selection' in admission
    require(0 < len(shards) <= (512 if saved else 192), 'audit shard cap')
    indexed = {(s['source_namespace'], s['source_shard']): s for s in shards}
    require(len(indexed) == len(shards), 'duplicate audited shard')
    if saved:
        # importlib: a `from scripts import` still counts in basedpyright's
        # import-cycle graph with derive_corpus_targets.
        # Reuse the audit's complete metadata admission, including typed WDL and
        # exact snapshot membership; never synthesize a collector receipt.
        import importlib
        auditor = importlib.import_module("scripts.audit_raw_baseline")

        admitted = auditor.selection(admission, lambda: None, max_shards=512)
        require(len(admitted) == len(shards) and all(
            all(observed.get(k) == v for k, v in expected.items())
            for observed, expected in zip(shards, admitted)), 'audit/receipt selection roster differs')
        selected_receipts = read(admission['receipt_selection'])
        admission_pins = [admission['receipt_selection'],
                          *[source['manifest'] for source in admission['sources']],
                          *[{k: ref[k] for k in ('path', 'sha256')}
                            for ref in selected_receipts['receipt_snapshots']]]
    else:
        collection = read(admission['collection'])
        require(collection['status'] == 'BOUNDED_RAW_LABEL_COLLECTION_COMPLETE', 'collection incomplete')
        actual = {(s['source_id'], s['source_shard'], s['source_sha256'], s['rows']) for s in shards}
        expected = {(s['source_id'], s['source_shard'], s['source_sha256'], s['positions']) for s in collection['receipts']}
        require(actual == expected and len(expected) == collection['new_shards'] == len(shards), 'audit/collection roster differs')
        require(sum(s['rows'] for s in shards) == collection['new_rows'], 'collection rows differ')
        admission_pins = [admission['collection']]
    chosen = [s for s in shards if s['source_dir'] == selection['source_dir']]
    require(bool(chosen), 'source absent from audit')
    require(all(s['config_sha256'] == selection['source_config_sha256'] for s in chosen), 'configuration differs')
    selected = {(s['source_shard'], s['source_sha256'], s['rows']) for s in selection['shards']}
    require(selected == {(s['source_shard'], s['source_sha256'], s['rows']) for s in chosen}
            and len(selected) == len(selection['shards']), 'full audited source roster required')
    sources = [s for s in audit['manifest']['sources'] if s['source_dir'] == selection['source_dir']]
    require(len(sources) == 1 and sources[0]['manifest']['sha256'] == selection['source_manifest_sha256'], 'source manifest differs')
    ref = manifest['diagnostics']
    p = Path(ref['path'])
    require(p.stat().st_size == audit['diagnostic_bytes'] <= 512 * 1024**2 and sha(p) == ref['sha256'], 'diagnostic size/hash')
    counts = {key: Counter() for key in indexed}
    seen = set()
    rows = []
    with p.open() as f:
        for line in f:
            r = json.loads(line)
            key = (r['source_namespace'], r['source_shard'])
            require(key in indexed, 'unknown diagnostic source')
            s = indexed[key]
            require(all(r[k] == s[k] for k in ('source_id', 'source_dir', 'config_sha256', 'source_sha256', 'rows', 'worker')), 'diagnostic source mismatch')
            require(type(r['source_row']) is int and 0 <= r['source_row'] < s['rows'], 'physical index outside shard')
            require(all(type(r[k]) is int and r[k] >= 0 for k in ('game_id', 'ply')), 'malformed row ID')
            identity = (*key, r['source_row'])
            require(identity not in seen, 'duplicate exclusion ID')
            seen.add(identity)
            require(r['eligible'] is False and type(r['no_result']) is bool, 'diagnostic eligibility')
            c = counts[key]
            c['no_result_rows'] += int(r['no_result'])
            for kind in ('policy', 'value'):
                error = r[kind + '_error']
                require(error is None or isinstance(error, str), 'malformed rejection reason')
                c[kind + '_rejected_rows'] += int(error is not None)
            if not r['no_result']:
                require(r['policy_error'] is not None or r['value_error'] is not None, 'exclusion without baseline failure')
                c['required_baseline_rejected_rows'] += 1
                if s in chosen:
                    rows.append(r)
    totals = Counter()
    for key, s in indexed.items():
        c = counts[key]
        c['physical_rows'] = s['rows']
        c['eligible_rows'] = s['rows'] - c['no_result_rows'] - c['required_baseline_rejected_rows']
        require(all(c[k] == s['counts'][k] for k in c), 'diagnostic/audit counts differ')
        totals.update(c)
    require(all(v == audit['counts'][k] for k, v in totals.items()), 'audit aggregate counts differ')
    pins = [manifest[k] for k in ('audit', 'selection', 'diagnostics')] + admission_pins
    proof = {'schema': 1, 'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
             'pins': [*pins, {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()}],
             'excluded_rows': len(rows),
             'physical_rows': sum(s['rows'] for s in chosen),
             'eligible_rows': sum(s['counts']['eligible_rows'] for s in chosen),
             'no_result_rows': sum(s['counts']['no_result_rows'] for s in chosen), 'scope': 'audited required baseline failures only; no-result unchanged'}
    return Exclusions(proof, selection, tuple(rows))
