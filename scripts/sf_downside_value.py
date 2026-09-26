"""Explicit Downside policy copy parent with authenticated V50/V100 bytes."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from scripts import combined_corpus_schedule as schedule


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def files(root: Path) -> dict[str, str]:
    result = {}
    for path in root.rglob('*'):
        require(not path.is_symlink(), 'value storage symlink')
        if path.is_file():
            result[str(path.relative_to(root))] = schedule.sha(path)
    return result


class ValueParent:
    """Reuse admitted value bytes; never recompute values or relax B100 admission."""
    def __init__(self, args: Any, source: Path, b100: Path, out: Path):
        self.root = Path(args.downside_value_source).resolve()
        require(all(self.root != p and self.root not in p.parents and p not in self.root.parents
                    for p in (source, b100, out, out.with_name(out.name + '.writing'))),
                'value parent overlaps input/output')
        require(not self.root.with_name(self.root.name + '.writing').exists()
                and not (self.root / 'failed.json').exists(), 'unfinished value parent')
        def ref(root: Path, name: str, digest: str) -> dict[str, str]:
            return {'path': str(root / name), 'sha256': digest}
        self.summary_ref = ref(self.root, 'derive_targets_summary.json', args.expected_downside_value_summary_sha256)
        self.recipe_ref = ref(self.root, 'bt4_value_rewrite_summary.json', args.expected_downside_value_recipe_sha256)
        self.summary = schedule.read_pin(self.summary_ref)
        self.recipe = schedule.read_pin(self.recipe_ref)
        alpha = self.recipe.get('bt4_weight')
        require(alpha in (.5, 1.), 'Downside value parent must be V50 or V100')
        arm = 'V100' if alpha == 1. else 'V50'
        cohort = {'rows': self.recipe['rows'], 'roots': {
            'source': {'summary': ref(source, 'derive_targets_summary.json', args.expected_source_summary_sha256)},
            'B100': {'summary': ref(b100, 'derive_targets_summary.json', args.expected_bt4_summary_sha256)},
            arm: {'summary': self.summary_ref}},
            'policy_recipe': ref(b100, 'bt4_policy_mix_summary.json', args.expected_bt4_mix_sha256),
            'value_recipe': self.recipe_ref}
        specs = schedule.validate_recipe(cohort, arm)
        require([p.name for p in sorted(self.root.glob('shard_*.zarr'))] == [s['path'] for s in specs],
                'value parent shard membership differs')
        self.outputs = {s['path']: s for s in self.recipe['outputs']}
        self.pins = {r['path']: r['sha256'] for r in (self.summary_ref, self.recipe_ref)}
        self.pins[str(self.root / 'bt4_policy_mix_summary.json')] = args.expected_bt4_mix_sha256
        self.binding = {'profile': arm, 'summary': self.summary_ref, 'value_recipe': self.recipe_ref,
                        'retained_arrays': 'all 16 nonpolicy arrays byte-identical to selected value parent'}

    def check_shard(self, name: str, b100: Path) -> tuple[Path, dict[str, str]]:
        """Bind actual value storage to writer receipt and unchanged B100 columns."""
        root = self.root / name
        actual = files(root)
        proof = self.outputs[name]
        require(hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
                == proof['files_manifest_sha256'], 'value parent content differs from completed writer')
        require(actual['.zattrs'] == proof['attrs_sha256'], 'value parent attrs differ')
        attrs = json.loads((root / '.zattrs').read_text())
        require(attrs.get('value_target_postprocess') == proof['stamp'], 'value shard stamp differs')
        def retained(mapping: dict[str, str]) -> dict[str, str]:
            return {k: v for k, v in mapping.items() if k != '.zattrs' and k.split('/')[0] != 'search_wdl'}
        b100_files = files(b100)
        require(retained(actual) == retained(b100_files), 'value parent changed B100 policy/nonvalue bytes')
        return root, b100_files
