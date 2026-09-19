"""Explicit CPU-only reduced-transformer checkpoint smoke; NEVER trained weights.

Not collected by ordinary pytest. The checkpoint is temporary and uses the same
strict loader/export/qualifier as a user-supplied checkpoint, with no seed fallback.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import shutil

import torch

from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig, build_model
from .checkpoint_probe import ROOT, qualify


def fixture_checkpoint(path: Path) -> None:
    cfg = ModelConfig(kind='transformer', embed_dim=32, num_layers=2, num_heads=4,
                      ffn_mult=1.5, use_smolgen=True, smolgen_mode='per_layer',
                      smolgen_relation_basis=True, smolgen_hidden_channels=4,
                      smolgen_hidden_sz=16, smolgen_gen_sz=16, input_pos_encoding='arc_adapter',
                      qkv_projection='split', use_deepnorm=True,
                      input_history_encoding='lc0_root_legacy_meta',
                      input_extra_features='v2_threats', history_rep_fix=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260918)
        model = build_model(cfg).eval()
    # Distinguish checkpoint values from a fresh same-seed model. No training claim.
    with torch.no_grad():
        next(model.parameters()).add_(0.001)
    torch.save({'model': model.state_dict(), 'arch': {**asdict(cfg), '_schema_version': ARCH_SCHEMA_VERSION},
                'fixture_status': 'untrained reduced transformer'}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--compiler-root', type=Path, default=ROOT / 'build/bend_u64_toolchain/source')
    parser.add_argument('--modes', nargs='+', choices=('generic', 'portable', 'native', 'ubsan'), default=['native', 'ubsan'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='bend-transformer-smoke-') as tmp:
        root = Path(tmp)
        checkpoint = root / 'fixture.pt'
        fixture_checkpoint(checkpoint)
        report = qualify(checkpoint, root / 'qualification', args.compiler_root,
                         modes=tuple(args.modes), bun=shutil.which('bun') or 'bun')
        report['fixture_status'] = 'UNTRAINED reduced transformer, not a production checkpoint'
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
