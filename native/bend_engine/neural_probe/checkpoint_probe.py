"""Qualify an explicit saved checkpoint in native inference and batched Bend search.

CPU F32 is executable on CPU hosts. CUDA BF16 requires an actual GPU and explicit
predeclared numerical tolerances; no fallback or unrun CUDA PASS is possible.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import torch

from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig, build_model
from native.bend_engine.session_probe import run_probe as sessions
from .backend import BATCHES, NativeEvaluator, build_worker
from .batch_probe import group
from .checkpoint import export_checkpoint, load_checkpoint, target

ROOT = Path(__file__).resolve().parents[3]


def fixture_checkpoint(path: Path) -> None:
    """Explicit test fixture, NOT a trained checkpoint or production topology."""
    cfg = ModelConfig(kind='transformer', embed_dim=32, num_layers=2, num_heads=4,
                      use_smolgen=True, use_dynamic_relations=True, smolgen_relation_basis=True,
                      input_history_encoding='lc0_root_legacy_meta', history_rep_fix=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260918)
        model = build_model(cfg)
    # Exclusive creation: never overwrite somebody's checkpoint.
    with path.open('xb') as stream:
        torch.save({'arch': {'_schema_version': ARCH_SCHEMA_VERSION, **asdict(cfg)},
                    'model': model.state_dict(), 'step': 0}, stream)


def tolerances(device: str, atol: float | None, rtol: float | None) -> tuple[float, float]:
    if device == 'cuda' and (atol is None or rtol is None):
        raise ValueError('CUDA requires explicit predeclared --atol and --rtol')
    a, r = (2e-6 if atol is None else atol), (2e-5 if rtol is None else rtol)
    if not math.isfinite(a) or not math.isfinite(r) or a < 0 or r < 0:
        raise ValueError('tolerances must be finite and nonnegative')
    return a, r


def validate_report_destination(report: Path, checkpoint: Path | None) -> None:
    """Never let a mistaken --report argument overwrite the source checkpoint."""
    if checkpoint is None:
        return
    source = checkpoint / 'trainer.pt' if checkpoint.is_dir() else checkpoint
    if (report.resolve() == source.resolve()
            or (report.exists() and source.exists() and report.samefile(source))):
        raise ValueError('report destination must differ from the source checkpoint')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--fixture-checkpoint', action='store_true', help='Create an UNTRAINED small transformer checkpoint for CI')
    parser.add_argument('--weights-key', choices=['model', 'swa_model'], default='model')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--batch', type=int, choices=BATCHES, default=4)
    parser.add_argument('--atol', type=float)
    parser.add_argument('--rtol', type=float)
    parser.add_argument('--compiler-root', type=Path, default=ROOT / 'build/bend_u64_toolchain/source')
    parser.add_argument('--bun', default=shutil.which('bun'))
    parser.add_argument('--cc', default=shutil.which('clang'))
    parser.add_argument('--cxx', default=shutil.which('clang++'))
    parser.add_argument('--modes', nargs='+', choices=list(sessions.MODES), default=['native'])
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not args.bun or not args.cc or not args.cxx:
        parser.error('Bun, Clang and Clang++ are required')
    # Outside the reporting finally block: a rejected destination must not be written.
    validate_report_destination(args.report, args.checkpoint)
    report: dict[str, object] = {'status': 'failed', 'device': args.device,
                               'scope': 'checkpoint-native-search qualification, not playing strength or throughput'}
    torch.set_num_threads(2)
    stage = 'preflight'
    try:
        atol, rtol = tolerances(args.device, args.atol, args.rtol)
        target(args.device, args.device_index)
        with tempfile.TemporaryDirectory(prefix='bend-checkpoint-') as temp:
            work = Path(temp)
            os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(work / 'inductor')
            path = args.checkpoint
            if args.fixture_checkpoint:
                path = work / 'untrained-transformer.pt'
                fixture_checkpoint(path)
            stage = 'checkpoint-load'
            loaded = load_checkpoint(path, weights_key=args.weights_key)
            report.update({'checkpoint': loaded.identity, 'fixture_untrained': args.fixture_checkpoint,
                           'atol': atol, 'rtol': rtol, 'torch_version': str(torch.__version__)})
            stage = 'export'
            package = work / 'checkpoint.pt2'
            eager = export_checkpoint(loaded, package, batch=args.batch, device=args.device, device_index=args.device_index)
            stage = 'native-build'
            worker = build_worker(work / 'worker', args.cxx)
            binaries = sessions.build(args.compiler_root, work / 'bend', args.bun, args.cc, args.modes)
            stage = 'native-search'
            evaluator = NativeEvaluator(worker, package)
            try:
                results = []
                oracle = sessions.Oracle(binaries['reference'], with_python_chess=True)
                for mode in args.modes:
                    control, original = group(binaries[mode], oracle, evaluator, eager, max_rows=1, atol=atol, rtol=rtol)
                    results.append({'mode': mode, **control})
                    for faults in (False, True):
                        observed, final = group(binaries[mode], oracle, evaluator, eager, faults=faults, atol=atol, rtol=rtol)
                        for a, b in zip(original, final, strict=True):
                            if a.structure != b.structure or any(a.summary[k] != b.summary[k] for k in ('best', 'nodes', 'completed', 'stop')):
                                raise AssertionError('checkpoint batching/recovery changed search; retain the mismatch')
                        results.append({'mode': mode, **observed})
                evaluator.finish()
                report.update({'status': 'passed', 'package': evaluator.manifest, 'results': results,
                               'native_calls': evaluator.sequence - 1,
                               'compiler_revision': sessions.check_compiler(args.compiler_root)['revision']})
            finally:
                evaluator.close()
    except Exception as error:
        report.update({'failed_stage': stage, 'error': str(error)})
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
