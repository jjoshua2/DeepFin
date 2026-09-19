"""Qualify an explicit saved checkpoint in native inference and batched Bend search.

CPU F32 is executable on CPU hosts. CUDA BF16 requires an actual GPU and explicit
predeclared numerical tolerances; no fallback or unrun CUDA PASS is possible.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import time

import torch

from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig, build_model
from native.bend_engine.session_probe import run_probe as sessions
from .backend import BATCHES, NativeEvaluator, build_worker
from .batch_probe import group
from .checkpoint import export_checkpoint, load_checkpoint
from .qualification import (
    compilation_cache, device_snapshot, protect_inputs, reuse_reference,
    tools, verified_package, workspace, write_report,
)

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
    parser.add_argument('--preflight-only', action='store_true',
                        help='Check CPU weights, toolchain and target readiness without export, forward or search')
    parser.add_argument('--work-dir', type=Path,
                        help='NEW directory retaining packages, builds and caches on success or failure')
    parser.add_argument('--reuse-package', type=Path,
                        help='Trusted immutable v3 package from this EXACT checkpoint/weights/target; skip export')
    args = parser.parse_args()
    if args.fixture_checkpoint and args.reuse_package:
        parser.error('--reuse-package requires --checkpoint, not a newly generated fixture')
    if args.preflight_only and args.work_dir:
        parser.error('--preflight-only does not create a --work-dir')
    # Outside the reporting finally block: a rejected destination must not be written.
    validate_report_destination(args.report, args.checkpoint)
    if args.reuse_package:
        protect_inputs(args.report, [args.reuse_package, args.reuse_package.with_suffix('.json')])
    report: dict[str, object] = {
        'status': 'running', 'device': args.device, 'qualification': 'not_run',
        'scope': 'checkpoint-native-search qualification, not playing strength or throughput',
        'fixture_untrained': args.fixture_checkpoint,
    }
    results: list[dict[str, object]] = []
    report['results'] = results
    timings: dict[str, float] = {}
    report['stage_seconds'] = timings
    torch.set_num_threads(2)
    stage, stage_start = 'preflight', time.monotonic()

    def enter(name: str) -> None:
        nonlocal stage, stage_start
        now = time.monotonic()
        timings[stage] = timings.get(stage, 0.0) + now - stage_start
        stage, stage_start = name, now
        report['stage'] = stage
        write_report(args.report, report)

    with workspace(args.work_dir) as work:
        protect_inputs(args.report, [work / 'checkpoint.pt2', work / 'checkpoint.json', work / 'untrained-transformer.pt'])
        report['artifacts'] = {'directory': str(work), 'retained': args.work_dir is not None}
        try:
            atol, rtol = tolerances(args.device, args.atol, args.rtol)
            report.update(atol=atol, rtol=rtol, batch=args.batch, modes=args.modes)
            report['environment'] = device_snapshot(args.device, args.device_index)
            # Validate the pin and native tools BEFORE paying for checkpoint export.
            report['compiler_revision'] = sessions.check_compiler(args.compiler_root)['revision']
            commands = tools(args.bun, args.cc, args.cxx)
            report['tools'] = commands
            path = args.checkpoint
            if args.fixture_checkpoint:
                path = work / 'untrained-transformer.pt'
                fixture_checkpoint(path)
            enter('checkpoint-load')
            loaded = load_checkpoint(path, weights_key=args.weights_key)
            report.update(checkpoint=loaded.identity, torch_version=str(torch.__version__),
                          input_shape=[args.batch, loaded.encoding.channels, 8, 8])
            package = args.reuse_package.resolve() if args.reuse_package else work / 'checkpoint.pt2'
            if args.reuse_package:
                enter('package-verification')
                report['package'] = verified_package(package, loaded, batch=args.batch,
                    device=args.device, device_index=args.device_index)
            if args.preflight_only:
                report['status'] = 'preflight_passed'
                # No export, model forward, native worker or search was executed.
                enter('complete')
            else:
                with compilation_cache(work):
                    if args.reuse_package:
                        enter('reference-prepare')
                        eager = reuse_reference(loaded, device=args.device, device_index=args.device_index)
                    else:
                        enter('export')
                        eager = export_checkpoint(loaded, package, batch=args.batch,
                            device=args.device, device_index=args.device_index)
                        report['package'] = verified_package(package, loaded, batch=args.batch,
                            device=args.device, device_index=args.device_index)
                    report.update(package_path=str(package), reused_package=args.reuse_package is not None)
                    enter('native-build')
                    worker = build_worker(work / 'worker', commands['cxx'])
                    binaries = sessions.build(args.compiler_root, work / 'bend', commands['bun'], commands['cc'], args.modes)
                    report['qualification'] = 'running'
                    enter('native-start')
                    evaluator = NativeEvaluator(worker, package)
                    try:
                        oracle = sessions.Oracle(binaries['reference'], with_python_chess=True)
                        for mode in args.modes:
                            enter(mode + '-control')
                            control, original = group(binaries[mode], oracle, evaluator, eager, max_rows=1, atol=atol, rtol=rtol)
                            results.append({'mode': mode, **control})
                            for faults in (False, True):
                                enter(mode + ('-recovery' if faults else '-batched'))
                                observed, final = group(binaries[mode], oracle, evaluator, eager, faults=faults, atol=atol, rtol=rtol)
                                # Bank completed groups even if a later comparison fails.
                                results.append({'mode': mode, **observed})
                                for a, b in zip(original, final, strict=True):
                                    if a.structure != b.structure or any(a.summary[k] != b.summary[k] for k in ('best', 'nodes', 'completed', 'stop')):
                                        raise AssertionError('checkpoint batching/recovery changed search; retain the mismatch')
                        evaluator.finish()
                        report.update(status='passed', qualification='passed')
                    finally:
                        report['native_calls'] = evaluator.sequence - 1
                        report['native_stderr_tail'] = evaluator.diagnostics()
                        evaluator.close()
                    enter('complete')
        except Exception as error:
            if report['qualification'] == 'running':
                report['qualification'] = 'failed'
            report.update(status='failed', failed_stage=stage, error=str(error), error_type=type(error).__name__)
            raise
        finally:
            timings[stage] = timings.get(stage, 0.0) + time.monotonic() - stage_start
            write_report(args.report, report)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
