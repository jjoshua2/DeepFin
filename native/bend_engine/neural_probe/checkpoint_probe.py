#!/usr/bin/env python3
"""Qualify a specified checkpoint through native AOTI and batched Bend searches.

Run explicitly in an isolated checkout on the requested CPU/GPU. Creates a NEW
output directory; never reads live YAML or overwrites existing artifacts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from native.bend_engine.session_probe import run_probe as sessions
from .adapter import probabilities
from .backend import BATCHES, NativeEvaluator, build_worker
from .batch_probe import group
from .batching import Batch
from .checkpoint import TupleOutputs, export_checkpoint, fingerprint

ROOT = Path(__file__).resolve().parents[3]


def compare_probabilities(actual: tuple[np.ndarray, np.ndarray], expected: tuple[np.ndarray, np.ndarray],
                          actions: np.ndarray) -> tuple[float, float]:
    """Per-row legal policy and WDL TV. Never average away a bad row."""
    aw, ap = probabilities(actual[0], actual[1], actions)
    ew, ep = probabilities(expected[0], expected[1], actions)
    return (float(np.abs(np.asarray(ap) - ep).sum() * 0.5),
            float(np.abs(np.asarray(aw) - ew).sum() * 0.5))


class Comparator:
    """Separate exact ABI/package identity from approximate compilation parity."""
    def __init__(self, package: Path, eager: torch.nn.Module, device: str, dtype: str,
                 *, policy_tv_limit: float, wdl_tv_limit: float):
        for value in (policy_tv_limit, wdl_tv_limit):
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError('TV budgets must be finite, positive and less than one')
        self.target = torch.device(device)
        self.dtype = torch.float32 if dtype == 'float32' else torch.bfloat16
        self.eager = TupleOutputs(eager).eval()
        self.packaged = torch._inductor.aoti_load_package(
            str(package), device_index=self.target.index if self.target.type == 'cuda' else -1)
        self.policy_tv_limit, self.wdl_tv_limit = policy_tv_limit, wdl_tv_limit
        self.rows = 0
        self.logit_error = self.policy_tv = self.wdl_tv = 0.0

    def __call__(self, batch: Batch, policy: np.ndarray, wdl: np.ndarray) -> float:
        with torch.no_grad():
            x = torch.from_numpy(batch.x.copy()).to(device=self.target, dtype=self.dtype)
            pp, pw = self.packaged(x)
            # Same package, device, shape and bits on both sides. No tolerance
            # hiding output order, stale weights, device or serialization errors.
            np.testing.assert_array_equal(policy, pp.detach().float().cpu().numpy())
            np.testing.assert_array_equal(wdl, pw.detach().float().cpu().numpy())
            for i, job in enumerate(batch.jobs):
                ep, ew = self.eager(torch.from_numpy(job.x.copy()).to(device=self.target, dtype=self.dtype))
                expected = (ep.detach().float().cpu().numpy(), ew.detach().float().cpu().numpy())
                actual = (policy[i:i+1], wdl[i:i+1])
                if self.dtype == torch.float32:
                    for a, e in zip(actual, expected, strict=True):
                        np.testing.assert_allclose(a, e, atol=2e-6, rtol=2e-5)
                pol_tv, wdl_tv = compare_probabilities(actual, expected, job.actions)
                if pol_tv > self.policy_tv_limit or wdl_tv > self.wdl_tv_limit:
                    raise AssertionError(f'checkpoint compiled/eager TV exceeded budget: {pol_tv=}, {wdl_tv=}')
                self.rows += 1
                self.policy_tv = max(self.policy_tv, pol_tv)
                self.wdl_tv = max(self.wdl_tv, wdl_tv)
                self.logit_error = max(self.logit_error, *(float(np.abs(a-e).max())
                                       for a, e in zip(actual, expected, strict=True)))
        return self.logit_error


def qualify(checkpoint: Path, output: Path, compiler_root: Path, *, device: str = 'cpu',
            dtype: str = 'float32', batch: int = 4, modes: tuple[str, ...] = ('native',),
            bun: str = 'bun', cc: str = 'clang', cxx: str = 'clang++',
            policy_tv_limit: float = 0.01, wdl_tv_limit: float = 0.01) -> dict[str, object]:
    if output.exists():
        raise FileExistsError('output directory already exists; refusing to overwrite')
    for limit in (policy_tv_limit, wdl_tv_limit):
        if not math.isfinite(limit) or not 0 < limit < 1:
            raise ValueError('TV budgets must be finite, positive and less than one')
    if not modes or any(mode not in sessions.MODES for mode in modes):
        raise ValueError('invalid Bend build modes')
    if batch > 8:
        # The general package schema supports 16, but this five-search scheduler
        # deliberately reserves only eight rows. Do not silently change its bound.
        raise ValueError('this bounded search qualification supports batches up to eight')
    pin = sessions.check_compiler(compiler_root)
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='bend-checkpoint-') as tmp:
        work = Path(tmp)
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(work / 'inductor')
        package, loaded = export_checkpoint(checkpoint, output, device=device, dtype=dtype, batch=batch)
        comparator = Comparator(package, loaded.model, device, dtype,
                                policy_tv_limit=policy_tv_limit, wdl_tv_limit=wdl_tv_limit)
        binaries = sessions.build(compiler_root, work / 'bend', bun, cc, list(modes))
        worker = build_worker(work / 'native', cxx)
        evaluator = NativeEvaluator(worker, package, timeout=60)
        results = []
        try:
            oracle = sessions.Oracle(binaries['reference'], with_python_chess=True)
            for mode in modes:
                control_report, control = group(binaries[mode], oracle, evaluator, loaded.model,
                                                max_rows=1, compare_batch=comparator)
                for faults in (False, True):
                    report, candidate = group(binaries[mode], oracle, evaluator, loaded.model,
                                              max_rows=batch, faults=faults, compare_batch=comparator)
                    for a, b in zip(control, candidate, strict=True):
                        if a.structure != b.structure or any(a.summary[k] != b.summary[k]
                                for k in ('best', 'nodes', 'completed', 'stop')):
                            raise AssertionError('checkpoint batching/cancellation changed control search')
                    results.append({'mode': mode, **report})
                results.append({'mode': mode, **control_report})
            evaluator.finish()
            final_report: dict[str, object] = {'scope': 'checkpoint-specific native inference and diagnostic Bend PUCT; not strength/throughput',
                      'package': evaluator.manifest, 'compiler': pin, 'results': results,
                      'native_calls': evaluator.sequence - 1, 'real_rows_compared': comparator.rows,
                      'native_vs_python_package': 'bit-exact for every batch including padding',
                      'max_eager_logit_absolute_error': comparator.logit_error,
                      'max_legal_policy_tv': comparator.policy_tv, 'max_wdl_tv': comparator.wdl_tv,
                      'policy_tv_limit': policy_tv_limit, 'wdl_tv_limit': wdl_tv_limit,
                      'trained_status': 'not inferred; weights supplied by checkpoint',
                      'cuda_executed': torch.device(device).type == 'cuda'}
        finally:
            evaluator.close()
        if fingerprint(checkpoint) != loaded.sha256:
            raise ValueError('source checkpoint changed before qualification completed')
        with (output / 'qualification.json').open('x') as stream:
            json.dump(final_report, stream, indent=2, allow_nan=False)
            stream.write('\n')
        return final_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--device', default='cpu', help='cpu or explicit cuda:N; never silently falls back')
    parser.add_argument('--dtype', choices=('float32', 'bfloat16'), default='float32')
    parser.add_argument('--batch', type=int, choices=[b for b in BATCHES if b <= 8], default=4)
    parser.add_argument('--modes', nargs='+', choices=list(sessions.MODES), default=['native'])
    parser.add_argument('--compiler-root', type=Path, default=ROOT / 'build/bend_u64_toolchain/source')
    parser.add_argument('--bun', default=shutil.which('bun'))
    parser.add_argument('--cc', default=shutil.which('clang'))
    parser.add_argument('--cxx', default=shutil.which('clang++'))
    parser.add_argument('--policy-tv-limit', type=float, default=0.01)
    parser.add_argument('--wdl-tv-limit', type=float, default=0.01)
    args = parser.parse_args()
    if not all((args.bun, args.cc, args.cxx)):
        parser.error('Bun, Clang and Clang++ are required')
    report = qualify(args.checkpoint, args.out_dir, args.compiler_root, device=args.device,
                     dtype=args.dtype, batch=args.batch, modes=tuple(args.modes), bun=args.bun,
                     cc=args.cc, cxx=args.cxx, policy_tv_limit=args.policy_tv_limit,
                     wdl_tv_limit=args.wdl_tv_limit)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
