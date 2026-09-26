"""Opt-in UCI interface for the bounded Bend prototype, NOT production DeepFin."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from pathlib import Path
import sys

from .backend import DiagnosticEvaluator, Evaluator, NativeSearch
from .protocol import Engine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bend-binary', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--diagnostic', action='store_true', help='Explicit non-neural material/test policy, not a trained model')
    source.add_argument('--checkpoint', type=Path)
    parser.add_argument('--reuse-package', type=Path)
    parser.add_argument('--worker-binary', type=Path)
    parser.add_argument('--weights-key', choices=['model', 'swa_model'], default='model')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--batch', type=int, choices=[1, 2, 4, 8, 16], default=4)
    args = parser.parse_args()
    if not args.bend_binary.is_file():
        parser.error('--bend-binary must name the existing compiled session executable')
    evaluator: Evaluator
    if args.diagnostic:
        if args.reuse_package or args.worker_binary or args.device != 'cpu' or args.device_index:
            parser.error('diagnostic mode cannot use neural artifacts or a CUDA target')
        evaluator = DiagnosticEvaluator()
    else:
        if not args.reuse_package or not args.worker_binary:
            parser.error('checkpoint mode requires --reuse-package and --worker-binary')
        # Imports/checkpoint/runtime messages never contaminate the UCI stdout stream.
        with redirect_stdout(sys.stderr):
            import torch
            from .evaluator import PackageEvaluator
            torch.set_num_threads(2)
            evaluator = PackageEvaluator(args.checkpoint, args.reuse_package, args.worker_binary,
                                         weights=args.weights_key, device=args.device,
                                         device_index=args.device_index, batch=args.batch)
    backend = NativeSearch(args.bend_binary, evaluator)
    label = 'diagnostic non-neural evaluator' if args.diagnostic else 'native checkpoint evaluator; experimental bounded PUCT'
    def emit(line: str) -> None:
        print(line, flush=True)
        if line == 'uciok':
            print('info string ' + label, flush=True)
    try:
        Engine(backend, emit).run(sys.stdin)
    except (BrokenPipeError, KeyboardInterrupt):
        backend.interrupt()
        backend.close()


if __name__ == '__main__':
    main()
