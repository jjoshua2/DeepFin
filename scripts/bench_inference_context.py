"""Compare no_grad and inference_mode around an inference-only forward."""
from __future__ import annotations

import argparse
import hashlib
import statistics
import time

import torch


class _Block(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(width)
        self.up = torch.nn.Linear(width, width * 4)
        self.down = torch.nn.Linear(width * 4, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.down(torch.nn.functional.gelu(self.up(self.norm(x))))


class _Model(torch.nn.Module):
    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block(width) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def _run_no_grad(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(x)


def _run_inference_mode(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return model(x)


def _checksum(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=9)
    args = parser.parse_args()

    torch.manual_seed(20260715)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = _Model(args.width, args.layers).eval()
    x = torch.randn(args.batch_size, args.width)
    for _ in range(3):
        _run_no_grad(model, x)
        _run_inference_mode(model, x)

    samples: dict[str, list[float]] = {"no_grad": [], "inference_mode": []}
    checksums: set[str] = set()
    for round_idx in range(args.rounds + 2):
        order = (
            ("no_grad", _run_no_grad),
            ("inference_mode", _run_inference_mode),
        ) if round_idx % 2 == 0 else (
            ("inference_mode", _run_inference_mode),
            ("no_grad", _run_no_grad),
        )
        for name, fn in order:
            start = time.perf_counter()
            result = x
            for _ in range(args.iterations):
                result = fn(model, x)
            elapsed = time.perf_counter() - start
            checksums.add(_checksum(result))
            if round_idx >= 2:
                samples[name].append(elapsed)

    if len(checksums) != 1:
        raise AssertionError(f"output mismatch: {checksums}")
    no_grad = statistics.median(samples["no_grad"])
    inference_mode = statistics.median(samples["inference_mode"])
    print(f"no_grad_seconds={no_grad:.9f}")
    print(f"inference_mode_seconds={inference_mode:.9f}")
    print(f"ratio={inference_mode / no_grad:.6f}")
    print(f"checksum={next(iter(checksums))}")


if __name__ == "__main__":
    main()
