from __future__ import annotations

import argparse
from pathlib import Path

import torch


class ProbeModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Exact binary arithmetic for the integer-valued probe inputs.
        return x * 2.0 + 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    model = ProbeModel().eval()
    example = torch.zeros((2, 4), dtype=torch.float32)
    with torch.no_grad():
        exported = torch.export.export(model, (example,))
        torch._inductor.aoti_compile_and_package(
            exported,
            package_path=str(out),
        )

    print(out)


if __name__ == "__main__":
    main()
