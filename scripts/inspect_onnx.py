#!/usr/bin/env python3
"""Print a foreign ONNX model's I/O schema.

Use to inspect models we want to load (e.g. CeresNets, LC0 nets) before
writing a loader. Reports input/output names, shapes, dtypes, and the
opset version. No GPU; pure metadata read.

    PYTHONPATH=. python3 scripts/inspect_onnx.py path/to/model.onnx
"""
from __future__ import annotations

import argparse
import sys

import onnx
import onnxruntime as ort


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", help="Path to .onnx file")
    p.add_argument("--show-initializers", action="store_true",
                   help="Also list parameter tensor names + shapes (large output)")
    args = p.parse_args()

    model = onnx.load(args.path)
    print(f"=== {args.path} ===")
    print(f"ir_version       : {model.ir_version}")
    print(f"producer         : {model.producer_name} {model.producer_version}")
    print(f"opset            : {[(o.domain or 'ai.onnx', o.version) for o in model.opset_import]}")
    print(f"graph nodes      : {len(model.graph.node)}")
    print(f"initializers     : {len(model.graph.initializer)}")
    print()

    sess = ort.InferenceSession(args.path, providers=["CPUExecutionProvider"])
    print("INPUTS:")
    for i in sess.get_inputs():
        print(f"  {i.name:30s} shape={i.shape}  dtype={i.type}")
    print()
    print("OUTPUTS:")
    for o in sess.get_outputs():
        print(f"  {o.name:30s} shape={o.shape}  dtype={o.type}")

    if args.show_initializers:
        print()
        print("INITIALIZERS (parameter tensors):")
        for init in model.graph.initializer:
            shape = list(init.dims)
            print(f"  {init.name:60s} shape={shape}")

if __name__ == "__main__":
    main()
    sys.exit(0)
