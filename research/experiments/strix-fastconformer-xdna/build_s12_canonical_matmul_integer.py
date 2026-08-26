#!/usr/bin/env python3
"""Build the S12-A standard A8W8 MatMulInteger assignment control.

This is deliberately not a VoiceChat fidelity graph. It asks only whether the
public Ryzen AI VitisAI provider assigns a plain ONNX quantized MatMul at the
real layer-0 linear1 shape to XDNA.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--activation", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=12012)
    args = ap.parse_args()

    width, rows = 1024, 4096
    rng = np.random.default_rng(args.seed)
    activation = rng.integers(-127, 128, size=(1, width), dtype=np.int8)
    weights = rng.integers(-127, 128, size=(width, rows), dtype=np.int8)
    graph = helper.make_graph(
        [helper.make_node("MatMulInteger", ["activation", "weights"], ["output"], name="canonical_a8w8_matmul")],
        "S12_CANONICAL_A8W8_MATMULINTEGER",
        [helper.make_tensor_value_info("activation", TensorProto.INT8, [1, width])],
        [helper.make_tensor_value_info("output", TensorProto.INT32, [1, rows])],
        initializer=[numpy_helper.from_array(weights, "weights")],
    )
    model = helper.make_model(
        graph,
        producer_name="Nemotron-VoiceChat-ROCm",
        opset_imports=[helper.make_opsetid("", 12)],
    )
    # Ryzen AI 1.7.1 ORT supports ONNX IR <= 11.
    model.ir_version = 11
    checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(model.SerializeToString())
    activation.tofile(args.activation)
    print({"model": str(args.output), "activation": str(args.activation), "input": [1, width], "output": [1, rows], "op": "MatMulInteger"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
