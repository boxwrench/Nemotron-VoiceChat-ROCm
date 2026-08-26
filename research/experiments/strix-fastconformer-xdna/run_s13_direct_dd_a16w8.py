#!/usr/bin/env python3
"""Run AMD DynamicDispatch's padded A16W8 MatMul route-control on XDNA.

This is intentionally not a VoiceChat fidelity candidate.  It exercises the
public DynamicDispatch transaction whose raw shape is 49x1024x4096 and whose
kernel shape is 64x1024x4096.  It establishes whether the direct XRT/DD path
is usable on this host independently of generic ONNX-provider partitioning.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import faulthandler
from pathlib import Path


M, K, N = 49, 1024, 4096


def create_model(out: Path):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    activation = helper.make_tensor_value_info("X", TensorProto.UINT16, [1, M, K])
    output = helper.make_tensor_value_info("Y", TensorProto.UINT16, [1, M, N])
    weights = helper.make_tensor("W0", TensorProto.UINT8, [K, N], np.zeros((K, N), dtype=np.uint8))
    # Match the public MatMul direct-DD operand form.  Zero constants plus a
    # zero activation make the expected integer-QDQ output unambiguously zero.
    qdq = helper.make_tensor("qdq", TensorProto.INT64, [N], np.zeros(N, dtype=np.int64))
    qdq_params = helper.make_tensor("qdq_params", TensorProto.INT32, [16], np.zeros(16, dtype=np.int32))
    node = helper.make_node("MatMul", ["X", "W0", "qdq", "qdq_params"], ["Y"], name="s13_a16w8_route_control")
    node.attribute.append(helper.make_attribute("input_shape", [1, M, K]))
    graph = helper.make_graph([node], "s13_a16w8_route_control", [activation], [output], initializer=[weights, qdq, qdq_params])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model_path = out / "s13-direct-dd-a16w8-49x1024x4096.onnx"
    onnx.save(model, model_path)
    return model, model_path


def main() -> int:
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--xclbin", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.xclbin = args.xclbin.resolve()
    if not args.xclbin.is_file():
        raise SystemExit(f"xclbin not found: {args.xclbin}")
    args.out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import onnx
    import ryzenai_dynamic_dispatch as dd
    from ryzenai_dynamic_dispatch import fuse
    from ryzenai_dynamic_dispatch import onnx_graph as ogm

    print("S13 phase: create ONNX route-control model", flush=True)
    model, model_path = create_model(args.out)
    print("S13 phase: build DynamicDispatch metadata", flush=True)
    metadata_tuple = fuse.prepare_metadata(ogm.ONNXGraph(model), tmp_dir=str(args.out), prefix="s13-", use_abs_paths=True)
    metadata_path = args.out / "s13-direct-dd-a16w8-meta.json"
    fuse.save_tensors_to_json(str(metadata_path), *metadata_tuple)
    print("S13 phase: load DynamicDispatch metadata", flush=True)
    metadata = dd.load_meta_json(str(metadata_path))

    # DynamicDispatch takes raw bf16 storage as uint16.  The all-zero control
    # remains zero under every valid bf16 interpretation.
    input_data = np.zeros((1, M, K), dtype=np.uint16)
    output_data = np.full((1, M, N), 0xFFFF, dtype=np.uint16)

    print("S13 phase: compile transaction", flush=True)
    compile_start = time.perf_counter()
    compiler = dd.FusionRuntime()
    compiler.compile(metadata, str(args.xclbin))
    compile_ms = (time.perf_counter() - compile_start) * 1000.0
    # The installed 1.7.1 binding crashes in ``load_state`` after a successful
    # compile.  Keep this one-process control on the supported compile -> init
    # -> execute path; the separate save/load failure is captured as a runtime
    # interface issue, not treated as an NPU result.
    print("S13 phase: initialize XRT runtime in compile process", flush=True)
    runtime = compiler
    runtime.init(metadata)
    timings = []
    for _ in range(args.repeats):
        output_data.fill(0xFFFF)
        print("S13 phase: execute XRT transaction", flush=True)
        started = time.perf_counter()
        runtime.execute([input_data], [output_data])
        timings.append((time.perf_counter() - started) * 1000.0)
    zero_output = bool(np.array_equal(output_data, np.zeros_like(output_data)))
    result = {
        "purpose": "direct DynamicDispatch route/shape control only; not VoiceChat Q8 fidelity",
        "shape": [M, K, N],
        "raw_transaction_shape": [49, 1024, 4096],
        "kernel_shape": [64, 1024, 4096],
        "logical_row_utilization": 1.0 / 64.0,
        "arithmetic": "A16W8acc16, not VoiceChat Q8_0 per-32/F16-scale/F32-accumulation",
        "xclbin": str(args.xclbin),
        "python": platform.python_version(),
        "onnx": onnx.__version__,
        "dynamicdispatch_module": str(Path(dd.__file__).resolve()),
        "model": str(model_path),
        "metadata": str(metadata_path),
        "state_reload": "not used: installed binding segfaulted at load_state after successful compile",
        "compile_ms": compile_ms,
        "samples_ms": timings,
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "output_all_zero": zero_output,
        "classification": "DIRECT_DD_ROUTE_PASS" if zero_output else "NUMERICAL_FAIL",
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if zero_output else 1


if __name__ == "__main__":
    raise SystemExit(main())
