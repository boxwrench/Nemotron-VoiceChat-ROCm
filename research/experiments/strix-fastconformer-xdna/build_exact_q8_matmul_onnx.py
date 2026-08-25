#!/usr/bin/env python3
"""Build and check one exact ggml-Q8_0-style VoiceChat MatMul in ONNX.

The graph deliberately uses explicit block reshape, runtime activation
quantization, integer products and per-block F16-derived scales.  It answers
representability on CPU ONNX only; it makes no VitisAI support claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, checker, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "build" / "llama-voicechat.cpp" / "tools" / "voicechat"
sys.path.insert(0, str(TOOLS))
from vc_gguf import GGUFSource  # noqa: E402

QK = 32


def init(name: str, value, dtype=None):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name)


def raw_q8(source: GGUFSource, name: str):
    tensor = source.take(name)
    if tensor["ty"] != "Q8_0":
        raise ValueError(f"{name}: expected Q8_0")
    rows, width = reversed(tensor["dims"])
    raw = np.frombuffer(source.raw(tensor), dtype=np.uint8).reshape(rows, width // QK, 34)
    scales = raw[..., :2].copy().view(np.float16).reshape(rows, width // QK).astype(np.float32)
    values = raw[..., 2:].view(np.int8).reshape(rows, width // QK, QK).astype(np.int32)
    return rows, width, scales, values


def metric(ref, got):
    delta = np.asarray(got, np.float64).reshape(-1) - np.asarray(ref, np.float64).reshape(-1)
    return {"rmse": float(np.sqrt(np.mean(delta * delta))), "max_abs": float(np.max(np.abs(delta)))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--tensor", default="stt_model.perception.encoder.layers.0.feed_forward1.linear1.weight")
    ap.add_argument("--activation", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    source = GGUFSource(args.source)
    rows, width, w_scale, w_q = raw_q8(source, args.tensor)
    if width % QK:
        raise ValueError("weight width must be Q8-block aligned")
    blocks = width // QK
    nodes = []
    def n(op, inputs, outputs, name, **attrs):
        nodes.append(helper.make_node(op, inputs, outputs if isinstance(outputs, list) else [outputs], name=name, **attrs))
    inits = [
        init("reshape_blocks", [1, blocks, QK], np.int64), init("reshape_scale", [1, 1, blocks], np.int64), init("limit", np.float32(127.0)),
        init("clip_low", np.float32(-128.0)), init("clip_high", np.float32(127.0)),
        init("weights_q", w_q[np.newaxis], np.int32), init("weights_scale", w_scale[np.newaxis], np.float32),
    ]
    n("Reshape", ["activation", "reshape_blocks"], "act_blocks", "reshape")
    n("Abs", ["act_blocks"], "act_abs", "abs")
    n("ReduceMax", ["act_abs"], "act_amax", "amax", axes=[2], keepdims=1)
    n("Div", ["act_amax", "limit"], "act_scale", "scale")
    n("Div", ["act_blocks", "act_scale"], "act_scaled", "normalize")
    n("Round", ["act_scaled"], "act_rounded", "round_nearest_even")
    n("Clip", ["act_rounded", "clip_low", "clip_high"], "act_clipped", "clip")
    n("Cast", ["act_clipped"], "act_q", "to_i32", to=TensorProto.INT32)
    n("Mul", ["weights_q", "act_q"], "products", "integer_products")
    n("ReduceSum", ["products"], "dot_i32", "integer_dot", axes=[3], keepdims=0)
    n("Cast", ["dot_i32"], "dot_f32", "dot_to_f32", to=TensorProto.FLOAT)
    n("Reshape", ["act_scale", "reshape_scale"], "act_scale_row", "reshape_activation_scale")
    n("Mul", ["weights_scale", "act_scale_row"], "block_scale", "block_scale_product")
    n("Mul", ["dot_f32", "block_scale"], "scaled_blocks", "scale_blocks")
    n("ReduceSum", ["scaled_blocks"], "output", "accumulate_blocks", axes=[2], keepdims=0)
    graph = helper.make_graph(nodes, "VOICECHAT_EXACT_Q8_0_MATMUL", [helper.make_tensor_value_info("activation", TensorProto.FLOAT, [1, width])], [
        helper.make_tensor_value_info("act_scale", TensorProto.FLOAT, [1, blocks, 1]),
        helper.make_tensor_value_info("act_q", TensorProto.INT32, [1, blocks, QK]),
        helper.make_tensor_value_info("products", TensorProto.INT32, [1, rows, blocks, QK]),
        helper.make_tensor_value_info("dot_i32", TensorProto.INT32, [1, rows, blocks]),
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, rows]),
    ], initializer=inits)
    model = helper.make_model(graph, producer_name="Nemotron-VoiceChat-ROCm", opset_imports=[helper.make_opsetid("", 12)])
    checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    activation = np.fromfile(args.activation, dtype="<f4").reshape(1, width)
    reference = np.fromfile(args.reference, dtype="<f4").reshape(1, rows)
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    result = session.run(None, {"activation": activation})[-1]
    report = {"nodes": len(nodes), "opset": 12, "providers": session.get_providers(), "q8_contract": "explicit 32-element blocks / runtime A8 / I32 dot / F16-derived scales", "parity": metric(reference, result)}
    if report["parity"]["max_abs"] < 2e-4:
        report["classification"] = "EXACT_Q8_ONNX_PRIMITIVE_PASS"
    elif report["parity"]["max_abs"] < 1e-2:
        report["classification"] = "EXACT_Q8_ONNX_PRIMITIVE_NUMERICAL_ENVELOPE"
    else:
        report["classification"] = "EXACT_Q8_ONNX_PRIMITIVE_FAIL"
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0 if report["classification"] != "EXACT_Q8_ONNX_PRIMITIVE_FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
