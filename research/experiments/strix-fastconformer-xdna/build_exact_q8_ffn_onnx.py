#!/usr/bin/env python3
"""Build and check layer-0's first macaron FFN with deployed Q8 arithmetic.

This is intentionally narrower than a complete Conformer layer.  It preserves
the live mixed-precision sequence:

    LayerNorm/affine -> Q8_0 x Q8_0 linear1 -> SiLU
        -> freshly quantized Q8_0 x Q8_0 linear2 -> half residual.

The two linear operators independently quantize their input activation into
32-element Q8_0 blocks.  Exported ONNX is a CPU representability/parity probe,
not an accelerator-compatibility claim.
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
EPS = 1e-5


def init(name: str, value, dtype=None):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name)


def metric(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, np.float32).reshape(-1)
    actual = np.asarray(actual, np.float32).reshape(-1)
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    denom = float(np.linalg.norm(reference) * np.linalg.norm(actual))
    return {
        "cosine": float(np.dot(reference, actual) / denom) if denom else 0.0,
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
    }


def raw_q8(source: GGUFSource, name: str):
    tensor = source.take(name)
    if tensor["ty"] != "Q8_0":
        raise ValueError(f"{name}: expected Q8_0, got {tensor['ty']}")
    rows, width = reversed(tensor["dims"])
    raw = np.frombuffer(source.raw(tensor), dtype=np.uint8).reshape(rows, width // QK, 34)
    scale = raw[..., :2].copy().view(np.float16).reshape(rows, width // QK).astype(np.float32)
    values = raw[..., 2:].view(np.int8).reshape(rows, width // QK, QK).astype(np.int32)
    return rows, width, scale, values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    source = GGUFSource(args.source)
    p = "stt_model.perception.encoder.layers.0."
    up_rows, hidden, up_scale, up_q = raw_q8(source, p + "feed_forward1.linear1.weight")
    down_rows, ffn, down_scale, down_q = raw_q8(source, p + "feed_forward1.linear2.weight")
    if (hidden, up_rows, down_rows, ffn) != (1024, 4096, 1024, 4096):
        raise ValueError("unexpected layer-0 first-FFN dimensions")

    nodes = []
    inits = []

    def n(op, inputs, outputs, name, **attrs):
        nodes.append(helper.make_node(op, inputs, outputs if isinstance(outputs, list) else [outputs], name=name, **attrs))

    def add_q8_linear(x: str, x_width: int, rows: int, w_scale: np.ndarray, w_q: np.ndarray, prefix: str) -> str:
        blocks = x_width // QK
        inits.extend([
            init(f"{prefix}.reshape_blocks", [1, blocks, QK], np.int64),
            init(f"{prefix}.reshape_scale", [1, 1, blocks], np.int64),
            init(f"{prefix}.limit", np.float32(127.0)),
            init(f"{prefix}.clip_low", np.float32(-128.0)),
            init(f"{prefix}.clip_high", np.float32(127.0)),
            init(f"{prefix}.axis_values", [3], np.int64),
            init(f"{prefix}.axis_blocks", [2], np.int64),
            init(f"{prefix}.weights_q", w_q[np.newaxis], np.int32),
            init(f"{prefix}.weights_scale", w_scale[np.newaxis], np.float32),
        ])
        n("Reshape", [x, f"{prefix}.reshape_blocks"], f"{prefix}.act_blocks", f"{prefix}.reshape")
        n("Abs", [f"{prefix}.act_blocks"], f"{prefix}.act_abs", f"{prefix}.abs")
        n("ReduceMax", [f"{prefix}.act_abs"], f"{prefix}.act_amax", f"{prefix}.amax", axes=[2], keepdims=1)
        n("Div", [f"{prefix}.act_amax", f"{prefix}.limit"], f"{prefix}.act_scale_f32", f"{prefix}.scale_f32")
        n("Div", [f"{prefix}.act_blocks", f"{prefix}.act_scale_f32"], f"{prefix}.act_scaled", f"{prefix}.normalize")
        n("Round", [f"{prefix}.act_scaled"], f"{prefix}.act_rounded", f"{prefix}.round_nearest_even")
        n("Clip", [f"{prefix}.act_rounded", f"{prefix}.clip_low", f"{prefix}.clip_high"], f"{prefix}.act_clipped", f"{prefix}.clip")
        n("Cast", [f"{prefix}.act_clipped"], f"{prefix}.act_q", f"{prefix}.to_i32", to=TensorProto.INT32)
        n("Mul", [f"{prefix}.weights_q", f"{prefix}.act_q"], f"{prefix}.products", f"{prefix}.integer_products")
        n("ReduceSum", [f"{prefix}.products", f"{prefix}.axis_values"], f"{prefix}.dot_i32", f"{prefix}.integer_dot", keepdims=0)
        n("Cast", [f"{prefix}.dot_i32"], f"{prefix}.dot_f32", f"{prefix}.dot_to_f32", to=TensorProto.FLOAT)
        # This F32->F16->F32 path represents ggml's separate stored block
        # scale.  Quantization above intentionally retained the F32 scale.
        n("Cast", [f"{prefix}.act_scale_f32"], f"{prefix}.act_scale_f16", f"{prefix}.store_scale_f16", to=TensorProto.FLOAT16)
        n("Cast", [f"{prefix}.act_scale_f16"], f"{prefix}.act_scale_stored", f"{prefix}.load_scale_f16", to=TensorProto.FLOAT)
        n("Reshape", [f"{prefix}.act_scale_stored", f"{prefix}.reshape_scale"], f"{prefix}.act_scale_row", f"{prefix}.reshape_scale")
        n("Mul", [f"{prefix}.weights_scale", f"{prefix}.act_scale_row"], f"{prefix}.block_scale", f"{prefix}.block_scale_product")
        n("Mul", [f"{prefix}.dot_f32", f"{prefix}.block_scale"], f"{prefix}.scaled_blocks", f"{prefix}.scale_blocks")
        output = f"{prefix}.output"
        n("ReduceSum", [f"{prefix}.scaled_blocks", f"{prefix}.axis_blocks"], output, f"{prefix}.accumulate_blocks", keepdims=0)
        return output

    inits.extend([
        init("norm.scale", source.f32(p + "norm_feed_forward1.weight"), np.float32),
        init("norm.bias", source.f32(p + "norm_feed_forward1.bias"), np.float32),
        init("half", np.float32(0.5)),
    ])
    n("LayerNormalization", ["input", "norm.scale", "norm.bias"], "norm", "layer_norm", axis=1, epsilon=EPS)
    up = add_q8_linear("norm", hidden, up_rows, up_scale, up_q, "linear1")
    n("Sigmoid", [up], "silu.sigmoid", "silu.sigmoid")
    n("Mul", [up, "silu.sigmoid"], "silu", "silu")
    down = add_q8_linear("silu", ffn, down_rows, down_scale, down_q, "linear2")
    n("Mul", [down, "half"], "scaled", "scale_half")
    n("Add", ["input", "scaled"], "residual", "residual")

    outputs = [
        helper.make_tensor_value_info("norm", TensorProto.FLOAT, [1, hidden]),
        helper.make_tensor_value_info(up, TensorProto.FLOAT, [1, up_rows]),
        helper.make_tensor_value_info("silu", TensorProto.FLOAT, [1, ffn]),
        helper.make_tensor_value_info(down, TensorProto.FLOAT, [1, down_rows]),
        helper.make_tensor_value_info("scaled", TensorProto.FLOAT, [1, down_rows]),
        helper.make_tensor_value_info("residual", TensorProto.FLOAT, [1, down_rows]),
    ]
    graph = helper.make_graph(
        nodes, "VOICECHAT_EXACT_Q8_0_LAYER0_FFN1",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, hidden])], outputs,
        initializer=inits,
    )
    model = helper.make_model(graph, producer_name="Nemotron-VoiceChat-ROCm", opset_imports=[helper.make_opsetid("", 17)])
    checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)

    files = {
        "norm": "vc_stream_l0_ffn1_norm_affine.f32",
        up: "vc_stream_l0_ffn1_up.f32",
        "silu": "vc_stream_l0_ffn1_silu.f32",
        down: "vc_stream_l0_ffn1_down.f32",
        "scaled": "vc_stream_l0_ffn1_scaled.f32",
        "residual": "layer-0-ffn1.f32",
    }
    input_value = np.fromfile(args.capture / "pre_enc.f32", dtype="<f4").reshape(1, hidden)
    expected = {name: np.fromfile(args.capture / filename, dtype="<f4").reshape(1, -1) for name, filename in files.items()}
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    actual = dict(zip([item.name for item in session.get_outputs()], session.run(None, {"input": input_value})))
    report = {
        "nodes": len(nodes), "opset": 17, "providers": session.get_providers(),
        "contract": "F32 LayerNorm/SiLU/residual plus independently F32-derived, F16-stored Q8_0 activation scales at each linear",
        "stages": {name: metric(expected[name], actual[name]) for name in files},
    }
    worst = max(stage["max_abs"] for stage in report["stages"].values())
    report["classification"] = "EXACT_Q8_FFN1_PASS" if worst <= 2e-4 else "EXACT_Q8_FFN1_NUMERICAL_ENVELOPE"
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0 if report["classification"] == "EXACT_Q8_FFN1_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
