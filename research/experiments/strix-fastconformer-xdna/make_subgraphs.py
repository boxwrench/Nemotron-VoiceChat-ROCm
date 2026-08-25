#!/usr/bin/env python3
"""Create provisional, VoiceChat-shaped ONNX compiler-spike subgraphs.

This intentionally does not load VoiceChat weights.  It creates static graph
shapes and operator semantics for the highest-risk structures so a host with
ONNX Runtime/VitisAI can test representation and partitioning without model
integration.  Generated .onnx files belong in an ignored output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import numpy as np
    import onnx
    from onnx import TensorProto, checker, helper, numpy_helper
except ImportError as exc:  # pragma: no cover - host dependency gate
    raise SystemExit(
        "BLOCKED: make_subgraphs.py requires the host 'onnx' package; "
        f"import failed: {exc}"
    ) from exc


OPSET = 17
T = 160  # provisional mel frames only; not the M4A-2 production contract
N = 21   # floor(floor(floor(T/2)+1)/2+1)/2+1 for VoiceChat causal subsampling
D = 1024
H = 8
DH = 128


def value_info(name: str, shape: list[int], dtype: int = TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


def initializer(name: str, array: np.ndarray):
    return numpy_helper.from_array(np.asarray(array), name=name)


def model(name: str, inputs, outputs, nodes, initializers):
    graph = helper.make_graph(
        nodes,
        name,
        inputs,
        outputs,
        initializer=initializers,
    )
    m = helper.make_model(
        graph,
        producer_name="Nemotron-VoiceChat-ROCm",
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    checker.check_model(m)
    return m


def make_fc1(fused: bool):
    # After pre_conv_0, VoiceChat has [channel=256, freq=65, time=81].
    x = value_info("x", [1, 256, 65, 81])
    y = value_info("y", [1, 256, 33, 41])
    w = initializer("depthwise_weight", np.zeros((256, 1, 3, 3), dtype=np.float32))
    nodes = []
    if fused:
        nodes.append(
            helper.make_node(
                "Conv",
                ["x", "depthwise_weight"],
                ["y"],
                name="pre_conv_2_depthwise_fused_pad",
                strides=[2, 2],
                pads=[2, 2, 1, 1],
                group=256,
            )
        )
    else:
        pads = initializer("causal_pads", np.array([0, 0, 2, 2, 0, 0, 1, 1], dtype=np.int64))
        zero = initializer("pad_value", np.array(0.0, dtype=np.float32))
        nodes.extend(
            [
                helper.make_node("Pad", ["x", "causal_pads", "pad_value"], ["padded"], name="pre_conv_2_depthwise_pad"),
                helper.make_node(
                    "Conv",
                    ["padded", "depthwise_weight"],
                    ["y"],
                    name="pre_conv_2_depthwise",
                    strides=[2, 2],
                    pads=[0, 0, 0, 0],
                    group=256,
                ),
            ]
        )
        w = [w, pads, zero]
    return model(
        "FC_SUBGRAPH_1_FUSED" if fused else "FC_SUBGRAPH_1_PAD_DEPTHWISE",
        [x],
        [y],
        nodes,
        w if isinstance(w, list) else [w],
    )


def make_fc2():
    x = value_info("x", [1, N, D])
    y = value_info("y", [1, N, D])
    scale = initializer("ln_scale", np.ones((D,), dtype=np.float32))
    bias = initializer("ln_bias", np.zeros((D,), dtype=np.float32))
    pw1 = initializer("pointwise_up", np.zeros((D, 2 * D), dtype=np.float32))
    dw = initializer("depthwise_weight", np.zeros((D, 1, 9), dtype=np.float32))
    pw2 = initializer("pointwise_down", np.zeros((D, D), dtype=np.float32))
    pads = initializer("causal_conv_pads", np.array([0, 0, 8, 0, 0, 0], dtype=np.int64))
    zero = initializer("pad_value", np.array(0.0, dtype=np.float32))
    split = initializer("split_sizes", np.array([D, D], dtype=np.int64))
    nodes = [
        helper.make_node("LayerNormalization", ["x", "ln_scale", "ln_bias"], ["norm_in"], name="conv_norm_in", axis=2, epsilon=1e-5),
        helper.make_node("MatMul", ["norm_in", "pointwise_up"], ["pw1"], name="conv_pw1"),
        helper.make_node("Split", ["pw1", "split_sizes"], ["signal", "gate"], name="glu_split", axis=2),
        helper.make_node("Sigmoid", ["gate"], ["gate_sigmoid"], name="glu_sigmoid"),
        helper.make_node("Mul", ["signal", "gate_sigmoid"], ["glu"], name="glu_mul"),
        helper.make_node("Transpose", ["glu"], ["channels_first"], name="to_channels_first", perm=[0, 2, 1]),
        helper.make_node("Pad", ["channels_first", "causal_conv_pads", "pad_value"], ["padded"], name="causal_depthwise_pad"),
        # Input is N,C,L, so a 1-D Conv has two pad values, not three.
        helper.make_node("Conv", ["padded", "depthwise_weight"], ["dw_out"], name="enc_conv_dw", group=D, pads=[0, 0], strides=[1]),
        helper.make_node("Transpose", ["dw_out"], ["channels_last"], name="to_channels_last", perm=[0, 2, 1]),
        helper.make_node("LayerNormalization", ["channels_last", "ln_scale", "ln_bias"], ["conv_norm_out"], name="conv_norm_out", axis=2, epsilon=1e-5),
        helper.make_node("Sigmoid", ["conv_norm_out"], ["silu_gate"], name="silu_sigmoid"),
        helper.make_node("Mul", ["conv_norm_out", "silu_gate"], ["silu"], name="silu"),
        helper.make_node("MatMul", ["silu", "pointwise_down"], ["y"], name="conv_pw2"),
    ]
    return model("FC_SUBGRAPH_2_CONV_MODULE", [x], [y], nodes, [scale, bias, pw1, dw, pw2, pads, zero, split])


def make_fc3():
    x = value_info("x", [1, N, D])
    mask = value_info("attn_mask", [1, 1, N, N])
    rel_bias = value_info("relative_bias", [1, H, N, N])
    y = value_info("y", [1, N, D])
    q_w = initializer("q_weight", np.zeros((D, D), dtype=np.float32))
    k_w = initializer("k_weight", np.zeros((D, D), dtype=np.float32))
    v_w = initializer("v_weight", np.zeros((D, D), dtype=np.float32))
    shape = initializer("head_shape", np.array([1, N, H, DH], dtype=np.int64))
    merged_shape = initializer("merged_shape", np.array([1, N, D], dtype=np.int64))
    nodes = [
        helper.make_node("MatMul", ["x", "q_weight"], ["q_flat"], name="q_projection"),
        helper.make_node("MatMul", ["x", "k_weight"], ["k_flat"], name="k_projection"),
        helper.make_node("MatMul", ["x", "v_weight"], ["v_flat"], name="v_projection"),
        helper.make_node("Reshape", ["q_flat", "head_shape"], ["q_heads_last"], name="q_reshape"),
        helper.make_node("Reshape", ["k_flat", "head_shape"], ["k_heads_last"], name="k_reshape"),
        helper.make_node("Reshape", ["v_flat", "head_shape"], ["v_heads_last"], name="v_reshape"),
        helper.make_node("Transpose", ["q_heads_last"], ["q"], name="q_heads", perm=[0, 2, 1, 3]),
        helper.make_node("Transpose", ["k_heads_last"], ["k"], name="k_heads", perm=[0, 2, 3, 1]),
        helper.make_node("Transpose", ["v_heads_last"], ["v"], name="v_heads", perm=[0, 2, 1, 3]),
        helper.make_node("MatMul", ["q", "k"], ["scores"], name="content_scores"),
        helper.make_node("Add", ["scores", "relative_bias"], ["relative_scores"], name="relative_position_add"),
        helper.make_node("Add", ["relative_scores", "attn_mask"], ["masked_scores"], name="f32_causal_mask_add"),
        helper.make_node("Softmax", ["masked_scores"], ["probs"], name="attention_softmax", axis=-1),
        helper.make_node("MatMul", ["probs", "v"], ["context"], name="value_aggregation"),
        helper.make_node("Transpose", ["context"], ["context_last"], name="context_merge", perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", ["context_last", "merged_shape"], ["y"], name="output_reshape"),
    ]
    return model("FC_SUBGRAPH_3_RELATIVE_ATTENTION", [x, mask, rel_bias], [y], nodes, [q_w, k_w, v_w, shape, merged_shape])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {
        "fc-subgraph-1-pad-depthwise.onnx": make_fc1(False),
        "fc-subgraph-1-fused-depthwise.onnx": make_fc1(True),
        "fc-subgraph-2-conv-module.onnx": make_fc2(),
        "fc-subgraph-3-relative-attention.onnx": make_fc3(),
    }
    for name, graph in outputs.items():
        path = args.out / name
        onnx.save(graph, path)
    print(json.dumps({"provisional_mel_frames": T, "encoded_frames": N, "outputs": [str(args.out / p) for p in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
