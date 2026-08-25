#!/usr/bin/env python3
"""Export real VoiceChat perception components from GGUF to ONNX.

This is deliberately a direct graph builder.  It reads the source GGUF with
the runtime's ``GGUFSource`` helper, dequantizes only the tensors needed by the
selected component, and constructs ONNX nodes explicitly.  It does not use
NeMo, PyTorch, torch.onnx, or a transcript-producing substitute.

The static dimensions are compiler-spike dimensions only.  M4A-2 still owns
the eventual production invocation shape.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper, shape_inference


ROOT = Path(__file__).resolve().parents[3]
VOICECHAT_TOOLS = ROOT / "build" / "llama-voicechat.cpp" / "tools" / "voicechat"
sys.path.insert(0, str(VOICECHAT_TOOLS))
from vc_gguf import GGUFSource  # noqa: E402


SRC = "stt_model.perception."
OPSET = 17
D = 1024
HEADS = 8
D_HEAD = 128
FFN = 4096
MEL = 128
KERNEL = 9
EPS = 1e-5


def vi(name: str, shape: list[int], dtype: int = TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


def init(name: str, array: np.ndarray | list | float, dtype=None):
    if dtype is not None:
        array = np.asarray(array, dtype=dtype)
    return numpy_helper.from_array(np.ascontiguousarray(array), name=name)


def add_node(nodes, op: str, inputs: list[str], output: str, name: str, **attrs):
    outputs = output if isinstance(output, list) else [output]
    nodes.append(helper.make_node(op, inputs, outputs, name=name, **attrs))
    return output


def add_linear(nodes, initializers, source: GGUFSource, source_name: str,
               x: str, output: str, name: str):
    # GGUFSource exposes checkpoint weights as [out, in].  ONNX MatMul uses
    # [batch..., in] @ [in, out], so transpose the dequantized source weight.
    weight = np.asarray(source.f32(source_name), dtype=np.float32).T
    wname = f"{name}.weight"
    initializers.append(init(wname, weight))
    return add_node(nodes, "MatMul", [x, wname], output, name)


def add_affine_linear(nodes, initializers, source: GGUFSource, weight_name: str,
                      bias_name: str, x: str, output: str, name: str):
    add_linear(nodes, initializers, source, weight_name, x, f"{name}.matmul", name)
    bname = f"{name}.bias"
    initializers.append(init(bname, source.f32(bias_name)))
    add_node(nodes, "Add", [f"{name}.matmul", bname], output, f"{name}.bias_add")
    return output


def add_layer_norm(nodes, initializers, source: GGUFSource, prefix: str,
                   x: str, output: str, name: str):
    scale = f"{name}.scale"
    bias = f"{name}.bias"
    initializers.extend([
        init(scale, source.f32(prefix + ".weight")),
        init(bias, source.f32(prefix + ".bias")),
    ])
    return add_node(
        nodes, "LayerNormalization", [x, scale, bias], output, name,
        axis=2, epsilon=EPS,
    )


def add_silu(nodes, x: str, output: str, name: str):
    sig = f"{name}.sigmoid"
    add_node(nodes, "Sigmoid", [x], sig, sig)
    return add_node(nodes, "Mul", [x, sig], output, name)


def causal_pad_2d(nodes, initializers, x: str, output: str, name: str):
    # NCHW: left/right in the final two axes are [2,1] for both frequency and
    # time, matching C++ ggml_pad_ext(cur, 2, 1, 2, 1, ...).
    pads = f"{name}.pads"
    initializers.append(init(pads, [0, 0, 2, 2, 0, 0, 1, 1], np.int64))
    value = f"{name}.value"
    initializers.append(init(value, np.array(0.0, dtype=np.float32)))
    return add_node(nodes, "Pad", [x, pads, value], output, name)


def causal_pad_1d(nodes, initializers, x: str, output: str, name: str):
    # N,C,L: eight zeros before the sequence and none after it.
    pads = f"{name}.pads"
    initializers.append(init(pads, [0, 0, KERNEL - 1, 0, 0, 0], np.int64))
    value = f"{name}.value"
    initializers.append(init(value, np.array(0.0, dtype=np.float32)))
    return add_node(nodes, "Pad", [x, pads, value], output, name)


def model_for(component: str, graph, metadata: dict[str, str]):
    model = helper.make_model(
        graph,
        producer_name="Nemotron-VoiceChat-ROCm",
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    for key, value in metadata.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value
    checker.check_model(model)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as exc:  # shape inference is useful but not a graph gate
        print(f"warning: ONNX shape inference skipped: {exc}", file=sys.stderr)
    checker.check_model(model)
    return model


def export_pre_encode(source: GGUFSource, frames: int, output: Path):
    if frames < 8:
        raise SystemExit("--frames must be at least 8 for the three stride-2 stages")

    n_time = frames
    for _ in range(3):
        n_time = n_time // 2 + 1
    freq = MEL
    for _ in range(3):
        freq = freq // 2 + 1

    nodes = []
    initializers = []
    inputs = [vi("mel", [1, 1, MEL, frames])]
    x = "mel"
    for index, group, relu in ((0, 1, True), (2, 256, False), (3, 1, True),
                               (5, 256, False), (6, 1, True)):
        if index in (0, 2, 5):
            padded = f"pre_encode.conv{index}.padded"
            causal_pad_2d(nodes, initializers, x, padded, f"pre_encode.conv{index}.causal_pad")
            x = padded
        stem = SRC + f"encoder.pre_encode.conv.{index}."
        weight = np.asarray(source.f32(stem + "weight"), dtype=np.float32)
        bias = np.asarray(source.f32(stem + "bias"), dtype=np.float32).reshape(1, -1, 1, 1)
        wname = f"pre_encode.conv{index}.weight"
        bname = f"pre_encode.conv{index}.bias"
        initializers.extend([init(wname, weight), init(bname, bias)])
        out = f"pre_encode.conv{index}.conv"
        add_node(
            nodes, "Conv", [x, wname], out, f"pre_encode.conv{index}",
            strides=[2, 2] if index in (0, 2, 5) else [1, 1],
            pads=[0, 0, 0, 0], group=group,
        )
        x = f"pre_encode.conv{index}.biased"
        add_node(nodes, "Add", [out, bname], x, f"pre_encode.conv{index}.bias_add")
        if relu:
            x = add_node(nodes, "Relu", [x], f"pre_encode.conv{index}.relu", f"pre_encode.conv{index}.relu")

    # Conv output is N,C,freq,time.  C++ permutes its ggml [freq,time,C]
    # tensor to [freq,C,time], then flattens with frequency fastest.  N,T,C,F
    # followed by reshape reproduces that exact C-major/frequency-inner order.
    transposed = "pre_encode.time_channel_freq"
    add_node(nodes, "Transpose", [x], transposed, "pre_encode.to_time_channel_freq", perm=[0, 3, 1, 2])
    shape_name = "pre_encode.flatten_shape"
    initializers.append(init(shape_name, [1, n_time, 256 * freq], np.int64))
    flattened = "pre_encode.flattened"
    add_node(nodes, "Reshape", [transposed, shape_name], flattened, "pre_encode.flatten")

    out_weight = np.asarray(source.f32(SRC + "encoder.pre_encode.out.weight"), dtype=np.float32).T
    out_bias = np.asarray(source.f32(SRC + "encoder.pre_encode.out.bias"), dtype=np.float32)
    initializers.extend([init("pre_encode.out.weight", out_weight), init("pre_encode.out.bias", out_bias)])
    projected = "pre_encode.out.matmul"
    add_node(nodes, "MatMul", [flattened, "pre_encode.out.weight"], projected, "pre_encode.out")
    output_name = "pre_encode_out"
    add_node(nodes, "Add", [projected, "pre_encode.out.bias"], output_name, "pre_encode.out.bias_add")

    graph = helper.make_graph(
        nodes,
        "VOICECHAT_PRE_ENCODE_S1",
        inputs,
        [vi(output_name, [1, n_time, D])],
        initializer=initializers,
    )
    return model_for("pre-encode", graph, {
        "voicechat.component": "pre-encode",
        "voicechat.source": "stt_model.perception.encoder.pre_encode.*",
        "voicechat.static_frames": str(frames),
        "voicechat.encoded_frames": str(n_time),
        "voicechat.production_shape_status": "provisional; M4A-2 pending",
    })


def export_layer(source: GGUFSource, layer_index: int, frames: int, output: Path):
    if frames < 1:
        raise SystemExit("--frames must be positive for --component layer")
    p = SRC + f"encoder.layers.{layer_index}."
    b = f"layer{layer_index}"
    window = 2 * frames - 1
    nodes = []
    initializers = []
    inputs = [
        vi("encoder_in", [1, frames, D]),
        vi("attn_mask", [1, 1, frames, frames]),
        # VoiceChat builds positional frequencies at n_state/2 (512), then
        # concatenates sin/cos to the full 1024-wide positional embedding.
        vi("pos_freqs", [D // 2, 1]),
        vi("rel_positions", [1, window]),
    ]

    # Macaron FFN 1.
    x = add_layer_norm(nodes, initializers, source, p + "norm_feed_forward1", "encoder_in", f"{b}.ffn1.norm", f"{b}.ffn1.norm")
    add_linear(nodes, initializers, source, p + "feed_forward1.linear1.weight", x, f"{b}.ffn1.up", f"{b}.ffn1.up")
    add_silu(nodes, f"{b}.ffn1.up", f"{b}.ffn1.silu", f"{b}.ffn1.silu")
    add_linear(nodes, initializers, source, p + "feed_forward1.linear2.weight", f"{b}.ffn1.silu", f"{b}.ffn1.down", f"{b}.ffn1.down")
    half = f"{b}.ffn1.half"
    initializers.append(init(half + ".scale", np.array(0.5, dtype=np.float32)))
    add_node(nodes, "Mul", [f"{b}.ffn1.down", half + ".scale"], half, f"{b}.ffn1.scale")
    cur = f"{b}.ffn1.residual"
    add_node(nodes, "Add", ["encoder_in", half], cur, f"{b}.ffn1.residual")

    # Relative-position self-attention.  The rel-shift is explicit as a fixed
    # GatherElements table for this provisional frame count.
    attn_in = add_layer_norm(nodes, initializers, source, p + "norm_self_att", cur, f"{b}.attn.norm", f"{b}.attn.norm")
    qflat = add_linear(nodes, initializers, source, p + "self_attn.linear_q.weight", attn_in, f"{b}.attn.qflat", f"{b}.attn.q")
    kflat = add_linear(nodes, initializers, source, p + "self_attn.linear_k.weight", attn_in, f"{b}.attn.kflat", f"{b}.attn.k")
    vflat = add_linear(nodes, initializers, source, p + "self_attn.linear_v.weight", attn_in, f"{b}.attn.vflat", f"{b}.attn.v")
    head_shape = b + ".attn.head_shape"
    initializers.append(init(head_shape, [1, frames, HEADS, D_HEAD], np.int64))
    for flat, shaped in ((qflat, f"{b}.attn.qshape"), (kflat, f"{b}.attn.kshape"), (vflat, f"{b}.attn.vshape")):
        add_node(nodes, "Reshape", [flat, head_shape], shaped, shaped)
    q = add_node(nodes, "Transpose", [f"{b}.attn.qshape"], f"{b}.attn.q", f"{b}.attn.q_heads", perm=[0, 2, 1, 3])
    k = add_node(nodes, "Transpose", [f"{b}.attn.kshape"], f"{b}.attn.k", f"{b}.attn.k_heads", perm=[0, 2, 3, 1])
    v = add_node(nodes, "Transpose", [f"{b}.attn.vshape"], f"{b}.attn.v", f"{b}.attn.v_heads", perm=[0, 2, 1, 3])

    pos_freqs = "pos.freqs"
    rel_positions = "pos.relative_positions"
    add_node(nodes, "Mul", ["pos_freqs", "rel_positions"], "pos.theta", "pos.theta")
    add_node(nodes, "Sin", ["pos.theta"], "pos.sin", "pos.sin")
    add_node(nodes, "Cos", ["pos.theta"], "pos.cos", "pos.cos")
    add_node(nodes, "Concat", ["pos.sin", "pos.cos"], "pos.dc", "pos.concat", axis=0)
    initializers.append(init("pos.batch_axis", [0], np.int64))
    add_node(nodes, "Unsqueeze", ["pos.dc", "pos.batch_axis"], "pos.dc_batch", "pos.batch")
    add_node(nodes, "Transpose", ["pos.dc_batch"], "pos.window_width", "pos.to_window_width", perm=[0, 2, 1])
    add_linear(nodes, initializers, source, p + "self_attn.linear_pos.weight", "pos.window_width", "pos.projected", f"{b}.attn.linear_pos")
    pos_shape = b + ".attn.pos_shape"
    initializers.append(init(pos_shape, [1, window, HEADS, D_HEAD], np.int64))
    add_node(nodes, "Reshape", ["pos.projected", pos_shape], f"{b}.attn.pos_reshaped", f"{b}.attn.pos_reshape")
    pos_heads = add_node(nodes, "Transpose", [f"{b}.attn.pos_reshaped"], f"{b}.attn.pos_heads", f"{b}.attn.pos_heads", perm=[0, 2, 1, 3])
    pos_heads_t = add_node(nodes, "Transpose", [pos_heads], f"{b}.attn.pos_heads_t", f"{b}.attn.pos_heads_t", perm=[0, 1, 3, 2])

    bias_shape = b + ".attn.bias_shape"
    initializers.append(init(bias_shape, [1, HEADS, 1, D_HEAD], np.int64))
    for suffix in ("u", "v"):
        raw = source.f32(p + f"self_attn.pos_bias_{suffix}")
        bias_name = f"{b}.attn.pos_bias_{suffix}"
        initializers.append(init(bias_name, raw.reshape(1, HEADS, 1, D_HEAD)))
        add_node(nodes, "Add", [q, bias_name], f"{b}.attn.q_{suffix}", f"{b}.attn.q_{suffix}")

    add_node(nodes, "MatMul", [f"{b}.attn.q_u", k], f"{b}.attn.content_scores", f"{b}.attn.content_scores")
    add_node(nodes, "MatMul", [f"{b}.attn.q_v", pos_heads_t], f"{b}.attn.relative_raw", f"{b}.attn.relative_raw")
    # For query i/key j the Transformer-XL relative index is centered at
    # window-1 and shifted by key-query.  GatherElements preserves this exact
    # static rel-shift semantics without introducing a framework op.
    rel_idx = np.empty((1, 1, frames, frames), dtype=np.int64)
    for i in range(frames):
        for j in range(frames):
            rel_idx[0, 0, i, j] = (frames - 1) + j - i
    initializers.append(init(f"{b}.attn.rel_indices", rel_idx))
    initializers.append(init(f"{b}.attn.rel_expand_shape", [1, HEADS, frames, frames], np.int64))
    add_node(nodes, "Expand", [f"{b}.attn.rel_indices", f"{b}.attn.rel_expand_shape"], f"{b}.attn.rel_indices_heads", f"{b}.attn.rel_indices_expand")
    add_node(nodes, "GatherElements", [f"{b}.attn.relative_raw", f"{b}.attn.rel_indices_heads"], f"{b}.attn.relative_scores", f"{b}.attn.rel_shift", axis=3)
    add_node(nodes, "Add", [f"{b}.attn.content_scores", f"{b}.attn.relative_scores"], f"{b}.attn.scores", f"{b}.attn.score_add")
    initializers.append(init(f"{b}.attn.inv_sqrt_dh", np.array(1.0 / np.sqrt(D_HEAD), dtype=np.float32)))
    add_node(nodes, "Mul", [f"{b}.attn.scores", f"{b}.attn.inv_sqrt_dh"], f"{b}.attn.scaled", f"{b}.attn.scale")
    add_node(nodes, "Add", [f"{b}.attn.scaled", "attn_mask"], f"{b}.attn.masked", f"{b}.attn.mask_add")
    add_node(nodes, "Softmax", [f"{b}.attn.masked"], f"{b}.attn.probs", f"{b}.attn.softmax", axis=-1)
    add_node(nodes, "MatMul", [f"{b}.attn.probs", v], f"{b}.attn.context_heads", f"{b}.attn.value_aggregation")
    add_node(nodes, "Transpose", [f"{b}.attn.context_heads"], f"{b}.attn.context_last", f"{b}.attn.context_merge", perm=[0, 2, 1, 3])
    context_shape = b + ".attn.context_shape"
    initializers.append(init(context_shape, [1, frames, D], np.int64))
    add_node(nodes, "Reshape", [f"{b}.attn.context_last", context_shape], f"{b}.attn.context", f"{b}.attn.context_reshape")
    add_linear(nodes, initializers, source, p + "self_attn.linear_out.weight", f"{b}.attn.context", f"{b}.attn.output", f"{b}.attn.out")
    cur = f"{b}.attn.residual"
    add_node(nodes, "Add", [f"{b}.attn.output", f"{b}.ffn1.residual"], cur, f"{b}.attn.residual")

    # Convolution module: LayerNorm -> pointwise GLU -> causal depthwise
    # Conv -> channel LayerNorm -> explicit SiLU -> pointwise projection.
    conv_in = add_layer_norm(nodes, initializers, source, p + "norm_conv", cur, f"{b}.conv.norm", f"{b}.conv.norm")
    add_linear(nodes, initializers, source, p + "conv.pointwise_conv1.weight", conv_in, f"{b}.conv.pw1", f"{b}.conv.pw1")
    initializers.append(init(f"{b}.conv.glu_split_sizes", [D, D], np.int64))
    add_node(nodes, "Split", [f"{b}.conv.pw1", f"{b}.conv.glu_split_sizes"], [f"{b}.conv.signal", f"{b}.conv.gate"], f"{b}.conv.glu_split", axis=2)
    add_node(nodes, "Sigmoid", [f"{b}.conv.gate"], f"{b}.conv.gate_sigmoid", f"{b}.conv.gate_sigmoid")
    add_node(nodes, "Mul", [f"{b}.conv.signal", f"{b}.conv.gate_sigmoid"], f"{b}.conv.glu", f"{b}.conv.glu")
    add_node(nodes, "Transpose", [f"{b}.conv.glu"], f"{b}.conv.channels_first", f"{b}.conv.to_channels_first", perm=[0, 2, 1])
    causal_pad_1d(nodes, initializers, f"{b}.conv.channels_first", f"{b}.conv.padded", f"{b}.conv.causal_pad")
    dw = np.asarray(source.f32(p + "conv.depthwise_conv.weight"), dtype=np.float32)
    initializers.append(init(f"{b}.conv.dw.weight", dw))
    add_node(nodes, "Conv", [f"{b}.conv.padded", f"{b}.conv.dw.weight"], f"{b}.conv.dw", f"{b}.conv.depthwise", group=D, strides=[1], pads=[0, 0])
    add_node(nodes, "Transpose", [f"{b}.conv.dw"], f"{b}.conv.channels_last", f"{b}.conv.to_channels_last", perm=[0, 2, 1])
    conv_norm = add_layer_norm(nodes, initializers, source, p + "conv.batch_norm", f"{b}.conv.channels_last", f"{b}.conv.norm_out", f"{b}.conv.norm_out")
    add_silu(nodes, conv_norm, f"{b}.conv.silu", f"{b}.conv.silu")
    add_linear(nodes, initializers, source, p + "conv.pointwise_conv2.weight", f"{b}.conv.silu", f"{b}.conv.pw2", f"{b}.conv.pw2")
    cur = f"{b}.conv.residual"
    add_node(nodes, "Add", [f"{b}.conv.pw2", f"{b}.attn.residual"], cur, f"{b}.conv.residual")

    # Macaron FFN 2 and final affine LayerNorm.
    ffn2 = add_layer_norm(nodes, initializers, source, p + "norm_feed_forward2", cur, f"{b}.ffn2.norm", f"{b}.ffn2.norm")
    add_linear(nodes, initializers, source, p + "feed_forward2.linear1.weight", ffn2, f"{b}.ffn2.up", f"{b}.ffn2.up")
    add_silu(nodes, f"{b}.ffn2.up", f"{b}.ffn2.silu", f"{b}.ffn2.silu")
    add_linear(nodes, initializers, source, p + "feed_forward2.linear2.weight", f"{b}.ffn2.silu", f"{b}.ffn2.down", f"{b}.ffn2.down")
    initializers.append(init(f"{b}.ffn2.half.scale", np.array(0.5, dtype=np.float32)))
    add_node(nodes, "Mul", [f"{b}.ffn2.down", f"{b}.ffn2.half.scale"], f"{b}.ffn2.half", f"{b}.ffn2.scale")
    add_node(nodes, "Add", [cur, f"{b}.ffn2.half"], f"{b}.ffn2.residual", f"{b}.ffn2.residual")
    output = add_layer_norm(nodes, initializers, source, p + "norm_out", f"{b}.ffn2.residual", "layer_out", f"{b}.final_norm")

    graph = helper.make_graph(
        nodes,
        f"VOICECHAT_CONFORMER_LAYER_{layer_index}_S2",
        inputs,
        [vi(output, [1, frames, D])],
        initializer=initializers,
    )
    return model_for("layer", graph, {
        "voicechat.component": "conformer-layer",
        "voicechat.layer": str(layer_index),
        "voicechat.static_encoded_frames": str(frames),
        "voicechat.relative_left_context": "70; provisional graph uses full static prefix",
        "voicechat.production_shape_status": "provisional; M4A-2 pending",
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="source VoiceChat GGUF")
    ap.add_argument("--output", type=Path, required=True, help="ignored/generated ONNX output")
    ap.add_argument("--component", choices=("pre-encode", "layer"), required=True)
    ap.add_argument("--frames", type=int, default=None, help="mel frames for pre-encode; encoded frames for layer")
    ap.add_argument("--layer", type=int, default=0, help="Conformer layer for --component layer")
    args = ap.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"missing GGUF input: {args.input}")
    if args.component == "layer" and not 0 <= args.layer < 24:
        raise SystemExit("--layer must be in [0, 23]")
    frames = args.frames if args.frames is not None else (160 if args.component == "pre-encode" else 21)
    source = GGUFSource(args.input)
    model = export_pre_encode(source, frames, args.output) if args.component == "pre-encode" else export_layer(source, args.layer, frames, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(json.dumps({
        "component": args.component,
        "layer": args.layer if args.component == "layer" else None,
        "input": str(args.input),
        "output": str(args.output),
        "frames": frames,
        "initializer_count": len(model.graph.initializer),
        "node_count": len(model.graph.node),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
