#!/usr/bin/env python3
"""Build a steady-state, one-frame VoiceChat Conformer ONNX graph.

This is a compiler/parity probe, not a production runtime.  It represents the
bounded D2 encoder boundary directly:

    one pre_enc_out frame + 70-frame K/V state + 8-frame convolution state
        -> one encoder output frame + updated bounded state

Weights are read directly from the VoiceChat source GGUF through GGUFSource.
No transcript-producing model or training-framework export is involved.
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
OPS = 17
D = 1024
H = 8
DH = 128
FF = 4096
HIST = 70
CONV_HIST = 8
EPS = 1e-5
QK = 32
# S7 enables this only for a research graph.  The default retained F32 graph
# remains the topology/reference exporter used by the earlier S4/S5 work.
QUANTIZED_LINEAR = False


def vi(name: str, shape: list[int], dtype: int = TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


def init(name: str, value, dtype=None):
    array = np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)
    return numpy_helper.from_array(np.ascontiguousarray(array), name=name)


def node(nodes, op: str, inputs: list[str], output: str | list[str], name: str, **attrs):
    nodes.append(helper.make_node(op, inputs, [output] if isinstance(output, str) else output, name=name, **attrs))
    return output


def linear(nodes, initializers, source: GGUFSource, source_name: str, x: str, out: str, name: str):
    # The deployed layer is mixed arithmetic.  S7 represents only true Q8_0
    # MatMuls explicitly; F16 convolution pointwise projections retain their
    # ordinary F32 ONNX representation instead of being gratuitously int8.
    source_type = source.take(source_name)["ty"]
    if QUANTIZED_LINEAR and source_type == "Q8_0":
        return q8_linear(nodes, initializers, source, source_name, x, out, name)
    if QUANTIZED_LINEAR and source_type == "F16":
        return f16_linear(nodes, initializers, source, source_name, x, out, name)
    weight = np.asarray(source.f32(source_name), dtype=np.float32)
    if weight.ndim > 2:
        weight = np.squeeze(weight, axis=tuple(i for i, size in enumerate(weight.shape) if size == 1))
    weight = weight.T
    wn = f"{name}.weight"
    initializers.append(init(wn, weight, np.float32))
    return node(nodes, "MatMul", [x, wn], out, name)


def f16_linear(nodes, initializers, source: GGUFSource, source_name: str, x: str, out: str, name: str):
    """Preserve a deployed F16 weight's F16 activation conversion boundary.

    The CPU type trait selects F16 as its vec-dot type, so ggml converts the
    F32 activation before its F16 dot kernel.  S7 makes that conversion
    explicit instead of silently widening the source weights into an F32
    MatMul.  The F32 cast after MatMul is the graph's surrounding arithmetic
    type; it is not a new quantization policy.
    """
    weight = np.asarray(source.f32(source_name), dtype=np.float16)
    if weight.ndim > 2:
        weight = np.squeeze(weight, axis=tuple(i for i, size in enumerate(weight.shape) if size == 1))
    weight = weight.T
    prefix = f"{name}.f16"
    initializers.append(init(f"{prefix}.weight", weight, np.float16))
    node(nodes, "Cast", [x], f"{prefix}.activation", f"{prefix}.to_f16", to=TensorProto.FLOAT16)
    # ggml's F16 vec-dot converts both operands to the F16 representation but
    # accumulates the products into F32.  A plain ONNX F16 MatMul may round its
    # result back to F16 before the surrounding graph sees it, so widen the
    # already-quantized operands before the standard F32 MatMul.
    node(nodes, "Cast", [f"{prefix}.activation"], f"{prefix}.activation_f32", f"{prefix}.activation_to_f32", to=TensorProto.FLOAT)
    node(nodes, "Cast", [f"{prefix}.weight"], f"{prefix}.weight_f32", f"{prefix}.weight_to_f32", to=TensorProto.FLOAT)
    return node(nodes, "MatMul", [f"{prefix}.activation_f32", f"{prefix}.weight_f32"], out, name)


def q8_linear(nodes, initializers, source: GGUFSource, source_name: str, x: str, out: str, name: str):
    """Emit a mixed-precision ggml-Q8_0 MatMul with shape-preserving output.

    ggml derives activation integers from an F32 block maximum, stores a
    separate F16 scale, and uses the re-expanded F16 value only when scaling
    the I32 dot product.  Every invocation of this helper re-quantizes its
    activation, matching the deployed execution boundary rather than sharing
    an earlier Q8 representation across operators.
    """
    tensor = source.take(source_name)
    if tensor["ty"] != "Q8_0":
        raise ValueError(f"{source_name}: Q8 research export expected Q8_0, got {tensor['ty']}")
    rows, width = reversed(tensor["dims"])
    if width % QK:
        raise ValueError(f"{source_name}: width {width} is not Q8_0 block aligned")
    blocks = width // QK
    raw = np.frombuffer(source.raw(tensor), dtype=np.uint8).reshape(rows, blocks, 34)
    w_scale = raw[..., :2].copy().view(np.float16).reshape(rows, blocks).astype(np.float32)
    w_q = raw[..., 2:].view(np.int8).reshape(rows, blocks, QK).astype(np.int32)
    prefix = f"{name}.q8"
    initializers.extend([
        init(f"{prefix}.reshape_in", [-1, blocks, QK], np.int64),
        init(f"{prefix}.axis_one", [1], np.int64),
        init(f"{prefix}.axis_two", [2], np.int64),
        init(f"{prefix}.axis_three", [3], np.int64),
        init(f"{prefix}.limit", np.float32(127.0)),
        init(f"{prefix}.clip_low", np.float32(-128.0)),
        init(f"{prefix}.clip_high", np.float32(127.0)),
        init(f"{prefix}.weights_q", w_q[np.newaxis], np.int32),
        init(f"{prefix}.weights_scale", w_scale[np.newaxis], np.float32),
        init(f"{prefix}.slice_start", [0], np.int64),
        init(f"{prefix}.slice_end", [-1], np.int64),
        init(f"{prefix}.slice_axis", [0], np.int64),
        init(f"{prefix}.slice_step", [1], np.int64),
        init(f"{prefix}.output_width", [rows], np.int64),
    ])
    node(nodes, "Reshape", [x, f"{prefix}.reshape_in"], f"{prefix}.act_blocks", f"{prefix}.reshape")
    node(nodes, "Abs", [f"{prefix}.act_blocks"], f"{prefix}.act_abs", f"{prefix}.abs")
    node(nodes, "ReduceMax", [f"{prefix}.act_abs"], f"{prefix}.act_amax", f"{prefix}.amax", axes=[2], keepdims=1)
    node(nodes, "Div", [f"{prefix}.act_amax", f"{prefix}.limit"], f"{prefix}.act_scale_f32", f"{prefix}.scale_f32")
    node(nodes, "Div", [f"{prefix}.act_blocks", f"{prefix}.act_scale_f32"], f"{prefix}.act_scaled", f"{prefix}.normalize")
    node(nodes, "Round", [f"{prefix}.act_scaled"], f"{prefix}.act_rounded", f"{prefix}.round_nearest_even")
    node(nodes, "Clip", [f"{prefix}.act_rounded", f"{prefix}.clip_low", f"{prefix}.clip_high"], f"{prefix}.act_clipped", f"{prefix}.clip")
    node(nodes, "Cast", [f"{prefix}.act_clipped"], f"{prefix}.act_q", f"{prefix}.to_i32", to=TensorProto.INT32)
    node(nodes, "Unsqueeze", [f"{prefix}.act_q", f"{prefix}.axis_one"], f"{prefix}.act_q_expanded", f"{prefix}.expand_activation")
    node(nodes, "Mul", [f"{prefix}.weights_q", f"{prefix}.act_q_expanded"], f"{prefix}.products", f"{prefix}.integer_products")
    node(nodes, "ReduceSum", [f"{prefix}.products", f"{prefix}.axis_three"], f"{prefix}.dot_i32", f"{prefix}.integer_dot", keepdims=0)
    node(nodes, "Cast", [f"{prefix}.dot_i32"], f"{prefix}.dot_f32", f"{prefix}.dot_to_f32", to=TensorProto.FLOAT)
    node(nodes, "Cast", [f"{prefix}.act_scale_f32"], f"{prefix}.act_scale_f16", f"{prefix}.store_scale_f16", to=TensorProto.FLOAT16)
    node(nodes, "Cast", [f"{prefix}.act_scale_f16"], f"{prefix}.act_scale_stored", f"{prefix}.load_scale_f16", to=TensorProto.FLOAT)
    node(nodes, "Squeeze", [f"{prefix}.act_scale_stored", f"{prefix}.axis_two"], f"{prefix}.act_scale_flat", f"{prefix}.squeeze_scale")
    node(nodes, "Unsqueeze", [f"{prefix}.act_scale_flat", f"{prefix}.axis_one"], f"{prefix}.act_scale_row", f"{prefix}.expand_scale")
    node(nodes, "Mul", [f"{prefix}.weights_scale", f"{prefix}.act_scale_row"], f"{prefix}.block_scale", f"{prefix}.block_scale_product")
    node(nodes, "Mul", [f"{prefix}.dot_f32", f"{prefix}.block_scale"], f"{prefix}.scaled_blocks", f"{prefix}.scale_blocks")
    node(nodes, "ReduceSum", [f"{prefix}.scaled_blocks", f"{prefix}.axis_two"], f"{prefix}.flat_output", f"{prefix}.accumulate_blocks", keepdims=0)
    # Restore all leading dimensions of the input and replace only its final
    # feature width.  This covers both [1,1,D] frame paths and [T,D] relative
    # position projection without hard-coding their batch rank.
    node(nodes, "Shape", [x], f"{prefix}.input_shape", f"{prefix}.input_shape")
    node(nodes, "Slice", [f"{prefix}.input_shape", f"{prefix}.slice_start", f"{prefix}.slice_end", f"{prefix}.slice_axis", f"{prefix}.slice_step"], f"{prefix}.leading_shape", f"{prefix}.leading_shape")
    node(nodes, "Concat", [f"{prefix}.leading_shape", f"{prefix}.output_width"], f"{prefix}.output_shape", f"{prefix}.output_shape", axis=0)
    return node(nodes, "Reshape", [f"{prefix}.flat_output", f"{prefix}.output_shape"], out, name)


def affine_norm(nodes, initializers, source: GGUFSource, prefix: str, x: str, out: str, name: str):
    scale = f"{name}.scale"
    bias = f"{name}.bias"
    initializers.extend([
        init(scale, source.f32(prefix + ".weight"), np.float32),
        init(bias, source.f32(prefix + ".bias"), np.float32),
    ])
    return node(nodes, "LayerNormalization", [x, scale, bias], out, name, axis=2, epsilon=EPS)


def silu(nodes, x: str, out: str, name: str):
    sigmoid = f"{name}.sigmoid"
    node(nodes, "Sigmoid", [x], sigmoid, sigmoid)
    return node(nodes, "Mul", [x, sigmoid], out, name)


def add_ffn(nodes, initializers, source: GGUFSource, weight_prefix: str, norm_prefix: str,
            x: str, residual: str, out: str, name: str):
    norm = affine_norm(nodes, initializers, source, norm_prefix, x, f"{name}.norm", f"{name}.norm")
    up = linear(nodes, initializers, source, weight_prefix + ".linear1.weight", norm, f"{name}.up", f"{name}.up")
    activated = silu(nodes, up, f"{name}.silu", f"{name}.silu")
    down = linear(nodes, initializers, source, weight_prefix + ".linear2.weight", activated, f"{name}.down", f"{name}.down")
    half = f"{name}.half"
    initializers.append(init(f"{name}.half.scale", np.float32(0.5)))
    node(nodes, "Mul", [down, f"{name}.half.scale"], half, f"{name}.scale")
    return node(nodes, "Add", [residual, half], out, f"{name}.residual")


def build_layer(source: GGUFSource, layer_index: int = 0, debug_intermediates: bool = False,
                dynamic_history_mask: bool = False):
    p = SRC + f"encoder.layers.{layer_index}."
    b = f"layer{layer_index}"
    nodes = []
    initializers = []

    inputs = [
        vi("pre_enc_out", [1, 1, D]),
        vi("k_hist", [1, H, HIST, DH]),
        vi("v_hist", [1, H, HIST, DH]),
        vi("conv_hist", [1, CONV_HIST, D]),
        vi("pos_freqs", [D // 2]),
        vi("rel_positions", [2 * (HIST + 1) - 1]),
    ]
    if dynamic_history_mask:
        inputs.append(vi("attn_mask", [1, 1, 1, HIST + 1]))

    cur = "pre_enc_out"
    cur = add_ffn(nodes, initializers, source, p + "feed_forward1", p + "norm_feed_forward1",
                  cur, cur, f"{b}.ffn1_residual", f"{b}.ffn1")
    if debug_intermediates:
        # Keep the production topology intact while asking ONNX
        # LayerNormalization for its optional sufficient statistics.  The
        # analysis tool reconstructs the raw normalized vector as
        # (pre_enc_out - mean) * inv_std, then compares it with the opt-in
        # ggml capture before the first Q8 MatMul.
        for candidate in nodes:
            if candidate.name == f"{b}.ffn1.norm":
                candidate.output.extend([f"{b}.ffn1.norm_mean", f"{b}.ffn1.norm_inv_std"])
                break
        else:
            raise RuntimeError("could not locate layer-0 first-FFN LayerNormalization node")

    # Relative Transformer-XL attention.  The steady-state query sees the
    # retained 70 keys/values followed by the new key/value.
    attn_norm = affine_norm(nodes, initializers, source, p + "norm_self_att", cur, f"{b}.attn.norm", f"{b}.attn.norm")
    q = linear(nodes, initializers, source, p + "self_attn.linear_q.weight", attn_norm, f"{b}.attn.qflat", f"{b}.attn.q")
    k = linear(nodes, initializers, source, p + "self_attn.linear_k.weight", attn_norm, f"{b}.attn.kflat", f"{b}.attn.k")
    v = linear(nodes, initializers, source, p + "self_attn.linear_v.weight", attn_norm, f"{b}.attn.vflat", f"{b}.attn.v")
    head_shape = f"{b}.attn.head_shape"
    initializers.append(init(head_shape, [1, 1, H, DH], np.int64))
    for flat, shaped in ((q, f"{b}.attn.qshape"), (k, f"{b}.attn.kshape"), (v, f"{b}.attn.vshape")):
        node(nodes, "Reshape", [flat, head_shape], shaped, shaped)
    qh = node(nodes, "Transpose", [f"{b}.attn.qshape"], f"{b}.attn.qh", f"{b}.attn.q_heads", perm=[0, 2, 1, 3])
    kh_new = node(nodes, "Transpose", [f"{b}.attn.kshape"], f"{b}.attn.kh_new", f"{b}.attn.k_heads", perm=[0, 2, 1, 3])
    vh_new = node(nodes, "Transpose", [f"{b}.attn.vshape"], f"{b}.attn.vh_new", f"{b}.attn.v_heads", perm=[0, 2, 1, 3])
    node(nodes, "Concat", ["k_hist", kh_new], f"{b}.attn.k_all", f"{b}.attn.k_concat", axis=2)
    node(nodes, "Concat", ["v_hist", vh_new], f"{b}.attn.v_all", f"{b}.attn.v_concat", axis=2)

    q_u_bias = f"{b}.attn.pos_bias_u"
    q_v_bias = f"{b}.attn.pos_bias_v"
    initializers.extend([
        # GGUFSource returns this [heads, head_dim] tensor in the same
        # head-major order used by the live ggml add.  Transposing it before
        # reshaping silently interleaves heads and corrupts both attention
        # score paths despite Q/K/V themselves matching.
        init(q_u_bias, source.f32(p + "self_attn.pos_bias_u").reshape(1, H, 1, DH), np.float32),
        init(q_v_bias, source.f32(p + "self_attn.pos_bias_v").reshape(1, H, 1, DH), np.float32),
    ])
    q_u = node(nodes, "Add", [qh, q_u_bias], f"{b}.attn.q_u", f"{b}.attn.q_u")
    q_v = node(nodes, "Add", [qh, q_v_bias], f"{b}.attn.q_v", f"{b}.attn.q_v")
    k_for_scores = node(nodes, "Transpose", [f"{b}.attn.k_all"], f"{b}.attn.k_t", f"{b}.attn.k_transpose", perm=[0, 1, 3, 2])
    content = node(nodes, "MatMul", [q_u, k_for_scores], f"{b}.attn.content", f"{b}.attn.content_scores")

    # The first HIST+1 relative positions are exactly the one-query slice of
    # the runtime's Transformer-XL rel-shift: [70, 69, ..., 0].
    pf = f"{b}.pos_freqs_2d"
    rp = f"{b}.rel_positions_2d"
    initializers.extend([
        init(f"{b}.axis_zero", [0], np.int64),
        init(f"{b}.axis_one", [1], np.int64),
        init(f"{b}.pos_interleaved_shape", [2 * (HIST + 1) - 1, D], np.int64),
    ])
    node(nodes, "Unsqueeze", ["pos_freqs", f"{b}.axis_one"], pf, f"{b}.pos_freqs_unsqueeze")
    node(nodes, "Unsqueeze", ["rel_positions", f"{b}.axis_zero"], rp, f"{b}.rel_positions_unsqueeze")
    node(nodes, "Mul", [pf, rp], f"{b}.theta", f"{b}.theta")
    node(nodes, "Sin", [f"{b}.theta"], f"{b}.sin", f"{b}.sin")
    node(nodes, "Cos", [f"{b}.theta"], f"{b}.cos", f"{b}.cos")
    # ggml concatenates sin/cos along its unit leading dimension before
    # reshaping, yielding [sin(f0), cos(f0), sin(f1), cos(f1), ...] for each
    # relative position.  A plain axis-0 ONNX concat would instead place all
    # sine features before all cosine features and changes the deployed model.
    node(nodes, "Unsqueeze", [f"{b}.sin", f"{b}.axis_one"], f"{b}.sin_pair", f"{b}.sin_pair")
    node(nodes, "Unsqueeze", [f"{b}.cos", f"{b}.axis_one"], f"{b}.cos_pair", f"{b}.cos_pair")
    node(nodes, "Concat", [f"{b}.sin_pair", f"{b}.cos_pair"], f"{b}.pos_pair", f"{b}.pos_pair", axis=1)
    node(nodes, "Transpose", [f"{b}.pos_pair"], f"{b}.pos_time_3d", f"{b}.pos_time_3d", perm=[2, 0, 1])
    node(nodes, "Reshape", [f"{b}.pos_time_3d", f"{b}.pos_interleaved_shape"], f"{b}.pos_time", f"{b}.pos_time")
    pos = linear(nodes, initializers, source, p + "self_attn.linear_pos.weight", f"{b}.pos_time", f"{b}.pos_projected", f"{b}.attn.linear_pos")
    pos_time_shape = f"{b}.attn.pos_time_shape"
    initializers.append(init(pos_time_shape, [2 * (HIST + 1) - 1, H, DH], np.int64))
    # `pos` is time-major [W, H*Dh].  The live ggml view consumes the same
    # values as [Dh, H, W], so transpose time/head/channel before making the
    # [B,H,Dh,W] MatMul operand.  A direct reshape changes relative attention
    # even when the position projection values themselves match.
    node(nodes, "Reshape", [pos, pos_time_shape], f"{b}.pos_time_heads", f"{b}.pos_time_heads")
    node(nodes, "Transpose", [f"{b}.pos_time_heads"], f"{b}.pos_heads_unbatched", f"{b}.pos_heads_transpose", perm=[1, 2, 0])
    node(nodes, "Unsqueeze", [f"{b}.pos_heads_unbatched", f"{b}.axis_zero"], f"{b}.pos_heads_raw", f"{b}.pos_heads_batch")
    # q_v is [B,H,1,Dh]; the raw position projection is [B,H,Dh,W].
    rel_all = node(nodes, "MatMul", [q_v, f"{b}.pos_heads_raw"], f"{b}.attn.rel_all", f"{b}.attn.rel_scores")
    starts = init(f"{b}.rel_starts", [0, 0, 0, 0], np.int64)
    ends = init(f"{b}.rel_ends", [1, H, 1, HIST + 1], np.int64)
    axes = init(f"{b}.rel_axes", [0, 1, 2, 3], np.int64)
    steps = init(f"{b}.rel_steps", [1, 1, 1, 1], np.int64)
    initializers.extend([starts, ends, axes, steps])
    node(nodes, "Slice", [rel_all, f"{b}.rel_starts", f"{b}.rel_ends", f"{b}.rel_axes", f"{b}.rel_steps"], f"{b}.attn.rel", f"{b}.attn.rel_slice")
    node(nodes, "Add", [content, f"{b}.attn.rel"], f"{b}.attn.scores", f"{b}.attn.score_add")
    initializers.append(init(f"{b}.attn.scale", np.float32(1.0 / np.sqrt(DH))))
    node(nodes, "Mul", [f"{b}.attn.scores", f"{b}.attn.scale"], f"{b}.attn.scaled", f"{b}.attn.scale_node")
    if dynamic_history_mask:
        node(nodes, "Add", [f"{b}.attn.scaled", "attn_mask"], f"{b}.attn.masked", f"{b}.attn.mask_add")
    else:
        initializers.append(init(f"{b}.attn.mask", np.zeros((1, 1, 1, HIST + 1), dtype=np.float32)))
        node(nodes, "Add", [f"{b}.attn.scaled", f"{b}.attn.mask"], f"{b}.attn.masked", f"{b}.attn.mask_add")
    node(nodes, "Softmax", [f"{b}.attn.masked"], f"{b}.attn.probs", f"{b}.attn.softmax", axis=-1)
    context = node(nodes, "MatMul", [f"{b}.attn.probs", f"{b}.attn.v_all"], f"{b}.attn.context_heads", f"{b}.attn.value_aggregation")
    node(nodes, "Transpose", [context], f"{b}.attn.context_last", f"{b}.attn.context_merge", perm=[0, 2, 1, 3])
    context_shape = f"{b}.attn.context_shape"
    initializers.append(init(context_shape, [1, 1, D], np.int64))
    node(nodes, "Reshape", [f"{b}.attn.context_last", context_shape], f"{b}.attn.context", f"{b}.attn.context_reshape")
    attn_out = linear(nodes, initializers, source, p + "self_attn.linear_out.weight", f"{b}.attn.context", f"{b}.attn.output", f"{b}.attn.out")
    cur = node(nodes, "Add", [f"{b}.ffn1_residual", attn_out], f"{b}.attn_residual", f"{b}.attn.residual")

    # Causal convolution uses the eight retained GLU vectors plus this frame.
    conv_norm = affine_norm(nodes, initializers, source, p + "norm_conv", cur, f"{b}.conv.norm", f"{b}.conv.norm")
    pw1 = linear(nodes, initializers, source, p + "conv.pointwise_conv1.weight", conv_norm, f"{b}.conv.pw1", f"{b}.conv.pw1")
    split_sizes = f"{b}.conv.split_sizes"
    initializers.append(init(split_sizes, [D, D], np.int64))
    node(nodes, "Split", [pw1, split_sizes], [f"{b}.conv.signal", f"{b}.conv.gate"], f"{b}.conv.split", axis=2)
    node(nodes, "Sigmoid", [f"{b}.conv.gate"], f"{b}.conv.gate_sigmoid", f"{b}.conv.gate_sigmoid")
    node(nodes, "Mul", [f"{b}.conv.signal", f"{b}.conv.gate_sigmoid"], f"{b}.conv.glu", f"{b}.conv.glu")
    node(nodes, "Concat", ["conv_hist", f"{b}.conv.glu"], f"{b}.conv.seq", f"{b}.conv.concat", axis=1)
    node(nodes, "Transpose", [f"{b}.conv.seq"], f"{b}.conv.ncl", f"{b}.conv.to_ncl", perm=[0, 2, 1])
    dw = np.asarray(source.f32(p + "conv.depthwise_conv.weight"), dtype=np.float32)
    initializers.append(init(f"{b}.conv.dw.weight", dw))
    node(nodes, "Conv", [f"{b}.conv.ncl", f"{b}.conv.dw.weight"], f"{b}.conv.dw", f"{b}.conv.depthwise", group=D, strides=[1], pads=[0, 0])
    node(nodes, "Transpose", [f"{b}.conv.dw"], f"{b}.conv.channels_last", f"{b}.conv.to_channels_last", perm=[0, 2, 1])
    conv_norm_out = affine_norm(nodes, initializers, source, p + "conv.batch_norm", f"{b}.conv.channels_last", f"{b}.conv.norm_out", f"{b}.conv.norm_out")
    conv_silu = silu(nodes, conv_norm_out, f"{b}.conv.silu", f"{b}.conv.silu")
    conv_pw2 = linear(nodes, initializers, source, p + "conv.pointwise_conv2.weight", conv_silu, f"{b}.conv.pw2", f"{b}.conv.pw2")
    cur = node(nodes, "Add", [f"{b}.attn_residual", conv_pw2], f"{b}.conv_residual", f"{b}.conv.residual")

    cur = add_ffn(nodes, initializers, source, p + "feed_forward2", p + "norm_feed_forward2",
                  cur, cur, f"{b}.ffn2_residual", f"{b}.ffn2")
    output = affine_norm(nodes, initializers, source, p + "norm_out", cur, "layer_out", f"{b}.final_norm")

    outputs = [
        vi(output, [1, 1, D]),
        vi("kh_new", [1, H, 1, DH]),
        vi("vh_new", [1, H, 1, DH]),
        vi("conv_new", [1, 1, D]),
    ]
    if debug_intermediates:
        # Exposed only for ggml attribution.  These are the exact first-FFN
        # boundaries needed to distinguish a norm/layout error from Q8
        # matrix-multiply arithmetic.  They are never part of the production
        # graph contract.
        outputs.extend([
            vi(f"{b}.ffn1.norm", [1, 1, D]),
            vi(f"{b}.ffn1.norm_mean", [1, 1, 1]),
            vi(f"{b}.ffn1.norm_inv_std", [1, 1, 1]),
            vi(f"{b}.ffn1.up", [1, 1, FF]),
            vi(f"{b}.ffn1.silu", [1, 1, FF]),
            vi(f"{b}.ffn1.down", [1, 1, D]),
            vi(f"{b}.ffn1.half", [1, 1, D]),
            vi(f"{b}.ffn1_residual", [1, 1, D]),
            vi(f"{b}.attn.norm", [1, 1, D]),
            vi(f"{b}.attn.qflat", [1, 1, D]),
            vi(f"{b}.pos_time", [2 * (HIST + 1) - 1, D]),
            vi(f"{b}.pos_projected", [2 * (HIST + 1) - 1, D]),
            vi(f"{b}.attn.content", [1, H, 1, HIST + 1]),
            vi(f"{b}.attn.rel", [1, H, 1, HIST + 1]),
            vi(f"{b}.attn.scaled", [1, H, 1, HIST + 1]),
            vi(f"{b}.attn.probs", [1, H, 1, HIST + 1]),
            vi(f"{b}.attn.output", [1, 1, D]),
            vi(f"{b}.attn_residual", [1, 1, D]),
            vi(f"{b}.conv.norm", [1, 1, D]),
            vi(f"{b}.conv.pw1", [1, 1, 2 * D]),
            vi(f"{b}.conv.dw", [1, D, 1]),
            vi(f"{b}.conv.silu", [1, 1, D]),
            vi(f"{b}.conv.pw2", [1, 1, D]),
            vi(f"{b}.conv_residual", [1, 1, D]),
        ])
    graph = helper.make_graph(
        nodes,
        f"VOICECHAT_STATEFUL_LAYER_{layer_index}_H{HIST}",
        inputs,
        outputs,
        initializer=initializers,
    )
    # Expose the state tensors without copying them through extra graph nodes.
    graph.output[1].name = f"{b}.attn.kh_new"
    graph.output[2].name = f"{b}.attn.vh_new"
    graph.output[3].name = f"{b}.conv.glu"
    model = helper.make_model(
        graph,
        producer_name="Nemotron-VoiceChat-ROCm",
        opset_imports=[helper.make_opsetid("", OPS)],
    )
    for key, value in {
        "voicechat.component": "stateful-conformer-layer",
        "voicechat.layer": str(layer_index),
        "voicechat.history": str(HIST),
        "voicechat.convolution_history": str(CONV_HIST),
        "voicechat.contract_sha": "6da91b8c6e5035110721dd3319f0511376d7487c",
        "voicechat.production_integration": "not authorized; compiler feasibility only",
    }.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value
    checker.check_model(model)
    model = shape_inference.infer_shapes(model)
    checker.check_model(model)
    return model


def build_stack(source: GGUFSource, layers: int = 24, validate: bool = True,
                dynamic_history_mask: bool = False):
    """Merge steady-state layer graphs and add the 4480-wide projection.

    Each layer keeps its own external state tensors.  This intentionally uses
    explicit host-visible state for the first compiler experiment; it does not
    claim that host round-trips are the eventual serving placement.
    """
    graph_nodes = []
    graph_initializers = []
    graph_inputs = [vi("pre_enc_out", [1, 1, D]), vi("pos_freqs", [D // 2]),
                    vi("rel_positions", [2 * (HIST + 1) - 1])]
    if dynamic_history_mask:
        graph_inputs.append(vi("attn_mask", [1, 1, 1, HIST + 1]))
    layer_outputs = []
    state_outputs = []

    for il in range(layers):
        sub = build_layer(source, il, dynamic_history_mask=dynamic_history_mask)
        input_map = {
            "pre_enc_out": "pre_enc_out" if il == 0 else f"layer{il - 1}.output",
            "k_hist": f"k_hist_{il}",
            "v_hist": f"v_hist_{il}",
            "conv_hist": f"conv_hist_{il}",
            "pos_freqs": "pos_freqs",
            "rel_positions": "rel_positions",
            "attn_mask": "attn_mask",
            "layer_out": f"layer{il}.output",
        }
        for state_name, shape in ((f"k_hist_{il}", [1, H, HIST, DH]),
                                  (f"v_hist_{il}", [1, H, HIST, DH]),
                                  (f"conv_hist_{il}", [1, CONV_HIST, D])):
            graph_inputs.append(vi(state_name, shape))
        for item in sub.graph.initializer:
            graph_initializers.append(item)
        for item in sub.graph.node:
            cloned = helper.make_node(
                item.op_type,
                [input_map.get(x, x) for x in item.input],
                [input_map.get(x, x) for x in item.output],
                name=item.name,
            )
            cloned.attribute.extend(item.attribute)
            graph_nodes.append(cloned)
        layer_outputs.append(vi(f"layer{il}.output", [1, 1, D]))
        state_outputs.extend([
            vi(f"layer{il}.attn.kh_new", [1, H, 1, DH]),
            vi(f"layer{il}.attn.vh_new", [1, H, 1, DH]),
            vi(f"layer{il}.conv.glu", [1, 1, D]),
        ])

    # IdentityConnector projection consumed by the Nemotron perception path.
    proj_w = np.asarray(source.f32(SRC + "proj.weight"), dtype=np.float32).T
    proj_b = np.asarray(source.f32(SRC + "proj.bias"), dtype=np.float32)
    graph_initializers.extend([init("projected.weight", proj_w), init("projected.bias", proj_b)])
    node(graph_nodes, "MatMul", [f"layer{layers - 1}.output", "projected.weight"], "projected.matmul", "projected")
    node(graph_nodes, "Add", ["projected.matmul", "projected.bias"], "projected", "projected.bias_add")

    graph = helper.make_graph(
        graph_nodes,
        f"VOICECHAT_STATEFUL_ENCODER_H{HIST}_L{layers}",
        graph_inputs,
        [vi("projected", [1, 1, 4480])] + state_outputs,
        initializer=graph_initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="Nemotron-VoiceChat-ROCm",
        opset_imports=[helper.make_opsetid("", OPS)],
    )
    for key, value in {
        "voicechat.component": "stateful-conformer-encoder",
        "voicechat.layers": str(layers),
        "voicechat.history": str(HIST),
        "voicechat.convolution_history": str(CONV_HIST),
        "voicechat.projected_width": "4480",
        "voicechat.contract_sha": "6da91b8c6e5035110721dd3319f0511376d7487c",
        "voicechat.production_integration": "not authorized; compiler feasibility only",
    }.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value
    if validate:
        checker.check_model(model)
        model = shape_inference.infer_shapes(model)
        checker.check_model(model)
    return model


def externalize_initializers(model, output: Path) -> Path:
    """Move large initializer payloads beside an ONNX graph.

    A 24-layer F32 graph is larger than protobuf's practical single-message
    limit when weights are embedded.  ONNX external data keeps the graph
    structure checkable and lets ORT/compiler tooling load the same tensors.
    """
    data_dir = Path(str(output) + ".data")
    data_dir.mkdir(parents=True, exist_ok=True)
    for tensor in model.graph.initializer:
        array = numpy_helper.to_array(tensor)
        # Keep shape/axis/mask constants inline.  ONNX Runtime's loader and
        # shape inferencer need these small values before external data is
        # materialized; only learned weights and other large tensors need the
        # sidecar representation.
        if array.nbytes <= 1024 * 1024:
            continue
        filename = tensor.name.replace("/", "_") + ".bin"
        path = data_dir / filename
        path.write_bytes(np.ascontiguousarray(array).tobytes())
        tensor.ClearField("raw_data")
        tensor.ClearField("float_data")
        tensor.ClearField("double_data")
        tensor.ClearField("int32_data")
        tensor.ClearField("int64_data")
        tensor.data_location = TensorProto.EXTERNAL
        tensor.external_data.clear()
        entry = tensor.external_data.add()
        entry.key = "location"
        entry.value = f"{data_dir.name}/{filename}"
    return data_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--layers", type=int, choices=(1, 24), default=1)
    ap.add_argument("--debug-intermediates", action="store_true",
                    help="expose first-FFN stages/statistics for ggml attribution (one layer only)")
    ap.add_argument("--dynamic-history-mask", action="store_true",
                    help="make the causal attention validity mask an input for startup-state research")
    ap.add_argument("--q8-linear", action="store_true",
                    help="research-only: represent Q8_0 ggml MatMuls with explicit activation Q8 blocks")
    args = ap.parse_args()
    if not 0 <= args.layer < 24:
        raise SystemExit("--layer must be in [0, 23]")
    global QUANTIZED_LINEAR
    QUANTIZED_LINEAR = args.q8_linear
    source = GGUFSource(args.input)
    if args.debug_intermediates and args.layers != 1:
        raise SystemExit("--debug-intermediates is only supported with --layers 1")
    model = (build_stack(source, args.layers, validate=args.layers != 24,
                         dynamic_history_mask=args.dynamic_history_mask)
             if args.layers == 24 else build_layer(source, args.layer, args.debug_intermediates,
                                                    dynamic_history_mask=args.dynamic_history_mask))
    external_dir = None
    if args.layers == 24:
        external_dir = externalize_initializers(model, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    if args.layers == 24:
        # The checker needs the saved graph path to resolve external-data
        # locations relative to the .onnx file, not the caller's cwd.
        checker.check_model(str(args.output))
    print(json.dumps({
        "output": str(args.output),
        "layer": None if args.layers == 24 else args.layer,
        "layers": args.layers,
        "external_data_dir": str(external_dir) if external_dir else None,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "inputs": [(x.name, [d.dim_value for d in x.type.tensor_type.shape.dim]) for x in model.graph.input],
        "outputs": [(x.name, [d.dim_value for d in x.type.tensor_type.shape.dim]) for x in model.graph.output],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
