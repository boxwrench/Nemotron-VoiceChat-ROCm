#!/usr/bin/env python3
"""Attribute the first authoritative ggml-versus-ONNX D2 mismatch.

This research-only tool consumes an opt-in scratch capture from
``VC_D2_STATE_DUMP_*`` plus a debug one-layer ONNX graph.  It keeps three
distinct arithmetic contracts visible:

* A: the deployed ggml Q8 path captured from the live D2 graph;
* B: a direct F32 control using GGUFSource.f32() weights; and
* C: CPU ONNX using the exported F32 initializers.

Agreement between B and C alone is intentionally not treated as authoritative:
both originate from the same dequantized GGUF interpretation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]
VOICECHAT_TOOLS = ROOT / "build" / "llama-voicechat.cpp" / "tools" / "voicechat"
sys.path.insert(0, str(VOICECHAT_TOOLS))
from vc_gguf import GGUFSource  # noqa: E402

EPS = np.float32(1e-5)


def f32(path: Path, count: int) -> np.ndarray:
    value = np.fromfile(path, dtype="<f4")
    if value.size != count:
        raise ValueError(f"{path}: expected {count} F32 values, found {value.size}")
    return value.astype(np.float32, copy=False)


def metric(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    actual = np.asarray(actual, dtype=np.float32).reshape(-1)
    if reference.size != actual.size:
        raise ValueError(f"shape mismatch: {reference.shape} vs {actual.shape}")
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    denom = float(np.linalg.norm(reference) * np.linalg.norm(actual))
    return {
        "cosine": float(np.dot(reference, actual) / denom) if denom else 0.0,
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
    }


def feed_from_capture(capture: Path) -> dict[str, np.ndarray]:
    meta = json.loads((capture / "state.json").read_text(encoding="utf-8"))
    if int(meta["history"]) != 70:
        raise ValueError("the steady-state debug graph requires history=70")
    hidden, heads, d_head = int(meta["hidden"]), int(meta["heads"]), int(meta["head_dim"])
    k = f32(capture / "layer-0-k-in.f32", 70 * heads * d_head).reshape(70, heads, d_head).transpose(1, 0, 2)[None]
    v = f32(capture / "layer-0-v-in.f32", 70 * heads * d_head).reshape(70, heads, d_head).transpose(1, 0, 2)[None]
    return {
        "pre_enc_out": f32(capture / "pre_enc.f32", hidden).reshape(1, 1, hidden),
        "k_hist": k,
        "v_hist": v,
        "conv_hist": f32(capture / "layer-0-conv-in.f32", 8 * hidden).reshape(1, 8, hidden),
        "pos_freqs": np.exp(-(np.arange(hidden // 2, dtype=np.float32) * 2 * np.log(np.float32(10000)) / np.float32(hidden))).astype(np.float32),
        "rel_positions": np.arange(70, -71, -1, dtype=np.float32),
    }


def f32_control(source: GGUFSource, x: np.ndarray) -> dict[str, np.ndarray]:
    """The direct dequantized-F32 control (B), not the deployed Q8 path."""
    prefix = "stt_model.perception.encoder.layers.0."
    x = x.astype(np.float32, copy=False)
    mean = np.mean(x, axis=-1, keepdims=True, dtype=np.float32)
    centered = x - mean
    variance = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
    raw = centered / np.sqrt(variance + EPS, dtype=np.float32)
    affine = raw * source.f32(prefix + "norm_feed_forward1.weight").reshape(1, 1, -1) + source.f32(prefix + "norm_feed_forward1.bias").reshape(1, 1, -1)
    up = affine @ source.f32(prefix + "feed_forward1.linear1.weight").T
    silu = up / (np.float32(1.0) + np.exp(-up, dtype=np.float32))
    down = silu @ source.f32(prefix + "feed_forward1.linear2.weight").T
    scaled = down * np.float32(0.5)
    return {
        "norm_raw": raw,
        "norm_affine": affine,
        "up": up,
        "silu": silu,
        "down": down,
        "scaled": scaled,
        "residual": x + scaled,
    }


def source_audit(source: GGUFSource, model: onnx.ModelProto) -> list[dict[str, object]]:
    prefix = "stt_model.perception.encoder.layers.0."
    bindings = [
        ("norm_feed_forward1.weight", "layer0.ffn1.norm.scale"),
        ("norm_feed_forward1.bias", "layer0.ffn1.norm.bias"),
        ("feed_forward1.linear1.weight", "layer0.ffn1.up.weight"),
        ("feed_forward1.linear2.weight", "layer0.ffn1.down.weight"),
    ]
    inits = {item.name: item for item in model.graph.initializer}
    out = []
    for source_suffix, onnx_name in bindings:
        tensor = source.take(prefix + source_suffix)
        initializer = inits[onnx_name]
        out.append({
            "source_tensor": tensor["name"],
            "gguf_dtype": tensor["ty"],
            "gguf_dims": tensor["dims"],
            "gguf_block": "32 elements / 34 bytes" if tensor["ty"] == "Q8_0" else None,
            "f32_control_shape": list(source.f32(tensor["name"]).shape),
            "onnx_initializer": onnx_name,
            "onnx_initializer_dtype": onnx.TensorProto.DataType.Name(initializer.data_type),
            "export_dequantized": tensor["ty"] in {"Q8_0", "Q4_0"},
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    feed = feed_from_capture(args.capture)
    session = ort.InferenceSession(str(args.graph), providers=["CPUExecutionProvider"])
    actual = dict(zip([item.name for item in session.get_outputs()], session.run(None, feed)))
    source = GGUFSource(args.source)
    control = f32_control(source, feed["pre_enc_out"])

    ggml_files = {
        "norm_raw": "vc_stream_l0_ffn1_norm_raw.f32",
        "norm_affine": "vc_stream_l0_ffn1_norm_affine.f32",
        "up": "vc_stream_l0_ffn1_up.f32",
        "silu": "vc_stream_l0_ffn1_silu.f32",
        "down": "vc_stream_l0_ffn1_down.f32",
        "scaled": "vc_stream_l0_ffn1_scaled.f32",
        "residual": "layer-0-ffn1.f32",
    }
    onnx_values = {
        "norm_raw": (feed["pre_enc_out"] - actual["layer0.ffn1.norm_mean"]) * actual["layer0.ffn1.norm_inv_std"],
        "norm_affine": actual["layer0.ffn1.norm"],
        "up": actual["layer0.ffn1.up"],
        "silu": actual["layer0.ffn1.silu"],
        "down": actual["layer0.ffn1.down"],
        "scaled": actual["layer0.ffn1.half"],
        "residual": actual["layer0.ffn1_residual"],
    }
    report: dict[str, object] = {
        "capture": str(args.capture),
        "graph": str(args.graph),
        "providers": session.get_providers(),
        "stages": {},
        "tensor_audit": source_audit(source, onnx.load(args.graph)),
    }
    stages: dict[str, object] = {}
    for name, filename in ggml_files.items():
        ggml = f32(args.capture / filename, int(np.prod(control[name].shape))).reshape(control[name].shape)
        stages[name] = {
            "ggml_q8_vs_f32_control": metric(ggml, control[name]),
            "ggml_q8_vs_onnx_f32": metric(ggml, onnx_values[name]),
            "f32_control_vs_onnx_f32": metric(control[name], onnx_values[name]),
        }
    report["stages"] = stages

    norm_ok = max(stages[key]["ggml_q8_vs_onnx_f32"]["max_abs"] for key in ("norm_raw", "norm_affine")) < 2e-4
    # CPU ORT and NumPy use different reduction/matmul kernels.  Their worst
    # absolute difference here (sub-1e-3) is three orders of magnitude below
    # the deployed-Q8-versus-F32 FFN split, so preserve it as a rounding
    # envelope instead of misclassifying this result as a graph-semantics bug.
    control_ok = max(stages[key]["f32_control_vs_onnx_f32"]["max_abs"] for key in stages) < 1e-3
    first_matmul_gap = stages["up"]["ggml_q8_vs_f32_control"]["max_abs"]
    report["classification"] = (
        "GGML_Q8_VS_F32_DIFFERENT_PRECISION_CONTRACT"
        if norm_ok and control_ok and first_matmul_gap >= 1e-3
        else "ONNX_SEMANTICS_FAIL"
    )
    report["decision_inputs"] = {
        "norm_matches": norm_ok,
        "f32_control_matches_onnx": control_ok,
        "first_matmul_ggml_q8_vs_f32_max_abs": first_matmul_gap,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
