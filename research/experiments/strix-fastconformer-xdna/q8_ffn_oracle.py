#!/usr/bin/env python3
"""Independent Q8_0 FFN oracle for the deployed VoiceChat arithmetic path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "build" / "llama-voicechat.cpp" / "tools" / "voicechat"
sys.path.insert(0, str(TOOLS))
from vc_gguf import GGUFSource  # noqa: E402

QK = 32


def metric(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    reference, actual = np.asarray(reference, np.float32).reshape(-1), np.asarray(actual, np.float32).reshape(-1)
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    denom = float(np.linalg.norm(reference) * np.linalg.norm(actual))
    return {"cosine": float(np.dot(reference, actual) / denom), "rmse": float(np.sqrt(np.mean(delta * delta))), "max_abs": float(np.max(np.abs(delta)))}


def quantize_q8_0(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blocks = np.asarray(values, np.float32).reshape(-1, QK)
    scale_f32 = np.max(np.abs(blocks), axis=1) / np.float32(127.0)
    # ggml stores the scale as IEEE F16 before multiplying the rounded int8s.
    scale = scale_f32.astype(np.float16)
    # The active x86 AVX path derives the integer multiplier from the original
    # F32 max, then stores the separate F16 scale.  `_MM_ROUND_NEAREST` is
    # round-to-nearest-even, not the scalar reference path's roundf rule.
    inverse = np.divide(np.float32(127.0), np.max(np.abs(blocks), axis=1), out=np.zeros_like(scale_f32), where=scale_f32 != 0)
    quant = np.clip(np.rint(blocks * inverse[:, None]), -128, 127).astype(np.int8)
    return scale, quant


def raw_q8_rows(source: GGUFSource, name: str) -> tuple[np.ndarray, np.ndarray]:
    tensor = source.take(name)
    if tensor["ty"] != "Q8_0":
        raise ValueError(f"{name}: expected Q8_0, got {tensor['ty']}")
    rows, width = reversed(tensor["dims"])
    raw = np.frombuffer(source.raw(tensor), dtype=np.uint8).reshape(rows, width // QK, 34)
    scales = raw[..., :2].copy().view(np.float16).reshape(rows, width // QK)
    quants = raw[..., 2:].view(np.int8).reshape(rows, width // QK, QK)
    return scales, quants


def q8_matvec(scales: np.ndarray, weights: np.ndarray, activation: np.ndarray, details: bool = False):
    a_scales, a_q = quantize_q8_0(activation)
    dot = np.sum(weights.astype(np.int32) * a_q[None].astype(np.int32), axis=2, dtype=np.int64)
    products = scales.astype(np.float32) * a_scales.astype(np.float32)[None]
    output = np.sum(dot.astype(np.float32) * products, axis=1, dtype=np.float32)
    if not details:
        return output
    return output, {"activation_scale": a_scales.astype(np.float32), "activation_q": a_q,
                    "weight_scale": scales.astype(np.float32), "integer_dot": dot, "scale_product": products}


def f32_matvec(source: GGUFSource, name: str, activation: np.ndarray) -> np.ndarray:
    return np.asarray(source.f32(name), np.float32) @ np.asarray(activation, np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    source = GGUFSource(args.source)
    prefix = "stt_model.perception.encoder.layers.0.feed_forward1."
    affine = np.fromfile(args.capture / "vc_stream_l0_ffn1_norm_affine.f32", dtype="<f4")
    ggml_up = np.fromfile(args.capture / "vc_stream_l0_ffn1_up.f32", dtype="<f4")
    ggml_silu = np.fromfile(args.capture / "vc_stream_l0_ffn1_silu.f32", dtype="<f4")
    ggml_down = np.fromfile(args.capture / "vc_stream_l0_ffn1_down.f32", dtype="<f4")
    up_scale, up_weight = raw_q8_rows(source, prefix + "linear1.weight")
    down_scale, down_weight = raw_q8_rows(source, prefix + "linear2.weight")
    q0_up, up_detail = q8_matvec(up_scale, up_weight, affine, details=True)
    q0_down, down_detail = q8_matvec(down_scale, down_weight, ggml_silu, details=True)
    # Q2 retains the source's Q8-valued weights but removes activation
    # quantization at each matrix multiply.  Q3 is the inverse: F32 weights
    # plus the Q8 activation reconstruction.  Since Q8_0 dequantizes exactly
    # to the source F32 control, Q2 and Q4 are intentionally equivalent here.
    q2_up = f32_matvec(source, prefix + "linear1.weight", affine)
    q2_silu = q2_up / (np.float32(1.0) + np.exp(-q2_up, dtype=np.float32))
    q2_down = f32_matvec(source, prefix + "linear2.weight", q2_silu)
    a1_scale, a1_q = quantize_q8_0(affine)
    q3_up = f32_matvec(source, prefix + "linear1.weight", (a1_q.astype(np.float32) * a1_scale.astype(np.float32)[:, None]).reshape(-1))
    q3_silu = q3_up / (np.float32(1.0) + np.exp(-q3_up, dtype=np.float32))
    a2_scale, a2_q = quantize_q8_0(q3_silu)
    q3_down = f32_matvec(source, prefix + "linear2.weight", (a2_q.astype(np.float32) * a2_scale.astype(np.float32)[:, None]).reshape(-1))
    report = {
        "contract": {"weight_type": "Q8_0", "activation_type": "Q8_0", "block_elements": QK,
                     "scale": "IEEE F16 per block (quant multiplier retains source F32 max)", "quant_round": "x86 AVX round-to-nearest-even", "dot": "I32 product accumulation; SIMD F32 scaled block accumulation", "output": "F32"},
        "q8_oracle": {"linear1": metric(ggml_up, q0_up), "linear2": metric(ggml_down, q0_down),
                      "first_block": {"activation_q": up_detail["activation_q"][0].tolist(), "activation_scale": float(up_detail["activation_scale"][0]), "integer_dot_row0_block0": int(up_detail["integer_dot"][0, 0]), "scale_product_row0_block0": float(up_detail["scale_product"][0, 0])}},
        "variants_vs_ggml": {"Q0_exact_q8": {"linear1": metric(ggml_up, q0_up), "linear2": metric(ggml_down, q0_down)}, "Q2_q8_weights_f32_activation": {"linear1": metric(ggml_up, q2_up), "linear2": metric(ggml_down, q2_down)}, "Q3_f32_weights_q8_activation": {"linear1": metric(ggml_up, q3_up), "linear2": metric(ggml_down, q3_down)}, "Q4_dequantized_f32": {"linear1": metric(ggml_up, q2_up), "linear2": metric(ggml_down, q2_down)}},
    }
    report["classification"] = "Q8_ORACLE_PASS" if max(report["q8_oracle"][key]["max_abs"] for key in ("linear1", "linear2")) < 2e-4 else "Q8_ORACLE_FAIL"
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if report["classification"] == "Q8_ORACLE_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
