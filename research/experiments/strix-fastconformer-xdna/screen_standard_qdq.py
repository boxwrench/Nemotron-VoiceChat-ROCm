#!/usr/bin/env python3
"""Cheap, non-promotional A8W8/A16W8 arithmetic screen for VoiceChat Q8.

This is intentionally *not* a calibrated AMD QDQ model.  It uses a per-row W8
scale and a dynamic per-invocation activation scale only to decide whether a
standard representation is near enough to deserve a separately calibrated
candidate.  It must not be used for a timeline or accelerator claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build" / "llama-voicechat.cpp" / "tools" / "voicechat"))
from vc_gguf import GGUFSource  # noqa: E402


def metric(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, np.float32).reshape(-1)
    actual = np.asarray(actual, np.float32).reshape(-1)
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    return {
        "cosine": float(np.dot(reference, actual) / (np.linalg.norm(reference) * np.linalg.norm(actual))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
    }


def q8_per_row(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.max(np.abs(weight), axis=1) / np.float32(127.0)
    scale = np.where(scale == 0, 1, scale).astype(np.float32)
    return np.rint(np.clip(weight / scale[:, None], -127, 127)).astype(np.int8), scale


def a8w8(weight_q: np.ndarray, weight_scale: np.ndarray, activation: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(activation)) / np.float32(127.0)
    scale = np.float32(1.0) if scale == 0 else scale
    aq = np.rint(np.clip(activation / scale, -127, 127)).astype(np.int8)
    dot = weight_q.astype(np.int32) @ aq.astype(np.int32)
    return dot.astype(np.float32) * weight_scale * scale


def a16w8(weight_q: np.ndarray, weight_scale: np.ndarray, activation: np.ndarray) -> np.ndarray:
    # A16 is represented as F16 activation storage followed by F32 accumulation.
    return (weight_q.astype(np.float32) * weight_scale[:, None]) @ activation.astype(np.float16).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    source = GGUFSource(args.source)
    prefix = "stt_model.perception.encoder.layers.0.feed_forward1."
    w1 = np.asarray(source.f32(prefix + "linear1.weight"), np.float32)
    w2 = np.asarray(source.f32(prefix + "linear2.weight"), np.float32)
    w1q, w1s = q8_per_row(w1)
    w2q, w2s = q8_per_row(w2)
    activation = np.fromfile(args.capture / "vc_stream_l0_ffn1_norm_affine.f32", dtype="<f4")
    ggml_up = np.fromfile(args.capture / "vc_stream_l0_ffn1_up.f32", dtype="<f4")
    ggml_down = np.fromfile(args.capture / "vc_stream_l0_ffn1_down.f32", dtype="<f4")
    report: dict[str, object] = {
        "scope": "dynamic arithmetic screen only; no calibration corpus and no timeline promotion",
        "weight": "symmetric per-output-row W8, derived from raw Q8-dequantized values",
        "activation": "A8W8 dynamic symmetric per-invocation scalar; A16W8 F16 storage",
        "calibration": "none; this is explicitly not an AMD deployment QDQ configuration",
        "candidates": {},
    }
    for name, linear in (("A8W8", a8w8), ("A16W8", a16w8)):
        up = linear(w1q, w1s, activation)
        silu = up / (np.float32(1.0) + np.exp(-up, dtype=np.float32))
        down = linear(w2q, w2s, silu)
        report["candidates"][name] = {"linear1": metric(ggml_up, up), "linear2": metric(ggml_down, down)}
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
