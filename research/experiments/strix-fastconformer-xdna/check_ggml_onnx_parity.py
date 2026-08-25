#!/usr/bin/env python3
"""Compare a captured D2 ggml state step with its ONNX representation.

The capture is produced by the uncommitted, opt-in VC_D2_STATE_DUMP_* runtime
hook documented in PARITY-INSTRUMENTATION.md.  It keeps this verifier outside
the production runtime and makes layout conversion explicit: ggml stores K/V
as [history, heads, head_dim] in contiguous memory while ONNX receives
[1, heads, history, head_dim].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def metric(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    actual = np.asarray(actual, dtype=np.float32).reshape(-1)
    if reference.size != actual.size:
        raise ValueError(f"shape mismatch: {reference.shape} vs {actual.shape}")
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    denom = float(np.linalg.norm(reference) * np.linalg.norm(actual))
    cosine = float(np.dot(reference, actual) / denom) if denom else 0.0
    return {
        "cosine": cosine,
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
    }


def f32(path: Path, count: int | None = None) -> np.ndarray:
    value = np.fromfile(path, dtype="<f4")
    if count is not None and value.size != count:
        raise ValueError(f"{path}: expected {count} F32s, found {value.size}")
    return value


def load_capture(capture: Path, one_layer: bool) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    meta = json.loads((capture / "state.json").read_text(encoding="utf-8"))
    hist, layers, heads, d_head, hidden = (
        int(meta["history"]), int(meta["layers"]), int(meta["heads"]),
        int(meta["head_dim"]), int(meta["hidden"]),
    )
    use_layers = 1 if one_layer else layers
    if hist != 70:
        raise ValueError(f"S5 steady-state graphs require history=70, capture has {hist}")
    feed: dict[str, np.ndarray] = {
        "pre_enc_out": f32(capture / "pre_enc.f32", hidden).reshape(1, 1, hidden),
        "pos_freqs": np.exp(-(np.arange(hidden // 2, dtype=np.float32) * 2 * np.log(np.float32(10000)) / np.float32(hidden))).astype(np.float32),
        "rel_positions": np.arange(hist, -hist - 1, -1, dtype=np.float32),
    }
    expected: dict[str, np.ndarray] = {}
    for il in range(use_layers):
        prefix = capture / f"layer-{il}"
        # ggml tensor shape [Dh, H, T] -> semantic [1, H, T, Dh].
        k_in = f32(prefix.with_name(prefix.name + "-k-in.f32"), hist * heads * d_head).reshape(hist, heads, d_head).transpose(1, 0, 2)[None]
        v_in = f32(prefix.with_name(prefix.name + "-v-in.f32"), hist * heads * d_head).reshape(hist, heads, d_head).transpose(1, 0, 2)[None]
        conv_in = f32(prefix.with_name(prefix.name + "-conv-in.f32"), 8 * hidden).reshape(1, 8, hidden)
        k_out = f32(prefix.with_name(prefix.name + "-k-out.f32"), hidden).reshape(1, heads, 1, d_head)
        v_out = f32(prefix.with_name(prefix.name + "-v-out.f32"), hidden).reshape(1, heads, 1, d_head)
        conv_out = f32(prefix.with_name(prefix.name + "-conv-out.f32"), hidden).reshape(1, 1, hidden)
        suffix = "" if one_layer else f"_{il}"
        feed[f"k_hist{suffix}"] = k_in
        feed[f"v_hist{suffix}"] = v_in
        feed[f"conv_hist{suffix}"] = conv_in
        expected[f"layer{il}.attn.kh_new"] = k_out
        expected[f"layer{il}.attn.vh_new"] = v_out
        expected[f"layer{il}.conv.glu"] = conv_out
        ffn1_path = prefix.with_name(prefix.name + "-ffn1.f32")
        if one_layer and ffn1_path.exists():
            expected[f"layer{il}.ffn1_residual"] = f32(ffn1_path, hidden).reshape(1, 1, hidden)
    if not one_layer:
        expected["projected"] = f32(capture / "projected.f32", 4480).reshape(1, 4480)
    return feed, expected, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    session = ort.InferenceSession(str(args.graph), providers=["CPUExecutionProvider"])
    one_layer = len(session.get_inputs()) == 6
    feed, expected, meta = load_capture(args.capture, one_layer)
    outputs = session.run(None, feed)
    actual = {info.name: value for info, value in zip(session.get_outputs(), outputs)}
    report = {
        "classification": "GGML_ONNX_PARITY_PASS",
        "capture": str(args.capture),
        "graph": str(args.graph),
        "history": meta["history"],
        "ort_providers": session.get_providers(),
        "outputs": {},
    }
    # The complete graph has an authoritative projected output.  For one layer,
    # only state outputs are captured by the current runtime hook.
    for name, reference in expected.items():
        if name not in actual:
            raise ValueError(f"ONNX graph omitted expected output {name}")
        report["outputs"][name] = metric(reference, actual[name])
    max_abs = max(item["max_abs"] for item in report["outputs"].values())
    if max_abs >= 1e-2:
        report["classification"] = "GGML_ONNX_FAIL"
    elif max_abs >= 1e-3:
        report["classification"] = "GGML_ONNX_NUMERICAL_ENVELOPE"
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["classification"] != "GGML_ONNX_FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
