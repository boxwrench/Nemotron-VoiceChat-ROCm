#!/usr/bin/env python3
"""Run the bounded F32 ONNX encoder over a captured pre-encoder sequence.

This is an S5 fidelity probe only.  It drives the fixed 70-slot graph with a
valid-history mask and right-aligned startup state, reproducing the D2 cache
contract without declaring any particular ONNX shape a production API.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

D, H, DH, HIST, CONV_HIST = 1024, 8, 128, 70, 8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--preenc", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    raw = np.fromfile(args.preenc, dtype="<f4")
    if raw.size % D:
        raise SystemExit(f"{args.preenc}: {raw.size} values is not divisible by {D}")
    frames = raw.reshape(-1, D)
    session = ort.InferenceSession(str(args.graph), providers=["CPUExecutionProvider"])
    required = {"attn_mask", "pre_enc_out"}
    missing = required - {item.name for item in session.get_inputs()}
    if missing:
        raise SystemExit(f"graph lacks the required S5 startup inputs: {sorted(missing)}")
    k = [np.zeros((1, H, HIST, DH), dtype=np.float32) for _ in range(24)]
    v = [np.zeros((1, H, HIST, DH), dtype=np.float32) for _ in range(24)]
    conv = [np.zeros((1, CONV_HIST, D), dtype=np.float32) for _ in range(24)]
    freqs = np.exp(-(np.arange(D // 2, dtype=np.float32) * 2 * np.log(np.float32(10000)) / np.float32(D))).astype(np.float32)
    outputs: list[np.ndarray] = []
    timings_ms: list[float] = []
    for index, frame in enumerate(frames):
        history = min(index, HIST)
        mask = np.full((1, 1, 1, HIST + 1), np.float32(-1e9), dtype=np.float32)
        mask[..., HIST - history:] = 0.0
        rel = np.zeros(2 * (HIST + 1) - 1, dtype=np.float32)
        rel[HIST - history:HIST + 1] = np.arange(history, -1, -1, dtype=np.float32)
        feed: dict[str, np.ndarray] = {
            "pre_enc_out": frame.reshape(1, 1, D),
            "pos_freqs": freqs,
            "rel_positions": rel,
            "attn_mask": mask,
        }
        for layer in range(24):
            feed[f"k_hist_{layer}"] = k[layer]
            feed[f"v_hist_{layer}"] = v[layer]
            feed[f"conv_hist_{layer}"] = conv[layer]
        start = time.perf_counter_ns()
        result = session.run(None, feed)
        timings_ms.append((time.perf_counter_ns() - start) / 1e6)
        outputs.append(result[0].reshape(D * 0 + 4480).copy())
        offset = 1
        for layer in range(24):
            new_k, new_v, new_conv = result[offset:offset + 3]
            offset += 3
            k[layer] = np.concatenate([k[layer][:, :, 1:, :], new_k], axis=2)
            v[layer] = np.concatenate([v[layer][:, :, 1:, :], new_v], axis=2)
            conv[layer] = np.concatenate([conv[layer][:, 1:, :], new_conv], axis=1)
    assembled = np.stack(outputs).astype("<f4", copy=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assembled.tofile(args.output)
    values = np.asarray(timings_ms, dtype=np.float64)
    report = {
        "frames": int(frames.shape[0]),
        "providers": session.get_providers(),
        "dynamic_history_mask": True,
        "timing_ms": {key: float(np.percentile(values, pct)) for key, pct in (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))},
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
