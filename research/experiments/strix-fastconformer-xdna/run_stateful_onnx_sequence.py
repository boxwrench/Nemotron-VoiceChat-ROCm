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
    ap.add_argument("--start-frame", type=int, default=0,
                    help="logical frame index to start (requires --state-in when nonzero)")
    ap.add_argument("--frame-count", type=int,
                    help="maximum frames to execute from --start-frame")
    ap.add_argument("--state-in", type=Path,
                    help="NPZ state checkpoint produced by a prior bounded run")
    ap.add_argument("--state-out", type=Path,
                    help="write updated state checkpoint after this bounded run")
    ap.add_argument("--append-output", action="store_true",
                    help="append this run's embeddings to an existing --output file")
    ap.add_argument("--intra-op-threads", type=int,
                    help="override ORT CPU intra-op thread count for dispatch studies")
    args = ap.parse_args()
    raw = np.fromfile(args.preenc, dtype="<f4")
    if raw.size % D:
        raise SystemExit(f"{args.preenc}: {raw.size} values is not divisible by {D}")
    frames = raw.reshape(-1, D)
    start_frame = args.start_frame
    if start_frame < 0 or start_frame > frames.shape[0]:
        raise SystemExit(f"--start-frame {start_frame} is outside 0..{frames.shape[0]}")
    stop = frames.shape[0] if args.frame_count is None else min(frames.shape[0], start_frame + args.frame_count)
    if start_frame and args.state_in is None:
        raise SystemExit("--start-frame requires --state-in so cache history is not silently reset")
    k = [np.zeros((1, H, HIST, DH), dtype=np.float32) for _ in range(24)]
    v = [np.zeros((1, H, HIST, DH), dtype=np.float32) for _ in range(24)]
    conv = [np.zeros((1, CONV_HIST, D), dtype=np.float32) for _ in range(24)]
    if args.state_in:
        checkpoint = np.load(args.state_in)
        checkpoint_index = int(checkpoint["next_frame"])
        if checkpoint_index != start_frame:
            raise SystemExit(f"{args.state_in}: next_frame={checkpoint_index}, expected {start_frame}")
        for layer in range(24):
            k[layer] = checkpoint[f"k_{layer}"]
            v[layer] = checkpoint[f"v_{layer}"]
            conv[layer] = checkpoint[f"conv_{layer}"]
    freqs = np.exp(-(np.arange(D // 2, dtype=np.float32) * 2 * np.log(np.float32(10000)) / np.float32(D))).astype(np.float32)
    outputs: list[np.ndarray] = []
    timings_ms: list[float] = []
    session_options = ort.SessionOptions()
    if args.intra_op_threads:
        session_options.intra_op_num_threads = args.intra_op_threads
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(args.graph), sess_options=session_options, providers=["CPUExecutionProvider"])
    required = {"attn_mask", "pre_enc_out"}
    missing = required - {item.name for item in session.get_inputs()}
    if missing:
        raise SystemExit(f"graph lacks the required S5 startup inputs: {sorted(missing)}")
    for index in range(start_frame, stop):
        frame = frames[index]
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
        compute_start = time.perf_counter_ns()
        result = session.run(None, feed)
        timings_ms.append((time.perf_counter_ns() - compute_start) / 1e6)
        outputs.append(result[0].reshape(D * 0 + 4480).copy())
        offset = 1
        for layer in range(24):
            new_k, new_v, new_conv = result[offset:offset + 3]
            offset += 3
            k[layer] = np.concatenate([k[layer][:, :, 1:, :], new_k], axis=2)
            v[layer] = np.concatenate([v[layer][:, :, 1:, :], new_v], axis=2)
            conv[layer] = np.concatenate([conv[layer][:, 1:, :], new_conv], axis=1)
    assembled = np.stack(outputs).astype("<f4", copy=False) if outputs else np.empty((0, 4480), dtype="<f4")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.append_output and args.output.exists():
        with args.output.open("ab") as handle:
            assembled.tofile(handle)
    else:
        assembled.tofile(args.output)
    if args.state_out:
        args.state_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"next_frame": np.asarray(stop, dtype=np.int64)}
        for layer in range(24):
            payload[f"k_{layer}"] = k[layer]
            payload[f"v_{layer}"] = v[layer]
            payload[f"conv_{layer}"] = conv[layer]
        np.savez(args.state_out, **payload)
    values = np.asarray(timings_ms, dtype=np.float64)
    report = {
        "frames": int(stop - start_frame),
        "start_frame": int(start_frame),
        "next_frame": int(stop),
        "providers": session.get_providers(),
        "dynamic_history_mask": True,
        "timing_ms": ({key: float(np.percentile(values, pct)) for key, pct in (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))} if values.size else {}),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
