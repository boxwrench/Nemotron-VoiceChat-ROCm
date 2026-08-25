#!/usr/bin/env python3
"""Create an ONNX Runtime VitisAI session and optionally run one graph.

Session creation is the compile trigger for the AMD Parakeet-style path. This
helper records providers, I/O shapes, profiling output, and output shapes. It
does not load VoiceChat weights or call the VoiceChat runtime.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        print(f"BLOCKED: ONNX Runtime/numpy import failed: {exc}")
        return 2

    available = ort.get_available_providers()
    print(json.dumps({"onnxruntime": ort.__version__, "available_providers": available}, indent=2))
    if "VitisAIExecutionProvider" not in available:
        print("BLOCKED: VitisAIExecutionProvider is not available in this host Python environment")
        return 3

    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    graphs = sorted(args.graph_dir.glob("*.onnx"))
    if not graphs:
        print(f"BLOCKED: no .onnx graphs found in {args.graph_dir}")
        return 4

    for graph in graphs:
        session_options = ort.SessionOptions()
        session_options.enable_profiling = True
        session_options.profile_file_prefix = str(args.out / graph.stem)
        started = time.perf_counter()
        try:
            session = ort.InferenceSession(
                str(graph),
                sess_options=session_options,
                providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
            )
            load_ms = (time.perf_counter() - started) * 1000.0
            inputs = []
            feed = {}
            for item in session.get_inputs():
                shape = [1 if not isinstance(dim, int) or dim <= 0 else dim for dim in item.shape]
                if "int64" in item.type:
                    data = np.zeros(shape, dtype=np.int64)
                else:
                    data = np.zeros(shape, dtype=np.float32)
                inputs.append({"name": item.name, "shape": shape, "type": item.type})
                feed[item.name] = data

            run_ms = None
            output_shapes = None
            if args.run:
                started = time.perf_counter()
                outputs = session.run(None, feed)
                run_ms = (time.perf_counter() - started) * 1000.0
                output_shapes = [list(value.shape) for value in outputs]

            profile = session.end_profiling()
            result = {
                "graph": str(graph),
                "providers": session.get_providers(),
                "inputs": inputs,
                "session_load_ms": load_ms,
                "run_ms": run_ms,
                "output_shapes": output_shapes,
                "profile": profile,
                "status": "PASS",
            }
            print(json.dumps(result, indent=2))
            results.append(result)
        except Exception as exc:
            result = {"graph": str(graph), "status": "FAIL", "error": repr(exc)}
            print(json.dumps(result, indent=2))
            results.append(result)

    summary = {"run": args.run, "graphs": results}
    (args.out / ("run-summary.json" if args.run else "compile-summary.json")).write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if all(item["status"] == "PASS" for item in results) else 5


if __name__ == "__main__":
    raise SystemExit(main())
