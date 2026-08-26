#!/usr/bin/env python3
"""Run one single-output ONNX quantized MatMul through CPU and VitisAI EP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assignment_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"report_exists": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    devices = [str(item.get("device", "UNKNOWN")) for item in data.get("nodeStat", [])]
    cpu = sum(device.upper() == "CPU" for device in devices)
    return {
        "report_exists": True,
        "devices": sorted(set(devices)),
        "total_nodes": len(devices),
        "cpu_nodes": cpu,
        "accelerator_nodes": len(devices) - cpu,
    }


def timing(session, feed, repeats: int, warmup: int):
    import numpy as np

    for _ in range(warmup):
        session.run(["output"], feed)
    values, output = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        output = session.run(["output"], feed)[0]
        values.append((time.perf_counter() - start) * 1000.0)
    return output, {"p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)), "p99_ms": float(np.percentile(values, 99)), "max_ms": float(np.max(values)), "samples_ms": values}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--activation", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()
    args.model, args.activation, args.out, args.cache_dir = (args.model.resolve(), args.activation.resolve(), args.out.resolve(), args.cache_dir.resolve())

    import numpy as np
    import onnxruntime as ort

    if "VitisAIExecutionProvider" not in ort.get_available_providers():
        raise SystemExit("VitisAIExecutionProvider unavailable")
    if args.cache_dir.exists() and any(args.cache_dir.iterdir()):
        raise SystemExit(f"refusing non-empty cache directory: {args.cache_dir}")
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    report = args.out / "operator-assignment-report.txt"
    os.environ["XLNX_ONNX_EP_REPORT_FILE"] = str(report)
    os.environ["XLNX_ONNX_EP_VERBOSE"] = "1"
    activation = np.fromfile(args.activation, dtype=np.int8).reshape(1, 1024)
    feed = {"activation": activation}
    cpu = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    cpu_output, cpu_timing = timing(cpu, feed, args.repeats, args.warmup)
    options = {"cache_dir": str(args.cache_dir), "cache_key": "voicechat-s12-canonical-a8w8", "enable_cache_file_io_in_mem": "0", "target": "X2"}
    start = time.perf_counter()
    vitis = ort.InferenceSession(str(args.model), providers=[("VitisAIExecutionProvider", options), "CPUExecutionProvider"])
    session_create_ms = (time.perf_counter() - start) * 1000.0
    if "VitisAIExecutionProvider" not in vitis.get_providers():
        classification = "RUNTIME_REJECT"
        vitis_timing = None
        equal = None
        profile = None
    else:
        vitis_output, vitis_timing = timing(vitis, feed, args.repeats, args.warmup)
        equal = bool(np.array_equal(cpu_output, vitis_output))
        profile = vitis.end_profiling()
        assignment = assignment_summary(report)
        classification = "CPU_ONLY" if assignment.get("accelerator_nodes") == 0 else "FULL_NPU" if assignment.get("cpu_nodes") == 0 else "PARTIAL_NPU"
    assignment = assignment_summary(report)
    result = {
        "classification": classification,
        "onnxruntime": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "session_providers": vitis.get_providers(),
        "model_sha256": sha256(args.model),
        "activation_sha256": sha256(args.activation),
        "session_create_ms": session_create_ms,
        "cpu_timing": cpu_timing,
        "vitisai_timing": vitis_timing,
        "cpu_vitisai_exact": equal,
        "assignment": assignment,
        "profile": profile,
        "cache_entries": sorted(str(p.relative_to(args.cache_dir)) for p in args.cache_dir.rglob("*")),
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
