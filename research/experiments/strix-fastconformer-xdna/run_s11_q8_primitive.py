#!/usr/bin/env python3
"""Qualify the single-output VoiceChat exact-Q8 primitive through VitisAI EP.

This is deliberately the first provider gate: one real captured activation,
one exact-Q8 MatMul, and one output.  It records provider compilation,
assignment-report availability, CPU-exact ONNX parity, and VitisAI parity. It
does not claim NPU execution merely because the provider is registered; the
host wrapper captures xdna-top telemetry concurrently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(reference, actual):
    import numpy as np

    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    actual = np.asarray(actual, dtype=np.float32).reshape(-1)
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(actual))
    return {
        "cosine": float(np.dot(reference, actual) / denominator) if denominator else 0.0,
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
    }


def timed_runs(session, activation, repeats: int, warmup: int):
    import numpy as np

    feed = {"activation": activation}
    for _ in range(warmup):
        session.run(["output"], feed)
    samples = []
    output = None
    for _ in range(repeats):
        start = time.perf_counter()
        output = session.run(["output"], feed)[0]
        samples.append((time.perf_counter() - start) * 1000.0)
    return output, {
        "samples_ms": samples,
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": float(np.percentile(samples, 99)),
        "min_ms": float(np.min(samples)),
        "max_ms": float(np.max(samples)),
    }


def assignment_summary(report_path: Path) -> dict[str, object]:
    """Summarize VAIP's report without guessing from provider registration."""
    if not report_path.exists():
        return {"report_exists": False, "devices": [], "cpu_nodes": None, "accelerator_nodes": None}
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        devices = [str(item.get("device", "UNKNOWN")) for item in data.get("nodeStat", [])]
        cpu_nodes = sum(device.upper() == "CPU" for device in devices)
        accelerator_nodes = len(devices) - cpu_nodes
        return {
            "report_exists": True,
            "devices": sorted(set(devices)),
            "total_nodes": len(devices),
            "cpu_nodes": cpu_nodes,
            "accelerator_nodes": accelerator_nodes,
        }
    except Exception as exc:  # The raw report remains authoritative evidence.
        return {"report_exists": True, "parse_error": repr(exc), "devices": [], "cpu_nodes": None, "accelerator_nodes": None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--activation", type=Path, required=True)
    ap.add_argument("--ggml-reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    # VAIP interprets report/cache paths relative to its per-key compiler
    # directory.  Resolve them before session creation so assignment evidence
    # is written exactly once beside this host experiment.
    args.model = args.model.resolve()
    args.activation = args.activation.resolve()
    args.ggml_reference = args.ggml_reference.resolve()
    args.out = args.out.resolve()
    args.cache_dir = args.cache_dir.resolve()

    import numpy as np
    import onnxruntime as ort

    if "VitisAIExecutionProvider" not in ort.get_available_providers():
        raise SystemExit("VitisAIExecutionProvider is unavailable in this Python environment")
    if args.cache_dir.exists() and any(args.cache_dir.iterdir()):
        raise SystemExit(f"refusing a non-empty cache directory: {args.cache_dir}")
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    report_path = args.out / "operator-assignment-report.txt"
    os.environ["XLNX_ONNX_EP_REPORT_FILE"] = str(report_path)
    os.environ["XLNX_ONNX_EP_VERBOSE"] = "1"
    activation = np.fromfile(args.activation, dtype="<f4").reshape(1, -1)
    ggml_reference = np.fromfile(args.ggml_reference, dtype="<f4").reshape(1, -1)

    cpu = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    cpu_output, cpu_timing = timed_runs(cpu, activation, args.repeats, args.warmup)

    options = {
        "cache_dir": str(args.cache_dir),
        "cache_key": "voicechat-s11-exact-q8-primitive",
        "enable_cache_file_io_in_mem": "0",
        "target": "X2",
    }
    session_options = ort.SessionOptions()
    session_options.enable_profiling = True
    session_options.profile_file_prefix = str(args.out / "vitisai-profile")
    start = time.perf_counter()
    try:
        vitis = ort.InferenceSession(
            str(args.model),
            sess_options=session_options,
            providers=[("VitisAIExecutionProvider", options), "CPUExecutionProvider"],
        )
    except Exception as exc:
        result = {
            "classification": "COMPILER_REJECT",
            "error": repr(exc),
            "onnxruntime": ort.__version__,
            "available_providers": ort.get_available_providers(),
            "model_sha256": sha256(args.model),
        }
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 10
    create_ms = (time.perf_counter() - start) * 1000.0
    if "VitisAIExecutionProvider" not in vitis.get_providers():
        # ORT can register the provider but fail while dlopen-ing its compiler
        # libraries.  It then silently constructs a CPU-only session, which is
        # environment evidence—not a VoiceChat provider-assignment result.
        result = {
            "classification": "RUNTIME_REJECT",
            "error": "VitisAIExecutionProvider was requested but is absent from the created session",
            "onnxruntime": ort.__version__,
            "available_providers": ort.get_available_providers(),
            "session_providers": vitis.get_providers(),
            "session_create_ms": create_ms,
            "model_sha256": sha256(args.model),
        }
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 11
    try:
        vitis_output, vitis_timing = timed_runs(vitis, activation, args.repeats, args.warmup)
        profile = vitis.end_profiling()
    except Exception as exc:
        result = {
            "classification": "RUNTIME_REJECT",
            "error": repr(exc),
            "onnxruntime": ort.__version__,
            "available_providers": ort.get_available_providers(),
            "session_providers": vitis.get_providers(),
            "session_create_ms": create_ms,
            "model_sha256": sha256(args.model),
        }
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 11

    vitis_vs_cpu = metrics(cpu_output, vitis_output)
    vitis_vs_ggml = metrics(ggml_reference, vitis_output)
    cpu_vs_ggml = metrics(ggml_reference, cpu_output)
    assignment = assignment_summary(report_path)
    if vitis_vs_ggml["max_abs"] > 2e-4:
        classification = "NUMERICAL_FAIL"
    elif assignment.get("accelerator_nodes") == 0:
        classification = "CPU_ONLY"
    elif assignment.get("cpu_nodes") == 0:
        classification = "FULL_NPU"
    else:
        classification = "PARTIAL_NPU"
    result = {
        "classification": classification,
        "onnxruntime": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "session_providers": vitis.get_providers(),
        "provider_options": options,
        "model_sha256": sha256(args.model),
        "activation_sha256": sha256(args.activation),
        "ggml_reference_sha256": sha256(args.ggml_reference),
        "model_input_shape": list(activation.shape),
        "model_output_shape": list(vitis_output.shape),
        "session_create_ms": create_ms,
        "cpu_timing": cpu_timing,
        "vitisai_timing": vitis_timing,
        "cpu_exact_onnx_vs_ggml": cpu_vs_ggml,
        "vitisai_vs_cpu_exact_onnx": vitis_vs_cpu,
        "vitisai_vs_ggml": vitis_vs_ggml,
        "operator_assignment_report": str(report_path),
        "operator_assignment": assignment,
        "cache_entries": sorted(str(path.relative_to(args.cache_dir)) for path in args.cache_dir.rglob("*")),
        "profile": profile,
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if classification != "NUMERICAL_FAIL" else 12


if __name__ == "__main__":
    raise SystemExit(main())
