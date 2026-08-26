#!/usr/bin/env bash
set -euo pipefail

# Host-only S13 route-control.  It uses a separate Ryzen AI userspace and the
# public DynamicDispatch source/xclbin; it does not alter drivers or host tools.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMD_PYTHON="${AMD_PYTHON:?set AMD_PYTHON to the isolated Ryzen AI Python}"
DD_ROOT="${DD_ROOT:?set DD_ROOT to a public amd/DynamicDispatch checkout}"
DD_ONNX_OVERLAY="${DD_ONNX_OVERLAY:?set DD_ONNX_OVERLAY to the isolated ONNX overlay}"
OUT_DIR="${1:?usage: host_s13_direct_dd_a16w8_control.sh OUTPUT_DIR}"
DD_XCLBIN="${DD_XCLBIN:-$DD_ROOT/xclbin/stx/mladf_4x2_gemm_a16w8_qdq.xclbin}"

test -f "$DD_XCLBIN" || { echo "xclbin not found: $DD_XCLBIN" >&2; exit 2; }
test -d "$DD_ONNX_OVERLAY" || { echo "ONNX overlay not found: $DD_ONNX_OVERLAY" >&2; exit 2; }
mkdir -p "$OUT_DIR"
if find "$OUT_DIR" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
    echo "refusing non-empty output directory: $OUT_DIR" >&2
    exit 2
fi

VOE_LIB_DIR="$("$AMD_PYTHON" - <<'PY'
from pathlib import Path
import voe
print(Path(voe.__file__).resolve().parent / "lib")
PY
)"
export LD_LIBRARY_PATH="$VOE_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$DD_ONNX_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
exec > >(tee "$OUT_DIR/s13-host.log") 2>&1
"$AMD_PYTHON" - <<'PY'
import json
import numpy as np
import onnx
import ryzenai_dynamic_dispatch as dd
print(json.dumps({"numpy": np.__version__, "onnx": onnx.__version__, "dynamicdispatch": dd.__file__}, indent=2))
PY
xdna-top record --duration 60 --interval 0.25 --out "$OUT_DIR/xdna-top.jsonl" &
TELEMETRY_PID=$!
set +e
"$AMD_PYTHON" "$ROOT/run_s13_direct_dd_a16w8.py" --out "$OUT_DIR" --xclbin "$DD_XCLBIN"
STATUS=$?
set -e
wait "$TELEMETRY_PID" || true
exit "$STATUS"
