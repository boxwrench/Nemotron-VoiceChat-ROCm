#!/usr/bin/env bash
set -euo pipefail

# S11 host-only probe.  This intentionally uses an isolated AMD userspace
# environment and neither installs nor changes host drivers or Python.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMD_PYTHON="${AMD_PYTHON:?set AMD_PYTHON to the isolated Ryzen AI 1.7.1 Python}"
MODEL="${1:?usage: host_s11_q8_primitive.sh MODEL ACTIVATION GGML_REFERENCE OUTPUT_DIR}"
ACTIVATION="${2:?usage: host_s11_q8_primitive.sh MODEL ACTIVATION GGML_REFERENCE OUTPUT_DIR}"
REFERENCE="${3:?usage: host_s11_q8_primitive.sh MODEL ACTIVATION GGML_REFERENCE OUTPUT_DIR}"
OUT_DIR="${4:?usage: host_s11_q8_primitive.sh MODEL ACTIVATION GGML_REFERENCE OUTPUT_DIR}"
CACHE_DIR="$OUT_DIR/provider-cache"
mkdir -p "$OUT_DIR"
if find "$CACHE_DIR" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
    echo "refusing non-empty cache directory: $CACHE_DIR" >&2
    exit 2
fi

exec > >(tee "$OUT_DIR/s11-host.log") 2>&1
VOE_LIB_DIR="$("$AMD_PYTHON" - <<'PY'
from pathlib import Path
import voe
print(Path(voe.__file__).resolve().parent / "lib")
PY
)"
test -f "$VOE_LIB_DIR/libxcompiler-core-without-symbol.so"
# The public wheel carries VAIP/VOE compiler libraries, but its provider
# binary retains AMD's build-time RUNPATH.  Supply the wheel's lib directory
# only to this isolated process; do not modify ldconfig or the host stack.
export LD_LIBRARY_PATH="$VOE_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "=== S11 exact-Q8 primitive provider probe ==="
echo "voe_lib_dir=$VOE_LIB_DIR"
echo "model_sha256=$(sha256sum "$MODEL" | awk '{print $1}')"
echo "activation_sha256=$(sha256sum "$ACTIVATION" | awk '{print $1}')"
echo "ggml_reference_sha256=$(sha256sum "$REFERENCE" | awk '{print $1}')"
"$AMD_PYTHON" --version
"$AMD_PYTHON" - <<'PY'
import json
import numpy as np
import onnxruntime as ort
print(json.dumps({"numpy": np.__version__, "onnxruntime": ort.__version__, "providers": ort.get_available_providers()}, indent=2))
if set(ort.get_available_providers()) != {"VitisAIExecutionProvider", "CPUExecutionProvider"}:
    raise SystemExit("unexpected provider set")
PY
xrt-smi examine

# The telemetry is separate evidence: provider registration alone is not proof
# that the graph reaches XDNA.  It writes only into the ignored host-results
# directory supplied by the caller.
xdna-top record --duration 60 --interval 0.25 --out "$OUT_DIR/xdna-top.jsonl" &
TELEMETRY_PID=$!
set +e
"$AMD_PYTHON" "$ROOT/run_s11_q8_primitive.py" \
    --model "$MODEL" \
    --activation "$ACTIVATION" \
    --ggml-reference "$REFERENCE" \
    --out "$OUT_DIR" \
    --cache-dir "$CACHE_DIR"
STATUS=$?
set -e
wait "$TELEMETRY_PID" || true
exit "$STATUS"
