#!/usr/bin/env bash
set -euo pipefail

# Run from an ordinary Strix host shell, not a device-isolated agent
# namespace. This performs no installation or driver/package changes.
GRAPH_DIR="${1:?usage: host_stateful_compile_probe.sh GRAPH_DIR OUTPUT_DIR}"
OUT_DIR="${2:?usage: host_stateful_compile_probe.sh GRAPH_DIR OUTPUT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT_DIR"
exec > >(tee "$OUT_DIR/stateful-compile.log") 2>&1

echo "=== VoiceChat stateful XDNA host compile probe ==="
echo "graph_dir=$GRAPH_DIR"
echo "output_dir=$OUT_DIR"
echo "python=$PYTHON_BIN"
uname -a
"$PYTHON_BIN" --version
for tool in xrt-smi flm xdna-top; do
    if command -v "$tool" >/dev/null 2>&1; then
        echo "$tool=$(command -v "$tool")"
        "$tool" --version 2>&1 | head -5 || true
    else
        echo "$tool=MISSING"
    fi
done

echo "=== device nodes ==="
ls -l /dev/kfd /dev/dri/render* /dev/accel/accel0

echo "=== runtime/provider ==="
"$PYTHON_BIN" - <<'PY'
import json
import onnxruntime as ort
print(json.dumps({"onnxruntime": ort.__version__, "providers": ort.get_available_providers()}, indent=2))
if "VitisAIExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("VitisAIExecutionProvider is unavailable")
PY

echo "=== XRT before ==="
xrt-smi examine
echo "=== ORT session creation ==="
"$PYTHON_BIN" "$ROOT/run_ort_vitisai_probe.py" \
    --graph-dir "$GRAPH_DIR" --out "$OUT_DIR/ort-compile"
echo "PASS: stateful graph session creation completed"
