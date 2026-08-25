#!/usr/bin/env bash
set -euo pipefail

GRAPH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:?usage: host_compile_probe.sh OUTPUT_DIR}"
GEN_DIR="$OUT_DIR/generated"
mkdir -p "$OUT_DIR"

exec > >(tee "$OUT_DIR/compile-probe.log") 2>&1

echo "=== STRIX FastConformer/XDNA compile probe ==="
echo "graph_root=$GRAPH_ROOT"
echo "output_dir=$OUT_DIR"
echo "=== tool versions ==="
uname -a
python3 --version
command -v xrt-smi
xrt-smi --version
command -v flm && flm --version
command -v xdna-top && xdna-top --version || true

echo "=== Python capability check ==="
python3 - <<'PY'
import importlib.util
for name in ("onnx", "onnxruntime", "numpy"):
    print(f"{name}={'present' if importlib.util.find_spec(name) else 'missing'}")
PY

if ! python3 -c 'import onnx' >/dev/null 2>&1; then
    echo "BLOCKED: host Python lacks onnx; no graph export can be generated."
    exit 2
fi
if ! python3 -c 'import onnxruntime' >/dev/null 2>&1; then
    echo "BLOCKED: host Python lacks onnxruntime; no VitisAI session can be created."
    exit 3
fi

echo "=== generate provisional static subgraphs ==="
python3 "$GRAPH_ROOT/make_subgraphs.py" --out "$GEN_DIR"

echo "=== XDNA device snapshot before session creation ==="
xrt-smi examine 2>&1 | tee "$OUT_DIR/xrt-before.log"
xdna-top --json snapshot 2>&1 | tee "$OUT_DIR/xdna-before.json" || true

echo "=== create VitisAI sessions (compile trigger) ==="
python3 "$GRAPH_ROOT/run_ort_vitisai_probe.py" \
    --graph-dir "$GEN_DIR" \
    --out "$OUT_DIR/ort-compile" \
    2>&1 | tee "$OUT_DIR/ort-compile.log"

echo "PASS: session creation completed; inspect compile summary and profiles."
