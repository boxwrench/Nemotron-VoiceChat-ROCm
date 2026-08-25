#!/usr/bin/env bash
set -euo pipefail

GRAPH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_DIR="${1:?usage: host_run_probe.sh GENERATED_GRAPH_DIR OUTPUT_DIR}"
OUT_DIR="${2:?usage: host_run_probe.sh GENERATED_GRAPH_DIR OUTPUT_DIR}"
mkdir -p "$OUT_DIR"

exec > >(tee "$OUT_DIR/run-probe.log") 2>&1

echo "=== STRIX FastConformer/XDNA run probe ==="
echo "graph_dir=$GEN_DIR"
echo "output_dir=$OUT_DIR"
python3 --version
flm --version || true
xrt-smi --version

echo "=== XDNA snapshot before run ==="
xrt-smi examine 2>&1 | tee "$OUT_DIR/xrt-before.log"
xdna-top --json snapshot 2>&1 | tee "$OUT_DIR/xdna-before.json" || true

echo "=== run static subgraphs through VitisAI EP ==="
python3 "$GRAPH_ROOT/run_ort_vitisai_probe.py" \
    --graph-dir "$GEN_DIR" \
    --out "$OUT_DIR/ort-run" \
    --run \
    2>&1 | tee "$OUT_DIR/ort-run.log"

echo "=== XDNA snapshot after run ==="
xrt-smi examine 2>&1 | tee "$OUT_DIR/xrt-after.log"
xdna-top --json snapshot 2>&1 | tee "$OUT_DIR/xdna-after.json" || true

echo "NOTE: a second host terminal should sample xdna-top during session creation"
echo "to prove a transient context when the workload is shorter than one sample."
