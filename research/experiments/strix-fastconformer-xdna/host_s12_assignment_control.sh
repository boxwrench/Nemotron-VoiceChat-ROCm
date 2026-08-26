#!/usr/bin/env bash
set -euo pipefail

# Isolated host-only S12-A provider probe; never alters host software.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMD_PYTHON="${AMD_PYTHON:?set AMD_PYTHON to isolated Ryzen AI Python}"
MODEL="${1:?usage: host_s12_assignment_control.sh MODEL ACTIVATION OUTPUT_DIR}"
ACTIVATION="${2:?usage: host_s12_assignment_control.sh MODEL ACTIVATION OUTPUT_DIR}"
OUT_DIR="${3:?usage: host_s12_assignment_control.sh MODEL ACTIVATION OUTPUT_DIR}"
CACHE_DIR="$OUT_DIR/provider-cache"
mkdir -p "$OUT_DIR"
if find "$CACHE_DIR" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
    echo "refusing non-empty cache directory: $CACHE_DIR" >&2
    exit 2
fi
VOE_LIB_DIR="$("$AMD_PYTHON" - <<'PY'
from pathlib import Path
import voe
print(Path(voe.__file__).resolve().parent / "lib")
PY
)"
export LD_LIBRARY_PATH="$VOE_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec > >(tee "$OUT_DIR/s12-host.log") 2>&1
"$AMD_PYTHON" - <<'PY'
import json
import numpy as np
import onnxruntime as ort
print(json.dumps({"numpy": np.__version__, "onnxruntime": ort.__version__, "providers": ort.get_available_providers()}, indent=2))
PY
xdna-top record --duration 60 --interval 0.25 --out "$OUT_DIR/xdna-top.jsonl" &
TELEMETRY_PID=$!
set +e
"$AMD_PYTHON" "$ROOT/run_s12_assignment_control.py" --model "$MODEL" --activation "$ACTIVATION" --out "$OUT_DIR" --cache-dir "$CACHE_DIR"
STATUS=$?
set -e
wait "$TELEMETRY_PID" || true
exit "$STATUS"
