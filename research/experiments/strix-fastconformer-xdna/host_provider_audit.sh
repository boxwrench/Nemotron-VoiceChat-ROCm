#!/usr/bin/env bash
# Run from an ordinary Strix host shell.  Read-only audit: no installation,
# compilation, model execution, or system configuration changes.
set -euo pipefail

echo '=== host identity / device visibility ==='
id
ls -ld /dev/kfd /dev/dri /dev/accel 2>&1 || true

echo '=== XDNA software ==='
for tool in xrt-smi flm lemonade xdna-top; do
    if command -v "$tool" >/dev/null 2>&1; then
        printf '%s=' "$tool"
        "$tool" --version 2>&1 | head -n 2 || true
    else
        printf '%s=MISSING\n' "$tool"
    fi
done

echo '=== ONNX Runtime providers by Python ==='
for python_bin in python3 "${PYTHON_BIN:-}"; do
    [ -n "$python_bin" ] || continue
    command -v "$python_bin" >/dev/null 2>&1 || continue
    "$python_bin" - <<'PY'
import sys
try:
    import onnxruntime as ort
    print({'python': sys.executable, 'onnxruntime': ort.__version__, 'providers': ort.get_available_providers()})
except Exception as exc:
    print({'python': sys.executable, 'onnxruntime': 'unavailable', 'error': repr(exc)})
PY
done

echo '=== VitisAI / VAIP libraries ==='
for root in /opt /usr/local /usr/lib /usr/lib64 "$HOME/.local" "$HOME/.cache"; do
    [ -d "$root" ] || continue
    find "$root" -type f \( -name 'libonnxruntime_providers_vitisai.so' -o -name 'libvaip*.so' \) -print 2>/dev/null || true
done

echo '=== installed packages ==='
dpkg-query -W -f='${Package} ${Version}\n' 2>/dev/null | grep -Ei 'ryzen|vitis|vaip|xrt|amdxdna|fastflow|onnx' || true
