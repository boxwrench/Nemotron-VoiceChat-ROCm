#!/usr/bin/env bash
# Create an isolated public-wheel Ryzen AI 1.7.1 userspace.
#
# This is intentionally a userspace-only provisioner.  It must not be run as
# root and never invokes apt, dpkg, sudo, or a driver/firmware installer.
# It does not create an inference session: provider compilation/execution is
# an S9 follow-up gate, not part of provisioning.
set -euo pipefail

target_root="${1:?usage: provision_public_ryzenai_1_7_1.sh TARGET_ROOT}"
python_bin="${PYTHON_BIN:-python3.12}"
amd_index="https://pypi.amd.com/ryzenai_llm/1.7.1/linux/simple/"

command -v "$python_bin" >/dev/null || {
    echo "missing required CPython 3.12 interpreter: $python_bin" >&2
    exit 2
}
"$python_bin" - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"expected CPython 3.12, got {sys.version}")
PY

test ! -e "$target_root" || {
    echo "target already exists; refusing to overwrite: $target_root" >&2
    exit 2
}

"$python_bin" -m venv "$target_root"
pip="$target_root/bin/pip"
python="$target_root/bin/python"

# AMD's ORT 1.23.3 wheel was built against NumPy 1.x.  Do not let pip choose
# NumPy 2.x: import then fails before the provider can register.
"$pip" install --disable-pip-version-check --no-input \
    'numpy==1.26.4' flatbuffers protobuf packaging coloredlogs sympy \
    colorlog ml-dtypes onnx onnx-tool pyyaml rich

# The AMD index contains the complete 1.7.1 Linux userspace chain.  Keep it
# separate from the normal Python environment and install the exact ABI set.
"$pip" install --disable-pip-version-check --no-input \
    --index-url "$amd_index" --extra-index-url https://pypi.org/simple \
    'voe==1.7.1' \
    'onnxruntime-vitisai==1.23.3' \
    'onnxruntime-providers-ryzenai==0.11.1' \
    'ryzenai-dynamic-dispatch==1.7.1' \
    'ryzenai-onnx-utils==0.12.0'

voe_lib="$target_root/lib/python3.12/site-packages/voe/lib"
test -d "$voe_lib" || {
    echo "VOE library directory was not installed: $voe_lib" >&2
    exit 3
}

cat >"$target_root/activate-ryzenai.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
if [ -r /opt/xilinx/xrt/setup.sh ]; then
    # shellcheck disable=SC1091
    source /opt/xilinx/xrt/setup.sh
fi
source "$target_root/bin/activate"
export RYZEN_AI_INSTALLATION_PATH="$target_root"
export LD_LIBRARY_PATH="$voe_lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
EOF
chmod 0755 "$target_root/activate-ryzenai.sh"

"$python" - <<'PY'
import onnxruntime as ort
providers = ort.get_available_providers()
print(f"onnxruntime={ort.__version__}")
print(f"providers={providers}")
required = {"VitisAIExecutionProvider", "CPUExecutionProvider"}
missing = required.difference(providers)
if missing:
    raise SystemExit(f"missing providers: {sorted(missing)}")
PY

echo "Provisioned isolated public Ryzen AI 1.7.1 userspace: $target_root"
echo "Activate only when needed: source $target_root/activate-ryzenai.sh"
echo "No NPU session was created; run the separate S9 provider quicktest next."
