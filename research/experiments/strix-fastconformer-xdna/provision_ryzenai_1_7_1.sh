#!/usr/bin/env bash
# Provision AMD Ryzen AI 1.7.1 into an explicitly supplied isolated path.
# This script never installs .deb packages or mutates system Python/XRT/FLM.
set -euo pipefail

archive="${1:?usage: provision_ryzenai_1_7_1.sh RYZEN_AI_ARCHIVE TARGET_ROOT}"
target_root="${2:?usage: provision_ryzenai_1_7_1.sh RYZEN_AI_ARCHIVE TARGET_ROOT}"

case "$(basename "$archive")" in
    ryzen_ai-1.7.1.tgz) ;;
    *) echo "expected AMD archive named ryzen_ai-1.7.1.tgz, got $(basename "$archive")" >&2; exit 2 ;;
esac
test -f "$archive" || { echo "archive not found: $archive" >&2; exit 2; }
test ! -e "$target_root" || { echo "target already exists; refuse to overwrite: $target_root" >&2; exit 2; }
test -r /opt/xilinx/xrt/setup.sh || { echo "missing /opt/xilinx/xrt/setup.sh; do not repair drivers from this script" >&2; exit 3; }

mkdir -p "$target_root"
tar -xzf "$archive" -C "$target_root"
installer="$(find "$target_root" -maxdepth 3 -type f -name install_ryzen_ai.sh -print -quit)"
test -n "$installer" || { echo "AMD installer not found after extraction" >&2; exit 4; }

bash "$installer" -a yes -p "$target_root/venv"
echo "Provisioned isolated Ryzen AI environment: $target_root/venv"
echo "Next: source /opt/xilinx/xrt/setup.sh && source $target_root/venv/bin/activate"
