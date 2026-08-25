#!/usr/bin/env bash
set +e

run_probe() {
    printf '\n## %s\n\n' "$1"
    shift
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    "$@" 2>&1
    local rc=$?
    printf '\nexit=%d\n' "$rc"
}

printf '# STRIX host accelerator probe\n\n'
printf 'Run this from an ordinary host shell, not the coding-agent sandbox.\n'
printf 'Collection is read-only; no permissions, drivers, or packages are changed.\n'

run_probe 'identity' id
run_probe 'groups' groups
run_probe '/dev/kfd' ls -l /dev/kfd
run_probe '/dev/dri' ls -la /dev/dri
run_probe '/dev/accel' ls -la /dev/accel
run_probe 'stat /dev/kfd' stat /dev/kfd
run_probe 'stat render nodes' bash -c 'stat /dev/dri/render*'
run_probe 'stat NPU node' stat /dev/accel/accel0
run_probe 'loaded accelerator modules' bash -c "lsmod | grep -E 'amdgpu|amdxdna'"
run_probe 'amdxdna module metadata' modinfo amdxdna
run_probe 'accelerator package versions' dpkg-query -W -f='${Package} ${Version}\n' amdxdna-dkms libxrt-npu2 libxrt-utils-npu fastflowlm
run_probe 'PCI devices and drivers' lspci -nnk
run_probe 'ROCm version/probe' rocminfo
run_probe 'ROCm SMI' rocm-smi
run_probe 'XRT version' xrt-smi --version
run_probe 'XRT device examine' xrt-smi examine
run_probe 'FastFlowLM version' flm --version
run_probe 'FastFlowLM validation' flm validate --json
run_probe 'installed FastFlowLM models' flm list --filter installed --json
