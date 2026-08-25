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

printf '# STRIX accelerator host probe\n\n'
printf 'Collection is read-only; failures are preserved verbatim.\n'

run_probe 'uname' uname -a
run_probe 'identity' id
run_probe 'groups' groups
run_probe 'OS release' cat /etc/os-release
run_probe 'virtualization' systemd-detect-virt
run_probe 'container virtualization' systemd-detect-virt --container
run_probe 'PID 1 cgroup' cat /proc/1/cgroup
run_probe 'probe-shell cgroup' cat /proc/self/cgroup
run_probe 'mount visibility for accelerator paths' bash -c "grep -E '(/dev|kfd|render|accel)' /proc/self/mountinfo"
run_probe 'namespace links' readlink /proc/1/ns/mnt
run_probe 'namespace links (self)' readlink /proc/self/ns/mnt

run_probe '/dev/kfd' ls -l /dev/kfd
run_probe '/dev/dri' ls -la /dev/dri
run_probe '/dev/accel' ls -la /dev/accel
run_probe 'DRM sysfs classes' ls -la /sys/class/drm
run_probe 'accelerator sysfs classes' ls -la /sys/class/accel
run_probe 'device ownership groups' getent group video
run_probe 'device ownership groups' getent group render
run_probe 'device ownership groups' getent group input

run_probe 'PCI devices and drivers' lspci -nnk
run_probe 'loaded AMD modules' bash -c "lsmod | grep -E '(amdgpu|amdxdna|xrt)'"
run_probe 'amdxdna module metadata' modinfo amdxdna
run_probe 'installed accelerator packages' dpkg-query -W -f='${Package} ${Version}\n' amdxdna-dkms libxrt-npu2 libxrt-utils-npu libxrt2 fastflowlm 2>&1

if command -v rocminfo >/dev/null 2>&1; then
    run_probe 'rocminfo' rocminfo
else
    printf '\n## rocminfo\n\nnot installed\n'
fi
if command -v rocm-smi >/dev/null 2>&1; then
    run_probe 'rocm-smi' rocm-smi
else
    printf '\n## rocm-smi\n\nnot installed\n'
fi
if command -v amd-smi >/dev/null 2>&1; then
    run_probe 'amd-smi list' amd-smi list
else
    printf '\n## amd-smi\n\nnot installed\n'
fi
if command -v xrt-smi >/dev/null 2>&1; then
    run_probe 'xrt-smi version' xrt-smi --version
    run_probe 'xrt-smi examine' xrt-smi examine
else
    printf '\n## xrt-smi\n\nnot installed\n'
fi
if command -v flm >/dev/null 2>&1; then
    run_probe 'flm version' flm --version
    run_probe 'flm validate' flm validate
    run_probe 'flm validate JSON' flm validate --json
else
    printf '\n## flm\n\nnot installed\n'
fi
if command -v xdna-top >/dev/null 2>&1; then
    run_probe 'xdna-top snapshot' xdna-top --json snapshot
else
    printf '\n## xdna-top\n\nnot installed\n'
fi
