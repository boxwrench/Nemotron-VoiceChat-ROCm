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

run_timed_probe() {
    printf '\n## %s\n\n' "$1"
    shift
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    /usr/bin/time -f 'elapsed_seconds=%e exit_status=%x' "$@" 2>&1
    local rc=$?
    printf '\nexit=%d\n' "$rc"
}

printf '# STRIX-BRINGUP-2 agent-visible NPU stack probe\n\n'
printf 'This capture is from the coding-agent shell. It must not be interpreted as a host-namespace result.\n'

run_probe 'uname' uname -a
run_probe 'identity' id
run_probe 'groups' groups
run_probe 'FastFlowLM version' flm --version
run_probe 'FastFlowLM help' flm help
run_probe 'FastFlowLM installed models' flm list --filter installed --json
run_probe 'FastFlowLM validation' flm validate --json
run_probe 'XRT version' xrt-smi --version
run_probe 'XRT examine' xrt-smi examine
run_probe 'xdna-top snapshot' xdna-top --json snapshot
run_probe 'ROCm probe' rocminfo
run_probe 'ROCm SMI' rocm-smi
run_timed_probe 'FLM Llama 1B smoke' bash -c "printf 'Reply with one short greeting.\\n' | flm run llama3.2:1b -i /dev/stdin"
run_timed_probe 'FLM Gemma ASR smoke' bash -c "printf 'Reply with one short greeting.\\n' | flm run gemma4-it:e2b --asr 1 -i /dev/stdin"
