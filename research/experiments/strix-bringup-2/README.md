# STRIX-BRINGUP-2

Status: host execution milestones passed; graph compatibility remains open.
No VoiceChat runtime, model, driver, or system package changes are part of this
batch.

The objective was to execute known Linux NPU and speech workloads on this
exact Strix Halo, prove XDNA context ownership and completions, repeat the
LLM run, and freeze the software stack. The coding-agent namespace hides the
device nodes, so the repository preserves both the agent-side control capture
and sanitized host-side evidence.

## Scripts

```text
host_probe.sh
    Run from an ordinary host shell. Read-only node, driver, package, ROCm,
    XRT, and FLM checks.

run_known_npu_probe.sh
    Runs the installed FLM/XRT/ROCm checks and two no-download smoke attempts
    from the current agent shell. Its result is not host evidence.
```

The host probe was run by the machine owner in a shell that could see
`/dev/kfd`, `/dev/dri/render*`, and `/dev/accel/accel0`. It remains in the
repository as a read-only reproduction script. Do not use `sudo` to change
permissions as part of a rerun.

## Results

```text
accelerator access          A HOST_OK_SANDBOX_HIDDEN — CONFIRMED
XDNA-LINUX-M0               PASS
XDNA-SPEECH-LINUX-M0        PASS
Linux XDNA hardware/runtime PROVEN
Linux XDNA LLM execution    PROVEN
Linux XDNA speech execution PROVEN
FastConformer on XDNA       QUALIFY — graph compatibility remains open
```

The prior agent-side failures are conclusively classified as execution-
namespace isolation, not host accelerator failure. The host proof does not
justify replacing FastConformer with Whisper or integrating any workload into
VoiceChat.

See the sanitized [host LLM evidence](host/xnda-linux-m0.md) and [host speech
evidence](host/xnda-speech-linux-m0.md).

## Prior agent-side control result

Installed stack:

```text
FLM v0.9.39
XRT 2.21.75
amdxdna kernel interface 6.17.0-35-generic
```

Installed FLM models include `llama3.2:1b`, `llama3.2:3b`, and
`gemma4-it:e2b` with an ASR capability flag. `flm validate --json` reports
`amd_device_found=false`, `devices=[]`, and `ready=false`.

The smallest LLM smoke and the ASR-enabled Gemma smoke failed immediately with
`No such device with index '0'`; neither loaded a model or created an XDNA
context in that namespace. See `generated-agent-stack-probe.md`.

## Remaining FastConformer questions

The open problem is graph compatibility, not Linux/XDNA execution:

```text
VoiceChat graph representation
static-shape strategy
depthwise-convolution transformation
SiLU/operator support
causal/chunked attention representation
embedding parity
production invocation shape from M4A-2
```

Do not begin VoiceChat integration, substitute Whisper for FastConformer, or
modify the production runtime until the M4A-2 production perception contract
is frozen.
