# Strix accelerator-access probe

This probe collects host and session visibility only. It does not install
packages, load modules, change permissions, or reconfigure the kernel.

The target hardware is Ryzen AI MAX+ 395 / Strix Halo with Radeon 8060S
(`gfx1151`) and an XDNA2 NPU. The probe distinguishes the layers needed by the
AMD Linux stack:

```text
kernel/device nodes  ->  amdxdna + /dev/accel/accel0
XRT userspace        ->  xrt-smi examine / xrt::device(0)
FastFlowLM           ->  flm validate and model execution
ROCm/iGPU            ->  /dev/kfd and DRM render nodes
```

`probe.sh` emits each command, exit status, and output. A non-zero probe is
evidence about the current shell, not proof that the host hardware is absent.
Run it from a host shell with enough visibility to compare against this
session.

## Run

```bash
research/experiments/strix-accelerator-access/probe.sh \
  > research/experiments/strix-accelerator-access/HOST-PROBE.md
```

The generated file is intentionally a local probe artifact. It records
machine-specific values and should be reviewed before deciding whether it is
durable evidence.
