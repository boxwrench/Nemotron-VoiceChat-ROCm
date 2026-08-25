# XDNA-LINUX-M0 host evidence

This is a sanitized capture summary from an ordinary host shell on the Strix
Halo system. Usernames, hostnames, PIDs, and local absolute paths are omitted.
No driver, firmware, package, permission, or production-runtime changes were
made for this capture.

## Stack

```text
FastFlowLM                  0.9.39
XRT                         2.21.75
amdxdna                     6.17.0-35-generic
NPU firmware                1.1.2.65
XRT device                  RyzenAI-npu5
```

## Validation

`flm validate --json` reported:

```text
ready=true
amd_device_found=true
all_fw_ok=true
kernel_ok=true
memlock_ok=true
```

## Repeated LLM execution

```text
model                       llama3.2:1b
host executions             2
result                      both successful
```

During execution, `xdna-top` showed active XDNA work owned by the FLM
process. This proves Linux XDNA LLM execution on the actual host, rather than
merely a model-list or software-validation result.

## Acceptance

```text
XDNA-LINUX-M0               PASS
accelerator access          A HOST_OK_SANDBOX_HIDDEN — CONFIRMED
```

The earlier `No such device with index '0'` and `amd_device_found=false`
results belong to the coding-agent execution namespace, whose synthetic `/dev`
hid the host device nodes.
