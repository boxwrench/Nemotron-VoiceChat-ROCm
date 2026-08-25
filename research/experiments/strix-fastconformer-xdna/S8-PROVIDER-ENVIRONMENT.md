# S8 provider-environment audit

Status: **VITISAI_EP_NOT_INSTALLED**

This is environment evidence only. It does not run a VoiceChat graph on XDNA,
does not change a Python, driver, firmware, or package installation, and does
not revise the S7 Q8 fidelity gate.

## Result

The inspected Strix userspace contains the known-good NPU execution stack:

```text
XRT                         2.21.75
amdxdna package             7.0.0-rc1+git20260310.6b13cb8f4-noble1
FastFlowLM                  0.9.39
Lemonade                    10.6.0
```

Earlier host evidence separately proves that XRT, FLM LLM execution, and FLM
Whisper speech execution work on the physical NPU. The current coding-agent
namespace still hides `/dev/kfd`, `/dev/dri`, and `/dev/accel`; that namespace
fact is not used to downgrade the prior host hardware proof.

The ordinary Python has no ONNX Runtime installation. The isolated exporter
environment has ONNX Runtime 1.29.0 with only:

```text
AzureExecutionProvider
CPUExecutionProvider
```

No `VitisAIExecutionProvider`, `libonnxruntime_providers_vitisai.so`, VAIP
library, or vendor ONNX Runtime build was found in the inspected package,
prefix, cache, and library locations. FastFlowLM installs model-specific NPU
libraries (including its Whisper runtime), not a general custom-ONNX VitisAI
Execution Provider.

## Classification

```text
VITISAI_EP_READY              NO
VITISAI_EP_PRESENT_BUT_BROKEN NO
VITISAI_EP_NOT_INSTALLED      YES

S8 provider execution         PAUSED
XDNA hardware capability      previously proven by FLM host runs
VoiceChat Q8 graph fidelity   retained from S7
```

This is not `PROVIDER_REPRESENTATION_BLOCKED`: the provider has not seen the
exact-Q8 primitive, so no assignment, partitioning, execution, or state
transport conclusion is justified.

## Next required authority

Use `host_provider_audit.sh` from an ordinary host shell to search any
vendor-provided Ryzen AI environment not visible to the coding agent. If that
also reports no provider, the next action requires an explicit decision to
provision a **separate vendor-compatible RyzenAI/VitisAI environment**. Do not
replace the ordinary Python or mutate the working FLM/XRT installation.

Only after a provider is found or provisioned may S8 resume at the existing
exact-Q8 primitive assignment probe:

```text
ggml Q8 capture
  -> CPU exact-Q8 ONNX control
  -> VitisAI assignment report + XRT/xdna-top evidence
```

The first provider result must distinguish full assignment, partial assignment,
fragmentation, CPU-only fallback, compiler rejection, runtime rejection, and
numerical-fidelity failure.
