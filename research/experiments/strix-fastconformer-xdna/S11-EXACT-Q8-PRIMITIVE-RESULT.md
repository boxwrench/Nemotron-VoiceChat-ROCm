# S11 — exact-Q8 primitive VitisAI qualification

## Scope

This is the first real VoiceChat custom-ONNX provider experiment on Strix. It
uses one real layer-0 `linear1` activation captured from the deployed Q8 ggml
runtime and the behaviorally validated, single-output, exact-Q8 ONNX MatMul.
It is deliberately not a one-layer or full-encoder result.

## Environment

The host stack was not changed. The provider ran only from the isolated public
Ryzen AI 1.7.1 CPython 3.12 userspace with `numpy==1.26.4`:

```text
ONNX Runtime / VitisAI EP  1.23.3.dev20260320
XRT                        2.21.75
amdxdna                    6.17.0-35-generic
NPU firmware               1.1.2.65
target                     RyzenAI-npu5 / XDNA2
```

The public `voe` wheel contains the VAIP compiler libraries. Its library
directory had to be added to `LD_LIBRARY_PATH` for the isolated process only:
the provider binary's embedded AMD build-time RUNPATH is not valid for this
distro-installed XRT layout. No system linker, driver, package, or Python
environment was modified.

The local exporter originally wrote ONNX IR v13. Ryzen AI 1.7.1's bundled ORT
accepts IR v11 at most, so the S11 graph is serialized as IR v11 while retaining
opset 12 and identical arithmetic topology.

## Graph and fidelity gate

```text
graph                 17 nodes, one input, one output
arithmetic            deployed Q8_0: dynamic A8 blocks, I32 dots,
                      F32-derived / F16-stored activation scales
CPU exact-ONNX vs ggml
  RMSE                 6.65e-7
  max abs              4.77e-6
```

## Provider result

```text
provider session       PASS
provider cache/context PASS (compiled context and xmodel emitted)
assignment report      PASS
NPU-assigned nodes     0 / 17
CPU-assigned nodes     17 / 17
XDNA contexts/activity none observed by xdna-top
VitisAI vs CPU ONNX    exact for this run
VitisAI vs ggml        same 4.77e-6 primitive envelope

classification         CPU_ONLY
```

All graph operators—`Reshape`, `Abs`, `ReduceMax`, `Div`, `Round`, `Clip`,
`Cast`, integer `Mul`, and `ReduceSum`—remain on CPU. Provider registration or
the emitted cache is therefore **not** evidence of XDNA execution.

## Decision

```text
XDNA hardware/runtime                 remains PROVEN by existing FLM evidence
public VitisAI provider session        PASS
exact-Q8 graph numerical fidelity      PASS
generic exact-Q8 provider assignment   CPU_ONLY

S11 decision                           PROVIDER_REPRESENTATION_BLOCKED
```

This does not reject XDNA perception. It shows that this explicit standard
ONNX decomposition is not an economical representation for VitisAI EP. Do not
proceed to the multi-output control, mixed-Q8 layer, or 24-layer graph until a
separate review chooses whether a fused/custom XDNA representation is worth
investigating.

## Reproduction

Build the ignored graph and use real captured input/reference tensors, then
run from an ordinary host shell:

```bash
AMD_PYTHON=<isolated-ryzen-ai-python> \
research/experiments/strix-fastconformer-xdna/host_s11_q8_primitive.sh \
  <primitive.onnx> <activation.f32> <ggml-linear1.f32> <ignored-output-dir>
```

The script uses a fresh cache, sets `enable_cache_file_io_in_mem=0`, writes an
absolute `XLNX_ONNX_EP_REPORT_FILE`, and records `xdna-top` alongside the
provider evidence. Generated ONNX files, caches, xmodels, and host evidence
remain ignored by policy.
