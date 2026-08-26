# S12-A — canonical quantized MatMul provider control

## Question

Can the isolated public Ryzen AI 1.7.1 VitisAI provider assign a standard
quantized matrix multiplication at the real VoiceChat layer-0 `linear1` shape
to XDNA2?

This is a compiler-path control, not a VoiceChat-fidelity graph.

## Control graph

```text
operator       standard ONNX MatMulInteger
input          int8 [1, 1024]
weight         int8 [1024, 4096]
output         int32 [1, 4096]
graph          one node, one output
target         X2
```

The input and weights are deterministic synthetic int8 values. It deliberately
contains no GGUF Q8 block scales, runtime activation quantization, F16 scale
storage, or VoiceChat-specific arithmetic.

## Result

```text
provider session             PASS
assignment report            PASS
MatMulInteger NPU assignment 0 / 1
MatMulInteger CPU assignment 1 / 1
NPU contexts/activity        none observed by xdna-top
CPU vs provider output       exact

classification               CPU_ONLY
```

The provider emitted a cache/context but assigned the only
`MatMulInteger` node to CPU. Generated cache artifacts are not evidence of
NPU execution.

## Decision

```text
S12-A                         CPU_ONLY
generic VitisAI shape/path    BLOCKED for this control
S12-B exact-Q8 MatMulInteger  NOT RUN
```

Per the S12 decision gate, the exact VoiceChat Q8 rewrite is not attempted:
the simpler standard quantized MatMul already fails to offload at the target
shape. This is not evidence against VoiceChat Q8 fidelity, XDNA hardware, or
a future fused/custom/DynamicDispatch representation. It is evidence that the
current generic VitisAI EP route does not provide the necessary quantized
MatMul assignment on this Linux Strix environment.

## Next boundary

Do not continue generic ONNX rewriting. A separate review may decide whether
AMD DynamicDispatch/custom operators or direct AIE/XRT are justified; no such
work is started by this result.
