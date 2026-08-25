# S5 authoritative parity and XDNA qualification

Status: **STOPPED AT GGML↔ONNX PARITY**

This is not an XDNA compiler failure. The S5 graph was not submitted to the
VitisAI provider because its semantics must match the D2 ggml reference first.

## Host boundary

The ordinary Strix host, unlike the agent namespace, exposes:

```text
/dev/kfd
/dev/dri/renderD128
/dev/accel/accel0
XRT 2.21.75 / amdxdna 6.17.0-35-generic / firmware 1.1.2.65
RyzenAI-npu5
```

This confirms the earlier namespace classification:

```text
A HOST_OK_SANDBOX_HIDDEN
```

The experiment-local Python environment and the ordinary host `python3` both
provide ONNX Runtime 1.29.0 with only `AzureExecutionProvider` and
`CPUExecutionProvider`; neither exposes `VitisAIExecutionProvider`. FastFlowLM
continues to prove the host NPU stack for its supported workloads, but it does
not provide a Python VitisAI EP installation for this custom ONNX graph.

## GGML truth capture

An uncommitted, opt-in `VC_D2_STATE_DUMP_*` scratch hook captured exact D2
stream-boundary inputs and outputs from real corpus clips. Captures include
VC01, VC03, and VC05; VC03 step 70 is the first steady-state (`history=70`)
case used below. Captures remain in `/tmp` and are not repository artifacts.

The hook records:

```text
pre_enc_out[1024]
K/V state before the step
conv state before the step
projected[4480]
new K/V/conv state
```

## Authoritative parity result

The source-level NumPy oracle still agrees with CPU ONNX. That result only
validates the exporter transcription against itself. The real steady-state
ggml capture does not agree sufficiently with the ONNX graph.

| boundary | result | evidence |
|---|---|---|
| layer-0 first macaron FFN | first meaningful divergence | cosine 0.9999981, RMSE 0.2973, max abs 1.0497 |
| layer-0 new K | divergent before attention-state reuse | RMSE 0.00571, max abs 0.02247 |
| layer-0 new V | divergent before attention-state reuse | RMSE 0.00699, max abs 0.02371 |
| layer-0 causal-conv state | substantial divergence | cosine 0.98711, RMSE 0.44416, max abs 2.71949 |
| full 24-layer projected output | unacceptable accumulated difference | cosine 0.68701, RMSE 0.03073, max abs 0.11612 |

`GGML_ONNX_FAIL` is therefore the current S5 classification. The initial
runtime dump named an intermediate that the scheduler could reuse; making the
first-FFN tensor an explicit scratch graph output confirmed the earliest
meaningful mismatch is already before relative-position attention and state
transport.

## Consequence

```text
GRAPH REPRESENTATION                 PASS
SOURCE-LEVEL NumPy ↔ ONNX            PASS
AUTHORITATIVE GGML ↔ ONNX             FAIL
XDNA COMPILER ASSIGNMENT              NOT TESTED
XDNA EXECUTION                        NOT TESTED
STATE-TRANSFER ECONOMICS              NOT TESTED
gfx1151 bounded-state control         NOT TESTED
```

No XDNA compiler has rejected a VoiceChat operator or graph. The next single
experiment is exporter attribution and repair at the layer-0 first-macaron-FFN
boundary, followed by the same captured-state parity gate. Only a passing
authoritative parity result authorizes VitisAI compilation.
