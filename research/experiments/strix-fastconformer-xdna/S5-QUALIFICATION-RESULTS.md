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

## Authoritative parity attribution

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

The initial runtime dump named an intermediate that the scheduler could reuse;
making the first-FFN tensor an explicit scratch graph output confirmed the
earliest meaningful mismatch is already before relative-position attention and
state transport. A second scratch pass rooted every layer-0 first-FFN stage:

| stage | ggml Q8 ↔ F32 control max abs | F32 control ↔ CPU ONNX max abs |
|---|---:|---:|
| normalized, pre-affine | 4.77e-7 | 9.54e-7 |
| normalized, affine | 5.96e-8 | 4.77e-7 |
| `linear1` | **0.15956** | 1.14e-5 |
| SiLU | 0.11029 | 1.14e-5 |
| `linear2` | 2.10016 | 6.71e-4 |
| 0.5 scale | 1.05008 | 3.36e-4 |
| first-FFN residual | 1.05008 | 3.36e-4 |

The source tensor audit is unambiguous:

```text
norm_feed_forward1.weight/bias     F16 -> F32 ONNX initializer
feed_forward1.linear1.weight       Q8_0 -> dequantized F32 initializer
feed_forward1.linear2.weight       Q8_0 -> dequantized F32 initializer
```

The layer-norm outputs match. The first material split is the first Q8 matmul;
the direct `GGUFSource.f32()` control and CPU ONNX agree within a small
CPU-kernel rounding envelope. S5 therefore reclassifies the old generic
`GGML_ONNX_FAIL` as:

```text
graph semantic representation             PASS
GGML-Q8 ↔ dequantized-F32 arithmetic       DIFFERENT PRECISION CONTRACT
dequantized-F32 as accelerator candidate   REJECTED by downstream fidelity
```

This is not evidence that the ONNX topology is wrong, and it is not an XDNA
compiler result.

## F32/dequantized downstream gate

The full 24-layer F32 ONNX encoder was driven frame-by-frame from real VC01
`pre_enc_out` inputs, including a right-aligned startup cache and an explicit
valid-history causal mask. Its projected embeddings were then replayed through
the unchanged VoiceChat timeline using a scratch-only injection hook.

```text
VC01 ggml-Q8:  The capital of France is Paris.
VC01 F32 ONNX: The capital of Valles is the town of Valles itself.
```

This deterministic first-fixture failure is sufficient to reject the
dequantized-F32 contract before spending time on the remaining fixtures:
continuing a VC02–VC06 sweep cannot make it a valid drop-in perception backend.
The result is a fidelity failure, not a claim about XDNA's eventual lowered
precision. Any future accelerator path must reproduce an acceptable deployed
Q8-equivalent arithmetic contract or establish a separately approved model
fidelity envelope.

## Consequence

```text
GRAPH REPRESENTATION                 PASS
SOURCE-LEVEL NumPy ↔ ONNX            PASS
AUTHORITATIVE GGML ↔ ONNX             FAIL
XDNA COMPILER ASSIGNMENT              NOT AUTHORIZED
XDNA EXECUTION                        NOT AUTHORIZED
STATE-TRANSFER ECONOMICS              NOT TESTED
gfx1151 bounded-state control         NOT TESTED
```

No XDNA compiler has rejected a VoiceChat operator or graph. The next single
experiment is not XDNA compilation: it is an isolated investigation of a
Q8-equivalent or otherwise behaviorally valid arithmetic representation for
the first FFN. Only that passing fidelity gate authorizes VitisAI compilation.
