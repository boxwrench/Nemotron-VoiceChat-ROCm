# S4 bounded FastConformer/XDNA feasibility results

This is a compiler/parity checkpoint, not a VoiceChat integration result.
Generated ONNX and external-data files are ignored and remain local.

## Imported contract

The experiment uses the steady-state D2 boundary in
[`D2-IMPORT-CONTRACT.md`](D2-IMPORT-CONTRACT.md): one `pre_enc_out[1024]`
frame, 70-frame K/V history and eight-frame causal-convolution history per
layer, followed by one projected `[4480]` output.

## S4A — stateful encoder graph

### One real layer

`export_stateful_voicechat_onnx.py` reads layer-0 tensors directly from the
VoiceChat source GGUF and represents the real operations:

```text
macaron FFN
relative-position attention over 70 retained frames
causal depthwise convolution over 8 retained GLU frames
channel LayerNorm and SiLU
second macaron FFN
final LayerNorm and state outputs
```

Result:

```text
ONNX checker       PASS
nodes              66
initializers       40
CPU ORT execution  PASS; finite output, repeatable
```

The independent direct-NumPy graph oracle also passes the compiler-spike
parity gate. This is a graph-construction parity result, not yet the ggml
runtime parity gate:

| output | cosine | RMSE | max absolute error |
|---|---:|---:|---:|
| layer output | 0.999999999999645 | 8.25e-6 | 2.07e-4 |
| new K | 0.999999999999886 | 6.22e-7 | 1.97e-6 |
| new V | 0.999999999999848 | 8.21e-7 | 2.80e-6 |
| new convolution state | 0.999999999999828 | 1.63e-6 | 1.26e-5 |

The remaining parity step is a comparison against the actual ggml D2-S1
oracle at identical input/state positions. No production runtime was changed
for this graph experiment.

### All 24 layers and projection

The exporter can merge all 24 real stateful layers and the 1024→4480
projection. Because the embedded F32 initializer payload exceeds a practical
single protobuf message, learned tensors are written as ONNX external data;
small shape/axis/mask constants remain inline.

```text
ONNX checker                  PASS
nodes                         1,586
initializers                  962
inputs                        75
outputs                       73 (projected + 72 state outputs)
CPU ORT load                  1,195 ms on the current host
CPU ORT one-step invocation   123.4 ms on the current host
outputs                       finite
```

The CPU timing is a graph smoke result, not a gfx1151 or XDNA serving result.
The explicit state inputs/outputs intentionally expose the state-transfer
question; no claim is made that host round-tripping 14.55 MiB every 80 ms is
acceptable.

## S4C — causal preencoder/subgraph probes

The existing VoiceChat-shaped probes were regenerated with provisional
`T=160` mel frames and checked on CPU:

```text
FC-SUBGRAPH-1 Pad + depthwise Conv     checker PASS; CPU run PASS
FC-SUBGRAPH-1 fused depthwise Conv     checker PASS; CPU run PASS
FC-SUBGRAPH-2 causal conv module       checker PASS; CPU run PASS
FC-SUBGRAPH-3 relative attention       checker PASS; CPU run PASS
```

The probe harness exposed and fixed one exporter-only error: a 1-D ONNX Conv
must use two spatial pad values, not three. This was not an XDNA rejection.
The real learned preencoder exporter remains separate from these provisional
operator probes and preserves asymmetric causal padding.

## XDNA/compiler boundary

The current Codex execution namespace has no accelerator device nodes:

```text
/dev/kfd             absent in this namespace
/dev/dri/render*     absent in this namespace
/dev/accel/accel0    absent in this namespace
```

The local ORT environment is:

```text
onnx             1.22.0
onnxruntime      1.29.0
providers        AzureExecutionProvider, CPUExecutionProvider
VitisAI EP       unavailable
```

Therefore:

```text
stateful ONNX export             PASS
CPU graph execution              PASS
XDNA compiler compatibility      NOT TESTED
XDNA execution                   NOT TESTED in this namespace
```

This is an execution-environment/provider boundary, not a compiler rejection
of the VoiceChat graph. The existing host-side `XDNA-LINUX-M0` and
`XDNA-SPEECH-LINUX-M0` evidence still proves the Strix host XDNA stack; this
S4 graph has not yet been handed to that host runtime. Use
`host_stateful_compile_probe.sh` with a host environment that exposes the XDNA
provider and device nodes.

## gfx1151 control

The new D2 stateful path was not run on gfx1151 in this namespace. The durable
historical whole-prefix control remains useful only as a reference:

```text
VC01 perception: mean 27.8 ms, p95 29.0 ms
VC01 total S2S: mean 6,342.0 ms, p95 6,396.0 ms
```

Those values are not a control for the new bounded-state graph and must not be
used to claim a D2 cross-backend result.

## Fallback map carried into S4

```text
CONV_2D_DW  RELATED_PATTERN
  pre_conv_2, pre_conv_5 and encoder depthwise convolution; related to AMD
  Parakeet Pad→depthwise-Conv compiler work, exact equivalence unproven.

UNARY ×24  VoiceChat-specific SiLU lead
  one convolution-module SiLU per encoder layer; not AMD's attention-mask
  rewrite issue.
```

## Decision

```text
one-layer stateful graph representation     PASS
24-layer CPU graph representation           PASS
CPU graph parity                            PASS against direct NumPy oracle
ggml-runtime parity                         NOT YET TESTED
XDNA compiler compatibility                 NOT TESTED
XDNA execution/latency/state placement      BLOCKED by current namespace
```

No XDNA compiler has rejected the VoiceChat graph. The next exact experiment
is a host-shell VitisAI/XDNA session-creation and one-step run using the
steady-state layer graph first, then the external-data 24-layer graph if the
provider accepts the layer.
