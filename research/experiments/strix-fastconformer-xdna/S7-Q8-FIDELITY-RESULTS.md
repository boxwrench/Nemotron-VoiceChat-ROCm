# S7 exact-Q8 arithmetic and behavioral gate

Status: **Q8_XDNA_CANDIDATE — AUTHORIZED FOR PROVIDER QUALIFICATION**

This experiment establishes an accelerator candidate only. It does not run an
XDNA compiler or provider, does not claim an XDNA performance result, and does
not change the production VoiceChat runtime.

## Scope and provenance

The authoritative deployed reference is the local D2 ggml CPU capture from
the VoiceChat Q8 runtime. The candidate graph is constructed from raw Q8_0
GGUF blocks and uses the mixed arithmetic identified in
`S6-ARITHMETIC-RESULTS.md`. Generated captures, ONNX artifacts, external data,
and temporary runtime instrumentation remain outside Git.

## S7-0 — stored-scale correction

The original ONNX primitive mistakenly reused the F32 activation scale in its
scaled-dot product. The deployed contract has two distinct values:

```text
act_scale_f32     = amax / 127; derives activation int8 values
act_scale_stored  = F32 -> F16 -> F32; multiplies the integer block dot
```

| implementation | max abs vs ggml `linear1` |
|---|---:|
| independent raw-Q8 oracle | 4.77e-6 |
| prior ONNX, F32 scale product | 7.8688e-3 |
| corrected ONNX, stored-F16 scale product | 4.77e-6 |

The first material discrepancy was scale-storage semantics, not an unexplained
ONNX reduction envelope. This establishes `EXACT_Q8_ONNX_PRIMITIVE_PASS` for
the captured primitive.

## S7-2 — first macaron FFN

The candidate freshly quantizes each Q8 linear input; it does not reuse the
`linear1` activation blocks for `linear2`.

| boundary | max abs vs ggml |
|---|---:|
| LayerNorm | 4.77e-7 |
| Q8 `linear1` | 4.77e-6 |
| SiLU | 2.86e-6 |
| Q8 `linear2` | 4.58e-5 |
| 0.5 branch | 2.29e-5 |
| residual | 6.10e-5 |

Classification: **EXACT_Q8_FFN1_PASS**.

## S7-3 — mixed-arithmetic one-layer contract

The complete real layer preserves Q8 arithmetic only for deployed Q8 matrix
multiplies. LayerNorm, softmax, relative-position arithmetic, causal
depthwise convolution, SiLU, and residuals retain their deployed F32/F16
semantics. Relative-position parity required three source-semantic details:

- interleave sinusoid/cosine features per frequency;
- keep `pos_bias_u` and `pos_bias_v` head-major, without an extra transpose;
- transpose projected relative-position time/head/channel axes before score
  multiplication.

Authoritative step-70 results (VC03) are:

| required output/state | max abs vs ggml |
|---|---:|
| layer output | 1.14e-4 |
| new K | 4.77e-7 |
| new V | 4.77e-7 |
| new convolution state | 5.25e-6 |

Intermediate relative-score and convolution-residual differences are bounded
F32 reduction-order envelopes (2.84e-3 and 1.63e-3 respectively); softmax
reduces the former to 2.09e-7 probability error. They do not precede a
required-output failure. Classification: **ONE_LAYER_Q8_CONTRACT_PASS**.

## S7-4 — 24-layer graph and behavioral kill gate

The generated steady-state graph contains the real 24 stateful layers plus
the 1024-to-4480 projection. It exposes the intended logical boundary:

```text
pre_enc[1024]
+ 24 * (K[70] + V[70] + conv[8])
-> projected[4480] + updated bounded state
```

The logical persistent state remains 14,548,992 bytes. A captured full-step
comparison gives projected cosine 0.9999651, RMSE 3.54e-4, and max abs
1.38e-3. Later-layer state tensors accumulate bounded SIMD/ONNX F32
reduction-order differences, so this is not claimed to be bit-identical.

The required deterministic behavioral kill gate passed:

```text
fixture: VC01-short
candidate: full 24-layer exact-Q8 ONNX sequence
timeline result: The capital of France is Paris.
classification: PASS
```

This is the first behaviorally validated `Q8_XDNA_CANDIDATE`.

## Deliberate CPU limit

The standard-ONNX exact-Q8 form expands each Q8 block computation into
reshape, runtime quantization, integer dot, stored-scale, and reduction nodes.
It is a semantic/provider-qualification graph, not a CPU serving design. On
the current CPU provider, a complete stateful step measured approximately
3.7–7.3 seconds depending on ORT thread setting; one-thread sequential mode
was worse at 18.7 seconds. Completing VC01–VC06 via this CPU graph would take
roughly 80 minutes and would characterize dispatch overhead rather than the
candidate arithmetic.

Therefore the full six-fixture *ONNX CPU* timeline sweep is deferred until an
accelerator-capable provider can execute the same graph. This is not a failed
fidelity gate and must not be reported as XDNA evidence. The future provider
round must rerun VC01–VC06, including token/function/turn traces, against the
same unchanged VoiceChat timeline.

## Standard quantization screening

A deliberately cheap, non-promotional screen used symmetric per-output-row
W8 values and dynamic per-invocation activation scaling. It has **no
calibration corpus** and is not an AMD deployment-QDQ configuration; it only
asks whether either standard arithmetic family is close enough to justify that
separate work.

| candidate | `linear1` max abs | `linear2` max abs | result |
|---|---:|---:|---|
| A8W8 proxy | 0.834 | 14.356 | suppress before layer/timeline |
| A16W8 proxy | 0.752 | 14.247 | suppress before layer/timeline |

Neither proxy is close enough to promote. A future calibrated candidate would
need frozen non-evaluation data plus explicit granularity, scale, symmetry,
clipping, and rounding policy. This preserves the existing result that higher
nominal precision is not automatically faithful to the deployed Q8 model.

## Decision

```text
deployed Q8 arithmetic oracle       PASS
exact-Q8 ONNX primitive             PASS
exact-Q8 first FFN                  PASS
one-layer mixed arithmetic          PASS
24-layer candidate                  REPRESENTABLE / VC01 PASS
F32/dequantized candidate           REJECTED (previous S6 gate)
A8W8 / A16W8                        SCREENED / NOT PROMOTED
XDNA compiler/execution             NOT TESTED

XDNA authorization                  YES
```

The next single experiment is provider qualification: locate a vendor-matched
VitisAI Execution Provider environment, obtain its assignment report for the
one-layer Q8 graph, then execute it on real captured tensors. Do not treat the
current absence of that provider as a model or compiler rejection.
