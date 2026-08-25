# S6 deployed-Q8 arithmetic qualification

Status: **STOPPED AT THE ARITHMETIC-REPRESENTATION GATE**

S6 does not authorize an XDNA compilation. Its purpose was to establish the
deployed numerical contract after S5 proved that dequantized-F32 topology is
not behaviorally equivalent to VoiceChat Q8.

## Q8-ARITHMETIC-CONTRACT-M1

The S5 reference capture executes the perception graph on the ggml CPU
backend. This local scratch build targets the Strix x86_64 CPU with
`-march=native`; its Q8_0 type trait is:

```text
weight tensor          Q8_0
activation conversion  quantize_row_q8_0
vec_dot type           Q8_0
vec_dot                ggml_vec_dot_q8_0_q8_0
weight block           32 signed int8 values + one F16 scale
activation block       32 signed int8 values + one F16 scale
activation rounding    x86 AVX round-to-nearest-even
quant multiplier       source F32 block amax / 127, then F16 scale stored
integer arithmetic     signed I8 products accumulated as I32
scaled accumulation     SIMD/F32; output F32
```

This is the actual contract for the captured layer-0 FFN, not a generic
assumption about another ggml backend.

## Independent oracle

`q8_ffn_oracle.py` reads raw Q8_0 GGUF blocks and a captured post-LayerNorm
activation. It does not call the live ggml graph for its candidate result.

| boundary | ggml vs independent Q8 max abs |
|---|---:|
| layer-0 `linear1` | 4.77e-6 |
| layer-0 `linear2` | 6.10e-5 |

`Q8_ORACLE_PASS` is therefore established. The oracle also exposes activation
quantization, integer block dot products, and scale products for individual
blocks so the first mismatch can be localized without compensating later.

## Controlled arithmetic attribution

| candidate | linear1 max abs vs ggml | linear2 max abs vs ggml | interpretation |
|---|---:|---:|---|
| Q0/Q1 exact Q8_0 × Q8_0 | 4.77e-6 | 6.10e-5 | deployed control / independent oracle |
| Q2 Q8-valued weights + F32 activation | 0.15956 | 2.10004 | removes the critical activation quantization |
| Q3 dequantized-F32 weights + Q8 activation | 6.68e-6 | 2.44e-4 | preserves the decisive activation-side behavior locally |
| Q4 dequantized-F32 | 0.15956 | 2.10004 | known rejected control |

The source Q8_0 weight values dequantize exactly to the F32 control, so Q2 and
Q4 collapse mathematically in this experiment. The important distinction is
not merely the stored weight format; it is the per-32-element activation Q8
conversion and its scale/rounding behavior.

## Exact-Q8 ONNX primitive

`build_exact_q8_matmul_onnx.py` represents one real `linear1` as standard
ONNX operations: reshape into 32-element blocks, runtime activation quantize,
I32 products/reduction, F16-derived scale products, and block accumulation.

```text
ONNX checker / CPU execution   PASS
topology                        15 nodes, opset 12
CPU ONNX vs ggml linear1        max abs 0.00787
```

The graph represents the block arithmetic, but CPU ONNX reduction/FMA order
does not reproduce the ggml SIMD result at the same tight envelope as the
independent oracle. This is an `EXACT_Q8_ONNX_PRIMITIVE_NUMERICAL_ENVELOPE`,
not an XDNA compiler result and not yet a valid one-layer fidelity result.

## Candidate gate

No standard A16W8/A8W8/XINT8 candidate was promoted to a VC01 timeline test:
the only completed arithmetic comparison shows that removing the deployed
activation quantization already restores the S5-rejected divergence. A
full-layer, behaviorally tested implementation is required before treating any
standard QDQ graph as a candidate. BF16 remains a negative precision control.

```text
Q8_XDNA_CANDIDATE                  NOT YET ESTABLISHED
XDNA_PERCEPTION_BLOCKED_BY_ARITHMETIC  NOT YET ESTABLISHED
XDNA_PERCEPTION_REJECT             NOT YET ESTABLISHED
XDNA                               NOT AUTHORIZED
```

The next single experiment is a complete one-layer Q0 ONNX graph with the
same explicit Q8 activation conversion at every FFN/linear boundary, compared
against a captured ggml layer and then its VC01 downstream gate. Only a
behaviorally passing candidate may proceed to provider assignment or XDNA.
