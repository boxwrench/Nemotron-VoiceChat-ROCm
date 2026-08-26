# S13 — DynamicDispatch feasibility

## Question

Can an existing public AMD DynamicDispatch (DD) operator/transaction execute
the VoiceChat layer-0 `1 × 1024 → 4096` deployed-Q8 MatMul class directly on
Strix XDNA2, without relying on generic ONNX-provider assignment?

This is a feasibility result only.  It does not change the proven VoiceChat
Q8 arithmetic contract and does not start custom AIE work.

## Inputs and evidence boundary

```text
host runtime          public isolated Ryzen AI 1.7.1 userspace
XRT                   2.21.75
NPU                   RyzenAI-npu5 / XDNA2
DynamicDispatch       installed ryzenai_dynamic_dispatch binding
public source audit   amd/DynamicDispatch b3051f03e20aab237cda3bbe4cd2081f76b72b06
```

The public source is used to inspect operator tables and public transaction
assets.  The installed 1.7.1 binding is used for the host control.  They are
not asserted to be the same source revision.

## Required VoiceChat arithmetic

`Q8-ARITHMETIC-CONTRACT-M1` requires:

```text
activation             F32 → independent signed Q8 per 32 values
integer rounding       deployed ggml rounding-to-nearest-even path
activation scale       F32 for quantization, then F32 → F16 → F32 for use
weights                original GGUF Q8_0 int8 values + F16 per-32 scales
dot                    signed I8 × signed I8 → I32 for each 32-value block
accumulation           block-scale-weighted F32 row accumulation
```

The F16 stored-scale step is behaviorally material; replacing it with the
unrounded F32 scale previously left a 7.8688e-3 primitive error.

## Existing DD operator inventory

| Operator / transaction family | Relevant supported shape | Arithmetic | Contract classification | Result |
|---|---|---|---|---|
| generic `MatMul` / `gemm_4x2_a16w8acc16` | raw `49×1024×4096`, mapped to kernel `64×1024×4096` | A16W8acc16, fixed QDQ/output formatting | CLOSE_BUT_DIFFERENT | usable only as route/shape control |
| generic `MatMul` / `gemm_4x2_a16w8acc16` | `1×1024×768`; no `1×1024×4096` entry | A16W8acc16 | INCOMPATIBLE | no exact M=1 target transaction |
| `MLADFMATMULA16W8` | no public table entry for `M=1,K=1024,N=4096` found | A16W8 MLADF packing | INCOMPATIBLE | neither target shape nor Q8_0 semantics established |
| `act_const_matvec_add` / `matvecadd` | small M=1 families exist | A16A16-style matvec/add families | INCOMPATIBLE | not a `1024→4096` Q8 MatMul |
| `MatMulNBits` helpers | packing helper/public ONNX integration found | low-bit packing, no matching direct transaction table found | INCOMPATIBLE | no direct Q8_0 `1×1024×4096` transaction established |

The public generic table maps the raw `49×1024×4096` entry to the padded
`64×1024×4096` transaction.  It does not accept a raw `1×1024×4096` request.
Consequently, a one-row use would need 48 padded zero input rows and the NPU
would compute 64 rows to retain one: `Dc=64`, `De=1`.  This is a *held
potential padded-row lead*, not DEC_CORE: no fidelity-compatible XDNA
transaction establishes that this is the candidate domain for VoiceChat.

## Direct route control

A reproducible host-only control was added:

```text
host_s13_direct_dd_a16w8_control.sh
run_s13_direct_dd_a16w8.py
```

It constructs the public four-operand A16W8 DD form at raw
`49×1024×4096`, uses all-zero inputs/weights/QDQ constants, and expects an
all-zero `49×4096` output.  The test uses the public
`mladf_4x2_gemm_a16w8_qdq.xclbin` only as a direct-route control.  It makes
no VoiceChat-fidelity claim.

Observed host behavior:

```text
metadata generation         PASS
DD transaction compile      PASS
DD state serialization      PASS
fresh FusionRuntime.load_state
                           SEGMENTATION FAULT
same-process compile→init   IndexError: unordered_map::at
XRT transaction execute     NOT REACHED
xdna-top context/activity   none observed
```

The crash occurs after compile at the binding's state-load interface.  Bypassing
state reload leaves the same installed binding unable to initialize the
compiled metadata in process.  This is a DynamicDispatch userspace/interface
failure; it is not an XDNA hardware failure and does not test VoiceChat Q8
arithmetic on NPU.

The host runs with packaged XRT libraries (`libxrt_coreutil.so.2`), not an
SDK-style `/opt/xilinx/xrt/setup.sh`; the latter path is absent on this host.
No system package or driver was changed.

## Decision

```text
existing generic VitisAI realization     REJECTED
existing DynamicDispatch realization     REJECTED
custom XDNA realization                  UNTESTED

VoiceChat DD finding                      DD_ARITHMETIC_BLOCK
direct-DD tooling finding                 DIRECT_DD_TOOLING_BLOCK
```

`DD_ARITHMETIC_BLOCK` is the primary rejection reason: available DD operators
do not preserve the deployed VoiceChat Q8 arithmetic. `DIRECT_DD_TOOLING_BLOCK`
is independent, secondary evidence: the installed binding failed before XRT
execution and is **not** the reason the VoiceChat DD route is rejected.

Custom XDNA remains untested. A separate whole-system economic gate must
justify a custom transaction/AIE campaign before any such implementation is
authorized.

## Next boundary

Do not continue generic ONNX rewrites or attempt to map VoiceChat Q8 onto
A16W8 merely because the padded dimensions fit.  Any later custom-path study
must start from `Q8-ARITHMETIC-CONTRACT-M1`, establish an executable and
version-matched DD/XRT interface, and separately decide whether the held
`64:1` padded-row observation is economical at the 80 ms cadence.
