# Strix XDNA Parakeet / FastConformer feasibility

Status: `QUALIFY` — strong structural similarity and a credible compiler
reuse lead, but no custom VoiceChat graph has been compiled or executed.

This is a source study only. It does not modify the VoiceChat runtime, export a
new model, or begin a FastConformer-on-XDNA implementation.

## Pinned AMD source

The inspected AMD `RyzenAI-SW` source is pinned to commit
`0b65628f1caacf0fbe3fd2cb4ed6bae0437a4155`.

Inspected files:

```text
Demos/ASR/Parakeet-TDT/README.md
Demos/ASR/Parakeet-TDT/OPTIMIZATION.md
Demos/ASR/Parakeet-TDT/preprocess_for_npu.py
Demos/ASR/Parakeet-TDT/inference/transcriber.py
Demos/ASR/Parakeet-TDT/inference/mel.py
Demos/ASR/Parakeet-TDT/models/static_config.json
Demos/ASR/Parakeet-TDT/models/vai_ep_config.json
```

The public demo describes CPU mel extraction, an XDNA/NPU Conformer encoder,
and an iGPU TDT decoder. It is a Windows/Ryzen AI Software demonstration, not
evidence that this VoiceChat graph already runs on Linux.

## A. Architecture comparison

| Property | AMD Parakeet-TDT demo | VoiceChat FastConformer | Consequence |
| --- | --- | --- | --- |
| Input features | 128-bin, 16 kHz log-mel; 10 ms hop and 25 ms window | 128-bin VoiceChat mel input; exact feature normalization is `NA` in the converter | Feature dimensions are a promising match, but preprocessing must be identical |
| Input shape | Static exported window; the public demo uses a fixed 15-second shape | M4A-1 selected zero-lookahead; M4A-2 still controls growing-prefix length and invocation shape | Static conversion is likely central; do not freeze the shape before M4A-2 |
| Subsampling | Conformer encoder with static encoded length | Causal VoiceChat subsampling factor 8; padded dimensions produce the VoiceChat-specific frequency shape | Same broad family, but padding and output shape must be reproduced |
| Convolution | AMD explicitly rewrites/fuses depthwise Pad → Conv patterns | Five pre-encode convolutions plus a depthwise convolution in each of 24 encoder blocks | The fallback lead may be related, but operator traces are still required |
| Normalization | AMD graph-specific ONNX/VitisAI handling | `conv_norm_type=layer_norm`; checkpoint names still say `batch_norm` | Parameter/configuration difference, not a safe graph equivalence assumption |
| Attention | AMD rewrites a boolean Slice/Where attention-mask pattern for VitisAI | `chunked_limited`, `att_context_size=[70, 0]`: causal attention with a 70-frame left window | Mask construction and causal semantics need a VoiceChat export, not a Parakeet copy |
| Depthwise context | AMD compiler-facing depthwise graph is transformed before NPU compilation | `conv_context_size=causal`; kernel-1 left context only | Causal padding is part of the embedding contract |
| Layers / width | Public demo is a Parakeet-TDT 0.6B encoder/decoder system; exact VoiceChat equivalence is not claimed | 24 layers, `d_model=1024`, 8 heads, FFN width 4096 | Width/layer similarity makes reuse plausible; learned embedding identity still matters |
| Output | Encoder representation feeds an iGPU TDT decoder for transcript tokens | Learned perception embedding feeds Nemotron directly | Whisper/Parakeet transcript output is not a drop-in replacement |

VoiceChat-specific parameters already established by the runtime converter are:

```text
causal_downsampling
conv_norm_type=layer_norm
conv_context_size=causal
att_context_style=chunked_limited, att_context_size=[70, 0]
normalize=NA
use_bias=false
```

The important conclusion is narrow: the two encoders occupy similar
Conformer/compiler territory, but VoiceChat requires its learned embedding
output and its causal/context semantics. A transcript-producing Parakeet or
Whisper path would be a different architecture.

## B. Relation to the gfx1151 fallback

The validated gfx1151 logs report unsupported perception operators including
`CONV_2D_DW` and repeated `UNARY` operations. AMD's Parakeet optimization notes
show two related compiler-facing interventions:

```text
24 depthwise Pad -> Conv pairs fused before compilation
boolean Slice/Where attention-mask pattern rewritten to avoid unsupported
  or unknown-type behavior
```

This is an encouraging convergence, not proof of a shared bug.

| Observed VoiceChat issue | AMD Parakeet evidence | Current interpretation |
| --- | --- | --- |
| `CONV_2D_DW` fallback | Pad → depthwise Conv graph patterns are fused in the AMD preprocessing script | Possibly adjacent to the same depthwise graph limitation; generic operator coverage and the exact VoiceChat padding pattern remain untraced |
| `UNARY` fallback | AMD rewrites a boolean attention-mask graph and reports a nearly single NPU partition | Could be an attention/mask or activation lowering issue, but the VoiceChat operation names and semantics are not yet mapped |
| CPU backend/scheduler buffer | AMD reports 99.8% of the Parakeet graph in the VitisAI partition after transforms | Suggests graph partition quality is a first-order question; it does not show the VoiceChat graph will partition the same way |

Required next evidence is an operator-level VoiceChat export/trace and a
static-shape compilation attempt on an accelerator-visible host. Do not
optimize the current fallback from this analogy.

## C. Reusable code and boundaries

| AMD component / script | What it does | Reusable unchanged? | Reusable conceptually? | VoiceChat adaptation | Linux blocker |
| --- | --- | --- | --- | --- | --- |
| `preprocess_for_npu.py` static-shape conversion | Makes the encoder input/output dimensions explicit and prepares a VitisAI graph | No | Yes | Freeze the exact M4A-2 production-shaped perception input and preserve learned embedding output | Public script is tied to AMD's Windows/Ryzen AI Software path |
| `fuse_pad_to_conv_depthwise` | Rewrites depthwise Pad → Conv pairs | Only if the exported VoiceChat graph has the same pattern | Yes | Match causal left padding and verify numerical equivalence | Compiler/tool availability on Linux is unproven |
| `patch_bool_slice_for_171` | Rewrites a boolean attention-mask pattern | No | Yes | Rebuild the VoiceChat causal/chunked mask with its own shapes and semantics | VitisAI EP/compiler version coupling |
| `vai_ep_config.json` | Selects VitisAI EP/provider and cache behavior | No | Partly | Produce a Linux-compatible provider/cache configuration if the runtime supports it | Official demo is Windows-oriented |
| ONNX Runtime/VitisAI wrapper | Places encoder on NPU and decoder on another engine | No | Yes | Expose embedding tensor and preserve 12.5 Hz timeline ownership | Custom graph loading boundary is not demonstrated |
| CPU mel handoff | Produces the 128-bin feature tensor before NPU invocation | Potentially | Yes | Must match VoiceChat mel windowing/normalization exactly | Transfer cost is small only if the shape/buffer contract is stable |
| Compiled cache artifacts | Avoids recompiling a known graph | Not proven | Yes | Cache key must include graph, shape, precision, compiler, and driver | Portability between Windows build and Linux execution is unknown |

## Linux bridge assessment

Keep these questions separate:

```text
Can Linux execute a compiled graph?       plausible in the XRT/amdxdna stack
Can Linux compile this graph?              not demonstrated here
Can FLM load a custom graph?               not demonstrated here
Can ORT/VitisAI load it on this Strix?     blocked by current device access
```

Evidence collected in this batch:

- FastFlowLM's Linux guide requires the `amdxdna` kernel path, XRT, NPU
  firmware, `/dev/accel/accel0`, and a successful XRT device check.
- The local shell has `FLM v0.9.39`, XRT `2.21.75`, and an `amdxdna` package,
  but `/dev/accel/accel0` is hidden by the executing shell's synthetic `/dev`
  namespace. `xrt-smi examine` reports no device and `flm validate` cannot
  validate an NPU.
- PCI and sysfs still expose the Radeon 8060S/amdgpu and Strix NPU/amdxdna
  devices, so this is a host-access blocker rather than evidence that the
  hardware or kernel stack is absent.
- The public AMD Parakeet demo is Windows-oriented and uses Ryzen AI
  Software/VitisAI EP/FlexML. Its precompiled artifacts have not been shown
  portable to the Linux FLM path.
- FastFlowLM's public Linux ASR path is useful prior art, but the public
  boundary is Whisper/ASR rather than a custom VoiceChat embedding encoder.

The current route decision is therefore:

```text
PATH B/C boundary:
  compiling elsewhere and executing on Linux is plausible but artifact
  portability is unproven; a small FLM/FlexML custom-audio-encoder capability
  may be required.

Current practical status:
  HOLD for implementation until accelerator access is restored and M4A-2
  freezes the production-shaped graph.
```

## Sources

- [Pinned AMD RyzenAI-SW source tree](https://github.com/amd/RyzenAI-SW/tree/0b65628f1caacf0fbe3fd2cb4ed6bae0437a4155/Demos/ASR/Parakeet-TDT)
- [AMD Parakeet-TDT README](https://github.com/amd/RyzenAI-SW/blob/0b65628f1caacf0fbe3fd2cb4ed6bae0437a4155/Demos/ASR/Parakeet-TDT/README.md)
- [AMD Parakeet-TDT optimization notes](https://github.com/amd/RyzenAI-SW/blob/0b65628f1caacf0fbe3fd2cb4ed6bae0437a4155/Demos/ASR/Parakeet-TDT/OPTIMIZATION.md)
- [AMD NPU preprocessing](https://github.com/amd/RyzenAI-SW/blob/0b65628f1caacf0fbe3fd2cb4ed6bae0437a4155/Demos/ASR/Parakeet-TDT/preprocess_for_npu.py)
- [FastFlowLM Linux guide](https://github.com/ROCm/FastFlowLM/blob/main/docs/linux-getting-started.md)
- [FastFlowLM CLI/ASR guide](https://github.com/ROCm/FastFlowLM/blob/main/docs/docs/instructions/cli.md)
