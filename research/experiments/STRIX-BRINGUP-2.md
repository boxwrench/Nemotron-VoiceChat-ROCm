# STRIX-BRINGUP-2

Date: 2026-08-25

Scope: establish real Linux/XDNA execution on this Strix Halo and make the
shortest credible route to a VoiceChat perception graph concrete. No VoiceChat
runtime, model, driver, or system-package changes were made.

## Decision summary

```text
XDNA-LINUX-M0                         PASS
XDNA-SPEECH-LINUX-M0                  PASS
accelerator access                    A HOST_OK_SANDBOX_HIDDEN — CONFIRMED
known NPU workload                    PROVEN on host
known speech workload                PROVEN on host
AMD Parakeet -> VoiceChat reuse       MEDIUM
LEAD-GFX1151-0001 relation            RELATED_PATTERN
Linux custom graph route              D, with C as a later artifact test
static Conformer candidate             NOT YET JUSTIFIED
FastConformer on XDNA                 QUALIFY — graph compatibility remains open
next intervention                     after M4A-2, one bounded graph-compatibility probe
```

Project conclusions:

```text
Linux XDNA hardware/runtime            PROVEN
Linux XDNA LLM execution               PROVEN
Linux XDNA speech execution            PROVEN
FastConformer on XDNA                  QUALIFY — graph compatibility remains open
```

These are host-shell results, not coding-agent namespace results. No VoiceChat
runtime, model, driver, or system-package changes were made.

The `A` classification is now **confirmed**. The previous probe's PCI/sysfs and
loaded-driver evidence showed that the host exposes the Radeon 8060S/amdgpu
and NPU/amdxdna devices to sysfs, while the coding-agent shell has a synthetic
`/dev` that hides the required character nodes. The ordinary host-shell probe
found the device nodes and a working XRT/FLM stack. The earlier failures are
therefore execution-namespace isolation, not host accelerator failure.

## 1. Accelerator boundary

Observed from the agent shell:

```text
PCI: 0000:c5:00.0 Strix Halo Radeon Graphics [1002:1586], amdgpu
PCI: 0000:c6:00.1 Strix/Strix Halo NPU [1022:17f0], amdxdna
sysfs: /sys/class/drm/card0 and renderD128 present
sysfs: /sys/class/accel/accel0 present
/dev/kfd: absent
/dev/dri/render*: absent
/dev/accel/accel0: absent
```

The prior process/mount evidence shows the shell and PID 1 share the agent
mount namespace, whose `/dev` is a synthetic tmpfs containing only basic nodes.
This makes `HOST_OK_SANDBOX_HIDDEN` the correct working classification, not a
driver-failure diagnosis. The host probe is the reproducible handoff for an
ordinary shell; it does not install or reconfigure anything.

## 2. Known-good Linux workloads

The installed stack is:

```text
/dev/kfd                    present
/dev/dri/renderD128         present
/dev/accel/accel0           present
FLM v0.9.39
XRT 2.21.75
virtio-pci / amdxdna kernel interface: 6.17.0-35-generic
NPU firmware                1.1.2.65
XRT device                  RyzenAI-npu5
Linux Mint 22.3, Ryzen AI MAX+ 395 / Radeon 8060S
```

`flm list --filter installed --json` showed NPU2 packages including
`llama3.2:1b`, `llama3.2:3b`, and `gemma4-it:e2b`; the host executions used
the installed stack and did not require a driver or system-package change.

### Agent namespace control

The earlier coding-agent attempts remain useful as a control. They failed
because the synthetic agent `/dev` hid the accelerator nodes:

| Attempt | Result | XDNA proof |
| --- | --- | --- |
| `flm validate --json` | `ready=false`, `amd_device_found=false`, `devices=[]` | no device/context |
| `flm run llama3.2:1b -i /dev/stdin` | `No such device with index '0'` | no context |
| `flm run gemma4-it:e2b --asr 1 -i /dev/stdin` | `No such device with index '0'` | no context |
| `xrt-smi examine` | `0 devices found` | no context |

`xdna-top --json snapshot` saw iGPU sysfs telemetry but reported the NPU as
degraded with `accel_absent`, no device, and zero contexts. These results
classify the agent namespace, not the host. The exact command/output capture
is [here](strix-bringup-2/generated-agent-stack-probe.md).

### Host XDNA LLM execution — `XDNA-LINUX-M0: PASS`

The ordinary host-shell capture established:

```text
FLM                         0.9.39
XRT                         2.21.75
amdxdna                     6.17.0-35-generic
NPU firmware                1.1.2.65
XRT device                  RyzenAI-npu5
flm validate ready          true
amd_device_found            true
all_fw_ok                   true
kernel_ok                   true
memlock_ok                  true
```

`llama3.2:1b` completed two host executions successfully. `xdna-top` showed
active NPU execution for the FLM process, proving that the model created and
used an XDNA context. See the sanitized [host LLM evidence](strix-bringup-2/host/xnda-linux-m0.md).

### Host XDNA speech execution — `XDNA-SPEECH-LINUX-M0: PASS`

FastFlowLM standalone ASR also completed successfully:

```text
runtime                     FastFlowLM standalone ASR
model                       Whisper-V3-Turbo-NPU2
endpoint                    POST /v1/audio/transcriptions
fixture duration            3.920 s
transcript                  The capital of France is Paris.
runtime evidence            NPU Locked -> Transforming audio to text
                            -> NPU Lock Released
```

During the request, `xdna-top` showed the FLM process owning active XDNA
contexts with matched submissions/completions:

```text
ctx4 440/440    ctx1 200/200    ctx3 132/132
ctx2  32/32     ctx5  11/11
```

This is a speech-runtime proof, not a FastConformer compatibility proof.
See the sanitized [host speech evidence](strix-bringup-2/host/xnda-speech-linux-m0.md).

## 3. AMD Parakeet -> VoiceChat operator matrix

The AMD source remains pinned to
`0b65628f1caacf0fbe3fd2cb4ed6bae0437a4155`. AMD's implementation statically
prepares the Parakeet encoder, fuses depthwise Pad→Conv patterns, rewrites a
boolean attention-mask pattern, and reports an almost completely fused NPU
partition. [AMD README](https://github.com/amd/RyzenAI-SW/blob/0b65628f1caacf0fbe3fd2cb4ed6bae0437a4155/Demos/ASR/Parakeet-TDT/README.md),
[optimization notes](https://github.com/amd/RyzenAI-SW/blob/0b65628f1caacf0fbe3fd2cb4ed6bae0437a4155/Demos/ASR/Parakeet-TDT/OPTIMIZATION.md).

| Operation/pattern | AMD Parakeet representation | VoiceChat representation | AMD transform | Reuse/risk | Evidence |
| --- | --- | --- | --- | --- | --- |
| depthwise convolution | Two depthwise subsampling convolutions plus depthwise Conformer modules | `pre_conv_2` and `pre_conv_5` call `ggml_conv_2d_dw_direct`; each encoder layer has `enc_%d_conv_dw` via `ggml_ssm_conv` | Pad→Conv fusion covers a depthwise graph pattern | `CONV_2D_DW` is a **related structural lead**, not proven identical compiler input | VoiceChat `models/voicechat.cpp`; AMD `fuse_pad_to_conv_depthwise` |
| Pad→Conv | Static graph rewrite for depthwise pairs; 24 pairs reported by AMD | Causal subsampling uses left/right `ggml_pad_ext` before the two pre-encode depthwise convolutions; encoder depthwise uses pad+roll for left-only context | Directly relevant in shape, but VoiceChat padding is causal and must be preserved | **RELATED_PATTERN** until a graph export proves the exact pair | VoiceChat `pre_conv` code and AMD optimization notes |
| layer norm | Encoder normalization as represented by the exported Parakeet graph | `ggml_norm` plus learned weight/bias in FFN, attention, and `conv_norm`; converter records `conv_norm_type=layer_norm` | No direct AMD transform established | Likely representable; epsilon/layout/provider behavior remains a risk | VoiceChat `voicechat.cpp`, converter comments |
| unary ops | Parakeet graph contains activations and mask-related lowering; AMD's documented mask rewrite is boolean Slice/Where, not a generic unary fix | The 24 repeated fallback tensors have shape `[1024, 376, 1, 1]`; source maps the repeated per-layer operation to `ggml_silu` after `conv_norm` | No direct match to AMD's boolean-mask rewrite | **RELATED_PATTERN**, not same pattern; exact XDNA lowering is unknown | gfx1151 VC01 log; VoiceChat `enc_%d_conv_dw` block |
| activations | ReLU/subsampling and Conformer nonlinearities | ReLU in `pre_conv_0`, `pre_conv_3`, `pre_conv_6`; SiLU in each encoder convolution module and FFNs | AMD graph/compiler handles its own activation placement | Op support and fusion need an exported VoiceChat graph | VoiceChat `voicechat.cpp` and Parakeet source |
| causal convolution padding | AMD demo uses a fixed offline/static encoder shape | Subsampling pads 2 left/1 right on both axes; depthwise encoder conv pads kernel-1 left via pad+roll | No direct reuse without preserving causality | High risk for embedding equivalence | VoiceChat converter and graph comments |
| attention | Relative-position Transformer-XL style attention | `content_scores` + relative-position scores, 8 heads, 70-frame left window, no future context | AMD rewrites a boolean mask pattern for its graph | Same family, different mask/context semantics | VoiceChat `attn_mask`, `rel_positions`, `enc_%d_attn_probs` |
| attention masks | Static Parakeet mask input; AMD patches bool Slice/Where | F32 `attn_mask` is an input to every VoiceChat graph build | Boolean rewrite is not directly reusable | Potentially related compiler issue, but no same node/pattern evidence | AMD `patch_bool_slice_for_171`; VoiceChat graph |
| relative positional encoding | Parakeet relative position inputs/embedding | `pos_freqs`, `rel_positions`, `pos_emb`, `linear_pos`, `pos_bias_u/v` | No specific transform identified | Likely compiler-shape risk, not a demonstrated blocker | VoiceChat graph source |
| subsampling | Static 128-bin input and fixed encoded length | 128 mel bins, 8× subsampling, causal 17-frequency output before `Linear(4352,1024)` | Static-shape preparation is conceptually reusable | Shape must come from M4A-2, not the 15-second Parakeet demo | Converter and AMD preprocessing |
| reshape/transposes | ONNX preprocessing/export must satisfy VitisAI layout expectations | Explicit permute/contiguous/reshape before `pre_enc_out`, plus Q/K/V and relative-position transposes | Conceptually reusable only | Layout and memory-copy costs are unknown | Both graph sources |
| static dimensions | Fixed input frames and encoded output in AMD config | M4A-2 still controls growing-prefix/window shape and cadence | AMD static conversion is a strong prior-art pattern | Cannot freeze VoiceChat shape early | AMD `static_config.json`; M4A-2 boundary |
| output embedding | Encoder output feeds iGPU TDT decoder | `encoder_out` is projected to 4480 and consumed directly by Nemotron | Tensor handoff pattern is reusable; decoder is not | Learned embedding parity is the acceptance criterion | VoiceChat `encoder_out`/`projected` nodes |

### LEAD-GFX1151-0001 answer

Classification: **`RELATED_PATTERN`**.

The `CONV_2D_DW` observation is specifically tied to VoiceChat's two
pre-encode depthwise stride-2 nodes (`pre_conv_2` and `pre_conv_5`), exactly the
kind of depthwise Pad→Conv family AMD had to transform in Parakeet. That is the
strongest concrete connection.

The repeated `UNARY` fallback is also now localized to the per-layer SiLU
operation (`enc_%d_conv_dw` block, one `[1024,376]` tensor per layer; 24 total), but AMD's
documented unary-adjacent transform is a boolean attention-mask rewrite. It is
not evidence that AMD already solved this exact VoiceChat SiLU lowering.

This is enough to prioritize graph comparison, not enough to optimize ROCm or
claim XDNA compatibility.

## 4. Linux custom-graph route

| Capability | Result | Evidence / boundary |
| --- | --- | --- |
| Linux executes a known XDNA model | `YES — PROVEN` on the host | FLM `llama3.2:1b` completed twice; `xdna-top` showed active XDNA execution |
| Linux executes an arbitrary compiled XDNA graph | `UNKNOWN` | No local custom graph load was attempted; artifact portability is unproven |
| Linux compiles an arbitrary graph | `UNKNOWN` | AMD public Parakeet flow is Windows-oriented; no Linux compile was available |
| FLM/Lemonade hosts an arbitrary encoder graph | `UNKNOWN` | Installed FLM exposes model tags/ASR, not a demonstrated custom embedding-encoder API |

Selected route: **D — evaluate a small FastFlowLM/Lemonade upstream capability
addition**, with C (compile elsewhere, execute a portable artifact on Linux)
as the first artifact experiment if its contract is documented. The host proof
establishes Linux execution of known XDNA models, but arbitrary graph loading,
compilation, and custom encoder hosting remain open. ONNX Runtime + VitisAI
direct Linux use remains an unproven alternative, not the selected
implementation path.

This preserves the distinction between the Linux runtime proof, arbitrary
graph execution, graph compilation, and a serving API for custom encoders.

## 5. Static candidate

`NOT YET JUSTIFIED`.

The host can execute known XDNA workloads, but M4A-2 has not frozen the
production perception shape and no VoiceChat-compatible graph has been built.
The next static candidate must be a bounded compatibility test for that exact
production-shaped graph, with embedding parity and XDNA execution evidence.

## 6. TTS and synchronization

Kokoro remains parked at:

```text
QUALIFY — viable renderer architecture, not currently compelling enough
to displace native TTS
```

No new TTS sweep or integration was performed.

At the beginning of this batch, product `origin/main` remained at
`2f8b361bdacf521019c73543dffb06c222bc799f`. The local runtime checkout was
`38a76719e2b31a4dfc574bf750bb9ad44c434b81`, while the remote PC reference
branch `origin/amd/rocm` was `5cc03186ab7db2c61efce2c3f3ce9455c8a70318`
(`research: freeze R9700 Q8 VoiceChat baseline`). No M4A-2 production
perception SHA is published. The exact synchronization check must be repeated
at batch end; if either runtime advances, record its SHA and stop before any
VoiceChat graph work.

End-of-batch synchronization was unchanged: product `origin/main` remained
`2f8b361bdacf521019c73543dffb06c222bc799f`, and runtime `origin/amd/rocm`
remained `5cc03186ab7db2c61efce2c3f3ce9455c8a70318`.

## Next intervention

Wait for the PC to publish the frozen M4A-2 production perception contract.
Then run one bounded, offline/static VoiceChat graph-compatibility experiment
using that exact contract:

```text
production-shaped input tensor and cadence
        -> candidate XDNA2 FastConformer graph
        -> embedding tensor
```

Measure graph compilation/loading, XDNA partitioning, CPU fallbacks, invocation
latency, XDNA execution, and embedding parity against the reference path. Do
not substitute Whisper embeddings, connect the candidate to Nemotron, or alter
the production runtime.

## Evidence

- [agent-side command capture](strix-bringup-2/generated-agent-stack-probe.md)
- [host XDNA LLM evidence](strix-bringup-2/host/xnda-linux-m0.md)
- [host XDNA speech evidence](strix-bringup-2/host/xnda-speech-linux-m0.md)
- [host-side probe script](strix-accelerator-access/host_probe.sh)
- [Bringup-1 accelerator probe](strix-accelerator-access/HOST-PROBE.md)
- [M4A-2 boundary](strix-m4a2-boundary.md)
- [AMD Parakeet-TDT source](https://github.com/amd/RyzenAI-SW/tree/0b65628f1caacf0fbe3fd2cb4ed6bae0437a4155/Demos/ASR/Parakeet-TDT)
- [FastFlowLM Linux guide](https://github.com/ROCm/FastFlowLM/blob/main/docs/linux-getting-started.md)
