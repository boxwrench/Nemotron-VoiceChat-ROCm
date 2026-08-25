# AMD Parakeet transform reuse map

## Provenance warning

The requested pin is recorded exactly:

```text
repository: amd/RyzenAI-SW
requested pin: 0b65628f1caacf0fbe3fd2cb4ed6bae0437a4155
```

That commit does **not** contain `Demos/ASR/Parakeet-TDT` or the requested
`preprocess_for_npu.py`, `OPTIMIZATION.md`, `static_config.json`, or
`vai_ep_config.json` files. Its tree contains the older Whisper ASR demo.
The Parakeet files are present in the later public tree inspected at:

```text
reference tree: 43b2dabe4d1bf084d0421953b134707b8cb7275a
release: RAI 1.7.0 Release
path: Demos/ASR/Parakeet-TDT
```

The mapping below therefore separates the **requested historical pin** from
the **later Parakeet reference implementation**. No later AMD source is being
treated as if it were present at the requested pin.

## Reuse map

| AMD component/source | What it does | Reusable unchanged? | Reusable with adaptation | VoiceChat-specific replacement required | Not relevant / blocker |
|---|---|---:|---:|---|---|
| `preprocess_for_npu.py:fix_shapes` | freezes ONNX input/output dimensions and computes encoded length | no | yes | use `n_time=f(T)` from VoiceChat causal subsampling; do not copy Parakeet `ceil(T/8)` blindly | final serving shape waits for D2 |
| `fuse_pad_to_conv_depthwise` | folds standalone depthwise Pad into Conv pads; AMD reports 24 pairs | no | yes | preserve VoiceChat asymmetric causal pad and verify embedding parity | direct graph export still absent |
| `patch_bool_slice_for_171` | changes BOOL Slice/Where attention masking to numeric arithmetic | no | maybe | current VoiceChat graph supplies an external F32 mask and has no equivalent BOOL mask subgraph in ggml | do not apply without an ONNX graph showing that pattern |
| `models/static_config.json` | 15 s / 1498-frame Parakeet static shape, encoded length 188 | no | concept only | VoiceChat provisional shapes must use causal `f(T)`; 1498 is not a production contract | D2 controls deployment shape |
| `models/vai_ep_config.json` | VAIML EP, optimize level 3, `ops-blocklist=SiLUBf16` | no | candidate config | test whether SiLU blocklisting is needed for VoiceChat; do not assume it preserves the graph | Linux host EP availability unknown |
| `inference/transcriber.py` | CPU mel + NPU encoder + iGPU/CPU decoder orchestration | no | concept only | VoiceChat needs embedding tensor handoff, not transcript decoding | Whisper/Parakeet decoder is not VoiceChat perception |
| `inference/mel.py` | vectorized CPU mel extraction | no | concept only | match VoiceChat preemphasis, log, no normalization, 128-bin parameters | audio frontend parity must be measured |
| ONNX Runtime + VitisAI EP | static graph session creation and NPU compilation | no | likely boundary | determine whether Linux host has this provider and whether arbitrary custom graphs load | Codex lacks `onnx` and `onnxruntime` |
| XDNA/XRT device proof | host execution/context telemetry | yes as evidence method | no code reuse | use `xrt-smi`/`xdna-top` to prove the process and submissions | does not establish graph compatibility |

## Directly transferable hypotheses

1. Static shapes are a compiler requirement, but the VoiceChat static shape must
   be selected after D2.
2. The VoiceChat `CONV_2D_DW` fallback is a related depthwise Pad→Conv problem;
   AMD's folding transform is the first concrete candidate for the pre-encode
   and causal depthwise subgraphs.
3. The VoiceChat `UNARY ×24` observation is per-layer SiLU and is not the AMD
   attention-mask issue. The VAIML `ops-blocklist=SiLUBf16` setting is therefore
   an explicit test variable, not a solution claim.
4. AMD's report of one monolithic NPU encoder partition is a target property to
   measure, not an assumption about VoiceChat.

## Current reuse decision

```text
reuse unchanged                 none yet
reuse with adaptation           static-shape logic; depthwise Pad->Conv idea
VoiceChat-specific work         graph export, causal semantics, mask, SiLU, parity
not yet justified                Linux arbitrary-graph compile/load route
```
