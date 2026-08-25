# STRIX-BRINGUP-1

Date: 2026-08-24

Scope: bounded Strix serving research only. No VoiceChat integration, model
changes, permanent runtime changes, FastConformer XDNA implementation, or
replacement-TTS integration was performed.

## Decisions

### 1. TTS renderer — `QUALIFY Kokoro`

Kokoro is `PROMISING` as an external renderer concept and the current CPU
stream adapter is `QUALIFY`. It is not yet a fair performance comparison with
native VoiceChat because native B1 repetitions use the cold full runtime while
Kokoro uses a warm reused model, and native public output is final-WAV only.

Native B1 recovered five repetitions per fixture while the renderer was
resident during `--say`:

| fixture | warm TTS step median | first decodable PCM median | sustained RTF median |
| --- | ---: | ---: | ---: |
| short | 2.615 s | 3.903 s | 1.53x |
| medium | 8.241 s | 9.566 s | 1.57x |
| long | 16.515 s | 17.853 s | 1.48x |

A supplemental scratch trace distinguished native lifecycle state: the first
non-silent frame appeared at 1.557 s from process start, while the public path
still exposed only the final WAV boundary at 3.949 s. This is not a streaming
PCM measurement.

Kokoro's synthetic 12.5 Hz stream simulation produced no modeled underrun for
the tested fixtures. The bounded-word policy produced first PCM in about
0.975–0.994 s in the latest run, with 12.72–14.11x synthesis RTF and 4–7
chunks for medium/long fixtures. Sentence and clause policies delayed the
first chunk to about 1.38–2.76 s in the same run. These are warm CPU simulator
measurements, not VoiceChat timings.

Cancellation is only partially qualified: queued PCM can be discarded and the
model can be reused without reload, but a synchronous in-flight CPU Kokoro call
cannot be preempted. Real playback and cancellation need the next isolated
renderer test. Evidence: [TTS-B1 report](strix-tts-bakeoff/generated/TTS-B1/REPORT.md).

### 2. Accelerator environment — `HOST-ACCESS-BLOCKED`

The agent shell cannot execute accelerator workloads. `/dev/kfd`, DRM render
nodes, and `/dev/accel/accel0` are absent from its synthetic `/dev` namespace;
`rocminfo`, `xrt-smi examine`, and `flm validate` therefore fail to see the
devices. This is not evidence that Strix lacks the accelerators: PCI/sysfs
show the Radeon 8060S with `amdgpu` and the NPU with `amdxdna`, and host group
database entries include the expected video/render groups. The missing device
nodes apply to the resumed automation shell, not the successful historical
inference runs.

Evidence: [host probe](strix-accelerator-access/HOST-PROBE.md) and
[probe script](strix-accelerator-access/probe.sh).

### 3. XDNA FastConformer feasibility — `QUALIFY`

AMD's pinned Parakeet-TDT source demonstrates compiler-facing work in the same
general territory: 128-bin speech features, static encoder shapes, depthwise
Pad → Conv fusion, attention-mask rewriting, and a nearly complete NPU
partition. VoiceChat is also a 24-layer, 1024-wide causal FastConformer family
encoder. That is a strong feasibility lead.

It is not yet a port: VoiceChat has causal downsampling, layer-norm
convolution normalization, causal depthwise context, chunked-limited attention,
and a learned embedding output consumed directly by Nemotron. The exact
operator correspondence, Linux compiler/load path, embedding parity, and
accelerator execution remain unproven. Evidence: [XDNA feasibility study](strix-xdna-parakeet-feasibility/README.md).

### 4. First implementation after M4A-2

Run one bounded control-versus-XDNA perception experiment using the exact
production-shaped runtime SHA and tensor contract frozen by M4A-2:

```text
A — frozen perception path on Strix control
B — one static XDNA graph candidate with the same learned embedding output
```

Measure embedding agreement, first embedding latency, p95 80 ms deadline
misses, queue/transfer overhead, CPU/gfx1151/XDNA activity, UMA contention,
and package power. Keep Nemotron's conversational timeline unchanged. Do not
substitute a Whisper transcript and do not integrate the graph into the
conversation runtime until this isolated result is positive.

## Strongest new findings

1. The accelerator blocker is execution-context visibility: host PCI/sysfs and
   installed amdgpu/amdxdna/XRT/FLM layers are present, while the agent shell's
   `/dev` is synthetic and hides the required device nodes.
2. AMD's official Parakeet preprocessing work is a concrete compiler-facing
   lead for the same broad Conformer/depthwise/attention territory as the
   VoiceChat perception graph. Similarity is not yet identity.
3. Kokoro's bounded-word policy starts earlier than sentence/clause flushing in
   the isolated 12.5 Hz simulation without modeled underruns, making it a real
   renderer candidate rather than only a batch TTS name.

## Largest remaining uncertainties

- Whether the exact VoiceChat FastConformer graph can be statically compiled,
  loaded, and kept on XDNA2 on Linux.
- Whether AMD's Pad/Conv and mask transformations address the actual
  `CONV_2D_DW`/`UNARY` fallbacks rather than merely related graph patterns.
- Whether Kokoro can provide production-grade incremental prosody, playback
  cancellation, and drain state without delaying or destabilizing conversation.
- Whether a heterogeneous placement improves the conversation rather than only
  increasing accelerator utilization.

## Can start immediately

```text
S3 source/prior-art study and serving role map
isolated Kokoro renderer contract/playback characterization
host-side device-access remediation by the machine owner
M4A-2 handoff preparation and exact-shape checklist
```

## Must wait for M4A-2

```text
FastConformer XDNA graph export/implementation
production-shaped gfx1151 perception profiling
first heterogeneous VoiceChat intervention
replacement-TTS integration into the conversation runtime
```

## Evidence index

- [TTS B0 README](strix-tts-bakeoff/README.md)
- [TTS B0 report](strix-tts-bakeoff/generated/REPORT.md)
- [TTS B1 report](strix-tts-bakeoff/generated/TTS-B1/REPORT.md)
- [TTS B1 environment](strix-tts-bakeoff/generated/TTS-B1/environment.json)
- [B1 harness](strix-tts-bakeoff/bench_tts_b1.py)
- [accelerator-access README](strix-accelerator-access/README.md)
- [M4A-2 boundary](strix-m4a2-boundary.md)
