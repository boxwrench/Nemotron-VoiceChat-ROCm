# D2: production continuous perception state

Status: **QUALIFIED — D2-S1 bounded encoder state and D2-S2 chunked PCM
frontend pass their parity gates; exact downstream token fidelity is not yet
universal and R9700 qualification remains unavailable here**.

The D2-S1 prototype computes one new frame from finite retained state through
all 24 VoiceChat encoder layers and preserves downstream VoiceChat behavior on
the available fixtures. D2-S2 then completed a source-faithful chunked
waveform/pre-emphasis/STFT/mel path and exercised the causal subsampling
frontier. The authoritative VoiceChat projector uses Parakeet with
`norm_per_feature=false`; the earlier `norm_per_feature=true` blocker was the
separate conformer path, not VoiceChat.

## Frozen evidence inherited from M4

These questions are not being repeated:

```text
future lookahead                         negligible; zero-lookahead is viable
naive growing-prefix re-encode           misses 80 ms around 20–24 s
simple bounded historical window         no tested window preserved timing and fidelity
embedding cosine alone                   insufficient as a correctness gate
```

The D2 gate is downstream VoiceChat behavior, with embedding comparisons used
as diagnostics rather than as the acceptance criterion.

## Current runtime state boundary

Source inspected at the D1 research runtime:

```text
14676822b9b973070ee04d1d8ebf5ba11fff22b2
```

The product still consumes the reconstructed M3.1 runtime:

```text
09f4c7b4a414ae060a2325714e612dfd9c057811
```

The old production call chain is:

```text
mtmd_encode_chunk()
  -> mtmd_encode_chunk_impl()
  -> clip_image_batch_encode()
  -> clip_encode()
       ggml_backend_sched_reset()
       build a graph for the whole supplied audio prefix
       allocate the graph
       compute the graph
       copy all output embeddings
```

`clip_encode()` accepts serialized `state_in/state_out` only for GEN_WAV
generation models. The VoiceChat perception path does not populate or consume
those slots. `mtmd_context` has an output-embedding buffer and an audio
preprocessor, but no persistent VoiceChat encoder state.

The VoiceChat graph constructs a fresh variable-length attention mask and
relative-position input for every call. Its selected graph contract is:

```text
128-bin NeMo mel input
three causal stride-2 subsampling stages
24 encoder layers
1024 hidden width
8 heads / 128 head dimension
70-frame causal attention left context
kernel-9 causal depthwise convolution
LayerNorm convolution normalization
SiLU FFN activations
4480-wide projected embedding output
one learned embedding per 80 ms VoiceChat timeline frame
```

The audio preprocessor also constructs a fresh pre-emphasized/padded sample
buffer and mel output on each call. Its persistent cache contains FFT tables,
window data, and mel filters—not streaming waveform history. Exact stateful
maintenance must therefore define sample/frame alignment as well as encoder
state.

## D2-S1 result: bounded encoder state

Runtime research commit:

```text
83f1829e8e89306579ffb18c11c840460f62050e
```

The prototype is exposed only through `VC_D2_STATE_TEST=1`, with optional
`VC_D2_STATEFUL_TIMELINE=1` for a research-only downstream run. The normal
VoiceChat path is unchanged.

For each new 1024-wide pre-encoder frame it retains:

```text
24 × attention K history, at most 70 frames per layer
24 × attention V history, at most 70 frames per layer
24 × causal convolution histories, 8 frames × 1024 values per layer
relative-position cursor represented by the bounded current history
```

The K/V cache evicts the oldest frame before appending the new one. The
convolution history is fixed at eight frames. Measured state memory is:

```text
VC01  38 frames:  8,257,536 bytes
VC02  54 frames: 11,403,264 bytes
VC03 281 frames: 14,548,992 bytes
VC05 101 frames: 14,548,992 bytes
```

The 14,548,992-byte plateau is the 70-frame attention limit plus fixed
convolution state; it does not grow with session duration.

## Minimum sufficient state to investigate

The first state decomposition is:

| Region | New input dependency | Candidate retained state | Required proof |
| --- | --- | --- | --- |
| waveform → mel | new samples plus pre-emphasis/STFT overlap and frame phase | previous pre-emphasis sample, overlap samples, frame cursor | exact mel parity at chunk boundaries |
| causal subsampling | new mel columns plus causal boundary/context at each stride-2 stage | per-stage boundary feature maps and stride phase | exact `pre_enc_out` parity |
| encoder attention | current query plus prior 70 encoded positions per layer | per-layer K/V history and relative-position cursor | exact attention output parity |
| encoder convolution | current sequence plus kernel-9 left context | per-layer last 8 channel states at the convolution boundary | exact conv-module parity |
| FFN/residual/norm | current frame only after required layer inputs exist | no long history beyond layer boundary state | full-layer parity |
| projection | current encoder output | no temporal cache | projected embedding parity |

The first four rows are now split by evidence. The encoder attention,
convolution, residual, normalization, and projection rows are implemented and
validated in D2-S1. The waveform/mel and causal-subsampling rows remain the
production completion work.

Important consequences:

- caching only attention K/V is insufficient if causal convolution and
  subsampling boundaries are not also retained;
- retaining a raw hidden window without preserving relative-position and
  stride alignment can change the learned embedding;
- a cache may be bounded in time while still growing in memory with layers,
  channels, and per-layer K/V/context tensors;
- output selection must remain the production rule: the embedding aligned to
  the newly available 80 ms frame, not an arbitrary final-prefix row.

## D2-S1 parity and downstream fidelity

The bounded encoder stack was driven by the exact full-prefix `pre_enc_out`
frames and compared with the full-prefix 4480-wide embeddings. Diagnostic
floating-point drift accumulates slowly, but downstream behavior remained
exact in the saved token/function traces:

| Fixture | Frames | Minimum cosine | Max RMSE | Max abs | Downstream result |
| --- | ---: | ---: | ---: | ---: | --- |
| VC01 short | 38 | 0.999862075 | 0.000716533 | 0.00263504 | exact token/function trace |
| VC03 long | 281 | 0.999477744 | 0.0014307 | 0.00563987 | exact token/function trace |
| VC05 pause | 101 | 0.999720812 | 0.000905603 | 0.00348644 | exact token/function trace |

The VC03 trace contains 291 downstream dump rows and VC05 contains 159; both
matched their full-prefix controls exactly. VC01's final text was
`</s><s>The capital of France is Paris.` in both paths. These are stronger
acceptance evidence than embedding cosine alone.

On the available CPU build, the p99-instrumented samples were:

```text
VC01 mean 11.678 ms, p95 13.627 ms, p99 13.719 ms
VC03 mean 15.735 ms, p95 18.821 ms, p99 19.930 ms
VC05 mean 57.153 ms, p95 131.981 ms, p99 200.366 ms
```

VC05's tail is currently **UNKNOWN**, not explained away as a host outlier:
the present instrumentation records whole-step timing but does not decompose
the slow step into graph build, allocation, compute, copies, synchronization,
and host scheduling. This is not the R9700/GPU production curve and is not a
D3 readiness claim.

## D2-CONTRACT-M1: encoder-state import boundary

This contract is frozen only for the bounded encoder-stage result, with the
frontend boundary intentionally explicit:

exact llama-voicechat.cpp SHA: exact runtime research commit in D2-CONTRACT-M1
history strategy: bounded causal encoder state; no growing raw prefix
cached state: per-layer K/V history (≤70 frames) plus 8-frame conv history
memory growth: constant after the 70-frame attention limit
input cadence: one pre-encoder frame per 80 ms timeline step (12.5 Hz)
input tensor: F32 `pre_enc_out`, `[1024]` for the current frame
dynamic/static shape: graph is rebuilt per bounded history length in prototype;
                      production static-shape choice remains open
output rule: projected 4480-wide embedding for the newly supplied frame
reset/session: destroy or reinitialize the stream; no state crosses sessions
warmup: first bounded graph allocation/compilation is outside steady state
service curve: CPU-only evidence above; R9700/GPU curve not yet measured
deadline misses: not established on the production GPU path
fidelity: exact downstream token/function traces for VC01, VC03, and VC05
```

The missing fields for a production import are specifically the streaming
frontend state, causal subsampling boundary/stride state, final production
static-shape strategy, and R9700 service/deadline curve. Strix must not treat
the `[1024]` pre-encoder boundary as the final live audio API until those are
closed.

## D2-S2 decision

See [D2-S2-FRONTEND-BLOCKER.md](D2-S2-FRONTEND-BLOCKER.md) for the source
correction, normalization bakeoff, exact streaming frontend parity, and
downstream qualification.

```text
VoiceChat normalization:       raw log-mel / no normalization
PCM → mel parity:               PASS across chunk boundaries
causal preencoder mapping:     PASS on bounded graph probe
complete PCM→embedding path:  QUALIFIED
downstream exact-token gate:   QUALIFIED, not universal
R9700 service/deadline curve:  BLOCKED by unavailable device access
```

## Decision gate

```text
D2-S1 bounded encoder state: PASS
D2-S2 complete frontend:     QUALIFIED
D2 production contract:     QUALIFIED, pending drift policy and R9700 curve
XDNA impact:                 Strix may import the logical frontend/state shape;
                             production placement still waits for qualification
```

The remaining blocker is specifically the full-session contributor domain from
per-feature normalization. It is not missing ONNX, XDNA access, or an
unsupported operator. A causal/running normalization change would be a new
model-behavior experiment, not an implementation cleanup.

## Wake-up information for Strix

Strix needs the eventual D2 runtime handoff, not this hypothesis:

```text
exact runtime SHA
history/cache strategy and state tensors
input cadence and shapes
output-frame rule
reset/session/warmup behavior
fidelity requirement and suite
R9700 mean/p95/p99 service curve by session duration
deadline-miss behavior
```

Until those values are frozen, no static XDNA input shape or FastConformer
cache topology should be treated as production architecture.
