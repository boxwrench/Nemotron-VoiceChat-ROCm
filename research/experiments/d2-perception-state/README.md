# D2: production continuous perception state

Status: **BLOCKED — contract not frozen**.

This is an architecture boundary, not an XDNA or compiler failure. The current
VoiceChat runtime has no stateful production perception path to measure.

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

The current call chain is:

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

This is a hypothesis about a causal incremental decomposition, not a frozen
production design. The current graph's all-at-once implementation does not
prove that this state layout is sufficient; it identifies the state that a
minimal exactness experiment must expose.

Important consequences:

- caching only attention K/V is insufficient if causal convolution and
  subsampling boundaries are not also retained;
- retaining a raw hidden window without preserving relative-position and
  stride alignment can change the learned embedding;
- a cache may be bounded in time while still growing in memory with layers,
  channels, and per-layer K/V/context tensors;
- output selection must remain the production rule: the embedding aligned to
  the newly available 80 ms frame, not an arbitrary final-prefix row.

## Required D2 contract

No D2 contract SHA exists yet. The following fields remain unfilled until an
incremental implementation passes downstream fidelity and timing gates:

```text
exact llama-voicechat.cpp SHA
history strategy
cached tensor/state definition
memory growth behavior
input cadence
input tensor dimensions and dynamic/static dimensions
output-frame selection rule
reset/session semantics
warmup semantics
mean/p95/p99 service time versus session duration
80 ms deadline misses
downstream behavioral fidelity suite and results
```

## Decision gate

```text
D2: BLOCKED / NOT FROZEN
XDNA impact: none yet; Strix must wait for this contract
```

The blocker is the missing stateful runtime implementation and its validation
suite, not missing ONNX, missing XDNA access, or an unsupported operator.

The next single experiment is **D2-S1: exact one-frame state decomposition**:

1. add temporary instrumentation to the reference graph to capture mel,
   `pre_enc_out`, and per-layer boundary tensors for one next frame;
2. implement a scratch frame-step candidate that retains pre-emphasis/STFT,
   subsampling, per-layer attention K/V, and convolution context;
3. compare its 4480-wide embedding against the full-prefix graph at identical
   frame positions;
4. run the resulting embeddings through the existing VoiceChat timeline and
   downstream fidelity suite before measuring long-session service time.

This is deliberately one state-decomposition experiment, not another window
sweep. It must remain a temporary research change until exactness and
downstream behavior are established.

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
