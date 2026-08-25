# D2-CONTRACT-M1: bounded VoiceChat encoder state

Status: **encoder-state milestone PASS; production D2 BLOCKED**

This contract records the first real stateful VoiceChat perception result. It
is intentionally an import boundary, not a claim that the complete live
waveform frontend has already been replaced.

## Exact runtime

```text
repository: boxwrench/llama-voicechat.cpp
branch:     research/d2-stateful-perception
commit:     5e6761ad727c91aed868dcb983a17e41a264ff6f
parent:     83f1829e8e89306579ffb18c11c840460f62050e
```

The commit is a research-only opt-in path. Normal VoiceChat execution is
unchanged. `VC_D2_STATE_TEST=1` runs the full-prefix oracle comparison;
`VC_D2_STATEFUL_TIMELINE=1` additionally feeds the bounded-state embeddings
to the research timeline.

## Import boundary and state

The current import boundary is one exact pre-encoder output frame per
VoiceChat timeline step:

```text
cadence:       12.5 Hz, one frame per 80 ms
input:         F32 pre_enc_out[1024] for the newly available frame
output:        F32 projected[4480] for that frame
encoder:       all 24 VoiceChat Conformer layers
attention:     per-layer K/V history, oldest-to-newest, capped at 70 frames
convolution:   per-layer causal history, 8 frames × 1024 values
relative pos:  bounded cursor derived from retained history, no growing prefix
eviction:      drop oldest K/V frame before appending the new frame
memory:        constant after the 70-frame attention cap
```

At the cap, the measured state is 14,548,992 bytes:

```text
24 × 70 × 1024 × 2 × sizeof(float)   attention K/V
24 ×  8 × 1024     × sizeof(float)   convolution history
                                      = 14,548,992 bytes
```

Reset destroys or reinitializes the stream. State does not cross sessions.
The first bounded graph allocation/compilation is warmup; steady-state
measurements exclude it.

## Evidence

The stateful stack consumes the exact `pre_enc_out` frames produced by the
existing full-prefix graph and matches the full-prefix downstream behavior:

| Fixture | Frames | State bytes | Min cosine | Max RMSE | Max abs | Downstream trace |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| VC01-short | 38 | 8,257,536 | 0.999862075 | 0.000716533 | 0.00263504 | exact token/function rows |
| VC03-long | 281 | 14,548,992 | 0.999477744 | 0.0014307 | 0.00563987 | exact token/function rows |
| VC05-pause | 101 | 14,548,992 | 0.999720812 | 0.000905603 | 0.00348644 | exact token/function rows |

The saved VC01, VC03, and VC05 comparisons showed identical downstream
token/function traces. VC01 produced the same final text in both paths:

```text
</s><s>The capital of France is Paris.
```

Embedding drift is retained as a diagnostic, not used as the sole acceptance
gate.

CPU-only service samples from the p99-instrumented prototype:

```text
VC01: mean 11.678 ms, p95 13.627 ms, p99 13.719 ms
VC03: mean 15.735 ms, p95 18.821 ms, p99 19.930 ms
VC05: mean 57.153 ms, p95 131.981 ms, p99 200.366 ms
```

VC05's tail is currently **UNKNOWN**, not explained away as a host outlier.
The present comparison records whole-step timing but does not decompose the
slow step into graph build, allocation, compute, copies, synchronization, and
host scheduling. This accounting work is deferred by the earlier D2-S2
frontend semantics blocker. These are not the R9700 production curve and do
not establish 80 ms deadline behavior on the reference GPU.

## Deliberately open production fields

The prototype is fed the full-prefix graph's exact `pre_enc_out` oracle. The
following are not yet frozen and are required before Strix wakes:

```text
current per-feature normalization contract (the present contract is not
    streamable with exact future-independent output)
streaming waveform/pre-emphasis/STFT state
streaming mel frame alignment
causal subsampling boundary features and stride phase
final production static/dynamic graph shape
R9700 mean/p95/p99 service curve versus session length
80 ms deadline misses on the production path
```

This contract therefore proves bounded state for the encoder stack, while the
production D2 contract remains **BLOCKED** pending an explicit normalization
decision, exact frontend/subsampling parity, and reference-host timing.
