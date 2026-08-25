# D2-CONTRACT-M1: bounded VoiceChat encoder state

Status: **encoder-state PASS; complete PCM/frontend path QUALIFIED**

This contract records the first real stateful VoiceChat perception result. It
is intentionally an import boundary, not a claim that the complete live
waveform frontend has already been replaced.

## Exact runtime

```text
repository: boxwrench/llama-voicechat.cpp
branch:     research/d2-stateful-perception
commit:     6da91b8c6e5035110721dd3319f0511376d7487c
parent:     5e6761ad727c91aed868dcb983a17e41a264ffb2
```

The commit is a research-only opt-in path. Normal VoiceChat execution is
unchanged. `VC_D2_STATE_TEST=1` runs the full-prefix oracle comparison;
`VC_D2_STATEFUL_TIMELINE=1` additionally feeds the bounded-state embeddings
to the research timeline.

## Import boundary and state

The research import boundary now includes the exact chunked PCM frontend and
one pre-encoder output frame per VoiceChat timeline step:

```text
cadence:       12.5 Hz, one frame per 80 ms
input:         chunked PCM; logical output boundary is F32 pre_enc_out[1024]
                for the newly available frame
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
streaming frontend. The full-prefix graph remains an oracle for diagnostics:

| Fixture | Frames | State bytes | Min cosine | Max RMSE | Max abs | Downstream trace |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| VC01-short | 38 | 8,257,536 | 0.999862075 | 0.000716533 | 0.00263504 | exact token/function rows |
| VC03-long | 281 | 14,548,992 | 0.999477744 | 0.0014307 | 0.00563987 | semantic; token rows diverge |
| VC05-pause | 101 | 14,548,992 | 0.999720812 | 0.000905603 | 0.00348644 | exact token/function rows |

VC01, VC04, VC05, and VC06 retain exact token/function traces. VC02 and VC03
retain semantic response behavior but their small bounded-encoder numerical
drift crosses sampler boundaries. VC01 produced the same final text in both
paths:

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

VC05's earlier tail is **UNKNOWN**, not explained away as a host outlier. The
new opt-in decomposition records approximately 0.4 ms graph build, 0.2 ms
allocation, 0.4–0.8 ms input staging, 13–15 ms compute, and 0.8–1.1 ms output
and state copies; the captured 101-frame run peaked at 16.9 ms. The earlier
approximately 200 ms event did not recur and remains provisionally attributed
to host scheduling or measurement artifact. These are not the R9700
production curve and do not establish 80 ms deadline behavior on the
reference GPU.

## Deliberately open production fields

The prototype now accepts chunked PCM. The following remain qualification
fields before a production import:

```text
VoiceChat normalization contract: raw log-mel, `norm_per_feature=false`
minimal persistent subsampling stage state versus the 33-row probe
final production static/dynamic graph shape
R9700 mean/p95/p99 service curve versus session length
80 ms deadline misses on the production path
```

This contract therefore proves bounded encoder state and exact PCM→mel and
pre-encoder parity in the research path. The production D2 contract remains
**QUALIFIED** pending the bounded-encoder drift/trace decision and
reference-host timing.
