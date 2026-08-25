# D2-S2: causal frontend and normalization decision

Status: **QUALIFIED — the actual VoiceChat PCM frontend is streamable; exact
downstream token fidelity is not yet universal and R9700 qualification is
still unavailable in this execution namespace.**

## Source correction

The earlier D2-S2 blocker was derived from the wrong preprocessor path. The
pinned VoiceChat projector is initialized in `tools/mtmd/mtmd.cpp` as:

```cpp
std::make_unique<mtmd_audio_preprocessor_parakeet>(ctx_a, false)
```

The `false` selects `norm_per_feature=false`: VoiceChat uses raw natural-log
mel features, with no full-utterance per-feature normalization. The
`norm_per_feature=true` implementation belongs to the conformer preprocessor
used by another projector type; it is not the production VoiceChat path.

That distinction changes the D2 decision. Full-session normalization remains
a genuine dependency for the conformer path, but it is not a blocker for the
authoritative VoiceChat frontend.

## Normalization policy bakeoff

The offline harness evaluated the requested causal alternatives on the same
raw log-mel frames and then drove the existing bounded VoiceChat encoder
prototype. `N4` was not run because no legitimate frozen calibration
statistics were available. Results below are diagnostic evidence, not a
request to change the VoiceChat model frontend:

| Policy | VC01 result | Multi-fixture observation | Decision |
| --- | --- | --- | --- |
| N0 full-utterance oracle | wrong response | impossible online | control only |
| N1 cumulative mean/variance | wrong response | causal but startup-sensitive | reject |
| N2 trailing 100/200/400/800 frames | VC01 pass | later fixtures varied in wording; no exact contract | counterfactual |
| N3 EWMA .01/.05/.10/.20 | .05/.10 VC01 pass | .01/.20 changed response | counterfactual |
| N4 frozen/global stats | not run | no uncontaminated calibration stats | unavailable |
| N5 50/100/200-frame freeze | wrong response | calibration choice changes behavior | reject |
| N6 no normalization | VC01 pass | authoritative VoiceChat contract; PCM/frontend parity passes all controls | select |

The selected D2 normalization contract is therefore:

```text
raw natural-log mel
norm_per_feature = false
no normalization state
```

The bakeoff remains useful as evidence that inventing a causal normalizer
would be a model-behavior change, not a harmless streaming implementation
detail.

## Bounded causal frontend result

Runtime checkpoint:

```text
boxwrench/llama-voicechat.cpp
6da91b8c6e5035110721dd3319f0511376d7487c
```

The research-only frontend now delivers the same PCM in bounded chunks while
retaining only:

```text
previous raw sample for causal pre-emphasis
bounded pre-emphasized waveform frontier
centered STFT frame cursor
352-sample inter-frame overlap / end-padding state
```

It reproduces the authoritative Parakeet framing exactly:

```text
sample rate 16 kHz
FFT 512
window 400
hop 160
center zero padding 256
natural log
128 mel bins
```

Chunk sizes 1280, 257, 4096, and 333 samples all produced:

```text
mel first_bad_frame = -1
mel minimum cosine   = 1.000000000
mel RMSE / max abs   = 0 / 0
```

This also passed the long-silence and abrupt-speech-onset controls.

The causal subsampling mapping is preserved without neighboring-row repair:

```text
stage 1: output i uses mel       [2i-2, 2i]
stage 2: output j uses stage-1   [2j-2, 2j]
stage 3: output k uses stage-2   [2k-2, 2k]
         raw mel frontier        [8k-14, 8k]
```

The current implementation uses a 33-row aligned bounded graph probe for
the three subsampling stages. It is deliberately not presented as the final
minimal stage-cache implementation or production static shape. Its
pre-encoder output passed the cosine/first-bad-frame gate on VC01–VC06 and
the onset/silence controls. VC01–VC04 and VC06 were numerically exact; VC05
has a tail envelope of RMSE 0.0317651 and max absolute difference 0.101608
while retaining cosine 1.0 and the exact downstream token/function trace. The
probe consumes the streaming PCM frontend;
the full-prefix tensors are retained only for parity diagnostics.

## Downstream fidelity

The bounded encoder stage retains the D2-S1 70-frame attention history and
8-frame causal-convolution history per layer. State reaches the existing
14,548,992-byte plateau. With the streaming PCM frontend:

```text
VC01 / VC04 / VC05 / VC06: exact token/function traces
VC02 / VC03: same semantic response class, but not exact token rows
```

VC02 and VC03 reproduce the bounded-encoder control's response, while their
full-prefix production controls differ after small accumulated embedding
drift crosses a sampler boundary. The drift envelope remains the previously
measured D2-S1 range:

```text
VC02  min cosine 0.999813020, max RMSE 0.000977375, max abs 0.00376237
VC03  min cosine 0.999477744, max RMSE 0.0014307, max abs 0.00563987
```

This is a downstream fidelity qualification, not an exact-token PASS. The
candidate is semantically viable, but D2 is not yet a production-ready
contract until the drift/trace policy is explicitly accepted or the bounded
encoder implementation closes that gap.

## Bounded state and timing

The logical state is:

```text
frontend waveform frontier: bounded by one centered STFT window and phase
subsampling: bounded causal boundary rows and independent stride phases
encoder: 14,548,992-byte plateau at 70 attention frames
```

The current CPU state-step service samples are approximately flat with
session length after the attention cap. In the VC05 decomposition, steady
state was compute-dominated:

```text
graph build       ~0.4 ms
allocation        ~0.2 ms
input staging     ~0.4–0.8 ms
compute            ~13–15 ms
output/state copy  ~0.8–1.1 ms
total              <=16.9 ms in the captured 101-frame run
```

The earlier approximately 200 ms VC05 event did not recur under decomposition
and remains `UNKNOWN`, attributed provisionally to host scheduling or a
measurement artifact rather than an algorithmic frontend component. R9700
GPU service curves and deadline behavior remain unmeasured because this
namespace lacks `/dev/kfd` and `/dev/dri`.

## Algebraic escape falsification

For the separate conformer path with true global affine normalization,

```text
y = (z - mean) / std
```

the affine operation could be folded through the first linear operation only
if `mean` and `std` were fixed. Once those statistics change with future
frames, the historical pre-activation changes; SiLU, attention, convolution,
and residual paths then require historical activations to correct the result.
No compact exact correction was found. The lead is killed for that path:

```text
Scope: NOT_DEC / genuine dependency (conformer norm path)
Recommendation: SUPPRESS
Reason: De == full utterance for the current normalized frame;
        no removable candidate-domain gap exists.
VoiceChat relevance: not applicable; VoiceChat passes false and uses N6.
```

The broader repeated-history replacement remains `DEC_EXTENDED_STATE`.

## D2-S2 decision

```text
VoiceChat normalization contract:      N6 / raw log-mel
PCM → mel parity:                       PASS
chunk-boundary invariance:              PASS
causal preencoder mapping:              PASS on bounded probe
complete PCM → embedding path:          QUALIFIED
downstream exact-token fidelity:        QUALIFIED, not universal
R9700 production qualification:         BLOCKED by device access
```

This is a real bounded frontend result, not an XDNA/compiler failure. The
next D2 decision is whether to close the small bounded-encoder numerical
drift before calling the contract production-ready; no new normalization
project is justified for the actual VoiceChat path.
