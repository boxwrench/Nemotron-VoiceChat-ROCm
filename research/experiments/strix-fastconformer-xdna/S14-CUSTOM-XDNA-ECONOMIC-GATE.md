# S14 — custom XDNA economic gate

## Decision

```text
CUSTOM_XDNA_NOT_CURRENTLY_JUSTIFIED
status: ECONOMIC_HOLD
```

This is an economic hold on a custom transaction/AIE campaign.  It is **not**
an XDNA hardware rejection, an arithmetic-fidelity rejection, or evidence
that a future fused implementation cannot work.

The currently established realization boundary is:

```text
existing generic VitisAI realization     REJECTED
existing DynamicDispatch realization     REJECTED
custom XDNA realization                  UNTESTED
```

The existing realizations were rejected for different, independently recorded
reasons:

```text
DD_ARITHMETIC_BLOCK
  Existing public DynamicDispatch transactions do not preserve
  Q8-ARITHMETIC-CONTRACT-M1.

DIRECT_DD_TOOLING_BLOCK
  The installed direct-DD binding failed before XRT execution.  This is
  secondary tooling evidence, not the reason the VoiceChat DD route was
  rejected.
```

## Question

Would removing current perception work from the Strix CPU/gfx1151 resource
materially improve the emerging duplex system enough to justify building a
custom XDNA implementation of the exact VoiceChat Q8 arithmetic?

The answer needs a matched duplex control.  That control does not yet exist.

## Evidence used

| Evidence | What it establishes | What it does not establish |
| --- | --- | --- |
| gfx1151 Q8 validation, VC01 | Historical whole-prefix perception was 27.8 ms mean, 28 ms p50, and 29 ms p95 across five samples. | A bounded-state D2 service curve, p99, or duplex contention. |
| gfx1151 Q8 validation, VC03 | The historical whole-prefix case reported 105 ms perception for 281 frames at a 22.48 s input, an 80 ms miss. | Per-step p99 or a current production-shaped result. |
| gfx1151 validation logs | The historical perception graph used a 126.09 MiB CPU compute buffer and reported unsupported ROCm `CONV_2D_DW` and repeated `UNARY`; peak sampled GPU busy was 91% during S2S. | Per-frame CPU/GPU attribution or contention with a live renderer. |
| D2 import contract | A research one-frame encoder boundary has a bounded 14,548,992-byte F32 state at 12.5 Hz. | A frozen production frontend/cache contract or matched gfx1151 D2 timing. |
| D1 research prototype | A worker-owned codec scheduler, bounded PCM ring, publish/drain/cancel metrics, and discard sink exist. | Real ALSA playback, duplex backlog, underruns, or interruption behavior under load. |
| Product roadmap | D1 and D2 remain next; D3 live timeline is blocked on them, with interruption later. | Any completed Strix duplex scheduler metric. |

Sources are the frozen gfx1151 validation package, `D2-IMPORT-CONTRACT.md`,
the D1 research runtime, and the current product roadmap.  The old gfx1151
perception result must not be relabeled as a D2 bounded-state control.

## Current whole-system accounting

### Perception service time and deadline evidence

```text
historical gfx1151 whole-prefix VC01
  p50                 28 ms
  p95                 29 ms
  p99                 not reported (n=5; do not infer)
  >80 ms misses       none observed in VC01

historical gfx1151 whole-prefix VC03
  reported perception 105 ms for 281 frames
  >80 ms event        observed

bounded-state D2 on gfx1151
  p50/p95/p99         not measured
  >80 ms misses       not measured
```

The historical CPU fallback and 91% sampled GPU busy make perception a valid
placement hypothesis.  They do not quantify how much current CPU or GPU time
is removable from an 80 ms live frame.

### Duplex, renderer, and interruption evidence

```text
current product mode              push-to-talk / record-then-submit
continuous timeline backlog       not measurable yet
mic backlog                       not measurable yet
renderer interaction              not qualified with ALSA or a live timeline
output underruns                  not measured in a live playback sink
interruption latency              not applicable before D5/barge-in
```

The D1 prototype's discard sink is useful scheduling preparation, but it is
not an audio-device or duplex qualification.  Therefore an isolated
perception improvement cannot yet be converted into a defensible
first-audible, timeline-lag, or interruption benefit.

### Idealized offload upper bound

For the old VC01 measurement only, an impossible perfect offload (zero NPU
latency, zero synchronization, and no replacement cost) could remove at most
the measured 28 ms p50 / 29 ms p95 perception service.  This is an upper
bound, not a forecast.  It is not transferable to D2 because D2 has a
different bounded-state contract and has not been measured on gfx1151.

The D2 logical state also sets an explicit economic question:

```text
state per step                         14,548,992 bytes = 13.875 MiB
cadence                                12.5 steps/s
one-direction logical state traffic    173.4375 MiB/s
host round-trip logical traffic         346.875 MiB/s
```

These are boundary sizes, not measured DMA/API costs.  A custom realization
could keep some state resident, but that is untested.  The state boundary and
the current lack of a duplex control mean an accelerator-only latency number
would not establish serving value.

## Why a custom transaction is not authorized now

1. The product has no measured continuous Strix control: D3 has not landed,
   so timeline backlog and interruption responsiveness have no baseline.
2. The D2 encoder import is research-only.  The production frontend/cache
   policy and matched gfx1151 bounded-state service curve are still open.
3. D1 has not been qualified with real ALSA playback alongside perception,
   so renderer/GPU contention and audio underruns are unknown.
4. Generic VitisAI and existing DD failures establish representation limits,
   not the maximum system value a custom implementation could recover.

Building a custom transaction now would choose a high-cost realization before
the system-level success criterion is measurable.

## Reopen condition

Resume custom XDNA investigation only after current D2 plus duplex
measurements establish that perception resource placement materially
contributes to frame deadline misses, timeline backlog, GPU/CPU contention,
interruption latency, or another whole-system serving metric.

## Required evidence to reopen custom XDNA

All of the following should be available before authorizing a custom XDNA
implementation campaign:

```text
1. frozen D2 production contract
   - exact runtime SHA
   - frontend/subsampling/cache semantics
   - state ownership and reset rules

2. matched gfx1151 bounded-state control
   - perception p50/p95/p99/max and >80 ms misses
   - CPU/GPU attribution, copies, synchronization, and fallback map

3. D1 real playback qualification
   - ALSA PCM sink, ring depth/backlog, underruns, drain, cancellation
   - same-GPU renderer/perception contention

4. D3-compatible duplex A/B harness
   - identical audio/model/timeline/renderer controls
   - current placement versus a measured perception-offload candidate
   - timeline lag, main-step deadline, first audio, and interruption metrics

5. explicit system benefit threshold
   - a material measured improvement in deadline reliability, critical-path
     latency, GPU/CPU headroom, power/thermal behavior, or another stated
     serving objective without unacceptable behavioral regression
```

The required future measurements are:

```text
D2 gfx1151 p50/p95/p99
>80 ms misses
CPU/GPU contention
ALSA/renderer interaction
timeline backlog
interruption latency
whole-system idealized perception-removal ceiling
```

Only then can a typed hardware-realization contract be emitted for the exact
`M=1, K=1024, N=4096` Q8 primitive.  **No such contract is emitted by S14.**

## Wrenchwork hand-off

This result is a hardware-realization decision record, not raw benchmark
truth.  It should enter Wrenchwork as:

```text
Decision: HOLD / CUSTOM_XDNA_NOT_CURRENTLY_JUSTIFIED
Subject: exact-Q8 VoiceChat perception realization on Strix XDNA2
Evidence class: hardware-realization economics
Positive evidence:
  XDNA host and speech execution proven; exact-Q8 graph and VC01 fidelity pass
Negative realization evidence:
  generic VitisAI CPU-only; existing DD arithmetic/tooling blocks
Missing decision evidence:
  matched D2 gfx1151 control and live duplex A/B economics
Revisit trigger: the five evidence gates above
```

No private raw traces, model weights, compiled blobs, or machine-local
identifiers are included here.

## Next action

Keep custom XDNA unimplemented.  The next useful work is the product-critical
D1/D2/D3 measurement path; then use the resulting Strix A/B control to decide
whether perception relocation earns a custom realization campaign.
