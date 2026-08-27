# D3 / D1 async-renderer qualification — gfx1151

Status: **BLOCKED — no optimization applied**

Classification: **D3_D1_RENDERER_CONTENTION_BLOCK**

## Scope

This is a bounded integration experiment from the promoted D6/C10 baseline:

```text
runtime base:  e9736d680b4f893b1a3b4fd06352d76a7e75aa2a
product base:  565abf45d5034f71b91e102ba22e52de709c4d65
DEC base:      5b97e0f4ac426561aaeb4768b59c2e83b8448ba7
hardware:      Ryzen AI MAX+ 395 / Radeon 8060S / gfx1151
input:         VC01, 37 consecutive 1,280-sample / 80 ms slices
preencoder:    C10 preencoder-only graph
function head: VC_FHEAD_GPU=1
```

No model, precision, tensor layout, device placement, D2 state, causal
authorization, preencoder extent, or renderer architecture was changed.
XDNA remains on `ECONOMIC_HOLD`; D4/D5 were not started.

## Lifecycle and correctness

The existing D1 path was exercised as:

```text
native TTS publication
  -> immutable snapshot
  -> worker-owned codec scheduler
  -> bounded PCM ring (640 ms configured capacity)
  -> default `aplay` ALSA sink
```

The host default ALSA sink accepted a one-second silent raw-PCM probe. The D3
renderer/TTS run emitted `playback_begin`, published on 36 frames, and recorded
a PCM-ring high-water mark of 3,528 samples. The renderer started once at
`live_start`; it never authorized a timeline step.

For the renderer-off, native-TTS-on, and renderer/ALSA-on runs alike:

```text
37 / 37 captured slices -> exactly 37 contiguous timeline steps
function-token trace: identical (12 on every frame)
D2 state hashes: exact match to renderer/TTS-off control
projected embedding hashes: exact match to renderer/TTS-off control
logical D2 state-byte trace: exact match
```

Thus the observed failure is not a D3 causality or semantic/state regression.

## Service evidence

Steady statistics exclude cold frame 0 (frames 1–36). The renderer/TTS-off
control was repeated because one run had two transient preencoder tails. Its
repeat p95 was healthy, but it retained one 83.231 ms outlier; therefore the
old zero-miss C10 result should not be treated as a universal distribution.

| Configuration | Perception p50 / p95 / max | Main p50 / p95 / max | Total p50 / p95 / max | Misses |
| --- | ---: | ---: | ---: | ---: |
| TTS off, renderer off — control A | 14.348 / 31.159 / 37.533 ms | 47.495 / 48.511 / 50.943 ms | 62.047 / 78.417 / 84.883 ms | 2 / 36 |
| TTS off, renderer off — control B | 14.479 / 18.216 / 35.700 ms | 47.505 / 48.436 / 51.038 ms | 62.098 / 66.363 / 83.231 ms | 1 / 36 |
| TTS on, renderer off | 14.151 / 17.142 / 37.080 ms | 79.626 / 87.176 / 115.377 ms | 93.846 / 102.864 / 152.457 ms | 36 / 36 |
| TTS on, renderer + real ALSA — repeat | 18.663 / 41.822 / 45.809 ms | 108.019 / 174.322 / 448.859 ms | 128.314 / 198.252 / 467.217 ms | 36 / 36 |

The first renderer-on run independently reproduced the direction of the
result: total p95 197.937 ms and 36/36 misses. Its telemetry harness omitted
the terminal `bye` because asynchronous `playback_begin` interleaved with a
command response; the repeat used a corrected evidence-only harness and
completed normally.

## Attribution boundary

The matched TTS-on, renderer-off control already has 36/36 misses. Therefore
the native TTS work remaining in the main D3 step is the first sufficient
deadline blocker. Enabling the existing async renderer then increases both
main-path and perception tails, but this experiment does **not** establish the
specific added mechanism: the present telemetry cannot separate worker CPU/GPU
contention, scheduler interference, or ALSA write blocking.

The worker did produce bounded PCM and the real sink began playback, but the
fixture has no turn-completion event. It therefore did not exercise natural
drain/settle or cancellation behavior. Those are open D1 gates, not failures
of the tested causal/state contract.

## Decision and handoff

```text
D3 causal invariant:              PASS
semantic/state parity:             PASS
real ALSA playback:                PASS
renderer produced bounded PCM:     PASS (3,528-sample high water)
renderer/TTS D3 deadline coexist:  BLOCKED
natural drain/cancel lifecycle:    NOT EXERCISED
```

Do not optimize from this result. The next work must first decide how native
TTS publication leaves the D3 main critical path; only then can a renderer
ON/OFF interference study isolate D1 worker cost. This report does not
authorize D4, D5, XDNA, or another DEC intervention.
