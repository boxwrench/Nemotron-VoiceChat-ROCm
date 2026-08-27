# D3 integration gap map

Status: **ACTIVE — integration/qualification frontier**

This document reconciles the historical M4 plan with durable runtime research
before D3 changes the live path. It preserves the original investigations;
it does not rewrite an old result as if it had already shipped.

## Status vocabulary

| Term | Meaning here |
| --- | --- |
| RESEARCH QUALIFIED | A mechanism has a bounded implementation and evidence sufficient to use as a D3 integration input. |
| IMPLEMENTED | Code exists on a runtime research branch. |
| INTEGRATED | The normal runtime/client path invokes it. |
| HARDWARE VALIDATED | Its relevant end-to-end behavior was measured on the named hardware. |
| PRODUCT READY | Integrated, broadly qualified, and suitable for release. |

## Reconciled program state

| Item | Research | Implemented | Integrated | Hardware validated | Product ready | Remaining gate |
| --- | --- | --- | --- | --- | --- | --- |
| M3 persistent PTT | complete | yes | yes | R9700/gfx1151 turn path | yes for PTT | not live duplex |
| D1 async renderer | QUALIFIED | runtime `14676822b9b973070ee04d1d8ebf5ba11fff22b2` | no | no real ALSA/live-timeline qualification | no | real sink, contention, bounded lifecycle in D3 |
| D2 bounded encoder state | QUALIFIED | runtime `6da91b8c6e5035110721dd3319f0511376d7487c` | no | research fidelity/probe evidence only | no | persistent frontend/preencoder API and matched service curve |
| Strix XDNA perception | ECONOMIC_HOLD | probes only | no | XDNA host/LLM/speech, not VoiceChat perception | no | D2 + duplex placement economics |
| D3 live causal timeline | active | runtime research branch | research serve path only | gfx1151 causality PASS; renderer/TTS-off effect-boundary cut reaches 65.093 ms p95 with no VC01 steady-state misses | no | renderer/ALSA qualification, live capture client, broader workload qualification |

The stale README rows that called D1/D2 “NEXT” and D3 “blocked” were accurate
before the two research branches existed. D3 is **not architecturally
blocked** now. It remains a substantial integration phase because D1/D2 were
deliberately kept out of the production PTT path.

## D1: contribution and open gates

Existing D1 pieces:

```text
voicechat_tts_stream_reset / publish / step / flush
immutable published TTS-frame snapshots
renderer worker with a distinct codec scheduler boundary
bounded PCM ring
drain/settle accounting
cancel_pending_audio that discards unheard PCM without resetting model state
```

Open D1 acceptance gates:

```text
real ALSA PCM sink rather than the research discard sink
worker/backend ownership proved safe under simultaneous main TTS work
bounded backpressure measured with the main 80 ms loop
actual underrun, drain, settle, and cancellation latency on hardware
continuous-session lifecycle rather than one completed turn
```

D1 is a runtime-quality queue mechanism needing live wiring and
qualification, not a finished renderer product.

## D2: contribution and open gates

Existing D2 pieces:

```text
zero-future-lookahead contract
one-frame FastConformer encoder step
24 layers of bounded K/V history (70 frames) and convolution history (8)
14,548,992-byte bounded F32 logical encoder state
raw natural-log Parakeet mel/no-normalization frontend probe
derived causal preencoder frontier: raw mel [8k-14, 8k]
downstream stateful-encoder fidelity evidence
```

Open D2 acceptance gates:

```text
persistent microphone-PCM -> mel -> preencoder state API
minimal production stage-cache/phase representation (not the 33-row probe)
arbitrary capture-chunk-boundary parity and reset semantics in the live path
normal mtmd encode-path integration
R9700 and gfx1151 bounded-state p50/p95/p99/max, misses, and attribution
```

D2 is a qualified model/state contract, not yet a production microphone
frontend.

## D3 causality and ownership model

```text
capture source (client-owned)
  -> bounded input ring of timestamped 16 kHz F32 80 ms slices
  -> single timeline owner
       -> D2 perception state
       -> vc_session persistent main-model state
       -> TTS frame publication
  -> D1 renderer worker (own codec scheduler/backend boundary)
       -> bounded PCM ring
       -> PCM sink
```

| State | Owner | Cross-thread rule |
| --- | --- | --- |
| microphone slices and capture timestamps | capture source/input ring | capture never writes model state |
| D2 frontend, subsampling, encoder state | timeline owner | one authorized slice at a time |
| `vc_session` timeline (`t`, `prev`, `fprev`, LLM KV) | timeline owner | no renderer access |
| native TTS generation state | timeline owner | renderer receives immutable frame snapshots only |
| codec/ISTFT scheduler state | renderer worker | never share its scheduler/backend with the main loop |
| PCM queue/sink | renderer worker + sink | bounded queue; cancellation drops unheard PCM only |

The causality invariant is exact:

```text
captured slice N
    authorizes timeline step N

no captured slice
    no ordinary timeline step
```

Tool/system frames are the only explicit exception and remain outside the
first D3 slice.

## Interfaces to connect

| Boundary | Existing input/output | D3 work |
| --- | --- | --- |
| capture -> timeline | PTT currently sends a completed WAV in a `turn` JSON command | add a timestamped fixed-slice live-ingest protocol and bounded input queue |
| PCM -> D2 | D2 probe accepts a complete PCM buffer and emits a sequence | expose a persistent one-slice frontend/preencoder step with reset/session semantics |
| D2 -> main | `clip_voicechat_stream_step(pre_enc[1024]) -> projected[4480]` | call once for each authorized slice and pass that embedding to `vc_session::step` |
| main -> TTS | `vc_session::step` already calls `tts_feed` | retain it; publish snapshot after each TTS step |
| TTS -> renderer | D1 publish/worker/PCM ring exist | start once per live session and attach real PCM sink |
| renderer -> timeline | D1 has drain/cancel state | surface lifecycle only; do not implement D5 interruption policy yet |

## Queue, synchronization, and telemetry

The first D3 slice uses bounded queues and has an explicit policy rather than
silently retaining unbounded latency:

```text
input ring: capture timestamp + exactly one 80 ms slice
renderer queue: immutable TTS frame snapshots
PCM ring: rendered/unheard PCM
```

The timeline owner is the only thread that may call the D2 encoder, main
`llama_decode`, sampler, or native TTS generator. The renderer worker may only
consume published snapshots using its own codec scheduling state.

Every authorized frame must emit or record:

```text
capture timestamp
timeline/frame id
perception start/end
main start/end
TTS publish
renderer enqueue/start/end
PCM queue depth
timeline backlog
deadline miss (>80 ms)
```

## Smallest coherent D3 slice

`D3-0` is deliberately narrower than fluent conversation:

```text
always-running capture source
-> fixed 80 ms ingestion
-> one bounded D2 perception step per authorized slice
-> one persistent vc_session step
-> native TTS snapshot publication
-> D1 async renderer
-> real PCM sink
```

It must run on real hardware with a deterministic capture/replay control and
produce the telemetry above. It does **not** decide model turn-taking (D4),
barge-in (D5), replacement TTS, custom XDNA, or multi-GPU placement.

## What later work changed from the original D3 assumptions

```text
old assumption: perception required a growing-prefix or a risky short window
current: D2 supplies a bounded causal encoder-state mechanism; the remaining
         work is production frontend/state integration

old assumption: codec streaming was an unknown math problem
current: causal codec + streaming ISTFT are numerically valid; the issue is
         worker scheduling and live sink ownership

old assumption: the frame loop could simply run as fast as compute permits
current: capture availability is an authorization gate; compute ahead of the
         microphone is causally invalid

old assumption: XDNA placement might be required to make D3 viable
current: custom XDNA is an economic hold and is not on D3's critical path
```

## D3 entry decision

```text
D3 implementation: UNBLOCKED
D3 product acceptance: OPEN
```

The next result is an end-to-end D3-0 causal-timeline run, not D4/D5 work.

## D3-HW initial host attempt (2026-08-26)

Classification: **D3_RUNTIME_INTEGRATION_BLOCK**. This is not a causality,
perception, renderer, or XDNA result.

The D3 runtime at `b4692dee6b765b21419899137af291ed05bfdefb` built for
`gfx1151` against `/opt/rocm-7.2.2`; the ordinary host namespace exposed
`/dev/kfd` and `/dev/dri/renderD128`. The controlled VC01 80 ms replay did
not reach its `ready` event, so it authorized **zero** live timeline frames.
During startup, an unrelated host `llama-server` already held the GPU at
approximately 94% busy. The original harness also piped verbose stderr during
model load; this could block the JSON handshake when that pipe fills.

The next hardware attempt must run with an idle or explicitly coordinated GPU
and must first prove the `ready` JSON handshake. It must not reuse this run for
latency, renderer, or placement conclusions. The controlled replay harness is
`research/scripts/harness/run_d3_live.py`.

The subsequent handshake-recovery preflight found 0% instantaneous busy but
the same unrelated ROCm process still retained substantial gfx1151 memory.
No retry was launched; an idle/coordinate device-owner gate is required before
the `ready` probe.

## Coordinated-gfx1151 gate result (2026-08-26)

With gfx1151 at 0% busy and no KFD owner, Gate 1 reached `ready`; Gate 2 ran
`live_start` only and exited at `t: 0`; and Gate 3 sent exactly one VC01 80 ms
slice. That slice emitted exactly one `d3_frame` and exited at `t: 1`, so the
causality/authorization invariant passed on hardware. Its service time was
146.157 ms (perception 93.218 ms; main 52.939 ms), a true >80 ms miss. The
correct current classification is **D3_CORRECT_BUT_DEADLINE_UNSTABLE**. The
one-frame result deliberately stops before renderer/ALSA work and does not
attribute the miss or make a placement recommendation.

## Renderer-disabled service curve (2026-08-26)

Runtime `11b808ec73aa331f06e6ce357eaa18dcbb00b959` adds explicit research
controls used only for this measurement: `live_start` ran with `tts:false` and
`renderer:false`. The 37 consecutive VC01 slices therefore exercised the same
causal D2/main state path without native TTS publication, D1 codec work, or an
ALSA sink. Every input slice produced exactly one `d3_frame`; frame ids were
the contiguous range 0–36 and the session exited at `t: 37`.

| Metric | Frame 0 | Frames 1–36 |
| --- | ---: | ---: |
| perception | 77.389 ms | p50 24.909 ms; p95 32.423 ms; p99/max 33.329 ms |
| main | 52.251 ms | p50 52.264 ms; p95 62.932 ms; p99/max 68.616 ms |
| total | 129.640 ms | p50 78.206 ms; p95 88.590 ms; p99/max 93.704 ms |
| deadline misses | 1 / 1 | 8 / 36 (22.2%) |

The logical retained state grew from 988,064 to 8,078,240 bytes while the
encoder history filled; VC01 is only 37 frames, so it does not reach the known
14,548,992-byte encoder-state plateau. Perception did fall sharply after the
first cold frame and did not show session-length growth over this partial
occupancy range. The recorded runtime queue was zero because this controlled
driver serializes each request. Under a real 80 ms capture cadence, cumulative
service-minus-cadence debt is projected to peak at 62.964 ms and finish at
15.426 ms; the per-frame inputs and timings are preserved in
`research/hardware-validation/gfx1151/generated/D3-service-curve/renderer-off-vc01.csv`.

Classification: **D3_COMBINED_SERVICE_BLOCK**. Neither steady perception nor
the main step independently exceeded 80 ms, but their combined p95 did. This
does not yet reopen the custom-XDNA economic hold: renderer, ALSA, real capture
backlog, and full duplex scheduler interaction are still unmeasured.

The first D6 control used the existing `VC_FHEAD_GPU=1` option with the same
renderer/TTS-off 37-frame sequence. It preserved exact frame authorization but
logged `VC_FHEAD_GPU set but no GPU backend device found, staying on cpu`.
That is **D6_FHEAD_GPU_CONTROL_NOT_ACTIVATED**, not a GPU-head performance or
fidelity result. It must not be used to judge the GPU-head mechanism.

Runtime `7a8ab71e8f5a5e981096f1ca83783dd47c92477e` makes the narrow portable
discovery correction: try `IGPU` after `GPU`, matching ggml's generic backend
preference. One-slice CPU/GPU probes on gfx1151 then emitted the same function
token (12), while the GPU probe reported `function_head_gpu:true` and mirrored
the head to `ROCm0`. The one-slice probe is activation/parity proof only.

The subsequent renderer/TTS-off 37-frame curve kept exact function-token
parity (all 37 tokens were 12) and active GPU-head telemetry. It reduced main
p50/p95 from 52.315/62.932 ms to 47.419/48.296 ms and reduced misses from 8 to
3, but perception p95 rose to 47.174 ms and total p95 remained 94.329 ms.
Thus the correct D6 result is **D6_FHEAD_GPU_CORRECT_BUT_INSUFFICIENT**: main
placement is valid, but a new perception-tail mechanism must be attributed
before another intervention. Renderer and XDNA remain out of scope.

## D6 / C10 preencoder effect-boundary cut — promoted

Runtime instrumentation based on `722e22a0aae1dd5cff4fee8e5c81aaddc6086bc3`
now provides two graph extents for the same D3 preencoder request:

```text
control:  complete VoiceChat projector graph -> pre_enc_out
cut:      causal preencoder graph -> pre_enc_out
```

The cut changes graph extent only. It leaves the D2 one-frame bounded encoder,
its K/V and convolution state, the main timeline, GPU function head, precision,
and scheduling unchanged. The full graph remains the control path.

The excluded 24-layer encoder/projection cannot contribute to `pre_enc_out`:
the cut ends after the causal subsampler/linear preencoder output. The actual
state required by later frames is owned by the separate bounded D2 encoder step;
it is not updated by the generic full-prefix graph. This was verified rather
than assumed: in consecutive full-versus-cut runs, every `d3_state_hash` and
`d3_embedding_hash` matched exactly.

Eight-frame parity (including startup) established all of:

```text
pre_enc_out:     bitwise equal on 8 / 8 frames
timeline ids:    identical 0..7
function tokens: identical (12 on every frame)
D2 state hashes: identical
embedding hashes: identical
state bytes:     identical
```

The matched 37-slice renderer/TTS-off VC01 curve retained those exact timeline,
function-token, state, and embedding results on every frame. Excluding cold
frame 0:

| Metric | Full graph control | Preencoder-only cut | Change |
| --- | ---: | ---: | ---: |
| preencoder graph nodes | 1,925 | 42 | -1,883 (-97.8%) |
| preencoder compute p50 / p95 | 11.694 / 19.190 ms | 1.661 / 3.199 ms | -10.033 / -15.991 ms |
| perception p50 / p95 | 24.679 / 29.680 ms | 13.884 / 18.315 ms | -10.795 / -11.365 ms |
| main p50 / p95 | 47.501 / 48.055 ms | 47.448 / 48.261 ms | unchanged within run variation |
| total p50 / p95 | 72.104 / 79.026 ms | 61.510 / 65.093 ms | -10.594 / -13.933 ms |
| deadline misses | 1 / 36 | 0 / 36 | recovered |
| timeline backlog peak | 0 | 0 | unchanged |

The control's one steady miss was frame 4 (80.576 ms, 20.955 ms preencoder
compute). The cut's largest steady frame was 67.193 ms; its residual
preencoder-compute maximum was 6.222 ms. This is a direct effect-boundary
mechanism result, not a graph-build/allocation or placement claim.

Classification: **D6_DEC_PREENC_PROMOTE**. The D3 renderer/TTS-off causal
service curve now has useful margin for this fixed VC01 control. Renderer,
ALSA, live capture, and broader workload qualification remain separate gates;
this result does not authorize D4, XDNA, or a stacked optimization.
