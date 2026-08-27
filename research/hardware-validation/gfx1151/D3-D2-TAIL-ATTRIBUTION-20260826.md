# D3 / D2 perception-tail attribution — gfx1151

Status: **ATTRIBUTED — no optimization applied**

Runtime measurement build: `722e22a0a` (`voicechat: instrument D3 perception stages`),
which is behaviorally based on `7a8ab71e8f5a5e981096f1ca83783dd47c92477e`.

Configuration:

```text
VC_FHEAD_GPU=1 for the primary and GPU-repeat runs
renderer=false
native TTS=false
VC01 fixed 37 consecutive 1,280-sample / 80 ms slices
same D2 bounded state and D3 causal timeline
```

All runs authorized exactly 37 contiguous ordinary timeline steps (0–36). The
GPU-head runs reported `function_head_gpu:true` for every frame and token 12,
matching the CPU function-token trace exactly.

## Measurement decomposition

The D3 step now records:

```text
PCM -> mel
causal preencoder preparation
preencoder graph build / allocation / input / compute / output
bounded encoder graph build / allocation / input / compute / output-state
host state-cache update
```

For the first instrumented GPU-head run, the two historic tail positions (21
and 30) were normal. Frame 25 was the only new steady miss (80.893 ms): its
perception time was 34.153 ms, including 20.518 ms in preencoder graph
compute. Its encoder compute was 9.802 ms.

Across its frames 2–37, normal/tail component behavior was:

| Component | Median | p95 | Maximum | Attribution |
| --- | ---: | ---: | ---: | --- |
| PCM -> mel | 0.168 ms | 0.171 ms | 0.187 ms | not a tail source |
| preencoder preparation | 0.007 ms | 0.008 ms | 0.009 ms | not a tail source |
| preencoder graph build | 0.260 ms | 0.309 ms | 0.399 ms | not a tail source |
| preencoder allocation | 0.380 ms | 0.487 ms | 0.987 ms | not a tail source |
| preencoder input/output | 0.056 ms | 0.078 ms | 0.084 ms | not a tail source |
| **preencoder graph compute** | **11.556 ms** | **20.026 ms** | **20.518 ms** | **tail source** |
| encoder graph build/allocation | 0.583 ms | 0.758 ms | 0.967 ms | not a tail source |
| encoder input | 0.293 ms | 0.408 ms | 0.412 ms | not a tail source |
| encoder compute | 9.764 ms | 10.169 ms | 10.327 ms | stable |
| encoder output/state update | 1.277 ms | 1.741 ms | 1.960 ms | stable |
| D3 state-cache update | 0.002 ms | 0.003 ms | 0.003 ms | not a tail source |

The graph-compute timer covers HIP graph execution plus its required scheduler
synchronization. It does not yet separate kernel execution from device wait;
therefore this evidence does not attribute the tail to one of those two
sub-mechanisms.

## Matched CPU/GPU/GPU/CPU repeat

The original frames 21 and 30 were not treated as a stable position-specific
phenomenon. A matched A/B/B/A repeat used the same VC01 input and settings,
changing only `VC_FHEAD_GPU`.

| Run | Head | Perception p50 / p95 / max | Main p50 / p95 | Total p50 / p95 / max | Misses |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU-A | CPU | 24.421 / 32.495 / 32.889 ms | 51.235 / 58.653 ms | 75.912 / 88.782 / 95.319 ms | 6 / 36 |
| GPU-A | gfx1151 | 24.547 / 38.460 / 49.219 ms | 47.464 / 48.404 ms | 72.066 / 86.025 / 96.816 ms | 5 / 36 |
| GPU-B | gfx1151 | 24.145 / 44.310 / 46.038 ms | 47.493 / 48.131 ms | 71.817 / 91.268 / 93.745 ms | 4 / 36 |
| CPU-B | CPU | 24.337 / 31.350 / 31.489 ms | 50.689 / 52.904 ms | 75.236 / 82.122 / 82.483 ms | 2 / 36 |

The two CPU runs have perception p95 31.350–32.495 ms. The two GPU-head runs
have perception p95 38.460–44.310 ms, with late tail positions varying by run
(GPU-A: 11/23/36; GPU-B: 6/26/27). In every such tail, the excess remains in
preencoder graph compute; encoder compute remains about 9–10 ms.

This is evidence of a **GPU-head-associated preencoder compute/synchronization
tail**, not a proof of a specific HIP kernel or OS scheduler cause. It rules
out graph construction, allocation, frontend state, bounded encoder state
movement, and main-token behavior as the direct source of the perception
excursions. A separate scheduler/queue telemetry experiment would be required
to split GPU execution from HIP synchronization or host scheduling.

## DEC qualification

`clip_image_batch_encode_with_preenc()` requests a preencoder tensor but calls
`clip_encode()`, which builds `clip_graph_voicechat::build()` and computes its
complete graph. That graph continues after `pre_enc_out` through the 24-layer
encoder and final projection. D3 then separately calls the bounded
`clip_voicechat_stream_step()` for the one newly authorized preencoder row.

Thus the current preencoder request executes a downstream 24-layer graph that
cannot affect the requested `pre_enc_out`. This is a qualified `C10` recursive
residual-over-enumeration candidate:

```text
current candidate domain:
  33-row causal preencoder graph + downstream full 24-layer projector graph

required effect domain:
  latest pre_enc_out[1024] only
  -> existing bounded one-frame encoder step
```

The measured preencoder graph-compute interval is 11.556 ms median and
20.026 ms p95 in the primary run. It includes both the genuine causal
preencoder and the unnecessary downstream graph, so no exact saving is claimed
yet. A preencoder-only graph with pre-encoder numerical parity is the cheap
falsifier and the only candidate presently plausibly capable of recovering the
needed 10–15 ms.

The existing causal Pad -> depthwise-Conv lead remains distinct: it concerns
whether the necessary preencoder itself still evaluates an overcomplete causal
domain. Its direct runtime count and savings are not yet measured. Fixed graph
build/allocation is a Wrenchwork/state-representation concern, but its
sub-millisecond contribution cannot explain these tails.

## Current decision

```text
D6_FHEAD_GPU_CORRECT_BUT_INSUFFICIENT
GPU function-head mechanism: VALIDATED on gfx1151
D2 perception tail: preencoder graph compute / synchronization interval
XDNA: ECONOMIC_HOLD
renderer: still disabled
```

No D2, D3, renderer, XDNA, or placement optimization was applied by this
attribution pass.
