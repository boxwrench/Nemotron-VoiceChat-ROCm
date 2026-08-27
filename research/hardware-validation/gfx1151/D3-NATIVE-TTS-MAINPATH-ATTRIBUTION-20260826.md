# D3 native-TTS main-path attribution — gfx1151

Status: **ATTRIBUTED — no optimization applied**

Classification: **D3_TTS_MAINPATH_ATTRIBUTED**

## Scope

This is the bounded follow-up to `D3_D1_RENDERER_CONTENTION_BLOCK`. It changes
only observability. It does not change TTS arithmetic, device placement,
precision, queue depths, renderer behavior, or the D3 causality contract.

```text
runtime base:        e9736d680b4f893b1a3b4fd06352d76a7e75aa2a
runtime timings:     025259170fd7d243a2009f3204396f6b49636e15
product base:        533a7cab1a19ad5a61559ec766e84ccca364fb82
DEC base:            fa8355c8c8cd554a0ac433b3b4c0ac4016d91495
hardware:            Ryzen AI MAX+ 395 / Radeon 8060S / gfx1151
input:               VC01; 37 consecutive 1,280-sample / 80 ms slices
configuration:       VC_D3_PREENCODER_ONLY=1, VC_FHEAD_GPU=1
renderer / ALSA:     off / off
```

The four-run matched order was `OFF -> ON -> ON -> OFF`. Statistics below
exclude cold frame 0 and therefore cover frames 1–36. The raw JSON events and
stderr logs are retained as local measurement artifacts under
`/tmp/d3-hw/tts-mainpath-attribution/`; they contain machine-local paths and
are intentionally not committed.

## Source-level synchronous path

For an authorized D3 slice, `vc_session::d3_step()` computes the D2 embedding
and calls `vc_session::step(embedding, ..., d3_feed_tts)`. With native TTS
enabled, the sampled text token enters the following additional synchronous
path before that `step()` can return:

```text
vc_session::step
  -> vc_session::tts_feed(text token)
       -> update tts_q / EOS-release decision
       -> voicechat_tts_step(tts, output token)
            -> depthsum(prev_code) + code embedding preparation
            -> frame_cond(token)                         [cond_cache read/build]
            -> backbone_step()
                 -> construct 28-layer Gemma-TTS graph
                 -> scheduler allocation / uploads
                 -> scheduler graph_compute() and required completion wait
                 -> read conditional / unconditional hidden states
            -> generate_codes()
                 -> prepare each active MoG code step
                 -> construct code-generation graph(s)
                 -> scheduler allocation / uploads
                 -> scheduler graph_compute() and required completion wait
                 -> read logits / residual outputs
                 -> CPU nucleus sampling and RVQ residual search
            -> append generated code to frames / frame_toks; update prev_code
       -> voicechat_tts_silence_cos() -> tts_quiet / tts_voiced lifecycle update
       -> [only when renderer is enabled] immutable snapshot publication
```

The runtime reports `voicechat-tts: device CPU` on this Strix run. That labels
the native generator scheduler/backend for these measurements; individual
`graph_compute()` intervals include the scheduler dispatch and the completion
wait required before its outputs are read. They are not claimed to be pure
kernel time.

## Ownership and dependency boundary

| Work / state | Owner | Required before the current D3 step returns? | Reason |
| --- | --- | ---: | --- |
| `tts_q`, EOS release, `tts_quiet`, `tts_wait`, `tts_voiced` | timeline thread | yes, today | controls the next emitted speech token and speech lifecycle |
| `prev_code`, TTS position, TTS K/V cache, `cond_cache`, RNG | native TTS generator | yes, today | sequential state for the next native TTS frame |
| generated `frames` / `frame_toks` | native TTS generator | yes, today | the snapshot is made from this post-step state |
| immutable `stream_frames` snapshot | renderer boundary | no codec dependency | published only *after* generation; renderer consumes a copy |
| codec / streaming ISTFT state and PCM ring | renderer worker | no | D1 worker-owned future-audio work |

The existing immutable snapshot is therefore a codec/rendering handoff, not a
native-generator handoff. The measured 32–33 ms generator work occurs before a
snapshot can exist and mutates sequential TTS state. Moving it off the D3
thread would require a new ordered native-TTS state ownership and publication
contract; it is not a safe movement across the existing D1 snapshot boundary.

## Matched service results

| Run | Perception p50 / p95 / max | Main p50 / p95 / max | Total p50 / p95 / max | Steady misses |
| --- | ---: | ---: | ---: | ---: |
| TTS OFF A | 14.548 / 23.424 / 26.175 ms | 47.516 / 48.229 / 50.433 ms | 62.302 / 70.843 / 73.779 ms | 0 / 36 |
| TTS ON A | 14.797 / 17.099 / 21.060 ms | 79.935 / 85.797 / 111.983 ms | 94.702 / 103.556 / 125.985 ms | 36 / 36 |
| TTS ON B | 14.678 / 24.061 / 39.044 ms | 81.502 / 85.801 / 90.471 ms | 96.529 / 105.342 / 121.250 ms | 36 / 36 |
| TTS OFF B | 14.227 / 20.116 / 33.021 ms | 47.499 / 48.106 / 50.406 ms | 61.986 / 66.853 / 80.495 ms | 1 / 36 |

Every run authorized exactly 37 contiguous timeline steps. For both matched
ON/OFF pairs, all timeline ids, function tokens, D2 state hashes, D2 state
bytes, and projected-embedding hashes matched exactly. The two ON runs also
produced the same native-TTS last-frame hash at every steady frame. Thus the
measured cost is not accompanied by a causal, main-model, or D2-state change.

## TTS stage attribution

All values are milliseconds, ON-run A / ON-run B as `p50 / p95`.

| Synchronous TTS stage | ON A | ON B | What it represents |
| --- | ---: | ---: | --- |
| frame preparation | 0.357 / 0.364 | 0.357 / 0.377 | depthsum, embedding and input preparation |
| text conditioning | 0.001 / 1.796 | 0.001 / 1.678 | 28/36 steady frames were cache hits; misses build conditioning |
| backbone graph build + alloc + upload | 0.704 / 0.973 | 0.714 / 0.984 | graph construction and scheduler preparation |
| backbone scheduler compute + wait | 16.139 / 19.854 | 16.150 / 21.179 | required generator backbone completion |
| backbone output readback | 0.002 / 0.003 | 0.002 / 0.003 | conditional/unconditional hidden states |
| code graph build + preparation + alloc + upload | 1.828 / 2.710 | 1.840 / 2.609 | MoG code-step CPU preparation and graph setup |
| code scheduler compute + wait | 3.670 / 4.899 | 3.774 / 5.912 | required logits/residual completion |
| code output readback | 0.007 / 0.009 | 0.007 / 0.009 | sampled-code inputs |
| CPU nucleus/RVQ search | 8.788 / 12.026 | 8.794 / 12.449 | CPU sampling and residual search |
| append native TTS state | 0.001 / 0.003 | 0.001 / 0.003 | `frames`, `frame_toks`, `prev_code` |
| **instrumented `voicechat_tts_step`** | **32.130 / 37.884** | **33.439 / 38.767** | synchronous native generator |
| queue + silence lifecycle + snapshot publication | 0.013 / 0.018 | 0.013 / 0.017 | snapshot cost is zero because renderer is off |

The internal stage sums were 32.087 / 37.844 ms (A) and 33.392 / 38.720 ms
(B), leaving only 0.044 / 0.050 ms and 0.045 / 0.050 ms respectively relative
to the outer `voicechat_tts_step` timer. The timer therefore closes the
native-generator step rather than leaving an unmeasured substage.

Matched main-path closure is also direct:

| Pair | Main ON–OFF delta p50 / p95 | TTS wrapper p50 / p95 | Residual p50 / p95 |
| --- | ---: | ---: | ---: |
| ON A − OFF A | 32.538 / 38.306 ms | 32.144 / 37.898 ms | 0.106 / 0.657 ms |
| ON B − OFF B | 33.515 / 38.914 ms | 33.450 / 38.778 ms | 0.025 / 1.386 ms |

## Result and bounded next candidate

```text
D3 native-TTS main-path increase:   attributed
primary contributor:                synchronous native TTS generator
largest measured substage:          backbone scheduler compute + required wait
secondary contributors:             native code scheduler work + CPU RVQ/sampling
renderer/ALSA contribution:         excluded by design
existing D1 snapshot boundary:      codec/rendering only, not generator-safe
```

Classification: **D3_TTS_MAINPATH_ATTRIBUTED**.

The strongest justified next candidate is an **ordered asynchronous native-TTS
generator-state handoff**: retain the same token order and all generator
lifecycle state, but make generated-frame ownership/publishing independent of
the microphone-authorized main step. That is a new architecture contract, not
an edit to the existing renderer snapshot queue. It is documented here only;
no asynchronous move, queue-depth change, renderer run, or optimization was
performed by this experiment.
