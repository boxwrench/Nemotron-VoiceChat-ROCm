# M4B: incremental audio streaming (M4B-1) and codec graph reuse (M4B-2)

Frozen research result. See
[docs/M4-DUPLEX-DESIGN.md](../../../docs/M4-DUPLEX-DESIGN.md) "Incremental
audio events" and M4-0B for why this question exists: `codec_decode()` is
already causal and already chunks internally, and a ready-built streaming
ISTFT (`mtmd_audio_streaming_istft`) already exists unused. M4B-1 wires
those two pieces together and measures whether the result is a real-time
duplex-viable path. M4B-2 follows up on M4B-1's one negative finding
(synchronous per-call latency) and tests the cheapest available fix
(graph reuse) before escalating to anything more invasive.

## Question

M4B-1: can the existing causal codec plus the existing streaming ISTFT
produce playable PCM incrementally, correctly, with enough performance
headroom for the 80ms causal frame budget?

M4B-2: M4B-1 found aggregate throughput is fine but per-call latency
isn't -- is that latency dominated by the per-call `sched.begin()` graph
rebuild (fixable cheaply by building the graph once and reusing it), or by
something else?

## Classification

**M4B-1: streaming PCM correctness PASS, aggregate codec throughput
PASS, synchronous live integration FAIL.**
**M4B-2: graph-reuse lead KILL.**

```
incremental PCM correctness        PASS
aggregate codec throughput         PASS
synchronous live integration       FAIL
graph-reuse lead                   KILL
async / codec-kernel work          POST-v0.1
```

> The codec is throughput-capable and numerically valid for streaming.
> The blocker is synchronous execution latency/dispatch structure, not
> an inability to generate incremental audio.

Fork resolution taken from this result: **ship PTT v0.1 first, continue
duplex (async codec worker or node-count reduction) afterward.** This is
the last cheap test that was run before that decision; closing the gap
found here needs either a background/async decode thread or kernel-level
work to cut the codec graph's node count, both out of scope for a "quick
fix before v0.1" pass.

## M4B-1: streaming architecture and wiring

New functions in `voicechat-tts.h`/`.cpp` (runtime repo, uncommitted, see
"Reproducing" below):

- `voicechat_tts_stream_reset(tts, first)` -- (re)starts a streaming pass
  at frame `first`, resets the streaming ISTFT's overlap/window-sum
  state.
- `voicechat_tts_stream_step(tts, timing, max_frames)` -- decodes up to
  `max_frames` newly-appended RVQ code frames since the last call,
  through the **existing, unmodified** causal `codec_decode()` (the same
  lead-context run-up trick `voicechat_tts_write_wav()` already uses for
  mid-stream starts), then feeds each resulting spectrogram row through
  the **existing, previously-unused** `mtmd_audio_streaming_istft`.
  Returns the int16 PCM this call adds.
- `voicechat_tts_stream_flush(tts)` -- drains the ISTFT's remaining
  overlap-add tail at end of turn.
- `voicechat_tts_sample_rate(tts)` -- small accessor, needed to write a
  streaming-assembled wav for comparison.

Wired into `voicechat-cli.cpp` at the single existing `tts_feed()` call
site (called once per real 80ms frame): every `VC_TTS_STREAM_CHUNK`-th
call, gated by `VC_TTS_STREAM=1`, drives the streaming path instead of
waiting for end-of-turn. `vc_say()` drains remaining frames at turn end
and, if `VC_TTS_STREAM_OUT` is set, writes the streaming-assembled PCM to
its own wav file alongside the existing reference wav, for direct A/B
comparison. No TTS redesign -- pure call-site wiring around functions
that were already there or already unused.

## M4B-1: the one-frame-lag bug, found and fixed

The first wiring version had a permanent one-frame lag: each
`tts_feed()` call appended a new RVQ frame and *then* called
`voicechat_tts_stream_step()`, but the step's `n_new` window only ever
covered frames already fully settled as of the *previous* call, so the
very last frame appended in a turn was never streamed before end-of-turn
cleanup ran -- silently dropping the final ~80ms of every turn.

Found via a total-sample-count mismatch: streaming output 169344 samples
vs. reference 167586 (short by roughly one frame's worth of samples).
Fixed by draining all pending frames (looping `voicechat_tts_stream_step`
until it returns empty) before calling `voicechat_tts_stream_flush()` at
turn end. After the fix: 169350 (streaming) vs. 169344 (reference) -- a
6-sample (0.27ms) difference, which is expected and explained below
(trim/clamp asymmetry), not a residual instance of the same bug.

## M4B-1: waveform equivalence

Streaming-assembled PCM vs. `voicechat_tts_write_wav()`'s reference
decode, same deterministic `--say` sequence (fixed seed, temp=0), across
`VC_TTS_STREAM_CHUNK` in {1, 2, 4, 8}:

```
correlation:    0.9999999 - 1.0000000
RMSE:           0.08 - 0.37            (int16 scale; signal RMS ~694 -> SNR ~65dB)
max abs diff:   16 / 32767
length diff:    +6 samples (0.27ms), streaming output longer
```

The 6-sample length difference is a real, understood, cosmetic
divergence, not noise: `voicechat_tts_write_wav()` trims
`pad = (n_fft - hop) / 2 = 6` samples off **both ends** of its output;
the streaming flush only trims the equivalent at the start (via the
ISTFT's internal `padding_to_remove`) and drains the full tail, so the
streaming output is 6 samples longer at the end. Easily matched if
wanted, not attempted here since it's below the threshold of anything
that would matter perceptually or for the correctness gate.

A second, separate, deliberate divergence was found and documented (not
silently absorbed into the error budget): the streaming path has no hook
for `write_wav`'s post-IFFT `constrain_value_range` clamp
(`s = clamp(s, -win[n], win[n])` before the window multiply) --
`mtmd_audio_streaming_istft::process_frame()` doesn't expose an
equivalent step. Measured impact is folded into the max-abs-diff figure
above (16/32767) -- negligible, no correctness-gate failures attributable
to it specifically.

Zero-offset alignment (no drift) across the full clip. No missing or
duplicated chunks. No truncated final word -- silence-to-silence match
confirmed at both tails. Chunk-boundary check at chunk=8 (12 boundaries
across a 7.68s clip): every boundary-crossing sample delta sits well
inside that boundary's own local max delta -- no discontinuity spikes at
any boundary. Deterministic: two full runs of the same sequence produced
byte-identical wav files.

No audio playback was available in the environment these measurements
were taken in; verification was numerical/waveform (correlation, RMSE,
SNR, per-boundary delta vs. local signal variation) rather than literal
listening.

## M4B-1: chunk-size sweep and per-call latency

Raw per-call data: `m4b1_stream_timing_c{1,2,4,8}.csv` (columns: frame,
codec_us, istft_us, total_us, n_samples).

| chunk (frames/call) | calls | codec_us mean | codec_us p95 | istft_us mean | audio ms/call | realtime factor (mean) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 95 | 103,315 | 108,320 | 137 | 80 | 0.77x |
| 2 | 47 | 114,796 | 118,454 | ~140 | 160 | 1.39x |
| 4 | 23 | 132,553 | 132,992 | ~140 | 320 | 2.41x |
| 8 | 11 | 163,693 | 231,549 | ~140 | 640 | 3.91x (worst call 2.77x) |

ISTFT cost is negligible throughout (~0.14ms/call) at every chunk size.
First-code-to-first-PCM latency at chunk=1: 189ms (dominated by the same
fixed per-call cost on the very first call).

At chunk=8 -- matching `codec_decode()`'s own internal chunk=8 causal
granularity -- aggregate realtime factor is comfortably above 1 (3.9x
mean, 2.8x worst observed), but each individual call costs
**164-232ms: 2-3x an 80ms frame budget** when injected synchronously into
the same thread as the main frame loop. This is the core M4B-1 finding:
good aggregate throughput, bad worst-case per-call latency.

## M4B-1: real-frame-loop regression

One real `--audio VC01-short.wav` turn, GPU function head enabled
(`VC_FHEAD_GPU=1`, see [../m4p-3 in the M4P-3 report -- not a separate
committed package, see the design doc's M4 journey summary]), native
TTS, streaming codec at chunk=8, all synchronous/single-threaded as
wired:

```
baseline turn (no streaming):   1936ms
streaming turn (chunk=8):       2621ms   (+685ms)
```

Confirms the per-call cost isn't a microbenchmark artifact -- injecting
it synchronously directly and measurably elongates a real turn.

## M4B-2: phase decomposition of `codec_decode()`

Raw per-chunk data: `m4b2_codec_decompose_steadystate.csv` (columns:
start, len, begin_us, build_us, alloc_us, upload_us, gpu_us, readback_us,
end_us, n_splits, n_nodes). Steady-state chunks only (chunk=8, real
`--say` run, n=35 chunks after excluding the first-call warmup outlier).

| phase | mean | p50 | p95 | max |
| --- | --- | --- | --- | --- |
| begin (sched_reset + ctx init) | 1.9us | 2us | 2us | 3us |
| graph construction (build ops) | 22.4us | 22us | 25us | 26us |
| alloc_graph | 92.9us | 55us | 229us | 818us |
| upload | 5.3us | 6us | 7us | 7us |
| **graph_compute (GPU + sync)** | **75,012.9us** | **54,837us** | **95,976us** | **228,759us** |
| readback | 50.7us | 48us | 76us | 79us |
| sched.end (free) | 0.1us | 0us | 1us | 1us |

`begin + build + alloc + upload + readback + end` together sum to
**under 200us**, out of a ~55-96ms total call. Graph construction and
allocation are **not** the dominant cost -- under 0.3% of it. This
directly falsifies the graph-rebuild hypothesis before any reuse
prototype was built.

## M4B-2: root cause (measured, not inferred)

`ggml_backend_sched_get_n_splits()` = 2 (not fragmented across many
CPU/GPU boundaries). `ggml_graph_n_nodes()` = 593 for a single
steady-state chunk -- from the depthwise-conv-as-unrolled-sum-of-`kk`-
shifted-views pattern, across 3 upsample stages x 3 blocks each. The
~54-96ms lives entirely inside `ggml_backend_sched_graph_compute()`
itself: genuine GPU kernel dispatch/execution cost across ~593
fine-grained ops, confirmed to scale with input length at the same node
count (54ms at L0=8, 96ms at L0=16) -- real per-op cost, not allocation
overhead.

## M4B-2: graph reuse benchmark

Built and empirically tested, not just reasoned about:
`voicechat_tts_codec_reuse_bench(tts, n_iters)` (runtime repo,
uncommitted) builds the steady-state (L0=16) graph exactly once, then
loops `n_iters` iterations re-uploading only the input tensor and calling
`ggml_backend_sched_graph_compute()` directly -- no reset, no rebuild, no
realloc -- against `n_iters` iterations of the normal rebuild-every-call
path, same synthetic data, same process:

```
normal:  n=40  mean=99034.4us  p50=94529us  p95=94762us  max=279894us
reuse:   n=40  mean=94760.3us  p50=94748us  p95=94945us  max=94982us
```

Statistically indistinguishable -- reuse's p50 isn't even lower than
normal's. Confirms the decomposition's implication directly: reuse
recovers ~0ms.

Because the reuse prototype itself showed no timing benefit before
touching the production streaming path, wiring it into
`voicechat_tts_stream_step()` for a full waveform-correctness comparison
was not attempted -- would have been effort spent chasing a lead the
measurement had already killed. M4B-1's streaming path is unaffected by
M4B-2 and remains exactly as correct as the waveform-equivalence section
above describes.

## Retained / deferred leads

Not pursued in M4B, kept for the record:

- **Background/async codec worker.** The natural next step per the
  design doc's own M4C framing ("simultaneous input/output... requires
  threading where today there is none"). Not attempted here -- M4B-2 was
  explicitly scoped as "the cheap test before that decision," not the
  decision itself.
- **Reduction/fusion of the depthwise-conv graph node count.** The
  593-node-per-chunk structure comes from expanding each depthwise conv
  as a sum of `kk` shifted-view multiplies rather than a single fused
  depthwise-conv op. Replacing that pattern with a real fused op (if one
  exists in the ggml backend used) would cut node count directly and
  might reduce dispatch/execution cost without needing async at all --
  unexplored, noted for the record, not pursued in this pass.
- **External/replacement TTS.** Explicitly out of scope per the user's
  standing M4P-1 instruction; not evaluated in M4B either.
- **Multi-GPU.** Still blocked by the visibility crash found in M4P-1
  (segfault whenever more than one ROCm device is visible to the
  process, independent of device assignment) -- not debugged here,
  carried forward untouched.

## Reproducing

Requires this repo's frozen corpus/`--say` harness and a HIP/ROCm build
of the runtime fork with the M4B-1 + M4B-2 patches applied (not shipped,
not committed to the runtime repo -- see below).

**Runtime provenance**: `boxwrench/llama-voicechat.cpp`,
`feature/m4b-streaming-audio`, branched from `perf/m4-function-head-gpu`
@ `a05335bb3` (the committed, pushed, PROMOTE-classified GPU
function-head change), itself branched from `amd/rocm` @ `5cc03186a`.
The M4B-1 + M4B-2 patches themselves are **uncommitted** on
`feature/m4b-streaming-audio` -- 3 files changed
(`tools/voicechat/voicechat-cli.cpp`,
`tools/voicechat/voicechat-tts.cpp`, `tools/voicechat/voicechat-tts.h`,
+465/-4 combined). This branch is experimental/not-for-release and is
explicitly not merged into `amd/rocm`.

**Key new entry points** (full diffs live only in the uncommitted
branch; the essential shapes, for anyone rebuilding this from scratch):

```cpp
// voicechat-tts.h
struct vc_stream_timing { int64_t codec_us = 0; int64_t istft_us = 0; };

std::vector<int16_t> voicechat_tts_stream_step(voicechat_tts * tts,
    vc_stream_timing * timing = nullptr, int max_frames = 1);
void voicechat_tts_stream_reset(voicechat_tts * tts, int first);
std::vector<int16_t> voicechat_tts_stream_flush(voicechat_tts * tts);
int voicechat_tts_sample_rate(const voicechat_tts * tts);

// M4B-2 only, never wired into the real frame loop:
void voicechat_tts_codec_reuse_bench(voicechat_tts * tts, int n_iters);
```

`voicechat_tts_stream_step()` decodes `[max_frames]` newly-appended RVQ
code frames via the existing causal `codec_decode()` (using the same
`lead = min(idx0, 8)` causal run-up trick `voicechat_tts_write_wav()`
already uses), then feeds the resulting spectrogram rows one at a time
through `mtmd_audio_streaming_istft::process_frame()`
(`tools/mtmd/mtmd-audio.h`/`.cpp`, pre-existing, previously unused).

Gating environment variables on `voicechat-cli.cpp`'s `tts_feed()` /
`vc_say()` call sites: `VC_TTS_STREAM=1` enables the streaming path;
`VC_TTS_STREAM_CHUNK=<n>` (default 1) sets how many `tts_feed()` calls
accumulate between `voicechat_tts_stream_step()` calls;
`VC_TTS_STREAM_DUMP=<path>` appends one CSV line per streaming call
(frame,codec_us,istft_us,total_us,n_samples); `VC_TTS_STREAM_OUT=<path>`
writes the streaming-assembled PCM as a wav for comparison against the
reference decode. `VC_CODEC_REUSE_BENCH=<n_iters>` (requires `--tts`,
exits before any real turn) runs the M4B-2 reuse benchmark.
`VC_CODEC_DECOMPOSE=<path>` (in `codec_decode()` itself) appends one CSV
line per chunk with the phase breakdown used in the table above.

Raw data in this directory: `m4b1_stream_timing_c{1,2,4,8}.csv`,
`m4b2_codec_decompose_steadystate.csv`.

## Next

Per the fork resolution above, M4 is closed here. The retained leads
(async codec worker, depthwise-conv node-count reduction) are the entry
points for duplex work after v0.1 ships, not before.
