# M4 design: native continuous duplex VoiceChat

Status: design only. No runtime or client code has changed as a result of
this document. It exists to scope M4 before any implementation branch is
opened.

Reviewed once; accepted directionally with three corrections applied
below (assistant-overlap vs. user-interruption are distinct phenomena,
a live-timeline causality invariant, and playback cancellation separated
from internal TTS/model-state cancellation). The M4-0 feasibility spike
(see bottom) has since run, from source reading alone, no experiment
needed: M4-0B (chunked audio decode) is RESOLVED -- not a blocker.
M4-0A (streaming perception encode) is BLOCKER CHARACTERIZED -- exact
online-equivalent embeddings are not achievable at all given this
encoder's bidirectional attention, so the open work was choosing among a
characterized set of tradeoffs (fixed lookahead / bounded sliding
context / causal encoder change / replay), not closing a gap with more
compute.

**M4A-1 has since resolved that choice empirically: PROMOTE
zero-lookahead.** Across an initial sweep and an adversarial follow-up
targeting speech onset, mid-word, pre/post-pause, silence, and noisy
regions, future-audio context produced no material change to perception
embeddings relative to natural frame-to-frame variation. See
`research/experiments/m4a-1-lookahead-spike/README.md` for the full
methodology and results; summary under M4-0A below. M4A-2 (measuring
whether the resulting naive growing-prefix re-encode stays inside the
80ms causal budget) is next, not yet started.

## Why this milestone, and why now

M3 (push-to-talk) proved the plumbing: process lifecycle, the `--serve`
JSON protocol, persistent multi-turn state, audio capture/playback, all
work on real AMD hardware (see `research/baselines/R9700-Q8-M1/` and
`app/push-to-talk/README.md`). But PTT's shape -- record a complete
utterance, submit it, wait for a complete reply, play it back -- is a
conventional voice-assistant interaction pasted onto a model that was not
built that way.

Nemotron VoiceChat's actual architecture is a single continuous timeline
running at a fixed frame rate, where `input[t] = embed_tokens(text_out[t-1])
+ perception(audio)[t]` (see `tools/voicechat/README.md` in the runtime
repo). Speech in and speech out are two channels of the same ongoing
process, not a request and a response. The reason this model was chosen
over a bolted-together VAD-STT-LLM-TTS pipeline was to get that behavior:
always listening, responding incrementally, and capable of being
interrupted mid-response because the model itself, not an external VAD,
decides when to speak.

M4's target, stated plainly: **a continuous, low-latency spoken
conversation -- always listening, speaking incrementally, naturally
taking turns, and interruptible while it talks.** PTT is now frozen as a
diagnostic/fallback interface (see its README's "Known limitations"); it
is not extended further as part of reaching this goal.

## What the runtime does today (read from source, not assumed)

All references are to `tools/voicechat/voicechat-cli.cpp` and
`voicechat-tts.cpp` in `boxwrench/llama-voicechat.cpp` at the pinned
commit in `runtime/README.md`, unless noted otherwise.

### 1. Perception encoding happens once per turn, on the whole clip

`vc_session::run_turn()` (`voicechat-cli.cpp:761-817`) loads an entire wav
file via `mtmd_helper_bitmap_init_from_file`, then encodes it in a single
`mtmd_encode_chunk` call. There is no per-frame streaming encode path in
mtmd/the FastConformer encoder today. **This is the largest structural
gap for continuous input** -- everything else about the frame loop
already operates one frame at a time; only the input encode is bulk.

### 2. The frame loop is real, but it is not paced to a clock

`run_turn()`'s `while (true)` loop (`:855-1002`) drives
`vc_session::step()` (`:639-718`, one `llama_decode` per call) once per
"frame." It runs as fast as compute allows; "12.5 Hz" describes what one
embedding step represents to the trained model (80ms of audio), not an
actual timer anywhere in this code.

State that already persists across turns (session-scoped): `t` (KV
position), `prev`/`fprev` (fed-back text/function tokens), the llama KV
cache itself, `tts_q`, and the TTS backbone's own internal state.

State that is currently local to one `run_turn()` call and would need to
become continuous: the audio cursor `ai` and the precomputed `aud`
buffer, `n_ext`/`n_pad_streak` quiet-detection counters, `rep_hist`
loop-detection history, tool-call freeze state (`in_call`/`inject`/
`want_eotr`), and `transcript`.

### 3. Text output already streams per frame -- no redesign needed here

`vc_events::text_delta()` (`:281-288`) emits one JSON line per non-pad
frame immediately, with `fflush(stdout)` on every line (`:271-279`). This
part of the pipeline is already incremental. M4 should not touch it.

### 4. TTS codes are generated incrementally; PCM audio is not

`voicechat_tts_step()` (`voicechat-tts.cpp:1176-1204`) runs the TTS
backbone and MoG sampler once per frame and appends the resulting RVQ
code vector to an append-only `frames` vector. That part is already
frame-by-frame. But the actual decode to audio -- ConvNeXt decoder plus
ISTFT, in `voicechat_tts_write_wav()` -- runs exactly once, at the end of
a turn (`voicechat-cli.cpp:1099`), after both the frame loop and a drain
loop have finished. **This is the second major gap**: codes exist
incrementally, nothing currently turns them into incremental audio.

### 5. `VC_NO_BARGE` / `VC_FORCE_BOS` are suppressors, not turn logic

`VC_NO_BARGE` (`:679-681`) rewrites a sampled `bos` (turn-open) token back
to `pad` for as long as `in_audio` is true -- i.e. it forcibly silences
the model's own decision to start talking early. `VC_FORCE_BOS`
(`:876`, `:682-685`) forces `bos` onto the first frame right after input
audio ends. Both exist specifically because, left alone, the model *will*
try to open its own turn while the user is still speaking, "usually about
a second in" (`tools/voicechat/README.md:453-465`). PTT sets both
unconditionally because whole-utterance record-then-submit needs strict
turn boundaries; M4 needs the opposite.

### 6. No barge-in signal is surfaced anywhere -- and the signal that does
exist is the wrong direction for the demo we actually want

Point 5 is proof the model already wants to interrupt on its own -- but
that signal (`bos` sampled while `in_audio` is true) is the *assistant*
electing to speak while the *user's* audio is still arriving:

```
USER is speaking
       |
model emits bos
       |
ASSISTANT wants to start speaking
```

That's **assistant overlap**, not what M4D's demo needs. The scenario we
actually care about is the reverse -- the user interrupting the
assistant:

```
ASSISTANT is speaking
       |
USER begins speaking
       |
model receives that audio
       |
what does the model do?
```

Nothing in the codebase today answers that question. There is no known
token/state transition for "new user speech arrives while the assistant
is mid-response" -- it could close the text channel, go pad-heavy, emit
a fresh bos/eos pair, change TTS behavior, or something not yet
anticipated. **Do not assume the bos-while-in_audio signal (assistant
overlap) is also the signal for user interruption** -- they are two
named, distinct phenomena, and the second one's behavior is unknown
until observed (see M4-0 below). There is a separate sotc/eotc/eotr
function-head channel, but it is scoped to tool-calls, not
user/assistant speech overlap; it is not a source for either signal.

### 7. `vc_serve()` is strictly one-shot request/response

`voicechat-cli.cpp:1130-1224`. A blocking `std::getline` reads one JSON
command; `"cmd":"turn"` calls `run_turn()` synchronously to completion and
emits exactly one `turn_end` before returning to `getline`. No
concurrency, no partial-turn boundaries, no way to keep both directions
open at once.

### 8. No ring buffers or backpressure exist in this code today

The only queue-like structure anywhere in `tools/voicechat/` is
`std::deque<llama_token> tts_q`, unbounded and with no backpressure. No
threads exist in this directory at all -- a search for `std::thread` /
`std::mutex` in `tools/voicechat/` returns nothing. A duplex redesign
needs real ring buffers and flow control built from scratch; `tts_q`'s
FIFO shape is the only existing precedent to build from.

## Design

### Continuous PCM input

Replace the whole-clip `mtmd_encode_chunk` call with an incremental path
that can encode fixed 80ms frames as they arrive from a ring buffer, with
no VAD deciding where a "turn" begins or ends -- turn-taking stays a
property of the model's own output (point 5/6), not an externally
imposed boundary.

**Resolved by M4-0A (real blocker, confirmed architecturally):** naive
80ms-chunked encoding does **not** reproduce whole-clip embeddings. The
FastConformer/parakeet encoder graph (`clip_graph_parakeet::build()`,
`tools/mtmd/models/parakeet.cpp:7-421`) has no cross-call state --
`mtmd_encode_chunk_impl()` (`tools/mtmd/mtmd.cpp:1738-1781`) runs
`clip_image_batch_encode()` fresh every call, and `mtmd_audio_cache`
(`mtmd-audio.h:28-51`) only holds fixed constants (mel filterbank,
window), not sequence state. Its default path is full bidirectional
self-attention over the entire submitted chunk (`parakeet.cpp:264-336`,
active whenever `n_time <= 8192` frames -- i.e. always, at conversational
scale), so a frame's embedding is a function of the whole clip it was
encoded with. The "local attention" fallback only kicks in past ~10.9
minutes and even then uses a **symmetric** window (`att_left =
att_right = 128`, `:74-75`) -- it needs ~10s of *future* audio, not just
causal history. It is not a causal/streaming encoder in any mode
relevant to a live conversation.

This is a genuine M4A blocker, not a formality: every embedding measured
to date (including the R9700 baseline) was produced with cross-frame
context that 80ms-chunked encoding would discard.

**Clarification on top of the M4-0A finding:** because this encoder path
is bidirectional, exact offline-equivalent embeddings are not available
at the instant a frame arrives at all -- an embedding for frame N can
change once audio after frame N becomes available, since the encoder
attends over the whole submitted clip. A growing/sliding re-encode
*alone* does not solve this: it changes how much context is used, not
the fact that the "true" (whole-clip) embedding for any frame is
unknowable until the clip is known to have ended. M4A is therefore a
latency/context/equivalence problem, not just a compute-cost problem --
re-encoding more context on every frame does not remove the tradeoff, it
only lets you choose where on it you sit.

A live implementation must choose among:

- **Fixed lookahead**: delay emission of frame N's embedding until some
  fixed amount of future audio (e.g. N+k frames) has arrived, trading
  latency for closer-to-exact embeddings.
- **Bounded sliding context**: approximate the whole-clip encoder with a
  finite left/right context window re-encoded per frame -- cheaper than
  fixed lookahead in latency terms, but an *approximation* of the
  offline embedding, not a reproduction of it; how close the
  approximation needs to be is unmeasured.
- **A genuinely causal/chunked encoder modification**: change the
  encoder itself (causal conv, little/no lookahead attention) so a
  streaming embedding is not an approximation at all -- the most
  invasive option, not attempted by this design.
- **Downstream replay/revision**: let early embeddings be provisional and
  revise/replay generation as more audio arrives -- likely too expensive
  and architecturally complex for the primary M4 path; noted as a
  fallback of last resort, not a direction to pursue first.

None of these four is selected by this document. Choosing between them
needs its own follow-up (measuring how much fixed lookahead or how wide
a bounded context is actually required before embeddings are close
enough to whole-clip to preserve VoiceChat's output quality) before an
implementation branch commits to one.

### The live-timeline causality invariant

Today the frame loop can run as fast as the GPU allows because the
entire user clip is already known in advance (point 2: "not paced to a
clock"). Once the microphone is live, that stops being safe. If the GPU
can compute five frames ahead of what the user has actually said so far,
those frames implicitly assume future silence that hasn't happened yet:

```
wall clock:       0ms     80     160     240     320
mic available:     A       B       C
model timeline:    A B C D E F G       <- WRONG, D-G assume unheard silence
```

If the user speaks during that window, the model has already generated
past the point where that speech should have entered its state. So:

> The VoiceChat timeline must never advance beyond the latest microphone
> frame actually captured, except for explicitly defined inserted states
> such as tool-call frames.

This makes 80ms/frame a hard causal budget, not just a throughput number:
below it, the runtime keeps pace with real time; above it, conversational
lag accumulates and compounds. The scoreboard at the bottom of this
document is revised accordingly -- frame service time and deadline misses
matter more here than tokens/sec.

### Persistent VoiceChat state

Promote the turn-scoped locals identified in point 2 (`ai`/`aud`,
quiet-detection counters, tool-call freeze state) into `vc_session`
fields, so `step()` can run indefinitely across what are currently turn
boundaries, the same way `t`/`prev`/`fprev`/the KV cache already do.

### Incremental text events

No change. Already streams per frame (point 3). Calling this out
explicitly so an implementation pass doesn't spend time "fixing"
something that isn't broken.

### Incremental audio events

**Resolved by M4-0B (not a blocker -- the codec is already causal and
already chunks internally):** `codec_decode()`
(`voicechat-tts.cpp:1237-1309`) already decodes in 8-frame chunks with an
8-frame causal left-context overlap (`const int chunk = 8, overlap = 8;`,
`:1245`) that gets discarded before use (`ggml_concat(ctx, zpad, x, 1)`
-- an explicit causal left-pad, no right/future context anywhere in the
ConvNeXt block, `:1272-1273`). `voicechat_tts_write_wav()` already
exploits exactly this for mid-stream starts: a `lead = min(first, 8)`
frame run-up gets decoded and thrown away (`:1326`, comment in the code
itself: "the codec is causal, so a range that starts mid-stream is
decoded with a few frames of run-up that are dropped again; that is the
same trick codec_decode uses for its own chunking").

The one piece not yet chunked is the ISTFT/overlap-add stage, currently
one batch pass over the full spectrogram (`:1363-1401`) -- but a
ready-built streaming replacement already exists, unused:
`mtmd_audio_streaming_istft` (`tools/mtmd/mtmd-audio.h:161-191`, impl
`mtmd-audio.cpp:1324-1429`), with exactly the `process_frame()`/
`flush()`/`reset()` shape this needs. Incremental audio output is
therefore an **integration task** -- decode `codec_decode`'s existing
8-frame chunks incrementally and route them through
`mtmd_audio_streaming_istft` instead of one batch ISTFT call -- not an
open architectural risk. The original framing of this as an "open
question" parallel to the encoder side was too cautious.

### Simultaneous input and output

Requires real threading where none exists today (point 8): a capture
thread reading the mic into an input ring buffer, the existing
generation loop consuming from it and producing frames, and a playback
thread consuming decoded audio chunks independently. The `tts_q` FIFO
pattern is the only existing precedent to extend; input-side and
output-side ring buffers are new.

### Interruption events

Two named, separately-handled phenomena (point 6) -- do not conflate
them:

- **Assistant overlap**: turn `VC_NO_BARGE`'s currently-silent
  bos-while-`in_audio` rewrite into a surfaced event instead of
  swallowing it. This is the direct token-level hook already found in
  the code, not new detection logic.
- **User interruption**: no known signal exists for this yet. What the
  model does when new user speech arrives while the assistant is
  mid-response is unobserved -- M4-0 (below) is where this gets watched
  for the first time, not designed from assumption.

And within user interruption, two separate actions that must not be
conflated:

- **A. Playback cancellation** (UX decision, safe to do immediately): stop
  playing assistant PCM the human has not yet heard, the instant an
  interruption is recognized.
- **B. Model/TTS internal-state cancellation** (needs evidence, not yet
  decided): whether the underlying TTS/timeline state should reset at
  all. The runtime intentionally runs the text channel ahead of the
  spoken channel, with TTS state tracking that gap -- earlier runtime
  work found that arbitrarily disturbing that timing degrades speech
  quality. **Do not assume that muting audible output implies clearing
  internal TTS/model state** -- those are two different systems (what
  the user hears vs. what the model's timeline is doing), and only A is
  a safe default today.

### Bounded buffers and backpressure

Explicit sizing and drop/block policy needed for the new mic-in and
speaker-out ring buffers (none of this exists today, point 8). Not
specified further in this pass -- needs the encoder/decoder chunking
questions above resolved first, since buffer sizing depends on what
chunk granularity ends up being feasible.

### What blocks continuous operation in the current protocol

Named concretely, for the implementation branch to target directly:

- `vc_serve()`'s one-shot `getline`-per-command loop (`:1168`) -- there
  is no way to keep receiving input while a turn is in flight.
- The once-per-turn `turn_end` response shape (`:1073-1082`,
  `:1130-1224`) -- there is no concept of a response that is still open
  while new input keeps arriving.

## Repo split

Runtime-side changes (frame loop, protocol, encoder/decoder chunking)
belong in `boxwrench/llama-voicechat.cpp`, on a new branch off `amd/rocm`
-- not merged into `amd/rocm` without explicit authorization, per the
established branch/pin policy in `runtime/README.md`. Client-side changes
(continuous mic capture ring buffer, continuous playback) belong in this
repo, under `app/`, once the runtime side has something to talk to.

## M4-0: feasibility and causality spike -- DONE

Both questions resolved from source reading alone; no file-driven
experiment was needed for either (the code itself was decisive). No live
audio, no threading, no ring buffers, no protocol/client changes were
touched, per scope.

### M4-0A feasibility: BLOCKER CHARACTERIZED -- selection since resolved by M4A-1

See "Continuous PCM input" above for the full finding: the
parakeet/FastConformer encoder graph has no cross-call state and uses
full bidirectional self-attention over whatever's submitted. Exact
online equivalence to the whole-clip embeddings every prior measurement
(including the R9700 baseline) was produced with is **not achievable at
all** without future context or downstream replay -- this isn't a matter
of picking a big-enough re-encode window and calling it solved. What
M4-0A produced is a characterization of the tradeoff (fixed lookahead vs.
bounded sliding context vs. causal encoder modification vs.
replay/revision), not a resolution of it.

**M4A-1 (`research/experiments/m4a-1-lookahead-spike/`) has since made
the selection: PROMOTE zero-lookahead.**

```
initial sweep:      VC03-long, 3 positions x 9 lookaheads
adversarial sweep:  VC01-short, VC04-noisy, VC05-pause,
                     16 transition frames x 6 lookaheads
                     (onset, mid-word, pre/post-pause, silence, noise)

observed range:      cosine 0.99970-1.00000, RMSE 0.00000-0.00114
natural adjacent-
frame cosine:        0.53-0.9994

classification:      PROMOTE zero-lookahead
```

> Exact online equivalence remains impossible in principle for a
> bidirectional encoder, but measured future-context dependence is
> negligible relative to natural embedding variation across all tested
> speech and boundary conditions.

"Zero-lookahead" means no future audio context is required beyond a
frame's own arrival -- it does not mean zero latency; the current 80ms
of audio still has to arrive and the prefix seen so far still has to be
encoded. What that encode costs, and whether it stays inside the 80ms
causal budget as conversation length grows, is M4A-2, not yet started.

### M4-0B feasibility: RESOLVED

See "Incremental audio events" above: `codec_decode()` already chunks
internally in causal 8-frame windows with discarded run-up context, the
same trick `voicechat_tts_write_wav()` already uses for mid-stream
starts. Only the ISTFT stage needs wiring up, and a ready-built streaming
implementation (`mtmd_audio_streaming_istft`) already exists unused in
the codebase. M4B is an integration task, not an open feasibility
question.

### Constraint carried forward into M4A

Headphones are a requirement, not a recommendation, for the first live-
microphone test in M4A. Playing VoiceChat through speakers while feeding
the mic back in introduces acoustic echo -- a separate problem from
whether the native duplex model works at all, and one to defer to later
productization (AEC), not solve while still validating the core premise.

## Sequencing (bounded steps, not started by this document)

- **M4A -- continuous input.**
  - **M4A-1 (done)** resolved the context-vs-equivalence choice
    empirically: PROMOTE zero-lookahead (see M4-0A above and
    `research/experiments/m4a-1-lookahead-spike/`). No future audio
    context is required for a usably-close perception embedding.
  - **M4A-2 (next, not started)**: zero-lookahead still means the
    growing prefix seen so far must be re-encoded on every new frame
    (no encoder-side state exists). Measure whether that naive
    re-encode's cost stays inside the 80ms causal budget (see "The
    live-timeline causality invariant" above) as conversation length
    grows -- a systems/performance question now, not an open
    architectural one.
  - Only once M4A-2 establishes a viable encode strategy does feeding
    VoiceChat continuously at its native cadence with no artificial
    turn boundary become meaningful to build. Headphones required for
    the first live test.
- **M4B -- incremental output.** Replace "wait for a complete WAV" with
  streaming playback as TTS frames decode. The metric that matters
  changes from complete speech-to-speech time (the R9700 baseline's 4.244s
  mean) to time from user-stops-speaking to first audible assistant
  speech.
- **M4C -- simultaneous input and output.** Microphone and speaker both
  live at once, no pausing capture while VoiceChat talks.
- **M4D -- barge-in.** The real test: interrupt VoiceChat mid-response and
  have it react correctly, without a push-to-talk button, without an
  artificial "you may speak now," and without resetting the conversation.

## Strix Halo roadmap handoff

The Strix product goal is a fluent, continuous, low-latency spoken
conversation. Preserve behavior, not component identity: Nemotron's
conversation core and timeline remain the behavioral anchor, while perception,
TTS, codec, auxiliary ASR, AEC/VAD, and serving infrastructure may be kept,
relocated, or replaced when evidence says the conversation improves.

The prior-art and serving-options study starts now, before the full PC M4
implementation. See [docs/STRIX-ROADMAP.md](STRIX-ROADMAP.md) and
[docs/STRIX-SERVING-OPTIONS.md](STRIX-SERVING-OPTIONS.md).

Strix waits for the PC only at the production-shaped perception gate: M4A-2
must measure the promoted zero-lookahead/growing-prefix perception path against
the 80 ms budget and freeze its exact runtime SHA. PC M4B/M4C/M4D may continue
in parallel. No XDNA implementation, replacement-TTS integration, or current
fallback optimization begins merely because the source study is underway.

## Explicitly out of scope for this document

- No runtime or client code changes.
- No merge decision on `feature/push-to-talk` -- left pushed and
  unmerged, unaffected by this document.
- No implementation of M4-0A's sliding-window re-encode strategy or
  M4-0B's ISTFT streaming integration -- both are now scoped, neither is
  built.
- No live-microphone testing was part of M4-0 -- both findings came from
  source reading, no experiment was even needed.
- No benchmarking, no kernel/graph tuning.

## Revised scoreboard

The R9700-Q8-M1 baseline's 4.244s complete-turn number stays a valid
frozen reference point, but it stops being the primary optimization
target once M4 begins. The metrics that matter for a continuous duplex
system:

```
mean frame service time
p95/p99 frame service time
deadline misses (frame service time > 80 ms)
input backlog depth
output underruns
timeline lag vs. wall clock
user stops speaking -> first assistant text
user stops speaking -> first audible assistant speech
interruption -> assistant stops (playback cancellation)
interruption -> new response begins
missed interruptions
false interruptions
conversation continuity across interruptions
```

The causality invariant above makes the first six of these the real
scoreboard: whether the R9700 can execute each 80ms slice before the
next slice of reality arrives is the actual question M4 answers,
underneath all of the above.

And, ultimately, the qualitative question none of the above fully
capture on their own: does it feel like talking to something.
