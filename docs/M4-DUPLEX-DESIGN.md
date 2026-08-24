# M4 design: native continuous duplex VoiceChat

Status: design only. No runtime or client code has changed as a result of
this document. It exists to scope M4 before any implementation branch is
opened.

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

### 6. No barge-in signal is surfaced anywhere

Point 5 is proof the model already wants to interrupt on its own -- but
nothing in the current code turns "the model sampled `bos` while
`in_audio` was true" into an event a caller can react to. It's silently
rewritten to `pad` and dropped. There is a separate sotc/eotc/eotr
function-head channel, but it is scoped to tool-calls, not
user/assistant speech overlap; it is not a source for M4's interruption
signal.

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

**Open question, not resolved by this document:** whether the
FastConformer encoder as wired through mtmd supports chunked/streaming
encoding at all, or whether it fundamentally needs a fixed-size window of
context to produce a comparable embedding to the whole-clip path today.
This needs a dedicated read of mtmd's encoder implementation before an
implementation branch can commit to an approach. Flagging it here rather
than guessing.

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

Extend `voicechat_tts_step()`'s existing per-frame RVQ code output with
an incremental decode-to-PCM path -- chunked ConvNeXt/ISTFT decode per
frame or per small group of frames, replacing the single end-of-turn
`voicechat_tts_write_wav()` call. Whether the ConvNeXt/ISTFT stage can
decode a partial code sequence without artifacts at chunk boundaries is
the open question here, parallel to the mtmd question above -- also not
resolved by this document.

### Simultaneous input and output

Requires real threading where none exists today (point 8): a capture
thread reading the mic into an input ring buffer, the existing
generation loop consuming from it and producing frames, and a playback
thread consuming decoded audio chunks independently. The `tts_q` FIFO
pattern is the only existing precedent to extend; input-side and
output-side ring buffers are new.

### Interruption events

Turn `VC_NO_BARGE`'s currently-silent bos-while-`in_audio` rewrite
(point 5) into a surfaced `barge_in` event instead of swallowing it --
this is the direct token-level hook already found in the code, not new
detection logic. Document what state resets and what carries over when a
barge-in is acted on: the in-progress assistant response's TTS queue and
any not-yet-decoded audio should stop/drain, while the underlying
timeline/KV-cache state stays live and continuous (per point 2, this is
one continuous session, not a reset between "turns").

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

## Sequencing (bounded steps, not started by this document)

- **M4A -- continuous input.** Prove VoiceChat can be fed microphone audio
  continuously at its native cadence with no artificial turn boundary
  imposed from outside. Headphones recommended for the first test so
  speaker echo doesn't confound the result.
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

## Explicitly out of scope for this document

- No runtime or client code changes.
- No merge decision on `feature/push-to-talk` -- left pushed and
  unmerged, unaffected by this document.
- No resolution of the two open feasibility questions (streaming
  FastConformer encode, chunked ConvNeXt/ISTFT decode) -- flagged as
  follow-up reads, not answered here.
- No benchmarking, no kernel/graph tuning.

## Revised scoreboard

The R9700-Q8-M1 baseline's 4.244s complete-turn number stays a valid
frozen reference point, but it stops being the primary optimization
target once M4 begins. The metrics that matter for a continuous duplex
system:

```
continuous frame budget           < 80 ms
user stops speaking -> first assistant text
user stops speaking -> first audible assistant speech
interruption -> assistant stops
interruption -> new response begins
audio underruns
missed interruptions
false interruptions
conversation continuity across interruptions
```

And, ultimately, the qualitative question none of the above fully
capture on their own: does it feel like talking to something.
