# Architecture

## Repository lineage

```
ggml-org/llama.cpp
        ^
sansamour/llama-voicechat.cpp
        ^
boxwrench/llama-voicechat.cpp (voicechat)   fork default branch, upstream lineage
        ^
        amd/q8-bringup                       frozen: first working AMD Q8 bring-up
        ^
        amd/rocm                             ongoing AMD runtime integration branch
        |
        | explicitly validated commit pinned / consumed by
        v
boxwrench/Nemotron-VoiceChat-ROCm    this repo: install, reproduce, benchmark,
                                      use; AMD-facing UX and research
```

This repository does not vendor the llama.cpp source tree and is not where
low-level runtime, HIP/backend, or Q8-converter changes happen. Those belong
in `boxwrench/llama-voicechat.cpp`. `amd/rocm` is where candidate AMD
runtime changes accumulate; this repository always consumes an explicitly
validated commit pinned in `runtime/README.md`, never the floating branch
tip.

## Client/server split (push-to-talk, M3+)

Microphone capture does not go inside inference-core code. The runtime
exposes a persistent `--serve` process with a JSON turn protocol; a thin
client owns everything audio-input-related:

```
llama-voicechat --serve
      |
      | persistent session / JSON
      v
thin AMD VoiceChat PTT client (app/push-to-talk/)
      |
      +-- microphone capture
      +-- hold/release key
      +-- send audio
      +-- play returned speech
      +-- display latency/events
```

The model stays resident between turns; the client only opens/reuses a
session. Initial UX: hold SPACE to record, release to submit the turn, hear
the response.

## Perception: what M4 actually found

The R9700 baseline's ~19.9ms perception-encode number (against a ~4.244s
end-to-end speech-to-speech turn) was never evidence perception was fine
for a live 80ms frame -- it was one whole-clip encode, not a per-frame
cost. The M4 investigation (see
[docs/M4-DUPLEX-DESIGN.md](M4-DUPLEX-DESIGN.md)) profiled the real
interactive workload (main VoiceChat timeline, function head, TTS
backbone, MoG, RVQ, codec, CPU work, synchronization, buffering/playback)
and found: perception's *future*-context dependence is negligible
(zero-lookahead is viable), but a naive growing-prefix re-encode of the
*past* gets too expensive above roughly 20-24s of conversation, and no
bounded historical window tested so far has both adequate timing margin
and reliable downstream fidelity. Later D2 work superseded that dead end
with a bounded encoder-state mechanism and a no-normalization frontend
probe. Those mechanisms are research-qualified, not yet the normal mtmd
encode path; their production integration is D3 work. Do not reopen the old
window sweep or treat the historical whole-prefix timing as the D2 control.

## North star and track structure (post-v0.1)

The primary program from here is **fluent VoiceChat on a single AMD
GPU**. Acceptance is behavioral, not throughput: no push-to-talk button,
an always-open microphone, a continuous session, the model's timeline
staying causal to real microphone time (it must never advance past the
latest actually-captured audio), the assistant deciding for itself when
to speak, speech beginning incrementally, the microphone staying open
while the assistant talks, the user being able to interrupt, the stale
response stopping, the assistant reacting to the interruption, and the
conversation continuing without a reset.

One primary track carries that program (Track A: R9700 live duplex, the
D1-D7 phases in the [README](../README.md#roadmap)), with four
supporting tracks:

- **Track B** -- bottleneck optimization / component substitution,
  triggered only by measurements Track A actually produces, never chosen
  because an optimization sounds attractive.
- **Track C** -- the Strix Halo serving/XDNA program, running in
  parallel and importing stable contracts (renderer queue, perception,
  live-timeline protocol) from Track A as they land, rather than
  independently inventing another VoiceChat runtime. See
  [docs/STRIX-ROADMAP.md](STRIX-ROADMAP.md).
- **Track D** -- gfx1100/hardware portability (H1), and later a
  multi-GPU diagnostic ceiling experiment (H2) once the known
  multi-visible-device crash has its own bounded fix.
- **Track E** -- release/product maintenance. v0.1.x takes real install
  bugs, runtime crashes, bad documentation, security issues, and serious
  regressions -- nothing experimental. There is no reason for duplex
  research to compromise itself to stay releasable every day once a
  real release already exists.

R9700/PC development is the **reference-runtime** track: it is where
new contracts (renderer queue shape, perception context strategy, live
timeline protocol) are proven first. Strix and other hardware targets
consume those contracts once stable, rather than each re-deriving their
own VoiceChat duplex architecture independently.

## D3 integration boundary

`D3` is active. The qualified D1 renderer and D2 perception mechanisms have
not yet entered the release runtime, so D3 is both an integration and a
production-qualification phase. Its first slice has one non-negotiable rule:

> One captured 80 ms microphone slice authorizes exactly one VoiceChat
> timeline step. The timeline never runs ahead of captured microphone audio,
> except for explicit tool/system frames.

The detailed gap map, ownership model, telemetry contract, and first-slice
scope are in [D3-INTEGRATION-GAP.md](D3-INTEGRATION-GAP.md). D4 turn-taking,
D5 interruption, custom XDNA, multi-GPU, and component optimization remain
outside this first D3 slice.

## Function/tool channel

The tool-call channel is QUALIFIED, not blocking: the model selects and
invokes the correct tool and consumes its result, but emits tool-call JSON
missing a colon after `arguments`. Tracked in
[docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md); not fixed as part of the
initial voice experience.
