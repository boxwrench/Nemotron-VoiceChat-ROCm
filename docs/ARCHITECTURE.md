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
and reliable downstream fidelity. That is now a live open question (D2
in the roadmap), not a closed one -- do not reopen FastConformer
optimization from the stale baseline number, but do not treat perception
as settled either.

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

## Current D1/D2 contract status

The reference track has now produced a qualified D1 renderer architecture:
the producer publishes immutable TTS-frame snapshots to a worker-owned codec
scheduler and bounded PCM ring. It is documented in
[research/experiments/d1-async-renderer](../research/experiments/d1-async-renderer/README.md)
and still needs R9700/GPU-contention and real playback-device qualification.

D2-S1 has a passing bounded-state encoder milestone, and D2-S2 now has a
research-qualified chunked PCM frontend. The path retains per-layer attention
K/V history and causal-convolution context, reaches a 14.55 MiB state plateau,
and reproduces the authoritative VoiceChat raw log-mel output exactly across
chunk boundaries, with a bounded pre-encoder parity gate and a documented VC05
tail envelope. VC01, VC04, VC05, and VC06 retain
exact token/function traces; VC02 and VC03 remain semantically coherent but
show bounded-encoder numerical drift that crosses sampler boundaries. The
R9700/GPU service curve is still unavailable in the current namespace. The
earlier normalization blocker was corrected at source: pinned VoiceChat uses
Parakeet with `norm_per_feature=false`; the full-session normalized path is a
separate conformer configuration. See
[research/experiments/d2-perception-state](../research/experiments/d2-perception-state/README.md).

Strix must not choose a production XDNA input shape or cache topology from
this encoder-stage contract. It wakes only when D2 supplies the completed
runtime SHA plus frontend/subsampling state, importable history/cache
contract, and the D3 live-timeline contract.

## Function/tool channel

The tool-call channel is QUALIFIED, not blocking: the model selects and
invokes the correct tool and consumes its result, but emits tool-call JSON
missing a colon after `arguments`. Tracked in
[docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md); not fixed as part of the
initial voice experience.
