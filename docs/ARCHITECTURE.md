# Architecture

## Repository lineage

```
ggml-org/llama.cpp
        ^
sansamour/llama-voicechat.cpp
        ^
boxwrench/llama-voicechat.cpp        runtime fork: Q8 converters, HIP/backend
        |                            fixes, changes appropriate to upstream
        | pinned / consumed by
        v
boxwrench/Nemotron-VoiceChat-ROCm    this repo: install, reproduce, benchmark,
                                      use; AMD-facing UX and research
```

This repository does not vendor the llama.cpp source tree and is not where
low-level runtime, HIP/backend, or Q8-converter changes happen. Those belong
in `boxwrench/llama-voicechat.cpp`, pinned here by commit (see
`runtime/README.md`).

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

## Why perception is not the current focus

The R9700 baseline measured perception (FastConformer encoder) at ~19.9 ms
against a ~4.244 s end-to-end speech-to-speech turn. Do not reopen
FastConformer optimization based on that baseline; the bottleneck, if one
exists for the interactive workload, has not yet been isolated. Once
push-to-talk exists, profile the actual interactive workload (main
VoiceChat timeline, function head, TTS backbone, MoG, RVQ, codec, CPU work,
synchronization, buffering/playback) before optimizing anything (M7).

## Function/tool channel

The tool-call channel is QUALIFIED, not blocking: the model selects and
invokes the correct tool and consumes its result, but emits tool-call JSON
missing a colon after `arguments`. Tracked in
[docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md); not fixed as part of the
initial voice experience.
