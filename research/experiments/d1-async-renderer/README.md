# D1: asynchronous native renderer prototype

Status: **QUALIFY** — the renderer-queue architecture works in a CPU-only
prototype; reference-GPU and real playback-device qualification remain open.
This is the D1-Q stopping point until an ordinary R9700 host shell can expose
the GPU and ALSA devices.

This experiment removes incremental codec/ISTFT work from the producer's
synchronous path. It does not claim that the complete continuous VoiceChat
timeline is implemented.

## Runtime provenance

```text
base/reconstructed M3.1: 09f4c7b4a414ae060a2325714e612dfd9c057811
D1 prototype runtime:   14676822b9b973070ee04d1d8ebf5ba11fff22b2
branch:                 research/d1-async-renderer
```

The product pin remains the reconstructed M3.1 commit. The D1 runtime is a
research branch and is not yet the product pin.

## Architecture

```text
VoiceChat producer / timeline
        |
        | immutable TTS frame snapshot; no codec wait
        v
worker-owned codec scheduler
        |
        v
bounded PCM ring (640 ms prototype capacity)
        |
        v
real-time sink (discard sink in this experiment)
```

The worker has separate ggml backend instances and a separate scheduler from
the main TTS path. An early version shared backend handles and aborted with a
heap-corruption failure; separate backend instances are required for this
prototype.

The producer still advances the native TTS model synchronously. D1 moves only
causal codec decode and streaming ISTFT off that producer path, which is the
M4B-1 bottleneck being tested.

## Contract candidate

The exact public API is not frozen in the runtime yet. The behavioral contract
for a future renderer boundary is:

```text
start/reset(first_frame)
append/publish immutable speech-frame snapshot
finish input
cancel_pending_audio
wait drained/settled
```

The renderer exposes these lifecycle facts independently of model state:

```text
started
speaking / first PCM queued
drained       all accepted render work has been rendered
settled       the playback sink has consumed the PCM ring
cancelled     queued/unheard output was discarded
reset         renderer state only; conversational state is preserved
```

Required semantics:

- text or TTS frames may run ahead of audible speech;
- accepted-but-unheard output has an explicit fate;
- input completion and audible completion are different events;
- bounded backpressure belongs to the worker/ring policy, not codec work on
  the 80 ms producer thread;
- cancellation discards queued PCM and stops at the next codec boundary;
- cancellation does not reset Nemotron state or the conversation timeline;
- an in-flight codec call is not preemptible in this prototype, so cancellation
  latency includes that call;
- a future real sink must emit `playback_begin` when its pipe receives the
  first real PCM, not merely when a renderer worker starts.

This preserves the native lifecycle distinction between text completion,
speech drain, and playback settlement.

## Measurements

The tests used the VoiceChat Q8 TTS model, deterministic `--say` fixtures,
`--tts-device CPU`, and a real-time discard sink. No ALSA device or R9700 GPU
was available in this execution namespace.

Normal drain run:

```text
published             42
publish_us            287
first_pcm_us          150248
pcm_samples           75858
max_ring_samples      14112
underruns             0
drain_us              2903080
settle_us             3972855
cancel_us             -1
codec_us              2672434
istft_us              10989
```

Cancellation at 500 ms:

```text
published             47
publish_us            302
first_pcm_us          171285
pcm_samples           10578
max_ring_samples      3528
underruns             0
drain_us              -1
settle_us              -1
cancel_us             366169
codec_us              835809
istft_us              2550
```

`first_pcm_us` is first PCM admitted to the prototype ring, not a hardware
playback timestamp. The cancellation result shows queued audio can be
discarded without a model reset, but the current worker cannot interrupt a
codec call already in progress.

Historical M4B control evidence remains the reason for this design: chunk-8
synchronous codec calls cost 164–232 ms, while aggregate throughput was above
real time; graph construction/reuse was not the cause. D1 therefore tests
scheduling and contention before any codec-kernel work.

## Classification and remaining qualification

```text
D1: QUALIFY

PASS in this prototype:
  producer publishes without waiting for codec completion
  codec and ISTFT execute on a background worker
  bounded PCM ring and discard sink operate without underruns
  drain/settle are distinct
  cancellation preserves conversation/model state

OPEN:
  R9700/GPU contention and main-frame p95/p99
  actual aplay/ALSA first-PCM and underrun behavior
  integration with D3's live timeline and playback event sink
  real text-delta scheduling rather than the current TTS-frame publisher
```

The current execution namespace has no `/dev/kfd`, GPU render node, or
`/dev/snd`, so it cannot produce the requested R9700/ALSA result. This is a
host-access boundary, not evidence against the queue architecture. The
qualification run must be repeated outside this namespace with the same
model/settings for the native-disabled and async cases, including real
`playback_begin`, ALSA underruns, GPU contention, and main-frame p95/p99.

The contract above is the D1 import candidate for later hardware work. Strix
must not import the research runtime branch until D2 and the live D3 timeline
freeze the complete production contract.
