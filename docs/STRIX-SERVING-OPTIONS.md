# Strix Halo serving options

Status: source/prior-art study started. No integration is approved by this
document.

Date of snapshot: 2026-08-24.

## Product north star

The target is a fluent, continuous, low-latency spoken conversation on:

```text
Ryzen AI MAX+ 395 / Strix Halo
Radeon 8060S gfx1151
XDNA2 NPU
```

The evaluation criterion is conversational behavior, not architectural purity:

> Does this make talking to the system feel faster, more natural, and more
> interruptible on Strix Halo?

The Nemotron conversational core and timeline are the strongest current
behavioral asset. They provide continuous state, learned turn-taking, overlap,
and interruption behavior. Perception, TTS, codec, auxiliary ASR, AEC/VAD,
audio preprocessing, and serving infrastructure are replaceable when another
configuration produces a better conversation.

The native NVIDIA TTS and codec are therefore the reference implementation,
not a permanent requirement.

## Functional role map

Every candidate is classified as one of:

```text
KEEP       retain the existing component or behavioral contract
RELOCATE   run the same component on another Strix engine
REPLACE    use an alternate implementation behind the same role contract
```

| Functional role | Current/reference choice | Candidate moves | Initial disposition |
| --- | --- | --- | --- |
| Audio input / perception | VoiceChat audio frames into learned FastConformer perception | Relocate FastConformer to XDNA2; replace only with separately evidenced perception | KEEP behavior; RELOCATE is the principal XDNA hypothesis |
| Conversational core / turn logic | Nemotron-H plus the continuous VoiceChat timeline | Relocate only if a replacement preserves learned state and interruption behavior | KEEP by default |
| Speech output / TTS | Native VoiceChat TTS producing incremental speech intent/output | Replace with Lemonade TTS, Kokoro, OpenMOSS, Chatterbox/another AMD-friendly renderer, or future XDNA2 TTS | KEEP as control; REPLACE explicitly open |
| Codec | Native VoiceChat codec and its causal streaming path | Replace or relocate only behind the streaming PCM contract | KEEP as control; replacement is allowed |
| Auxiliary ASR | None required by the Nemotron perception contract | FLM/Lemonade Whisper, GPU Whisper, or another transcript/caption channel | REPLACE only as an auxiliary tool; not a FastConformer substitute |
| AEC / VAD / audio preprocessing | Headphone-first validation; no AEC project yet | CPU DSP, GPU, NPU, or platform audio stack | REPLACE/RELOCATE only when productization needs it; VAD must not impose turn boundaries |
| Serving / routing | In-process VoiceChat timeline and runtime | Thin adapter, Lemonade router, FastFlowLM/other service, or a future generic backend | KEEP behavior; RELOCATE/REPLACE only if streaming, cancellation, and state survive |

The role map deliberately allows a mixed topology. The final system does not
need to be a literal NVIDIA-component port or a full NPU port.

## TTS is downstream of Nemotron state

Runtime source reading establishes an important boundary. The conversational
LLM feeds its sampled text token back into the next VoiceChat frame, while TTS
consumes that token separately. Replacing TTS therefore does **not** inherently
replace the Nemotron conversational state machine.

```text
Nemotron conversation timeline
          ├── sampled text token -> next VoiceChat frame
          │
          └── speech text token / delta -> speech renderer
```

The existing runtime nevertheless uses native TTS state for speech-progress and
speech-lifecycle control. In particular:

- the text stream can run far ahead of audible speech;
- EOS may be delayed while remaining speech finishes;
- the runtime tracks whether speech is still active;
- queued text can represent speech that has not yet been heard; and
- the end of a response includes a speech-drain phase.

The replacement boundary must preserve the behavior represented by those facts
without requiring the rest of VoiceChat to know about
`voicechat_tts_silence_cos()` or native codec internals.

## Evaluation scoreboard

Every candidate must eventually be evaluated in the same conversational
workload, against the native VoiceChat control. Record:

```text
first audible response latency
80 ms deadline misses and p95/p99 frame service time
timeline lag versus wall clock
user interruption -> playback stop
user interruption -> new response
audio continuity and underruns
conversation quality and naturalness
CPU load
gfx1151 utilization and headroom
XDNA2 activity and placement evidence
UMA / shared-memory contention
package power and thermal behavior
```

An NPU placement does not have to win every latency number. Equal
conversation latency with fewer deadline spikes, lower CPU load, more GPU
headroom, or materially better power/thermal behavior can be a serving win.
NPU utilization without a conversational benefit is not a win.

## Current bounded evidence

`STRIX-BRINGUP-1` keeps the first TTS bakeoff as `TTS-B0` and adds a repeatable
`TTS-B1` renderer study. Kokoro is currently `PROMISING`; the native-vs-Kokoro
numbers are **not yet a fair performance comparison** because native runs use
the cold full VoiceChat process while Kokoro uses a warm reused CPU model, and
the native public path exposes final WAV output rather than PCM chunks.

The isolated 12.5 Hz text-stream simulator found that a bounded-word policy
started Kokoro audio sooner than sentence or clause flushing on the tested
fixtures, with no modeled underruns. The current CPU adapter still cannot
preempt synchronous in-flight synthesis, so it is `QUALIFY`, not an approved
renderer integration. See [STRIX-BRINGUP-1](../research/experiments/STRIX-BRINGUP-1.md)
and the [TTS-B1 report](../research/experiments/strix-tts-bakeoff/generated/TTS-B1/REPORT.md).

Current TTS decision: **Kokoro `QUALIFY` — viable renderer architecture, not
currently compelling enough to displace native TTS.** Reopen it only if PC
work establishes native output as a real conversational bottleneck or another
renderer demonstrates genuinely streaming, preemptible synthesis.

The AMD Parakeet-TDT source study is also a qualified feasibility lead, not a
port. AMD's static-shape, depthwise Pad-to-Conv, and attention-mask rewrites
are relevant to the VoiceChat fallback observation, but the exact VoiceChat
embedding graph, Linux compile/load path, and operator correspondence remain
unproven. See the [XDNA feasibility study](../research/experiments/strix-xdna-parakeet-feasibility/README.md).

## Prior-art findings

### Lemonade Server

Lemonade is the most promising serving and routing prior art because it already
has a multi-backend server, model recipes, device-aware backend selection, and
OpenAI-compatible endpoints for completions, transcription, and speech. Its
backend registry is descriptor-based, and its authoring guide defines a path for
adding a backend with a model specification, install behavior, endpoint
implementation, and integration tests.

The current documented support picture is useful but not sufficient for this
product:

| Capability | Documented path | Directly reusable? | Boundary / blocker |
| --- | --- | --- | --- |
| XDNA2 LLM | `flm:npu` / FastFlowLM | Yes for a separate NPU model service | Does not preserve the in-process Nemotron timeline automatically |
| Strix GPU LLM | `llamacpp:rocm` and experimental `vllm:rocm` | Potentially, but current VoiceChat runtime is already the control | Router lifecycle and streaming behavior must not split the timeline |
| XDNA2 Whisper | Lemonade/FLM speech path and `whispercpp:npu` support claims | Prior art; isolated auxiliary proof possible | Whisper produces transcript/audio features, not VoiceChat FastConformer embeddings; documented backend/OS combinations differ |
| Kokoro TTS | `kokoro:cpu` on Linux/Windows; Metal on macOS | Maybe as an isolated renderer | Need first-chunk, PCM streaming, cancellation, voice-quality, and throughput proof on Strix |
| OpenMOSS TTS | Experimental `openmoss` on CUDA/ROCm/Vulkan | Maybe; AMD GPU path is attractive | Streaming/cancellation contract and stable gfx1151 behavior are not yet proven |
| RyzenAI / `ryzenai-server` | NPU/hybrid model serving through AMD's Ryzen AI stack | Prior art, not a renderer decision | Linux support and audio/TTS coverage must be verified; no direct VoiceChat boundary yet |
| Backend routing | `lemond`, recipes, descriptors, model registry, per-backend wrappers | Yes as infrastructure or upstream design reference | A generic HTTP router does not itself provide VoiceChat's continuous state or barge-in semantics |
| Custom backend | Descriptor + wrapper + model/install/test contracts | Yes in principle | A permanent VoiceChat backend should be considered for upstream contribution first |

Lemonade should not become a mandatory dependency prematurely. It is a
reusable infrastructure candidate, an implementation prior-art source, and a
possible upstream destination.

### FastFlowLM

FastFlowLM is the closest existing XDNA2 speech/runtime prior art. Its Linux
documentation describes `amdxdna`, XRT, NPU firmware, and memlock prerequisites;
it distinguishes `flm validate` from the XRT-backed execution path used to run
models. Its CLI documents an OpenAI-compatible server and a Whisper ASR mode
that loads alongside an LLM.

| Capability | Directly reusable? | Hardware engine | Streaming/state | Model/runtime | Linux status | Integration boundary | Obvious blocker |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FLM generative LLM | Yes for an auxiliary or replacement NPU LLM | XDNA2 NPU | Token streaming is supported by the serving API; continuous VoiceChat timeline is not implied | FLM model recipes and NPU kernels/runtime | Documented Linux path with `amdxdna` + XRT | OpenAI-compatible service | Replacing Nemotron would change the learned conversational behavior |
| FLM Whisper ASR | Yes as audio-encoder/runtime prior art or auxiliary transcript channel | XDNA2 NPU | Service-level streaming and incremental feature behavior require proof; ASR mode is documented with a concurrently loaded LLM | FLM Whisper model package and `libwhisper_npu` path | Linux support is documented, but version/driver details matter | Transcript/audio service boundary | It cannot be substituted for FastConformer embeddings without an architectural comparison |
| FLM custom audio encoder | Unknown | XDNA2 NPU | Unknown; static shapes and state boundaries are central questions | Likely requires FLM/IRON/compiler-facing model packaging, not a GGUF drop-in | Toolchain exists, custom path unproven here | Embedding tensor output into VoiceChat | No demonstrated custom FastConformer acceptance path |

The key distinction is:

```text
FastConformer -> XDNA2
```

is a relocation experiment, while:

```text
FastConformer -> Whisper transcript
```

is a model-architecture replacement. The latter requires separate evidence.

### AMD Ryzen AI Software / `ryzenai-server` / OGA / Vitis AI EP

AMD's Ryzen AI Software repository provides ONNX Runtime GenAI examples,
Vitis AI EP material, hybrid NPU/iGPU examples, and Whisper/Parakeet speech
examples. The general ONNX path is attractive for perception or audio
preprocessing because it is not restricted to LLM-shaped models.

| Candidate | Directly reusable? | Hardware engine | Streaming/state | Model/runtime | Linux status | Integration boundary | Obvious blocker |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ONNX Runtime GenAI / RyzenAI server path | Maybe for a separate model service or hybrid control | XDNA2 NPU plus CPU/iGPU depending on EP/model | Usually request/session shaped; streaming and persistent state must be verified | ONNX plus `genai_config.json`, EP/provider configuration, compiled/cache artifacts | AMD documents the stack, but Linux XDNA2 graph coverage and packaging are version-sensitive | ONNX/session or HTTP server boundary | Historical Linux EP/operator/packaging gaps and CPU fallback risk |
| ONNX FastConformer/auxiliary audio graph | Feasibility candidate | XDNA2 NPU with CPU fallback possible | Must prove bounded-lookahead, shape stability, and keep-up | ONNX opset/operator subset plus Vitis AI compilation/cache | Not assumed; verify on this Strix environment | Embedding or PCM tensor bridge | Unsupported operators can leave the critical path on CPU |

This is a possible general-purpose compiler/runtime route, not a reason to
start a conversion now. The first question is whether the resulting graph
helps the conversation after transfer, compile, and fallback costs.

### REM and `xdna-top`

REM provides useful workload-shape prior art: bounded streaming/background jobs
work best when deterministic state surrounds the model, and quality plus
keep-up rate matter together. Its Strix contention study found that an NPU
background generation job had roughly three times the throughput and roughly
three times the total-board performance-per-watt of a spare-CPU control in that
particular workload, while still causing measurable shared-memory contention.
Those numbers are not VoiceChat results; they justify measuring placement and
contention rather than assuming either isolation or free acceleration.

`xdna-top` is the measurement path. It attributes NPU contexts through XRT,
shows iGPU busy/power from kernel telemetry, and records concurrent activity.
It does not invent a generic NPU utilization percentage. For every future
placement claim, capture context ownership, submission/completion activity,
iGPU busy/power, and the conversation timeline together.

## Candidate cards by role

The cards below are the working classification for S3B. “Unknown” is an
explicit research result, not permission to assume compatibility.

### Conversational core — native Nemotron timeline

```text
classification       KEEP
reusable directly?    yes; this is the current behavioral reference
hardware engine       gfx1151 GPU in the validated Strix runtime
streaming capable?    target behavior; full production duplex awaits the PC SHA
stateful?             yes; continuous Nemotron/KV/timeline state is the asset
latency class         critical path; must meet the live 80 ms budget
model/runtime         VoiceChat runtime and pinned Nemotron Q8 model
Linux support         validated for the frozen turn-based Q8 path
integration boundary  in-process VoiceChat timeline
obvious blocker       production-shaped continuous-input runtime is not frozen
```

Do not replace this component merely because another runtime can serve an LLM.
Any replacement must reproduce the behavior that makes the system a
conversation rather than a VAD -> ASR -> LLM -> TTS pipeline.

### Perception — native FastConformer on gfx1151

```text
classification       KEEP control; current gfx1151 path includes CPU fallback
reusable directly?    yes as the control path after PC `D2` is frozen
hardware engine       gfx1151 plus CPU fallback in the current graph
streaming capable?    zero-lookahead production shape is still a cost question
stateful?             the current encoder graph has no useful cross-call state
latency class         critical 80 ms frame path; measure growing-prefix cost
model/runtime         native VoiceChat FastConformer/mtmd graph
Linux support         validated in the frozen Q8 turn-based workload
integration boundary  learned perception embeddings into Nemotron
obvious blocker       PC `D2` must establish a viable production-shaped path
```

`LEAD-GFX1151-0001` remains parked. Do not optimize its `CONV_2D_DW` or
`UNARY` fallback before the production-shaped perception path has been profiled.

### Perception — FastConformer relocated to XDNA2

```text
classification       RELOCATE candidate
reusable directly?    no; feasibility only
hardware engine       XDNA2 NPU, possibly with CPU fallback
streaming capable?    feasibility unknown on XDNA2; the currently selected
                      VoiceChat contract is zero-lookahead, with the
                      production execution shape pending PC `D2`
stateful?             current VoiceChat encoder has no useful cross-call
                      state; an XDNA implementation must reproduce the
                      production-shaped perception contract selected by
                      `D2` rather than inventing incompatible state semantics
latency class         critical; must fit the 80 ms frame budget plus transfers
model/runtime         likely ONNX/compiled XDNA graph or a FastFlowLM/IRON path
Linux support         runtime stack exists; this exact graph is unproven
integration boundary  NPU output tensor -> shared UMA -> Nemotron timeline
obvious blocker       no demonstrated custom FastConformer embedding path
```

This is the principal NPU relocation hypothesis, but it is not approved for
implementation until S4 profiling shows the role is worth moving.

### Auxiliary ASR — Whisper through FastFlowLM/Lemonade

```text
classification       REPLACE candidate for an auxiliary channel only
reusable directly?    yes as XDNA2 audio prior art and a possible tool channel
hardware engine       XDNA2 NPU through FLM; Lemonade may route the service
streaming capable?    request/token streaming is documented; frame-level
                      perception embeddings are not the contract
stateful?             service/model state may exist, but not Nemotron timeline state
latency class         auxiliary; may be critical only if product evidence says so
model/runtime         Whisper model recipe through FLM/Lemonade
Linux support         FastFlowLM Linux path documented; exact versions matter
integration boundary  transcript/caption/tool side channel
obvious blocker       Whisper transcript is not FastConformer perception
```

Use this to prove the AMD speech toolchain or add captions/tools later. Do not
call it a drop-in perception replacement.

### Speech output — native VoiceChat TTS plus codec

```text
classification       KEEP reference
reusable directly?    yes; current behavior is the control renderer
hardware engine       gfx1151 GPU in the native VoiceChat runtime
streaming capable?    intended incremental path; codec is causal, ISTFT wiring
                      remains part of the production integration
stateful?             yes; native TTS state participates in speech lifecycle,
                      but is downstream of the Nemotron conversational state
latency class         critical; first audible speech and underruns matter
model/runtime         native NVIDIA TTS/codec artifacts and runtime
Linux support         validated in the frozen turn-based Q8 workload
integration boundary  incremental speech intent -> native codec -> PCM
obvious blocker       full duplex renderer integration is not yet frozen
```

This is the reference, not a commitment to keep the implementation forever.
The important behavior is the lifecycle contract, not native TTS identity.

### Speech output — Lemonade Kokoro

```text
classification       REPLACE candidate
reusable directly?    isolated proof candidate; no VoiceChat integration yet
hardware engine       documented CPU on Linux/Windows; Metal on macOS
streaming capable?    Lemonade exposes a speech endpoint; PCM chunking and
                      first-chunk behavior must be measured on Strix
stateful?             renderer/session state, not conversational state
latency class         unknown; likely viable only if warm and incremental
model/runtime         Kokoro recipe and Kokoro-82M voice/model packaging
Linux support         documented CPU path
integration boundary  incremental text -> Lemonade audio/speech endpoint -> PCM
obvious blocker       throughput, cancellation, voice quality, and buffering
```

Kokoro is attractive because it is a known open TTS family and a simple
renderer boundary, but it is not presumed to beat the native path.

### Speech output — Lemonade OpenMOSS

```text
classification       REPLACE candidate
reusable directly?    isolated proof candidate; experimental backend
hardware engine       documented CUDA, Vulkan, and ROCm GPU paths
streaming capable?    unknown for the required incremental PCM/cancel contract
stateful?             likely request/render state, not Nemotron timeline state
latency class         unknown; measure first chunk and sustained real-time rate
model/runtime         Lemonade OpenMOSS recipe/backend
Linux support         documented Linux ROCm/Vulkan availability
integration boundary  incremental text -> renderer service -> PCM
obvious blocker       streaming response format, cancellation, and gfx1151 proof
```

OpenMOSS is explicitly open for investigation because an AMD GPU renderer may
fit the Strix better than the native NVIDIA TTS. Its “experimental” status is
part of the qualification, not a reason to reject it in advance.

### Speech output — Chatterbox ROCm or another AMD-friendly renderer

```text
classification       REPLACE candidate
reusable directly?    isolated proof only
hardware engine       likely gfx1151 ROCm where a maintained build exists
streaming capable?    unknown; many TTS systems render complete utterances
stateful?             renderer state may be local; timeline state must remain outside
latency class         unknown to potentially high; first audio is decisive
model/runtime         backend-specific weights and server/runtime packaging
Linux support         candidate-specific; verify on Strix
integration boundary  renderer adapter accepting incremental text and returning PCM
obvious blocker       streaming and cancellation are not implied by a TTS demo
```

This category stays open so the project can choose the best renderer rather
than prematurely naming a winner.

### Codec — native codec or alternate renderer codec

```text
classification       KEEP control; REPLACE allowed behind the PCM contract
reusable directly?    native codec is directly reusable as the reference
hardware engine       gfx1151 GPU or the engine selected by the renderer
streaming capable?    native ConvNeXt codec is causal and chunks internally;
                      end-to-end streaming still needs the production wiring
stateful?             yes; overlap/context and cancellation state matter
latency class         critical after the renderer emits speech
model/runtime         native codec or renderer-owned decoder/codec
Linux support         native path validated in the frozen workload
integration boundary  renderer frames -> PCM ring buffer
obvious blocker       format, sample rate, cancellation, and underrun contract
```

The codec is not sacred. It is sacred only that the speaker receives timely,
continuous PCM and that interruption does not corrupt the timeline.

### AEC / VAD / audio preprocessing

```text
classification       REPLACE/RELOCATE candidate when productization requires it
reusable directly?    no single chosen implementation yet
hardware engine       CPU DSP first; XDNA2/GPU are candidates for persistent work
streaming capable?    should be frame-streaming and bounded
stateful?             AEC/filter state yes; VAD state yes; turn ownership no
latency class         critical auxiliary path; must not consume the 80 ms budget
model/runtime         conventional PCM/DSP or small ONNX/NPU graph
Linux support         platform/library-specific
integration boundary  PCM frames before perception and playback reference
obvious blocker       acoustic validation, device routing, and shared-memory cost
```

Headphones remain the initial duplex validation setup. VAD may help with
diagnostics or audio quality, but must not silently become an externally imposed
turn boundary unless the evidence says that improves the product.

### Serving / routing — Lemonade or a thin adapter

```text
classification       RELOCATE/REPLACE candidate; KEEP the timeline contract
reusable directly?    Lemonade routing/backend infrastructure is reusable in principle
hardware engine       router is CPU; child engines may be XDNA2, gfx1151, or CPU
streaming capable?    endpoints exist; cross-component timeline streaming is unknown
stateful?             router/model sessions can be stateful; Nemotron state is separate
latency class         control-plane overhead must be negligible on the live path
model/runtime         Lemonade recipes/descriptors or a thin in-process adapter
Linux support         documented multi-backend Linux support, version-sensitive
integration boundary  incremental events, cancellation, PCM, telemetry, errors
obvious blocker       a generic server can accidentally turn VoiceChat into services
```

The preferred boundary is thin: preserve the Nemotron session and timeline,
then route replaceable roles behind explicit streaming/cancellation contracts.
If a generic backend capability is missing, evaluate contributing it upstream
to Lemonade/FastFlowLM before maintaining a permanent VoiceChat-only fork.

## Replaceable speech-renderer contract

The renderer must satisfy this contract regardless of implementation. The
conceptual data path is:

```text
Nemotron conversational timeline
          │
          └── push_text(token / delta)
                        │
                        ▼
                 speech renderer
                        ├── incremental PCM / audio
                        │
                        └── speech state
                             - started
                             - currently speaking
                             - finished / drained
                             - stop pending playback
                             - reset renderer state
```

The intended future structure is one speech-renderer interface with the native
implementation retained as the control and an alternate renderer behind the
same boundary:

```text
Nemotron conversational timeline
            │
            ▼
     speech-renderer interface
         /             \
        /               \
native VoiceChat      alternate renderer
TTS + codec            Lemonade / other
```

The renderer also exposes lifecycle state/events. Exact API names are not
selected yet; this is documentation only, not an API selection:

```text
started
speaking
settled / drained
cancel_pending_audio
reset
```

Minimum requirements:

1. `push_text(token/delta)` must accept text as fast as Nemotron emits it. Do
   not throttle the model's text stream to natural speaking rate; the renderer
   buffers internally as needed.
2. Begin PCM output incrementally rather than waiting for a complete utterance.
3. Expose `started` and `speaking` state so the timeline can distinguish
   accepted text from audible/rendered speech.
4. Expose `settled` / `drained` when all accepted speech has actually finished
   rendering and pending PCM has drained. EOS and the response's speech-drain
   phase must not rely on a native silence helper.
5. Support fast `cancel_pending_audio` when the user interrupts. Define what
   happens to text already accepted by the renderer but not yet spoken: discard,
   preserve, replay, or another explicit policy.
6. Distinguish playback cancellation from destructive renderer `reset`, and
   distinguish both from resetting Nemotron conversational/model state.
7. Provide enough lifecycle and queue state that the runtime does not depend on
   `voicechat_tts_silence_cos()` or native codec internals.
8. Sustain output faster than real time with enough bounded buffering to avoid
   starvation and underruns.
9. Declare sample rate, channel layout, sample format, frame boundaries, and
   ownership/lifetime for every output buffer.
10. Preserve acceptable voice quality and intelligibility under incremental
    synthesis, not only in a complete-utterance demo.

The first renderer proof should be isolated: text fixtures in, timestamped PCM
out, lifecycle events recorded, cancellation injected at controlled points, and
no change to the conversation model. Do not wire a candidate into VoiceChat
until it passes that contract and has a plausible serving benefit. This phase
documents the interface only; it does not implement it.

## Dependency and sequencing gates

```text
NOW
  S3A prior-art study
  S3B KEEP / RELOCATE / REPLACE role map
  S3C renderer and serving contracts

PC completes `D2` and freezes production-shaped perception
  ↓
Strix builds the exact runtime SHA on gfx1151
  ↓
profile the same perception path and complete the cost map
  ↓
rank serving candidates
  ↓
first bounded heterogeneous experiment

PC `D1` / `D3` / `D4` / `D5` may continue in parallel
```

The full `STRIX-DUPLEX-GPU-M1` baseline and later A/B/C system comparison are
required before claiming that a serving topology improves fluent conversation,
but they are not prerequisites for this source study.

## Explicit non-goals for this phase

- No full XDNA implementation.
- No FastConformer conversion before production-shaped profiling.
- No replacement-TTS integration except a cheap isolated proof that passes the
  renderer contract.
- No optimization of the current CPU perception fallback.
- No investigation of `LEAD-GFX1151-0001` before its production-shaped profile.
- No mandatory Lemonade dependency.
- No conventional VAD -> ASR -> LLM -> TTS rewrite without evidence that it is
  the better conversational system.
- No requirement to wait for PC `D1`/`D3`/`D4`/`D5` (or `H1`, RX 7900 XT
  validation) before completing this study.

## Sources checked

- [Lemonade overview and support matrix](https://github.com/lemonade-sdk/lemonade)
- [Lemonade backend reference](https://github.com/lemonade-sdk/lemonade/blob/main/docs/dev/backends-reference.md)
- [Lemonade adding-a-backend guide](https://github.com/lemonade-sdk/lemonade/blob/main/docs/dev/adding-a-backend.md)
- [Lemonade custom model configuration](https://github.com/lemonade-sdk/lemonade/blob/main/docs/guide/configuration/custom-models.md)
- [FastFlowLM Linux guide](https://github.com/ROCm/FastFlowLM/blob/main/docs/linux-getting-started.md)
- [FastFlowLM CLI and ASR guide](https://github.com/ROCm/FastFlowLM/blob/main/docs/docs/instructions/cli.md)
- [AMD Ryzen AI Software](https://github.com/amd/RyzenAI-SW)
- [AMD Ryzen AI model deployment](https://ryzenai.docs.amd.com/en/latest/modelrun.html)
- [xdna-top](https://github.com/boxwrench/xdna-top)
- [REM workload-pattern notes](https://github.com/boxwrench/REM)
