# Strix Halo serving roadmap

The product goal is a fluent, continuous, low-latency spoken conversation on
Strix Halo. The program therefore preserves behavior, not component identity.

The Nemotron conversational core and timeline are valuable because they provide
continuous state, learned turn-taking, overlap, and interruption behavior. But
perception, TTS, codec, auxiliary ASR, AEC/VAD, and serving infrastructure are
replaceable when another component produces a better conversational system.

The north-star question for every decision is:

> Does this make talking to the system feel faster, more natural, and more
> interruptible on Strix Halo?

Architectural purity, component loyalty, and NPU utilization are not success
criteria by themselves.

## S0 — Frozen reference state — DONE

Current frozen state:

```text
Ryzen AI MAX+ 395 / Strix Halo
Radeon 8060S gfx1151
VoiceChat 11B Q8

BUILD   PASS
LOAD    PASS
STT     PASS
TTS     PASS
S2S     PASS

VALIDATED
known perception/CLIP CPU fallback
LEAD-GFX1151-0001 parked
```

This result is immutable. Do not retroactively optimize the validation pass.
Its unsupported ROCm `CONV_2D_DW` / `UNARY` perception operators remain a
parked performance lead, not an active implementation task.

# Chapter 1 — Fluent conversation and production-shaped perception

## S1 — Import the production-shaped PC perception runtime

Do not independently reimplement the conversational timeline on Strix. The
PC/R9700 remains the development and reference machine for the Nemotron runtime
and its M4 behavior.

The prior-art and serving study starts now. Strix waits only for the PC to
finish the next perception gate:

```text
M4-0    DONE
M4A-1   PROMOTE zero-lookahead
M4A-2   measure growing-prefix perception cost versus the 80 ms budget
        then freeze a production-shaped continuous-input perception runtime
```

Once that exact runtime SHA exists, build it for gfx1151 and measure the same
perception path on Strix. PC M4B/M4C/M4D may continue in parallel; Strix does
not wait for the entire PC duplex milestone or for RX 7900 XT validation.

When the full continuous runtime is later frozen, Strix imports that exact
`llama-voicechat.cpp` SHA rather than independently implementing duplex. The
identical M4 acceptance suite remains:

```text
continuous PCM
incremental TTS
simultaneous input/output
native turn taking
interruption
```

Initial acceptance uses headphones. Do not begin an AEC project as part of
this step.

The Strix acceptance suite must demonstrate:

```text
no button / no Enter
open microphone
natural turn boundary
assistant begins speaking incrementally
microphone remains active while assistant speaks
user can interrupt assistant
session remains continuous
```

## S2 — Establish the Strix real-time baseline

Once the imported duplex runtime works, characterize conversation rather than
the old turn-based S2S path. Record this as:

```text
STRIX-DUPLEX-GPU-M1
```

Primary metrics:

```text
perception frame service time
mean / p95 / p99 frame time

80 ms deadline misses
timeline lag vs wall clock
input backlog depth

user-stop -> first text
user-stop -> first audible speech

user-interrupt -> playback stop
user-interrupt -> new response

audio underruns
conversation continuity
```

Also capture gfx1151 utilization, CPU utilization, GTT/UMA consumption,
package power, and perception fallback activity. This is the control baseline
every later NPU experiment must compare against. An NPU placement must
demonstrate a material serving benefit -- critical-path latency, deadline
reliability, compute headroom, power/thermal behavior, CPU relief, or another
explicitly measured objective -- without unacceptable conversation-quality
regression. It does not have to win every metric.

# Chapter 2 — Serving topology study and heterogeneous Strix

## S3A — Map serving prior art now

This source/prior-art phase starts immediately. It does not require a finished
PC duplex runtime and it does not authorize an integration. Study the existing
AMD machinery with a product question: which pieces can help produce a better
conversation while preserving the Nemotron timeline contract?

| Prior art | What to inspect | Current use in this program |
| --- | --- | --- |
| [FastFlowLM](https://github.com/ROCm/FastFlowLM) | XDNA2 execution, Whisper/audio path, Linux NPU runtime, model packaging, streaming APIs, static-shape and state limits | NPU speech/runtime prior art; not proof that Whisper can replace FastConformer embeddings |
| [Lemonade Server](https://github.com/lemonade-sdk/lemonade) | backend routing, model recipes, OpenAI-compatible APIs, streaming speech, Kokoro/OpenMOSS, FLM, RyzenAI and custom backend boundaries | possible reusable serving infrastructure, isolated renderer/ASR proof path, or upstream destination |
| [AMD Ryzen AI Software](https://github.com/amd/RyzenAI-SW) / `ryzenai-server` / OGA | ONNX Runtime GenAI, Vitis AI EP, hybrid NPU+iGPU execution, Whisper/Parakeet examples, model packaging | alternate NPU/compiler path; Linux/XDNA2 support and graph coverage must be verified |
| [REM](https://github.com/boxwrench/REM) | bounded streaming jobs, deterministic state around small models, keep-up rate, prefix stability, NPU placement | workload-shape and state-management prior art |
| [`xdna-top`](https://github.com/boxwrench/xdna-top) | NPU context attribution, iGPU busy/power, concurrent traces, contention experiments | required evidence that placement happened and what it costs the iGPU/UMA system |

The study must identify direct reuse, not merely summarize project names. Its
role-by-role result is in [docs/STRIX-SERVING-OPTIONS.md](STRIX-SERVING-OPTIONS.md).

## S3B — Build the serving-options map

Map the functional roles rather than assuming a final component graph:

```text
audio input / perception
conversational core / turn logic
speech output / TTS
codec
auxiliary ASR
AEC / VAD / audio preprocessing
serving / routing
```

For every option classify the move as:

```text
KEEP       existing VoiceChat component or behavior
RELOCATE   same component on another Strix engine
REPLACE    alternate implementation
```

The primary evaluation is fluent conversation: first audible response latency,
80 ms deadline reliability, interruption responsiveness, audio continuity and
underruns, conversation quality, CPU load, gfx1151 headroom, XDNA2 activity,
UMA contention, and package power/thermal behavior.

## S3C — Define replaceable renderer and serving interfaces

The Nemotron timeline remains the behavioral anchor, but the speech renderer is
an explicit replaceable boundary. Runtime source reading shows why this does
not inherently mean replacing the conversational state machine: the sampled
LLM text token is fed back into the next VoiceChat frame, while TTS consumes
that token separately.

```text
Nemotron conversational timeline
          ├── sampled text token -> next VoiceChat frame
          │
          └── push_text(token / delta)
                        │
                        ▼
                 speech renderer
                        │
                        ▼
                 streaming PCM
```

Candidate renderers include native VoiceChat TTS + codec, Lemonade-supported
TTS, Kokoro, OpenMOSS, another AMD-friendly streaming TTS, and a future XDNA2
implementation. The native NVIDIA path is the reference implementation, not a
requirement.

Native TTS state still participates in the current speech lifecycle: the text
channel may run far ahead of audible speech, EOS may be held until speech
settles, quiet/silence state participates in drain completion, and queued text
represents speech that has not yet been heard. The renderer boundary must expose
those lifecycle facts without making the timeline depend on
`voicechat_tts_silence_cos()` or native codec internals.

No renderer or serving integration starts in S3. The isolated proof gate is
cheap, explicit, and must preserve cancellation, streaming, and timeline state.

## S4 — Profile the production-shaped path and rank roles

After PC M4A-2 produces a frozen production-shaped continuous-input
perception runtime, measure the same path on gfx1151 before moving anything.
Then build the full component cost map for:

```text
perception
projector
Nemotron-H backbone
function / turn head
TTS
codec
audio preprocessing
```

For each component record wall time, critical-path contribution, compute
utilization, CPU involvement, memory footprint, frequency per 80 ms frame,
statefulness, tensor shapes, data movement, existing fallback, and whether a
replacement can meet the renderer/timeline contract.

`LEAD-GFX1151-0001` remains parked until this profile exists. The perception
CPU fallback may be a good relocation candidate because it could replace CPU
work with a dedicated accelerator, but that benefit must be measured rather
than assumed.

Rank every role as KEEP, RELOCATE, or REPLACE. The candidate must be evaluated
as a serving configuration, not as an isolated component benchmark.

## S5 — Rank candidate placements and serving configurations

Use this decision gate for every candidate:

```text
                 Critical?
                    │
           ┌────────┴────────┐
          yes               no
           │                 │
Can XDNA finish within       background/auxiliary
frame budget?                candidate
           │
         yes / no
```

Score critical-path reduction, deadline reliability, conversation quality,
XDNA compatibility, conversion effort, statefulness difficulty, memory traffic,
iGPU contention, CPU work eliminated, power/thermal behavior, and expected
latency benefit.

Initial hypotheses, subject to measurement:

```text
KEEP       Nemotron conversational core and timeline behavior
RELOCATE   FastConformer perception if XDNA2 can emit the required embeddings
REPLACE    TTS if an alternate renderer starts sooner, cancels faster, or
           preserves quality better
REPLACE    auxiliary ASR, AEC/VAD, or preprocessing when they improve the
           system without imposing a turn boundary
RELOCATE   serving/routing only when lifecycle and streaming contracts survive
AVOID      moving the 9B conversational backbone merely because it can run
```

The actual winner may be a mixed topology rather than a complete NPU port.

## S6 — Separate perception feasibility from auxiliary ASR

VoiceChat consumes learned FastConformer perception embeddings directly, not
simply an ASR transcript. Therefore:

```text
FastConformer -> XDNA2
```

is a plausible relocation experiment, while:

```text
FastConformer -> Whisper transcript
```

changes the model architecture and requires separate evidence. Whisper through
FastFlowLM/Lemonade is useful now as XDNA2 audio-encoder prior art, proof of the
AMD speech runtime/toolchain, and a possible auxiliary ASR/caption/tool channel
later. It is not a drop-in replacement for perception.

If perception remains attractive after S4, investigate whether AMD's existing
audio encoder machinery can host the VoiceChat FastConformer. This is not a
blanket instruction to port it.

Investigate in order:

1. Can FLM/Lemonade accept a custom audio encoder?
2. What format does their NPU Whisper encoder use?
3. Is there a reusable graph/compiler path?
4. Can the VoiceChat FastConformer operators be represented?
5. What precision is required?
6. What input shapes must be static?
7. Can bounded-lookahead streaming be represented?
8. What are model-load and invocation overheads?

Only convert or implement after these answers are favorable and the candidate
beats the control on an explicit serving objective.

## S7 — Build one bounded heterogeneous serving prototype

The first target is not a fixed component diagram. It is one bounded topology
chosen from the serving-options map, for example:

```text
                 STRIX HALO

MIC -> selected perception / preprocessing path
       │
       ▼
   Nemotron conversational timeline on the selected core
       │ incremental text / speech intent
       ▼
   selected speech renderer -> streaming PCM -> SPEAKER
```

Keep one continuous VoiceChat timeline. Do not turn the system into a
conventional VAD -> ASR -> LLM -> TTS assistant unless evidence shows that it
actually produces the better conversational system. Any NPU is a backend for a
role in VoiceChat, not a reason to discard the learned timeline.

## S8 — Prove the serving topology matters

Once full duplex is stable, record `STRIX-DUPLEX-GPU-M1` and run the same
conversational workload in at least three configurations:

```text
A — native VoiceChat control
B — best bounded KEEP/RELOCATE/REPLACE candidate
C — matched control for the moved role
```

Compare first audible response latency, 80 ms deadline misses, timeline lag,
interruption responsiveness, audio underruns, conversation quality, CPU load,
gfx1151 utilization/headroom, XDNA2 activity, package power/thermal behavior,
and UMA contention. Use `xdna-top` to prove that NPU placement happened and to
observe concurrent iGPU effects.

The question is whether the serving topology makes conversation faster, more
natural, or more interruptible. Equal latency with fewer deadline spikes,
lower CPU load, more GPU headroom, or better power/thermal behavior can be a
material win. NPU utilization alone is not a win.

## S9 — Decide what belongs upstream

This upstream gate may trigger as early as S6. If FastConformer/XDNA
feasibility reveals that the missing piece is a generic XDNA audio-encoder,
compiler, or runtime capability, evaluate the Lemonade/FastFlowLM upstream
implementation path before building a VoiceChat-specific version.

If the work produces a reusable capability such as FastConformer on XDNA2, a
generic audio-encoder backend, a streaming NPU audio path, a heterogeneous
audio/LLM serving primitive, or a new Lemonade backend integration, pause before
making it VoiceChat-specific.

```text
reusable VoiceChat result
        ↓
reveals missing generic AMD capability
        ↓
implement/reuse it in Lemonade/FastFlowLM
        ↓
VoiceChat consumes that capability
        ↓
other AMD applications gain it too
```

Gate:

```text
reusable capability found?

YES -> evaluate Lemonade/FastFlowLM contribution; keep VoiceChat integration thin
NO  -> implement narrowly in the VoiceChat runtime
```

## Branch and repository discipline

```text
boxwrench/llama-voicechat.cpp
    model/runtime mechanics, duplex timeline, backend interfaces,
    and any VoiceChat-specific XDNA integration

boxwrench/Nemotron-VoiceChat-ROCm
    Strix integration, launch/configuration, validation, benchmarks,
    placement research, and the user-facing app

boxwrench/REM
    prior NPU methodology, xdna-top usage, and contention findings

lemonade / FastFlowLM
    upstream candidates for generic XDNA capability
```

Never float Strix against a runtime branch. Pin exact known-good commits.

## Current instruction

Start now:

```text
S3A  Lemonade / FastFlowLM / AMD Ryzen AI / REM prior-art study
S3B  Strix serving-component and KEEP/RELOCATE/REPLACE map
S3C  replaceable renderer and serving-interface definition
```

Wait only for the PC to finish M4A-2 and freeze the production-shaped
continuous-input perception runtime. Then build it on gfx1151, measure the
same perception path, and rank the candidates before any full integration.

Meanwhile PC M4B -> M4C -> M4D may continue independently.

Rules:

- Do not optimize merely to increase NPU utilization.
- Do not investigate `LEAD-GFX1151-0001` or optimize the current CPU fallback
  before the production-shaped profile makes it relevant.
- Do not commit to FastConformer-on-NPU before profiling.
- Do not commit to NVIDIA TTS if a better renderer exists.
- Do not begin a full NPU or replacement-TTS integration during S3.
- Do not break Nemotron into a conventional VAD -> ASR -> LLM -> TTS assistant
  unless evidence shows that it produces the better conversation.
- Do not make Lemonade a mandatory dependency prematurely. Treat it as
  reusable infrastructure, implementation prior art, and a possible upstream
  destination.
- If a generic missing capability appears, evaluate contributing it to
  Lemonade/FastFlowLM before building a permanent VoiceChat-only version.

Success is a materially better fluent VoiceChat serving configuration on Strix
Halo, with measured evidence showing why each role belongs where it is placed.
