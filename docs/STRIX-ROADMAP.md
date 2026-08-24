# Strix Halo implementation roadmap

The Strix program has two deliberate chapters:

1. Make VoiceChat genuinely good on the gfx1151 GPU path.
2. Turn Strix Halo into a measured heterogeneous GPU/NPU version of the
   project.

The NPU chapter does not replace the first chapter and does not begin with a
blind model-layer port.

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

# Chapter 1 — Fluent GPU VoiceChat

## S1 — Import the proven M4 runtime from the PC

Do not independently implement continuous duplex on Strix. The PC/R9700
remains the development and reference machine for M4.

When the PC reaches a stable milestone covering:

```text
continuous PCM
incremental TTS
simultaneous input/output
native turn taking
interruption
```

freeze the exact `llama-voicechat.cpp` SHA. Strix then pulls that exact runtime
SHA, builds for gfx1151, and runs the identical M4 acceptance suite.

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

# Chapter 2 — Heterogeneous GPU/NPU Strix

## S3 — Map XDNA prior art

Begin only after `STRIX-DUPLEX-GPU-M1` exists. This is a source-study phase,
not implementation.

Study the existing AMD bodies of work with one question: what machinery can be
reused instead of building an XDNA backend from scratch?

| Prior art | Focus |
| --- | --- |
| REM / `xdna-top` | NPU/iGPU concurrency, placement, and contention measurement |
| FastFlowLM | XDNA2 execution, audio/Whisper implementation, Linux NPU libraries, model packaging, static-shape/runtime constraints |
| Lemonade | Backend routing and one serving API over heterogeneous engines |
| `ryzenai-server` / AMD NPU stack | NPU and hybrid execution, ONNX Runtime GenAI, reusable runtime boundaries |

Deliverable:

```text
docs/STRIX-XDNA-PRIOR-ART.md
```

It should map each VoiceChat component to an existing AMD analogue, possible
runtime reuse, likely XDNA fit, and unknowns. At minimum cover:

| VoiceChat component | Existing AMD analogue | Reusable runtime? | Likely XDNA fit | Unknowns |
| --- | --- | --- | --- | --- |
| FastConformer | FLM Whisper encoder | maybe | high interest | model conversion |
| projector | small tensor graph | TBD | possible | transfer overhead |
| 9B backbone | FLM LLM | yes conceptually | poor primary target | latency |
| turn head | small model | TBD | possible | coupling |
| TTS | unknown | TBD | investigate | state |
| codec | audio graph | maybe | possible | benefit |
| AEC/VAD | common NPU workload | likely | good auxiliary fit | later |

## S4 — Build the VoiceChat component cost map

Before moving anything, measure the existing GPU implementation for:

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
statefulness, tensor shapes, data movement, and existing fallback.

This is where `LEAD-GFX1151-0001` becomes relevant. Perception's CPU fallback
may be a good NPU candidate because it could replace CPU work with a dedicated
accelerator, but that benefit must be measured rather than assumed.

## S5 — Rank candidate NPU placements

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

Score critical-path reduction, XDNA compatibility, conversion effort,
statefulness difficulty, memory traffic, iGPU contention, CPU work eliminated,
and expected latency benefit.

Initial hypothesis, subject to measurement:

```text
Tier 1  FastConformer / perception
Tier 2  audio auxiliary models, AEC, VAD, small classifiers/control
Tier 3  projector / codec pieces
Tier 4  TTS pieces if naturally separable
Avoid initially  9B conversational backbone
```

## S6 — Test XDNA perception feasibility

The first serious NPU experiment should determine whether AMD's existing audio
encoder machinery can host the VoiceChat FastConformer. It is not a blanket
instruction to port FastConformer to XDNA.

Investigate in order:

1. Can FLM/Lemonade accept a custom audio encoder?
2. What format does their NPU Whisper encoder use?
3. Is there a reusable graph/compiler path?
4. Can the VoiceChat FastConformer operators be represented?
5. What precision is required?
6. What input shapes must be static?
7. Can bounded-lookahead streaming be represented?
8. What are model-load and invocation overheads?

Only convert or implement after these answers are favorable.

## S7 — Build one bounded heterogeneous prototype

The first target architecture is:

```text
                 STRIX HALO

MIC -> XDNA2 NPU: FastConformer perception
                  │
                  ▼ shared UMA
              gfx1151: Nemotron-H + turn logic
                  │
                  ▼
              gfx1151: TTS / codec -> SPEAKER
```

Keep one continuous VoiceChat timeline. Do not turn the system into separate
NPU-ASR -> GPU-LLM -> GPU-TTS services; the NPU is a backend for a VoiceChat
component, not a replacement architecture.

## S8 — Prove the placement matters

Run the same conversational workload in three configurations:

```text
A — current GPU/CPU reference
B — XDNA perception
C — CPU perception control
```

Compare 80 ms deadline misses, timeline lag, first audible speech, barge-in
response, gfx1151 utilization, CPU utilization, NPU utilization, package
power, UMA pressure, and conversation quality. Use `xdna-top` to prove that
the NPU actually executed the workload.

The question is whether moving perception onto XDNA2 improves fluent
conversation, not whether the NPU produces an attractive standalone benchmark.

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

The gfx1151 Q8 validation milestone is frozen. Do not begin NPU
implementation. The next primary milestone is importing the exact stable M4
runtime produced on the R9700 development machine, validating it on Strix, and
establishing `STRIX-DUPLEX-GPU-M1`.

Do not optimize the current CPU perception fallback before S4. Do not place the
9B conversational backbone on the NPU merely because it can run models. Do
not create a separate ASR -> LLM -> TTS pipeline. Preserve VoiceChat's unified
continuous conversational timeline.

Success is not “NPU utilized.” Success is a materially better fluent VoiceChat
serving configuration on Strix Halo, with measured evidence showing why the
NPU belongs where it is placed.
