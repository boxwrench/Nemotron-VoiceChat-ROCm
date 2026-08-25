# M4A-2 handoff boundary for Strix

This document records what the Strix prototype will need from the PC before
the first real FastConformer relocation experiment. It does not choose a new
perception shape and does not implement an XDNA path.

## Already invariant

These properties are not waiting on M4A-2:

```text
zero future lookahead
12.5 Hz VoiceChat timeline / 80 ms frame cadence
learned FastConformer perception embedding is required
Nemotron consumes that embedding directly
no transcript substitution
current encoder has no useful cross-call state
```

The last item means a future accelerator path must reproduce the selected
production execution contract; it must not invent state semantics merely to
make the graph look streaming.

## M4A-2 variables

M4A-2 must freeze the values that determine the actual serving cost:

```text
prefix or window length at each invocation
invocation cadence and scheduling relationship to the 80 ms timeline
input tensor shape(s), including padding/masking
output frame selection and embedding layout
maximum useful conversation prefix
acceptable per-call and p95 perception budget
re-encode policy and any bounded queue behavior
```

The leading hypothesis is zero-lookahead/growing-prefix re-encode, but this
document deliberately does not turn that hypothesis into an XDNA requirement.
The eventual runtime SHA is authoritative.

## Required frozen-runtime handoff

When the PC completes M4A-2, record:

```text
exact llama-voicechat.cpp commit SHA
build/configuration and backend flags
perception model and projector artifact identifiers
input sample rate, mel dimensions, and normalization
per-call input shape(s)
embedding output tensor shape, dtype, and frame selection
invocation cadence and timing budget
fixture/corpus and expected embedding comparison method
known CPU/GPU fallback operators
```

The handoff must include a production-shaped executable path, not only a
microbenchmark or an offline embedding dump. Strix can then build the same
runtime SHA on gfx1151 and measure the same perception path before choosing
KEEP, RELOCATE, or REPLACE.

## First bounded experiment after the handoff

The first implementation should be exactly one A/B measurement:

```text
A — frozen production-shaped perception path on the Strix control
B — the same input/output contract through one converted static XDNA candidate
```

Keep Nemotron, turn logic, TTS, and codec unchanged. Measure embedding
agreement/tolerance, first-embedding latency, p95 80 ms deadline behavior,
queue/transfer cost, CPU/gfx1151/XDNA activity, UMA contention, and power.
Use `xdna-top`/XRT evidence to prove placement. Stop if the candidate cannot
preserve the learned embedding contract or does not show a measured serving
benefit.
