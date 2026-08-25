# STRIX-BRINGUP-3 rescope

Status: documentation-only rescope; no production runtime, model, ONNX/XDNA
integration, or compiler test is authorized by this document.

Reference state:

```text
main:       ef4b45eab2b1617b46a7a94c707ed3ba59228d49
Bringup-2:  0fd0e1ec5083e374d5ffce0f002199e72bb08f9c
```

## What the PC work solved

The PC reference-runtime work closed several questions that Bringup-3 had
been treating as open:

```text
future audio is not required: zero-lookahead is promoted
naive growing-prefix re-encode misses 80 ms around 20–24 seconds
no tested short history window has both timing margin and downstream fidelity
downstream correctness, not embedding cosine alone, is the fidelity gate
GPU function-head execution is promoted with exact CPU parity
incremental codec decode and streaming ISTFT are numerically correct
codec graph reconstruction/reuse is not the synchronous bottleneck
```

These are reference findings, not new Strix experiments.

## Redundant Strix work

Do not repeat the M4 lookahead sweep, rediscover the growing-prefix curve,
choose a bounded window by intuition, rebuild the function head, or re-prove
codec/ISTFT math. Do not treat the old `onnx` import failure as an XDNA or
project-level compatibility result.

## What can be reused directly

```text
PC timing and downstream-fidelity methodology
GPU function-head runtime behavior and parity evidence
M4B incremental codec/ISTFT evidence and the release post-turn renderer
hardware-neutral benchmark harnesses
Bringup-3 source-derived graph contract and operator inventory
GGUF tensor extraction and AMD Parakeet correspondence notes
host-side XDNA context/submission telemetry probes
```

The current Bringup-3 ONNX exporter, local export environment, generated
graphs, and runtime parity instrumentation are preserved as uncommitted WIP.
They are preparation only and are not production contracts.

## Exact contracts Strix needs from D1/D2/D3

### D1 — async renderer

```text
queue item and text/delta acceptance unit
PCM format, chunk ownership, and output scheduling
backpressure policy and queue bounds
started/speaking/settled-drained events
fast playback cancellation semantics
cancel versus destructive renderer/model reset
handling of accepted but not-yet-spoken text after interruption
worker/thread ownership and first-audio timing
```

### D2 — production continuous perception

```text
exact runtime SHA and model/projector hashes
past-context strategy: cache, bounded history, or other proven mechanism
cross-call state/cache lifecycle and reset semantics
input layout and exact tensor shapes
invocation cadence and output-frame selection
maximum useful context/prefix rule
per-call latency curve, p50/p95/p99, and 80 ms deadline behavior
downstream correctness/fidelity evidence
memory, transfer, and CPU/GPU cost of the chosen path
```

### D3 — continuous causal timeline

```text
capture-to-tick causality rule and timeline cadence
input backlog and lag behavior
incremental text/speech handoff to D1
turn-boundary and interruption events
stale response cancellation and conversation continuity/reset rules
```

## XDNA work independent of D2

Safe work is limited to facts that survive any reasonable D2 result:

```text
exact VoiceChat operator inventory
GGUF tensor naming, shapes, and extraction
AMD Parakeet/XDNA operator correspondence
SiLU lowering/support research
causal/asymmetric Pad semantics
Linux custom-graph toolchain discovery
reproducible host compiler probes
XDNA process/context/submission telemetry methodology
```

Do not freeze a prefix length, bounded window, cache scheme, final ONNX input
shape, or full FastConformer XDNA integration before D2.

## Strategic Bringup-3 status

```text
EXPORT TOOLING                 not a production dependency; local experimental WIP is preserved
XDNA COMPILER COMPATIBILITY   not yet tested
PRODUCTION XDNA PERCEPTION    DEFERRED pending the D2 contract
```

The earlier export-environment blocker was an experiment setup state, not an
XDNA rejection. The current strategic blocker is contract timing: D2 is
specifically solving the past-context/caching problem that determines the
production invocation shape.

## Reference-runtime reproducibility issue

The current main documentation names post-turn streaming playback runtime
SHA `5e5b8628cf5db8e18b61fa8eb8a12fb80d68f79d`, but that object is not
available in the local runtime checkout, any local branch/ref or reflog, or
the public runtime remote. The runtime checkout has no reachable commit newer
than the known GPU function-head commit
`a05335bb37b4819e6802efe831cbbee3e584f50b`.

This is a reference-runtime synchronization/reproducibility issue, not
evidence against XDNA or FastConformer. No replacement SHA has been found, so
the Strix branch does not change the pin or claim a corrected runtime. The
reference-runtime owner must resolve whether the documented commit was never
pushed, rewritten, or recorded incorrectly before a fresh consumer build can
reproduce the post-turn playback contract.

## Single best Strix task while D1/D2 proceed

Keep Bringup-3 D2-neutral: finish the source-derived graph/operator and AMD
reuse evidence, keep the host compiler/telemetry probes reproducible, and
prepare a bounded subgraph experiment without selecting a production shape.
When D2 freezes, build that exact runtime SHA on gfx1151, profile it first,
and only then choose the first XDNA intervention.
