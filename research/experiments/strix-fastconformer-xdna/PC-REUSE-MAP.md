# PC-to-Strix reuse map

Status: D1/D2/D3 contract audit against `origin/main` at
`ef4b45eab2b1617b46a7a94c707ed3ba59228d49`.

The PC/R9700 is the reference-runtime track. Strix should import stable
behavioral and execution contracts rather than re-run PC feasibility work.
The current durable runtime references are:

```text
GPU function head: a05335bb37b4819e6802efe831cbbee3e584f50b
release runtime pin: 5e5b8628cf5db8e18b61fa8eb8a12fb80d68f79d
main roadmap:       ef4b45eab2b1617b46a7a94c707ed3ba59228d49
```

The release pin is documented by `runtime/README.md`; the local runtime
checkout is older and the release object is not present in this clone. The
documented contract is still sufficient for this audit.

## Runtime pin reproducibility audit

The documented post-turn streaming-playback pin is currently not
reproducible from the available runtime checkout or its public remote:

```text
documented pin: 5e5b8628cf5db8e18b61fa8eb8a12fb80d68f79d
local runtime object: absent
local branch/ref containing it: none
local reflog match: none
public fetch by object ID: rejected as not found
latest reachable related commit: a05335bb37b4819e6802efe831cbbee3e584f50b
replacement SHA: not found
```

Best classification from this checkout: **unresolved/orphaned or
misdocumented public pin**. The evidence rules out a locally available
post-turn commit, but cannot distinguish an unpublished PC-only commit from a
rewritten or incorrectly recorded commit without the authoritative PC
runtime checkout. Strix must not invent or substitute a runtime SHA. Resolve
the pin with the reference-runtime owner before treating the D1/D2/D3
contracts as consumable build inputs.

## ALREADY SOLVED

These are settled reference-runtime findings. Do not duplicate them on Strix
unless a later hardware-specific measurement gives a concrete reason.

| Question/role | PC result | Strix disposition |
|---|---|---|
| Future lookahead | M4A-1 promoted zero-lookahead: future audio did not materially change perception embeddings across adversarial cases | Import the zero-future-lookahead invariant; do not repeat a lookahead sweep |
| Growing-prefix behavior | M4A-2 characterized naive re-encode as missing the 80 ms budget around 20–24 s | Treat as a known failure mode; wait for D2's bounded/cached contract |
| Bounded historical windows | M4A-3 found no tested short window with both timing margin and reliable downstream fidelity | Do not choose a Strix window by embedding cosine or intuition |
| Perception timing harness | PC harness measured service time against the live 80 ms deadline | Reuse the measurement shape after D2; do not recreate the feasibility experiment |
| Downstream fidelity methodology | Downstream correctness disproved embedding-only shortcuts: a 1 s window failed while a 2 s window succeeded despite lower cosine similarity | Preserve task/output correctness as the gate for any Strix perception change |
| Function head | `a05335bb3` promoted GPU function-head projection with 909/909 exact token parity and about 9.7 ms/frame recovered | Import the exact runtime behavior/flag when the runtime SHA is consumed; do not reimplement it |
| Incremental codec decode | M4B proved causal codec decode can produce correct incremental PCM; aggregate throughput is above realtime | Use the proven path as the native control |
| Streaming ISTFT | M4B proved waveform fidelity (`0.9999999–1.0` correlation), no underruns, and no missing final word | Import the release/runtime implementation; do not re-prove codec mathematics |
| Codec graph-reuse hypothesis | M4B-2 killed graph reuse as the bottleneck: compute, not build/alloc, dominated the 593-node graph | Do not spend Strix effort on the dead graph-rebuild explanation |

## IMPORT WHEN CONTRACT FREEZES

These PC results are reusable, but the exact runtime/interface must be frozen
before Strix consumes them.

| Role | Contract Strix needs | Current source |
|---|---|---|
| TTS lifecycle | Accepted text may lead audible speech; renderer must expose started/speaking/settled, cancellation, drain, and reset semantics | `docs/STRIX-SERVING-OPTIONS.md`; D1 will make the async queue concrete |
| Renderer scheduling | Queue item/granularity, PCM format, worker ownership, backpressure, cancellation and drain behavior, and first-audio timing | D1 async native renderer |
| Incremental native PCM | Exact release runtime SHA and opt-in streaming-playback behavior; distinguish post-turn streaming from live duplex | `runtime/README.md`, M4B evidence, release pin |
| Live timeline | Tick cadence, capture boundary, causal lag rule, text/speech handoff, interruption and stale-output cancellation | D3 continuous causal timeline protocol |
| Perception timing | Same workload and metrics: p50/p95/p99 service time, deadline misses, backlog and duration curve | D2 production-shaped perception harness |
| Function head | Runtime SHA, enablement flag, parity evidence and fallback behavior | `a05335bb3` / release runtime lineage |

### Exact D2 handoff required by Strix

Before the first production-shaped FastConformer/XDNA experiment, D2 must
provide all of the following, not merely a branch name:

```text
exact llama-voicechat.cpp runtime SHA
model/projector artifact identifiers and hashes
zero-lookahead guarantee and any future-context assumptions
past-context strategy: cache, bounded history, or another proven mechanism
cache/state lifecycle, reset behavior, and cross-call tensor semantics
input feature layout and exact tensor shapes
invocation cadence and output-frame selection
maximum useful context/prefix rule
per-call service-time curve at representative conversation durations
mean/p95/p99 latency and 80 ms deadline results
downstream correctness/fidelity evidence, not embedding similarity alone
CPU/GPU memory and transfer behavior relevant to the serving path
```

Strix should build the exact SHA on gfx1151 and compare the same path before
choosing KEEP, RELOCATE, or REPLACE for perception.

## STILL STRIX-SPECIFIC

These questions were not answered by the PC work and remain valid Strix
research:

| Area | What remains specific to Strix |
|---|---|
| FastConformer graph representation | Map the authoritative VoiceChat GGUF tensors and source graph to an XDNA-friendly representation without changing learned embeddings |
| XDNA compiler path | Establish Linux custom-graph compile/load capability and identify the first unsupported VoiceChat pattern |
| Operator correspondence | Test causal/asymmetric Pad, depthwise Conv, channel LayerNorm, explicit/per-layer SiLU, relative/chunked attention, and layout transforms against AMD Parakeet prior art |
| XDNA placement/telemetry | Prove process ownership, submissions/completions, NPU utilization, iGPU contention, UMA pressure, power and thermal effects on this Strix |
| Strix service topology | After D2 and a continuous control exist, measure whether KEEP/RELOCATE/REPLACE improves fluent conversation rather than merely increasing NPU utilization |
| Alternative renderers | Only isolated characterization remains valid before D1; no replacement TTS integration is implied by the PC codec result |

## Explicitly removed duplicated work

The following Strix experiments are now redundant:

```text
future-lookahead sweep
naive growing-prefix feasibility sweep as a new discovery
choosing an arbitrary short bounded window
embedding-cosine-only perception acceptance
independent GPU function-head implementation
codec graph-rebuild optimization as the first explanation
re-proving incremental codec/ISTFT numerical correctness
```

Their scripts and evidence remain historical/reusable references. They are not
new Strix milestones.

## Reusable files and boundaries

```text
research/experiments/m4a-1-lookahead-spike/  historical lookahead evidence
research/experiments/m4b-streaming-audio/    codec/ISTFT evidence and timing
research/scripts/harness/bench_voicechat.py shared hardware-neutral harness
research/experiments/strix-fastconformer-xdna/ graph/operator/host-probe WIP
```

The Bringup-3 graph contract, GGUF extraction, operator inventory, AMD reuse
map, and host telemetry probes remain useful. Fixed static dimensions, cache
semantics, and a full FastConformer export do not become production decisions
until D2 freezes them.
