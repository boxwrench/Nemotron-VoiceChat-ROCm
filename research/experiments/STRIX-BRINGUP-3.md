# STRIX-BRINGUP-3

Status: **PAUSED / RESCOPED** after the PC D1/D2/D3 roadmap reset; no
VoiceChat integration. See [STRIX-BRINGUP-3-RESCOPE.md](STRIX-BRINGUP-3-RESCOPE.md).

The graph contract, operator inventory, AMD reuse map, exporter scaffolding,
and host probes remain valuable WIP. The production XDNA perception experiment
is deferred until PC D2 freezes the past-context/caching contract.

## Final classification

```text
STRIX-BRINGUP-3              PAUSED / RESCOPED
XDNA-LINUX-M0               PASS
XDNA-SPEECH-LINUX-M0        PASS
VoiceChat graph contract    CHARACTERIZED
operator inventory          CHARACTERIZED
AMD reuse potential         MEDIUM / RELATED_PATTERN
ONNX/export preparation     AVAILABLE WIP
XDNA compiler compatibility NOT TESTED
production XDNA perception  DEFERRED pending D2
runtime pin 5e5b8628...     UNRESOLVED / unreachable
last reachable related SHA  a05335bb37b4819e6802efe831cbbee3e584f50b
```

No XDNA compiler has rejected the VoiceChat graph. The pause is strategic,
not a hardware or compiler failure. Wake-up requires either the D2 contract
or repair and public verification of the runtime pin.

## Decision gate

```text
EXPORT TOOLING                 not a production dependency; experimental WIP preserved
XDNA COMPILER COMPATIBILITY    not yet tested
PRODUCTION XDNA PERCEPTION     DEFERRED pending PC D2
graph contract                 COMPLETE from pinned runtime source
operator inventory             COMPLETE; machine-readable JSON included
AMD reuse map                  COMPLETE with pin/tree provenance warning
host compiler/session probe    PREPARED for a normal host shell
```

Known Linux XDNA LLM and speech execution already passed in Bringup-2. The
earlier export environment limitation was only experiment setup; the
strategic reason to pause is now the D2 production contract:

```text
the D2 contract must select the past-context/cache and invocation semantics
before a production-shaped ONNX/XDNA graph is frozen
the XDNA compiler path has not yet been tested for this exact graph
the requested AMD pin predates the Parakeet-TDT files
```

The last item is recorded rather than silently resolved: AMD commit
`0b65628f1caacf0fbe3fd2cb4ed6bae0437a4155` does not contain the requested
Parakeet files. The later reference tree inspected for transform provenance is
`43b2dabe4d1bf084d0421953b134707b8cb7275a`.

## Graph contract

See [GRAPH-CONTRACT.md](strix-fastconformer-xdna/GRAPH-CONTRACT.md). The exact
source-derived contract is:

```text
128-bin log-mel input, no per-feature normalization
24 FastConformer layers
width 1024, 8 heads, head width 128
FFN width 4096, SiLU, two macaron branches
causal 3-stage stride-2 downsampling: 128 -> 17 frequency bins
causal depthwise convolution, kernel 9, left context 8
LayerNorm convolution normalization
chunked_limited attention [70 left, 0 right], chunk width 1
one 1024-wide encoder embedding per served frame
one 4480-wide projected embedding per served frame
```

D2 still controls the production prefix/window, cadence, output-frame
selection, cross-call state/cache semantics, and service budget. The spike's
static shapes are compiler probes, not serving architecture.

## Operator inventory

See [OPERATOR-INVENTORY.json](strix-fastconformer-xdna/OPERATOR-INVENTORY.json).
The known fallback mapping remains:

```text
CONV_2D_DW    RELATED_PATTERN to AMD Pad->depthwise-Conv work
UNARY x24     per-layer convolution-module SiLU; not AMD's mask issue
```

## Subgraphs and probes

`make_subgraphs.py` prepares four ignored ONNX files when the host has the ONNX
package:

```text
FC-SUBGRAPH-1   Pad + depthwise Conv, plus the AMD-style fused control
FC-SUBGRAPH-2   LayerNorm + GLU + causal depthwise Conv + SiLU
FC-SUBGRAPH-3   relative-position attention with external F32 causal mask
```

The provisional shape is `T=160` mel frames and `N=21` encoded frames. It is
deliberately not the D2 production shape.

## PC synchronization

The runtime remote was checked at the end of this spike:

```text
origin/amd/rocm           5cc03186ab7db2c61efce2c3f3ce9455c8a70318
local runtime checkout    38a76719e2b31a4dfc574bf750bb9ad44c434b81
D2 production contract    not published
```

No D2 production perception SHA or contract appeared remotely. The new
`perf/m4-function-head-gpu` branch is unrelated to the production perception
contract and was not integrated.

The documented post-turn playback pin `5e5b8628cf5db8e18b61fa8eb8a12fb80d68f79d`
is also unreachable from the available runtime checkout and public remote.
The latest reachable related runtime SHA is `a05335bb37b4819e6802efe831cbbee3e584f50b`.
See [PC-REUSE-MAP.md](strix-fastconformer-xdna/PC-REUSE-MAP.md) for the audit.

`host_compile_probe.sh` and `host_run_probe.sh` use explicit output
directories, record tool/provider versions, create sessions through the
VitisAI EP when available, capture ONNX Runtime profiles, and collect
`xrt-smi`/`xdna-top` evidence. They never modify system packages or call
Nemotron.

## Classification and next experiment

```text
XDNA-FC-COMPILE-M0: DEFERRED — not yet a production experiment
XDNA compiler compatibility: NOT TESTED
recommended next sequence: import D2 -> finish ONNX representation ->
  ggml/CPU parity -> XDNA compile -> host NPU execution -> compare against
  the gfx1151 production control
```

Once D2 freezes the production contract, the first bounded comparison is
FC-SUBGRAPH-1 original Pad plus depthwise Conv versus the AMD-style fused Conv.
Then test the LayerNorm/SiLU module and attention mask separately. Do not begin
a production-shaped VoiceChat export or feed any output to Nemotron until the
exact D2 shape and semantics are known.
