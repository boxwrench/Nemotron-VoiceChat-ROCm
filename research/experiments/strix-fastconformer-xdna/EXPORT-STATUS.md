# VoiceChat FastConformer export status

## Current strategic status after the PC roadmap reset

```text
EXPORT TOOLING                 not a production dependency; local WIP preserved
XDNA COMPILER COMPATIBILITY    not yet tested
PRODUCTION XDNA PERCEPTION     DEFERRED pending PC D2
```

The local exporter and generated graphs are preserved as preparation. They are
not evidence of XDNA compiler compatibility and do not select a production
prefix, cache, or invocation shape. The D2 contract must be imported before a
production-shaped graph is frozen.

The earlier setup result below is retained as experiment history, not as the
current strategic blocker.

## Current result

```text
full production-shaped export     DEFERRED pending D2
real S1/S2 export WIP             preserved; CPU graph checks completed
XDNA compiler handoff             NOT TESTED
model weights in repository       no (local source GGUF is outside Git)
ggml -> ONNX production serializer no
```

The pinned runtime constructs the perception graph directly in ggml from GGUF
weights. The local direct exporter WIP uses the runtime's `GGUFSource`; it does
not use NeMo, PyTorch, or `torch.onnx`. No XDNA compiler/provider handoff was
attempted, and no production input shape was selected.

The first generator attempt was run with:

```text
python3 make_subgraphs.py --out /tmp/strix-fastconformer-xdna-generated
result: BLOCKED — No module named 'onnx'
```

This is an export/toolchain blocker, not evidence that XDNA cannot execute the
graph. The exact contract and operator inventory are source-derived and are
ready for a host-side graph preparation run.

## Representative subgraphs

`make_subgraphs.py` defines three static, VoiceChat-shaped subgraphs:

```text
FC-SUBGRAPH-1   causal Pad + pre-encode depthwise Conv2D
FC-SUBGRAPH-2   LayerNorm + GLU + causal depthwise Conv + LayerNorm + SiLU
FC-SUBGRAPH-3   relative-position attention with external F32 causal mask
```

They use provisional prefix `T=160` mel frames (`N=21` encoded frames) only
for compiler feasibility. Generated ONNX files are ignored and must remain
outside commits.

## Host handoff

Run `host_compile_probe.sh` from a normal host shell with an environment that
provides `onnx` and an ONNX Runtime VitisAI/XDNA provider. It prints versions,
generates the subgraphs into an explicit output directory, creates sessions to
trigger compilation, and records provider/fallback/profile information.

Run `host_run_probe.sh` only after compilation succeeds. It runs the same
static graphs, records output shapes and latency, and captures XRT/`xdna-top`
observations. It does not feed embeddings to Nemotron.
