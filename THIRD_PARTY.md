# Third-party components

This document covers licensing of components this repository depends on or
distributes evidence about. It is separate from [LICENSE](LICENSE), which
covers only the source code in this repository (scripts, docs, harness code).

## Runtime

This repository does not vendor or redistribute the inference runtime. It
consumes a pinned commit of:

- [boxwrench/llama-voicechat.cpp](https://github.com/boxwrench/llama-voicechat.cpp)
  (fork of [sansamour/llama-voicechat.cpp](https://github.com/sansamour/llama-voicechat.cpp),
  itself based on [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)),
  MIT licensed.

See [runtime/README.md](runtime/README.md) for the exact pinned commit.

## Model weights

Model weights are never committed to or redistributed through this
repository. See [docs/MODELS.md](docs/MODELS.md) for the source, provenance,
and license of NVIDIA NemotronLabs VoiceChat 11B, which is documented there
rather than in this file since it governs a separate artifact under a
separate license from this repository's own code.

## Benchmark corpus

The fixed benchmark corpus recorded under `research/corpus/` and the raw
per-turn evidence under `research/baselines/` are original test inputs and
measurement data produced for this project. See
`research/corpus/README.md` for details.
