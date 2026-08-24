# Nemotron VoiceChat ROCm

Install, reproduce, benchmark, and use NVIDIA NemotronLabs VoiceChat 11B Q8
locally on AMD ROCm/Radeon hardware, working toward a usable spoken
conversational system:

```
microphone -> Nemotron VoiceChat 11B Q8 on AMD -> spoken response
```

## Status

| GPU | gfx target | Status |
| --- | --- | --- |
| AMD Radeon AI PRO R9700 | gfx1201 | VALIDATED -- reference, see [BENCHMARKS.md](docs/BENCHMARKS.md) |
| AMD Radeon RX 7900 XT | gfx1100 | PENDING |
| AMD Strix Halo | gfx1151 | PENDING |

R9700 warm speech-to-speech: mean 4.244 s, p95 4.264 s. Full results:
[research/baselines/R9700-Q8-M1](research/baselines/R9700-Q8-M1/README.md).

## Architecture

This repository does **not** vendor llama.cpp source. It consumes a pinned
commit of the runtime fork:

```
ggml-org/llama.cpp
        ^
sansamour/llama-voicechat.cpp
        ^
boxwrench/llama-voicechat.cpp   (runtime fork: Q8 converters, HIP/backend fixes)
        |
        | pinned / consumed by
        v
boxwrench/Nemotron-VoiceChat-ROCm   (this repo: install, reproduce, benchmark, use)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full picture,
including the planned push-to-talk client/server split.

## Quickstart

```
git clone https://github.com/boxwrench/Nemotron-VoiceChat-ROCm.git
cd Nemotron-VoiceChat-ROCm
scripts/setup.sh
```

`setup.sh` runs, in order: `download-q8.sh` (source GGUF + tokenizer
metadata) -> `build-rocm.sh` (pinned runtime, HIP build) -> `convert-q8.sh`
(split/verify Q8 artifacts) -> `smoke-test.sh` (BUILD/LOAD/STT/TTS/S2S).
See [docs/INSTALL.md](docs/INSTALL.md) for the full, deterministic flow and
what each step verifies.

## Model weights

Model weights are never committed to or redistributed through this Git
repository. Setup scripts download the source GGUF from Hugging Face,
run the Q8 conversion steps against the pinned runtime fork, and verify
output SHA256 hashes. See [docs/MODELS.md](docs/MODELS.md) for provenance
and licensing, which is documented separately from this repository's own
license.

## Roadmap

| Milestone | Scope |
| --- | --- |
| M1 | R9700 Q8 reproducible baseline (in progress) |
| M2 | gfx1100 + gfx1151 validation |
| M3 | persistent push-to-talk |
| M4 | multi-turn conversational state |
| M5 | user-visible latency instrumentation |
| M6 | true live duplex / barge-in |
| M7 | optimize bottlenecks from the actual interactive workload |
| M8 | optional multi-GPU experiments |
| M9 | public AMD VoiceChat release |

## Documentation

- [docs/INSTALL.md](docs/INSTALL.md)
- [docs/MODELS.md](docs/MODELS.md)
- [docs/HARDWARE.md](docs/HARDWARE.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md)
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
