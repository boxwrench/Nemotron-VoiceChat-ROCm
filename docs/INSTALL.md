# Install

Deterministic setup flow, orchestrated by `scripts/setup.sh`:

```
1. scripts/build-rocm.sh    clone boxwrench/llama-voicechat.cpp at the pinned
                             commit (see runtime/README.md) and build it with
                             HIP for the target GPU (GPU_TARGETS=gfx1201, ...)

2. scripts/download-q8.sh   fetch the source GGUF and tokenizer metadata from
                             Hugging Face; see docs/MODELS.md for provenance

3. scripts/convert-q8.sh    run the runtime repo's four Q8 conversion steps
                             (tools/voicechat/convert_voicechat_*.py) and
                             verify the four output SHA256 hashes

4. scripts/smoke-test.sh    BUILD / LOAD / STT / TTS / full S2S pass or fail,
                             no benchmarking

5. scripts/benchmark.sh     optional: run the fixed corpus and reproduce the
                             numbers in docs/BENCHMARKS.md
```

Each step is idempotent and can be re-run independently once its
prerequisites exist on disk.

This mirrors the exact protocol already recorded, by hand, in
`research/baselines/R9700-Q8-M1/commands.txt` and
`research/baselines/R9700-Q8-M1/artifact-hashes.txt` -- these scripts
formalize that protocol instead of replacing it.

## Requirements

- ROCm/HIP toolchain matching the target GPU (see docs/HARDWARE.md)
- Sufficient VRAM for Q8 (R9700 reference peak: 14.45 GiB)
- Hugging Face access to the NemotronLabs VoiceChat 11B source weights

<!-- TODO: fill in exact package/toolchain versions once build-rocm.sh exists -->
