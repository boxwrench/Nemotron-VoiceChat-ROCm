# gfx1151 validation

## Result

**VALIDATED — with known perception/CLIP CPU fallback** — the Q8 model builds,
loads, transcribes, synthesizes speech, and completes the full
speech-to-speech cases on gfx1151. The fallback is a performance qualification,
not a compatibility or correctness failure.

## Environment

- Host: AMD Ryzen AI MAX+ 395, 16 cores / 32 threads, 121 GiB system memory.
- GPU: AMD Radeon 8060S / `AMD Radeon Graphics`, `gfx1151`; UMA memory pool
  reported by ggml as 114,688 MiB.
- OS: Linux Mint 22.3, kernel `6.17.0-35-generic`.
- ROCm: 7.2.2; HIP `7.2.53211-671d39a71e`; AMD clang 22.0.0git.
- Runtime source: llama.cpp commit `38a76719e`, Release build with
  `GGML_HIP=ON`, `GPU_TARGETS=gfx1151`.
- Runtime settings: `--no-warmup`, `GGML_CUDA_DISABLE_GRAPHS=1`,
  `VC_NO_BARGE=1`, `VC_FORCE_BOS=1`, `--device ROCm0`,
  `--split-mode none`, `--gpu-layers all`, temperature 0, seed 42.

The captured inference processes were GPU-visible: their logs show
`ggml_cuda_init` finding gfx1151 and the runtime using `ROCm0`. The resumed
automation shell used for the final documentation pass did not expose
`/dev/kfd`, so it could not perform a new live inference run; that shell
limitation did not affect the BUILD/LOAD/STT/TTS/S2S artifacts recorded here.

## Validation evidence

- **BUILD: PASS.** `cmake --build build/hip-gfx1151 --target llama-voicechat`
  completed successfully.
- **Q8 LOAD: PASS.** The run found one ROCm device at `gfx1151`; the function
  head loaded, the TTS side loaded 559 tensors / 1083 MiB plus a 1210 MiB KV
  cache, and the service reached `ready`.
- **STT: PASS.** VC01 returned “The capital of France is Paris.”; VC02 returned
  the Rayleigh-scattering explanation; VC03 produced the long itinerary; VC04
  matched the clean factual answer; VC05 returned the rainbow explanation.
- **TTS: PASS.** Every case produced a non-empty mono 22050 Hz PCM WAV. The
  measured VC01 outputs were 3.92 seconds and byte-identical across all five
  measured turns.
- **FULL S2S: PASS.** All five cases exited 0 and produced text plus audio:

  | Case | Input | Output | Mean total | Mean S2S/output |
  | --- | ---: | ---: | ---: | ---: |
  | VC01 short factual | 2.96 s | 3.92 s | 6.342 s | 1.618 |
  | VC02 conversational | 4.24 s | 15.92 s | 17.356 s | 1.090 |
  | VC03 long constrained | 22.40 s | 68.48 s | 108.758 s | 1.588 |
  | VC04 fixed noise | 2.96 s | 3.92 s | 6.608 s | 1.686 |
  | VC05 internal pause | 7.94 s | 8.48 s | 15.086 s | 1.779 |

## Memory and latency

This is a unified-memory result, not a discrete-VRAM result. During the VC01
observer run, sysfs VRAM usage stayed near 0.156 GiB while GTT usage rose from
48.1 GiB to a 63.3 GiB peak; process RSS peaked at 2.92 GiB. The service logs
reported roughly 57.9 GiB free from the 114,688 MiB UMA pool after loading.
Peak sampled GPU busy was 91%.

Against the R9700 Q8 reference, the primary VC01 mean was 6.342 s versus
4.244 s: about **1.49x slower** (+49.5%). First text was 3.019 s versus 2.088 s
(+44.6%); the S2S/output ratio was 1.618 versus 1.083. These are **preliminary
observations**, not a controlled architectural comparison.

This comparison is intentionally rough: the current ignored corpus has the
same VC01 duration but different audio hashes from the frozen R9700 manifest,
and VC03 is 22.40 seconds here versus 28.24 seconds in that manifest. The
VC01 duration and prompt are aligned; the secondary cases should not be read as
controlled cross-machine comparisons.

The logs explicitly report:

- a 126.09 MiB CPU compute buffer for the CLIP graph;
- unsupported ROCm CLIP operators including depthwise `CONV_2D_DW` and
  repeated `UNARY` operations.

The main model and TTS report `ROCm0` as their device. No separate CPU fallback
for the main LLM or TTS path was observed. The CPU backend/scheduler is used for
part of perception because the CLIP graph does not have complete ROCm coverage.

## Known qualification and parked lead

Known qualification:

- The gfx1151 perception/CLIP graph reports unsupported depthwise
  `CONV_2D_DW` operations.
- It also reports repeated unsupported `UNARY` operations.
- The runtime reserves a 126.09 MiB CPU compute buffer for this graph.

Parked lead `LEAD-GFX1151-0001`:

- Observation: gfx1151 perception reports unsupported ROCm operators and uses
  CPU backend/scheduler support.
- Question: how much served latency comes from those perception fallbacks?
- Prediction: restoring HIP coverage would reduce S2S latency if fallback work
  accounts for a meaningful fraction of execution time.
- Status: parked pending a matched workload/profile. No optimization or
  investigation was performed in this validation pass.

Not yet assessed:

- human listening quality;
- matched cross-GPU performance;
- function-channel behavior.

## Files

- Modified: this README and the root `.gitignore`.
- Created: `research/scripts/harness/bench_voicechat.py`; it discovers a single
  `build/*/bin/llama-voicechat` binary or accepts `--binary`, and discovers a
  usable DRM sysfs device or accepts `--sysfs-device`, so it has no gfx1151 or
  gfx1201-specific assumptions.
- Created: `research/hardware-validation/gfx1151/corpus-manifest.tsv` with the
  actual input hashes and durations used by this pass.
- Created: `research/hardware-validation/gfx1151/model-manifest.tsv` and
  `environment.txt` with relative-path model hashes, hardware, software,
  runtime settings, and the final classification.
- Created: `research/hardware-validation/gfx1151/build.txt` with the path-free
  build command, configuration, source revision, and successful result.
- Created: `research/hardware-validation/gfx1151/generated/.gitignore`; WAV
  outputs remain locally available but are excluded from validation commits.
- Created: `research/hardware-validation/gfx1151/generated/VC01/` through
  `VC05/`, containing `ready.json`, `cold-load-ms.txt`, `raw-runs.csv`,
  `summary.csv`, `service.stderr.txt`, and `telemetry.csv` for VC01. Generated
  `turn-*.wav` files are ignored by policy.
- Scratch WAVs at the repository root (`test_vc01.wav`, `test_turn0.wav`, and
  `test_out_s2s.wav`) are ignored and will not be committed.
- Ignored build/model assets created under `build/` and `models/`; these are not
  source changes.

Proposed commit message:

```text
research: validate VoiceChat Q8 on gfx1151
```

This validation pass stops here. No optimization or product work was started,
and no commit has been made.
