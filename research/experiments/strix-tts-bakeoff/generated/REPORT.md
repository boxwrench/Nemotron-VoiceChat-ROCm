# STRIX-TTS-BAKEOFF-CPU-001

Kokoro: **PROMISING**
Native vs Kokoro: **NOT YET A FAIR PERFORMANCE COMPARISON**

CPU-only isolated renderer characterization. This is not a gfx1151 GPU performance result: `/dev/kfd` was absent in the executing shell.
Native rows are cold-process full-runtime measurements; Kokoro is warm-loaded once. The timing table is behavioral evidence, not a controlled cross-renderer latency ranking.

| renderer | fixture | first PCM (ms) | wall (s) | audio (s) | speed ratio | PCM chunks | CPU (% of one core) | peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native_voicechat | short | 4080.6 | 4.223 | 4.000 | 0.95x | 1 | 950.3 | 13053.7 |
| native_voicechat | medium | 9906.7 | 10.051 | 12.960 | 1.29x | 1 | 1190.5 | 13057.9 |
| native_voicechat | long | 17972.2 | 18.117 | 24.480 | 1.35x | 1 | 1269.0 | 13063.2 |
| kokoro_cpu | short | 342.5 | 0.343 | 3.575 | 10.44x | 1 | 3050.8 | 2554.9 |
| kokoro_cpu | medium | 547.5 | 1.146 | 14.325 | 12.50x | 2 | 1681.6 | 2745.1 |
| kokoro_cpu | long | 306.1 | 2.112 | 28.250 | 13.38x | 3 | 1415.8 | 3010.6 |

## Behavioral observations

- Native VoiceChat TTS generated incremental 80 ms model frames, but the isolated `--say` path exposed one final WAV write. It did not expose first PCM, playback cancellation, or a renderer-level drain event.
- Kokoro produced punctuation-delimited PCM chunks and exhausted its generator cleanly. Its cancellation measurement is only Python generator close; it does not prove cancellation of an in-flight accelerator operation.
- GPU utilization, NPU utilization, package power, and thermal behavior were not measured because `/dev/kfd` and render nodes were not visible to this shell. `amd-smi` could enumerate the card but did not provide live utilization in this context.
- Native measurements include cold full-runtime startup for each fixture; native RSS includes the LLM/projector. Kokoro was warm-loaded once. Do not compare these rows as a formal renderer speed ranking.
- No VoiceChat runtime integration, NPU implementation, alternate-TTS integration, or subjective listening evaluation was performed.

## Next gate

Repeat the unchanged fixture/measurement contract in a shell with `/dev/kfd` and render-node access before making a Strix GPU-serving decision. Keep renderer contract work separate from the VoiceChat runtime until cancellation, accepted-text buffering, and speech drain are specified.
