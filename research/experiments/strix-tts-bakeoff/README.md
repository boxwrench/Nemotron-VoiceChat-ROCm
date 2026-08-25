# Strix isolated TTS bakeoff

The first run is retained as `TTS-B0`. Its current interpretation is:

```text
Kokoro: PROMISING
Native vs Kokoro: NOT YET A FAIR PERFORMANCE COMPARISON
```

`TTS-B1` is a separate repeatability and conversational-text-stream study;
it must not overwrite the `TTS-B0` raw JSON, CSV, or logs.

This experiment characterizes speech renderers outside the VoiceChat runtime.
It is a serving-options study, not a replacement-TTS integration.

The current harness compares:

- `native_voicechat`: the VoiceChat TTS+codec driven by `llama-voicechat --say`;
- `kokoro_cpu`: Kokoro V1 through its local Python pipeline.

The native control is the actual VoiceChat renderer, but its current isolated
CLI writes one WAV after TTS frame generation and codec drain. It therefore
does not expose a first-PCM streaming event. The harness records that as
`pcm_streaming: final_file_only`, rather than treating file completion as
equivalent to a production streaming renderer.

Measured fields include:

- first PCM availability and total wall time;
- PCM chunk count, size, audio duration, and inter-chunk gaps;
- audio rate (`audio_seconds / wall_seconds`), reported as a speed ratio;
- drain completion;
- cancellation behavior at the exposed experiment boundary;
- CPU time, approximate CPU utilization, and peak RSS;
- GPU/NPU telemetry availability and any observed host-level limitation.

The fixtures are deliberately text-only and deterministic. No VoiceChat
conversation, model state, microphone input, NPU graph, or alternate renderer
is wired into the product runtime. Generated audio is temporary and ignored by
the experiment directory policy; JSON/CSV/log evidence is durable.

The current native rows are cold-process measurements: each `--say` fixture
loads the full validated runtime before exercising TTS. Kokoro is loaded once
and then measured warm. Those conditions are intentional and documented, but
they are not a controlled cross-renderer latency comparison.

The exact B0 controls are the validated Q8 runtime files:

```text
models/voicechat-q8/runtime/mmproj-voicechat-perception-Q8_0.gguf
models/voicechat-q8/runtime/nemotron_voicechat_11b-stt-llm-Q8_0.gguf
models/voicechat-q8/runtime/voicechat-tts-Q8_0.gguf
```

The native command is the repository's `llama-voicechat --say` path with
`--device none`, `--tts-device CPU`, 16 CPU threads, and a temporary WAV output.
Kokoro is the local V1 model with the `af_bella` voice, loaded once on CPU and
then warmed before fixture measurement. The frozen fixture text and complete
environment snapshot are in `generated/environment.json`.

## Run

From the repository root, use the Kokoro virtual environment already present on
the Strix host:

```bash
../AI-Box/kokoro/.venv/bin/python \
  research/experiments/strix-tts-bakeoff/bench_tts_bakeoff.py
```

The command writes evidence under `generated/` and removes all generated audio
before returning. The native control requires the validated Q8 runtime files
under `models/voicechat-q8/runtime/`; the Kokoro control requires the local
Kokoro V1 model and voice pack.

## Interpretation boundary

This is a CPU-only result if `/dev/kfd` is unavailable to the executing shell.
In that case, the report must not be used as a gfx1151 GPU performance result.
The experiment records that condition, along with the successful native and
alternate CPU renderer behavior, so a later GPU-enabled run can be compared
without changing the fixtures or measurement contract.
