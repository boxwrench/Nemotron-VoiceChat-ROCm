# STRIX-TTS-BAKEOFF-TTS-B1

Kokoro remains **PROMISING** as an external renderer candidate. Native vs Kokoro remains **NOT YET A FAIR PERFORMANCE COMPARISON**.

This run is CPU-only because `/dev/kfd`, DRM render nodes, and `/dev/accel/accel0` were unavailable to the executing shell.

## Native repeatability

Native `--say` repetitions kept the renderer resident while rendering, but each repetition was a cold full-runtime process. The available PCM event remains the final WAV write. The saved scratch patch adds per-frame silence/voicing instrumentation without changing production source.

One supplemental CPU scratch run (ignored build checkout, restored afterward)
observed the first native non-silent frame at 1,556.872 ms from process start
(frame 4; 105.375 ms after the `say:` event), first silence-threshold drain
frame at 2,537.873 ms, and final WAV write at 3,948.837 ms. This measures
native frame/lifecycle state, not a PCM stream; the public `--say` path still
does not expose first streaming PCM. See `native-scratch-short.json`.

## Kokoro stream simulator

The simulator emits one text delta every 80 ms and tests complete-sentence, clause-boundary, and bounded-word flushing. The bounded-word rule is at least 32 buffered characters or 640 ms since the prior flush, ending at a word boundary. It is intentionally a feasibility probe, not a product adapter.

| fixture | policy | first text delta → PCM (ms) | chunks | synthesis RTF | underrun ms |
|---|---|---:|---:|---:|---:|
| medium | sentence | 1613.6 | 2 | 12.16x | 0.0 |
| medium | clause | 2761.4 | 2 | 11.90x | 0.0 |
| medium | bounded_word | 974.5 | 4 | 12.72x | 0.0 |
| long | sentence | 1377.4 | 3 | 12.30x | 0.0 |
| long | clause | 2528.3 | 6 | 12.43x | 0.0 |
| long | bounded_word | 993.5 | 7 | 14.11x | 0.0 |

## Decision

`PROMOTE` the Kokoro-style external-renderer concept to the next bounded integration design review, subject to real cancellation and playback-queue tests. Keep the implementation outside VoiceChat until the renderer contract is explicit.

`QUALIFY` the current CPU stream adapter: it demonstrates buffering and chunk production, but the synchronous CPU call cannot preempt in-flight synthesis and the model was not exercised on gfx1151 or XDNA2.

Representative long-fixture WAVs are under `audio/` and are ignored by the experiment policy.
