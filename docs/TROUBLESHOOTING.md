# Troubleshooting

## Known issues

### HIP graph-enabled warmup crashes

Graph-enabled warmup currently crashes in HIP graph execution on the R9700
reference. Workaround, used throughout the R9700-Q8-M1 baseline:

```
GGML_CUDA_DISABLE_GRAPHS=1
--no-warmup
```

This is recorded as a required compatibility setting, not a resolved bug.
Track upstream fixes against the pinned runtime commit in
`runtime/README.md`.

### Function/tool-call channel emits malformed JSON

The model selects the correct tool, invokes it, and consumes the returned
result correctly, but its emitted tool-call text consistently omits the
colon after `arguments` (for example
`{"name": "get_current_weather", "arguments {"location": "Seattle"}}`
instead of `"arguments": {...}`). The transport path passes; strict JSON
tool parsing does not.

Status: QUALIFIED, not a blocker for the initial voice experience. Do not
fix this as part of M1-M3 work; it is a separate future lead for a client
or runtime-side parsing policy.

### Stock service exposes only complete-WAV audio

`audio_ready_ms` in the benchmark harness is complete-file availability,
not first-playable-audio latency. Push-to-talk (M3) needs incremental
playback to improve perceived latency; this is expected, not a defect in
the current baseline.

<!-- Add new entries above this line as they're found. -->
