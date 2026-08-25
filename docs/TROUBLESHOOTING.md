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

### Multi-GPU hosts crash on TTS init unless one ROCm device is isolated

Found during M4 development (see `docs/M4-DUPLEX-DESIGN.md`, M4P-1):
whenever more than one ROCm device is visible to the process at all, the
process crashes shortly after TTS's KV-cache init -- reproducible
regardless of which device TTS is actually assigned to. Not yet debugged
or fixed.

Workaround: explicitly isolate one ROCm device before running anything in
this repo on a multi-GPU host:

```
export ROCR_VISIBLE_DEVICES=<n>
```

Single-GPU hosts are unaffected.

### Global Space-hold push-to-talk (`ptt.py`) needs X11 or `input`-group access

On a strict Wayland session, `pynput`'s global key listener cannot
intercept Space -- held Space just types spaces into the terminal instead
of firing the press/release handlers. Needs X11/XWayland, or the user
added to the `input` group so the `evdev` backend can read
`/dev/input/event*` (log out/in after adding). Use
`app/push-to-talk/ptt_terminal.py` instead -- same protocol and audio
path, no global key capture, works on any session type. See
[../app/push-to-talk/README.md](../app/push-to-talk/README.md) "Wayland".

### No duplex or barge-in (v0.1)

Push-to-talk is strictly record-then-submit, one turn at a time. A
continuous, always-listening, interruptible conversation was investigated
as milestone M4 (see `docs/M4-DUPLEX-DESIGN.md`) and found to need real
async/runtime engineering (a background audio-decode worker, or reducing
GPU kernel dispatch overhead in the speech codec) beyond what fits before
this release -- not a fundamental blocker, a scheduling problem being
carried forward.

### Stock service exposes only complete-WAV audio

`audio_ready_ms` in the benchmark harness is complete-file availability,
not first-playable-audio latency. Push-to-talk (M3) needs incremental
playback to improve perceived latency; this is expected, not a defect in
the current baseline.

<!-- Add new entries above this line as they're found. -->
