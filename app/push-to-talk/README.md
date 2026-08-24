# Push-to-talk client (M3-v0)

A thin client, not inference code. It spawns the existing
`llama-voicechat --serve` process once, keeps it resident for the whole
session, and talks to it over its documented stdin/stdout JSON-lines
protocol (see `tools/voicechat/voicechat-cli.cpp` -> `vc_serve()` in the
runtime repo). See [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) for
the client/server split this follows.

```
llama-voicechat --serve
      |
      | persistent session / JSON lines on stdin/stdout
      v
ptt.py
      +-- microphone capture     (arecord)
      +-- hold/release SPACE     (pynput)
      +-- send audio             ({"cmd":"turn",...})
      +-- stream assistant text  (assistant_text_delta events)
      +-- play returned speech   (aplay)
      +-- print latency events
```

Audio in/out uses the system's normal ALSA command-line tools
(`arecord`/`aplay`), not an audio framework. The one extra Python
dependency is `pynput`, used only for global hold/release key detection.

Two entry points, same `Session`/protocol/audio code underneath, different
input trigger:

- `ptt.py` -- hold/release SPACE (`pynput`). Needs X11/XWayland or
  `input`-group access on Wayland; see the Wayland section below.
- `ptt_terminal.py` -- Enter to start, Enter to stop, no global key capture
  needed. Use this on a plain Wayland session.

## Setup

```
python3 -m venv app/push-to-talk/.venv
app/push-to-talk/.venv/bin/pip install -r app/push-to-talk/requirements.txt
```

## Run

```
export VC_BIN=/path/to/build/hip-gfx1201/bin/llama-voicechat
export VC_MODEL=/path/to/models/voicechat-q8/runtime/nemotron_voicechat_11b-stt-llm-Q8_0.gguf
export VC_MMPROJ=/path/to/models/voicechat-q8/runtime/mmproj-voicechat-perception-Q8_0.gguf
export VC_TTS=/path/to/models/voicechat-q8/runtime/voicechat-tts-Q8_0.gguf   # optional, omit for text-only

app/push-to-talk/.venv/bin/python app/push-to-talk/ptt.py
```

Hold SPACE, speak, release SPACE. The turn submits, streamed text prints
as it arrives, and the reply plays through `aplay` once ready. The model
stays loaded between turns; press Esc or Ctrl+C to quit.

`--serve-cmd 'full command line'` overrides `--bin`/`--model`/`--mmproj`/
`--tts` entirely, if a different invocation is needed. `--rec-device` /
`--play-device` (or `$VC_REC_DEVICE` / `$VC_PLAY_DEVICE`) pick a specific
ALSA device (`arecord -l` / `aplay -l` list what's available); otherwise
the ALSA default is used.

### GPU selection

Neither client sets `ROCR_VISIBLE_DEVICES` -- it is never invented on your
behalf, only passed through if you've set it yourself:

- **Single-GPU system**: no `ROCR_VISIBLE_DEVICES` setting required.
  Normal runtime enumeration plus `--device ROCm0` selects the only
  visible ROCm device.
- **Multi-GPU system**: explicitly set `ROCR_VISIBLE_DEVICES` yourself to
  isolate the card you want, for example (the R9700 reference workstation
  in `research/baselines/R9700-Q8-M1/` is a dual-GPU example, not a
  universal requirement):
  ```
  export ROCR_VISIBLE_DEVICES=1   # example: isolates the R9700 on that workstation
  ```

Both clients default `GGML_CUDA_DISABLE_GRAPHS=1` (override by setting it
yourself before running). This is not a leftover dev-time convenience --
the R9700-Q8-M1 baseline demonstrated graph-enabled execution crashing on
this runtime (see `research/baselines/R9700-Q8-M1/README.md`, "Required
compatibility settings"), so it's a known-good requirement until that's
fixed upstream.

### Non-interactive check

```
app/push-to-talk/.venv/bin/python app/push-to-talk/ptt.py --test
```

Runs two turns from the fixed corpus (`research/corpus/`) instead of a
live mic/keyboard, exercising the same session/protocol/playback path.
Useful where holding a physical key or a live mic isn't available (e.g.
over SSH); it is not a substitute for a real hold-and-speak test.

## Wayland: `pynput` Space-hold does not work without `input` group access

On a strict Wayland session, `pynput`'s global key listener cannot
intercept Space at all -- held Space just types spaces into the terminal
instead of firing `on_press`/`on_release`. This needs either an X11/XWayland
session, or the user added to the `input` group so the `evdev` backend can
read `/dev/input/event*` (verify with `groups`; log out/in after adding).

Until then, use the terminal fallback client, which needs no global key
capture at all:

```
app/push-to-talk/.venv/bin/python app/push-to-talk/ptt_terminal.py
```

Press Enter to start recording, Enter again to stop and submit, `q` to
quit. It reuses the same `Session`/`record_to`/`submit_turn` as `ptt.py`
(only the input trigger differs), so it's the same protocol and audio
path, not a separate implementation. A 0.8s minimum-recording guard, a
stdin-flush between turns, and a post-turn cooldown avoid the two bugs an
Enter-based trigger is otherwise prone to: the stop keypress being read as
the next turn's start keypress, and submitting before `arecord` has
actually opened the capture device.

## Verified (R9700 / gfx1201)

- `ptt.py --test`: two turns end to end against the real build and Q8
  model -- runtime started once, `ready` received, both turns completed
  with streamed text and a written WAV, process stayed resident (no
  reload), shut down cleanly. `arecord`/`aplay` and the SIGINT-based
  stop-recording path checked independently against the real capture
  device.
- `ptt_terminal.py` with a **live microphone**, driven automatically
  (pseudo-tty) and confirmed manually by a human tester: two full turns
  (record -> submit -> transcribe -> LLM -> TTS -> playback), second turn
  without a model reload, clean quit on `q`.
- `ptt.py`'s interactive Space-hold path: **not** verified working, per
  above -- Wayland without `input` group blocks it. This is a display-
  server/permissions limitation, not a bug in the client.

## Known limitations (v0)

- No continuous duplex / barge-in -- strictly record-then-submit, one
  turn at a time, whichever client you use.
- A tool call is auto-skipped (`tool_skip`) rather than answered, so a
  turn never hangs on the function channel; this does not attempt to fix
  or work around the known malformed tool-call JSON issue (see
  [docs/TROUBLESHOOTING.md](../../docs/TROUBLESHOOTING.md)).
- No packaging/systemd unit yet -- run directly with the venv's Python.
