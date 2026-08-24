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
export ROCR_VISIBLE_DEVICES=1   # isolate the target card, as in the R9700 baseline

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

### Non-interactive check

```
app/push-to-talk/.venv/bin/python app/push-to-talk/ptt.py --test
```

Runs two turns from the fixed corpus (`research/corpus/`) instead of a
live mic/keyboard, exercising the same session/protocol/playback path.
Useful where holding a physical key or a live mic isn't available (e.g.
over SSH); it is not a substitute for a real hold-and-speak test.

## Verified (R9700 / gfx1201, this session)

`--test` ran two turns end to end against the real build and Q8 model:
runtime started once, `ready` received, turn 1 (VC01, short factual) and
turn 2 (VC02, conversational) both completed with streamed text and a
written WAV, the process stayed resident across both turns (no reload),
and it shut down cleanly afterward. `arecord`/`aplay` and the SIGINT-based
stop-recording path were checked independently against the real capture
device.

Not verified in this session: an actual hold/release SPACE test with a
physical keyboard and live microphone end to end (no interactive terminal
in this environment) -- do that locally before relying on this for real
use.

## Known limitations (v0)

- No continuous duplex / barge-in -- strictly hold-to-record,
  release-to-submit, one turn at a time.
- A tool call is auto-skipped (`tool_skip`) rather than answered, so a
  turn never hangs on the function channel; this does not attempt to fix
  or work around the known malformed tool-call JSON issue (see
  [docs/TROUBLESHOOTING.md](../../docs/TROUBLESHOOTING.md)).
- `pynput`'s global key listener depends on the display server; it is
  known to work under X11/XWayland and may need adjustment under a
  Wayland-only compositor with stricter input restrictions.
- No packaging/systemd unit yet -- run directly with the venv's Python.
