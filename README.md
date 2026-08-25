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
| AMD Strix Halo | gfx1151 | VALIDATED, known perception/CLIP CPU fallback; see [validation notes](research/hardware-validation/gfx1151/README.md) |

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
including the push-to-talk client/server split (`app/push-to-talk/`).

## Quickstart

```
git clone https://github.com/boxwrench/Nemotron-VoiceChat-ROCm.git
cd Nemotron-VoiceChat-ROCm
scripts/setup.sh

app/push-to-talk/.venv/bin/python app/push-to-talk/ptt_terminal.py
```

`setup.sh` runs, in order: `download-q8.sh` (source GGUF + tokenizer
metadata) -> `build-rocm.sh` (pinned runtime, HIP build) -> `convert-q8.sh`
(split/verify Q8 artifacts) -> `smoke-test.sh` (BUILD/LOAD/STT/TTS/S2S).
See [docs/INSTALL.md](docs/INSTALL.md) for the full, deterministic flow and
what each step verifies.

Once setup finishes, `app/push-to-talk/` is how you actually talk to the
model: press Enter to start recording, speak, press Enter to stop and
submit, hear the reply. See
[app/push-to-talk/README.md](app/push-to-talk/README.md) for the venv
setup, environment variables, and the optional global-hotkey (Space-hold)
variant -- the terminal client above is the reliable default; it has no
display-server dependency.

## Model weights

Model weights are never committed to or redistributed through this Git
repository. Setup scripts download the source GGUF from Hugging Face,
run the Q8 conversion steps against the pinned runtime fork, and verify
output SHA256 hashes. See [docs/MODELS.md](docs/MODELS.md) for provenance
and licensing, which is documented separately from this repository's own
license.

## v0.1 release

**v0.1 is push-to-talk: a persistent, multi-turn conversation with native
Nemotron speech in and out, on a single AMD Radeon GPU running Linux +
ROCm.** The terminal client (`app/push-to-talk/ptt_terminal.py`) is the
reliable default interface; the global-hotkey Space-hold client
(`ptt.py`) is available but has Wayland/input-permission constraints the
terminal client doesn't.

Explicitly not in v0.1: continuous/live duplex, barge-in, streaming PCM
*during* generation, multi-GPU, replacement TTS, production tool calling,
non-Linux audio backends. These were investigated as the M4 milestone
(see [docs/M4-DUPLEX-DESIGN.md](docs/M4-DUPLEX-DESIGN.md)) and found to
need real async/runtime engineering beyond a quick addition -- that work
continues after this release, not before it.

Separately, an opt-in **post-turn** streaming-playback path
(`VC_TTS_STREAM_PLAYBACK=1`, see
[app/push-to-talk/README.md](app/push-to-talk/README.md)) is available
for this release's human validation round: assistant text still streams
during generation as before, and once the response text finishes, native
speech now starts playing as soon as the first already-available audio
is decoded, rather than waiting for the complete response WAV to render.
This is not speech streaming *during* generation, not duplex, and not
barge-in -- it only closes the gap between the model finishing talking
(in text) and the user hearing it. It remains opt-in, not the default,
until a human validation pass confirms it's ready to ship on by default.

### Known limitations

- **Multi-GPU hosts**: run with one ROCm GPU explicitly isolated
  (`ROCR_VISIBLE_DEVICES=<n>`). Multiple visible ROCm devices trigger a
  known TTS-init crash, found during M4 development and not yet fixed.
- **Global Space-hold (`ptt.py`)**: needs X11/XWayland, or `input`-group
  access on Wayland (see [app/push-to-talk/README.md](app/push-to-talk/README.md)
  "Wayland"). The terminal client (`ptt_terminal.py`) has neither
  constraint and is the recommended default.
- **Tool/function calls**: the model's tool-call JSON is consistently
  malformed (missing colon after `arguments`); a turn auto-skips a tool
  call rather than hanging. See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).
- **No duplex or barge-in**: strictly record-then-submit, one turn at a
  time. See [docs/M4-DUPLEX-DESIGN.md](docs/M4-DUPLEX-DESIGN.md) for why,
  and what's next.

## Roadmap

| Milestone | Scope |
| --- | --- |
| M1 | R9700 Q8 reproducible baseline -- DONE |
| M2 | gfx1100 + gfx1151 validation -- in progress |
| M3 | push-to-talk client -- DONE, the v0.1 release interface |
| M4 | native continuous duplex: always listening, incremental speech out, interruptible -- investigated and closed, see [docs/M4-DUPLEX-DESIGN.md](docs/M4-DUPLEX-DESIGN.md); resumes after v0.1 |
| M7 | optimize bottlenecks from the actual interactive workload |
| M8 | optional multi-GPU experiments |
| M9 | public AMD VoiceChat release |

M4 absorbed what earlier planning called M4 (multi-turn state, already
proven by M3), M5 (latency instrumentation), and M6 (duplex/barge-in) into
one milestone, since PTT already established multi-turn conversational
state and the real goal was always the duplex/barge-in behavior, not an
intermediate step on the way there. M4 ran to a closed feasibility result
(perception context resolved, real frame-budget bottlenecks decomposed, a
GPU function-head optimization promoted, incremental audio proven
numerically sound but not yet real-time-schedulable) without shipping
duplex -- the decision was to ship v0.1 on what M3 already proved works,
and continue M4's remaining leads (an async codec worker, or reducing the
codec's graph node count) afterward.

## Documentation

- [docs/INSTALL.md](docs/INSTALL.md)
- [docs/MODELS.md](docs/MODELS.md)
- [docs/HARDWARE.md](docs/HARDWARE.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md)
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
