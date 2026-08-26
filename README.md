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
| AMD Radeon RX 7900 XT | gfx1100 | PENDING (H1) |
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

**v0.1.0 is shipped** (tagged `v0.1.0`): push-to-talk, a persistent,
multi-turn conversation with native Nemotron speech in and out, on a
single AMD Radeon GPU running Linux + ROCm. The terminal client
(`app/push-to-talk/ptt_terminal.py`) is the reliable default interface;
the global-hotkey Space-hold client (`ptt.py`) is available but has
Wayland/input-permission constraints the terminal client doesn't.

Explicitly not in v0.1: continuous/live duplex, barge-in, streaming PCM
*during* generation, multi-GPU, replacement TTS, production tool calling,
non-Linux audio backends. These were investigated as the M4 milestone
(see [docs/M4-DUPLEX-DESIGN.md](docs/M4-DUPLEX-DESIGN.md)) and found to
need real async/runtime engineering beyond a quick addition. M4 was not
a failure to build duplex -- it turned a giant unknown problem into a
map. That map is now the post-v0.1 program below.

v0.1.0 also ships an opt-in **post-turn** streaming-playback path
(`VC_TTS_STREAM_PLAYBACK=1`, see
[app/push-to-talk/README.md](app/push-to-talk/README.md)): assistant
text still streams during generation as before, and once the response
text finishes, native speech now starts playing as soon as the first
already-available audio is decoded, rather than waiting for the
complete response WAV to render (main-complete -> first-audio ~157ms
measured, vs. several-to-tens-of-seconds before). This is not speech
streaming *during* generation, not duplex, and not barge-in -- it only
closes the gap between the model finishing talking (in text) and the
user hearing it. It passed five live human validation turns and is
integrated into the release's runtime pin; it remains opt-in (not the
default) pending broader validation across more hardware/sessions.

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

M1-M4 are historical and frozen. M1-M3 shipped what they set out to
build; M4 absorbed what earlier planning called M4 (multi-turn state,
already proven by M3), M5 (latency instrumentation), and M6
(duplex/barge-in) into one milestone, since PTT already established
multi-turn conversational state and the real goal was always the
duplex/barge-in behavior. M4 did not ship duplex, but it was not a
failure: it turned a giant unknown problem into a map -- perception's
future-context dependence is negligible, its real cost is re-encoding
growing history; the main-path GPU decode cost is real and unrecoverable
while the function head was a genuine, promoted optimization; the causal
codec and streaming ISTFT produce numerically correct incremental audio,
but synchronous invocation is too expensive for an 80ms live frame. See
[docs/M4-DUPLEX-DESIGN.md](docs/M4-DUPLEX-DESIGN.md) for the full record.

The forward program (D1-D7, H1-H2, F1, R2) replaces the old placeholder
M7-M9 rows with the actual plan that map produced:

| Milestone | Scope |
| --- | --- |
| M1 | R9700 Q8 reproducible baseline -- DONE |
| M2 | AMD hardware validation -- gfx1201 DONE, gfx1151 DONE (see [validation notes](research/hardware-validation/gfx1151/README.md)), gfx1100 PENDING (H1) |
| M3 | Persistent push-to-talk product path -- DONE, v0.1.0 shipped |
| M4 | Native duplex feasibility investigation -- DONE, produced perception, critical-path, and streaming-audio findings, see [docs/M4-DUPLEX-DESIGN.md](docs/M4-DUPLEX-DESIGN.md) |
| D1 | Async native audio renderer -- **RESEARCH QUALIFIED**; implementation boundary exists at runtime `14676822b9b973070ee04d1d8ebf5ba11fff22b2`, but real ALSA/live-timeline integration remains open |
| D2 | Bounded-state perception -- **RESEARCH QUALIFIED**; stateful encoder and bounded frontend probe exist at runtime `6da91b8c6e5035110721dd3319f0511376d7487c`, but the normal production encode path remains untouched |
| D3 | Continuous causal VoiceChat timeline -- **ACTIVE / hardware causality PASS**; one gfx1151 80 ms slice authorized exactly one persistent D2/main step, but the first integrated frame was 146.157 ms and is not deadline-stable yet |
| D4 | Native model turn-taking -- blocked on D3 |
| D5 | User interruption / barge-in -- blocked on D4 |
| D6 | Measured optimization / component substitution -- iterative across D1-D5, only where a real measurement demands it |
| D7 | Long-session quality / stress / stability -- blocked on working duplex |
| H1 | gfx1100 (RX 7900 XT) validation -- parallel / non-blocking |
| H2 | Multi-GPU diagnostic ceiling (fix the known visibility crash, then a matched placement experiment) -- later, not before single-R9700 duplex work |
| F1 | Strix Halo serving/XDNA program -- parallel, imports stable contracts from D1/D2/D3 as they land, see [docs/STRIX-ROADMAP.md](docs/STRIX-ROADMAP.md) and [serving options](docs/STRIX-SERVING-OPTIONS.md) |
| R2 | Public fluent-duplex release -- end state |

`D*` is deliberately new numbering rather than reusing M5-M9 for
different things -- history stays understandable.

**D1/D2/D3 reconciliation (2026-08-25)**: D1 and D2 are no longer merely
future investigations. Their research branches contain qualified mechanisms,
but neither mechanism is merged into the release runtime or connected to the
push-to-talk path. D3 is therefore unblocked for implementation, not declared
product-ready: its first job is to make the ownership, bounded queues,
causality, and telemetry contracts real. See
[docs/D3-INTEGRATION-GAP.md](docs/D3-INTEGRATION-GAP.md).

**Why D1 first**: M4B already proved the codec's math (correct
incremental PCM, correct streaming ISTFT) and its aggregate throughput
(comfortably above realtime); the specific thing that failed was
synchronous dispatch inside the live 80ms frame. D1 tests the next
cheapest mechanism -- scheduling, via an async renderer worker feeding a
PCM ring -- before reaching for codec kernel work, perception changes,
or a second GPU. It's the highest-information next experiment: it tells
us whether the existing native system is much closer to duplex than the
synchronous measurements suggest, or whether same-GPU contention between
the main model and the codec worker is itself the real ceiling.

**D2's actual open question**: M4 already showed perception's
*future*-context dependence is negligible (zero-lookahead is viable).
The real problem is *past* context: a naive growing-prefix re-encode
gets too expensive around 20-24s of conversation, and no bounded window
tested so far has both adequate timing margin and reliable downstream
fidelity. D2 is a bounded-context and/or cached-encode problem, not a
"does it need the future" problem.

**Dependency shape**:
```
v0.1.0 -----------> D1 (async renderer) --\
                                            --> D3 (live timeline) -> D4 (turn-taking) -> D5 (barge-in) -> D7 (stability) -> R2
M4 evidence -------> D2 (perception contract) --/
```
D6 is not a terminal stage -- it's fed continuously by measurements from
D1 through D5, and only acts where a real measured deadline miss demands
it, not because an optimization sounds attractive.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the north-star
acceptance criteria and the track structure this program runs under.

## Documentation

- [docs/INSTALL.md](docs/INSTALL.md)
- [docs/MODELS.md](docs/MODELS.md)
- [docs/HARDWARE.md](docs/HARDWARE.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md)
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- [docs/STRIX-ROADMAP.md](docs/STRIX-ROADMAP.md)
- [docs/STRIX-SERVING-OPTIONS.md](docs/STRIX-SERVING-OPTIONS.md)
