# Push-to-talk client (design stub, M3)

No code yet. This directory will hold a thin client, not inference code.
See [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) for the
client/server split this follows.

Planned responsibilities:

```
llama-voicechat --serve
      |
      | persistent session / JSON
      v
this client
      |
      +-- microphone capture
      +-- hold/release key
      +-- send audio
      +-- play returned speech
      +-- display latency/events
```

Initial UX: hold SPACE to record, release to submit the turn, hear the
spoken response. The runtime process stays resident between turns; this
client only opens/reuses a `--serve` session.

Explicitly out of scope until M3 begins: audio capture code, kernel/runtime
changes, and anything under `research/` -- this is a UX/client concern, not
a runtime-core concern.
