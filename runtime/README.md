# Runtime dependency

This repository does not vendor the llama.cpp source tree. All build,
inference, and Q8 conversion code lives in the runtime fork and is consumed
by pinned commit.

```
Repository:     boxwrench/llama-voicechat.cpp
Upstream chain: ggml-org/llama.cpp -> sansamour/llama-voicechat.cpp -> boxwrench/llama-voicechat.cpp
Pinned commit:  38a76719e2b31a4dfc574bf750bb9ad44c434b81
                "voicechat: support Q8_0 component conversion"
```

`scripts/build-rocm.sh` clones the runtime repo and checks out this pin
before building. `scripts/convert-q8.sh` invokes the runtime repo's
conversion scripts under `tools/voicechat/` against this same pin.

Update the pinned commit here, deliberately, when a new runtime change is
validated against this project's benchmark suite -- do not float on a
branch tip.

## Do not vendor unless a concrete requirement appears

Keep this repository's job to integration, reproducibility, UX, and
research. If a concrete requirement to vendor part of the runtime source
appears (for example, a required local patch with no clean way to pin it
upstream), record the reasoning here before doing so.
