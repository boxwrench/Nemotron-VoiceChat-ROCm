# Runtime dependency

This repository does not vendor the llama.cpp source tree. All build,
inference, and Q8 conversion code lives in the runtime fork and is consumed
by pinned commit -- never by a floating branch tip.

```
Repository:          boxwrench/llama-voicechat.cpp
Upstream chain:       ggml-org/llama.cpp -> sansamour/llama-voicechat.cpp -> boxwrench/llama-voicechat.cpp
Integration branch:   amd/rocm
Pinned known-good:    38a76719e2b31a4dfc574bf750bb9ad44c434b81
                      "voicechat: support Q8_0 component conversion"
```

## Integration branch vs. pinned commit

`amd/rocm` is where candidate AMD runtime changes accumulate in the runtime
fork (backend fixes, additional Q8 work, etc.). It moves independently of
this repository.

This repository always builds from the **pinned commit** above, not from
the current tip of `amd/rocm`. `scripts/build-rocm.sh` and
`scripts/convert-q8.sh` check out that exact SHA -- they do not check out
`amd/rocm` by branch name, and they must not be changed to do so.

The pin only advances when a newer `amd/rocm` commit has been explicitly
validated against this project's benchmark/smoke-test suite, and the
advance is a deliberate edit to this file, not an automatic follow of the
branch tip.

## Do not vendor unless a concrete requirement appears

Keep this repository's job to integration, reproducibility, UX, and
research. If a concrete requirement to vendor part of the runtime source
appears (for example, a required local patch with no clean way to pin it
upstream), record the reasoning here before doing so.
