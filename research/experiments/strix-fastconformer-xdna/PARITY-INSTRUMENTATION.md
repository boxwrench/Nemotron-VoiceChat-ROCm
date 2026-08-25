# Temporary runtime parity instrumentation

Bringup-3 included a scratch-only instrumentation pass in the runtime
submodule to prepare ggml-versus-ONNX intermediate comparisons. The runtime
changes are intentionally not part of this checkpoint and must not be pushed
or merged.

## Runtime changes

At the pinned/local runtime checkout, apply the temporary diff to:

```text
common/debug.cpp
tools/mtmd/clip.cpp
```

`common/debug.cpp` adds an environment-controlled tensor dump inside
`common_debug_cb_eval`:

```text
MTMD_DEBUG_DUMP_NAME   exact ggml tensor name to capture
MTMD_DEBUG_DUMP_TENSOR binary output path
```

When both variables are set and the tensor name matches, the hook copies the
tensor to host memory and writes this little-endian binary format:

```text
int32 ndim
int32 ne[4]
float32 values in i3, i2, i1, i0 traversal order
```

`tools/mtmd/clip.cpp` adds a diagnostic size check before setting the raw
perception input. It reports the expected and received element counts before
the existing assertion fires; it does not alter the tensor or model path.

## Scratch build and use

From the repository build directory, rebuild only the existing debug target:

```bash
cmake --build build/hip-gfx1151 --target llama-mtmd-debug
```

Run the debug encoder with an explicit model/projector pair and a controlled
audio fixture. The exact tensor name depends on the graph dump being studied:

```bash
MTMD_DEBUG_DUMP_NAME='<tensor-name>' \
MTMD_DEBUG_DUMP_TENSOR='<scratch-dir>/ggml-tensor.bin' \
build/hip-gfx1151/bin/llama-mtmd-debug \
    -m '<llm-gguf>' \
    --mmproj '<perception-gguf>' \
    --audio one \
    -n 20480 \
    -p encode
```

Read the five-int32 header and float32 payload with a small analysis script,
then compare the resulting shape and values against the corresponding ONNX
intermediate. Keep all dumps and logs outside Git.

## Observed limitation

The initial scratch attempt reached the existing raw-input assertion with:

```text
inp_raw expects 20480, got 61440
```

No parity result was accepted from that attempt. The instrumentation remains
useful as a reproducible starting point, but parity must be re-established
only after the input fixture and debug-tool path are aligned. This is separate
from XDNA compiler compatibility and is not a production-runtime change.
