# Temporary runtime parity instrumentation

Bringup-3 included a scratch-only instrumentation pass in the runtime
submodule to prepare ggml-versus-ONNX intermediate comparisons. The runtime
changes are intentionally not part of this checkpoint and must not be pushed
or merged.

## Runtime changes

At the pinned/local runtime checkout, apply the temporary diff to:

```text
tools/mtmd/clip.cpp
```

The current scratch diff adds an environment-controlled D2 stream-boundary
capture in `clip_voicechat_stream_step`:

```text
VC_D2_STATE_DUMP_DIR     existing scratch root
VC_D2_STATE_DUMP_STEPS   comma-separated D2 step indices, for example 0,70
```

For each selected step it writes an ignored `step-XXXXXX/` directory with
little-endian F32 raw tensors:

```text
pre_enc
K/V and convolution state before the step
projected output
new K/V/convolution state
first macaron-FFN output
state.json metadata
```

The first-FFN tensor is an explicit scratch graph output so scheduler reuse
cannot overwrite it before capture. This adds diagnostic outputs only when the
opt-in path is enabled and must not be committed to the runtime.

The S5-P1 attribution variant additionally roots only layer 0's first-FFN
boundaries: raw and affine LayerNorm output, `linear1`, SiLU, `linear2`, the
0.5-scaled branch, and its residual output. `analyze_ggml_ffn_stages.py`
compares the live Q8 capture with a direct dequantized-F32 control and CPU
ONNX. These extra roots exist solely to locate a representation split.

Two further scratch-only variables support the F32 fidelity falsification:

```text
VC_D2_PREENC_TRACE=<path>       append each [1024] D2 pre-encoder input
VC_D2_EMBD_INJECT_PATH=<path>   replace the timeline embedding sequence only
```

The injector checks the exact `n_frames * 4480 * sizeof(float)` contract and
does not alter model state or normal inference when absent. It must never
become a production renderer/perception API.

## Scratch build and use

From the runtime build directory, rebuild the VoiceChat binary:

```bash
cmake --build <scratch-build> --target llama-voicechat
```

Run the D2 comparison path with a real corpus WAV. The base STT model is used
for perception capture when the local runtime cannot load the separate
function-head architecture:

```bash
VC_D2_STATE_TEST=1 \
VC_D2_STATE_DUMP_DIR='<scratch-dir>' \
VC_D2_STATE_DUMP_STEPS='0,70' \
<scratch-build>/bin/llama-voicechat \
    -m '<stt-llm.gguf>' --mmproj '<perception.gguf>' \
    --audio '<fixture.wav>' -n 1
```

Use `check_ggml_onnx_parity.py` to convert the explicit ggml K/V layout into
the ONNX layout and compare a captured step. Keep all dumps and logs outside
Git.

## Observed limitation

The old generic raw-audio debug path reached a shape assertion and was not
used for S5. The stream-boundary capture replaced it. S5 then established a
real authoritative parity failure at the layer-0 first macaron FFN; see
`S5-QUALIFICATION-RESULTS.md`. No compiler conclusion is valid until that
exporter mismatch is repaired.
