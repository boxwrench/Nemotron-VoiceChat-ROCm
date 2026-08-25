# VoiceChat FastConformer graph contract

Status: source-frozen contract; production invocation shape is still pending
PC D2.

## Provenance

```text
VoiceChat runtime source checkout
commit: 6da91b8c6e5035110721dd3319f0511376d7487c
source: build/llama-voicechat.cpp/tools/mtmd/models/voicechat.cpp
converter: build/llama-voicechat.cpp/tools/voicechat/convert_voicechat_perception_to_mmproj.py
```

The contract below is extracted from the pinned runtime source and converter,
not inferred from generic Parakeet defaults.

## Input and output

| Item | Contract |
|---|---|
| audio sample rate | 16 kHz |
| FFT/window/hop | 512 / 400 / 160 samples |
| mel features | 128 bins, log mel, NeMo/Parakeet featurizer |
| feature normalization | disabled (`normalize: "NA"`; no per-feature normalization) |
| graph input | `inp_raw`: `[T, 128, 1, 1]` in ggml layout; semantic NCHW input is `[1, 1, 128, T]` after transpose |
| prefix length | `T` mel frames; dynamic until D2 freezes the production contract |
| output cadence | one embedding per 80 ms / 12.5 Hz VoiceChat frame |
| encoder output | `encoder_out`: width 1024, `n_time = f(T)` frames |
| projected output | `projected`: width 4480, `n_time` frames; consumed directly by Nemotron perception fusion |

The VoiceChat causal subsampler maps each dimension with `n -> floor(n/2)+1`
for three stride-2 stages. Therefore:

```text
frequency: 128 -> 65 -> 33 -> 17
time:      T   -> floor(T/2)+1 -> floor(n/2)+1 -> floor(n/2)+1 = n_time
pre_encode.out input width: 17 * 256 = 4352
pre_encode.out output: [1024, n_time]
```

## Pre-encode

```text
inp_raw [T,128,1,1]
  -> transpose/contiguous
  -> causal Pad(2 left, 1 right) + dense Conv2D(k=3,s=2)
  -> bias + ReLU
  -> causal Pad + depthwise Conv2D(k=3,s=2) + pointwise Conv2D(k=1)
  -> bias + ReLU
  -> causal Pad + depthwise Conv2D(k=3,s=2) + pointwise Conv2D(k=1)
  -> bias + ReLU
  -> permute to [freq,channel,time]
  -> reshape [17*256,n_time]
  -> Linear(4352,1024)
```

Known compiler-sensitive locations:

```text
pre_conv_2   depthwise stride-2 convolution
pre_conv_5   depthwise stride-2 convolution
```

These are the exact locations associated with the gfx1151 `CONV_2D_DW`
fallback observation.

## Encoder configuration

```text
layers                         24
model width / n_state          1024
attention heads                8
head width                     128
FFN width                      4096
activation                     SiLU in both macaron FFNs
FFN scaling                    0.5 for each macaron branch
convolution kernel             9
convolution context            causal; 8 zero positions on the left
convolution normalization      LayerNorm over the channel axis
normalization epsilon          1e-5
positional encoding            Transformer-XL relative position encoding
attention context              chunked_limited, [70 left, 0 right]
chunk width                    1 frame; no future lookahead
attention mask                 external F32 [n_time,n_time] input
projection                     Linear(1024,4480) + bias; IdentityConnector
```

## Per-layer execution

Each of the 24 layers executes, in order:

```text
1. macaron FFN: LayerNorm -> Linear(1024,4096) -> SiLU -> Linear(4096,1024)
2. relative self-attention:
   LayerNorm -> Q/K/V -> relative-position scores -> F32 mask add
   -> softmax -> V aggregation -> output Linear
3. convolution module:
   LayerNorm -> pointwise Linear(1024,2048) -> GLU
   -> causal depthwise convolution(k=9) -> channel LayerNorm -> SiLU
   -> pointwise Linear(1024,1024)
4. second macaron FFN: LayerNorm -> Linear -> SiLU -> Linear
5. final LayerNorm/affine
```

The original full-prefix graph has no useful cross-call encoder state. The D2
research runtime at `6da91b8c...` separately proves a bounded one-frame
encoder mechanism with 70-frame K/V history and eight-frame convolution
history per layer. Strix may use that as a compiler-feasibility contract, but
must not present the provisional steady-state graph as the final production
PCM/frontend or device-placement contract.

## Four VoiceChat-specific differences

| Difference | Exact source consequence | XDNA question |
|---|---|---|
| `causal_downsampling` | all three subsampling stages use Pad(2 left, 1 right) on time and frequency; 128 mel bins become 17 | can the compiler fold or represent asymmetric causal padding without changing embeddings? |
| `layer_norm` conv norm | checkpoint tensors retain `batch_norm` names, but runtime uses LayerNorm with no running statistics | is channel-axis LayerNorm supported/fusable in the target graph? |
| causal convolution context | depthwise k=9 receives eight left zeros through Pad + Roll; no right context | can causal Pad/roll be represented or rewritten as a legal depthwise Conv? |
| `chunked_limited` attention | `[70,0]` with chunk width 1 becomes F32 causal mask over the current frame and 70-frame history | can relative-position attention and the exact mask be compiled without semantic substitution? |

## D2 production-contract boundary

These are invariant now:

```text
zero future lookahead
12.5 Hz VoiceChat timeline
learned FastConformer embedding required
one 4480-wide projected embedding per served frame
no transcript substitution
```

D2 still controls for production serving:

```text
prefix/window length
invocation cadence
input tensor shape
output-frame selection
maximum useful conversation prefix
per-call service-time budget
```

For encoder/XDNA feasibility, the imported D2 steady-state boundary is:

```text
pre_enc_out [1,1,1024]
24 × K/V [1,8,70,128]
24 × conv [1,8,1024]
    -> projected [1,1,4480] + updated bounded state
```

This is a research handoff, not permission to integrate it into VoiceChat.

The compiler spike therefore uses provisional shapes only. They are not the
production serving contract. D2 is specifically resolving the past-context
and cross-call caching/state question that determines that contract.
