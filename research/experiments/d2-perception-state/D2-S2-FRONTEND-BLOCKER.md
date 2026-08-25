# D2-S2: frontend contributor frontier and blocker

Status: **BLOCKED under the current VoiceChat frontend semantics**

The D2-S1 bounded encoder state is valid. The complete PCM-to-embedding path
is blocked one stage earlier by the existing conformer mel normalization, not
by the causal subsampler or by XDNA.

## Authoritative source

`tools/mtmd/mtmd-audio.cpp`,
`mtmd_audio_preprocessor_conformer::preprocess()` constructs:

```text
center_padding   = true
preemph          = 0.97
use_natural_log  = true
norm_per_feature = true
sample rate      = 16,000 Hz
FFT              = 512
window           = 400
hop              = 160
mel bins         = 128
```

The normalization implementation then computes, independently for every mel
bin `m`:

```text
L       = n_samples_in / 160
mean_m  = sum(z[m,j], j = 0 .. L-1) / L
var_m   = sum((z[m,j] - mean_m)^2, j = 0 .. L-1) / (L-1)
y[m,j]  = (z[m,j] - mean_m) / sqrt(var_m + 1e-5)
```

`z` is the log-mel frame before normalization. This is a source-derived
fact, not an inference from model documentation.

## DEC contributor result

For an early normalized output `y[m,j]`, a later audio frame `z[m,k]`
contributes through both `mean_m` and `var_m`. Generically, changing any
future frame changes the already-requested normalized output. Across the
session, the contributor domain is therefore:

```text
candidate domain Dc: all supplied audio frames/samples used to form z
contributor domain De for y[m,j]: all session frames 0 .. L-1
```

The desired bounded space:

```text
new samples + finite frontend state
```

does not exist for exact preservation of the current normalized mel values.
This is the direct-enumeration conclusion: the normalization contributor
frontier is the full utterance, so there is no safe state-only elimination to
implement.

## What is still bounded before normalization

The local, pre-normalization contributor frontier is well-defined:

```text
pre-emphasis:
  new samples + previous sample + startup alignment

STFT/mel frame j:
  one 512-sample window at padded offset j*160
  previous 352 waveform samples can be retained between adjacent frames
  plus hop/frame phase and center-padding/end-of-input state
```

With center padding, frame `j` reads padded samples
`[160*j, 160*j + 511]`, corresponding to original sample coordinates
`[160*j - 256, 160*j + 255]` after the known zero boundaries. The
pre-emphasis boundary is one scalar previous sample, with the first original
sample as the initialized predecessor under the current implementation.

These states can produce exact **unnormalized** log-mel frames. They cannot
produce exact current **normalized** frames online without future statistics.

## Causal subsampling residue map

The three VoiceChat temporal convolution stages use kernel 3, stride 2,
left pad 2, and right pad 1. For temporal index `i`, each stage maps:

```text
stage 1: y1[i] depends on mel rows       [2i-2, 2i]
stage 2: y2[j] depends on stage-1 rows   [2j-2, 2j]
         therefore raw mel rows          [4j-6, 4j]
stage 3: y3[k] depends on stage-2 rows   [2k-2, 2k]
         therefore raw mel rows          [8k-14, 8k]
```

Negative coordinates are the explicit causal left-padding contributors. The
newest productive coordinate is direct-enumeration friendly:

```text
stage 1 output i is authorized when mel_count > 2i
stage 2 output j is authorized when stage1_count > 2j
stage 3 output k is authorized when stage2_count > 2k
pre_enc_out[k] is authorized when mel_count > 8k
```

The residue/phase is independently tracked at each stage; no neighboring-row
selection is valid. A future implementation can retain two temporal feature
rows at each stage plus the phase and exact boundary state. That result is
calculated from the source graph, but it is not promoted into runtime code
because the normalized mel contract currently prevents exact online input.

## Required architectural choice

One of these must be explicitly selected before D2 can continue to a complete
production path:

```text
A  remove/replace per-feature normalization and revalidate the model;
B  define a causal/running normalization contract and revalidate downstream;
C  delay perception until utterance statistics are known;
D  preserve full-utterance normalization and accept unbounded/future state.
```

The current D2 requirements reject C and D for fluent continuous operation,
and they do not authorize A or B without a model-behavior/downstream-fidelity
experiment. Therefore no streaming frontend implementation was committed.

## D2-S2 decision

```text
waveform local frontier:       DERIVABLE
STFT/mel local frontier:       DERIVABLE before normalization
causal subsampling frontier:   DERIVABLE, exact residue map above
current normalized mel frontier: FULL SESSION / not bounded
complete PCM-to-embedding path:  BLOCKED by frontend contract
```

This is an architectural semantics blocker, not an XDNA/compiler rejection.
