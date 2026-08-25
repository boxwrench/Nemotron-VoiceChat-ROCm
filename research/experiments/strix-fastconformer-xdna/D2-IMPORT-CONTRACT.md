# D2 encoder import contract for Strix feasibility

Status: research import only. This authorizes encoder/compiler feasibility
work; it does not authorize production VoiceChat integration.

## Provenance

```text
llama-voicechat.cpp: 6da91b8c6e5035110721dd3319f0511376d7487c
product evidence:    92c8dbd51a2d06c06813ff9c26f1cef4a966213d
source graph:        tools/mtmd/models/voicechat.cpp
stateful prototype:  tools/mtmd/clip.cpp
```

The imported D2 mechanism is a bounded one-frame encoder step. It is not the
final live PCM/frontend production contract and does not select a final cache
placement for XDNA.

## Steady-state graph boundary

```text
input:
  pre_enc_out       F32 [1, 1, 1024]
  per-layer K state  F32 [24, 8, 70, 128]
  per-layer V state  F32 [24, 8, 70, 128]
  per-layer conv     F32 [24, 8, 1024]

output:
  projected          F32 [1, 1, 4480]
  updated K/V state  one new frame per layer
  updated conv state one new frame per layer
```

The logical state is:

```text
24 layers × 70 attention frames × 2 × 1024 values
24 layers × 8 convolution frames × 1024 values
```

At F32 this is exactly 14,548,992 bytes (about 13.88 MiB). The state is
bounded with session length. The first Strix graph uses steady-state history
70; startup history 0..69 remains a separate serving-design question.

## Invariants imported from D2

```text
12.5 Hz / one 80 ms timeline step
zero future lookahead
raw natural-log mel in the active VoiceChat frontend
learned FastConformer embedding, not transcript substitution
relative-position attention with 70-frame left context
causal kernel-9 convolution with 8 retained positions
```

## Not frozen by this import

```text
PCM chunk size and frontend state placement
startup graph strategy
host/device state placement
production XDNA input/output tensor ownership
full duplex integration
```

The Strix experiment must preserve the D2 logical state contract while keeping
those serving decisions explicit and provisional.
