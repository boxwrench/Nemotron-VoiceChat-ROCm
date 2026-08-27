# D3 / C10 preencoder effect-boundary cut — gfx1151

Status: **PROMOTED**

Classification: **D6_DEC_PREENC_PROMOTE**

Runtime instrumentation base: `722e22a0aae1dd5cff4fee8e5c81aaddc6086bc3`

Configuration:

```text
Ryzen AI MAX+ 395 / Radeon 8060S / gfx1151
VC_FHEAD_GPU=1
renderer=false
native TTS=false
VC01, 37 consecutive 1,280-sample / 80 ms PCM slices
```

## Claim and boundary

D3 requests only `pre_enc_out` from the generic VoiceChat projector, then
passes that row into the separately maintained bounded one-frame encoder.

```text
Dc: causal preencoder + full 24-layer generic projector + projection
De: causal preencoder through pre_enc_out
```

The new research graph terminates after `pre_enc_out`; its first patch changes
only graph extent. It does not change precision, tensor layout, backend,
function-head selection, state representation, encoder arithmetic, or D3
scheduling. The former full graph remains available as the control.

The omitted encoder/projection has no required future-frame state effect:
the persistent state used after `pre_enc_out` belongs to
`clip_voicechat_stream_step()` and is updated once by the bounded D2 encoder.
Full-versus-cut state and embedding hashes provide the end-to-end check.

## Correctness gates

An eight-slice parity probe ran the cut and full graph for the same inputs.
`pre_enc_out` was bitwise equal on all eight frames (cosine 1.0, RMSE 0,
maximum absolute difference 0). An independent eight-slice full-graph control
then matched the cut on every timeline id, function token, D2 state hash,
D2-state byte count, and projected-embedding hash.

The 37-slice service runs preserved that result on all frames:

```text
37 / 37 captured slices -> exactly 37 contiguous timeline steps
function token: 12 on every control and cut frame
D2 state hashes: exact full == cut
projected embedding hashes: exact full == cut
timeline backlog peak: 0 in both runs
```

Thus no omitted node changed `pre_enc_out`, the bounded encoder cache, the
projected embedding, or the observable D3 timeline for this control.

## Mechanism and service curve

Steady state excludes cold frame 0 (frames 1–36):

| Metric | Full graph | Preencoder-only | Delta |
| --- | ---: | ---: | ---: |
| graph nodes | 1,925 | 42 | -1,883 (-97.8%) |
| preencoder compute p50 / p95 | 11.694 / 19.190 ms | 1.661 / 3.199 ms | -10.033 / -15.991 ms |
| perception p50 / p95 | 24.679 / 29.680 ms | 13.884 / 18.315 ms | -10.795 / -11.365 ms |
| main p50 / p95 | 47.501 / 48.055 ms | 47.448 / 48.261 ms | run variation only |
| total p50 / p95 | 72.104 / 79.026 ms | 61.510 / 65.093 ms | -10.594 / -13.933 ms |
| total maximum | 80.576 ms | 67.193 ms | -13.383 ms |
| deadline misses | 1 / 36 | 0 / 36 | recovered |

The full control's missed frame was frame 4: 80.576 ms total and 20.955 ms in
the preencoder graph-compute interval. The cut's largest residual
preencoder-compute interval was 6.222 ms (frame 35); its largest total frame
was 67.193 ms. The graph-compute timer includes HIP graph execution and the
required synchronization, so this result establishes work removal without
claiming a further kernel-versus-wait split.

## DEC result

This is a confirmed `C10` recursive residual-over-enumeration result. The
generic projector rebuilt and executed downstream encoder/projection work after
the requested effect boundary. That work was removed safely, with exact
preencoder and complete D3 state/output parity.

It is distinct from the retained causal Pad -> depthwise-Conv lead, which asks
whether the necessary preencoder itself still has an overcomplete contributor
domain. It is also distinct from graph-build/allocation and state-placement
work, whose measured contribution was too small to explain the original tail.

## Scope boundary

This promotes only the preencoder effect-boundary cut. It does not enable the
renderer or TTS, change device placement, reopen XDNA, or begin D4. Remaining
D3 gates include renderer/ALSA qualification, live capture behavior, and
broader workload service curves.
