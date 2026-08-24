# Hardware

| GPU | gfx target | Status | Evidence |
| --- | --- | --- | --- |
| AMD Radeon AI PRO R9700 | gfx1201 | PROVEN (reference) | `research/baselines/R9700-Q8-M1/` |
| AMD Radeon RX 7900 XT | gfx1100 | not yet validated | `research/hardware-validation/gfx1100/` |
| AMD Strix Halo | gfx1151 | not yet validated | `research/hardware-validation/gfx1151/` |

## Reference environment (R9700)

- ROCm 7.2.1, HIP 7.2.53211
- AMD Ryzen 7 9800X3D, 186 GiB system RAM
- `ROCR_VISIBLE_DEVICES` used to isolate the target card when multiple GPUs
  are present in the host; see
  `research/baselines/R9700-Q8-M1/README.md` for the exact required
  compatibility settings (HIP graphs disabled, `--no-warmup`).

## Validation bar for additional GPUs (M2)

For each new GPU, initially require only:

```
BUILD    LOAD    STT    TTS    full S2S    memory    rough latency    correctness
```

Mark the GPU VALIDATED if all pass. If any step fails, stop at the first
concrete blocker and record it under
`research/hardware-validation/<gfx-target>/` -- do not attempt to optimize
past a blocker during initial validation.
