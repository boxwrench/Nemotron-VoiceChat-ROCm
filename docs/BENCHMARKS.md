# Benchmarks

## R9700-Q8-M1 (reference)

AMD Radeon AI PRO R9700 (gfx1201), ROCm 7.2.1, Nemotron VoiceChat 11B Q8_0.
Full evidence, raw per-turn data, and methodology:
[research/baselines/R9700-Q8-M1/README.md](../research/baselines/R9700-Q8-M1/README.md).

| Metric | Value |
| --- | --- |
| Fresh process to ready, warm page cache | 1.543 s |
| Warm speech-to-speech, mean | 4.244 s |
| Warm speech-to-speech, p95 | 4.264 s |
| First response text event, mean | 2.088 s |
| Perception (FastConformer) | 19.9 ms |
| Peak VRAM | 14.45 GiB |
| Peak host RSS during load | 9.10 GiB |
| Telemetry observer overhead | +0.27% (admissible) |

35/35 measured turns across six deterministic cases succeeded, each with one
stable text result and one stable WAV result. Function/tool channel:
QUALIFIED (correct tool selection and consumption; malformed JSON, missing
a colon after `arguments` -- see
[docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md)).

## Reproducing this baseline

```
scripts/benchmark.sh
```

runs the fixed corpus (`research/corpus/`) through the harness under
`research/scripts/harness/` against a running `--serve` process built from
the pinned runtime commit, and compares output against
`research/baselines/R9700-Q8-M1/summary.csv`.

## Other hardware

No benchmark results yet for gfx1100 or gfx1151. See
[docs/HARDWARE.md](HARDWARE.md) for the M2 validation plan.
