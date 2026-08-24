# Research

- `baselines/` -- frozen, reproducible performance/experience baselines.
  Each is a named directory (e.g. `R9700-Q8-M1`) with its own README,
  raw data, and manifests. Nothing here is regenerated silently; a new
  baseline gets a new directory.
- `hardware-validation/` -- one directory per gfx target, tracking the M2
  validation bar (BUILD/LOAD/STT/TTS/S2S/memory/rough latency/correctness)
  independent of full baselines.
- `experiments/` -- speculative or one-off investigations, not yet frozen.
- `corpus/` -- the fixed, hash-verified benchmark input set (VC01-VC06)
  shared by all baselines and hardware-validation runs.
- `scripts/harness/` -- the benchmark/measurement harness itself
  (`bench_r9700_q8.py`, `summarize_r9700_q8.py`, `generate_corpus.sh`,
  `freeze_r9700_environment.sh`), migrated from the runtime repo's research
  tree. `scripts/benchmark.sh` at the repo root wraps this harness.
