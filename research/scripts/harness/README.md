# Benchmark harness

Migrated from the runtime repo's research tree (not from runtime source):

- `generate_corpus.sh` -- produces the fixed corpus under `research/corpus/`
- `bench_r9700_q8.py` -- persistent-service harness: drives `--serve`,
  runs warmup + measured turns, writes `raw-runs.csv`
- `summarize_r9700_q8.py` -- consolidates `raw-runs.csv` into `summary.csv`
- `freeze_r9700_environment.sh` -- captures `environment.txt`/`build.txt`

`scripts/benchmark.sh` at the repo root is the intended entry point; these
scripts are the implementation it wraps.
