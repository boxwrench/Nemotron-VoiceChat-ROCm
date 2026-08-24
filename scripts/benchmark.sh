#!/usr/bin/env bash
# Thin wrapper around the migrated benchmark harness under
# research/scripts/harness/ (bench_r9700_q8.py, summarize_r9700_q8.py).
# Reproduces the numbers in docs/BENCHMARKS.md against a running
# `--serve` process.
#
# TODO: implement / generalize. Expected shape:
#   1. ensure a `--serve` process is running (see docs/INSTALL.md)
#   2. run research/scripts/harness/bench_r9700_q8.py against
#      research/corpus/ for each case, writing raw-runs.csv
#   3. run research/scripts/harness/summarize_r9700_q8.py to produce
#      summary.csv
#   4. diff against the frozen baseline under research/baselines/ for a
#      regression signal
set -euo pipefail
echo "benchmark.sh: not yet implemented, see comments in this file" >&2
exit 1
