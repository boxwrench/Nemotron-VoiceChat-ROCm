#!/usr/bin/env bash
# Orchestrates the deterministic setup flow. See docs/INSTALL.md.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

./download-q8.sh "$@"
./build-rocm.sh "$@"
./convert-q8.sh "$@"
./smoke-test.sh "$@"

echo "setup.sh: done. Run ./benchmark.sh to reproduce docs/BENCHMARKS.md."
