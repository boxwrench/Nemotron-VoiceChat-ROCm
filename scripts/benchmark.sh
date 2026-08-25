#!/usr/bin/env bash
# Thin wrapper around the benchmark harness under research/scripts/harness/
# (bench_r9700_q8.py, summarize_r9700_q8.py). Reproduces the numbers in
# docs/BENCHMARKS.md against a running `--serve` process, without touching
# the frozen baseline evidence under research/baselines/R9700-Q8-M1/.
#
# Optional: this is not part of the required v0.1 install path
# (scripts/setup.sh does not call this). GPU_TARGETS is currently fixed to
# gfx1201 by the harness itself, matching the only validated reference.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

HARNESS="research/scripts/harness"
CORPUS="research/corpus"
BASELINE="research/baselines/R9700-Q8-M1"
RUN_DIR="research/baselines/R9700-Q8-M1/generated/run-$(date +%Y%m%d-%H%M%S)"

if [[ ! -x "build/hip-gfx1201/bin/llama-voicechat" ]]; then
    echo "benchmark.sh: build/hip-gfx1201/bin/llama-voicechat not found." >&2
    echo "  Run scripts/setup.sh (or scripts/build-rocm.sh) first." >&2
    exit 1
fi
if [[ ! -f "$CORPUS/VC01-short.wav" ]]; then
    echo "benchmark.sh: fixed corpus not found, generating it (research/scripts/harness/generate_corpus.sh)"
    "$HARNESS/generate_corpus.sh"
fi

mkdir -p "$RUN_DIR"

run_case() {
    local dir="$1" audio="$2" warmups="$3" runs="$4"
    shift 4
    echo "benchmark.sh: running $dir ($audio, $warmups warmup + $runs measured)"
    python3 "$HARNESS/bench_r9700_q8.py" \
        --audio "$CORPUS/$audio" \
        --warmups "$warmups" --runs "$runs" \
        --output-dir "$RUN_DIR/$dir" \
        "$@"
}

run_case primary            VC01-short.wav        3 20
run_case VC02-conversation  VC02-conversation.wav  1 3
run_case VC03-long          VC03-long.wav          1 3
run_case VC04-noisy         VC04-noisy.wav         1 3
run_case VC05-pause         VC05-pause.wav         1 3
run_case VC06-tool          VC06-tool.wav          1 3 \
    --system-file "$CORPUS/VC06-system.txt" \
    --tool-response-file "$CORPUS/VC06-tool-response.txt"

echo "benchmark.sh: summarizing"
python3 "$HARNESS/summarize_r9700_q8.py" \
    --baseline-dir "$BASELINE" \
    --generated-dir "$RUN_DIR" \
    --output-dir "$RUN_DIR"

echo
echo "benchmark.sh: this run's summary: $RUN_DIR/summary.csv"
echo "benchmark.sh: frozen baseline:    $BASELINE/summary.csv"
if command -v diff >/dev/null 2>&1; then
    echo "benchmark.sh: diff (this run vs frozen baseline, informational only):"
    diff "$BASELINE/summary.csv" "$RUN_DIR/summary.csv" || true
fi
echo "benchmark.sh: done. This run's raw data was NOT written over the frozen"
echo "  baseline evidence -- it lives entirely under $RUN_DIR."
