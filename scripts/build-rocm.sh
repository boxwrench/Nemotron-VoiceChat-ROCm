#!/usr/bin/env bash
# Clones boxwrench/llama-voicechat.cpp at the pinned commit (see
# runtime/README.md) and builds it with HIP for the target GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUNTIME_REPO_URL="https://github.com/boxwrench/llama-voicechat.cpp.git"
RUNTIME_DIR="build/runtime-src"
GPU_TARGETS="${GPU_TARGETS:-gfx1201}"

PIN="$(grep -A0 '^Pinned known-good:' runtime/README.md | awk '{print $3}')"
if [[ -z "$PIN" ]]; then
    echo "build-rocm.sh: could not read pinned commit from runtime/README.md" >&2
    exit 1
fi

if [[ -d "$RUNTIME_DIR/.git" ]]; then
    CURRENT="$(git -C "$RUNTIME_DIR" rev-parse HEAD)"
    if [[ "$CURRENT" != "$PIN" ]]; then
        echo "build-rocm.sh: $RUNTIME_DIR is checked out at $CURRENT," >&2
        echo "  but runtime/README.md pins $PIN." >&2
        echo "  Refusing to silently build a different revision. Remove" >&2
        echo "  $RUNTIME_DIR (or 'git checkout $PIN' inside it) and re-run." >&2
        exit 1
    fi
    echo "build-rocm.sh: $RUNTIME_DIR already at pinned commit $PIN"
else
    echo "build-rocm.sh: cloning $RUNTIME_REPO_URL"
    mkdir -p build
    git clone "$RUNTIME_REPO_URL" "$RUNTIME_DIR"
    git -C "$RUNTIME_DIR" checkout "$PIN"
fi

# Matches the layout the frozen R9700-Q8-M1 baseline and the benchmark
# harness (research/scripts/harness/bench_r9700_q8.py) both assume:
# build/hip-<target>/bin/llama-voicechat, directly under the repo root.
BUILD_DIR="build/hip-$GPU_TARGETS"
echo "build-rocm.sh: configuring $BUILD_DIR (GPU_TARGETS=$GPU_TARGETS)"
cmake -S "$RUNTIME_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON \
    -DGPU_TARGETS="$GPU_TARGETS" \
    -DLLAMA_CURL=OFF

echo "build-rocm.sh: building llama-voicechat"
cmake --build "$BUILD_DIR" --target llama-voicechat -j"$(nproc)"

BIN="$BUILD_DIR/bin/llama-voicechat"
if [[ ! -x "$BIN" ]]; then
    echo "build-rocm.sh: build finished but $BIN was not produced" >&2
    exit 1
fi

echo "build-rocm.sh: done. Binary: $BIN"
echo "build-rocm.sh: note -- this GPU currently requires GGML_CUDA_DISABLE_GRAPHS=1 --no-warmup"
echo "  at run time (see docs/TROUBLESHOOTING.md); this is a known runtime workaround,"
echo "  not something the build itself can fix."
