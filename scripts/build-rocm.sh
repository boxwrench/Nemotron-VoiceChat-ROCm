#!/usr/bin/env bash
# Clones boxwrench/llama-voicechat.cpp at the pinned commit (see
# runtime/README.md) and builds it with HIP for the target GPU.
#
# TODO: implement. Expected shape:
#   1. read pinned commit from runtime/README.md
#   2. clone/fetch boxwrench/llama-voicechat.cpp into a local build dir
#   3. git checkout <pinned commit>
#   4. cmake/HIP build for GPU_TARGETS (default gfx1201, override via env/arg)
set -euo pipefail
echo "build-rocm.sh: not yet implemented, see comments in this file" >&2
exit 1
