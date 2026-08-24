#!/usr/bin/env bash
# Runs the runtime repo's four Q8 conversion steps
# (tools/voicechat/convert_voicechat_*.py, at the pinned commit) and
# verifies the four output SHA256 hashes against docs/MODELS.md.
#
# TODO: implement. Expected shape:
#   1. locate the built runtime checkout from build-rocm.sh
#   2. run convert_voicechat_to_nemotron_h.py
#   3. run convert_voicechat_perception_to_mmproj.py
#   4. run convert_voicechat_tts_to_gguf.py (produces function-head + tts)
#   5. sha256sum the four runtime GGUFs, compare against a manifest
set -euo pipefail
echo "convert-q8.sh: not yet implemented, see comments in this file" >&2
exit 1
