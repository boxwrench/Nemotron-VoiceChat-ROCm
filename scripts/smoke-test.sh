#!/usr/bin/env bash
# Minimal pass/fail check for a built + converted install: BUILD is implied
# by getting here; this script checks LOAD, STT, TTS, and full S2S.
# No benchmarking -- see benchmark.sh for timing.
#
# This is the same bar used for new-hardware validation (docs/HARDWARE.md).
#
# TODO: implement. Expected shape:
#   1. start `llama-voicechat --serve` against the converted Q8 GGUFs
#   2. LOAD:  process reaches ready
#   3. STT:   send research/corpus/VC01-short.wav, expect a text response
#   4. TTS:   request TTS-only output, expect non-empty audio
#   5. S2S:   full speech-to-speech turn, expect non-empty audio + text
#   6. report BUILD/LOAD/STT/TTS/S2S as PASS/FAIL, stop at first failure
set -euo pipefail
echo "smoke-test.sh: not yet implemented, see comments in this file" >&2
exit 1
