#!/usr/bin/env bash
# Runs the pinned runtime's four Q8 conversion steps
# (tools/voicechat/convert_voicechat_*.py) and verifies the four output
# SHA256 hashes against the frozen R9700-Q8-M1 evidence.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUNTIME_DIR="build/runtime-src"
MODEL_DIR="models/voicechat-q8"
SOURCE_GGUF="$MODEL_DIR/source/nemotron_voicechat_11b-Q8_0.gguf"
REF_DIR="$MODEL_DIR/ref-nano9b"
OUT_DIR="$MODEL_DIR/runtime"
HASHES="research/baselines/R9700-Q8-M1/artifact-hashes.txt"

if [[ ! -d "$RUNTIME_DIR" ]]; then
    echo "convert-q8.sh: $RUNTIME_DIR not found. Run scripts/build-rocm.sh first." >&2
    exit 1
fi
if [[ ! -f "$SOURCE_GGUF" ]]; then
    echo "convert-q8.sh: $SOURCE_GGUF not found. Run scripts/download-q8.sh first." >&2
    exit 1
fi
if [[ ! -f "$REF_DIR/tokenizer.json" ]]; then
    echo "convert-q8.sh: $REF_DIR (tokenizer metadata) not found. Run scripts/download-q8.sh first." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

LLM_OUT="$OUT_DIR/nemotron_voicechat_11b-stt-llm-Q8_0.gguf"
FHEAD_OUT="$OUT_DIR/nemotron_voicechat_11b-stt-llm-Q8_0-function-head.gguf"
MMPROJ_OUT="$OUT_DIR/mmproj-voicechat-perception-Q8_0.gguf"
TTS_OUT="$OUT_DIR/voicechat-tts-Q8_0.gguf"

echo "convert-q8.sh: extracting STT LLM + function head"
python3 "$RUNTIME_DIR/tools/voicechat/convert_voicechat_to_nemotron_h.py" \
    "$SOURCE_GGUF" --ref-dir "$REF_DIR" -o "$LLM_OUT"

echo "convert-q8.sh: extracting perception encoder (mmproj)"
python3 "$RUNTIME_DIR/tools/voicechat/convert_voicechat_perception_to_mmproj.py" \
    "$SOURCE_GGUF" -o "$MMPROJ_OUT"

echo "convert-q8.sh: extracting TTS backbone + codec"
python3 "$RUNTIME_DIR/tools/voicechat/convert_voicechat_tts_to_gguf.py" \
    "$SOURCE_GGUF" --ref-dir "$REF_DIR" -o "$TTS_OUT"

echo "convert-q8.sh: verifying output hashes against $HASHES"
FAILED=0
for name in \
    "models/voicechat-q8/runtime/mmproj-voicechat-perception-Q8_0.gguf:$MMPROJ_OUT" \
    "models/voicechat-q8/runtime/nemotron_voicechat_11b-stt-llm-Q8_0-function-head.gguf:$FHEAD_OUT" \
    "models/voicechat-q8/runtime/nemotron_voicechat_11b-stt-llm-Q8_0.gguf:$LLM_OUT" \
    "models/voicechat-q8/runtime/voicechat-tts-Q8_0.gguf:$TTS_OUT"; do
    manifest_key="${name%%:*}"
    actual_path="${name##*:}"
    expected="$(grep -F " $manifest_key" "$HASHES" | awk '{print $1}')"
    if [[ -z "$expected" ]]; then
        echo "  SKIP (no frozen hash recorded): $manifest_key" >&2
        continue
    fi
    actual="$(sha256sum "$actual_path" | awk '{print $1}')"
    if [[ "$actual" != "$expected" ]]; then
        echo "  MISMATCH: $actual_path" >&2
        echo "    expected: $expected" >&2
        echo "    actual:   $actual" >&2
        FAILED=1
    else
        echo "  OK: $actual_path"
    fi
done

if [[ "$FAILED" -ne 0 ]]; then
    echo "convert-q8.sh: one or more artifact hashes did not match the frozen manifest." >&2
    echo "  This can be expected (a genuine change in a converter) or a real problem" >&2
    echo "  (wrong source file, wrong runtime pin). Investigate before proceeding." >&2
    exit 1
fi

echo "convert-q8.sh: done, all four artifacts verified. Output: $OUT_DIR"
