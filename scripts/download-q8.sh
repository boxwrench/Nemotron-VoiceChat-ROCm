#!/usr/bin/env bash
# Downloads the source GGUF and tokenizer metadata from Hugging Face.
# See docs/MODELS.md for provenance. Never commits weights to Git.
#
# Source (see docs/MODELS.md):
#   repo:     hoidhxd/NVIDIA-NemotronLabs-VoiceChat-11B-GGUF
#   revision: 89883a05a031557729771f94abb9998e4facdd45
#   file:     nemotron_voicechat_11b-Q8_0.gguf
#
# Tokenizer metadata (needed by scripts/convert-q8.sh's --ref-dir):
#   repo: nvidia/NVIDIA-Nemotron-Nano-9B-v2
#   files: config.json, tokenizer.json, tokenizer_config.json
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SOURCE_REPO="hoidhxd/NVIDIA-NemotronLabs-VoiceChat-11B-GGUF"
SOURCE_REVISION="89883a05a031557729771f94abb9998e4facdd45"
SOURCE_FILE="nemotron_voicechat_11b-Q8_0.gguf"
SOURCE_SHA256="47018a356c2d18ab4831ee2aa964a78baa35382f5d3c036697ccc248cde1479d"

REF_REPO="nvidia/NVIDIA-Nemotron-Nano-9B-v2"
REF_FILES=(config.json tokenizer.json tokenizer_config.json)

MODEL_DIR="models/voicechat-q8"
SOURCE_DIR="$MODEL_DIR/source"
REF_DIR="$MODEL_DIR/ref-nano9b"

if ! command -v hf >/dev/null 2>&1; then
    echo "download-q8.sh: 'hf' (huggingface_hub CLI) not found on PATH." >&2
    echo "  install: curl -LsSf https://hf.co/cli/install.sh | bash -s" >&2
    echo "  auth (if the source repo requires it): hf auth login" >&2
    exit 1
fi

mkdir -p "$SOURCE_DIR" "$REF_DIR"

verify_sha256() {
    local path="$1" expected="$2"
    local actual
    actual="$(sha256sum "$path" | awk '{print $1}')"
    if [[ "$actual" != "$expected" ]]; then
        echo "download-q8.sh: SHA256 mismatch for $path" >&2
        echo "  expected: $expected" >&2
        echo "  actual:   $actual" >&2
        return 1
    fi
}

SOURCE_PATH="$SOURCE_DIR/$SOURCE_FILE"
if [[ -f "$SOURCE_PATH" ]] && verify_sha256 "$SOURCE_PATH" "$SOURCE_SHA256" 2>/dev/null; then
    echo "download-q8.sh: $SOURCE_PATH already present and hash-valid, skipping download"
else
    echo "download-q8.sh: fetching $SOURCE_FILE from $SOURCE_REPO@$SOURCE_REVISION"
    if ! hf download "$SOURCE_REPO" "$SOURCE_FILE" \
        --revision "$SOURCE_REVISION" \
        --local-dir "$SOURCE_DIR"; then
        echo "download-q8.sh: download failed. If this model requires authentication," >&2
        echo "  run 'hf auth login' (or set HF_TOKEN) and retry." >&2
        exit 1
    fi
    verify_sha256 "$SOURCE_PATH" "$SOURCE_SHA256"
    echo "download-q8.sh: $SOURCE_FILE verified against frozen R9700-Q8-M1 evidence"
fi

NEED_REF=0
for f in "${REF_FILES[@]}"; do
    [[ -f "$REF_DIR/$f" ]] || NEED_REF=1
done
if [[ "$NEED_REF" -eq 0 ]]; then
    echo "download-q8.sh: tokenizer metadata already present in $REF_DIR, skipping"
else
    echo "download-q8.sh: fetching tokenizer metadata from $REF_REPO"
    if ! hf download "$REF_REPO" "${REF_FILES[@]}" --local-dir "$REF_DIR"; then
        echo "download-q8.sh: tokenizer metadata download failed." >&2
        exit 1
    fi
fi

echo "download-q8.sh: done. Source GGUF: $SOURCE_PATH"
echo "download-q8.sh: this repository does not redistribute these weights; see docs/MODELS.md"
