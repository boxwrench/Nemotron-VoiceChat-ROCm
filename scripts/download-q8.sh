#!/usr/bin/env bash
# Downloads the source GGUF and tokenizer metadata from Hugging Face.
# See docs/MODELS.md for provenance. Never commits weights to Git.
#
# TODO: implement. Expected shape:
#   1. read source repo/revision from docs/MODELS.md
#   2. hf download (or huggingface_hub) into a local, gitignored models dir
#   3. verify downloaded file hashes where already known
set -euo pipefail
echo "download-q8.sh: not yet implemented, see comments in this file" >&2
exit 1
