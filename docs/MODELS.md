# Models

Model: NVIDIA NemotronLabs VoiceChat 11B, Q8_0 quantization.

Model weights are never committed to or redistributed through this Git
repository. `scripts/download-q8.sh` fetches the source GGUF and required
tokenizer metadata from Hugging Face; `scripts/convert-q8.sh` runs the
Q8 split/conversion steps and verifies output hashes against a manifest.

## Frozen artifact hashes (R9700-Q8-M1 baseline)

See `research/baselines/R9700-Q8-M1/artifact-hashes.txt` for the exact
source and generated GGUF SHA256 identities used to produce that baseline's
results. `docs/BENCHMARKS.md` links the same evidence.

## Licensing

Document the model's own license and usage terms here, separately from
this repository's own [LICENSE](../LICENSE) (which covers only the code in
this repository). Do not assume the model license permits redistribution of
weights; this repository never redistributes them regardless.

<!-- TODO: fill in exact Hugging Face source repo, revision, and license text/link -->
