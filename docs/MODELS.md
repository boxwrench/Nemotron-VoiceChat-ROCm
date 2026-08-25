# Models

Model: NVIDIA NemotronLabs VoiceChat 11B, Q8_0 quantization.

Model weights are never committed to or redistributed through this Git
repository. `scripts/download-q8.sh` fetches the source GGUF and required
tokenizer metadata from Hugging Face; `scripts/convert-q8.sh` runs the
Q8 split/conversion steps and verifies output hashes against a manifest.

## Provenance

- **Original model:** NVIDIA NemotronLabs VoiceChat 11B
  ([nvidia/NVIDIA-NemotronLabs-VoiceChat-11B](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B)),
  SafeTensors, 1632 tensors, fine-tuned from
  `nvidia/NVIDIA-Nemotron-Nano-9B-v2`.
- **Q8 GGUF source used for this release:**
  [`hoidhxd/NVIDIA-NemotronLabs-VoiceChat-11B-GGUF`](https://huggingface.co/hoidhxd/NVIDIA-NemotronLabs-VoiceChat-11B-GGUF),
  file `nemotron_voicechat_11b-Q8_0.gguf`, community-converted single-container
  GGUF (all five component models in one file; `scripts/convert-q8.sh` splits
  it into the four files the runtime wants -- see
  [docs/ARCHITECTURE.md](ARCHITECTURE.md)).
- **Revision:** repo commit `89883a05a031557729771f94abb9998e4facdd45`, as
  resolved from the Hub API at the time this was written. The GGUF repo does
  not tag stable releases, so `scripts/download-q8.sh` pins this exact
  revision rather than following the repo tip.
- **Source SHA256:** `47018a356c2d18ab4831ee2aa964a78baa35382f5d3c036697ccc248cde1479d`
  (`nemotron_voicechat_11b-Q8_0.gguf`), matching
  `research/baselines/R9700-Q8-M1/artifact-hashes.txt`.
- **Tokenizer metadata:** `nvidia/NVIDIA-Nemotron-Nano-9B-v2`
  (`config.json`, `tokenizer.json`, `tokenizer_config.json`) -- the VoiceChat
  GGUF carries no tokenizer of its own; `scripts/convert-q8.sh` reads these
  via `--ref-dir`.

This repository does not redistribute weights in any form: `scripts/
download-q8.sh` only fetches from the Hugging Face sources above, and
`scripts/convert-q8.sh`'s output stays local and gitignored.

## Frozen artifact hashes (R9700-Q8-M1 baseline)

See `research/baselines/R9700-Q8-M1/artifact-hashes.txt` for the exact
source and generated GGUF SHA256 identities used to produce that baseline's
results. `docs/BENCHMARKS.md` links the same evidence.

## Licensing

The original model (`nvidia/NVIDIA-NemotronLabs-VoiceChat-11B`) is
distributed under the **OpenMDW License Agreement, version 1.1
(OpenMDW-1.1)**, per that repository's `LICENSE` file and Hub license tag.
In summary (the `LICENSE` file there is the authoritative text, not this
paraphrase): a permissive grant to deal in the "Model Materials" (the model,
its architecture/parameters, and related artifacts) without restriction,
conditioned on retaining the license text and origin notices in any
redistribution, with a patent-litigation-triggered termination clause, no
restrictions on outputs generated using the model, and the model provided
"AS IS" with no warranty.

The GGUF conversion repository (`hoidhxd/NVIDIA-NemotronLabs-VoiceChat-11B-GGUF`)
states no separate license for the converted files themselves beyond
crediting the original model; this project treats the OpenMDW-1.1 terms of
the original model as governing the weights regardless of container format.
**This is our own reading, not a legal opinion -- verify independently
before any use with different risk tolerance than personal/research use.**

This is separate from this repository's own [LICENSE](../LICENSE), which
covers only the code in this repository. This repository never
redistributes model weights regardless of their license.
