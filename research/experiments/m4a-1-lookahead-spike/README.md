# M4A-1: perception lookahead spike

Frozen research result. See
[docs/M4-DUPLEX-DESIGN.md](../../../docs/M4-DUPLEX-DESIGN.md) M4-0A for why
this question exists: the FastConformer/parakeet perception encoder is
bidirectional with no cross-call state, so exact online-equivalent
embeddings are impossible in principle for a live frame the instant it
arrives. M4-0A characterized four strategies to approximate this (fixed
lookahead, bounded sliding context, causal encoder modification, replay);
M4A-1 is the empirical measurement that picks among them.

## Question

How much future audio context does a perception embedding need before it
stabilizes close to its full-clip (offline) value?

## Classification

**PROMOTE zero-lookahead.**

> Exact online equivalence remains impossible in principle for a
> bidirectional encoder, but measured future-context dependence is
> negligible relative to natural embedding variation across all tested
> speech and boundary conditions.

"Zero-lookahead" means *no future audio context* is required beyond the
frame's own arrival -- it does not mean zero latency. The current 80ms of
audio still has to arrive, and the growing prefix seen so far still has
to be (re-)encoded; that cost is what M4A-2 measures next.

## Method

Two passes, both file-driven against frozen corpus WAVs, both using a
throwaway runtime debug hook (`VC_DUMP_EMBD`, see "Reproducing" below) to
dump raw per-frame perception embeddings before the LLM loop runs.

For a chosen historical frame index F in a WAV: encode the *full* clip
once (reference embedding at F), then truncate the same clip to
`F's timestamp + lookahead` and re-encode that truncation (candidate
embedding at F). Compare candidate vs. reference. This directly measures
what a live system would have if it waited `lookahead` ms of extra audio
before committing to frame F's embedding.

### Pass 1 -- initial sweep (`sweep.py` / `m4a1_lookahead_sweep.json`)

- Corpus: `VC03-long.wav` (28.32s monologue, 354 frames).
- Frame positions: 3, chosen uniformly (early=40 ~3.2s, mid=175 ~14.0s,
  late=300 ~24.0s).
- Lookaheads: 0, 80, 160, 240, 320, 480, 640, 1000, 2000 ms.
- Calibration: natural adjacent-frame cosine similarity within the same
  clip, to confirm cosine similarity is a meaningful metric in this
  4480-dim embedding space (i.e. not saturated near 1.0 for unrelated
  frames) before trusting it for the lookahead comparison.

### Pass 2 -- adversarial follow-up (`adversarial_sweep.py` /
`m4a1_adversarial_sweep.json`)

Pass 1 used arbitrary frame positions; pass 2 deliberately targets the
transitions most likely to expose future-context dependence, across
three WAVs chosen for distinct content:

- `VC01-short.wav` (clean, 2.96s): before speech onset, first voiced
  frame, mid-word, sentence-boundary/last-voiced frame.
- `VC04-noisy.wav` (pink-noise-mixed, 2.96s): the same four categories
  plus a pure-noise-floor region with no speech at all.
- `VC05-pause.wav` (7.94s, has an internal ~2.5s pause): before onset,
  first voiced, mid-word, last frame before the pause, first silent
  frame of the pause, first frame after the pause resumes, and the
  final sentence-boundary frame.

Frame indices were chosen from 80ms-frame RMS energy profiles of each
WAV (not uniform sampling) -- see the per-frame comments in `WAVS` at
the top of `adversarial_sweep.py` for the exact index/timestamp/RMS
value used to identify each category.

Lookaheads: a reduced sweep (0, 80, 160, 320, 640, 1000 ms) with an
automatic extension to (1500, 2000, 3000 ms) for any frame whose
1000ms-lookahead point still showed `cosine_sim < 0.999` or
`rmse > 0.002` against the pass-1 reference band -- this trigger never
fired, so no frame needed the extended sweep.

Calibration was recomputed per-WAV (not reused from pass 1), sampling
natural adjacent-frame cosine similarity at each tested frame index, so
the "is this deviation material" judgment is made against that WAV's own
content, not an arbitrary universal threshold.

## Results

### Pass 1 (VC03-long.wav, 3 positions x 9 lookaheads)

Cosine similarity to the full-clip reference was already **~0.9999 at
0ms lookahead**, RMSE ~0.0004, max error ~0.0014-0.0022, uniformly
across all three frame positions. Extending lookahead to 2000ms produced
no meaningful further improvement -- movement was noise-level jitter,
not a convergence trend. Perception encode wall time tracked total clip
length (300-390ms for this clip), not lookahead, as expected for a
whole-clip re-encode.

Calibration: natural adjacent-frame cosine similarity within VC03 ranged
0.23-0.98 depending on content, confirming the metric has real dynamic
range and the ~0.9999 lookahead result is not an artifact of a saturated
metric.

### Pass 2 (VC01/VC04/VC05, 16 transition frames x 6 lookaheads)

Cosine similarity across all 16 frames x 6 lookaheads: **0.99970 to
1.00000**, RMSE 0.00000-0.00114, max error 0.0000-0.0044. The single
worst case anywhere was `mid_word_seg1` in VC05-pause (the noisiest,
most speech-dense category tested) at cos>=0.9997 -- still the same
order of magnitude as pass 1's ~0.9999/0.0004 reference band, not
materially worse. No lookahead-vs-error trend was observed at any
category, including onset, mid-word, pre/post-pause, first-silence, and
pure-noise frames.

New per-WAV calibration (broader than pass 1's, spanning onset/pause/
noise content specifically): natural adjacent-frame cosine similarity
ranged **0.53-0.9994** depending on content -- e.g. VC01 frame 12
(mid-word) vs. its neighbor: 0.5349; VC05 frame 10 (mid-word) vs. next:
0.6342. Every lookahead-induced deviation measured across both passes is
one to two orders of magnitude smaller than this natural variation, at
every category tested.

### Downstream checks

Pass 1 included one qualitative spot check: a full LLM+TTS turn on the
full VC03 clip vs. on a genuinely mid-sentence truncation (14.16s into
28.32s, not a lookahead artifact -- this tests "does truncated input
produce a safe response," not lookahead sensitivity specifically). The
full clip produced a complete, coherent response; the truncated clip
correctly produced `</s>` rather than a premature or hallucinated
answer.

Pass 2's extend-on-deviation trigger never fired (no frame's embedding
deviation crossed into "material" territory relative to either
reference band), so per the task's own rule, no further downstream
STT/VoiceChat checks were run in pass 2 -- the embedding evidence never
required them.

## Reproducing

Requires this repo's frozen corpus (`research/corpus/`) and a HIP/ROCm
build of the runtime fork with one small throwaway debug hook applied
(not shipped, not committed to the runtime repo -- see below).

**Runtime provenance**: `boxwrench/llama-voicechat.cpp`, commit
`5cc03186a` (`bench/r9700-q8-baseline`, the same commit the frozen
R9700-Q8-M1 baseline was measured against) with the patch below applied
on top on a local-only branch `scratch/m4a-1-lookahead-spike`, never
committed or pushed. This is *not* the officially pinned commit in
[runtime/README.md](../../../runtime/README.md) (`38a76719e`) -- that pin
is for the product's build/install path; this spike ran against the
baseline-freeze commit already built and resident on the reference
workstation.

**Hardware**: R9700 (gfx1201, `ROCR_VISIBLE_DEVICES=1` on the same
dual-GPU workstation as R9700-Q8-M1), build target `hip-gfx1201`,
`GGML_CUDA_DISABLE_GRAPHS=1` (per the R9700-Q8-M1 baseline's documented
requirement).

**Debug hook** (apply to `tools/voicechat/voicechat-cli.cpp` after the
perception-encode block in `vc_session::run_turn`, immediately after the
existing `LOG_INF("perception: ...")` line, before the turn counter
increment):

```cpp
// VC_DUMP_EMBD=<path>: M4A-1 spike only, write the raw per-frame embeddings
// (int32 n_frames, int32 n_embd header, then n_frames*n_embd float32) and
// stop before the LLM loop. Throwaway, not part of any shipped path.
if (const char * dump_path = getenv("VC_DUMP_EMBD")) {
    FILE * f = fopen(dump_path, "wb");
    if (f) {
        int32_t hdr[2] = { n_frames, n_embd };
        fwrite(hdr, sizeof(int32_t), 2, f);
        fwrite(aud.data(), sizeof(float), aud.size(), f);
        fclose(f);
    }
    return true;
}
```

With that applied and rebuilt, `sweep.py` and `adversarial_sweep.py` in
this directory reproduce pass 1 and pass 2 respectively. Both scripts
hardcode this workstation's build/model paths (`BIN`/`MODEL`/`MMPROJ` at
the top of each file) -- update those for a different machine. Raw
output is `m4a1_lookahead_sweep.json` and `m4a1_adversarial_sweep.json`
in this directory.

## Next

M4A-2: perception is now known to need no future context, but the
runtime still has no encoder-side state, so a live implementation must
re-encode the growing prefix on every new frame. M4A-2 measures whether
that naive re-encode strategy stays inside the ~80ms causal budget (see
docs/M4-DUPLEX-DESIGN.md, "The live-timeline causality invariant") as
conversation length grows, before any more elaborate windowing or
encoder-surgery approach is considered.
