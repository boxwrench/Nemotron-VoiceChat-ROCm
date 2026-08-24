#!/usr/bin/env python3
"""M4A-1 adversarial follow-up: deliberately-chosen transition frames (speech
onset, mid-word, pre/post-pause, silence, sentence boundary, noisy region)
across VC01/VC04/VC05, reduced lookahead sweep, extended for any frame that
deviates materially from the initial sweep's noise-level behavior.
Throwaway spike script, not part of any shipped path."""
import json
import os
import struct
import subprocess
import time
import wave
from pathlib import Path

BIN = "/ai/github/llama-voicechat.cpp-ROCm/build/hip-gfx1201/bin/llama-voicechat"
MODEL = "/ai/github/llama-voicechat.cpp-ROCm/models/voicechat-q8/runtime/nemotron_voicechat_11b-stt-llm-Q8_0.gguf"
MMPROJ = "/ai/github/llama-voicechat.cpp-ROCm/models/voicechat-q8/runtime/mmproj-voicechat-perception-Q8_0.gguf"
CORPUS_DIR = "/ai/github/Nemotron-VoiceChat-ROCm/research/corpus"
WORKDIR = Path("/tmp/claude-1000/-ai/176b2eae-5a13-420f-89c0-b73a916f941f/scratchpad/m4a1_adv")

ENV = os.environ.copy()
ENV["ROCR_VISIBLE_DEVICES"] = "1"
ENV["GGML_CUDA_DISABLE_GRAPHS"] = "1"

FRAME_HZ = 12.5
LOOKAHEADS_MS = [0, 80, 160, 320, 640, 1000]
EXTENDED_LOOKAHEADS_MS = [1500, 2000, 3000]

# Deliberately chosen from 80ms-frame RMS energy profiles (see energy.py output),
# not uniform sampling. Categories per corpus README: VC01 clean short, VC04
# pink-noise-mixed, VC05 has a 2.5s internal pause.
WAVS = {
    "VC01-short.wav": {
        "before_onset": 2,        # t=0.24s rms=0.0, last silent frame pre-speech
        "first_voiced": 3,        # t=0.32s rms=478.8, onset
        "mid_word": 12,           # t=1.04s rms=909.8, mid-utterance
        "sentence_boundary_end": 21,  # t=1.76s rms=102.9, last voiced frame
    },
    "VC04-noisy.wav": {
        "noisy_before_onset": 0,      # t=0.08s rms=21.1, pink-noise floor only
        "noisy_first_voiced": 3,      # t=0.32s rms=355.2, onset over noise
        "noisy_mid_word": 12,         # t=1.04s rms=675.2, mid-utterance over noise
        "noisy_sentence_boundary_end": 21,  # t=1.76s rms=79.5, tail over noise
        "noisy_region_only": 25,      # t=2.08s rms=22.9, pure noise, no speech
    },
    "VC05-pause.wav": {
        "before_onset": 2,        # t=0.24s rms=0.0
        "first_voiced": 3,        # t=0.32s rms=491.4
        "mid_word_seg1": 10,      # t=0.88s rms=787.6, first-segment speech
        "before_pause": 12,       # t=1.04s rms=486.6, last strong frame pre-pause
        "first_silence": 14,      # t=1.20s rms=0.0, first frame of the pause
        "after_pause": 66,        # t=5.36s rms=83.1, first frame of renewed energy
        "sentence_boundary_end": 83,  # t=6.72s rms=204.9, last strongly voiced frame
    },
}


def truncate_wav(src: Path, dst: Path, duration_s: float) -> int:
    with wave.open(str(src), "rb") as w:
        params = w.getparams()
        n_keep = min(params.nframes, int(duration_s * params.framerate))
        frames = w.readframes(n_keep)
    with wave.open(str(dst), "wb") as w:
        w.setparams(params)
        w.writeframes(frames)
    return n_keep


def run_encode(wav_path: Path, dump_path: Path):
    env = ENV.copy()
    env["VC_DUMP_EMBD"] = str(dump_path)
    t0 = time.monotonic()
    proc = subprocess.run(
        [BIN, "-m", MODEL, "--mmproj", MMPROJ, "--no-warmup",
         "--device", "ROCm0", "--split-mode", "none", "--gpu-layers", "all",
         "--temp", "0", "--seed", "42", "--audio", str(wav_path)],
        env=env, capture_output=True, text=True, timeout=60,
    )
    wall_s = time.monotonic() - t0
    encode_ms = None
    for line in proc.stderr.splitlines():
        if "perception:" in line and " ms" in line:
            encode_ms = float(line.rsplit(" in ", 1)[1].rstrip(" ms\n"))
    if not dump_path.exists():
        raise RuntimeError(f"no dump produced for {wav_path}\nstderr:\n{proc.stderr[-2000:]}")
    return wall_s, encode_ms


def read_dump(path: Path):
    data = path.read_bytes()
    n_frames, n_embd = struct.unpack_from("<ii", data, 0)
    floats = struct.unpack_from(f"<{n_frames * n_embd}f", data, 8)
    rows = [floats[i * n_embd:(i + 1) * n_embd] for i in range(n_frames)]
    return n_frames, n_embd, rows


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na > 0 and nb > 0 else float("nan")


def rmse_max(a, b):
    diffs = [x - y for x, y in zip(a, b)]
    sq = [d * d for d in diffs]
    rmse = (sum(sq) / len(sq)) ** 0.5
    maxerr = max(abs(d) for d in diffs)
    return rmse, maxerr


def sweep_point(ref_embd, wav_src, frame_time_s, lookahead_ms, label_tag, n_embd_expected):
    trunc_duration = frame_time_s + lookahead_ms / 1000.0
    trunc_wav = WORKDIR / f"trunc_{label_tag}_{lookahead_ms}ms.wav"
    dump_path = WORKDIR / f"dump_{label_tag}_{lookahead_ms}ms.bin"
    truncate_wav(wav_src, trunc_wav, trunc_duration)
    wall_s, encode_ms = run_encode(trunc_wav, dump_path)
    n_frames_t, _, rows_t = read_dump(dump_path)
    return n_frames_t, rows_t, wall_s, encode_ms, trunc_duration


def main():
    WORKDIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    calibration = {}

    for wav_name, frames in WAVS.items():
        wav_src = Path(CORPUS_DIR) / wav_name
        print(f"\n########## {wav_name} ##########")
        full_dump = WORKDIR / f"ref_full_{wav_name}.bin"
        wall_s, encode_ms = run_encode(wav_src, full_dump)
        n_frames_ref, n_embd, ref_rows = read_dump(full_dump)
        print(f"reference full-clip: n_frames={n_frames_ref} n_embd={n_embd} "
              f"encode_ms={encode_ms} wall_s={wall_s:.2f}")

        # Natural adjacent-frame calibration for THIS wav's content, sampled at
        # a few points spanning its speech content (not just one).
        cal_pairs = []
        candidate_idxs = sorted(set(frames.values()))
        for idx in candidate_idxs:
            if 1 <= idx < n_frames_ref - 1:
                cos_prev = cosine(ref_rows[idx], ref_rows[idx - 1])
                cos_next = cosine(ref_rows[idx], ref_rows[idx + 1])
                cal_pairs.append({"frame_idx": idx, "cos_vs_prev": cos_prev, "cos_vs_next": cos_next})
        calibration[wav_name] = cal_pairs
        print("calibration (adjacent-frame cosine sim):")
        for c in cal_pairs:
            print(f"  frame {c['frame_idx']:3d}: cos_vs_prev={c['cos_vs_prev']:.4f} "
                  f"cos_vs_next={c['cos_vs_next']:.4f}")

        wav_results = {}
        for label, frame_idx in frames.items():
            frame_time_s = (frame_idx + 1) / FRAME_HZ
            ref_embd = ref_rows[frame_idx]
            label_tag = f"{wav_name.split('.')[0]}_{label}"
            print(f"\n=== {wav_name} / '{label}' idx={frame_idx} (~{frame_time_s:.2f}s) ===")
            points = []
            lookaheads = list(LOOKAHEADS_MS)
            i = 0
            while i < len(lookaheads):
                lookahead_ms = lookaheads[i]
                n_frames_t, rows_t, wall_s, encode_ms, trunc_dur = sweep_point(
                    ref_embd, wav_src, frame_time_s, lookahead_ms, label_tag, n_embd)
                if frame_idx >= n_frames_t:
                    print(f"  lookahead={lookahead_ms:5d}ms: frame not yet present, skipping")
                    i += 1
                    continue
                cand_embd = rows_t[frame_idx]
                cos = cosine(ref_embd, cand_embd)
                rmse, maxerr = rmse_max(ref_embd, cand_embd)
                row = {
                    "lookahead_ms": lookahead_ms, "trunc_duration_s": round(trunc_dur, 3),
                    "n_frames_truncated": n_frames_t, "cosine_sim": cos,
                    "rmse": rmse, "max_err": maxerr, "encode_ms": encode_ms,
                    "wall_s": round(wall_s, 3),
                }
                points.append(row)
                print(f"  lookahead={lookahead_ms:5d}ms: cos={cos:.6f} rmse={rmse:.5f} "
                      f"maxerr={maxerr:.4f} encode_ms={encode_ms} wall_s={wall_s:.2f}")
                i += 1
                # Extend if the LAST point in the base sweep still looks materially
                # off from the initial-sweep reference band; append extended points.
                if lookahead_ms == LOOKAHEADS_MS[-1] and (cos < 0.999 or rmse > 0.002):
                    print(f"  ** {label} at {lookahead_ms}ms still deviating (cos={cos:.6f}, "
                          f"rmse={rmse:.5f}) -- extending sweep **")
                    lookaheads.extend(EXTENDED_LOOKAHEADS_MS)
            wav_results[label] = {"frame_idx": frame_idx, "frame_time_s": frame_time_s, "points": points}
        all_results[wav_name] = wav_results

    out = {"sweep": all_results, "calibration": calibration,
           "lookaheads_ms": LOOKAHEADS_MS, "extended_lookaheads_ms": EXTENDED_LOOKAHEADS_MS}
    out_path = Path("/ai/github/Nemotron-VoiceChat-ROCm/research/experiments/m4a-1-lookahead-spike/m4a1_adversarial_sweep.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nraw results written to {out_path}")


if __name__ == "__main__":
    main()
