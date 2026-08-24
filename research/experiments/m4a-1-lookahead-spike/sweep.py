#!/usr/bin/env python3
"""M4A-1 lookahead sweep: how much future audio do perception embeddings need
to stabilize. Throwaway spike script, not part of any shipped path."""
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
CORPUS_WAV = "/ai/github/Nemotron-VoiceChat-ROCm/research/corpus/VC03-long.wav"
WORKDIR = Path("/tmp/claude-1000/-ai/176b2eae-5a13-420f-89c0-b73a916f941f/scratchpad/m4a1")

ENV = os.environ.copy()
ENV["ROCR_VISIBLE_DEVICES"] = "1"
ENV["GGML_CUDA_DISABLE_GRAPHS"] = "1"

FRAME_HZ = 12.5
LOOKAHEADS_MS = [0, 80, 160, 240, 320, 480, 640, 1000, 2000]
FRAME_INDICES = {"early": 40, "mid": 175, "late": 300}  # ~3.2s, ~14.0s, ~24.0s


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


def main():
    WORKDIR.mkdir(parents=True, exist_ok=True)
    full_dump = WORKDIR / "ref_full.bin"
    print("=== reference: full-clip encode ===")
    wall_s, encode_ms = run_encode(Path(CORPUS_WAV), full_dump)
    n_frames_ref, n_embd, ref_rows = read_dump(full_dump)
    print(f"full clip: n_frames={n_frames_ref} n_embd={n_embd} "
          f"encode_ms={encode_ms} wall_s={wall_s:.2f}")

    results = {}
    for label, frame_idx in FRAME_INDICES.items():
        frame_time_s = (frame_idx + 1) / FRAME_HZ
        ref_embd = ref_rows[frame_idx]
        results[label] = {"frame_idx": frame_idx, "frame_time_s": frame_time_s, "points": []}
        print(f"\n=== frame '{label}' idx={frame_idx} (~{frame_time_s:.2f}s) ===")
        for lookahead_ms in LOOKAHEADS_MS:
            trunc_duration = frame_time_s + lookahead_ms / 1000.0
            trunc_wav = WORKDIR / f"trunc_{label}_{lookahead_ms}ms.wav"
            dump_path = WORKDIR / f"dump_{label}_{lookahead_ms}ms.bin"
            n_kept = truncate_wav(Path(CORPUS_WAV), trunc_wav, trunc_duration)
            wall_s, encode_ms = run_encode(trunc_wav, dump_path)
            n_frames_t, _, rows_t = read_dump(dump_path)
            if frame_idx >= n_frames_t:
                print(f"  lookahead={lookahead_ms:5d}ms: frame {frame_idx} not yet "
                      f"present (n_frames_t={n_frames_t}), skipping")
                continue
            cand_embd = rows_t[frame_idx]
            cos = cosine(ref_embd, cand_embd)
            rmse, maxerr = rmse_max(ref_embd, cand_embd)
            row = {
                "lookahead_ms": lookahead_ms,
                "trunc_duration_s": round(trunc_duration, 3),
                "n_frames_truncated": n_frames_t,
                "cosine_sim": cos,
                "rmse": rmse,
                "max_err": maxerr,
                "encode_ms": encode_ms,
                "wall_s": round(wall_s, 3),
            }
            results[label]["points"].append(row)
            print(f"  lookahead={lookahead_ms:5d}ms: cos={cos:.6f} rmse={rmse:.5f} "
                  f"maxerr={maxerr:.4f} encode_ms={encode_ms} wall_s={wall_s:.2f}")

    out_path = WORKDIR / "m4a1_lookahead_sweep.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nraw results written to {out_path}")


if __name__ == "__main__":
    main()
