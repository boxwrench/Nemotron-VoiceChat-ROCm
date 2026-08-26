#!/usr/bin/env python3
"""Drive the D3-0 serve protocol with fixed 80 ms PCM from a WAV fixture.

This is a controlled replay harness, not a microphone client. It deliberately
uses the same 1,280-sample authorization quantum as live capture so every
ordinary timeline step has one recorded capture slice behind it.
"""
import argparse
import array
import json
import subprocess
import sys
import time

SAMPLE_RATE = 16000
SLICE = 1280


def pcm_f32(path: str) -> array.array:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-f", "f32le", "-"],
        check=True, stdout=subprocess.PIPE)
    data = array.array("f")
    data.frombytes(proc.stdout)
    if sys.byteorder != "little":
        data.byteswap()
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--tts", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--alsa", action="store_true")
    ap.add_argument("--ngl", default="99")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="send at most this many 80 ms slices (0 exercises live_start only)")
    args = ap.parse_args()

    samples = pcm_f32(args.wav)
    # D3 has no end-of-input semantic yet; align only to full captured slices.
    samples = samples[: len(samples) // SLICE * SLICE]
    cmd = [args.binary, "-m", args.model, "--mmproj", args.mmproj,
           "--tts", args.tts, "-ngl", args.ngl, "--serve"]
    # Runtime/model loading is verbose. Keeping stderr in a pipe can fill it
    # before the JSON `ready` event arrives and deadlock the harness. Retain
    # diagnostics in a sibling artifact without putting them on the handshake
    # path.
    stderr_path = args.events + ".stderr.txt"
    stderr_file = open(stderr_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=stderr_file, text=True, bufsize=1)
    assert proc.stdin and proc.stdout
    events = []

    def send(obj: dict) -> None:
        proc.stdin.write(json.dumps(obj, separators=(",", ":")) + "\n")
        proc.stdin.flush()

    ready = json.loads(proc.stdout.readline())
    events.append(ready)
    if ready.get("kind") != "ready":
        raise RuntimeError(f"expected ready, got {ready}")
    send({"cmd": "live_start", "alsa": args.alsa})
    events.append(json.loads(proc.stdout.readline()))

    capture0 = time.monotonic_ns() // 1000
    for index in range(0, len(samples), SLICE):
        if args.max_frames is not None and index // SLICE >= args.max_frames:
            break
        capture_us = capture0 + (index // SLICE) * 80_000
        send({"cmd": "live_frame", "capture_us": capture_us,
              "pcm_f32": samples[index:index + SLICE].tolist()})
        events.append(json.loads(proc.stdout.readline()))

    send({"cmd": "quit"})
    events.append(json.loads(proc.stdout.readline()))
    proc.stdin.close()
    rc = proc.wait()
    stderr_file.close()
    with open(args.events, "w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")
    if rc:
        raise RuntimeError(f"voicechat exited {rc}; see {args.events}.stderr.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
