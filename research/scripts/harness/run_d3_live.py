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
    ap.add_argument("--tts")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--alsa", action="store_true")
    ap.add_argument("--no-renderer", action="store_true",
                    help="disable the D1 renderer worker for service attribution")
    ap.add_argument("--no-tts", action="store_true",
                    help="exclude native TTS model/work from a D3 service probe")
    ap.add_argument("--ngl", default="99")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="send at most this many 80 ms slices (0 exercises live_start only)")
    args = ap.parse_args()

    if not args.no_tts and not args.tts:
        ap.error("--tts is required unless --no-tts is selected")
    if args.no_tts and not args.no_renderer:
        ap.error("--no-tts requires --no-renderer")

    samples = pcm_f32(args.wav)
    # D3 has no end-of-input semantic yet; align only to full captured slices.
    samples = samples[: len(samples) // SLICE * SLICE]
    cmd = [args.binary, "-m", args.model, "--mmproj", args.mmproj, "-ngl", args.ngl, "--serve"]
    if not args.no_tts:
        cmd[5:5] = ["--tts", args.tts]
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

    def receive(expected: set[str]) -> dict:
        """Return the next protocol reply while preserving async events.

        The D1 renderer may emit `playback_begin` between a live-frame command
        and its `d3_frame` reply.  Those events are part of the evidence, not
        a response to a later command.
        """
        while True:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError(f"voicechat closed stdout while waiting for {sorted(expected)}")
            event = json.loads(line)
            events.append(event)
            if event.get("kind") in expected:
                return event

    ready = receive({"ready"})
    send({"cmd": "live_start", "alsa": args.alsa,
          "renderer": not args.no_renderer, "tts": not args.no_tts})
    receive({"d3_live_start"})

    capture0 = time.monotonic_ns() // 1000
    for index in range(0, len(samples), SLICE):
        if args.max_frames is not None and index // SLICE >= args.max_frames:
            break
        capture_us = capture0 + (index // SLICE) * 80_000
        send({"cmd": "live_frame", "capture_us": capture_us,
              "pcm_f32": samples[index:index + SLICE].tolist()})
        receive({"d3_frame", "d3_frame_wait"})

    send({"cmd": "quit"})
    receive({"bye"})
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
