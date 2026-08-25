#!/usr/bin/env python3
"""Thin push-to-talk client for llama-voicechat --serve.

Hold SPACE to record, release to submit the turn, hear the response.
Does not touch inference/runtime code: it spawns the existing --serve
process once, keeps it resident for the whole session, and talks to it
over the documented stdin/stdout JSON-lines protocol (see
tools/voicechat/voicechat-cli.cpp in the runtime repo, vc_serve()).

Audio in/out uses the system's normal ALSA command-line tools (arecord/
aplay) rather than an audio framework. The only extra dependency is
pynput, used only for global hold/release key detection.
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


class ServeError(RuntimeError):
    pass


class Session:
    """Owns one long-lived `llama-voicechat --serve` process."""

    def __init__(self, command: list[str], env: dict[str, str]) -> None:
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # let runtime logs go straight to the terminal
            text=True,
            bufsize=1,
            env=env,
        )
        self._lock = threading.Lock()

    def _send(self, cmd: dict) -> None:
        assert self.process.stdin is not None
        with self._lock:
            self.process.stdin.write(json.dumps(cmd, separators=(",", ":")) + "\n")
            self.process.stdin.flush()

    def _read_event(self, timeout: float) -> dict:
        assert self.process.stdout is not None
        # readline() on a text-mode pipe has no timeout param; the runtime
        # is expected to respond quickly, so a watchdog thread is overkill
        # for a v0 thin client. A stuck runtime is a runtime bug, not a
        # client concern here.
        line = self.process.stdout.readline()
        if line == "":
            raise ServeError("voicechat process exited")
        event = json.loads(line)
        if event.get("kind") == "error":
            raise ServeError(event.get("message", "voicechat error"))
        return event

    def wait_for(self, kind: str, timeout: float = 300.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            event = self._read_event(deadline - time.monotonic())
            if event.get("kind") == kind:
                return event
        raise TimeoutError(f"timed out waiting for '{kind}'")

    def system(self, text: str) -> None:
        self._send({"cmd": "system", "text": text})
        self.wait_for("system")

    def run_turn(self, audio_in: Path, audio_out: Path, on_event) -> dict:
        """Submit one turn, stream events to on_event(event), return turn_end."""
        self._send({"cmd": "turn", "audio": str(audio_in), "out": str(audio_out)})
        while True:
            event = self._read_event(300.0)
            on_event(event)
            kind = event.get("kind")
            if kind == "tool_call":
                # Function-channel handling is out of scope for PTT v0;
                # skip so the turn does not hang waiting on a response.
                self._send({"cmd": "tool_skip"})
            elif kind == "turn_end":
                return event

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._send({"cmd": "quit"})
                self.process.wait(timeout=10)
            except Exception:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()


def record_to(path: Path, stop_event: threading.Event, device: str | None) -> None:
    """Record with arecord until stop_event is set, then finalize the WAV."""
    cmd = ["arecord", "-q", "-f", "cd", "-t", "wav"]
    if device:
        cmd += ["-D", device]
    cmd += [str(path)]
    proc = subprocess.Popen(cmd)
    stop_event.wait()
    proc.send_signal(signal.SIGINT)  # arecord finalizes the WAV header on SIGINT
    proc.wait(timeout=5)


def play(path: Path, device: str | None) -> None:
    cmd = ["aplay", "-q"]
    if device:
        cmd += ["-D", device]
    cmd += [str(path)]
    subprocess.run(cmd, check=False)


def build_command(args: argparse.Namespace) -> list[str]:
    if args.serve_cmd:
        return shlex.split(args.serve_cmd)
    missing = [name for name, val in (
        ("--bin", args.bin), ("--model", args.model),
        ("--mmproj", args.mmproj),
    ) if not val]
    if missing:
        sys.exit(
            "ptt.py: missing required arguments/env vars: " + ", ".join(missing) +
            "\n(or pass --serve-cmd 'full command line' directly)"
        )
    cmd = [
        args.bin, "-m", args.model, "--mmproj", args.mmproj,
        "--serve", "--no-warmup", "--device", args.device,
        "--split-mode", "none", "--gpu-layers", "all",
        "--temp", "0", "--seed", "42",
    ]
    if args.tts:
        cmd += ["--tts", args.tts]
    return cmd


def run_interactive(session: Session, args: argparse.Namespace) -> None:
    from pynput import keyboard

    print("Hold SPACE to record, release to submit. Ctrl+C to quit.")
    recording = threading.Event()
    stop_recording = threading.Event()
    rec_thread: threading.Thread | None = None
    rec_path: Path | None = None
    key_release_t = 0.0

    def on_press(key):
        nonlocal rec_thread, rec_path, stop_recording
        if key != keyboard.Key.space or recording.is_set():
            return
        recording.set()
        stop_recording = threading.Event()
        rec_path = Path(tempfile.mkstemp(suffix=".wav", prefix="ptt-in-")[1])
        rec_thread = threading.Thread(
            target=record_to, args=(rec_path, stop_recording, args.rec_device), daemon=True
        )
        rec_thread.start()
        print("[recording... release SPACE to submit]")

    def on_release(key):
        nonlocal key_release_t
        if key == keyboard.Key.esc:
            return False
        if key != keyboard.Key.space or not recording.is_set():
            return
        key_release_t = time.monotonic()
        stop_recording.set()
        rec_thread.join(timeout=10)
        recording.clear()
        submit_turn(session, rec_path, key_release_t, args, delete_input=True)

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


def submit_turn(session: Session, audio_in: Path, t0: float, args: argparse.Namespace,
                 delete_input: bool = False) -> None:
    out_path = Path(tempfile.mkstemp(suffix=".wav", prefix="ptt-out-")[1])
    text_parts: list[str] = []
    first_text_t = None
    audio_ready_t = None
    playback_begin_t = None

    def on_event(event: dict) -> None:
        nonlocal first_text_t, audio_ready_t, playback_begin_t
        kind = event.get("kind")
        if kind == "assistant_text_delta":
            if first_text_t is None:
                first_text_t = time.monotonic()
            delta = event.get("delta", "")
            text_parts.append(delta)
            print(delta, end="", flush=True)
        elif kind == "playback_begin":
            # M3.1 (VC_TTS_STREAM_PLAYBACK=1): the runtime's own aplay pipe
            # just received its first real PCM -- this is genuine first-audio
            # time, well before the "audio" event below, which under
            # streaming playback now fires only once the runtime's pipe has
            # finished playing the whole turn.
            if playback_begin_t is None:
                playback_begin_t = time.monotonic()
        elif kind == "audio":
            audio_ready_t = time.monotonic()

    submit_t = time.monotonic()
    try:
        result = session.run_turn(audio_in, out_path, on_event)
    except (ServeError, TimeoutError) as exc:
        print(f"\n[turn failed: {exc}]")
        return
    finally:
        # Only ever delete our own recorded temp file, never caller-owned
        # input such as a fixed corpus WAV (see --test).
        if delete_input:
            audio_in.unlink(missing_ok=True)

    print()  # end the streamed text line
    if audio_ready_t is not None and out_path.exists():
        play(out_path, args.play_device)
    total_t = time.monotonic()
    out_path.unlink(missing_ok=True)

    def since(t):
        return f"{(t - t0) * 1000:.0f} ms" if t is not None else "n/a"

    # Prefer the real playback_begin event (M3.1 streaming playback, emitted
    # when the runtime's own aplay pipe gets its first PCM). Falls back to
    # the legacy "audio" event (wav-ready time, immediately followed by this
    # client's own play() call) when streaming playback isn't active --
    # unchanged from before this fix.
    playback_metric_t = playback_begin_t if playback_begin_t is not None else audio_ready_t

    print(
        "[key_release->submit "
        + f"{(submit_t - t0) * 1000:.0f} ms, "
        + "first_text " + since(first_text_t) + ", "
        + "playback_begin " + since(playback_metric_t) + ", "
        + "total " + since(total_t) + "]"
    )


def run_test(session: Session, args: argparse.Namespace) -> None:
    """Non-interactive: run two turns from the frozen corpus, no live mic.

    Exercises the same Session/protocol/playback path as the interactive
    loop, for environments/CI where holding a physical key or capturing a
    live microphone is not available. Not a substitute for a real
    hold/release test.
    """
    corpus = Path(__file__).resolve().parents[2] / "research/corpus"
    inputs = [corpus / "VC01-short.wav", corpus / "VC02-conversation.wav"]
    for i, wav in enumerate(inputs):
        if not wav.exists():
            sys.exit(f"ptt.py --test: missing corpus file {wav}")
        print(f"\n--- test turn {i + 1}/{len(inputs)}: {wav.name} ---")
        submit_turn(session, wav, time.monotonic(), args)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bin", default=os.environ.get("VC_BIN"))
    p.add_argument("--model", default=os.environ.get("VC_MODEL"))
    p.add_argument("--mmproj", default=os.environ.get("VC_MMPROJ"))
    p.add_argument("--tts", default=os.environ.get("VC_TTS"))
    p.add_argument("--device", default=os.environ.get("VC_DEVICE", "ROCm0"))
    p.add_argument("--serve-cmd", default=os.environ.get("VC_SERVE_CMD"),
                    help="full command line, overrides --bin/--model/...")
    p.add_argument("--system", default=None, help="optional system prompt")
    p.add_argument("--rec-device", default=os.environ.get("VC_REC_DEVICE"))
    p.add_argument("--play-device", default=os.environ.get("VC_PLAY_DEVICE"))
    p.add_argument("--test", action="store_true",
                    help="run two turns from the fixed corpus instead of live mic/keyboard")
    args = p.parse_args()

    command = build_command(args)
    env = os.environ.copy()
    # No ROCR_VISIBLE_DEVICES default here: inventing a physical GPU index
    # is a machine-specific choice, not this client's to make. If the user
    # has set it, os.environ.copy() already carries it through; if not,
    # normal runtime enumeration plus --device ROCm0 picks the first
    # visible ROCm device. See docs/HARDWARE.md.
    #
    # GGML_CUDA_DISABLE_GRAPHS=1 is retained: the R9700-Q8-M1 baseline
    # demonstrated graph-enabled execution crashing on this runtime (see
    # research/baselines/R9700-Q8-M1/README.md, "Required compatibility
    # settings"), so this is a known-good requirement, not a leftover
    # dev-time default.
    env.setdefault("GGML_CUDA_DISABLE_GRAPHS", "1")
    env.setdefault("VC_NO_BARGE", "1")
    env.setdefault("VC_FORCE_BOS", "1")

    print("[starting voicechat --serve, waiting for ready...]")
    session = Session(command, env)
    try:
        ready = session.wait_for("ready")
        print(f"[ready: tts={ready.get('tts')} function_head={ready.get('function_head')}]")
        if args.system:
            session.system(args.system)
        if args.test:
            run_test(session, args)
        else:
            run_interactive(session, args)
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
