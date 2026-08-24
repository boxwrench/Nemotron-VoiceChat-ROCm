#!/usr/bin/env python3
"""Push-to-talk client using terminal Enter key for Wayland environments.

Press Enter to start recording, press Enter again to stop and submit.
Type 'q' or Ctrl+C to quit.

Designed to avoid the pynput/Wayland global-key-capture issue by using
terminal cbreak mode instead.
"""

import argparse
import os
import sys
import select
import termios
import tty
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ptt import Session, build_command, record_to, submit_turn


def flush_stdin():
    """Drain any buffered keystrokes so they don't cascade."""
    while select.select([sys.stdin], [], [], 0.0)[0]:
        sys.stdin.read(1)


def wait_for_key(valid_chars=None):
    """Block until a single keypress, return the character.
    If valid_chars is set, ignore anything not in that set."""
    while True:
        ch = sys.stdin.read(1)
        if valid_chars is None or ch in valid_chars:
            return ch


def run_terminal_mode(session: Session, args: argparse.Namespace) -> None:
    print()
    print("=" * 60)
    print("  TERMINAL PUSH-TO-TALK (Wayland Compatible)")
    print()
    print("  [Enter]  = start recording")
    print("  [Enter]  = stop recording & submit")
    print("  [q]      = quit")
    print("=" * 60)
    print()

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())

        turn = 0
        while True:
            flush_stdin()
            print("[Ready — press ENTER to talk, or 'q' to quit]", flush=True)
            ch = wait_for_key({'\n', '\r', ' ', 'q', 'Q', '\x1b', '\x03'})
            if ch in ('q', 'Q', '\x1b', '\x03'):
                print("\n[Exiting cleanly...]")
                break

            # --- Start recording ---
            turn += 1
            rec_path = Path(tempfile.mkstemp(suffix=".wav", prefix="ptt-in-")[1])
            stop_recording = threading.Event()
            rec_thread = threading.Thread(
                target=record_to,
                args=(rec_path, stop_recording, args.rec_device),
                daemon=True,
            )
            rec_thread.start()

            # Small delay to let arecord actually open the device
            time.sleep(0.15)
            flush_stdin()

            print(f"[● RECORDING (turn {turn})... press ENTER when done]", flush=True)

            # Wait for stop key, but enforce a minimum recording of 0.8s
            rec_start = time.monotonic()
            MIN_REC_SECS = 0.8
            while True:
                # Poll stdin with a short timeout so we can enforce minimum
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    ch = sys.stdin.read(1)
                    elapsed = time.monotonic() - rec_start
                    if ch in ('\n', '\r', ' '):
                        if elapsed >= MIN_REC_SECS:
                            break
                        # Too short — ignore this keypress
                        continue
                    if ch in ('q', 'Q', '\x1b', '\x03'):
                        # Abort recording and quit
                        stop_recording.set()
                        rec_thread.join(timeout=5)
                        rec_path.unlink(missing_ok=True)
                        print("\n[Exiting cleanly...]")
                        return

            # --- Stop recording ---
            t_stop = time.monotonic()
            stop_recording.set()
            rec_thread.join(timeout=10)

            # Validate the WAV file is real before submitting
            if not rec_path.exists() or rec_path.stat().st_size < 100:
                print("[Recording too short or failed, skipping turn]")
                rec_path.unlink(missing_ok=True)
                time.sleep(0.3)
                continue

            print("[Submitting audio...]", flush=True)
            submit_turn(session, rec_path, t_stop, args, delete_input=True)

            # Brief cooldown before next prompt to avoid key cascade
            time.sleep(0.5)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bin", default=os.environ.get("VC_BIN"))
    p.add_argument("--model", default=os.environ.get("VC_MODEL"))
    p.add_argument("--mmproj", default=os.environ.get("VC_MMPROJ"))
    p.add_argument("--tts", default=os.environ.get("VC_TTS"))
    p.add_argument("--device", default=os.environ.get("VC_DEVICE", "ROCm0"))
    p.add_argument("--serve-cmd", default=os.environ.get("VC_SERVE_CMD"))
    p.add_argument("--system", default=None)
    p.add_argument("--rec-device", default=os.environ.get("VC_REC_DEVICE"))
    p.add_argument("--play-device", default=os.environ.get("VC_PLAY_DEVICE"))
    args = p.parse_args()

    command = build_command(args)
    env = os.environ.copy()
    env.setdefault("ROCR_VISIBLE_DEVICES", "1")
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
        run_terminal_mode(session, args)
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
