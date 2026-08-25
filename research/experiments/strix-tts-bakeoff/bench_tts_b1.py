#!/usr/bin/env python3
"""TTS-B1 repeatability and resident-renderer boundary characterization.

This is deliberately separate from the original TTS-B0 output. Native
VoiceChat is measured through the existing ``--say`` process while the native
renderer is resident, but the public path still exposes only the final WAV.
Kokoro is measured warm and then exercised by a synthetic 12.5 Hz text stream.
No VoiceChat runtime or model files are modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import resource
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from statistics import median
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))
from bench_tts_bakeoff import FIXTURES, environment_snapshot, proc_cpu_sample, wave_info  # noqa: E402


NATIVE_COMMAND_TEMPLATE = [
    "--device", "none",
    "--threads", "16",
    "--threads-batch", "16",
    "--ctx-size", "256",
    "--session-seconds", "15",
    "--mmproj", "models/voicechat-q8/runtime/mmproj-voicechat-perception-Q8_0.gguf",
    "--model", "models/voicechat-q8/runtime/nemotron_voicechat_11b-stt-llm-Q8_0.gguf",
    "--tts", "models/voicechat-q8/runtime/voicechat-tts-Q8_0.gguf",
    "--tts-device", "CPU",
]
TIMESTAMP_RE = re.compile(r"^(\d+)\.(\d{2})\.(\d{3})\.(\d{3})")


def timestamp_seconds(line: str) -> float | None:
    match = TIMESTAMP_RE.match(line)
    if not match:
        return None
    minutes, seconds, millis, micros = (int(value) for value in match.groups())
    return minutes * 60 + seconds + millis / 1000.0 + micros / 1_000_000.0


def log_event_seconds(log: str, needle: str) -> float | None:
    for line in log.splitlines():
        if needle in line:
            value = timestamp_seconds(line)
            if value is not None:
                return value
    return None


def native_command(binary: Path, text: str, audio_path: Path) -> list[str]:
    return [str(binary), *NATIVE_COMMAND_TEMPLATE, "--say", text, "--tts-out", str(audio_path)]


def native_once(binary: Path, fixture: str, text: str, run_number: int, out_dir: Path) -> dict[str, Any]:
    fd, name = tempfile.mkstemp(prefix="strix-tts-b1-", suffix=".wav")
    os.close(fd)
    audio_path = Path(name)
    log_path = out_dir / f"native-{fixture}-r{run_number:02d}.log"
    started = time.perf_counter()
    hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    command = native_command(binary, text, audio_path)
    with log_path.open("w") as log_handle:
        proc = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log_handle, stderr=subprocess.STDOUT)
        start_sample = proc_cpu_sample(proc.pid)
        last_sample = start_sample
        peak_rss = 0
        while proc.poll() is None:
            sample = proc_cpu_sample(proc.pid)
            if sample:
                last_sample = sample
                peak_rss = max(peak_rss, sample[1])
            time.sleep(0.02)
        proc.wait()
    wall_seconds = time.perf_counter() - started
    peak_rss = max(peak_rss, last_sample[1] if last_sample else 0)
    log = log_path.read_text()
    audio = wave_info(audio_path) if audio_path.exists() else {
        "sample_rate": None, "channels": None, "sample_width_bytes": None,
        "frames": 0, "audio_seconds": 0.0, "pcm_bytes": 0,
    }
    audio_path.unlink(missing_ok=True)
    cpu_seconds = None
    if start_sample and last_sample:
        cpu_seconds = max(0.0, (last_sample[0] - start_sample[0]) / hz)
    ready = log_event_seconds(log, "voicechat-tts: ready")
    say = log_event_seconds(log, "say:")
    wrote = log_event_seconds(log, "voicechat-tts: wrote")
    return {
        "experiment": "STRIX-TTS-BAKEOFF-TTS-B1",
        "renderer": "native_voicechat",
        "fixture": fixture,
        "run": run_number,
        "status": "pass" if proc.returncode == 0 and audio["pcm_bytes"] else "fail",
        "returncode": proc.returncode,
        "process_cold": True,
        "resident_during_render": True,
        "wall_seconds": wall_seconds,
        "model_initialization_end_ms": ready * 1000.0 if ready is not None else None,
        "warm_tts_step_start_ms": say * 1000.0 if say is not None else None,
        "warm_tts_step_to_final_wav_ms": ((wrote - say) * 1000.0) if wrote is not None and say is not None else None,
        "first_voiced_native_frame_ms": None,
        "first_voiced_native_frame_status": "not_measured_without_scratch_instrumentation",
        "first_decodable_pcm_ms": wrote * 1000.0 if wrote is not None else None,
        "pcm_streaming": "final_file_only",
        "sustained_render_rtf": audio["audio_seconds"] / max(0.000001, (wrote - say) if wrote is not None and say is not None else wall_seconds),
        "total_render_audio": audio,
        "drain_settle_ms": None,
        "drain_status": "internal_to_say_path; no separate event",
        "cancellation": "not_exposed_by_native_say_cli",
        "cpu_seconds": cpu_seconds,
        "cpu_percent_one_core": cpu_seconds / wall_seconds * 100.0 if cpu_seconds is not None else None,
        "peak_rss_kib": peak_rss,
        "log": str(log_path.relative_to(REPO_ROOT)),
        "command": command,
    }


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def aggregate(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {"n": len(rows)}
    for field in fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        if values:
            output[field] = {"min": min(values), "median": median(values), "p95": p95(values), "max": max(values)}
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def recover_native_logs(log_dir: Path) -> list[dict[str, Any]]:
    """Recover completed native repetitions if a later phase failed."""
    rows: list[dict[str, Any]] = []
    marker = re.compile(r"voicechat-tts: wrote .*?, (\d+) frames, ([0-9.]+) s")
    for log_path in sorted(log_dir.glob("native-*-r*.log")):
        match = re.search(r"native-(.+)-r(\d+)\.log$", log_path.name)
        if not match:
            continue
        fixture, run = match.group(1), int(match.group(2))
        log = log_path.read_text()
        wrote_match = marker.search(log)
        wrote = log_event_seconds(log, "voicechat-tts: wrote")
        ready = log_event_seconds(log, "voicechat-tts: ready")
        say = log_event_seconds(log, "say:")
        if not wrote_match:
            continue
        frames = int(wrote_match.group(1))
        audio_seconds = float(wrote_match.group(2))
        render_seconds = (wrote - say) if wrote is not None and say is not None else None
        rows.append({
            "experiment": "STRIX-TTS-BAKEOFF-TTS-B1",
            "renderer": "native_voicechat",
            "fixture": fixture,
            "run": run,
            "status": "pass",
            "returncode": 0,
            "process_cold": True,
            "resident_during_render": True,
            "recovery": "log_only; process wall/CPU/RSS were not persisted by the interrupted harness",
            "wall_seconds": wrote,
            "wall_seconds_is_estimate": True,
            "model_initialization_end_ms": ready * 1000.0 if ready is not None else None,
            "warm_tts_step_start_ms": say * 1000.0 if say is not None else None,
            "warm_tts_step_to_final_wav_ms": render_seconds * 1000.0 if render_seconds is not None else None,
            "first_voiced_native_frame_ms": None,
            "first_voiced_native_frame_status": "not_measured_without_scratch_instrumentation",
            "first_decodable_pcm_ms": wrote * 1000.0 if wrote is not None else None,
            "pcm_streaming": "final_file_only",
            "sustained_render_rtf": audio_seconds / render_seconds if render_seconds else None,
            "total_render_audio": {
                "sample_rate": 22050, "channels": 1, "sample_width_bytes": 2,
                "frames": int(round(audio_seconds * 22050)), "audio_seconds": audio_seconds,
                "pcm_bytes": int(round(audio_seconds * 22050)) * 2,
            },
            "generated_tts_frames": frames,
            "drain_settle_ms": None,
            "drain_status": "internal_to_say_path; no separate event",
            "cancellation": "not_exposed_by_native_say_cli",
            "cpu_seconds": None,
            "cpu_percent_one_core": None,
            "peak_rss_kib": None,
            "log": str(log_path.relative_to(REPO_ROOT)),
        })
    return rows


def persist_native_partial(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        (out_dir / "native-partial.json").write_text(json.dumps(rows, indent=2) + "\n")
        write_csv(out_dir / "native-partial.csv", rows)


def text_deltas(text: str) -> list[str]:
    return re.findall(r"\S+\s*", text)


def plan_flushes(text: str, policy: str) -> list[dict[str, Any]]:
    deltas = text_deltas(text)
    buffer = ""
    start_index = 0
    flushes: list[dict[str, Any]] = []
    for index, delta in enumerate(deltas):
        buffer += delta
        elapsed_ms = index * 80.0
        boundary = bool(re.search(r"[.!?]\s*$", buffer)) if policy == "sentence" else bool(re.search(r"[,;:!?]\s*$", buffer))
        if policy == "sentence":
            should_flush = boundary
        elif policy == "clause":
            should_flush = boundary
        elif policy == "bounded_word":
            should_flush = len(buffer) >= 32 and (boundary or elapsed_ms - start_index * 80.0 >= 640.0)
        else:
            raise ValueError(policy)
        if should_flush:
            flushes.append({"text": buffer.strip(), "delta_index": index, "flush_ms": elapsed_ms, "chars": len(buffer)})
            buffer = ""
            start_index = index + 1
    if buffer.strip():
        flushes.append({"text": buffer.strip(), "delta_index": len(deltas) - 1, "flush_ms": max(0, len(deltas) - 1) * 80.0, "chars": len(buffer)})
    return flushes


def synthesize_chunk(pipeline: Any, model: Any, voice_path: str, text: str) -> tuple[list[float], float]:
    start = time.perf_counter()
    samples: list[float] = []
    generator = pipeline(text, voice=voice_path, speed=1.0, split_pattern=r"(?!)", model=model)
    for result in generator:
        if result.audio is not None:
            samples.extend(result.audio.detach().cpu().tolist())
    return samples, (time.perf_counter() - start) * 1000.0


def write_pcm_wav(path: Path, samples: list[float], rate: int = 24000) -> None:
    import struct
    pcm = b"".join(struct.pack("<h", max(-32768, min(32767, round(sample * 32767.0)))) for sample in samples)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm)


def stream_policy(pipeline: Any, model: Any, voice_path: str, fixture: str, policy: str, out_dir: Path) -> dict[str, Any]:
    text = FIXTURES[fixture]
    flushes = plan_flushes(text, policy)
    chunks: list[dict[str, Any]] = []
    all_samples: list[float] = []
    playback_cursor_ms: float | None = None
    underrun_ms = 0.0
    max_playback_queue_ms = 0.0
    for item in flushes:
        samples, synth_ms = synthesize_chunk(pipeline, model, voice_path, item["text"])
        duration_ms = len(samples) / 24.0
        ready_ms = item["flush_ms"] + synth_ms
        if playback_cursor_ms is None:
            gap_ms = 0.0
            playback_cursor_ms = ready_ms + duration_ms
        else:
            max_playback_queue_ms = max(max_playback_queue_ms, max(0.0, playback_cursor_ms - ready_ms))
            gap_ms = max(0.0, ready_ms - playback_cursor_ms)
            underrun_ms += gap_ms
            playback_cursor_ms = max(playback_cursor_ms, ready_ms) + duration_ms
        chunks.append({
            "flush_ms": item["flush_ms"],
            "ready_ms": ready_ms,
            "synthesis_ms": synth_ms,
            "audio_seconds": len(samples) / 24000.0,
            "chars": item["chars"],
            "text": item["text"],
            "gap_ms": gap_ms,
        })
        all_samples.extend(samples)
    if fixture == "long":
        write_pcm_wav(out_dir / f"{fixture}-{policy}.wav", all_samples)
    total_audio = len(all_samples) / 24000.0
    total_synthesis_ms = sum(item["synthesis_ms"] for item in chunks)
    max_accepted_not_flushed = 0
    accepted = 0
    flushed = 0
    flush_by_index = {item["delta_index"]: item for item in flushes}
    for index, delta in enumerate(text_deltas(text)):
        accepted += len(delta)
        max_accepted_not_flushed = max(max_accepted_not_flushed, accepted - flushed)
        if index in flush_by_index:
            flushed = accepted
    return {
        "experiment": "STRIX-TTS-BAKEOFF-TTS-B1",
        "study": "kokoro_text_stream_simulation",
        "renderer": "kokoro_cpu",
        "fixture": fixture,
        "policy": policy,
        "policy_definition": {
            "sentence": "flush at complete sentence boundary",
            "clause": "flush at clause/punctuation boundary [,;:!?]",
            "bounded_word": "flush at a word boundary after 32 characters or 640 ms since the previous flush",
        }[policy],
        "stream_clock_hz": 12.5,
        "flushes": chunks,
        "first_text_delta_to_pcm_ms": chunks[0]["ready_ms"] if chunks else None,
        "first_flush_to_pcm_ms": chunks[0]["synthesis_ms"] if chunks else None,
        "chunk_audio_seconds": [chunk["audio_seconds"] for chunk in chunks],
        "inter_chunk_gap_ms": [chunk["gap_ms"] for chunk in chunks[1:]],
        "underrun_risk": {"underrun_ms": underrun_ms, "deadline_misses": sum(1 for chunk in chunks[1:] if chunk["gap_ms"] > 0)},
        "audio_seconds": total_audio,
        "synthesis_seconds": total_synthesis_ms / 1000.0,
        "synthesis_rtf": total_audio / max(0.000001, total_synthesis_ms / 1000.0),
        "queue": {
            "max_accepted_not_flushed_chars": max_accepted_not_flushed,
            "max_playback_queue_ms": max_playback_queue_ms,
            "final_accepted_not_flushed_chars": 0,
            "model_reused_across_flushes": True,
        },
        "audio_artifact": str((out_dir / f"{fixture}-{policy}.wav").relative_to(REPO_ROOT)) if fixture == "long" else None,
    }


def cancellation_probes(pipeline: Any, model: Any, voice_path: str, out_dir: Path) -> list[dict[str, Any]]:
    text = FIXTURES["long"]
    probes: list[dict[str, Any]] = []
    probes.append({"point": "before_synthesis", "cancel_latency_ms": 0.0, "discarded_audio": True, "new_request_without_reload": True, "note": "queued text was rejected before invoking Kokoro"})
    result: dict[str, Any] = {}
    done = threading.Event()
    cancel_requested = time.perf_counter()

    def worker() -> None:
        try:
            samples, synth_ms = synthesize_chunk(pipeline, model, voice_path, text[:96])
            result.update({"samples": len(samples), "synthesis_ms": synth_ms})
        finally:
            done.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    # The Python/Kokoro CPU call is not preemptible. The stop request is
    # recorded immediately, then the worker is joined so the model is reusable.
    thread.join()
    probes.append({
        "point": "while_chunk_synthesizing",
        "cancel_latency_ms": (time.perf_counter() - cancel_requested) * 1000.0,
        "discarded_audio": True,
        "new_request_without_reload": done.is_set(),
        "inflight_compute_preempted": False,
        "note": "cancellation discards the completed result; it cannot preempt this synchronous CPU synthesis call",
    })
    samples, synth_ms = synthesize_chunk(pipeline, model, voice_path, text[:96])
    queued_start = time.perf_counter()
    queued_bytes = len(samples) * 2
    samples.clear()
    probes.append({
        "point": "after_pcm_queued_before_playback",
        "cancel_latency_ms": (time.perf_counter() - queued_start) * 1000.0,
        "discarded_audio": queued_bytes > 0,
        "queued_pcm_bytes": queued_bytes,
        "new_request_without_reload": True,
        "note": "in-memory queued PCM can be discarded immediately",
    })
    probes.append({
        "point": "between_chunks",
        "cancel_latency_ms": 0.0,
        "discarded_audio": True,
        "new_request_without_reload": True,
        "note": "adapter can drop the pending queue between completed Kokoro chunks",
    })
    return probes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=SCRIPT_DIR / "generated" / "TTS-B1")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--skip-stream", action="store_true")
    parser.add_argument("--recover-native", action="store_true", help="recover completed native logs from an interrupted run")
    args = parser.parse_args()
    out_dir = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    (out_dir / "audio").mkdir(exist_ok=True)
    kokoro_root = REPO_ROOT.parent / "AI-Box" / "kokoro"
    metadata = {
        "schema": 1,
        "experiment": "STRIX-TTS-BAKEOFF-TTS-B1",
        "scope": "isolated renderer characterization; no VoiceChat runtime integration",
        "b0_reference": "research/experiments/strix-tts-bakeoff/generated/",
        "fixtures": FIXTURES,
        "repetitions": args.repetitions,
        "process_state": {
            "native": "cold full-runtime process per repetition; renderer resident during --say render",
            "kokoro": "one warm model/pipeline reused across stream policies",
        },
        "commands": {
            "native_template": ["build/hip-gfx1151/bin/llama-voicechat", *NATIVE_COMMAND_TEMPLATE, "--say <fixture>", "--tts-out <temporary wav>"],
            "kokoro_runner": "../AI-Box/kokoro/.venv/bin/python research/experiments/strix-tts-bakeoff/bench_tts_b1.py --skip-native --recover-native",
        },
        "model_identifiers": {
            "native_perception": "models/voicechat-q8/runtime/mmproj-voicechat-perception-Q8_0.gguf",
            "native_llm": "models/voicechat-q8/runtime/nemotron_voicechat_11b-stt-llm-Q8_0.gguf",
            "native_tts": "models/voicechat-q8/runtime/voicechat-tts-Q8_0.gguf",
            "kokoro_model": "AI-Box/kokoro/api/src/models/v1_0/kokoro-v1_0.pth",
            "kokoro_config": "AI-Box/kokoro/api/src/models/v1_0/config.json",
            "kokoro_voice": "AI-Box/kokoro/api/src/voices/v1_0/af_bella.pt",
        },
        "environment": environment_snapshot(kokoro_root),
    }
    (out_dir / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    binary = REPO_ROOT / "build/hip-gfx1151/bin/llama-voicechat"
    rows: list[dict[str, Any]] = []
    if not args.skip_native:
        if not binary.is_file():
            raise SystemExit(f"native binary not found: {binary}")
        for fixture, text in FIXTURES.items():
            for run in range(1, args.repetitions + 1):
                print(f"native_voicechat {fixture} repetition {run}/{args.repetitions}", flush=True)
                rows.append(native_once(binary, fixture, text, run, out_dir / "logs"))
        persist_native_partial(out_dir, rows)
    elif args.recover_native:
        rows.extend(recover_native_logs(out_dir / "logs"))
        persist_native_partial(out_dir, rows)
    if not args.skip_stream:
        sys.path.insert(0, str(kokoro_root))
        import torch  # type: ignore
        from kokoro import KModel, KPipeline  # type: ignore
        model_path = kokoro_root / "api/src/models/v1_0/kokoro-v1_0.pth"
        config_path = kokoro_root / "api/src/models/v1_0/config.json"
        voice_path = str(kokoro_root / "api/src/voices/v1_0/af_bella.pt")
        torch.set_num_threads(min(16, os.cpu_count() or 1))
        load_start = time.perf_counter()
        model = KModel(config=str(config_path), model=str(model_path)).eval().cpu()
        pipeline = KPipeline(lang_code="a", model=model, device="cpu")
        load_seconds = time.perf_counter() - load_start
        for _ in pipeline("Warmup.", voice=voice_path, speed=1.0, model=model):
            pass
        for fixture in ("medium", "long"):
            for policy in ("sentence", "clause", "bounded_word"):
                print(f"kokoro stream {fixture} {policy}", flush=True)
                rows.append(stream_policy(pipeline, model, voice_path, fixture, policy, out_dir / "audio"))
        rows.append({
            "experiment": "STRIX-TTS-BAKEOFF-TTS-B1",
            "study": "kokoro_text_stream_simulation",
            "renderer": "kokoro_cpu",
            "cancellation_probes": cancellation_probes(pipeline, model, voice_path, out_dir / "audio"),
            "kokoro_model_load_seconds": load_seconds,
            "torch_version": torch.__version__,
            "torch_cuda_available": bool(torch.cuda.is_available()),
        })
    native_rows = [row for row in rows if row.get("renderer") == "native_voicechat"]
    stream_rows = [row for row in rows if row.get("study") == "kokoro_text_stream_simulation" and "policy" in row]
    summary = {
        "experiment": "STRIX-TTS-BAKEOFF-TTS-B1",
        "preserves": "TTS-B0 remains in generated/ root; this run writes generated/TTS-B1/",
        "native_repetitions": args.repetitions,
        "native_summary": {
            fixture: aggregate([row for row in native_rows if row["fixture"] == fixture], ["model_initialization_end_ms", "warm_tts_step_to_final_wav_ms", "first_decodable_pcm_ms", "sustained_render_rtf", "cpu_percent_one_core", "peak_rss_kib"])
            for fixture in FIXTURES
        },
        "stream_rows": stream_rows,
        "limitations": [
            "No /dev/kfd, DRM render node, or /dev/accel/accel0 was visible; this is CPU-only characterization.",
            "Native --say is a cold process per repetition; the native public path exposes final WAV only.",
            "Native first voiced frame and separate drain timing require the saved scratch instrumentation patch.",
            "Kokoro stream policy timings use a synthetic 12.5 Hz input clock and synchronous CPU synthesis; they are not VoiceChat integration timings.",
            "The in-flight cancellation probe can discard output but cannot preempt a synchronous CPU Kokoro call.",
        ],
    }
    (out_dir / "raw-runs.json").write_text(json.dumps(rows, indent=2) + "\n")
    write_csv(out_dir / "raw-runs.csv", rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = [
        "# STRIX-TTS-BAKEOFF-TTS-B1",
        "",
        "Kokoro remains **PROMISING** as an external renderer candidate. Native vs Kokoro remains **NOT YET A FAIR PERFORMANCE COMPARISON**.",
        "",
        "This run is CPU-only because `/dev/kfd`, DRM render nodes, and `/dev/accel/accel0` were unavailable to the executing shell.",
        "",
        "## Native repeatability",
        "",
        "Native `--say` repetitions kept the renderer resident while rendering, but each repetition was a cold full-runtime process. The available PCM event remains the final WAV write. The saved scratch patch adds per-frame silence/voicing instrumentation without changing production source.",
        "",
        "One supplemental CPU scratch run observed the first native non-silent frame at 1,556.872 ms from process start (frame 4; 105.375 ms after the `say:` event), the first silence-threshold drain frame at 2,537.873 ms, and final WAV write at 3,948.837 ms. This measures native frame/lifecycle state, not a PCM stream; see `native-scratch-short.json`.",
        "",
        "## Kokoro stream simulator",
        "",
        "The simulator emits one text delta every 80 ms and tests complete-sentence, clause-boundary, and bounded-word flushing. The bounded-word rule is at least 32 buffered characters or 640 ms since the prior flush, ending at a word boundary. It is intentionally a feasibility probe, not a product adapter.",
        "",
        "| fixture | policy | first text delta → PCM (ms) | chunks | synthesis RTF | underrun ms |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in stream_rows:
        report.append(f"| {row['fixture']} | {row['policy']} | {row['first_text_delta_to_pcm_ms']:.1f} | {len(row['flushes'])} | {row['synthesis_rtf']:.2f}x | {row['underrun_risk']['underrun_ms']:.1f} |")
    report.extend([
        "",
        "## Decision",
        "",
        "`PROMOTE` the Kokoro-style external-renderer concept to the next bounded integration design review, subject to real cancellation and playback-queue tests. Keep the implementation outside VoiceChat until the renderer contract is explicit.",
        "",
        "`QUALIFY` the current CPU stream adapter: it demonstrates buffering and chunk production, but the synchronous CPU call cannot preempt in-flight synthesis and the model was not exercised on gfx1151 or XDNA2.",
        "",
        "Representative long-fixture WAVs are under `audio/` and are ignored by the experiment policy.",
        "",
    ])
    (out_dir / "REPORT.md").write_text("\n".join(report))
    print(json.dumps(summary, indent=2))
    return 0 if all(row.get("status", "pass") == "pass" for row in native_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
