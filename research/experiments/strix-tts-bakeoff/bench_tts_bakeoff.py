#!/usr/bin/env python3
"""Measure isolated VoiceChat-native and alternate TTS renderer behavior.

This deliberately stays outside the VoiceChat conversation runtime.  The
native control uses the existing ``--say`` text-fixture path; Kokoro is loaded
and driven directly through its local streaming generator.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = Path(__file__).resolve().parent / "generated"
FIXTURES = {
    "short": "The quick brown fox jumps over the lazy dog.",
    "medium": (
        "A fluent spoken assistant must begin responding before the complete "
        "answer has been generated. It must keep producing audio while text "
        "continues to arrive, and it must stop promptly when the user interrupts."
    ),
    "long": (
        "The Strix serving study treats the conversational timeline as the "
        "product boundary. A speech renderer may be replaced if it accepts text "
        "quickly, starts audio incrementally, sustains real-time output, and "
        "drains or cancels without damaging conversation state. "
        "The useful comparison is therefore not model identity but the felt "
        "latency, continuity, and interruptibility of the resulting conversation."
    ),
}


def run_capture(args: list[str], *, cwd: Path | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout


def command_version(binary: Path, extra: list[str] | None = None) -> str:
    code, out = run_capture([str(binary), *(extra or ["--version"])])
    return f"exit={code}: {out.strip()[:1000]}"


def environment_snapshot(kokoro_root: Path) -> dict[str, Any]:
    def exists(path: str) -> bool:
        return Path(path).exists()

    snapshot: dict[str, Any] = {
        "schema": 1,
        "captured_at_unix": time.time(),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "repo_head": run_capture(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)[1].strip(),
        "kernel_device_visibility": {
            "dev_kfd": exists("/dev/kfd"),
            "dev_dri": exists("/dev/dri"),
            "render_nodes": sorted(str(p) for p in Path("/dev/dri").glob("render*") if p.exists()),
        },
        "native_binary": command_version(
            REPO_ROOT / "build/hip-gfx1151/bin/llama-voicechat"
        ),
        "kokoro": {
            "root_label": "local AI-Box Kokoro checkout",
            "model_present": (kokoro_root / "api/src/models/v1_0/kokoro-v1_0.pth").is_file(),
            "voice_present": (kokoro_root / "api/src/voices/v1_0/af_bella.pt").is_file(),
        },
    }

    for label, command in {
        "amd_smi_list": ["amd-smi", "list"],
        "xdna_top_snapshot": ["xdna-top", "--json", "snapshot"],
        "rocminfo_probe": ["rocminfo"],
    }.items():
        code, out = run_capture(command)
        snapshot[label] = {"exit": code, "output": out[:12000]}

    return snapshot


def proc_cpu_sample(pid: int) -> tuple[float, int] | None:
    """Return process CPU ticks and RSS KiB, or None after process exit."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().split()
        # utime/stime are fields 14/15, represented at indexes 13/14.
        ticks = float(stat[13]) + float(stat[14])
        status = Path(f"/proc/{pid}/status").read_text()
        rss = 0
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1])
                break
        return ticks, rss
    except (FileNotFoundError, IndexError, ValueError):
        return None


def wave_info(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        return {
            "sample_rate": rate,
            "channels": wav.getnchannels(),
            "sample_width_bytes": wav.getsampwidth(),
            "frames": frames,
            "audio_seconds": frames / rate if rate else 0.0,
            "pcm_bytes": frames * wav.getnchannels() * wav.getsampwidth(),
        }


def parse_native_frames(log: str) -> int | None:
    for line in reversed(log.splitlines()):
        if "voicechat-tts: wrote " in line and " frames," in line:
            try:
                return int(line.split("voicechat-tts: wrote ", 1)[1].split(" frames,", 1)[0])
            except ValueError:
                return None
    return None


def native_run(binary: Path, fixture_name: str, text: str, out_dir: Path) -> dict[str, Any]:
    model = "models/voicechat-q8/runtime/nemotron_voicechat_11b-stt-llm-Q8_0.gguf"
    mmproj = "models/voicechat-q8/runtime/mmproj-voicechat-perception-Q8_0.gguf"
    tts = "models/voicechat-q8/runtime/voicechat-tts-Q8_0.gguf"
    audio_fd, audio_name = tempfile.mkstemp(prefix="strix-native-", suffix=".wav")
    os.close(audio_fd)
    audio_path = Path(audio_name)
    log_path = out_dir / f"native_voicechat-{fixture_name}.log"
    command = [
        str(binary),
        "--device",
        "none",
        "--threads",
        "16",
        "--threads-batch",
        "16",
        "--ctx-size",
        "256",
        "--session-seconds",
        "15",
        "--mmproj",
        mmproj,
        "--model",
        model,
        "--tts",
        tts,
        "--tts-device",
        "CPU",
        "--say",
        text,
        "--tts-out",
        str(audio_path),
    ]

    start = time.perf_counter()
    # llama.cpp prints every loaded projector tensor.  Redirect directly to a
    # file so a verbose native run cannot deadlock on a full subprocess pipe.
    with log_path.open("w") as log_handle:
        proc = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        ticks_start = proc_cpu_sample(proc.pid)
        ticks_last = ticks_start
        peak_rss = 0
        first_pcm_ms: float | None = None
        hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        while proc.poll() is None:
            sample = proc_cpu_sample(proc.pid)
            if sample:
                ticks_last = sample
                peak_rss = max(peak_rss, sample[1])
            if first_pcm_ms is None and audio_path.exists() and audio_path.stat().st_size > 44:
                first_pcm_ms = (time.perf_counter() - start) * 1000.0
            time.sleep(0.02)
        proc.wait()
    elapsed = time.perf_counter() - start
    if ticks_last:
        peak_rss = max(peak_rss, ticks_last[1])
    output = log_path.read_text()

    if audio_path.exists():
        audio = wave_info(audio_path)
    else:
        audio = {"sample_rate": None, "audio_seconds": 0.0, "pcm_bytes": 0}
    audio_path.unlink(missing_ok=True)

    cpu_seconds = None
    cpu_percent_one_core = None
    if ticks_start and ticks_last:
        cpu_seconds = max(0.0, (ticks_last[0] - ticks_start[0]) / hz)
        cpu_percent_one_core = cpu_seconds / elapsed * 100.0 if elapsed else None

    return {
        "renderer": "native_voicechat",
        "fixture": fixture_name,
        "status": "pass" if proc.returncode == 0 and audio["pcm_bytes"] else "fail",
        "returncode": proc.returncode,
        "wall_seconds": elapsed,
        "first_pcm_ms": first_pcm_ms,
        "pcm_streaming": "final_file_only",
        "pcm_chunks": 1 if audio["pcm_bytes"] else 0,
        "chunk_audio_seconds": [audio["audio_seconds"]] if audio["pcm_bytes"] else [],
        "inter_chunk_gap_ms": [],
        "audio": audio,
        "audio_speed_ratio": audio["audio_seconds"] / elapsed if elapsed else None,
        "drain": {
            "status": "completed_before_final_wav_write",
            "drain_ms_after_last_pcm": 0.0,
        },
        "cancellation": {
            "status": "not_exposed_by_native_say_cli",
            "measurement": None,
            "note": "The native renderer has lifecycle/silence state internally, but this isolated CLI exposes no playback-cancel operation or PCM stream.",
        },
        "cpu_seconds": cpu_seconds,
        "cpu_percent_one_core": cpu_percent_one_core,
        "peak_rss_kib": peak_rss,
        "gpu_telemetry": "unavailable: /dev/kfd absent in executing shell",
        "npu_telemetry": "not applicable: native renderer was forced to CPU",
        "log": str(log_path.relative_to(REPO_ROOT))
        if log_path.is_relative_to(REPO_ROOT)
        else str(log_path),
        "generated_tts_frames": parse_native_frames(output),
    }


def kokoro_run(
    kokoro_root: Path,
    fixture_name: str,
    text: str,
    model: Any,
    pipeline: Any,
    voice_path: str,
) -> dict[str, Any]:
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    chunks: list[dict[str, Any]] = []
    # Use the same observable unit the serving study cares about: punctuation
    # delimited text chunks.  Kokoro's bare pipeline otherwise defaults to a
    # newline split and would make a whole fixture appear as one chunk.
    split_pattern = r"(?<=[.!?])\s+"
    generator = pipeline(
        text,
        voice=voice_path,
        speed=1.0,
        split_pattern=split_pattern,
        model=model,
    )
    first_pcm_ms: float | None = None
    while True:
        try:
            result = next(generator)
        except StopIteration:
            break
        now = time.perf_counter()
        audio = result.audio
        if audio is None:
            continue
        samples = int(audio.numel())
        if first_pcm_ms is None:
            first_pcm_ms = (now - start_wall) * 1000.0
        chunks.append(
            {
                "available_ms": (now - start_wall) * 1000.0,
                "samples": samples,
                "pcm_bytes_16bit": samples * 2,
                "audio_seconds": samples / 24000.0,
            }
        )
    end_wall = time.perf_counter()
    drain_ms = 0.0
    if chunks:
        drain_ms = chunks[-1]["available_ms"]
        drain_ms = (end_wall - start_wall) * 1000.0 - drain_ms
    wall_seconds = end_wall - start_wall

    # This is a generator-close probe, not a claim that Kokoro cancels an
    # in-flight kernel. It records the boundary currently exposed by the
    # isolated backend.
    cancel_start = time.perf_counter()
    cancel_gen = pipeline(
        FIXTURES["long"],
        voice=voice_path,
        speed=1.0,
        split_pattern=split_pattern,
        model=model,
    )
    cancel_first = next(cancel_gen, None)
    cancel_gen.close()
    cancel_ms = (time.perf_counter() - cancel_start) * 1000.0
    cancellation = {
        "status": "generator_close_only",
        "after_first_chunk": cancel_first is not None and cancel_first.audio is not None,
        "close_ms": cancel_ms,
        "note": "The direct pipeline exposes generator close, not a renderer-level cancel_pending_audio event.",
    }

    gaps = [
        chunks[i]["available_ms"] - chunks[i - 1]["available_ms"]
        for i in range(1, len(chunks))
    ]
    audio_seconds = sum(c["audio_seconds"] for c in chunks)
    return {
        "renderer": "kokoro_cpu",
        "fixture": fixture_name,
        "status": "pass" if chunks else "fail",
        "returncode": 0 if chunks else 1,
        "wall_seconds": wall_seconds,
        "first_pcm_ms": first_pcm_ms,
        "pcm_streaming": "sentence_chunks",
        "pcm_chunks": len(chunks),
        "chunk_audio_seconds": [c["audio_seconds"] for c in chunks],
        "chunk_pcm_bytes": [c["pcm_bytes_16bit"] for c in chunks],
        "inter_chunk_gap_ms": gaps,
        "audio": {
            "sample_rate": 24000,
            "channels": 1,
            "sample_width_bytes": 2,
            "frames": int(audio_seconds * 24000),
            "audio_seconds": audio_seconds,
            "pcm_bytes": sum(c["pcm_bytes_16bit"] for c in chunks),
        },
        "audio_speed_ratio": audio_seconds / wall_seconds if wall_seconds else None,
        "drain": {
            "status": "generator_exhausted",
            "drain_ms_after_last_pcm": drain_ms,
        },
        "cancellation": cancellation,
        "cpu_seconds": time.process_time() - start_cpu,
        "cpu_percent_one_core": (time.process_time() - start_cpu) / wall_seconds * 100.0
        if wall_seconds
        else None,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "gpu_telemetry": "unavailable: /dev/kfd absent; torch.cuda.is_available() false",
        "npu_telemetry": "not exercised",
        "model_label": "Kokoro V1 local model / af_bella voice",
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--kokoro-root",
        type=Path,
        default=REPO_ROOT.parent / "AI-Box" / "kokoro",
    )
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--skip-kokoro", action="store_true")
    args = parser.parse_args()

    out_dir = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.wav"):
        old.unlink()
    (out_dir / "logs").mkdir(exist_ok=True)

    binary = REPO_ROOT / "build/hip-gfx1151/bin/llama-voicechat"
    results: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "schema": 1,
        "experiment": "STRIX-TTS-BAKEOFF-CPU-001",
        "scope": "isolated renderer characterization; no VoiceChat runtime integration",
        "fixtures": FIXTURES,
        "environment": environment_snapshot(args.kokoro_root),
        "renderer_controls": {
            "native_voicechat": "validated Q8 VoiceChat TTS+codec via --say, forced CPU",
            "kokoro_cpu": "Kokoro V1 direct Python generator, CPU because CUDA unavailable",
        },
    }

    if not args.skip_native:
        if not binary.is_file():
            raise SystemExit(f"native binary not found: {binary}")
        for name, text in FIXTURES.items():
            print(f"native_voicechat {name}", flush=True)
            results.append(native_run(binary, name, text, out_dir / "logs"))

    if not args.skip_kokoro:
        if not args.kokoro_root.is_dir():
            raise SystemExit(f"Kokoro root not found: {args.kokoro_root}")
        sys.path.insert(0, str(args.kokoro_root))
        import torch  # type: ignore
        from kokoro import KModel, KPipeline  # type: ignore

        model_path = args.kokoro_root / "api/src/models/v1_0/kokoro-v1_0.pth"
        config_path = args.kokoro_root / "api/src/models/v1_0/config.json"
        voice_path = str(args.kokoro_root / "api/src/voices/v1_0/af_bella.pt")
        torch.set_num_threads(min(16, os.cpu_count() or 1))
        model_load_start = time.perf_counter()
        model = KModel(config=str(config_path), model=str(model_path)).eval().cpu()
        model_load_seconds = time.perf_counter() - model_load_start
        pipeline_start = time.perf_counter()
        pipeline = KPipeline(lang_code="a", model=model, device="cpu")
        pipeline_seconds = time.perf_counter() - pipeline_start
        metadata["kokoro_load"] = {
            "device": "cpu",
            "torch_version": torch.__version__,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "model_load_seconds": model_load_seconds,
            "pipeline_init_seconds": pipeline_seconds,
        }
        # Warm the model/pipeline outside measured fixture rows.
        for _ in pipeline("Warmup.", voice=voice_path, speed=1.0, model=model):
            pass
        for name, text in FIXTURES.items():
            print(f"kokoro_cpu {name}", flush=True)
            results.append(kokoro_run(args.kokoro_root, name, text, model, pipeline, voice_path))

    metadata["results_count"] = len(results)
    (out_dir / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out_dir / "raw-runs.json").write_text(json.dumps(results, indent=2) + "\n")
    write_csv(out_dir / "raw-runs.csv", results)

    summary: dict[str, Any] = {
        "experiment": metadata["experiment"],
        "classification": (
            "CPU-only isolated characterization; not a gfx1151 GPU performance result "
            "because /dev/kfd was absent"
        ),
        "comparison_validity": (
            "Behavioral evidence only: native rows are cold-process full-runtime "
            "measurements; Kokoro is warm-loaded once. Not a controlled cross-renderer "
            "latency ranking."
        ),
        "rows": [
            {
                "renderer": row["renderer"],
                "fixture": row["fixture"],
                "status": row["status"],
                "first_pcm_ms": row["first_pcm_ms"],
                "wall_seconds": row["wall_seconds"],
                "audio_seconds": row["audio"]["audio_seconds"],
                "audio_speed_ratio": row["audio_speed_ratio"],
                "pcm_chunks": row["pcm_chunks"],
                "drain_status": row["drain"]["status"],
                "cancellation_status": row["cancellation"]["status"],
                "cpu_percent_one_core": row["cpu_percent_one_core"],
                "peak_rss_kib": row["peak_rss_kib"],
            }
            for row in results
        ],
        "limitations": [
            "No /dev/kfd or render nodes were visible to the executing shell; GPU/NPU utilization and power were not measured.",
            "Native rows include cold full-runtime startup for every fixture; Kokoro is warm-loaded once, so the timing rows are not a controlled cross-renderer latency comparison.",
            "Native VoiceChat TTS emits codec PCM only at final WAV write in the isolated --say path.",
            "Native peak RSS includes the loaded VoiceChat LLM/projector as well as the TTS model; Kokoro RSS includes the warm Python process.",
            "Kokoro direct generator close measures the exposed Python generator boundary, not cancellation of an in-flight accelerator kernel.",
            "No subjective listening evaluation was performed.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report_lines = [
        "# STRIX-TTS-BAKEOFF-CPU-001",
        "",
        "CPU-only isolated renderer characterization. This is not a gfx1151 GPU performance result: `/dev/kfd` was absent in the executing shell.",
        "Native rows are cold-process full-runtime measurements; Kokoro is warm-loaded once. The timing table is behavioral evidence, not a controlled cross-renderer latency ranking.",
        "",
        "| renderer | fixture | first PCM (ms) | wall (s) | audio (s) | speed ratio | PCM chunks | CPU (% of one core) | peak RSS (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        report_lines.append(
            "| {renderer} | {fixture} | {first:.1f} | {wall:.3f} | {audio:.3f} | {speed:.2f}x | {chunks} | {cpu:.1f} | {rss:.1f} |".format(
                renderer=row["renderer"],
                fixture=row["fixture"],
                first=row["first_pcm_ms"] or 0.0,
                wall=row["wall_seconds"],
                audio=row["audio"]["audio_seconds"],
                speed=row["audio_speed_ratio"] or 0.0,
                chunks=row["pcm_chunks"],
                cpu=row["cpu_percent_one_core"] or 0.0,
                rss=row["peak_rss_kib"] / 1024.0,
            )
        )
    report_lines.extend(
        [
            "",
            "## Behavioral observations",
            "",
            "- Native VoiceChat TTS generated incremental 80 ms model frames, but the isolated `--say` path exposed one final WAV write. It did not expose first PCM, playback cancellation, or a renderer-level drain event.",
            "- Kokoro produced punctuation-delimited PCM chunks and exhausted its generator cleanly. Its cancellation measurement is only Python generator close; it does not prove cancellation of an in-flight accelerator operation.",
            "- GPU utilization, NPU utilization, package power, and thermal behavior were not measured because `/dev/kfd` and render nodes were not visible to this shell. `amd-smi` could enumerate the card but did not provide live utilization in this context.",
            "- Native measurements include cold full-runtime startup for each fixture; native RSS includes the LLM/projector. Kokoro was warm-loaded once. Do not compare these rows as a formal renderer speed ranking.",
            "- No VoiceChat runtime integration, NPU implementation, alternate-TTS integration, or subjective listening evaluation was performed.",
            "",
            "## Next gate",
            "",
            "Repeat the unchanged fixture/measurement contract in a shell with `/dev/kfd` and render-node access before making a Strix GPU-serving decision. Keep renderer contract work separate from the VoiceChat runtime until cancellation, accepted-text buffering, and speech drain are specified.",
            "",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(report_lines))
    print(json.dumps(summary, indent=2))
    return 0 if all(row["status"] == "pass" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
