# STRIX-BRINGUP-2 agent-visible NPU stack probe

This capture is from the coding-agent shell. It must not be interpreted as a host-namespace result.

## uname

+ uname -a
Linux <host> 6.17.0-35-generic #35~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Tue May 26 19:30:42 UTC 2 x86_64 x86_64 x86_64 GNU/Linux

exit=0

## identity

+ id
uid=1000(<user>) gid=1000(<user>) groups=1000(<user>),65534(nogroup)

exit=0

## groups

+ groups
<user> nogroup

exit=0

## FastFlowLM version

+ flm --version
FLM v0.9.39

exit=1

## FastFlowLM help

+ flm help
Usage: flm <command> [options] [model_tag]

Commands:
  run <model_tag>     - Run the model interactively
  serve <model_tag>   - Start the  server
  pull <model_tag>    - Download model files if not present
  remove <model_tag>  - Remove a model
  list                - List all available models
  version             - Show version information
  help                - Show this help message
  port                - Show the default server port
  validate            - Validate the NPU stack

Allowed options:
  -h [ --help ]                    Show help message
  -v [ --version ]                 Show version information
  --pmode arg (=performance)       Set power mode: powersaver, balanced,
                                   performance, turbo
  -a [ --asr ] arg (=0)            If load asr model
  -e [ --embed ] arg (=0)          If load embed model
  --host arg (=127.0.0.1)          Set the server address (for serve command)
  -p [ --port ] arg (=-1)          Set the server port number (for serve
                                   command)
  --force                          Force re-download even if model exists (for
                                   pull command)
  --filter arg (=all)              Show models: all | installed | not-installed
  --quiet                          Quiet mode, for sub-process usages
  -j [ --json ]                    Output in JSON format (for list, validate,
                                   version commands)
  -c [ --ctx-len ] arg (=-1)       Set context length
  -r [ --img-pre-resize ] arg (=2) Pre-resize the image, 0: original size, 1:
                                   height = 480, 2: height = 720, 3: height =
                                   1080, 4: height = 1440
  -s [ --socket ] arg (=10)        Set the maximum number of socket connections
                                   allowed (for serve command)
  -q [ --q-len ] arg (=10)         Set number of max npu queue length (for
                                   serve command)
  --cors arg (=1)                  Enable or disable Cross-Origin Resource
                                   Sharing (CORS) (for serve command)
  --preemption arg (=0)            Enable preemption
  -i [ --prompt ] arg              Direct file input

Examples:
	flm run llama3.2:1b
	flm run llama3.2:1b --asr 1
	flm serve llama3.2:1b --pmode balanced
	flm pull llama3.2:1b --force
	flm serve llama3.2:1b --ctx-len 8192
	flm serve llama3.2:1b --socket 10
	flm serve llama3.2:1b --q-len 10
	flm serve llama3.2:1b --port 8000
	flm serve llama3.2:1b --cors 0
	flm serve llama3.2:1b --asr 1
	flm serve llama3.2:1b --embed 1
	flm serve qwen3vl-it:4b --resize 1 (0: original size, 1: height = 480, 2: height = 720, 3: height = 1080)
	flm list
	flm list --quiet
	flm list --filter installed


exit=1

## FastFlowLM installed models

+ flm list --filter installed --json
{
    "models": [
        {
            "asr": true,
            "default_context_length": 32768,
            "details": {
                "family": "gemma4e",
                "format": "NPU2",
                "parameter_size": "5B",
                "quantization_level": "Q4_1",
                "think": true
            },
            "file_url": "https://huggingface.co/api/models/FastFlowLM/Gemma4-E2B-IT-NPU2/tree/main",
            "files": [
                "config.json",
                "model.q4nx",
                "tokenizer.json",
                "tokenizer_config.json",
                "vision_weight.q4nx",
                "audio_weight.q4nx",
                "chat_template.jinja"
            ],
            "flm_min_version": "0.9.39",
            "footprint": 6.0,
            "installed": true,
            "label": [
                "vision",
                "reasoning",
                "tool-calling",
                "audio",
                "transcription"
            ],
            "max_prefill_len": 4096,
            "model": "gemma4-it:e2b",
            "name": "gemma4-it:e2b",
            "size": 2000000000,
            "url": "https://huggingface.co/FastFlowLM/Gemma4-E2B-IT-NPU2",
            "vlm": true
        },
        {
            "default_context_length": 32768,
            "details": {
                "family": "lfm2",
                "format": "NPU2",
                "parameter_size": "1.2B",
                "quantization_level": "Q4_0",
                "think": false,
                "think_toggleable": false
            },
            "file_url": "https://huggingface.co/api/models/FastFlowLM/LFM2.5-1.2B-NPU2/tree/main",
            "files": [
                "config.json",
                "model.q4nx",
                "tokenizer.json",
                "tokenizer_config.json"
            ],
            "flm_min_version": "0.9.25",
            "footprint": 0.96,
            "installed": true,
            "max_prefill_len": 512,
            "model": "lfm2.5-it:1.2b",
            "name": "lfm2.5-it:1.2b",
            "size": 1200000000,
            "url": "https://huggingface.co/FastFlowLM/LFM2.5-1.2B-NPU2",
            "vlm": false
        },
        {
            "default_context_length": 131072,
            "details": {
                "family": "llama3",
                "parameter_size": "1B",
                "quantization_level": "Q4_1",
                "think": false
            },
            "file_url": "https://huggingface.co/api/models/FastFlowLM/Llama-3.2-1B-NPU2/tree/v0.9.21-merge-lm-head",
            "files": [
                "config.json",
                "model.q4nx",
                "tokenizer.json",
                "tokenizer_config.json"
            ],
            "flm_min_version": "0.9.21",
            "footprint": 1.3,
            "installed": true,
            "max_prefill_len": 512,
            "model": "llama3.2:1b",
            "modified_at": "2025-05-30T00:00:00Z",
            "name": "llama3.2:1b",
            "size": 1000000000,
            "url": "https://huggingface.co/FastFlowLM/Llama-3.2-1B-NPU2/resolve/v0.9.21-merge-lm-head"
        },
        {
            "default_context_length": 65536,
            "details": {
                "family": "llama3",
                "parameter_size": "3B",
                "quantization_level": "Q4_1",
                "think": false
            },
            "file_url": "https://huggingface.co/api/models/FastFlowLM/Llama-3.2-3B-NPU2/tree/v0.9.21-merging-lm-head",
            "files": [
                "config.json",
                "model.q4nx",
                "tokenizer.json",
                "tokenizer_config.json"
            ],
            "flm_min_version": "0.9.21",
            "footprint": 2.7,
            "installed": true,
            "max_prefill_len": 512,
            "model": "llama3.2:3b",
            "modified_at": "2025-05-30T00:00:00Z",
            "name": "llama3.2:3b",
            "size": 3000000000,
            "url": "https://huggingface.co/FastFlowLM/Llama-3.2-3B-NPU2/resolve/v0.9.21-merging-lm-head"
        },
        {
            "default_context_length": 32768,
            "details": {
                "family": "qwen2",
                "parameter_size": "3B",
                "quantization_level": "Q4_0",
                "think": true,
                "think_toggleable": false
            },
            "file_url": "https://huggingface.co/api/models/FastFlowLM/Qwen2.5-3B-Instruct-NPU2/tree/main",
            "files": [
                "config.json",
                "model.q4nx",
                "tokenizer.json",
                "tokenizer_config.json"
            ],
            "flm_min_version": "0.9.32",
            "footprint": 2.5,
            "installed": true,
            "max_prefill_len": 512,
            "model": "qwen2.5-it:3b",
            "modified_at": "2025-05-30T00:00:00Z",
            "name": "qwen2.5-it:3b",
            "size": 3000000000,
            "url": "https://huggingface.co/FastFlowLM/Qwen2.5-3B-Instruct-NPU2/resolve/main"
        },
        {
            "default_context_length": 32768,
            "details": {
                "family": "qwen3-it",
                "parameter_size": "4B",
                "quantization_level": "Q4_1",
                "think": false,
                "think_toggleable": false
            },
            "file_url": "https://huggingface.co/api/models/FastFlowLM/Qwen3-4B-Instruct-2507-NPU2/tree/v0.9.22-faster-q4-1",
            "files": [
                "config.json",
                "model.q4nx",
                "tokenizer.json",
                "tokenizer_config.json"
            ],
            "flm_min_version": "0.9.22",
            "footprint": 3.1,
            "installed": true,
            "label": [
                "tool-calling"
            ],
            "max_prefill_len": 512,
            "model": "qwen3-it:4b",
            "modified_at": "2025-05-30T00:00:00Z",
            "name": "qwen3-it:4b",
            "size": 4000000000,
            "url": "https://huggingface.co/FastFlowLM/Qwen3-4B-Instruct-2507-NPU2/resolve/v0.9.22-faster-q4-1"
        },
        {
            "default_context_length": 32768,
            "details": {
                "family": "qwen3.5",
                "format": "NPU2",
                "parameter_size": "2B",
                "quantization_level": "Q4_1",
                "think": true
            },
            "file_url": "https://huggingface.co/api/models/FastFlowLM/Qwen3.5-2B-NPU2/tree/main",
            "files": [
                "config.json",
                "model.q4nx",
                "tokenizer.json",
                "tokenizer_config.json",
                "vision_weight.q4nx",
                "chat_template.jinja"
            ],
            "flm_min_version": "0.9.37",
            "footprint": 3.2,
            "installed": true,
            "label": [
                "vision",
                "reasoning",
                "tool-calling"
            ],
            "max_prefill_len": 512,
            "model": "qwen3.5:2b",
            "name": "qwen3.5:2b",
            "size": 2000000000,
            "url": "https://huggingface.co/FastFlowLM/Qwen3.5-2B-NPU2",
            "vlm": true
        }
    ]
}

exit=0

## FastFlowLM validation

+ flm validate --json
{
    "all_fw_ok": true,
    "amd_device_found": false,
    "devices": [],
    "enough_cols": true,
    "kernel": "6.17.0-35-generic",
    "kernel_ok": true,
    "memlock_limit": "infinity",
    "memlock_ok": true,
    "object": "npu_stack_validation",
    "platform": "linux",
    "ready": false
}

exit=1

## XRT version

+ xrt-smi --version
  Version              : 2.21.75
  virtio-pci Version   : 6.17.0-35-generic
  amdxdna Version      : 6.17.0-35-generic

exit=0

## XRT examine

+ xrt-smi examine
System Configuration
  OS Name              : Linux
  Release              : 6.17.0-35-generic
  Machine              : x86_64
  CPU Cores            : 32
  Memory               : 124415 MB
  Distribution         : Linux Mint 22.3
  GLIBC                : 2.39
  Model                : MME3L
  BIOS Vendor          : American Megatrends International, LLC.
  BIOS Version         : 3.05
  Processor            : AMD RYZEN AI MAX+ 395 w/ Radeon 8060S

XRT
  Version              : 2.21.75
  virtio-pci Version   : 6.17.0-35-generic
  amdxdna Version      : 6.17.0-35-generic

Device(s) Present
  0 devices found

exit=0

## xdna-top snapshot

+ xdna-top --json snapshot
{
  "backends": {
    "igpu": {
      "primary": "sysfs",
      "signals": {
        "busy_pct": "sysfs",
        "power_w": "sysfs"
      }
    },
    "npu": {
      "fallbacks_used": [],
      "primary": "xrt_smi",
      "signals": {
        "contexts": null,
        "device": null,
        "driver": null,
        "power_state": null,
        "sensors": null
      }
    }
  },
  "captured_at": "2026-08-25T01:09:28.958785Z",
  "commands": {
    "xrt_smi": {
      "aie_partitions_returncode": 1,
      "available": true,
      "examine_returncode": 0,
      "path": "/usr/bin/xrt-smi",
      "version_output": "Version : 2.21.75 virtio-pci Version : 6.17.0-35-generic amdxdna Version : 6.17.0-35-generic"
    }
  },
  "degraded": {
    "igpu": {
      "degraded": false,
      "reasons": []
    },
    "npu": {
      "degraded": true,
      "reasons": [
        "aie_partitions_report_failed"
      ]
    },
    "overall": true
  },
  "devices": {
    "accel": {
      "entries": [
        {
          "exists": false,
          "path": "/dev/accel/accel0"
        }
      ]
    },
    "igpu": {
      "busy_path": "/sys/class/drm/card0/device/gpu_busy_percent",
      "busy_path_exists": true,
      "power_path": "/sys/class/hwmon/hwmon5/power1_input",
      "power_path_exists": true
    },
    "npu": {
      "bdf": null,
      "contexts": [],
      "detected": false,
      "driver": {
        "drm_version": null,
        "supports_sensors": false
      },
      "ioctl": {
        "aie": null,
        "aie_version": null,
        "available": false,
        "clocks": null,
        "driver": null,
        "firmware_version": null,
        "node": null,
        "reason": "accel_absent",
        "sensors": {
          "available": false,
          "items": [],
          "reason": null
        },
        "source": null,
        "supports_sensors": false
      },
      "name": null,
      "power_state": {
        "available": false,
        "dpm": null,
        "powerstate": null,
        "reason": "debugfs_accel_absent",
        "source": null
      },
      "report_shape": {
        "context_count": 0,
        "has_aie_partitions": false,
        "has_hw_contexts": false
      },
      "sensors": {
        "column_utilization_pct": {
          "source": null,
          "value": null
        },
        "power_w": {
          "source": null,
          "value": null
        }
      }
    }
  },
  "errors": [
    {
      "message": "ERROR: Please specify a device using --device option Available devices:",
      "probe": "xrt_smi.aie_partitions"
    }
  ],
  "host": {
    "hostname": "<host>",
    "kernel": {
      "release": "6.17.0-35-generic",
      "version": "#35~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Tue May 26 19:30:42 UTC 2"
    },
    "os": {
      "id": "linuxmint",
      "pretty_name": "Linux Mint 22.3",
      "version_id": "22.3"
    },
    "python": {
      "version": "3.12.3"
    },
    "xdna_top": {
      "version": "0.1.0"
    }
  },
  "kind": "xdna-top.snapshot",
  "schema_version": "1.0",
  "telemetry": {
    "gpu_busy_pct": 0,
    "gpu_power_w": 13.041,
    "igpu_degraded": false,
    "npu_active": false,
    "npu_degraded": true,
    "state": "IDLE",
    "ts": 1787620168.9585958
  }
}

exit=0

## ROCm probe

+ rocminfo
[37mROCk module is loaded[0m
[31mUnable to open /dev/kfd read-write: No such file or directory[0m
[31mFailed to get user name to check for video group membership[0m

exit=1

## ROCm SMI

+ rocm-smi


======================================== ROCm System Management Interface ========================================
================================================== Concise Info ==================================================
Device  Node  IDs              Temp    Power     Partitions          SCLK  MCLK  Fan  Perf  PwrCap  VRAM%  GPU%
[3m              (DID,     GUID)  (Edge)  (Socket)  (Mem, Compute, ID)                                              [0m
==================================================================================================================
0       1     0x1586,   51259  36.0°C  13.065W   N/A, N/A, 0         N/A   N/A   0%   auto  N/A     3%     0%
==================================================================================================================
============================================== End of ROCm SMI Log ===============================================

exit=0

## FLM Llama 1B smoke

+ bash -c printf\ \'Reply\ with\ one\ short\ greeting.\\n\'\ \|\ flm\ run\ llama3.2:1b\ -i\ /dev/stdin
Error:  No such device with index '0'
Command exited with non-zero status 1
elapsed_seconds=0.03 exit_status=1

exit=1

## FLM Gemma ASR smoke

+ bash -c printf\ \'Reply\ with\ one\ short\ greeting.\\n\'\ \|\ flm\ run\ gemma4-it:e2b\ --asr\ 1\ -i\ /dev/stdin
Error:  No such device with index '0'
Command exited with non-zero status 1
elapsed_seconds=0.03 exit_status=1

exit=1
