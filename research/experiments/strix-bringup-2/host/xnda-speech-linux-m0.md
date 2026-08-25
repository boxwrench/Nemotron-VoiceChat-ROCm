# XDNA-SPEECH-LINUX-M0 host evidence

This is a sanitized capture summary from an ordinary host shell. Usernames,
hostnames, PIDs, and local absolute paths are omitted.

## Workload

```text
runtime                     FastFlowLM standalone ASR
model                       Whisper-V3-Turbo-NPU2
endpoint                    POST /v1/audio/transcriptions
fixture duration            3.920 s
```

The observed transcript was:

```text
The capital of France is Paris.
```

FastFlowLM runtime evidence showed:

```text
NPU Locked
Transforming audio to text
NPU Lock Released
```

## XDNA activity proof

During the request, `xdna-top` showed active XDNA contexts owned by the FLM
process. The observed context submission/completion pairs were:

```text
ctx4 440/440
ctx1 200/200
ctx3 132/132
ctx2  32/32
ctx5  11/11
```

## Acceptance

```text
XDNA-SPEECH-LINUX-M0      PASS
```

This proves a Linux XDNA speech workload. It is auxiliary ASR/runtime evidence
only; Whisper is not being substituted for VoiceChat's learned FastConformer
perception embeddings.
