# S9 isolated Ryzen AI provider provisioning

Status: **BLOCKED — AMD account/EULA archive access required**

S9 needs AMD Ryzen AI Software 1.7.1 for Linux as a separate userspace
environment. AMD documents the archive name as `ryzen_ai-1.7.1.tgz` and its
Linux installer as `install_ryzen_ai.sh -a yes -p <target>/venv`. The release's
documented XRT version is 2.21.75, matching the existing host stack.

The archive is supplied through AMD's account/EULA download flow. The coding
environment cannot establish that authenticated session. No generic ONNX
Runtime, driver package, XRT package, or Python replacement was installed as
a substitute.

## Host-side bootstrap

Once an authenticated operator places the untouched AMD archive on the host,
run the committed bootstrap from an ordinary host shell:

```bash
research/experiments/strix-fastconformer-xdna/provision_ryzenai_1_7_1.sh \
  /path/to/ryzen_ai-1.7.1.tgz \
  /path/to/isolated/ryzen-ai-1.7.1
```

It extracts only to the supplied target and runs AMD's installer with an
isolated virtual-environment target below it. It does not invoke `apt`,
`dpkg`, `pip` outside AMD's installer, or alter the existing FLM/XRT/ROCm
stack.

Afterward, explicitly activate only that environment:

```bash
source /opt/xilinx/xrt/setup.sh
source /path/to/isolated/ryzen-ai-1.7.1/venv/bin/activate
python -c 'import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())'
```

S9 may proceed only if both `VitisAIExecutionProvider` and
`CPUExecutionProvider` appear. The supplied AMD `quicktest/quicktest.py` is
then the first executable gate, before any VoiceChat graph.

## Decision boundary

```text
Q8_XDNA_CANDIDATE           retained from S7
VitisAI provider status     not provisioned
XDNA provider execution     not tested
S9 decision                 BLOCKED by AMD archive authentication/access
```

This is not an XDNA hardware failure, generic-provider compiler rejection, or
VoiceChat representation result.

## DEC field-study update

```text
strongest lead: unchanged — exact-Q8 provider assignment remains the next gate
strongest suppression: do not infer arbitrary-graph provider capability from
  the separately proven FLM model-specific NPU path
compiler/backend asymmetry: not measured; the provider package is unavailable
validation lesson: archive access/provisioning is a separate prerequisite from
  hardware and application-runtime proof
taxonomy gap: none added; this is environment availability, not DEC_CORE
```
