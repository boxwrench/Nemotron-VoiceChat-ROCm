# S10 public Linux VitisAI provider recovery audit

Status: **PUBLIC_RECOVERY_POSSIBLE — userspace import gate passed**

This is packaging and ABI research only. It does not compile or execute a
VoiceChat ONNX graph, and it does not change any S7/S8/S9 arithmetic,
provider-assignment, or XDNA-execution classification.

## Decision

```text
PUBLIC_RECOVERY_POSSIBLE       YES
PUBLIC_SOURCE_BUILD_POSSIBLE   PARTIAL ONLY; not a complete VAIP replacement
PRECOMPILED_XDNA_ROUTE_POSSIBLE LIMITED; model-specific FLM deployment only
GATED_AMD_PAYLOAD_REQUIRED     NO for the isolated 1.7.1 provider userspace
                                YES only for AMD's archive/installer and any
                                driver-package replacement (not needed here)
```

AMD's public `pypi.amd.com/ryzenai_llm/1.7.1/linux/simple/` index supplies
the Linux wheels that the gated `ryzen_ai-1.7.1.tgz` installer materializes.
An isolated CPython 3.12 environment built from those wheels registered:

```text
onnxruntime 1.23.3.dev20260320
['VitisAIExecutionProvider', 'CPUExecutionProvider']
```

This establishes only that the provider userspace can be reconstructed. A
provider session, operator assignment, compile, cache, or NPU execution has
**not** been attempted in this audit.

## Dependency map

```text
VoiceChat custom ONNX
        |
        v
onnxruntime-vitisai 1.23.3 (Python binding + VitisAI EP)
        |
        +-- voe 1.7.1 (VAIP/VOE compiler/runtime libraries)
        |       |
        |       +-- libxcompiler-core-without-symbol.so
        |       +-- libvaiml2.so / libaiecompiler_client.so
        |       +-- libdyn_dispatch_core.so / VART / XIR
        |       `-- libonnxruntime_vitisai_ep.so
        |
        +-- onnxruntime-providers-ryzenai 0.11.1
        |       `-- DynamicDispatch custom-op runtime
        |
        +-- ryzenai-dynamic-dispatch 1.7.1
        |       `-- transaction/operator metadata and dispatch library
        |
        `-- XRT libxrt_coreutil.so.2 (host 2.21.75)
                |
                v
             amdxdna + XDNA2 firmware + /dev/accel/accel0
```

The hosted 1.7.1 provider binary has a `libxrt_coreutil.so.2` dependency and
its runpath names `/opt/xilinx/xrt/lib`. The current host exposes the same SONAME
through XRT 2.21.75, so no XRT replacement is justified. Activation puts the
isolated VOE library directory first in `LD_LIBRARY_PATH`; it does not modify
the global linker configuration.

## Component inventory

| Component | Ryzen AI 1.7.1 expectation | License / source | Public binary | Current host | Local-build assessment | ABI / package notes |
|---|---|---|---|---|---|---|
| CPython | 3.12 wheel ABI | PSF | distro/user interpreter | present | n/a | provider and DynamicDispatch wheels are `cp312` |
| NumPy | 1.x required in practice | BSD | PyPI | not part of provider env | n/a | `numpy==1.26.4`; NumPy 2.x prevents ORT import |
| ONNX Runtime VitisAI | `onnxruntime-vitisai==1.23.3` | MIT; ORT source public | AMD public wheel | absent before S10 | EP glue source is public | wheel requires `voe==1.7.1` |
| VitisAI EP library | `libonnxruntime_providers_vitisai.so` | delivered under AMD's MIT-labelled ORT wheel | bundled above | absent before S10 | cannot operate without VAIP/VOE | provider presence is not assignment proof |
| VAIP / VOE | `voe==1.7.1` | wheel metadata Apache-2.0; full compiler source not found in public audit | AMD public 385 MiB wheel | absent before S10 | **not** reconstructible from public source alone | includes xcompiler, VAIML, VART/XIR and EP runtime libs |
| Ryzen custom ops | `onnxruntime-providers-ryzenai==0.11.1` | MIT | AMD public wheel | absent before S10 | source-level alternatives incomplete | supplies `libonnxruntime_providers_ryzenai.so` and `libryzen_mm.so` |
| Dynamic Dispatch | `ryzenai-dynamic-dispatch==1.7.1` | MIT metadata | AMD public wheel | absent before S10 | interfaces/headers are public; complete operator binaries ship in wheel | includes transaction runtime; custom graph support remains untested |
| ONNX utilities | `ryzenai-onnx-utils==0.12.0` | license included in wheel | AMD public wheel | Lemonade cache had utility code only | Python package is public | graph tooling, not itself an NPU runtime |
| Model generation | `model-generate==1.7.1` | MIT | AMD public wheel | absent before S10 | Python wrapper is public | recipes target recognised LLM/VLM flows; not needed for, or evidence of, arbitrary VoiceChat support |
| XRT | `2.21.75` | Apache-2.0 user-space package | public AMD packages | **present**: 2.21.75 | public source exists | required SONAME: `libxrt_coreutil.so.2` |
| `amdxdna` | host driver/firmware | GPL-2.0 driver; AMD proprietary firmware | distro/AMD package | **present and working** | not relevant to userspace recovery | do not replace; exact host evidence remains authoritative |
| FLM / Lemonade | optional existing applications | respective packages | installed | FLM 0.9.39; Lemonade 10.6.0 | n/a | neither installs VitisAI EP/VAIP |

## What the gated archive contributes

The account-gated archive remains a supported convenience bundle: installer,
quicktest assets, examples, and the wheel set. It is not required to obtain the
1.7.1 provider userspace because the public AMD index supplies the required
wheels. The archive may still be useful for AMD's exact quicktest fixture, but
it is not a dependency of the custom-ONNX qualification path.

The audit did **not** find a public AMD apt repository that offers the missing
VAIP userspace as a distro package. The existing public AMD packages on this
host cover XRT/amdxdna, which must remain unchanged.

## Model packages and containers

AMD/FastFlowLM Hugging Face packages are deployment artifacts, not SDK
payloads. The inspected `Whisper-V3-Turbo-NPU2` package contains the model plus
fixed `encoder_*.xclbin`, `whisper_head.xclbin`, and `model.q4nx` assets. It
does not contain VAIP, VitisAI EP, a general ONNX compiler, or reusable compiler
configuration. This confirms:

```text
PRECOMPILED_MODEL_RUNTIME != ARBITRARY_ONNX_COMPILER
```

AMD's public container material found in this audit is older framework-specific
Vitis AI content, not a current documented Ryzen AI 1.7.1 Linux generic-EP
image. A container would also need explicit `/dev/accel/accel0` access and an
ABI-compatible host XRT boundary. It is not a lower-risk recovery route than
the isolated public-wheel environment and was not used.

## Route assessment

### Route A — public binary/package reconstruction: recommended

**Viable.** The public 1.7.1 wheel chain is sufficient to create an isolated
provider environment, with the exact XRT 2.21.75 ABI already present. Use
`provision_public_ryzenai_1_7_1.sh` with a new target directory. It creates no
session and performs no system mutation.

Remaining gate: host-side provider quicktest, then the S8 primitive assignment
probe. Provider import is not evidence that a custom graph compiles or offloads.

### Route B — public source build: not a complete recovery route

Public sources cover ONNX Runtime EP glue and XIR graph infrastructure, but
the required VAIP/VOE compiler/runtime implementation (`xcompiler`, VAIML,
FlexML/VART linkage and target/operator payloads) was not found as a complete
public source build. The public binary `voe` wheel supplies it instead.

Therefore a source-only rebuild is not justified while Route A exists. Building
only ORT or XIR would leave the compiler/runtime missing and risks an ABI split.

### Route C — precompiled/runtime reuse: limited, not arbitrary-ONNX capable

FastFlowLM already has model-specific native runtimes and XRT `xclbin` assets.
For example, its Whisper package contains model data plus fixed encoder/decoder
`*.xclbin` files and FLM native libraries. Lemonade exposes supporting tooling
but no discovered generic VAIP provider library or custom-graph API.

That is a **precompiled-model runtime**, not a demonstrated arbitrary-ONNX
compiler/deployment route. It is useful evidence for XRT execution only; it
cannot yet qualify the VoiceChat exact-Q8 graph without a new FLM/Lemonade
interface or independently produced compatible artifacts.

The public `model-generate==1.7.1` wrapper delegates model recipes to
`ryzenai-onnx-utils` and DynamicDispatch. It is useful prior art for packaging
compiled artifacts, but its documented modes target LLM/VLM recipes rather than
establishing support for the VoiceChat exact-Q8 stateful encoder.

## Isolated provisioning plan

From an ordinary host shell, with no NPU job holding the device:

```bash
cd <repo>
research/experiments/strix-fastconformer-xdna/provision_public_ryzenai_1_7_1.sh \
  <isolated-parent>/public-ryzenai-1.7.1
source <isolated-parent>/public-ryzenai-1.7.1/activate-ryzenai.sh
python -c 'import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())'
```

Expected provider gate:

```text
VitisAIExecutionProvider
CPUExecutionProvider
```

Only after this import check may S9 run a supplied/provider quicktest and then
the existing exact-Q8 primitive assignment probe. Use a fresh dedicated cache
per graph and provider/XRT version; do not reuse compiler contexts across ABI
versions.

## Evidence boundary

```text
Linux XDNA hardware/runtime             previously proven via FLM
Linux XDNA speech runtime               previously proven via FLM Whisper
Public VitisAI EP userspace import      proven in isolated CPython 3.12 venv
VitisAI provider session/assignment     not tested
VoiceChat exact-Q8 compilation           not tested
VoiceChat exact-Q8 NPU execution         not tested
```

## DEC field-study update

```text
methodology lesson:
  hardware capability, application-runtime capability, generic-provider
  availability, and compiler/operator assignment are separate evidence classes.

proprietary-packaging boundary:
  an account-gated SDK archive may not be the only distribution channel for
  its userspace components. Audit package indexes and wheel contents before
  classifying a compiler path as unavailable.

scope:
  SYSTEM_ATTRIBUTION / environment availability, not DEC_CORE. No candidate
  domain, contributor elimination, compiler assignment, or XDNA result is
  asserted by this audit.
```
