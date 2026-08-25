# Export environment

This is the isolated environment used for the preserved Bringup-3 ONNX
preparation WIP. It is deliberately local to the experiment and is not part
of the repository checkpoint.

## Recorded environment

```text
Python       3.12.3
numpy        2.5.2
onnx         1.22.0
onnxruntime  1.29.0
```

The environment was created with the system Python and contains only the
packages needed for direct ONNX construction/checking and CPU inspection:

```bash
cd <repo>
python3 -m venv research/experiments/strix-fastconformer-xdna/.venv
research/experiments/strix-fastconformer-xdna/.venv/bin/python -m pip install \
    numpy onnx onnxruntime
```

Use the experiment interpreter explicitly when inspecting or extending the
export WIP:

```bash
research/experiments/strix-fastconformer-xdna/.venv/bin/python \
    research/experiments/strix-fastconformer-xdna/export_voicechat_perception_onnx.py \
    --help
```

No NeMo, PyTorch, `torch.onnx`, VitisAI EP, XRT, driver, firmware, or system
package is required for graph construction and CPU ONNX checks. The local
environment, generated graphs, caches, and model weights remain ignored.
The production-shaped input contract is deferred to PC D2.
