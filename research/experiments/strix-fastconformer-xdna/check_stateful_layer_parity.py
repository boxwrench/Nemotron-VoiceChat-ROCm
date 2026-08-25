#!/usr/bin/env python3
"""Compare the stateful ONNX layer against a direct NumPy VoiceChat oracle.

The oracle follows the same source-level operations as tools/mtmd/clip.cpp.
It is intentionally independent of ONNX Runtime and is a graph-construction
check; a ggml-runtime comparison remains a separate host/runtime task.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "build" / "llama-voicechat.cpp" / "tools" / "voicechat"
sys.path.insert(0, str(TOOLS))
from vc_gguf import GGUFSource  # noqa: E402

D = 1024
H = 8
DH = 128
HIST = 70
EPS = 1e-5
SRC = "stt_model.perception.encoder.layers.0."


def w(source: GGUFSource, name: str) -> np.ndarray:
    a = np.asarray(source.f32(SRC + name), dtype=np.float32)
    if a.ndim > 2:
        a = np.squeeze(a, axis=tuple(i for i, size in enumerate(a.shape) if size == 1))
    return a


def linear(source: GGUFSource, name: str, x: np.ndarray) -> np.ndarray:
    return x @ w(source, name).T


def norm(source: GGUFSource, name: str, x: np.ndarray) -> np.ndarray:
    scale = w(source, name + ".weight")
    bias = w(source, name + ".bias")
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + EPS) * scale + bias


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def ffn(source: GGUFSource, prefix: str, norm_name: str, x: np.ndarray) -> np.ndarray:
    y = norm(source, norm_name, x)
    y = silu(linear(source, prefix + ".linear1.weight", y))
    y = linear(source, prefix + ".linear2.weight", y)
    return x + 0.5 * y


def oracle(source: GGUFSource, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
    cur = feed["pre_enc_out"].astype(np.float32)
    cur = ffn(source, "feed_forward1", "norm_feed_forward1", cur)

    y = norm(source, "norm_self_att", cur)
    q = linear(source, "self_attn.linear_q.weight", y).reshape(1, 1, H, DH).transpose(0, 2, 1, 3)
    k_new = linear(source, "self_attn.linear_k.weight", y).reshape(1, 1, H, DH).transpose(0, 2, 1, 3)
    v_new = linear(source, "self_attn.linear_v.weight", y).reshape(1, 1, H, DH).transpose(0, 2, 1, 3)
    k_all = np.concatenate([feed["k_hist"], k_new], axis=2)
    v_all = np.concatenate([feed["v_hist"], v_new], axis=2)

    bias_u = w(source, "self_attn.pos_bias_u").T.reshape(1, H, 1, DH)
    bias_v = w(source, "self_attn.pos_bias_v").T.reshape(1, H, 1, DH)
    content = (q + bias_u) @ np.swapaxes(k_all, -1, -2)

    freqs = feed["pos_freqs"][:, None]
    positions = feed["rel_positions"][None, :]
    theta = freqs * positions
    pos = np.concatenate([np.sin(theta), np.cos(theta)], axis=0).T
    pos = linear(source, "self_attn.linear_pos.weight", pos)
    pos = pos.reshape(1, H, DH, 2 * (HIST + 1) - 1)
    rel = (q + bias_v) @ pos
    scores = (content + rel[..., :HIST + 1]) / np.sqrt(DH)
    probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probs /= np.sum(probs, axis=-1, keepdims=True)
    context = (probs @ v_all).transpose(0, 2, 1, 3).reshape(1, 1, D)
    cur = cur + linear(source, "self_attn.linear_out.weight", context)

    y = norm(source, "norm_conv", cur)
    pw = linear(source, "conv.pointwise_conv1.weight", y)
    signal, gate = np.split(pw, 2, axis=-1)
    glu = signal * (1.0 / (1.0 + np.exp(-gate)))
    conv_seq = np.concatenate([feed["conv_hist"], glu], axis=1).transpose(0, 2, 1)
    dw = w(source, "conv.depthwise_conv.weight")
    if dw.ndim == 3:
        dw = dw[:, 0, :]
    conv = np.sum(conv_seq * dw[None, :, :], axis=-1)[:, :, None].transpose(0, 2, 1)
    conv = norm(source, "conv.batch_norm", conv)
    conv = silu(conv)
    cur = cur + linear(source, "conv.pointwise_conv2.weight", conv)

    cur = ffn(source, "feed_forward2", "norm_feed_forward2", cur)
    out = norm(source, "norm_out", cur)
    return [out, k_new, v_new, glu]


def metric(expected: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    delta = expected.astype(np.float64) - actual.astype(np.float64)
    denom = np.linalg.norm(expected.astype(np.float64)) * np.linalg.norm(actual.astype(np.float64))
    cosine = float(np.sum(expected.astype(np.float64) * actual.astype(np.float64)) / denom) if denom else 0.0
    return {
        "cosine": cosine,
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260825)
    args = ap.parse_args()

    source = GGUFSource(args.input)
    rng = np.random.default_rng(args.seed)
    feed = {
        "pre_enc_out": rng.standard_normal((1, 1, D), dtype=np.float32),
        "k_hist": rng.standard_normal((1, H, HIST, DH), dtype=np.float32),
        "v_hist": rng.standard_normal((1, H, HIST, DH), dtype=np.float32),
        "conv_hist": rng.standard_normal((1, 8, D), dtype=np.float32),
        "pos_freqs": np.exp(-(np.arange(D // 2, dtype=np.float32) * 2 * np.log(np.float32(10000)) / np.float32(D))).astype(np.float32),
        "rel_positions": np.arange(HIST, -HIST - 1, -1, dtype=np.float32),
    }
    expected = oracle(source, feed)
    session = ort.InferenceSession(str(args.graph), providers=["CPUExecutionProvider"])
    actual = session.run(None, feed)
    results = [metric(e, a) for e, a in zip(expected, actual)]
    report = {
        "graph": str(args.graph),
        "reference": "direct NumPy transcription of source-level stateful layer",
        "ort_providers": session.get_providers(),
        "outputs": [
            {"name": n.name, "shape": list(a.shape), **m}
            for n, a, m in zip(session.get_outputs(), actual, results)
        ],
        # ORT and the NumPy oracle both run the dequantized F32 graph, but
        # reduction order differs slightly.  The envelope is intentionally
        # tight and is not a substitute for the later ggml-runtime gate.
        "pass": all(x["max_abs"] < 5e-4 for x in results),
    }
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
