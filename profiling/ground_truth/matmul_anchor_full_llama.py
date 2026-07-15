#!/usr/bin/env python3
"""matmul-anchor: sum ALL matmul latencies of a Llama layer x32 -> vs real prefill 28.1ms.

Prefill is matmul-compute-bound, so summing just the matmuls (anchors) should give the
layer latency IF the matmul measurements were faithful. Our CompileModule measurements are
~12x inflated at S=2048, so this will show ~12x -- confirming the fidelity gap dominates
regardless of combiner (matmul-anchor vs n-gram).

Llama-3.1-8B S=2048 TP=8: 7 proj/mlp matmuls + 2 attention-core matmuls.
"""
import statistics as st
from collections import defaultdict

import torch
import furiosa.torch  # noqa: F401
from furiosa.torch import native_device as nd

nd.set_fusion(8)

from furiosa.torch.custom_ops import CompileModule  # noqa: E402

D, NH, HD, KV, INTER, L, VOCAB = 4096, 32, 128, 8, 14336, 32, 128256
B, S = 1, 2048
DTYPE = torch.bfloat16
GT_PREFILL_MS = 28.1


class MM(torch.nn.Module):
    """generic matmul x@w; 'oproj' also does the attention-output reshape."""
    def __init__(self, w_shape, reshape_oproj=False):
        super().__init__()
        self.register_buffer("w", torch.randn(*w_shape, dtype=DTYPE))
        self.reshape_oproj = reshape_oproj

    def forward(self, x):
        if self.reshape_oproj:
            x = x.permute(0, 2, 1, 3).reshape(B, S, NH * HD)
        return x @ self.w


def measure(in_shape, w_shape, reshape_oproj=False):
    dev = torch.device("rngd", 0)
    x = torch.randn(*in_shape, dtype=DTYPE)
    cm = CompileModule.from_exported(torch.export.export(MM(w_shape, reshape_oproj), (x,)))
    cm.to(dev); xd = x.to(dev)
    cm(xd, profiles=None, device=dev)
    ts = []
    for _ in range(3):
        pr = cm.generate_profiles(dev); cm(xd, profiles=pr, device=dev)
        di = [[p.device.index for p in inn] for inn in pr]
        pc = [[p.cpu() for p in inn] for inn in pr]
        sp = cm.edf.npu_node.build_tuc_profile_spans(pc, di, 10**9)
        ts.append(sum(e.time_range.elapsed_us() for e in sp if e.name == "Task"))
    return st.median(ts)


# (name, input_shape, weight_shape, reshape_oproj)
MATMULS = [
    ("q_proj",  (B, S, D),        (D, NH * HD),    False),
    ("k_proj",  (B, S, D),        (D, KV * HD),    False),
    ("v_proj",  (B, S, D),        (D, KV * HD),    False),
    ("qk",      (B, NH, S, HD),   (B, NH, HD, S),  False),
    ("av",      (B, NH, S, S),    (B, NH, S, HD),  False),
    ("o_proj",  (B, NH, S, HD),   (NH * HD, D),    True),
    ("gate",    (B, S, D),        (D, INTER),      False),
    ("up",      (B, S, D),        (D, INTER),      False),
    ("down",    (B, S, INTER),    (INTER, D),      False),
]


def main():
    print(f"=== matmul-anchor full Llama (S={S} TP=8) ===", flush=True)
    per_layer = 0.0
    for name, insh, wsh, rp in MATMULS:
        t = measure(insh, wsh, rp)
        per_layer += t
        print(f"  {name:8s} {t:9.1f}us", flush=True)
    lm_head = measure((B, 1, D), (D, VOCAB))
    total = per_layer * L + lm_head
    print(f"\n  per-layer matmul sum = {per_layer:.1f}us", flush=True)
    print(f"  x{L} + lm_head({lm_head:.0f}us) = {total/1000:.1f} ms", flush=True)
    print(f"  real prefill = {GT_PREFILL_MS} ms  -> {total/1000/GT_PREFILL_MS:.1f}x", flush=True)


if __name__ == "__main__":
    main()
