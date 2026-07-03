#!/usr/bin/env python3
"""예측: Llama-3.1-8B prefill(S=128)을 격리 커널 합으로 재구성 → ground truth 대조.

실제 모델 한 layer 가 쓰는 커널을 정확한 shape(GQA 포함)로 TP=8 격리 측정하고,
×32 layer + lm_head 로 합산 → ground truth prefill(S=128)=20.1ms 와 비교.
합과 실제의 차이 = 모델 규모 composition gap(fusion/스케줄 이득).

Llama-3.1-8B: D=4096 NH=32 KV=8 HD=128 INTER=14336 L=32 VOCAB=128256
GQA: q_proj D->4096, k/v_proj D->1024(KV*HD).

사용법: python prefill_kernels_s128.py
"""
import statistics as st
from collections import defaultdict

import torch
import torch.nn.functional as F
import furiosa.torch  # noqa: F401
from furiosa.torch import native_device as nd

nd.set_fusion(8)  # TP=8

from furiosa.torch.custom_ops import CompileModule  # noqa: E402

D, NH, KV, HD, INTER, L, VOCAB = 4096, 32, 8, 128, 14336, 32, 128256
B, S = 1, 128
DTYPE = torch.bfloat16
N_RUNS = 3


def union_us(iv):
    if not iv:
        return 0.0
    iv = sorted(iv); t = 0.0; cs, ce = iv[0]
    for s, e in iv[1:]:
        if s <= ce: ce = max(ce, e)
        else: t += ce - cs; cs, ce = s, e
    return t + (ce - cs)


class K(torch.nn.Module):
    def __init__(self, kind, w=None):
        super().__init__()
        self.kind = kind
        if w is not None:
            self.register_buffer("w", w)

    def forward(self, x):
        k = self.kind
        if k == "matmul":  return x @ self.w
        if k == "silu":    return F.silu(x)
        if k == "mul":     return x * self.w
        if k == "add":     return x + self.w
        if k == "softmax": return torch.softmax(x, dim=-1)
        if k == "rmsnorm":
            v = x.float(); v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-5)
            return (v * self.w.float()).to(x.dtype)
        raise ValueError(k)


def measure(kind, in_shape, w_shape):
    dev = torch.device("rngd", 0)
    w = torch.randn(*w_shape, dtype=DTYPE) if w_shape else None
    x = torch.randn(*in_shape, dtype=DTYPE)
    cm = CompileModule.from_exported(torch.export.export(K(kind, w), (x,)))
    cm.to(dev); xd = x.to(dev)
    cm(xd, profiles=None, device=dev)
    tasks = []
    for _ in range(N_RUNS):
        pr = cm.generate_profiles(dev); cm(xd, profiles=pr, device=dev)
        di = [[p.device.index for p in inner] for inner in pr]
        pc = [[p.cpu() for p in inner] for inner in pr]
        spans = cm.edf.npu_node.build_tuc_profile_spans(pc, di, 10**9)
        task = sum(e.time_range.elapsed_us() for e in spans if e.name == "Task")
        tasks.append(task)
    return st.median(tasks)


# 한 layer 의 prefill 커널 (name, count, in_shape, w_shape)
LAYER = [
    ("input_rmsnorm", 1, (B, S, D),      (D,)),
    ("q_proj",        1, (B, S, D),      (D, NH * HD)),      # D->4096
    ("k_proj",        1, (B, S, D),      (D, KV * HD)),      # D->1024 (GQA)
    ("v_proj",        1, (B, S, D),      (D, KV * HD)),      # D->1024 (GQA)
    ("qk_scores",     1, (B, NH, S, HD), (B, NH, HD, S)),    # ->[B,NH,S,S]
    ("softmax",       1, (B, NH, S, S),  None),
    ("av",            1, (B, NH, S, S),  (B, NH, S, HD)),    # ->[B,NH,S,HD]
    ("o_proj",        1, (B, S, NH * HD),(NH * HD, D)),
    ("post_rmsnorm",  1, (B, S, D),      (D,)),
    ("gate_proj",     1, (B, S, D),      (D, INTER)),
    ("up_proj",       1, (B, S, D),      (D, INTER)),
    ("silu",          1, (B, S, INTER),  None),
    ("mul",           1, (B, S, INTER),  (B, S, INTER)),
    ("down_proj",     1, (B, S, INTER),  (INTER, D)),
    ("residual_add",  2, (B, S, D),      (B, S, D)),
]

KIND = {"input_rmsnorm": "rmsnorm", "post_rmsnorm": "rmsnorm",
        "q_proj": "matmul", "k_proj": "matmul", "v_proj": "matmul",
        "qk_scores": "matmul", "softmax": "softmax", "av": "matmul",
        "o_proj": "matmul", "gate_proj": "matmul", "up_proj": "matmul",
        "silu": "silu", "mul": "mul", "down_proj": "matmul",
        "residual_add": "add"}


def main():
    print(f"=== Llama-3.1-8B prefill S={S} TP=8 커널 격리 합 (per layer) ===", flush=True)
    per_layer = 0.0
    for name, cnt, insh, wsh in LAYER:
        t = measure(KIND[name], insh, wsh)
        per_layer += cnt * t
        print(f"  {name:14s} x{cnt}  {t:8.1f}us  (합 {cnt*t:8.1f})", flush=True)
    print(f"  --- per-layer 합: {per_layer:.1f}us ---", flush=True)

    all_layers = per_layer * L
    # 1회성: final rmsnorm + lm_head (마지막 위치 1토큰만 logits)
    final_norm = measure("rmsnorm", (B, 1, D), (D,))
    lm_head = measure("matmul", (B, 1, D), (D, VOCAB))
    total = all_layers + final_norm + lm_head

    print(f"\n32 layers 합 : {all_layers/1000:.2f} ms", flush=True)
    print(f"final_norm   : {final_norm:.1f} us", flush=True)
    print(f"lm_head      : {lm_head:.1f} us", flush=True)
    print(f"===== 예측 prefill(S=128) = {total/1000:.2f} ms =====", flush=True)
    print(f"ground truth = 20.1 ms  →  차이 {total/1000 - 20.1:+.2f} ms "
          f"({total/1000/20.1*100:.0f}%)", flush=True)


if __name__ == "__main__":
    main()
