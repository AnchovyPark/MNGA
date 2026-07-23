#!/usr/bin/env python3
"""n-gram 예측 실험 (2026.3.0): 개별 격리 측정(재료)으로 실제 in-model Tokenwise(ground truth)를 예측.

- ground truth = 실제 Llama forward path의 Tokenwise supertask 실측 (128:334, 512:752, 1024:1299 us @1.1GHz).
- 재료 = op-identical 각 커널을 하나씩 따로 컴파일+실행한 latency.
- 조합 법칙 후보: naive_all(전부 합), naive_mm(matmul만 합). → ground truth와 대조.

사용: /home/furiosa/venv3030/bin/python materials_predict_30.py <seq>
"""
import os, sys, time, statistics as st

import torch
import furiosa.torch as ft
ft.set_fusion(8)
from furiosa.torch._C.config.compiler import CompilerConfig, TacticHintConfig

S = int(sys.argv[1]) if len(sys.argv) > 1 else 512
DEV = "furiosa:0"
D, INTER, NH, HD, KV = 2048, 8192, 32, 64, 8
DT = torch.bfloat16
EPS = 1e-5
ITERS, WARMUP = 30, 8
INMODEL = {128: 334, 512: 752, 1024: 1299}


def cfg():
    return CompilerConfig(tactic_hint=TacticHintConfig.ForLlmModelComputeBound,
                          scheduler_beam_search=True, use_attention_kernel=True)


class MM(torch.nn.Module):
    def __init__(s, K, N): super().__init__(); s.register_buffer("w", torch.randn(K, N, dtype=DT))
    def forward(s, x): return torch.mm(x, s.w)


class RMSNorm(torch.nn.Module):
    def __init__(s): super().__init__(); s.register_buffer("w", torch.randn(D, dtype=DT))
    def forward(s, x):
        v = (x.to(torch.float32) ** 2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(v.to(DT) + EPS)) * s.w


class RoPE(torch.nn.Module):
    def __init__(s, heads):
        super().__init__(); s.h = heads
        s.register_buffer("cos", torch.randn(1, 1, HD, dtype=DT))
        s.register_buffer("sin", torch.randn(1, 1, HD, dtype=DT))
    def forward(s, x):
        t = x.view(S, s.h, HD)
        x1 = t[..., :HD//2]; x2 = t[..., HD//2:]
        rot = torch.cat((-x2, x1), dim=-1)
        return (t * s.cos + rot * s.sin).reshape(S, s.h * HD)


class Resid(torch.nn.Module):
    def forward(s, a, b): return a + b


def timeit(mod, *xs):
    cm = ft.CompileModule.from_module(mod, xs, compiler_config=cfg()).to(DEV)
    xds = [x.to(DEV) for x in xs]; s = ft.current_stream(DEV)
    for _ in range(WARMUP): cm(*xds)
    s.synchronize()
    reps = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(ITERS): cm(*xds)
        s.synchronize(); reps.append((time.perf_counter()-t0)/ITERS*1e6)
    return st.median(reps)


def P(*a): print(*a, flush=True)


def main():
    P(f"=== 재료(개별) → 예측 vs ground truth : seq={S} TP=8 ===")
    mm = {
        "q":  timeit(MM(D, NH*HD), torch.randn(S, D, dtype=DT)),
        "k":  timeit(MM(D, KV*HD), torch.randn(S, D, dtype=DT)),
        "v":  timeit(MM(D, KV*HD), torch.randn(S, D, dtype=DT)),
        "o":  timeit(MM(NH*HD, D), torch.randn(S, NH*HD, dtype=DT)),
        "gate": timeit(MM(D, INTER), torch.randn(S, D, dtype=DT)),
        "up":   timeit(MM(D, INTER), torch.randn(S, D, dtype=DT)),
        "down": timeit(MM(INTER, D), torch.randn(S, INTER, dtype=DT)),
    }
    other = {
        "rmsnorm1": timeit(RMSNorm(), torch.randn(S, D, dtype=DT)),
        "rmsnorm2": timeit(RMSNorm(), torch.randn(S, D, dtype=DT)),
        "rope_q":   timeit(RoPE(NH), torch.randn(S, NH*HD, dtype=DT)),
        "rope_k":   timeit(RoPE(KV), torch.randn(S, KV*HD, dtype=DT)),
        "resid1":   timeit(Resid(), torch.randn(S, D, dtype=DT), torch.randn(S, D, dtype=DT)),
        "resid2":   timeit(Resid(), torch.randn(S, D, dtype=DT), torch.randn(S, D, dtype=DT)),
    }
    P("[재료 - matmul]")
    for k, val in mm.items(): P(f"    {k:>6}: {val:7.1f}us")
    P("[재료 - 비matmul]")
    for k, val in other.items(): P(f"    {k:>8}: {val:7.1f}us")
    sum_mm = sum(mm.values()); sum_all = sum_mm + sum(other.values())
    gt = INMODEL.get(S)
    P(f"\nnaive_mm  (matmul만 합) = {sum_mm:7.1f}us" + (f"  → GT {gt} 대비 {sum_mm/gt:.2f}x" if gt else ""))
    P(f"naive_all (전부 합)     = {sum_all:7.1f}us" + (f"  → GT {gt} 대비 {sum_all/gt:.2f}x" if gt else ""))
    if gt: P(f"ground truth (in-model) = {gt}us")
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
