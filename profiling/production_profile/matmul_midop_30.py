#!/usr/bin/env python3
"""matmul A → [중간op] → matmul B 의 overlap (2026.3.0, host). 중간 연산이 fusion 깨나 테스트.

중간op: none / silu / rmsnorm. overlap = U(A)+U(B)−W₂. 중간op별 overlap 비교 →
사이 연산이 HBM 왕복 강제해 overlap 줄이는지(fusion barrier) 확인. (floor는 3케이스 공통이라 상쇄.)

사용: /home/furiosa/venv3030/bin/python matmul_midop_30.py <M> <KA> <NA> <NB> <midop>
출력: MID M KA NA NB midop uA uB w2 overlap
"""
import os, sys, time, statistics as st

import torch
import furiosa.torch as ft
ft.set_fusion(8)
from furiosa.torch._C.config.compiler import CompilerConfig, TacticHintConfig

M, KA, NA, NB = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
MID = sys.argv[5]  # none / silu / rmsnorm
DEV = "furiosa:0"; DT = torch.bfloat16; EPS = 1e-5


def cfg():
    return CompilerConfig(tactic_hint=TacticHintConfig.ForLlmModelComputeBound,
                          scheduler_beam_search=True, use_attention_kernel=True)


def mid(t, w):
    if MID == "silu": return torch.nn.functional.silu(t)
    if MID == "rmsnorm":
        v = (t.to(torch.float32) ** 2).mean(-1, keepdim=True)
        return (t * torch.rsqrt(v.to(DT) + EPS)) * w
    return t


class Amod(torch.nn.Module):
    def __init__(s): super().__init__(); s.register_buffer("w", torch.randn(KA, NA, dtype=DT))
    def forward(s, x): return torch.mm(x, s.w)


class Bmod(torch.nn.Module):
    def __init__(s): super().__init__(); s.register_buffer("w", torch.randn(NA, NB, dtype=DT))
    def forward(s, x): return torch.mm(x, s.w)


class Chain(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.register_buffer("wa", torch.randn(KA, NA, dtype=DT)); s.register_buffer("wb", torch.randn(NA, NB, dtype=DT))
        s.register_buffer("nw", torch.randn(NA, dtype=DT))
    def forward(s, x): return torch.mm(mid(torch.mm(x, s.wa), s.nw), s.wb)


def host_us(mod, x):
    cm = ft.CompileModule.from_module(mod, (x,), compiler_config=cfg()).to(DEV)
    xd = x.to(DEV); s = ft.current_stream(DEV)
    for _ in range(10):
        cm(xd)
    s.synchronize()
    reps = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(30):
            cm(xd)
        s.synchronize(); reps.append((time.perf_counter() - t0) / 30 * 1e6)
    return st.median(reps)


def main():
    uA = host_us(Amod(), torch.randn(M, KA, dtype=DT))
    uB = host_us(Bmod(), torch.randn(M, NA, dtype=DT))
    w2 = host_us(Chain(), torch.randn(M, KA, dtype=DT))
    print(f"MID {M} {KA} {NA} {NB} {MID} {uA:.1f} {uB:.1f} {w2:.1f} {uA+uB-w2:.1f}", flush=True)
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
