#!/usr/bin/env python3
"""연쇄 matmul pair A→B의 overlap 측정 (2026.3.0, host stream-sync = robust).

A: (M,K_A)@(K_A,N_A) → (M,N_A).  B: (M,N_A)@(N_A,N_B) → (M,N_B).  B가 A 출력을 입력받음(연쇄).
U(A), U(B), W₂(A→B) 측정 → overlap = U(A)+U(B)−W₂.  가설: overlap ≈ min(A compute, B weight-load).

host 타이밍(모든 크기 robust, ~40us floor는 상수). 사용:
  /home/furiosa/venv3030/bin/python matmul_pair_30.py <M> <K_A> <N_A> <N_B>
출력: PAIR M K_A N_A N_B uA uB w2 overlap overlap_frac  computeA_flops weightB_bytes
"""
import os, sys, time, statistics as st

import torch
import furiosa.torch as ft
ft.set_fusion(8)
from furiosa.torch._C.config.compiler import CompilerConfig, TacticHintConfig

M, KA, NA, NB = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
DEV = "furiosa:0"
DT = torch.bfloat16


def cfg():
    return CompilerConfig(tactic_hint=TacticHintConfig.ForLlmModelComputeBound,
                          scheduler_beam_search=True, use_attention_kernel=True)


class A(torch.nn.Module):
    def __init__(s): super().__init__(); s.register_buffer("w", torch.randn(KA, NA, dtype=DT))
    def forward(s, x): return torch.mm(x, s.w)


class B(torch.nn.Module):
    def __init__(s): super().__init__(); s.register_buffer("w", torch.randn(NA, NB, dtype=DT))
    def forward(s, x): return torch.mm(x, s.w)


class Chain(torch.nn.Module):  # A→B (연쇄, B가 A출력 받음)
    def __init__(s):
        super().__init__(); s.register_buffer("wa", torch.randn(KA, NA, dtype=DT)); s.register_buffer("wb", torch.randn(NA, NB, dtype=DT))
    def forward(s, x): return torch.mm(torch.mm(x, s.wa), s.wb)


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
    uA = host_us(A(), torch.randn(M, KA, dtype=DT))
    uB = host_us(B(), torch.randn(M, NA, dtype=DT))
    w2 = host_us(Chain(), torch.randn(M, KA, dtype=DT))
    ov = uA + uB - w2
    frac = ov / (uA + uB) if (uA + uB) else 0
    computeA = 2 * M * KA * NA
    weightB = NA * NB * 2
    print(f"PAIR {M} {KA} {NA} {NB} {uA:.1f} {uB:.1f} {w2:.1f} {ov:.1f} {frac:.3f} {computeA} {weightB}", flush=True)
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
