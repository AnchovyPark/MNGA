#!/usr/bin/env python3
"""n-gram 예측 — device-cycle 정밀 버전 (2026.3.0).

host wall-clock의 op당 ~40us 바닥을 제거하기 위해 프로파일러 device cycle(cycle_actual)로 측정.
in-model GT도 cycle 단위이므로 cycle 대 cycle 직접 비교(클럭 변환 불필요).

- 재료 = 각 op-identical 커널의 device cycle (min across warmup 뺀 반복).
- ground truth = 실제 in-model Tokenwise supertask cycle (128:368000, 512:827000, 1024:1429000).
- 조합법칙 후보: naive_mm(matmul 합), naive_all(전부 합).

사용: TUC_PROFILE_LEVEL=info RUST_LOG=info,span::tuc=info \
      /home/furiosa/venv3030/bin/python dev_cycle_predict_30.py <seq>
"""
import os, sys, json, statistics as st

import torch
import furiosa.torch as ft
ft.set_fusion(8)
from furiosa.torch._C.config.compiler import CompilerConfig, TacticHintConfig
from furiosa.torch.profiler import Profiler

S = int(sys.argv[1]) if len(sys.argv) > 1 else 512
DEV = "furiosa:0"
D, INTER, NH, HD, KV = 2048, 8192, 32, 64, 8
DT = torch.bfloat16
EPS = 1e-5
SDIR = "/tmp/claude-1002/-home-furiosa------pjh-rngd/8464c59e-3432-4d59-9ccc-d34e43519088/scratchpad"
INMODEL_CYC = {128: 368000, 512: 827000, 1024: 1429000}


def cfg():
    return CompilerConfig(tactic_hint=TacticHintConfig.ForLlmModelComputeBound,
                          scheduler_beam_search=True, use_attention_kernel=True)


def dev_cyc(mod, xs, tag):
    cm = ft.CompileModule.from_module(mod, xs, compiler_config=cfg()).to(DEV)
    xds = [x.to(DEV) for x in xs]
    for _ in range(8):
        cm(*xds)
    pp = f"{SDIR}/dcp_{tag}_{S}.json"
    with Profiler(profile_path=pp):
        for _ in range(15):
            cm(*xds)
    d = json.load(open(pp)); seen = {}
    for e in d:
        if e.get("name") != "Task":
            continue
        a = e["args"]
        seen[(a["begin_cycle"], a["end_cycle"], a.get("cluster_index"))] = int(a["cycle_actual"])
    vals = sorted(seen.values())
    return min(vals) if vals else 0


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
        s.register_buffer("cos", torch.randn(1, 1, HD, dtype=DT)); s.register_buffer("sin", torch.randn(1, 1, HD, dtype=DT))
    def forward(s, x):
        t = x.view(S, s.h, HD); x1 = t[..., :HD//2]; x2 = t[..., HD//2:]
        return (t * s.cos + torch.cat((-x2, x1), -1) * s.sin).reshape(S, s.h*HD)


class Resid(torch.nn.Module):
    def forward(s, a, b): return a + b


def P(*a): print(*a, flush=True)


def main():
    r = lambda *sh: torch.randn(*sh, dtype=DT)
    mm = {
        "q": dev_cyc(MM(D, NH*HD), (r(S, D),), "q"),
        "k": dev_cyc(MM(D, KV*HD), (r(S, D),), "k"),
        "v": dev_cyc(MM(D, KV*HD), (r(S, D),), "v"),
        "o": dev_cyc(MM(NH*HD, D), (r(S, NH*HD),), "o"),
        "gate": dev_cyc(MM(D, INTER), (r(S, D),), "gate"),
        "up": dev_cyc(MM(D, INTER), (r(S, D),), "up"),
        "down": dev_cyc(MM(INTER, D), (r(S, INTER),), "down"),
    }
    other = {
        "rmsnorm1": dev_cyc(RMSNorm(), (r(S, D),), "n1"),
        "rmsnorm2": dev_cyc(RMSNorm(), (r(S, D),), "n2"),
        "rope_q": dev_cyc(RoPE(NH), (r(S, NH*HD),), "rq"),
        "rope_k": dev_cyc(RoPE(KV), (r(S, KV*HD),), "rk"),
        "resid1": dev_cyc(Resid(), (r(S, D), r(S, D)), "r1"),
        "resid2": dev_cyc(Resid(), (r(S, D), r(S, D)), "r2"),
    }
    P(f"=== device-cycle 재료 → 예측 vs in-model : seq={S} TP=8 (단위 cyc) ===")
    P("[matmul]  " + "  ".join(f"{k}={v}" for k, v in mm.items()))
    P("[비mm]    " + "  ".join(f"{k}={v}" for k, v in other.items()))
    sm = sum(mm.values()); sa = sm + sum(other.values()); gt = INMODEL_CYC.get(S)
    P(f"\nnaive_mm  = {sm:8d} cyc" + (f"  → GT {gt} 대비 {sm/gt:.2f}x" if gt else ""))
    P(f"naive_all = {sa:8d} cyc" + (f"  → GT {gt} 대비 {sa/gt:.2f}x" if gt else ""))
    if gt: P(f"ground truth(in-model Tokenwise) = {gt} cyc")
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
