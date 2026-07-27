#!/usr/bin/env python3
"""단일 matmul 커널 device-cycle 측정 (2026.3.0). kernel-based ML 데이터용.

atom = generic matmul (M,K)@(K,N). op 이름 무관, shape로만. → shape 공간 샘플링.
사용: TUC_PROFILE_LEVEL=info RUST_LOG=info,span::tuc=info \
      /home/furiosa/venv3030/bin/python kernel_measure_30.py <M> <K> <N>
출력: CYC <M> <K> <N> <cyc> <flops> <wbytes>
"""
import os, sys, json

import torch
import furiosa.torch as ft
ft.set_fusion(8)
from furiosa.torch._C.config.compiler import CompilerConfig, TacticHintConfig
from furiosa.torch.profiler import Profiler

M, K, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
DEV = "furiosa:0"
DT = torch.bfloat16
SDIR = "/tmp/claude-1002/-home-furiosa------pjh-rngd/8464c59e-3432-4d59-9ccc-d34e43519088/scratchpad"


def cfg():
    return CompilerConfig(tactic_hint=TacticHintConfig.ForLlmModelComputeBound,
                          scheduler_beam_search=True, use_attention_kernel=True)


class MM(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.register_buffer("w", torch.randn(K, N, dtype=DT))
    def forward(self, x):
        return torch.mm(x, self.w)


def dev_cyc():
    mod = MM(); x = torch.randn(M, K, dtype=DT)
    cm = ft.CompileModule.from_module(mod, (x,), compiler_config=cfg()).to(DEV)
    xd = x.to(DEV)
    for _ in range(8):
        cm(xd)
    pp = f"{SDIR}/km_{M}_{K}_{N}.json"
    for _ in range(4):
        with Profiler(profile_path=pp):
            for _ in range(15):
                cm(xd)
        d = json.load(open(pp)); seen = {}
        for e in d:
            if e.get("name") != "Task":
                continue
            a = e["args"]; seen[(a["begin_cycle"], a["end_cycle"], a.get("cluster_index"))] = int(a["cycle_actual"])
        v = sorted(seen.values())
        if v:
            return min(v)
    return None


def main():
    c = dev_cyc()
    flops = 2 * M * K * N
    wbytes = K * N * 2
    print(f"CYC {M} {K} {N} {c} {flops} {wbytes}", flush=True)
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
