#!/usr/bin/env python3
"""N개 Tokenwise 레이어(attention 제외) 이어붙여 측정 — cross-layer weight 겹침 특성화.

cross-layer 겹침의 핵심 = matmul weight 로드가 레이어 간 파이프라인 되는 것 (attention은 weight 없음).
attention의 S×S 텐서가 2층+ 컴파일서 memory explosion → attention 빼고 Tokenwise만 chain.
각 층: rmsnorm→q,k,v→o(가짜attn=q,k,v 결합)→residual→rmsnorm→gate,up→down→residual. 별도 weight/층.
q,k,v를 o 입력에 섞어 DCE 방지(모든 weight 로드 유지).

marginal L(N)-L(N-1)이 층 추가하며 줄어들면 = 겹침. saturate 지점 = 파이프라인 깊이.

사용: TUC_PROFILE_LEVEL=info RUST_LOG=info,span::tuc=info \
      /home/furiosa/venv3030/bin/python nlayer_tw_30.py <seq> <nlayers>
출력: CYC <nlayers> <seq> <cyc>
"""
import os, sys, json

import torch
import furiosa.torch as ft
ft.set_fusion(8)
from furiosa.torch._C.config.compiler import CompilerConfig, TacticHintConfig
from furiosa.torch.profiler import Profiler

S = int(sys.argv[1]) if len(sys.argv) > 1 else 512
NL = int(sys.argv[2]) if len(sys.argv) > 2 else 1
DEV = "furiosa:0"
D, INTER, NH, HD, KV = 2048, 8192, 32, 64, 8
DT = torch.bfloat16
EPS = 1e-5
SDIR = "/tmp/claude-1002/-home-furiosa------pjh-rngd/8464c59e-3432-4d59-9ccc-d34e43519088/scratchpad"


def cfg():
    return CompilerConfig(tactic_hint=TacticHintConfig.ForLlmModelComputeBound,
                          scheduler_beam_search=True, use_attention_kernel=True)


def rmsnorm(x, w):
    v = (x.to(torch.float32) ** 2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(v.to(DT) + EPS)) * w


class NLayersTW(torch.nn.Module):
    def __init__(self, n):
        super().__init__(); self.n = n
        for L in range(n):
            for nm, sh in [("wq",(D,NH*HD)),("wk",(D,KV*HD)),("wv",(D,KV*HD)),("wo",(NH*HD,D)),
                           ("wg",(D,INTER)),("wu",(D,INTER)),("wd",(INTER,D))]:
                self.register_buffer(f"{nm}_{L}", torch.randn(*sh, dtype=DT))
            self.register_buffer(f"n1_{L}", torch.randn(D, dtype=DT))
            self.register_buffer(f"n2_{L}", torch.randn(D, dtype=DT))

    def _layer(self, x, L):
        g = lambda nm: getattr(self, f"{nm}_{L}")
        h = rmsnorm(x, g("n1"))
        q = torch.mm(h, g("wq"))               # (S, NH*HD=2048)
        k = torch.mm(h, g("wk")); v = torch.mm(h, g("wv"))   # (S, 512)
        # 가짜 attention output: q + cat(k,v,k,v) → q,k,v 모두 사용(DCE 방지), S×S 없음
        attn_out = q + torch.cat([k, v, k, v], dim=-1)       # (S, 2048)
        o = torch.mm(attn_out, g("wo"))
        x2 = x + o
        h2 = rmsnorm(x2, g("n2"))
        mlp = torch.mm(torch.nn.functional.silu(torch.mm(h2, g("wg"))) * torch.mm(h2, g("wu")), g("wd"))
        return x2 + mlp

    def forward(self, x):
        for L in range(self.n):
            x = self._layer(x, L)
        return x


def dev_cyc(mod, x):
    # 큰 그래프는 device-cycle 프로파일러가 캡처 miss → host stream-sync 배치 타이밍(robust).
    import time, statistics as st
    cm = ft.CompileModule.from_module(mod, (x,), compiler_config=cfg()).to(DEV)
    xd = x.to(DEV)
    stream = ft.current_stream(DEV)
    for _ in range(10):
        cm(xd)
    stream.synchronize()
    reps = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(30):
            cm(xd)
        stream.synchronize()
        reps.append((time.perf_counter() - t0) / 30 * 1e6)  # per-call us
    return round(st.median(reps), 1)


def main():
    c = dev_cyc(NLayersTW(NL), torch.randn(S, D, dtype=DT))
    print(f"CYC {NL} {S} {c}", flush=True)
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
