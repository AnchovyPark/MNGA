#!/usr/bin/env python3
"""개별 matmul 합 vs 번들(supertask 단위) — 예전 "합 > 전체" 문제의 결정적 규명 (2026.3.0).

예전 문제: 격리 matmul들을 하나씩 따로 컴파일해 latency를 더하니 in-model 한 레이어보다 훨씬 컸다(부분합>전체).
가설: 따로 컴파일하면 op끼리 overlap을 잃어 각자 full weight-load를 냄 → 합이 뻥튀기. in-model 커널은
supertask(융합 번들)로 도니, 격리도 번들로 묶으면 in-model과 맞아야 한다.

이 스크립트(한 프로세스): 같은 config로
  (A) 7개 op 각각 따로 compile+run 해서 device time 합산  (= 예전 방식)
  (B) 7개를 한 그래프(번들)로 compile+run                 (= supertask 단위, 새 방식)
둘을 나란히 출력. A >> B ≈ in-model 이면 가설 확정.

사용: /home/furiosa/venv3030/bin/python individual_vs_bundle_30.py <seq> [hint]
  hint = ForLlmModelComputeBound(기본) | NoConstraint
"""
import os, sys, time, statistics as st

import torch
import furiosa.torch as ft
ft.set_fusion(8)
from furiosa.torch._C.config.compiler import CompilerConfig, TacticHintConfig

S = int(sys.argv[1])
HINT = sys.argv[2] if len(sys.argv) > 2 else "ForLlmModelComputeBound"
DEV = "furiosa:0"
D, INTER, NH, HD, KV = 2048, 8192, 32, 64, 8
DT = torch.bfloat16
ITERS, WARMUP = 30, 8

# op -> (M, K, N): (S,K)@(K,N)
OPS = {
    "q":    (S, D, NH*HD), "k": (S, D, KV*HD), "v": (S, D, KV*HD),
    "o":    (S, NH*HD, D),
    "gate": (S, D, INTER), "up": (S, D, INTER), "down": (S, INTER, D),
}


class MM(torch.nn.Module):
    def __init__(self, K, N):
        super().__init__(); self.register_buffer("w", torch.randn(K, N, dtype=DT))
    def forward(self, x):
        return torch.mm(x, self.w)


class Bundle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for n, sh in [("wq",(D,NH*HD)),("wk",(D,KV*HD)),("wv",(D,KV*HD)),("wo",(NH*HD,D)),
                      ("wg",(D,INTER)),("wu",(D,INTER)),("wd",(INTER,D))]:
            self.register_buffer(n, torch.randn(*sh, dtype=DT))
    def forward(self, x):
        q = torch.mm(x, self.wq); k = torch.mm(x, self.wk); v = torch.mm(x, self.wv)
        o = torch.mm(q, self.wo); x2 = x + o
        g = torch.mm(x2, self.wg); u = torch.mm(x2, self.wu)
        d = torch.mm(torch.nn.functional.silu(g) * u, self.wd)
        return (k, v, d)


def cfg():
    kw = dict(tactic_hint=getattr(TacticHintConfig, HINT))
    if HINT != "NoConstraint":
        kw.update(scheduler_beam_search=True, use_attention_kernel=True)
    return CompilerConfig(**kw)


def time_module(mod, x):
    cm = ft.CompileModule.from_module(mod, (x,), compiler_config=cfg()).to(DEV)
    xd = x.to(DEV)
    stream = ft.current_stream(DEV)
    for _ in range(WARMUP):
        cm(xd)
    stream.synchronize()
    reps = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(ITERS):
            cm(xd)
        stream.synchronize()
        reps.append((time.perf_counter() - t0) / ITERS * 1e6)
    return st.median(reps)


def main():
    print(f"=== 개별합 vs 번들 : seq={S} hint={HINT} TP={ft.get_fusion().num_fusion} ===", flush=True)
    # (A) 개별 op 따로
    print("[A] 개별 op (따로 compile+run):", flush=True)
    total = 0.0
    for name, (M, K, N) in OPS.items():
        t = time_module(MM(K, N), torch.randn(M, K, dtype=DT))
        total += t
        print(f"    {name:>5}: {t:8.1f}us", flush=True)
    print(f"    [개별 합] {total:8.1f}us", flush=True)
    # (B) 번들
    tb = time_module(Bundle(), torch.randn(S, D, dtype=DT))
    print(f"[B] 번들(supertask 단위): {tb:8.1f}us", flush=True)
    print(f"\n>>> 개별합/번들 = {total/tb:.2f}x  (개별합이 이만큼 뻥튀기)", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
