#!/usr/bin/env python3
"""n-gram 조합 법칙 실측 (2026.3.0). 예전 composition_sweep(컴파일러 예상치)의 실행-latency 버전.

op을 q→+k→+v→+o→+gate→+up→+down 누적하며 각 prefix 번들을 RNGD에서 실측.
  marginal(N) = L(N) - L(N-1)         (op N을 추가했을 때 실제로 늘어난 시간)
  absorbed    = standalone(N) - marginal(N)   (그 op 홀로 시간 중 overlap에 숨은 양)
  흡수율      = absorbed / standalone
→ "op을 묶으면 컴파일러가 얼마나 겹쳐서 숨기나"의 실측 법칙. 예전엔 예상치로만 봤음.

사용: /home/furiosa/venv3030/bin/python ngram_composition_30.py <seq> [hint]
"""
import os, sys, time, statistics as st, csv

import torch
import furiosa.torch as ft
ft.set_fusion(8)
from furiosa.torch._C.config.compiler import CompilerConfig, TacticHintConfig

S = int(sys.argv[1]) if len(sys.argv) > 1 else 512
HINT = sys.argv[2] if len(sys.argv) > 2 else "ForLlmModelComputeBound"
DEV = "furiosa:0"
D, INTER, NH, HD, KV = 2048, 8192, 32, 64, 8
DT = torch.bfloat16
ITERS, WARMUP = 30, 8
OPS = ["q", "k", "v", "o", "gate", "up", "down"]
# standalone op -> (M, K, N)
SHAPE = {"q": (S, D, NH*HD), "k": (S, D, KV*HD), "v": (S, D, KV*HD), "o": (S, NH*HD, D),
         "gate": (S, D, INTER), "up": (S, D, INTER), "down": (S, INTER, D)}


def cfg():
    kw = dict(tactic_hint=getattr(TacticHintConfig, HINT))
    if HINT != "NoConstraint":
        kw.update(scheduler_beam_search=True, use_attention_kernel=True)
    return CompilerConfig(**kw)


class One(torch.nn.Module):
    def __init__(self, K, N):
        super().__init__(); self.register_buffer("w", torch.randn(K, N, dtype=DT))
    def forward(self, x):
        return torch.mm(x, self.w)


class Prefix(torch.nn.Module):
    """첫 N개 op만 계산 (모두 반환해 DCE 방지). o는 q에, gate/up/down은 x2=x+o에 의존."""
    def __init__(self, N):
        super().__init__(); self.N = N
        for n, sh in [("wq",(D,NH*HD)),("wk",(D,KV*HD)),("wv",(D,KV*HD)),("wo",(NH*HD,D)),
                      ("wg",(D,INTER)),("wu",(D,INTER)),("wd",(INTER,D))]:
            self.register_buffer(n, torch.randn(*sh, dtype=DT))
    def forward(self, x):
        N = self.N; outs = [torch.mm(x, self.wq)]
        if N >= 2: outs.append(torch.mm(x, self.wk))
        if N >= 3: outs.append(torch.mm(x, self.wv))
        if N >= 4:
            o = torch.mm(outs[0], self.wo); outs.append(o)
        if N >= 5:
            x2 = x + outs[3]; g = torch.mm(x2, self.wg); outs.append(g)
        if N >= 6: outs.append(torch.mm(x2, self.wu))
        if N >= 7: outs.append(torch.mm(torch.nn.functional.silu(outs[4]) * outs[5], self.wd))
        return tuple(outs)


def time_mod(mod, x):
    cm = ft.CompileModule.from_module(mod, (x,), compiler_config=cfg()).to(DEV)
    xd = x.to(DEV); stream = ft.current_stream(DEV)
    for _ in range(WARMUP): cm(xd)
    stream.synchronize()
    reps = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(ITERS): cm(xd)
        stream.synchronize(); reps.append((time.perf_counter() - t0) / ITERS * 1e6)
    return st.median(reps)


def P(*a): print(*a, flush=True)


def main():
    P(f"=== n-gram 조합 실측 : seq={S} hint={HINT} TP=8 ===")
    x = torch.randn(S, D, dtype=DT)
    # 1) 누적 prefix
    L = {}
    for N in range(1, 8):
        L[N] = time_mod(Prefix(N), x)
    # 2) standalone
    single = {op: time_mod(One(SHAPE[op][1], SHAPE[op][2]), torch.randn(SHAPE[op][0], SHAPE[op][1], dtype=DT))
              for op in OPS}
    # 3) 흡수 분석
    P(f"\n{'추가op':>6} {'prefix L':>9} {'marginal':>9} {'standalone':>10} {'흡수':>8} {'흡수율':>7}")
    prev = 0.0; rows = []
    for N in range(1, 8):
        op = OPS[N-1]; marg = L[N] - prev; prev = L[N]
        so = single[op]; ab = so - marg; rate = ab/so*100 if so else 0
        P(f"{op:>6} {L[N]:9.1f} {marg:9.1f} {so:10.1f} {ab:8.1f} {rate:6.0f}%")
        rows.append(dict(seq=S, op=op, prefix_L=round(L[N],1), marginal=round(marg,1),
                         standalone=round(so,1), absorbed=round(ab,1), absorb_pct=round(rate,1)))
    naive = sum(single.values())
    P(f"\n개별합(naive)={naive:.1f}us  전체번들 L(7)={L[7]:.1f}us  개별합/번들={naive/L[7]:.2f}x")
    out = f"profiling/production_profile/ngram_composition_s{S}_{HINT}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    P(f"저장: {out}")
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
