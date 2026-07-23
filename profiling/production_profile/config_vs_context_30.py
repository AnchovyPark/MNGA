#!/usr/bin/env python3
"""config vs context 분리 실험 (컴파일러 2026.3.0 전용). 한 프로세스=한 (target,seq,hint) 측정.

핵심 질문: "격리 커널 == in-model 커널? latency 같나?"
격차 = (a) config 성분(격리 저품질) + (b) context 성분(cross-op fusion 상실).
2026.3.0은 World-1 compile()이 production tactic(ForLlmModelComputeBound)+beam+attn을 받고
Runnable을 TpModule/CompileModule로 device 실행까지 함 → 처음으로 격리를 near-production 품질로
컴파일+실행 가능. 이걸 in-model 실측과 비교.

주의(2026.3.0): set_fusion(8)은 import 직후 맨 먼저. aten::matmul 미지원→2D aten::mm. 입력을 .to(device).
teardown 크래시(device 정리) 회피 위해 os._exit(0).

사용: /home/furiosa/venv3030/bin/python config_vs_context_30.py <seq> <target> <hint>
  target = oproj | tokenwise ;  hint = NoConstraint | ForLlmModelComputeBound
출력: RESULT <hint> <target> <seq> <median_us> <min_us>
"""
import os, sys, time, statistics as st

import torch
import furiosa.torch as ft
ft.set_fusion(8)  # ★맨 먼저★
from furiosa.torch._C.config.compiler import CompilerConfig, TacticHintConfig

S = int(sys.argv[1])
TARGET = sys.argv[2]
HINT = sys.argv[3]
DEV = "furiosa:0"
D, INTER, NH, HD, KV = 2048, 8192, 32, 64, 8
DT = torch.bfloat16
ITERS, WARMUP = 40, 10


class OProj(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.register_buffer("w", torch.randn(NH*HD, D, dtype=DT))
    def forward(self, x):
        return torch.mm(x, self.w)


class Tokenwise(torch.nn.Module):
    """full Tokenwise 번들: q,k,v,o proj + gate/up/down MLP. in-model Tokenwise supertask와 같은 op 구성."""
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


def main():
    mod, x = (OProj(), torch.randn(S, NH*HD, dtype=DT)) if TARGET == "oproj" \
        else (Tokenwise(), torch.randn(S, D, dtype=DT))
    kw = dict(tactic_hint=getattr(TacticHintConfig, HINT))
    if HINT != "NoConstraint":
        kw.update(scheduler_beam_search=True, use_attention_kernel=True)
    cfg = CompilerConfig(**kw)
    cm = ft.CompileModule.from_module(mod, (x,), compiler_config=cfg).to(DEV)
    xd = x.to(DEV)
    stream = ft.current_stream(DEV)
    for _ in range(WARMUP):
        cm(xd)
    stream.synchronize()
    # ★배치 타이밍★: ITERS 호출을 큐잉 후 마지막에 stream.synchronize() 한 번 → 총 wall÷ITERS = device time
    reps = []
    for _ in range(5):  # 5블록 median
        t0 = time.perf_counter()
        for _ in range(ITERS):
            cm(xd)
        stream.synchronize()
        reps.append((time.perf_counter() - t0) / ITERS * 1e6)  # per-call us
    print(f"RESULT {HINT} {TARGET} {S} {st.median(reps):.1f} {min(reps):.1f}", flush=True)
    sys.stdout.flush()
    os._exit(0)  # teardown 크래시 회피


if __name__ == "__main__":
    main()
