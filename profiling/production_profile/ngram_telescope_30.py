#!/usr/bin/env python3
"""진짜 n-gram 방식 검증 (2026.3.0, device-cycle).

지금까지는 unigram(op 하나씩) 합이었음. 이건 진짜 n-gram:
op을 연속 2~3개 묶음(window)으로 측정 → telescoping으로 전체 예측.
목표: n-gram이 unigram 합보다 실제 전체 번들(및 in-model)을 잘 맞추나?

chain 순서: [q,k,v,o,gate,up,down] (Tokenwise forward 순).
- unigram U(op) = 그 op 혼자 device-cycle
- 2-gram W2(i)  = 연속 2개(op_i,op_{i+1}) 묶음 device-cycle  (그 안에 국소 overlap 포함됨)
- 3-gram W3(i)  = 연속 3개
- full          = 7개 전체 번들 (예측 대상 = 같은버전 clean truth)
telescoping:
  unigram_sum = ΣU
  tele2 = W2(0) + Σ_{i≥1}(W2(i) − U(op_i))         # 공유 op 빼며 이어붙임
  tele3 = W3(0) + Σ_{i≥1}(W3(i) − W2'(i))          # 공유 2-gram 빼며

사용: TUC_PROFILE_LEVEL=info RUST_LOG=info,span::tuc=info \
      /home/furiosa/venv3030/bin/python ngram_telescope_30.py <seq>
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
SDIR = "/tmp/claude-1002/-home-furiosa------pjh-rngd/8464c59e-3432-4d59-9ccc-d34e43519088/scratchpad"
INMODEL = {128: 368000, 512: 827000, 1024: 1429000}

OPS = ["q", "k", "v", "o", "gate", "up", "down"]
# op -> (input_key, K, N).  input_key: x(D) / attn(NH*HD) / interin(INTER)
SPEC = {"q": ("x", D, NH*HD), "k": ("x", D, KV*HD), "v": ("x", D, KV*HD),
        "o": ("attn", NH*HD, D), "gate": ("x", D, INTER), "up": ("x", D, INTER),
        "down": ("interin", INTER, D)}


def cfg():
    return CompilerConfig(tactic_hint=TacticHintConfig.ForLlmModelComputeBound,
                          scheduler_beam_search=True, use_attention_kernel=True)


class Window(torch.nn.Module):
    """subset(op 리스트)의 matmul들을 한 그래프로. 각 op은 제 입력(x/attn/interin)에서."""
    def __init__(self, subset):
        super().__init__(); self.subset = list(subset)
        for op in self.subset:
            _, K, N = SPEC[op]
            self.register_buffer(f"w_{op}", torch.randn(K, N, dtype=DT))

    def forward(self, x, attn, interin):
        src = {"x": x, "attn": attn, "interin": interin}
        outs = []
        for op in self.subset:
            ik, _, _ = SPEC[op]
            outs.append(torch.mm(src[ik], getattr(self, f"w_{op}")))
        return tuple(outs)


def dev_cyc(subset):
    mod = Window(subset)
    xs = (torch.randn(S, D, dtype=DT), torch.randn(S, NH*HD, dtype=DT), torch.randn(S, INTER, dtype=DT))
    cm = ft.CompileModule.from_module(mod, xs, compiler_config=cfg()).to(DEV)
    xds = [x.to(DEV) for x in xs]
    for _ in range(8):
        cm(*xds)
    tag = "_".join(subset)
    pp = f"{SDIR}/ng_{tag}_{S}.json"
    with Profiler(profile_path=pp):
        for _ in range(15):
            cm(*xds)
    d = json.load(open(pp)); seen = {}
    for e in d:
        if e.get("name") != "Task":
            continue
        a = e["args"]; seen[(a["begin_cycle"], a["end_cycle"], a.get("cluster_index"))] = int(a["cycle_actual"])
    v = sorted(seen.values())
    return min(v) if v else 0


def P(*a): print(*a, flush=True)


def main():
    P(f"=== 진짜 n-gram telescoping : seq={S} TP=8 (device-cycle) ===")
    U = {op: dev_cyc([op]) for op in OPS}
    W2 = [dev_cyc(OPS[i:i+2]) for i in range(len(OPS)-1)]      # 6개
    W3 = [dev_cyc(OPS[i:i+3]) for i in range(len(OPS)-2)]      # 5개
    full = dev_cyc(OPS)
    gt = INMODEL.get(S)

    P("[unigram U]  " + "  ".join(f"{o}={U[o]}" for o in OPS))
    P("[2-gram W2]  " + "  ".join(f"{OPS[i]}{OPS[i+1]}={W2[i]}" for i in range(len(W2))))
    P("[3-gram W3]  " + "  ".join(f"{''.join(OPS[i:i+3])}={W3[i]}" for i in range(len(W3))))

    uni = sum(U.values())
    tele2 = W2[0] + sum(W2[i] - U[OPS[i]] for i in range(1, len(W2)))
    # 3-gram telescope: 공유되는 2-gram(내부 겹침) 빼기. W3(i),W3(i+1) 공유 = ops[i+1:i+3] ≈ W2[i+1]
    tele3 = W3[0] + sum(W3[i] - W2[i] for i in range(1, len(W3)))

    P(f"\n{'방식':16s} {'예측(cyc)':>10} {'vs full':>8}" + (f" {'vs in-model':>11}" if gt else ""))
    def row(name, val):
        s = f"{name:16s} {val:10d} {val/full:8.2f}"
        if gt: s += f" {val/gt:11.2f}"
        P(s)
    row("unigram 합", uni)
    row("2-gram tele", tele2)
    row("3-gram tele", tele3)
    P(f"{'full(측정)':16s} {full:10d} {'1.00':>8}" + (f" {full/gt:11.2f}" if gt else ""))
    if gt: P(f"{'in-model GT':16s} {gt:10d}")
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
