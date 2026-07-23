#!/usr/bin/env python3
"""Attention supertask n-gram 분해 (2026.3.0, device-cycle). Tokenwise n-gram과 동일 방식.

Attention = [matmul_qk, softmax, matmul_score_v] 3-stage chain (GQA). Tokenwise를 7 matmul로 쪼개
telescoping 했듯, attention도 3 stage로 쪼개 n-gram window 측정 + telescoping → full attention 예측.
  U(stage), W2(연속쌍), full(3개).  tele2 = W2(qk,sm) + (W2(sm,sv) − U(sm)).

사용: TUC_PROFILE_LEVEL=info RUST_LOG=info,span::tuc=info \
      /home/furiosa/venv3030/bin/python attention_ngram_30.py <seq>
"""
import os, sys, json, math

import torch
import furiosa.torch as ft
ft.set_fusion(8)
from furiosa.torch._C.config.compiler import CompilerConfig, TacticHintConfig
from furiosa.torch.profiler import Profiler

S = int(sys.argv[1]) if len(sys.argv) > 1 else 512
DEV = "furiosa:0"
NH, KV, HD = 32, 8, 64
GROUP = NH // KV
SCALE = 1.0 / math.sqrt(HD)
DT = torch.bfloat16
NEG = -30000.0
SDIR = "/tmp/claude-1002/-home-furiosa------pjh-rngd/8464c59e-3432-4d59-9ccc-d34e43519088/scratchpad"


def cfg():
    # n-gram 분해엔 use_attention_kernel=False (flash 융합 끄고 qk/softmax/sv를 분리 op으로).
    # True면 부분 window(sv 없는 것)가 flash 패턴 불일치로 컴파일 실패.
    return CompilerConfig(tactic_hint=TacticHintConfig.ForLlmModelComputeBound,
                          scheduler_beam_search=True, use_attention_kernel=False)


class AttnWin(torch.nn.Module):
    def __init__(self, stages):
        super().__init__(); self.stages = stages
        m = torch.triu(torch.full((S, S), NEG, dtype=DT), diagonal=1)
        self.register_buffer("mask", m)

    def _qk(self, q, k):
        q = q.view(S, NH, HD).transpose(0, 1)
        k = k.view(S, KV, HD).transpose(0, 1).unsqueeze(1).expand(KV, GROUP, S, HD).reshape(NH, S, HD)
        return torch.bmm(q, k.transpose(1, 2)) * SCALE + self.mask

    def _sm(self, scores):
        return torch.softmax(scores, dim=-1)

    def _sv(self, attn, v):
        v = v.view(S, KV, HD).transpose(0, 1).unsqueeze(1).expand(KV, GROUP, S, HD).reshape(NH, S, HD)
        return torch.bmm(attn, v).transpose(0, 1).reshape(S, NH * HD)

    def forward(self, *ins):
        st = self.stages
        if st[0] == "qk":
            x = self._qk(ins[0], ins[1])
            if "sm" in st: x = self._sm(x)
            if "sv" in st: x = self._sv(x, ins[2])
            return x
        if st[0] == "sm":
            x = self._sm(ins[0])
            if "sv" in st: x = self._sv(x, ins[1])
            return x
        return self._sv(ins[0], ins[1])   # ["sv"]


def win_inputs(stages):
    q = torch.randn(S, NH*HD, dtype=DT); k = torch.randn(S, KV*HD, dtype=DT); v = torch.randn(S, KV*HD, dtype=DT)
    scores = torch.randn(NH, S, S, dtype=DT); attn = torch.randn(NH, S, S, dtype=DT)
    if stages[0] == "qk":
        return (q, k, v) if "sv" in stages else (q, k)
    if stages[0] == "sm":
        return (scores, v) if "sv" in stages else (scores,)
    return (attn, v)   # ["sv"]


def dev_cyc(stages):
    mod = AttnWin(stages); xs = win_inputs(stages)
    cm = ft.CompileModule.from_module(mod, xs, compiler_config=cfg()).to(DEV)
    xds = [x.to(DEV) for x in xs]
    for _ in range(8):
        cm(*xds)
    pp = f"{SDIR}/an_{'_'.join(stages)}_{S}.json"
    for _ in range(4):
        with Profiler(profile_path=pp):
            for _ in range(15):
                cm(*xds)
        d = json.load(open(pp)); seen = {}
        for e in d:
            if e.get("name") != "Task":
                continue
            a = e["args"]; seen[(a["begin_cycle"], a["end_cycle"], a.get("cluster_index"))] = int(a["cycle_actual"])
        v = sorted(seen.values())
        if v:
            return min(v)
    return None


def P(*a): print(*a, flush=True)


def main():
    P(f"=== Attention n-gram : seq={S} TP=8 (device-cycle) ===")
    U = {s: dev_cyc([s]) for s in ["qk", "sm", "sv"]}
    W2 = {"qk_sm": dev_cyc(["qk", "sm"]), "sm_sv": dev_cyc(["sm", "sv"])}
    full = dev_cyc(["qk", "sm", "sv"])
    P(f"[unigram] qk={U['qk']}  sm={U['sm']}  sv={U['sv']}")
    P(f"[2-gram]  qk_sm={W2['qk_sm']}  sm_sv={W2['sm_sv']}")
    uni = U['qk'] + U['sm'] + U['sv']
    tele2 = W2['qk_sm'] + (W2['sm_sv'] - U['sm'])
    P(f"\n{'방식':14s} {'예측(cyc)':>10} {'vs full':>8}")
    P(f"{'unigram 합':14s} {uni:10d} {uni/full:8.2f}")
    P(f"{'2-gram tele':14s} {tele2:10d} {tele2/full:8.2f}")
    P(f"{'full(측정)':14s} {full:10d} {'1.00':>8}")
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
