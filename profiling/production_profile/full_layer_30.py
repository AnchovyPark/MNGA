#!/usr/bin/env python3
"""연속 N개 레이어 격리 측정 (2026.3.0, device-cycle) — cross-layer overlap 측정용.

한 레이어(op-identical): rmsnorm→QKV(+rope)→flash attention→O→residual→rmsnorm→MLP(silu)→residual.
N층을 실제 연결로 chain(각 층 별도 weight = cross-layer weight-load 반영). flash attention=use_attention_kernel=True.

레이어-레벨 telescoping: T_full = 16·U1 − 15·(2U1 − W2), U1=1층, W2=2층.

사용: TUC_PROFILE_LEVEL=info RUST_LOG=info,span::tuc=info \
      /home/furiosa/venv3030/bin/python full_layer_30.py <seq> <nlayers>
출력: CYC <nlayers> <seq> <cyc>
"""
import os, sys, json, math

import torch
import furiosa.torch as ft
ft.set_fusion(8)
from furiosa.torch._C.config.compiler import CompilerConfig, TacticHintConfig
from furiosa.torch.profiler import Profiler

S = int(sys.argv[1]) if len(sys.argv) > 1 else 512
NL = int(sys.argv[2]) if len(sys.argv) > 2 else 1
DEV = "furiosa:0"
D, INTER, NH, HD, KV = 2048, 8192, 32, 64, 8
GROUP = NH // KV
SCALE = 1.0 / math.sqrt(HD)
DT = torch.bfloat16
EPS = 1e-5
NEG = -30000.0
SDIR = "/tmp/claude-1002/-home-furiosa------pjh-rngd/8464c59e-3432-4d59-9ccc-d34e43519088/scratchpad"


def cfg():
    return CompilerConfig(tactic_hint=TacticHintConfig.ForLlmModelComputeBound,
                          scheduler_beam_search=True, use_attention_kernel=True)


def rmsnorm(x, w):
    v = (x.to(torch.float32) ** 2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(v.to(DT) + EPS)) * w


class NLayers(torch.nn.Module):
    def __init__(self, n):
        super().__init__(); self.n = n
        for L in range(n):
            for nm, sh in [("wq",(D,NH*HD)),("wk",(D,KV*HD)),("wv",(D,KV*HD)),("wo",(NH*HD,D)),
                           ("wg",(D,INTER)),("wu",(D,INTER)),("wd",(INTER,D))]:
                self.register_buffer(f"{nm}_{L}", torch.randn(*sh, dtype=DT))
            self.register_buffer(f"n1_{L}", torch.randn(D, dtype=DT))
            self.register_buffer(f"n2_{L}", torch.randn(D, dtype=DT))
        self.register_buffer("cos", torch.randn(1, 1, HD, dtype=DT))
        self.register_buffer("sin", torch.randn(1, 1, HD, dtype=DT))
        self.register_buffer("mask", torch.triu(torch.full((S, S), NEG, dtype=DT), diagonal=1))

    def _rope(self, t, heads):
        t = t.view(S, heads, HD); x1 = t[..., :HD//2]; x2 = t[..., HD//2:]
        return (t * self.cos + torch.cat((-x2, x1), -1) * self.sin).reshape(S, heads * HD)

    def _attn(self, q, k, v):
        q = q.view(S, NH, HD).transpose(0, 1)
        k = k.view(S, KV, HD).transpose(0, 1).unsqueeze(1).expand(KV, GROUP, S, HD).reshape(NH, S, HD)
        v = v.view(S, KV, HD).transpose(0, 1).unsqueeze(1).expand(KV, GROUP, S, HD).reshape(NH, S, HD)
        a = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * SCALE + self.mask, dim=-1)
        return torch.bmm(a, v).transpose(0, 1).reshape(S, NH * HD)

    def _layer(self, x, L):
        g = lambda nm: getattr(self, f"{nm}_{L}")
        h = rmsnorm(x, g("n1"))
        q = self._rope(torch.mm(h, g("wq")), NH); k = self._rope(torch.mm(h, g("wk")), KV); v = torch.mm(h, g("wv"))
        o = torch.mm(self._attn(q, k, v), g("wo"))
        x2 = x + o
        h2 = rmsnorm(x2, g("n2"))
        mlp = torch.mm(torch.nn.functional.silu(torch.mm(h2, g("wg"))) * torch.mm(h2, g("wu")), g("wd"))
        return x2 + mlp

    def forward(self, x):
        for L in range(self.n):
            x = self._layer(x, L)
        return x


def dev_cyc(mod, x):
    cm = ft.CompileModule.from_module(mod, (x,), compiler_config=cfg()).to(DEV)
    xd = x.to(DEV)
    for _ in range(8):
        cm(xd)
    pp = f"{SDIR}/fl_{NL}_{S}.json"
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
    c = dev_cyc(NLayers(NL), torch.randn(S, D, dtype=DT))
    print(f"CYC {NL} {S} {c}", flush=True)
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
