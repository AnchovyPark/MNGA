#!/usr/bin/env python3
"""n-gram telescoping — ★chain(데이터 흐름) 보존★ 버전 (2026.3.0, device-cycle).

이전 ngram_telescope_30.py는 op에 독립 input을 줘서 순차 연결(o→gate, up→down)이 끊겼음.
이 버전은 실제 Tokenwise 흐름을 보존:
  q,k,v ← h=rmsnorm(x) (공유)
  o     ← attn_out (외부, attention 출력)
  gate,up ← h2=rmsnorm(x+o)         (o에 의존)
  down  ← silu(gate)*up             (gate,up에 의존)
연속 window가 시퀀스 중간서 시작하면 그 진입 지점 값만 외부 dummy, 창 내부는 실제 연결.

사용: TUC_PROFILE_LEVEL=info RUST_LOG=info,span::tuc=info \
      /home/furiosa/venv3030/bin/python ngram_telescope_chained_30.py <seq>
"""
import os, sys, json, math

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
OPS = ["q", "k", "v", "o", "gate", "up", "down"]


def cfg():
    return CompilerConfig(tactic_hint=TacticHintConfig.ForLlmModelComputeBound,
                          scheduler_beam_search=True, use_attention_kernel=True)


def rmsnorm(x, w):
    v = (x.to(torch.float32) ** 2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(v.to(DT) + EPS)) * w


def rope(t, heads, cos, sin):
    t = t.view(S, heads, HD)
    x1 = t[..., :HD//2]; x2 = t[..., HD//2:]
    return (t * cos + torch.cat((-x2, x1), -1) * sin).reshape(S, heads * HD)


class WChain(torch.nn.Module):
    """연속 window(ops)를 실제 연결 보존해 계산. 외부 dummy = 창 진입 입력만."""
    def __init__(self, ops):
        super().__init__(); self.ops = ops
        for n, sh in [("wq",(D,NH*HD)),("wk",(D,KV*HD)),("wv",(D,KV*HD)),("wo",(NH*HD,D)),
                      ("wg",(D,INTER)),("wu",(D,INTER)),("wd",(INTER,D))]:
            self.register_buffer(n, torch.randn(*sh, dtype=DT))
        self.register_buffer("n1", torch.randn(D, dtype=DT)); self.register_buffer("n2", torch.randn(D, dtype=DT))
        self.register_buffer("cos", torch.randn(1, 1, HD, dtype=DT)); self.register_buffer("sin", torch.randn(1, 1, HD, dtype=DT))

    def forward(self, x, attn_out, h2_ext, din_ext):
        ops = self.ops; out = {}
        if any(o in ops for o in ("q", "k", "v")):
            h = rmsnorm(x, self.n1)
            if "q" in ops: out["q"] = rope(torch.mm(h, self.wq), NH, self.cos, self.sin)
            if "k" in ops: out["k"] = rope(torch.mm(h, self.wk), KV, self.cos, self.sin)
            if "v" in ops: out["v"] = torch.mm(h, self.wv)
        if "o" in ops:
            out["o"] = torch.mm(attn_out, self.wo)
        if "gate" in ops or "up" in ops:
            h2 = rmsnorm(x + out["o"], self.n2) if "o" in ops else h2_ext
            if "gate" in ops: out["gate"] = torch.mm(h2, self.wg)
            if "up" in ops: out["up"] = torch.mm(h2, self.wu)
        if "down" in ops:
            din = (torch.nn.functional.silu(out["gate"]) * out["up"]) if ("gate" in ops and "up" in ops) else din_ext
            out["down"] = torch.mm(din, self.wd)
        return tuple(out[o] for o in ops)


def externals():
    return (torch.randn(S, D, dtype=DT), torch.randn(S, NH*HD, dtype=DT),
            torch.randn(S, D, dtype=DT), torch.randn(S, INTER, dtype=DT))


def dev_cyc(ops):
    mod = WChain(ops); xs = externals()
    cm = ft.CompileModule.from_module(mod, xs, compiler_config=cfg()).to(DEV)
    xds = [x.to(DEV) for x in xs]
    for _ in range(8):
        cm(*xds)
    pp = f"{SDIR}/nc_{'_'.join(ops)}_{S}.json"
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


def main():
    # 단일 window 모드: argv[2] = 쉼표구분 ops (fresh 프로세스로 degrade 회피)
    win = sys.argv[2].split(",")
    c = dev_cyc(win)
    print(f"CYC {'_'.join(win)} {S} {c}", flush=True)
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
