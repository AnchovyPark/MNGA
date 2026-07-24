#!/usr/bin/env python3
"""레이어 경계 op 창 (2026.3.0, device-cycle) — ★chain 완전 보존★ (down←silu(gate)·up 포함).

경계: ... gate_L,up_L,down_L → [out=x2+down] → rmsnorm → q_{L+1},k_{L+1},v_{L+1} ...
왼쪽 = [gate,up,down]의 down 포함 suffix, 오른쪽 = [q,k,v] prefix. 실제 연결 보존:
  - gate,up ← h2_ext (창에 있으면), down ← silu(gate)·up  (up이 창에 있으면 그 up을 씀 = chain)
  - q,k,v ← rmsnorm(x2_ext + down)  (형제, h 공유)
창 밖 값만 외부 dummy(gate_ext,up_ext는 down의 예측 입력 구성용, 값 무관).

n-gram-3 경계 문맥: "up,down|q", "down|q,k", "up,down|q,k" 등 겹치는 창으로 경계 overlap 추출.

사용: TUC_PROFILE_LEVEL=info RUST_LOG=info,span::tuc=info \
      /home/furiosa/venv3030/bin/python boundary_window_30.py <seq> "<left>|<right>"
출력: CYC <left>_<right> <seq> <cyc>
"""
import os, sys, json

import torch
import furiosa.torch as ft
ft.set_fusion(8)
from furiosa.torch._C.config.compiler import CompilerConfig, TacticHintConfig
from furiosa.torch.profiler import Profiler

S = int(sys.argv[1]) if len(sys.argv) > 1 else 512
SPEC = sys.argv[2] if len(sys.argv) > 2 else "down|q"
LEFT = SPEC.split("|")[0].split(",")
RIGHT = SPEC.split("|")[1].split(",")
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


class Boundary(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for nm, sh in [("wg",(D,INTER)),("wu",(D,INTER)),("wd",(INTER,D)),
                       ("wq",(D,NH*HD)),("wk",(D,KV*HD)),("wv",(D,KV*HD))]:
            self.register_buffer(nm, torch.randn(*sh, dtype=DT))
        self.register_buffer("n1", torch.randn(D, dtype=DT))
        self.register_buffer("cos", torch.randn(1, 1, HD, dtype=DT))
        self.register_buffer("sin", torch.randn(1, 1, HD, dtype=DT))

    def _rope(self, t, heads):
        t = t.view(S, heads, HD); x1 = t[..., :HD//2]; x2 = t[..., HD//2:]
        return (t * self.cos + torch.cat((-x2, x1), -1) * self.sin).reshape(S, heads * HD)

    def forward(self, h2_ext, x2_ext, gate_ext, up_ext):
        outs = []
        # 왼쪽: gate,up (창에 있으면 h2_ext에서, 없으면 외부 dummy). down = silu(gate)*up (chain 보존)
        g = torch.mm(h2_ext, self.wg) if "gate" in LEFT else gate_ext
        u = torch.mm(h2_ext, self.wu) if "up" in LEFT else up_ext
        down = torch.mm(torch.nn.functional.silu(g) * u, self.wd)
        if "gate" in LEFT: outs.append(g)
        if "up" in LEFT: outs.append(u)
        outs.append(down)   # down은 항상 왼쪽에 포함
        # 경계: residual + rmsnorm → 레이어 L+1 진입
        h = rmsnorm(x2_ext + down, self.n1)
        if "q" in RIGHT: outs.append(self._rope(torch.mm(h, self.wq), NH))
        if "k" in RIGHT: outs.append(self._rope(torch.mm(h, self.wk), KV))
        if "v" in RIGHT: outs.append(torch.mm(h, self.wv))
        return tuple(outs)


def dev_cyc():
    mod = Boundary()
    xs = (torch.randn(S, D, dtype=DT), torch.randn(S, D, dtype=DT),
          torch.randn(S, INTER, dtype=DT), torch.randn(S, INTER, dtype=DT))
    cm = ft.CompileModule.from_module(mod, xs, compiler_config=cfg()).to(DEV)
    xds = [x.to(DEV) for x in xs]
    for _ in range(8):
        cm(*xds)
    pp = f"{SDIR}/bw_{'_'.join(LEFT)}_{'_'.join(RIGHT)}_{S}.json"
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
    c = dev_cyc()
    print(f"CYC {'_'.join(LEFT)}_{'_'.join(RIGHT)} {S} {c}", flush=True)
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
