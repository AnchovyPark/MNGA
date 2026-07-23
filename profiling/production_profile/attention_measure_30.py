#!/usr/bin/env python3
"""Attention supertask 격리 측정 (2026.3.0, device-cycle). 실제 RNGD 실행.

Attention supertask = matmul_qk + causal softmax + matmul_score_v (GQA: NH=32 q헤드, KV=8 kv헤드).
Tokenwise와 함께 한 레이어를 이룸. in-model Attention GT(1B, cyc): 128:41K, 256:75K, 512:143K, 1024:477K.
prefill attention은 scores가 S×S라 latency가 S에 비선형(∝~S^1.2, 작은 seq는 오버헤드 지배).

사용:
  검증(수치): /home/furiosa/venv3030/bin/python attention_measure_30.py <seq> check
  측정(cyc) : TUC_PROFILE_LEVEL=info RUST_LOG=info,span::tuc=info \
              /home/furiosa/venv3030/bin/python attention_measure_30.py <seq> time
"""
import os, sys, json, math

import torch
import furiosa.torch as ft
ft.set_fusion(8)
from furiosa.torch._C.config.compiler import CompilerConfig, TacticHintConfig
from furiosa.torch.profiler import Profiler

S = int(sys.argv[1]) if len(sys.argv) > 1 else 512
MODE = sys.argv[2] if len(sys.argv) > 2 else "time"
DEV = "furiosa:0"
NH, KV, HD = 32, 8, 64
GROUP = NH // KV
SCALE = 1.0 / math.sqrt(HD)
DT = torch.bfloat16
SDIR = "/tmp/claude-1002/-home-furiosa------pjh-rngd/8464c59e-3432-4d59-9ccc-d34e43519088/scratchpad"
INMODEL = {128: 41000, 256: 75000, 512: 143000, 1024: 477000}
NEG = -30000.0  # bf16-safe causal mask (softmax→0)


def cfg():
    return CompilerConfig(tactic_hint=TacticHintConfig.ForLlmModelComputeBound,
                          scheduler_beam_search=True, use_attention_kernel=True)


class Attention(torch.nn.Module):
    """q:(S,NH*HD), k,v:(S,KV*HD) → out:(S,NH*HD). matmul_qk + causal softmax + score_v."""
    def __init__(self):
        super().__init__()
        m = torch.triu(torch.full((S, S), NEG, dtype=DT), diagonal=1)
        self.register_buffer("mask", m)

    def forward(self, q, k, v):
        q = q.view(S, NH, HD).transpose(0, 1)                 # (NH,S,HD)
        k = k.view(S, KV, HD).transpose(0, 1)                 # (KV,S,HD)
        v = v.view(S, KV, HD).transpose(0, 1)
        k = k.unsqueeze(1).expand(KV, GROUP, S, HD).reshape(NH, S, HD)   # GQA
        v = v.unsqueeze(1).expand(KV, GROUP, S, HD).reshape(NH, S, HD)
        scores = torch.bmm(q, k.transpose(1, 2)) * SCALE + self.mask     # (NH,S,S) matmul_qk
        attn = torch.softmax(scores, dim=-1)                            # softmax
        out = torch.bmm(attn, v)                                        # (NH,S,HD) score_v
        return out.transpose(0, 1).reshape(S, NH * HD)


def inputs():
    return (torch.randn(S, NH*HD, dtype=DT), torch.randn(S, KV*HD, dtype=DT), torch.randn(S, KV*HD, dtype=DT))


def check():
    m = Attention(); q, k, v = inputs()
    cm = ft.CompileModule.from_module(m, (q, k, v), compiler_config=cfg()).to(DEV)
    out = cm(q.to(DEV), k.to(DEV), v.to(DEV)).to("cpu").float()
    # CPU ref (같은 mask)
    def ref():
        qq = q.float().view(S,NH,HD).transpose(0,1); kk=k.float().view(S,KV,HD).transpose(0,1); vv=v.float().view(S,KV,HD).transpose(0,1)
        kk = kk.unsqueeze(1).expand(KV,GROUP,S,HD).reshape(NH,S,HD); vv=vv.unsqueeze(1).expand(KV,GROUP,S,HD).reshape(NH,S,HD)
        sc = torch.bmm(qq, kk.transpose(1,2))*SCALE + m.mask.float()
        return torch.bmm(torch.softmax(sc,-1), vv).transpose(0,1).reshape(S,NH*HD)
    r = ref(); rel = ((out-r).abs().max()/(r.abs().max()+1e-9)).item()
    print(f"Attention@{S} NPU vs CPU 상대오차 = {rel:.4f}", flush=True)


def dev_cyc(mod, xs):
    cm = ft.CompileModule.from_module(mod, xs, compiler_config=cfg()).to(DEV)
    xds = [x.to(DEV) for x in xs]
    for _ in range(8):
        cm(*xds)
    pp = f"{SDIR}/attn_{S}.json"
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


def time_it():
    c = dev_cyc(Attention(), inputs())
    gt = INMODEL.get(S)
    msg = f"Attention@{S}: {c} cyc"
    if gt and c:
        msg += f"  | in-model {gt}  → 격리/in-model = {c/gt:.2f}x"
    print(msg, flush=True)


if __name__ == "__main__":
    (check if MODE == "check" else time_it)()
    sys.stdout.flush(); os._exit(0)
