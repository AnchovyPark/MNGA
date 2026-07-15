#!/usr/bin/env python3
"""prefill S=128 layer-DEPTH 스윕 → per-layer latency가 깊이로 실제값에 수렴하나.

prefill_layer_joint_s128 에서 full_layer joint 가 1.71x, 2-layer 1.65x 로 정체.
2-layer에서 -40us/layer 이득 → 깊이가 깊어질수록 계속 닫히는가(=deep-pipeline
amortization)를 N=1,2,4,8 layer joint-compile 로 직접 확인. per-layer=task/N 이
실제 per-layer(~570-628us)로 수렴하면 gap은 깊이-amortization, plateau면
컴파일 경로 차이(CompileModule vs production).

TP=8, B=1 S=128, Llama-3.1-8B GQA faithful. 사용법: python prefill_layer_depth_s128.py
"""
import statistics as st
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
import furiosa.torch  # noqa: F401
from furiosa.torch import native_device as nd

nd.set_fusion(8)

from furiosa.torch.custom_ops import CompileModule  # noqa: E402

D, NH, HD, KV, INTER = 4096, 32, 128, 8, 14336
B = 1
S = int(sys.argv[1]) if len(sys.argv) > 1 else 128
DEPTHS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2, 4, 8]
SCALE = HD ** -0.5
DTYPE = torch.bfloat16
N_RUNS = 3
# real per-layer ≈ (prefill - lm_head) / 32 : S128 20.1ms→570us, S2048 28.1ms→825us
REAL_PER_LAYER = {128: 570.0, 512: 700.0, 2048: 825.0}.get(S, 570.0)


def union_us(iv):
    if not iv:
        return 0.0
    iv = sorted(iv); t = 0.0; cs, ce = iv[0]
    for s, e in iv[1:]:
        if s <= ce: ce = max(ce, e)
        else: t += ce - cs; cs, ce = s, e
    return t + (ce - cs)


class AttnBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("g", torch.randn(D, dtype=DTYPE))
        self.register_buffer("wq", torch.randn(D, NH * HD, dtype=DTYPE))
        self.register_buffer("wk", torch.randn(D, KV * HD, dtype=DTYPE))
        self.register_buffer("wv", torch.randn(D, KV * HD, dtype=DTYPE))
        self.register_buffer("wo", torch.randn(NH * HD, D, dtype=DTYPE))

    def forward(self, h):
        v0 = h.float()
        n = (v0 * torch.rsqrt(v0.pow(2).mean(-1, keepdim=True) + 1e-5) * self.g.float()).to(h.dtype)
        q = (n @ self.wq).view(B, S, NH, HD).transpose(1, 2)
        k = (n @ self.wk).view(B, S, KV, HD).transpose(1, 2)
        v = (n @ self.wv).view(B, S, KV, HD).transpose(1, 2)
        k = k.view(B, KV, 1, S, HD).expand(B, KV, NH // KV, S, HD).reshape(B, NH, S, HD)
        v = v.view(B, KV, 1, S, HD).expand(B, KV, NH // KV, S, HD).reshape(B, NH, S, HD)
        scores = torch.softmax((q @ k.transpose(-1, -2)) * SCALE, dim=-1)
        ctx = (scores @ v).transpose(1, 2).reshape(B, S, NH * HD)
        return h + ctx @ self.wo


class MLPBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("g", torch.randn(D, dtype=DTYPE))
        self.register_buffer("wg", torch.randn(D, INTER, dtype=DTYPE))
        self.register_buffer("wu", torch.randn(D, INTER, dtype=DTYPE))
        self.register_buffer("wd", torch.randn(INTER, D, dtype=DTYPE))

    def forward(self, h):
        v0 = h.float()
        n = (v0 * torch.rsqrt(v0.pow(2).mean(-1, keepdim=True) + 1e-5) * self.g.float()).to(h.dtype)
        x = F.silu(n @ self.wg) * (n @ self.wu)
        return h + x @ self.wd


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = AttnBlock()
        self.mlp = MLPBlock()

    def forward(self, h):
        return self.mlp(self.attn(h))


class NLayers(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer() for _ in range(n)])

    def forward(self, h):
        for lyr in self.layers:
            h = lyr(h)
        return h


def measure(mod):
    dev = torch.device("rngd", 0)
    x = torch.randn(B, S, D, dtype=DTYPE)
    cm = CompileModule.from_exported(torch.export.export(mod, (x,)))
    cm.to(dev); xd = x.to(dev)
    cm(xd, profiles=None, device=dev)
    runs = []
    for _ in range(N_RUNS):
        pr = cm.generate_profiles(dev); cm(xd, profiles=pr, device=dev)
        di = [[p.device.index for p in inn] for inn in pr]
        pc = [[p.cpu() for p in inn] for inn in pr]
        spans = cm.edf.npu_node.build_tuc_profile_spans(pc, di, 10**9)
        by = defaultdict(list); task = 0.0
        for e in spans:
            if e.name == "Task": task += e.time_range.elapsed_us()
            else: by[e.name].append((e.time_range.start, e.time_range.end))
        runs.append({
            "task": task,
            "tu": union_us(by.get("Renegade::TuExec", [])),
            "dma": union_us(by.get("DMA", [])),
            "cluster": union_us(by.get("Cluster", [])),
        })
    return {k: st.median(r[k] for r in runs) for k in runs[0]}


def main():
    print(f"=== prefill layer DEPTH sweep S={S} TP=8 (Llama GQA) ===", flush=True)
    print(f"{'N':>3} {'task':>9} {'per-layer':>10} {'vs real':>8} | "
          f"{'tu/L':>7} {'dma/L':>7} {'cluster/L':>9}", flush=True)
    prev_pl = None
    for n in DEPTHS:
        try:
            m = measure(NLayers(n))
            pl = m["task"] / n
            delta = f"{pl-prev_pl:+.1f}" if prev_pl is not None else "  -"
            print(f"{n:>3} {m['task']:9.1f} {pl:10.1f} {pl/REAL_PER_LAYER:7.2f}x | "
                  f"{m['tu']/n:7.1f} {m['dma']/n:7.1f} {m['cluster']/n:9.1f}   "
                  f"(Δ/L {delta})", flush=True)
            prev_pl = pl
        except Exception as exc:
            print(f"{n:>3}  FAIL: {type(exc).__name__}: {str(exc)[:150]}", flush=True)
    print(f"\n실제 per-layer ≈ {REAL_PER_LAYER:.0f}us (GT 20.1ms - lm_head, /32)", flush=True)
    print("per-layer가 깊이로 실제값에 수렴하면 gap=깊이 amortization, "
          "plateau면 컴파일 경로 차이.", flush=True)


if __name__ == "__main__":
    main()
