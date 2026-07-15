#!/usr/bin/env python3
"""prefill S=128 layer-level joint-compile ladder → 2× gap이 어느 경계에서 닫히나.

context-접근의 확장: "연산 순서/문맥으로 예측"이 그래프 경계를 넘어 layer 까지
버티는가. 작은 단위 → full layer → 2-layer 로 joint-compile 범위를 키우며
per-unit latency 가 어떻게 줄어드는지 측정. 단순 합(격리)이 2× 틀리는 게
(a) 그냥 더해서인지 vs (b) 경계를 넘는 문맥(weight-prefetch)이 빠져서인지 가린다.

Llama-3.1-8B faithful (GQA): D=4096 NH=32 KV=8 HD=128 INTER=14336.
  attn block: rmsnorm → q,k,v(GQA) → eager attn → o_proj → residual
  mlp  block: rmsnorm → gate,up → silu*up → down → residual
  layer     : attn block → mlp block
(RoPE/causal mask/KV paging 생략 — latency 지배 커널엔 무관, 주석 참고)

TP=8 (set_fusion(8)), B=1 S=128. 사용법: python prefill_layer_joint_s128.py
"""
import statistics as st
from collections import defaultdict

import torch
import torch.nn.functional as F
import furiosa.torch  # noqa: F401
from furiosa.torch import native_device as nd

nd.set_fusion(8)

from furiosa.torch.custom_ops import CompileModule  # noqa: E402

D, NH, HD, KV, INTER, L, VOCAB = 4096, 32, 128, 8, 14336, 32, 128256
B, S = 1, 128
SCALE = HD ** -0.5
DTYPE = torch.bfloat16
N_RUNS = 3

# 참조 상수 (이전 측정)
ISO_SUM_PER_LAYER = 1569.0     # Σ 격리 single (prefill_kernels_s128)
CHAIN_SUM_PER_LAYER = 1333.0   # chain-fused 합 (prefill_chains_s128)
REAL_PER_LAYER = 628.0         # ground truth 20.1ms / 32
FINAL_NORM, LM_HEAD = 6.7, 1765.4
GT_PREFILL_MS = 20.1


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
        self.register_buffer("wk", torch.randn(D, KV * HD, dtype=DTYPE))  # GQA
        self.register_buffer("wv", torch.randn(D, KV * HD, dtype=DTYPE))  # GQA
        self.register_buffer("wo", torch.randn(NH * HD, D, dtype=DTYPE))

    def forward(self, h):
        v0 = h.float()
        n = (v0 * torch.rsqrt(v0.pow(2).mean(-1, keepdim=True) + 1e-5) * self.g.float()).to(h.dtype)
        q = (n @ self.wq).view(B, S, NH, HD).transpose(1, 2)             # (B,NH,S,HD)
        k = (n @ self.wk).view(B, S, KV, HD).transpose(1, 2)             # (B,KV,S,HD)
        v = (n @ self.wv).view(B, S, KV, HD).transpose(1, 2)
        # GQA: KV → NH heads (그룹 반복)
        k = k.view(B, KV, 1, S, HD).expand(B, KV, NH // KV, S, HD).reshape(B, NH, S, HD)
        v = v.view(B, KV, 1, S, HD).expand(B, KV, NH // KV, S, HD).reshape(B, NH, S, HD)
        scores = torch.softmax((q @ k.transpose(-1, -2)) * SCALE, dim=-1)  # (B,NH,S,S)
        ctx = (scores @ v).transpose(1, 2).reshape(B, S, NH * HD)          # (B,S,D)
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
    names = set()
    for _ in range(N_RUNS):
        pr = cm.generate_profiles(dev); cm(xd, profiles=pr, device=dev)
        di = [[p.device.index for p in inn] for inn in pr]
        pc = [[p.cpu() for p in inn] for inn in pr]
        spans = cm.edf.npu_node.build_tuc_profile_spans(pc, di, 10**9)
        by = defaultdict(list); task = 0.0
        for e in spans:
            names.add(e.name)
            if e.name == "Task": task += e.time_range.elapsed_us()
            else: by[e.name].append((e.time_range.start, e.time_range.end))
        runs.append({
            "task": task,
            "tu": union_us(by.get("Renegade::TuExec", [])),
            "dma": union_us(by.get("DMA", [])),
            "sto": union_us(by.get("Renegade::StoTrf", [])),
            "parcopy": union_us(by.get("Renegade::ParallelCopy", [])),
            "cluster": union_us(by.get("Cluster", [])),
        })
    med = {k: st.median(r[k] for r in runs) for k in runs[0]}
    med["spans"] = "|".join(sorted(names))
    return med


def main():
    print(f"=== prefill layer joint ladder S={S} TP=8 (Llama-3.1-8B GQA) ===", flush=True)
    units = []
    for name, mod in [("attn_block", AttnBlock()),
                      ("mlp_block", MLPBlock()),
                      ("full_layer", Layer()),
                      ("two_layer", NLayers(2))]:
        try:
            m = measure(mod)
            m["name"] = name
            units.append(m)
            print(f"  {name:11s} task={m['task']:8.1f}us  tu={m['tu']:7.1f} dma={m['dma']:7.1f} "
                  f"sto={m['sto']:6.1f} parcopy={m['parcopy']:6.1f} cluster={m['cluster']:6.1f}",
                  flush=True)
        except Exception as exc:
            print(f"  {name:11s} FAIL: {type(exc).__name__}: {str(exc)[:160]}", flush=True)

    d = {u["name"]: u["task"] for u in units}
    print("\n=== 사다리 (per-layer, us) ===", flush=True)
    print(f"  Σ 격리 single      : {ISO_SUM_PER_LAYER:8.1f}   ({ISO_SUM_PER_LAYER/REAL_PER_LAYER:.2f}x)", flush=True)
    print(f"  chain-fused 합      : {CHAIN_SUM_PER_LAYER:8.1f}   ({CHAIN_SUM_PER_LAYER/REAL_PER_LAYER:.2f}x)", flush=True)
    if "attn_block" in d and "mlp_block" in d:
        block_sum = d["attn_block"] + d["mlp_block"]
        print(f"  attn_blk+mlp_blk    : {block_sum:8.1f}   ({block_sum/REAL_PER_LAYER:.2f}x)   "
              f"[attn {d['attn_block']:.0f} + mlp {d['mlp_block']:.0f}]", flush=True)
    if "full_layer" in d:
        fl = d["full_layer"]
        print(f"  full_layer joint    : {fl:8.1f}   ({fl/REAL_PER_LAYER:.2f}x)", flush=True)
        if "attn_block" in d and "mlp_block" in d:
            print(f"    → 블록경계 fusion 이득: {fl - block_sum:+.1f}us "
                  f"({(fl-block_sum)/block_sum*100:+.0f}%)", flush=True)
    if "two_layer" in d:
        pl = d["two_layer"] / 2
        print(f"  2-layer joint ÷2    : {pl:8.1f}   ({pl/REAL_PER_LAYER:.2f}x)", flush=True)
        if "full_layer" in d:
            print(f"    → layer간 prefetch 이득: {pl - d['full_layer']:+.1f}us/layer", flush=True)
    print(f"  실제 per-layer (GT) : {REAL_PER_LAYER:8.1f}   (1.00x)", flush=True)

    if "full_layer" in d:
        pred = d["full_layer"] * L + FINAL_NORM + LM_HEAD
        print(f"\n=== full_layer×{L} + lm_head 예측 prefill ===", flush=True)
        print(f"  = {pred/1000:.2f} ms  vs GT {GT_PREFILL_MS} ms  → {pred/1000/GT_PREFILL_MS:.2f}x", flush=True)


if __name__ == "__main__":
    main()
