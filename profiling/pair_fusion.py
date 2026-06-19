#!/usr/bin/env python3
"""LLM 블록 내 인접 2-커널 쌍의 fusion 경향 측정.

커널 시간 = Task span (device latency, 겹침 반영).
각 쌍 (A,B) 에 대해:
  iso : A 단독, B 단독 (각각 따로 compile/실행)         → task_A, task_B
  fused: A→B joint-compile (한 그래프, 컴파일러 fusion)  → task_AB
  fusion_delta = task_AB - (task_A + task_B)   (음수 = fusion 이득)
  fusion_ratio = task_AB / (task_A + task_B)
DMA/연산(union) 도 같이 떠서 fusion 이 HBM 트래픽을 줄였는지 본다.

블록 내부 인접쌍만 (attention / MLP), 블록 경계 넘는 쌍 제외.
matmul↔vector 패턴별로 본다.

사용법: python pair_fusion.py [decode|prefill|all]
"""
import csv
import os
import statistics as st
import sys
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
import furiosa.torch  # noqa
from furiosa.torch.custom_ops import CompileModule

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "pair_fusion_results.csv")

# Llama-3.1-8B
D, NH, HD, INTER = 4096, 32, 128, 14336
DTYPE = torch.bfloat16
N_RUNS = 3
REGIME = sys.argv[1] if len(sys.argv) > 1 else "all"


class Mod(torch.nn.Module):
    """ops=[(name, aux)] 를 순서대로 적용. aux 는 buffer."""
    def __init__(self, ops):
        super().__init__()
        self.names = [o[0] for o in ops]
        self.has = []
        for i, (_n, aux) in enumerate(ops):
            if aux is not None:
                self.register_buffer(f"a{i}", aux); self.has.append(True)
            else:
                self.has.append(False)

    def _ap(self, n, h, a):
        if n == "matmul":  return h @ a
        if n == "mul":     return h * a
        if n == "add":     return h + a
        if n == "silu":    return F.silu(h)
        if n == "softmax": return torch.softmax(h, dim=-1)
        if n == "rmsnorm":
            v = h.float(); v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-5)
            return (v * a.float()).to(h.dtype)
        raise ValueError(n)

    def forward(self, h):
        for i, n in enumerate(self.names):
            h = self._ap(n, h, getattr(self, f"a{i}") if self.has[i] else None)
        return h


def union(iv):
    if not iv: return 0.0
    iv = sorted(iv); t = 0.0; cs, ce = iv[0]
    for s, e in iv[1:]:
        if s <= ce: ce = max(ce, e)
        else: t += ce - cs; cs, ce = s, e
    return t + (ce - cs)


def measure(ops, in_shape, dev):
    """ops 모듈을 compile/실행 → {task, tu, dma, env} (us, median)."""
    mod = Mod(ops)
    h = torch.randn(*in_shape, dtype=DTYPE)
    cm = CompileModule.from_exported(torch.export.export(mod, (h,)))
    cm.to(dev); hd = h.to(dev)
    cm(hd, profiles=None, device=dev)
    runs = []
    for _ in range(N_RUNS):
        profiles = cm.generate_profiles(dev)
        cm(hd, profiles=profiles, device=dev)
        di = [[p.device.index for p in inner] for inner in profiles]
        pc = [[p.cpu() for p in inner] for inner in profiles]
        spans = cm.edf.npu_node.build_tuc_profile_spans(pc, di, 10**9)
        by = defaultdict(list)
        task = 0.0
        for e in spans:
            if e.name == "Task":
                task += e.time_range.elapsed_us()
            else:
                by[e.name].append((e.time_range.start, e.time_range.end))
        tu = union(by.get("Renegade::TuExec", []))
        dma = union(by.get("DMA", []))
        allv = [iv for v in by.values() for iv in v]
        env = (max(e for _, e in allv) - min(s for s, _ in allv)) if allv else 0.0
        runs.append((task or env, tu, dma, env))
    return {k: st.median([r[i] for r in runs])
            for i, k in enumerate(["task", "tu", "dma", "env"])}


def aux(kind, shape):
    return torch.randn(*shape, dtype=DTYPE) if kind else None


def build_pairs(B, S, ctx):
    """블록 내부 인접쌍: (name, block, patternAB, opA, opB, inA, inB).
    opX=(opname, aux_shape_or_None). inB = A 의 출력 shape."""
    P = []
    # ---- MLP ----
    P.append(("rmsnorm→gate_proj", "mlp", "vec→mm",
              ("rmsnorm", (D,)), ("matmul", (D, INTER)), (B, S, D), (B, S, D)))
    P.append(("gate_proj→silu", "mlp", "mm→vec",
              ("matmul", (D, INTER)), ("silu", None), (B, S, D), (B, S, INTER)))
    P.append(("silu→mul(swiglu)", "mlp", "vec→vec",
              ("silu", None), ("mul", (B, S, INTER)), (B, S, INTER), (B, S, INTER)))
    P.append(("mul→down_proj", "mlp", "vec→mm",
              ("mul", (B, S, INTER)), ("matmul", (INTER, D)), (B, S, INTER), (B, S, INTER)))
    P.append(("down_proj→residual", "mlp", "mm→vec",
              ("matmul", (INTER, D)), ("add", (B, S, D)), (B, S, INTER), (B, S, D)))
    # ---- Attention ----  (q:[B,NH,S,HD], k:[B,NH,HD,ctx], v:[B,NH,ctx,HD])
    P.append(("QK→softmax", "attn", "mm→vec",
              ("matmul", (B, NH, HD, ctx)), ("softmax", None),
              (B, NH, S, HD), (B, NH, S, ctx)))
    P.append(("softmax→AV", "attn", "vec→mm",
              ("softmax", None), ("matmul", (B, NH, ctx, HD)),
              (B, NH, S, ctx), (B, NH, S, ctx)))
    return P


def main():
    dev = torch.device("rngd", 0)
    regimes = []
    if REGIME in ("decode", "all"):
        regimes.append(("decode", 8, 1, 2048))
    if REGIME in ("prefill", "all"):
        regimes.append(("prefill", 1, 2048, 2048))   # furiosa prefill_buckets=[(1,b)] → B=1

    rows = []
    for reg, B, S, ctx in regimes:
        for (name, block, pat, opA, opB, inA, inB) in build_pairs(B, S, ctx):
            try:
                auxA = (opA[0], aux(opA[1], opA[1]) if opA[1] else None)
                auxB = (opB[0], aux(opB[1], opB[1]) if opB[1] else None)
                mA = measure([auxA], inA, dev)
                mB = measure([auxB], inB, dev)
                mAB = measure([auxA, auxB], inA, dev)
                naive = mA["task"] + mB["task"]
                delta = mAB["task"] - naive
                ratio = mAB["task"] / naive if naive else 0
                dma_naive = mA["dma"] + mB["dma"]
                dma_save = mAB["dma"] - dma_naive
                print(f"[{reg}] {name:20s}({pat}) "
                      f"A={mA['task']:.0f} B={mB['task']:.0f} fused={mAB['task']:.0f} "
                      f"Δ={delta:+.0f}({ratio*100:.0f}%) | "
                      f"dma A+B={dma_naive:.0f}→{mAB['dma']:.0f}({dma_save:+.0f})", flush=True)
                rows.append(dict(regime=reg, pair=name, block=block, pattern=pat,
                                 task_A=round(mA["task"], 1), task_B=round(mB["task"], 1),
                                 task_fused=round(mAB["task"], 1),
                                 naive_sum=round(naive, 1), fusion_delta=round(delta, 1),
                                 fusion_ratio=round(ratio, 3),
                                 dma_A=round(mA["dma"], 1), dma_B=round(mB["dma"], 1),
                                 dma_fused=round(mAB["dma"], 1), dma_save=round(dma_save, 1),
                                 tu_A=round(mA["tu"], 1), tu_B=round(mB["tu"], 1),
                                 tu_fused=round(mAB["tu"], 1)))
            except Exception as e:
                print(f"[{reg}] {name} FAIL: {type(e).__name__}: {str(e)[:120]}", flush=True)
                rows.append(dict(regime=reg, pair=name, block=block, error=str(e)[:160]))

    fields = ["regime", "block", "pair", "pattern", "task_A", "task_B", "task_fused",
              "naive_sum", "fusion_delta", "fusion_ratio", "dma_A", "dma_B",
              "dma_fused", "dma_save", "tu_A", "tu_B", "tu_fused", "error"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\n[DONE] {OUT_CSV} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
