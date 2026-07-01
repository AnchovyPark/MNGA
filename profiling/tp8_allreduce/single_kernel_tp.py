#!/usr/bin/env python3
"""Attention single-kernel 격리 측정을 TP=1 vs TP=8 로.

attention_chain_tp.py 의 짝. 체인의 composition gap 을 TP=8 기준으로 계산하려면
격리 컴포넌트도 TP=8 로 재야 한다. 여기서 attention 쪽 single kernel 7종을 전부 잰다:
  qk_matmul, softmax_scores, av_matmul, rmsnorm, qkv_proj, o_proj, residual_add

set_fusion(NPE) 은 프로세스당 1회 → NPE 인자로 두 번 실행(1, 8).

사용법:
  python single_kernel_tp.py 1
  python single_kernel_tp.py 8
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

import torch
import furiosa.torch  # noqa: F401
from furiosa.torch import native_device as nd

NPE = int(sys.argv[1]) if len(sys.argv) > 1 else 8
nd.set_fusion(NPE)

from furiosa.torch.custom_ops import CompileModule  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, f"single_kernel_tp_np{NPE}_results.csv")

B, NH, HD = 1, 32, 128
D = NH * HD
DTYPE = torch.bfloat16
N_RUNS = 3
DEFAULT_S = [128, 512, 2048]


def union_us(intervals):
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    return total + (cur_e - cur_s)


class Component(torch.nn.Module):
    def __init__(self, kind, s):
        super().__init__()
        self.kind = kind
        self.s = s
        if kind == "qk_matmul":
            self.register_buffer("k", torch.randn(B, NH, HD, s, dtype=DTYPE))
        elif kind == "av_matmul":
            self.register_buffer("v", torch.randn(B, NH, s, HD, dtype=DTYPE))
        elif kind == "rmsnorm":
            self.register_buffer("gain", torch.randn(D, dtype=DTYPE))
        elif kind == "qkv_proj":
            self.register_buffer("weight", torch.randn(D, 3 * D, dtype=DTYPE))
        elif kind == "o_proj":
            self.register_buffer("weight", torch.randn(D, D, dtype=DTYPE))
        elif kind == "residual_add":
            self.register_buffer("residual", torch.randn(B, s, D, dtype=DTYPE))

    def forward(self, x):
        if self.kind == "qk_matmul":
            return x @ self.k
        if self.kind == "softmax_scores":
            return torch.softmax(x, dim=-1)
        if self.kind == "av_matmul":
            return x @ self.v
        if self.kind == "rmsnorm":
            y = x.float()
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-5)
            return (y * self.gain.float()).to(x.dtype)
        if self.kind == "qkv_proj":
            return x @ self.weight
        if self.kind == "o_proj":
            y = x.permute(0, 2, 1, 3).reshape(B, self.s, D)
            return y @ self.weight
        if self.kind == "residual_add":
            return x + self.residual
        raise ValueError(self.kind)


def shape_for(kind, s):
    if kind == "qk_matmul":
        return (B, NH, s, HD)
    if kind in ("softmax_scores", "av_matmul"):
        return (B, NH, s, s)
    if kind in ("rmsnorm", "qkv_proj", "residual_add"):
        return (B, s, D)
    if kind == "o_proj":
        return (B, NH, s, HD)
    raise ValueError(kind)


def measure(kind, s, dev):
    x = torch.randn(*shape_for(kind, s), dtype=DTYPE)
    cm = CompileModule.from_exported(torch.export.export(Component(kind, s), (x,)))
    cm.to(dev)
    xd = x.to(dev)
    cm(xd, profiles=None, device=dev)
    runs = []
    names = set()
    for _ in range(N_RUNS):
        pr = cm.generate_profiles(dev)
        cm(xd, profiles=pr, device=dev)
        di = [[p.device.index for p in inner] for inner in pr]
        pc = [[p.cpu() for p in inner] for inner in pr]
        spans = cm.edf.npu_node.build_tuc_profile_spans(pc, di, 10**9)
        by = defaultdict(list)
        task = 0.0
        for e in spans:
            names.add(e.name)
            if e.name == "Task":
                task += e.time_range.elapsed_us()
            else:
                by[e.name].append((e.time_range.start, e.time_range.end))
        runs.append({
            "task": task,
            "tu": union_us(by.get("Renegade::TuExec", [])),
            "dma": union_us(by.get("DMA", [])),
            "parallelcopy": union_us(by.get("Renegade::ParallelCopy", [])),
            "cluster": union_us(by.get("Cluster", [])),
        })
    med = {k: st.median(r[k] for r in runs) for k in runs[0]}
    med["span_names"] = "|".join(sorted(names))
    return med


def main():
    components = ["qk_matmul", "softmax_scores", "av_matmul", "rmsnorm",
                  "qkv_proj", "o_proj", "residual_add"]
    dev = torch.device("rngd", 0)
    rows = []
    print(f"=== NPE={NPE} ({nd.get_device_configuration()[0]} PE fused) ===")
    for s in DEFAULT_S:
        for kind in components:
            try:
                m = measure(kind, s, dev)
                m.update(npe=NPE, name=kind, S=s)
                rows.append(m)
                print(f"[NPE={NPE} S={s}] {kind:16s} task={m['task']:.1f} "
                      f"tu={m['tu']:.1f} dma={m['dma']:.1f} "
                      f"parcopy={m['parallelcopy']:.1f} cluster={m['cluster']:.1f}",
                      flush=True)
            except Exception as exc:
                print(f"[NPE={NPE} S={s}] {kind} FAIL: {type(exc).__name__}: {exc}",
                      flush=True)
                rows.append(dict(npe=NPE, name=kind, S=s,
                                 error=f"{type(exc).__name__}: {exc}"))

    fields = ["npe", "name", "S", "task", "tu", "dma", "parallelcopy",
              "cluster", "span_names", "error"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n[DONE] {OUT_CSV} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
