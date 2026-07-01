#!/usr/bin/env python3
"""Attention 멀티커널 체인을 TP=1 vs TP=8 로 측정.

가설: attention 은 head-parallel 이라 TP=8 이면 NH=32 를 8 PE 에 4헤드씩 분산 →
score (B,NH,S,S) 가 PE 당 (B,4,S,S) 로 쪼개져 SRAM 압박이 1/8. S=2048 에서 TP=1 때
나타난 심각한 anti-fusion 이 TP=8 에선 완화/해소되는지 본다.

set_fusion(NPE) 은 프로세스당 1회 → NPE 인자로 두 번 실행(1, 8) 후 offline diff.
격리 컴포넌트 + 체인을 같은 프로세스에서 재서 TP=8 gap 도 계산 가능.

사용법:
  python attention_chain_tp.py 1
  python attention_chain_tp.py 8
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
OUT_CSV = os.path.join(HERE, f"attention_chain_tp_np{NPE}_results.csv")

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
        elif kind == "o_proj":
            self.register_buffer("weight", torch.randn(D, D, dtype=DTYPE))

    def forward(self, x):
        if self.kind == "qk_matmul":
            return x @ self.k
        if self.kind == "softmax_scores":
            return torch.softmax(x, dim=-1)
        if self.kind == "av_matmul":
            return x @ self.v
        if self.kind == "o_proj":
            y = x.permute(0, 2, 1, 3).reshape(B, self.s, D)
            return y @ self.weight
        raise ValueError(self.kind)


class Chain(torch.nn.Module):
    def __init__(self, chain, s):
        super().__init__()
        self.chain = chain
        self.s = s
        if chain == "qk_softmax_av":
            self.register_buffer("k", torch.randn(B, NH, HD, s, dtype=DTYPE))
            self.register_buffer("v", torch.randn(B, NH, s, HD, dtype=DTYPE))
        elif chain == "softmax_av_oproj_residual":
            self.register_buffer("v", torch.randn(B, NH, s, HD, dtype=DTYPE))
            self.register_buffer("o_weight", torch.randn(D, D, dtype=DTYPE))
            self.register_buffer("residual", torch.randn(B, s, D, dtype=DTYPE))

    def forward(self, x):
        if self.chain == "qk_softmax_av":
            x = x @ self.k
            x = torch.softmax(x, dim=-1)
            return x @ self.v
        if self.chain == "softmax_av_oproj_residual":
            y = torch.softmax(x, dim=-1)
            y = y @ self.v
            y = y.permute(0, 2, 1, 3).reshape(B, self.s, D)
            y = y @ self.o_weight
            return y + self.residual
        raise ValueError(self.chain)


def shape_for(name, s):
    if name in ("qk_matmul",):
        return (B, NH, s, HD)
    if name in ("softmax_scores", "av_matmul", "softmax_av_oproj_residual"):
        return (B, NH, s, s)
    if name in ("o_proj",):
        return (B, NH, s, HD)
    if name == "qk_softmax_av":
        return (B, NH, s, HD)
    raise ValueError(name)


def measure(mod, in_shape, dev):
    x = torch.randn(*in_shape, dtype=DTYPE)
    cm = CompileModule.from_exported(torch.export.export(mod, (x,)))
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
    components = ["qk_matmul", "softmax_scores", "av_matmul", "o_proj"]
    chains = ["qk_softmax_av", "softmax_av_oproj_residual"]
    dev = torch.device("rngd", 0)
    rows = []
    print(f"=== NPE={NPE} ({nd.get_device_configuration()[0]} PE fused) ===")
    for s in DEFAULT_S:
        for name in components + chains:
            kind = "chain" if name in chains else "component"
            mod = Chain(name, s) if name in chains else Component(name, s)
            try:
                m = measure(mod, shape_for(name, s), dev)
                m.update(npe=NPE, kind=kind, name=name, S=s)
                rows.append(m)
                print(f"[NPE={NPE} S={s}] {name:26s} task={m['task']:.1f} "
                      f"tu={m['tu']:.1f} parcopy={m['parallelcopy']:.1f} "
                      f"cluster={m['cluster']:.1f}", flush=True)
            except Exception as exc:
                print(f"[NPE={NPE} S={s}] {name} FAIL: {type(exc).__name__}: {exc}",
                      flush=True)
                rows.append(dict(npe=NPE, kind=kind, name=name, S=s,
                                 error=f"{type(exc).__name__}: {exc}"))

    fields = ["npe", "kind", "name", "S", "task", "tu", "dma",
              "parallelcopy", "cluster", "span_names", "error"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n[DONE] {OUT_CSV} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
