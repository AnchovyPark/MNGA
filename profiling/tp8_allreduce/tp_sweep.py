#!/usr/bin/env python3
"""TP=1 vs TP=8 latency sweep on RNGD — isolate the collective (all-reduce) cost.

RNGD 의 tensor parallelism 은 8개 PE 를 하나의 device 로 융합(set_fusion(8))하고,
reduce/통신을 네이티브 컴파일러가 on-chip PE interconnect 로 처리한다.
FX 레벨 all_reduce 노드는 생기지 않는다(README 참고). 대신 프로파일러 span 에
TP=8 에서만 나타나는 `Renegade::ParallelCopy`(PE 간 데이터 이동 = collective) 와
`Cluster` span 이 등장한다. 이걸로 all-reduce 비용을 분리 측정한다.

set_fusion(num_pe) 은 프로세스당 1회만 호출 가능 → 스크립트를 NPE 인자로 두 번
실행(1, 8)하고 offline 으로 diff 한다.

사용법:
  python tp_sweep.py 1
  python tp_sweep.py 8
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
import furiosa.torch  # noqa: F401
from furiosa.torch import native_device as nd

NPE = int(sys.argv[1]) if len(sys.argv) > 1 else 8
nd.set_fusion(NPE)  # 전역 1회. 반드시 device 초기화/compile 전에.

from furiosa.torch.custom_ops import CompileModule  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, f"tp_sweep_np{NPE}_results.csv")

# Llama-3.1-8B-like
D, NH, HD, INTER = 4096, 32, 128, 14336
DTYPE = torch.bfloat16
N_RUNS = 3
B, S = 1, 512  # prefill-ish


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


# ---- workloads (한 그래프로 compile → 네이티브가 TP 로 쪼갬) ----
class MatmulDD(torch.nn.Module):
    """o_proj 류 단일 GEMM (D->D)."""
    def __init__(self):
        super().__init__()
        self.register_buffer("w", torch.randn(D, D, dtype=DTYPE))

    def forward(self, x):
        return x @ self.w


class MLP(torch.nn.Module):
    """up(D->INTER) -> silu -> down(INTER->D). row-parallel down 뒤에 reduce 필요."""
    def __init__(self):
        super().__init__()
        self.register_buffer("w1", torch.randn(D, INTER, dtype=DTYPE))
        self.register_buffer("w2", torch.randn(INTER, D, dtype=DTYPE))

    def forward(self, x):
        return F.silu(x @ self.w1) @ self.w2


class Attn(torch.nn.Module):
    """qkv proj -> sdpa(eager) -> o_proj. head-parallel + o_proj reduce."""
    def __init__(self):
        super().__init__()
        self.register_buffer("wq", torch.randn(D, D, dtype=DTYPE))
        self.register_buffer("wk", torch.randn(D, D, dtype=DTYPE))
        self.register_buffer("wv", torch.randn(D, D, dtype=DTYPE))
        self.register_buffer("wo", torch.randn(D, D, dtype=DTYPE))

    def forward(self, x):
        b, s, _ = x.shape
        q = (x @ self.wq).view(b, s, NH, HD).permute(0, 2, 1, 3)
        k = (x @ self.wk).view(b, s, NH, HD).permute(0, 2, 1, 3)
        v = (x @ self.wv).view(b, s, NH, HD).permute(0, 2, 1, 3)
        scores = torch.softmax((q @ k.transpose(-1, -2)) * (HD ** -0.5), dim=-1)
        ctx = (scores @ v).permute(0, 2, 1, 3).reshape(b, s, D)
        return ctx @ self.wo


def measure(mod, in_shape):
    dev = torch.device("rngd", 0)
    x = torch.randn(*in_shape, dtype=DTYPE)
    cm = CompileModule.from_exported(torch.export.export(mod, (x,)))
    cm.to(dev)
    xd = x.to(dev)
    cm(xd, profiles=None, device=dev)

    runs = []
    span_names = set()
    for _ in range(N_RUNS):
        profiles = cm.generate_profiles(dev)
        cm(xd, profiles=profiles, device=dev)
        di = [[p.device.index for p in inner] for inner in profiles]
        pc = [[p.cpu() for p in inner] for inner in profiles]
        spans = cm.edf.npu_node.build_tuc_profile_spans(pc, di, 10**9)
        by = defaultdict(list)
        task = 0.0
        for e in spans:
            span_names.add(e.name)
            if e.name == "Task":
                task += e.time_range.elapsed_us()
            else:
                by[e.name].append((e.time_range.start, e.time_range.end))
        runs.append({
            "task": task,
            "tu": union_us(by.get("Renegade::TuExec", [])),
            "dma": union_us(by.get("DMA", [])),
            # TP=8 에서만: PE 간 통신(collective) / cluster wrapper
            "parallelcopy": union_us(by.get("Renegade::ParallelCopy", [])),
            "cluster": union_us(by.get("Cluster", [])),
            "stotrf": union_us(by.get("Renegade::StoTrf", [])),
        })
    med = {k: st.median(r[k] for r in runs) for k in runs[0]}
    med["span_names"] = "|".join(sorted(span_names))
    return med


def main():
    workloads = [
        ("matmul_DxD", MatmulDD(), (B, S, D)),
        ("mlp", MLP(), (B, S, D)),
        ("attn", Attn(), (B, S, D)),
    ]
    rows = []
    print(f"=== NPE={NPE}  device_config={nd.get_device_configuration()[0]} PE fused ===")
    for name, mod, shape in workloads:
        try:
            m = measure(mod, shape)
            m.update(name=name, npe=NPE, B=B, S=S)
            rows.append(m)
            print(f"[NPE={NPE}] {name:12s} task={m['task']:.1f} tu={m['tu']:.1f} "
                  f"dma={m['dma']:.1f} parcopy={m['parallelcopy']:.1f} "
                  f"cluster={m['cluster']:.1f} | task-tu={m['task']-m['tu']:.1f}",
                  flush=True)
        except Exception as exc:
            print(f"[NPE={NPE}] {name} FAIL: {type(exc).__name__}: {exc}", flush=True)
            rows.append(dict(name=name, npe=NPE, error=f"{type(exc).__name__}: {exc}"))

    fields = ["npe", "name", "B", "S", "task", "tu", "dma", "parallelcopy",
              "cluster", "stotrf", "span_names", "error"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n[DONE] {OUT_CSV} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
