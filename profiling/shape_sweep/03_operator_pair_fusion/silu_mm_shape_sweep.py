#!/usr/bin/env python3
"""Prefill shape sweep for SiLU -> MatMul composition on RNGD.

Measures:
  isolated SiLU + isolated MatMul vs joint-compiled SiLU -> MatMul

Default sweep changes sequence length S while keeping Llama-3.1-8B-like
hidden dimensions:
  INTER = 14336
  D = 4096

Usage:
  python silu_mm_shape_sweep.py
  python silu_mm_shape_sweep.py 128,256,512,1024,2048
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
import furiosa.torch  # noqa: F401
from furiosa.torch.custom_ops import CompileModule

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "silu_mm_shape_sweep_results.csv")

B = 1
D = 4096
INTER = 14336
DTYPE = torch.bfloat16
N_RUNS = 3
DEFAULT_S = [128, 256, 512, 1024, 2048]


class OpChain(torch.nn.Module):
    def __init__(self, ops, weight=None):
        super().__init__()
        self.ops = ops
        if weight is not None:
            self.register_buffer("weight", weight)

    def forward(self, x):
        for op in self.ops:
            if op == "silu":
                x = F.silu(x)
            elif op == "matmul":
                x = x @ self.weight
            else:
                raise ValueError(op)
        return x


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


def measure(ops, input_shape, weight_shape, dev):
    weight = torch.randn(*weight_shape, dtype=DTYPE) if weight_shape else None
    x = torch.randn(*input_shape, dtype=DTYPE)
    cm = CompileModule.from_exported(torch.export.export(OpChain(ops, weight), (x,)))
    cm.to(dev)
    xd = x.to(dev)
    cm(xd, profiles=None, device=dev)

    runs = []
    for _ in range(N_RUNS):
        profiles = cm.generate_profiles(dev)
        cm(xd, profiles=profiles, device=dev)
        device_indices = [[p.device.index for p in inner] for inner in profiles]
        profiles_cpu = [[p.cpu() for p in inner] for inner in profiles]
        spans = cm.edf.npu_node.build_tuc_profile_spans(
            profiles_cpu, device_indices, 10**9
        )
        by_name = defaultdict(list)
        task = 0.0
        for event in spans:
            if event.name == "Task":
                task += event.time_range.elapsed_us()
            else:
                by_name[event.name].append(
                    (event.time_range.start, event.time_range.end)
                )
        tu = union_us(by_name.get("Renegade::TuExec", []))
        dma = union_us(by_name.get("DMA", []))
        runs.append((task, tu, dma))

    return {
        "task_us": st.median(r[0] for r in runs),
        "tu_us": st.median(r[1] for r in runs),
        "dma_us": st.median(r[2] for r in runs),
    }


def main():
    seq_lens = DEFAULT_S
    if len(sys.argv) > 1:
        seq_lens = [int(x) for x in sys.argv[1].split(",") if x]

    dev = torch.device("rngd", 0)
    rows = []
    for s in seq_lens:
        input_shape = (B, s, INTER)
        weight_shape = (INTER, D)

        silu = measure(["silu"], input_shape, None, dev)
        matmul = measure(["matmul"], input_shape, weight_shape, dev)
        fused = measure(["silu", "matmul"], input_shape, weight_shape, dev)

        isolated_sum = silu["task_us"] + matmul["task_us"]
        gap = fused["task_us"] - isolated_sum
        ratio = fused["task_us"] / isolated_sum if isolated_sum else 0.0

        row = dict(
            B=B,
            S=s,
            INTER=INTER,
            D=D,
            silu_task_us=round(silu["task_us"], 2),
            matmul_task_us=round(matmul["task_us"], 2),
            fused_task_us=round(fused["task_us"], 2),
            isolated_sum_us=round(isolated_sum, 2),
            composition_gap_us=round(gap, 2),
            fusion_ratio=round(ratio, 4),
            silu_dma_us=round(silu["dma_us"], 2),
            matmul_dma_us=round(matmul["dma_us"], 2),
            fused_dma_us=round(fused["dma_us"], 2),
            silu_tu_us=round(silu["tu_us"], 2),
            matmul_tu_us=round(matmul["tu_us"], 2),
            fused_tu_us=round(fused["tu_us"], 2),
        )
        rows.append(row)
        print(
            f"[S={s}] A+B={isolated_sum:.1f}us fused={fused['task_us']:.1f}us "
            f"gap={gap:+.1f}us ratio={ratio:.3f}",
            flush=True,
        )

    fields = [
        "B", "S", "INTER", "D",
        "silu_task_us", "matmul_task_us", "fused_task_us",
        "isolated_sum_us", "composition_gap_us", "fusion_ratio",
        "silu_dma_us", "matmul_dma_us", "fused_dma_us",
        "silu_tu_us", "matmul_tu_us", "fused_tu_us",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[DONE] {OUT_CSV} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
