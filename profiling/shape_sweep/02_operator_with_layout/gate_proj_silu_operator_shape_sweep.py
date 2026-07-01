#!/usr/bin/env python3
"""Operator-level shape sweep for gate projection and SiLU on RNGD.

Stage 2: measure isolated operator-level blocks for MatMul -> SiLU.
There is no reshape in this pair, but the names match the MLP operator path:
  gate projection -> activation

Usage:
  python gate_proj_silu_operator_shape_sweep.py
  python gate_proj_silu_operator_shape_sweep.py 128,256,512,1024,2048
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
OUT_CSV = os.path.join(HERE, "gate_proj_silu_operator_shape_sweep_results.csv")

B = 1
D = 4096
INTER = 14336
DTYPE = torch.bfloat16
N_RUNS = 3
DEFAULT_S = [128, 256, 512, 1024, 2048]


class OperatorBlock(torch.nn.Module):
    def __init__(self, kind, weight=None):
        super().__init__()
        self.kind = kind
        if weight is not None:
            self.register_buffer("weight", weight)

    def forward(self, x):
        if self.kind == "gate_proj":
            return x @ self.weight
        if self.kind == "silu":
            return F.silu(x)
        raise ValueError(self.kind)


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


def measure(kind, input_shape, weight_shape, dev):
    weight = torch.randn(*weight_shape, dtype=DTYPE) if weight_shape else None
    x = torch.randn(*input_shape, dtype=DTYPE)
    cm = CompileModule.from_exported(
        torch.export.export(OperatorBlock(kind, weight), (x,))
    )
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
        runs.append((
            task,
            union_us(by_name.get("Renegade::TuExec", [])),
            union_us(by_name.get("DMA", [])),
        ))

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
        gate_proj = measure("gate_proj", (B, s, D), (D, INTER), dev)
        silu = measure("silu", (B, s, INTER), None, dev)
        row = dict(
            B=B, S=s, D=D, INTER=INTER,
            gate_proj_task_us=round(gate_proj["task_us"], 2),
            silu_task_us=round(silu["task_us"], 2),
            isolated_sum_us=round(gate_proj["task_us"] + silu["task_us"], 2),
            gate_proj_dma_us=round(gate_proj["dma_us"], 2),
            silu_dma_us=round(silu["dma_us"], 2),
            gate_proj_tu_us=round(gate_proj["tu_us"], 2),
            silu_tu_us=round(silu["tu_us"], 2),
        )
        rows.append(row)
        print(
            f"[S={s}] gate_proj={gate_proj['task_us']:.1f}us "
            f"silu={silu['task_us']:.1f}us sum={row['isolated_sum_us']:.1f}us",
            flush=True,
        )

    fields = [
        "B", "S", "D", "INTER",
        "gate_proj_task_us", "silu_task_us", "isolated_sum_us",
        "gate_proj_dma_us", "silu_dma_us", "gate_proj_tu_us", "silu_tu_us",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[DONE] {OUT_CSV} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
