#!/usr/bin/env python3
"""Prefill shape sweep for O projection -> residual add on RNGD.

Measures:
  isolated O projection + isolated residual add
  vs joint-compiled O projection -> residual add

O projection here includes the attention-output layout transform:
  (B, NH, S, HD) -> (B, S, D) -> MatMul(D, D)

Usage:
  python o_proj_residual_shape_sweep.py
  python o_proj_residual_shape_sweep.py 128,256,512,1024,2048
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

import torch
import furiosa.torch  # noqa: F401
from furiosa.torch.custom_ops import CompileModule

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "o_proj_residual_shape_sweep_results.csv")

B = 1
NH = 32
HD = 128
D = 4096
DTYPE = torch.bfloat16
N_RUNS = 3
DEFAULT_S = [128, 256, 512, 1024, 2048]


class OpChain(torch.nn.Module):
    def __init__(self, ops, weight=None, residual=None):
        super().__init__()
        self.ops = ops
        if weight is not None:
            self.register_buffer("weight", weight)
        if residual is not None:
            self.register_buffer("residual", residual)

    def forward(self, x):
        for op in self.ops:
            if op == "o_proj":
                bsz, n_heads, seq_len, head_dim = x.shape
                x = x.permute(0, 2, 1, 3).reshape(
                    bsz, seq_len, n_heads * head_dim
                )
                x = x @ self.weight
            elif op == "residual_add":
                x = x + self.residual
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


def measure(ops, input_shape, weight_shape, residual_shape, dev):
    weight = torch.randn(*weight_shape, dtype=DTYPE) if weight_shape else None
    residual = (
        torch.randn(*residual_shape, dtype=DTYPE) if residual_shape else None
    )
    x = torch.randn(*input_shape, dtype=DTYPE)
    cm = CompileModule.from_exported(
        torch.export.export(OpChain(ops, weight, residual), (x,))
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
        o_proj_input_shape = (B, NH, s, HD)
        residual_input_shape = (B, s, D)
        weight_shape = (D, D)
        residual_shape = (B, s, D)

        o_proj = measure(["o_proj"], o_proj_input_shape, weight_shape, None, dev)
        residual = measure(["residual_add"], residual_input_shape, None,
                           residual_shape, dev)
        fused = measure(["o_proj", "residual_add"], o_proj_input_shape,
                        weight_shape, residual_shape, dev)

        isolated_sum = o_proj["task_us"] + residual["task_us"]
        gap = fused["task_us"] - isolated_sum
        ratio = fused["task_us"] / isolated_sum if isolated_sum else 0.0

        row = dict(
            B=B,
            S=s,
            NH=NH,
            HD=HD,
            D=D,
            o_proj_task_us=round(o_proj["task_us"], 2),
            residual_task_us=round(residual["task_us"], 2),
            fused_task_us=round(fused["task_us"], 2),
            isolated_sum_us=round(isolated_sum, 2),
            composition_gap_us=round(gap, 2),
            fusion_ratio=round(ratio, 4),
            o_proj_dma_us=round(o_proj["dma_us"], 2),
            residual_dma_us=round(residual["dma_us"], 2),
            fused_dma_us=round(fused["dma_us"], 2),
            o_proj_tu_us=round(o_proj["tu_us"], 2),
            residual_tu_us=round(residual["tu_us"], 2),
            fused_tu_us=round(fused["tu_us"], 2),
        )
        rows.append(row)
        print(
            f"[S={s}] A+B={isolated_sum:.1f}us fused={fused['task_us']:.1f}us "
            f"gap={gap:+.1f}us ratio={ratio:.3f}",
            flush=True,
        )

    fields = [
        "B", "S", "NH", "HD", "D",
        "o_proj_task_us", "residual_task_us", "fused_task_us",
        "isolated_sum_us", "composition_gap_us", "fusion_ratio",
        "o_proj_dma_us", "residual_dma_us", "fused_dma_us",
        "o_proj_tu_us", "residual_tu_us", "fused_tu_us",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[DONE] {OUT_CSV} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
