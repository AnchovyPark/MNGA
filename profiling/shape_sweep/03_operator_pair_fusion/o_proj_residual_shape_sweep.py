#!/usr/bin/env python3
"""Prefill shape sweep for O projection -> residual add on RNGD.

Stage 3: measure only the joint operator pair.
Use stage 2 outputs to compute composition gap offline.

O projection here includes the attention-output layout transform:
  (B, NH, S, HD) -> (B, S, D) -> MatMul(D, D)

Usage:
  python o_proj_residual_shape_sweep.py
  python o_proj_residual_shape_sweep.py 128,256,512,1024,2048
  python o_proj_residual_shape_sweep.py "1,32,512,128;1,16,512,128"
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
D = NH * HD
DTYPE = torch.bfloat16
N_RUNS = 3
DEFAULT_CONFIGS = [
    # B, NH, S, HD
    (1, 32, 128, 128),
    (1, 32, 512, 128),
    (1, 32, 2048, 128),
    (1, 16, 512, 128),
    (1, 64, 512, 128),
    (1, 32, 512, 64),
    (1, 32, 512, 256),
    (2, 32, 512, 128),
    (4, 32, 512, 128),
]


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


def parse_configs():
    if len(sys.argv) <= 1:
        return DEFAULT_CONFIGS
    arg = sys.argv[1]
    if ";" in arg:
        return [tuple(int(x) for x in item.split(",")) for item in arg.split(";") if item]
    return [(B, NH, int(s), HD) for s in arg.split(",") if s]


def main():
    dev = torch.device("rngd", 0)
    rows = []
    for b, nh, s, hd in parse_configs():
        d = nh * hd
        o_proj_input_shape = (b, nh, s, hd)
        weight_shape = (d, d)
        residual_shape = (b, s, d)

        fused = measure(["o_proj", "residual_add"], o_proj_input_shape,
                        weight_shape, residual_shape, dev)

        row = dict(
            B=b,
            S=s,
            NH=nh,
            HD=hd,
            D=d,
            pair="o_proj_to_residual_add",
            pair_task_us=round(fused["task_us"], 2),
            pair_dma_us=round(fused["dma_us"], 2),
            pair_tu_us=round(fused["tu_us"], 2),
        )
        rows.append(row)
        print(
            f"[B={b} NH={nh} S={s} HD={hd}] "
            f"o_proj_to_residual_add={fused['task_us']:.1f}us",
            flush=True,
        )

    fields = [
        "B", "S", "NH", "HD", "D", "pair",
        "pair_task_us", "pair_dma_us", "pair_tu_us",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[DONE] {OUT_CSV} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
