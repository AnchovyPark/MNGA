#!/usr/bin/env python3
"""Single-kernel/component sweep for attention-chain shapes on RNGD.

This provides isolated measurements for the components used by:

  - qk_softmax_av
  - rmsnorm_qkv_proj
  - softmax_av_oproj_residual

Usage:
  python attention_single_kernel_shape_sweep.py
  python attention_single_kernel_shape_sweep.py 128,512,2048
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
OUT_CSV = os.path.join(HERE, "attention_single_kernel_shape_sweep_results.csv")

B = 1
NH = 32
HD = 128
D = NH * HD
DTYPE = torch.bfloat16
N_RUNS = 3
DEFAULT_S = [128, 512, 2048]


class SingleComponent(torch.nn.Module):
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
        elif kind == "softmax_scores":
            pass
        else:
            raise ValueError(kind)

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


def input_shape_for(kind, s):
    if kind == "qk_matmul":
        return (B, NH, s, HD)
    if kind in ("softmax_scores", "av_matmul"):
        return (B, NH, s, s)
    if kind in ("rmsnorm", "qkv_proj", "residual_add"):
        return (B, s, D)
    if kind == "o_proj":
        return (B, NH, s, HD)
    raise ValueError(kind)


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


def measure(kind, s, dev):
    x = torch.randn(*input_shape_for(kind, s), dtype=DTYPE)
    cm = CompileModule.from_exported(
        torch.export.export(SingleComponent(kind, s), (x,))
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

    components = [
        "qk_matmul",
        "softmax_scores",
        "av_matmul",
        "rmsnorm",
        "qkv_proj",
        "o_proj",
        "residual_add",
    ]
    dev = torch.device("rngd", 0)
    rows = []
    for s in seq_lens:
        for component in components:
            result = measure(component, s, dev)
            row = dict(
                B=B,
                S=s,
                NH=NH,
                HD=HD,
                D=D,
                component=component,
                task_us=round(result["task_us"], 2),
                dma_us=round(result["dma_us"], 2),
                tu_us=round(result["tu_us"], 2),
            )
            rows.append(row)
            print(f"[S={s}] {component}={result['task_us']:.1f}us", flush=True)

    fields = ["B", "S", "NH", "HD", "D", "component", "task_us", "dma_us", "tu_us"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[DONE] {OUT_CSV} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
