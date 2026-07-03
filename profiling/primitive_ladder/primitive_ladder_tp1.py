#!/usr/bin/env python3
"""Primitive composition ladder for TP=1.

Pilot experiment:
  - primitives: M=MatMul, A=Add, S=Softmax
  - shape: [B, T, D] -> [B, T, D]
  - default workload depth: Single and Double only

Usage:
  python primitive_ladder_tp1.py
  python primitive_ladder_tp1.py --max-len 2 --T 512 --D 1024 --runs 3
"""
import argparse
import csv
import itertools
import os
import statistics as st
from collections import defaultdict

import torch
import furiosa.torch  # noqa: F401
from furiosa.torch import native_device as nd

nd.set_fusion(1)

from furiosa.torch.custom_ops import CompileModule  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
OPS = ("M", "A", "S")
DTYPE = torch.bfloat16


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


class PrimitiveChain(torch.nn.Module):
    def __init__(self, sequence, b, t, d):
        super().__init__()
        self.sequence = sequence
        for i, op in enumerate(sequence):
            if op == "M":
                self.register_buffer(f"weight_{i}", torch.randn(d, d, dtype=DTYPE))
            elif op == "A":
                self.register_buffer(f"residual_{i}", torch.randn(b, t, d, dtype=DTYPE))

    def forward(self, x):
        for i, op in enumerate(self.sequence):
            if op == "M":
                x = x @ getattr(self, f"weight_{i}")
            elif op == "A":
                x = x + getattr(self, f"residual_{i}")
            elif op == "S":
                x = torch.softmax(x, dim=-1)
            else:
                raise ValueError(op)
        return x


def measure(sequence, args, dev):
    x = torch.randn(args.B, args.T, args.D, dtype=DTYPE)
    module = PrimitiveChain(sequence, args.B, args.T, args.D)
    cm = CompileModule.from_exported(torch.export.export(module, (x,)))
    cm.to(dev)
    xd = x.to(dev)
    cm(xd, profiles=None, device=dev)

    runs = []
    span_names = set()
    for _ in range(args.runs):
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
            span_names.add(event.name)
            if event.name == "Task":
                task += event.time_range.elapsed_us()
            else:
                by_name[event.name].append(
                    (event.time_range.start, event.time_range.end)
                )
        runs.append({
            "task_us": task,
            "tu_us": union_us(by_name.get("Renegade::TuExec", [])),
            "dma_us": union_us(by_name.get("DMA", [])),
            "sto_trf_us": union_us(by_name.get("Renegade::StoTrf", [])),
            "parallelcopy_us": union_us(by_name.get("Renegade::ParallelCopy", [])),
            "cluster_us": union_us(by_name.get("Cluster", [])),
        })

    med = {k: st.median(r[k] for r in runs) for k in runs[0]}
    med["span_names"] = "|".join(sorted(span_names))
    return med


def workload_kind(length):
    if length == 1:
        return "single"
    if length == 2:
        return "double"
    if length == 3:
        return "triple"
    if length == 4:
        return "quad"
    return f"len{length}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--T", type=int, default=512)
    parser.add_argument("--D", type=int, default=1024)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=2)
    parser.add_argument(
        "--out",
        default=os.path.join(HERE, "primitive_ladder_tp1_medium_results.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dev = torch.device("rngd", 0)
    rows = []

    print(
        f"=== primitive ladder TP=1 B={args.B} T={args.T} D={args.D} "
        f"max_len={args.max_len} runs={args.runs} ===",
        flush=True,
    )
    for length in range(1, args.max_len + 1):
        for seq_tuple in itertools.product(OPS, repeat=length):
            sequence = "".join(seq_tuple)
            row = {
                "npe": 1,
                "B": args.B,
                "T": args.T,
                "D": args.D,
                "dtype": "bfloat16",
                "kind": workload_kind(length),
                "length": length,
                "sequence": sequence,
                "ops": "->".join(seq_tuple),
                "error": "",
            }
            try:
                metrics = measure(sequence, args, dev)
                row.update(metrics)
                print(
                    f"[{row['kind']:6s} {sequence:4s}] "
                    f"task={row['task_us']:.2f}us "
                    f"tu={row['tu_us']:.2f}us dma={row['dma_us']:.2f}us",
                    flush=True,
                )
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[{row['kind']:6s} {sequence:4s}] FAIL {row['error']}", flush=True)
            rows.append(row)

    fields = [
        "npe", "B", "T", "D", "dtype", "kind", "length", "sequence", "ops",
        "task_us", "tu_us", "dma_us", "sto_trf_us", "parallelcopy_us",
        "cluster_us", "span_names", "error",
    ]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[DONE] {args.out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
