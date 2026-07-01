#!/usr/bin/env python3
"""Profile isolated inference-like operator blocks on RNGD.

This is the first rung of the operator composition ladder:
measure one operator-level block at a time, using the same low-level
CompileModule -> generate_profiles -> build_tuc_profile_spans path used by
the existing MNGA profiling scripts.

Usage:
  python operator_blocks.py [decode|prefill|all]
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
OUT_CSV = os.path.join(HERE, "operator_blocks_results.csv")

# Llama-3.1-8B-like shapes.
D, NH, HD, INTER = 4096, 32, 128, 14336
DTYPE = torch.bfloat16
N_RUNS = 3
REGIME = sys.argv[1] if len(sys.argv) > 1 else "all"


class OperatorBlock(torch.nn.Module):
    def __init__(self, kind, aux=None):
        super().__init__()
        self.kind = kind
        if aux is not None:
            self.register_buffer("aux", aux)

    def forward(self, x):
        if self.kind == "rmsnorm":
            v = x.float()
            v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-5)
            return (v * self.aux.float()).to(x.dtype)
        if self.kind in ("q_proj", "gate_proj", "down_proj"):
            return x @ self.aux
        if self.kind == "activation":
            return F.silu(x)
        if self.kind == "qk_scores":
            return x @ self.aux
        if self.kind == "softmax":
            return torch.softmax(x, dim=-1)
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


def measure(name, kind, input_shape, aux_shape, dev):
    aux = torch.randn(*aux_shape, dtype=DTYPE) if aux_shape else None
    x = torch.randn(*input_shape, dtype=DTYPE)
    mod = OperatorBlock(kind, aux)
    cm = CompileModule.from_exported(torch.export.export(mod, (x,)))
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
            elapsed = event.time_range.elapsed_us()
            if event.name == "Task":
                task += elapsed
            else:
                by_name[event.name].append(
                    (event.time_range.start, event.time_range.end)
                )
        tu = union_us(by_name.get("Renegade::TuExec", []))
        dma = union_us(by_name.get("DMA", []))
        all_intervals = [iv for intervals in by_name.values() for iv in intervals]
        envelope = (
            max(e for _, e in all_intervals) - min(s for s, _ in all_intervals)
            if all_intervals else 0.0
        )
        runs.append((task or envelope, tu, dma, envelope))

    return {
        "name": name,
        "kind": kind,
        "task_us": st.median(r[0] for r in runs),
        "tu_us": st.median(r[1] for r in runs),
        "dma_us": st.median(r[2] for r in runs),
        "envelope_us": st.median(r[3] for r in runs),
    }


def blocks_for(regime):
    if regime == "decode":
        B, S, CTX = 8, 1, 2048
    else:
        B, S, CTX = 1, 2048, 2048
    return [
        ("rmsnorm", "rmsnorm", (B, S, D), (D,)),
        ("q_proj", "q_proj", (B, S, D), (D, D)),
        ("gate_proj", "gate_proj", (B, S, D), (D, INTER)),
        ("qk_scores", "qk_scores", (B, NH, S, HD), (B, NH, HD, CTX)),
        ("softmax_scores", "softmax", (B, NH, S, CTX), None),
    ]


def main():
    dev = torch.device("rngd", 0)
    regimes = []
    if REGIME in ("decode", "all"):
        regimes.append("decode")
    if REGIME in ("prefill", "all"):
        regimes.append("prefill")

    rows = []
    for regime in regimes:
        for name, kind, input_shape, aux_shape in blocks_for(regime):
            try:
                row = measure(name, kind, input_shape, aux_shape, dev)
                row.update(regime=regime, input_shape=str(input_shape),
                           aux_shape=str(aux_shape))
                rows.append(row)
                print(
                    f"[{regime}] {name:15s} task={row['task_us']:.1f}us "
                    f"tu={row['tu_us']:.1f}us dma={row['dma_us']:.1f}us",
                    flush=True,
                )
            except Exception as exc:
                rows.append(dict(regime=regime, name=name, kind=kind,
                                 error=f"{type(exc).__name__}: {exc}"))
                print(f"[{regime}] {name} FAIL: {type(exc).__name__}: {exc}",
                      flush=True)

    fields = [
        "regime", "name", "kind", "input_shape", "aux_shape",
        "task_us", "tu_us", "dma_us", "envelope_us", "error",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[DONE] {OUT_CSV} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
