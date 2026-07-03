#!/usr/bin/env python3
"""Llama-3.1-8B S=128 contextual-window profiler.

This measures Llama-shaped contiguous windows instead of only prefix chains.
The goal is to produce enough local context data to test whether Single/Double/
Triple windows can predict Quad windows at real Llama tensor shapes.

Families:
  MLP tail:  G=gate_proj, I=silu, X=mul(up_act), D=down_proj, R=residual
  ATTN tail: Q=qk_scores, S=softmax, A=av, O=o_proj, R=residual

Examples:
  python llama_context_windows_s128.py --max-len 4 --runs 3
  python llama_context_windows_s128.py --families mlp_tail attn_tail --list-only
"""
import argparse
import csv
import os
import statistics as st
from collections import defaultdict

import torch
import torch.nn.functional as F
import furiosa.torch  # noqa: F401
from furiosa.torch import native_device as nd

nd.set_fusion(8)

from furiosa.torch.custom_ops import CompileModule  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))

D, NH, KV, HD, INTER, L, VOCAB = 4096, 32, 8, 128, 14336, 32, 128256
B, S = 1, 128
DTYPE = torch.bfloat16

SHAPES = {
    "hidden": (B, S, D),
    "inter": (B, S, INTER),
    "attn_q": (B, NH, S, HD),
    "attn_scores": (B, NH, S, S),
    "attn_ctx": (B, NH, S, HD),
}

MLP_OPS = ("G", "I", "X", "D", "R")
ATTN_OPS = ("Q", "S", "A", "O", "R")

OP_NAMES = {
    "G": "gate_proj",
    "I": "silu",
    "X": "mul_up_act",
    "D": "down_proj",
    "R": "residual_add",
    "Q": "qk_scores",
    "S": "softmax",
    "A": "av",
    "O": "o_proj",
}


def union_us(intervals):
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    cur_s, cur_e = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_e:
            cur_e = max(cur_e, end)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = start, end
    return total + (cur_e - cur_s)


def workload_kind(length):
    return {
        1: "single",
        2: "double",
        3: "triple",
        4: "quad",
    }.get(length, f"len{length}")


def input_shape_for(family, first_op):
    if family == "mlp_tail":
        if first_op == "G":
            return SHAPES["hidden"]
        if first_op in ("I", "X", "D"):
            return SHAPES["inter"]
        if first_op == "R":
            return SHAPES["hidden"]
    if family == "attn_tail":
        if first_op == "Q":
            return SHAPES["attn_q"]
        if first_op in ("S", "A"):
            return SHAPES["attn_scores"]
        if first_op == "O":
            return SHAPES["attn_ctx"]
        if first_op == "R":
            return SHAPES["hidden"]
    raise ValueError(f"unknown family/op: {family}/{first_op}")


class ContextWindow(torch.nn.Module):
    def __init__(self, family, sequence):
        super().__init__()
        self.family = family
        self.sequence = sequence

        if family == "mlp_tail":
            if "G" in sequence:
                self.register_buffer("wg", torch.randn(D, INTER, dtype=DTYPE))
            if "X" in sequence:
                self.register_buffer("up_act", torch.randn(B, S, INTER, dtype=DTYPE))
            if "D" in sequence:
                self.register_buffer("wd", torch.randn(INTER, D, dtype=DTYPE))
            if "R" in sequence:
                self.register_buffer("res", torch.randn(B, S, D, dtype=DTYPE))
        elif family == "attn_tail":
            if "Q" in sequence:
                self.register_buffer("k", torch.randn(B, NH, HD, S, dtype=DTYPE))
            if "A" in sequence:
                self.register_buffer("v", torch.randn(B, NH, S, HD, dtype=DTYPE))
            if "O" in sequence:
                self.register_buffer("wo", torch.randn(NH * HD, D, dtype=DTYPE))
            if "R" in sequence:
                self.register_buffer("res", torch.randn(B, S, D, dtype=DTYPE))
        else:
            raise ValueError(family)

    def forward(self, x):
        for op in self.sequence:
            if self.family == "mlp_tail":
                if op == "G":
                    x = x @ self.wg
                elif op == "I":
                    x = F.silu(x)
                elif op == "X":
                    x = x * self.up_act
                elif op == "D":
                    x = x @ self.wd
                elif op == "R":
                    x = x + self.res
                else:
                    raise ValueError(op)
            elif self.family == "attn_tail":
                if op == "Q":
                    x = x @ self.k
                elif op == "S":
                    x = torch.softmax(x, -1)
                elif op == "A":
                    x = x @ self.v
                elif op == "O":
                    x = x.permute(0, 2, 1, 3).reshape(B, S, NH * HD) @ self.wo
                elif op == "R":
                    x = x + self.res
                else:
                    raise ValueError(op)
        return x


class MLPFullBlock(torch.nn.Module):
    def __init__(self, with_residual=True):
        super().__init__()
        self.with_residual = with_residual
        self.register_buffer("wg", torch.randn(D, INTER, dtype=DTYPE))
        self.register_buffer("wu", torch.randn(D, INTER, dtype=DTYPE))
        self.register_buffer("wd", torch.randn(INTER, D, dtype=DTYPE))
        if with_residual:
            self.register_buffer("res", torch.randn(B, S, D, dtype=DTYPE))

    def forward(self, x):
        gate = x @ self.wg
        up = x @ self.wu
        x = F.silu(gate) * up
        x = x @ self.wd
        if self.with_residual:
            x = x + self.res
        return x


def build_workloads(args):
    workloads = []
    families = set(args.families)
    if "mlp_tail" in families:
        workloads.extend(window_workloads("mlp_tail", MLP_OPS, args.min_len, args.max_len))
    if "attn_tail" in families:
        workloads.extend(window_workloads("attn_tail", ATTN_OPS, args.min_len, args.max_len))
    if args.include_blocks and "mlp_tail" in families:
        workloads.append(("mlp_full", "MLP", "full_mlp_no_residual", "G+U->I->X->D", "hidden"))
        workloads.append(("mlp_full", "MLPR", "full_mlp_residual", "G+U->I->X->D->R", "hidden"))
    return workloads


def window_workloads(family, ops, min_len, max_len):
    workloads = []
    for length in range(min_len, max_len + 1):
        for start in range(0, len(ops) - length + 1):
            seq = "".join(ops[start:start + length])
            names = "->".join(OP_NAMES[op] for op in seq)
            shape_name = input_shape_name_for(family, seq[0])
            workloads.append((family, seq, workload_kind(length), names, shape_name))
    return workloads


def input_shape_name_for(family, first_op):
    if family == "mlp_tail":
        if first_op == "G":
            return "hidden"
        if first_op in ("I", "X", "D"):
            return "inter"
        if first_op == "R":
            return "hidden"
    if family == "attn_tail":
        if first_op == "Q":
            return "attn_q"
        if first_op in ("S", "A"):
            return "attn_scores"
        if first_op == "O":
            return "attn_ctx"
        if first_op == "R":
            return "hidden"
    raise ValueError(f"unknown family/op: {family}/{first_op}")


def make_module(family, sequence):
    if family == "mlp_full":
        return MLPFullBlock(with_residual=(sequence == "MLPR")), torch.randn(
            *SHAPES["hidden"], dtype=DTYPE
        )
    return ContextWindow(family, sequence), torch.randn(
        *input_shape_for(family, sequence[0]), dtype=DTYPE
    )


def measure(family, sequence, args, dev):
    module, x = make_module(family, sequence)
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

    med = {key: st.median(run[key] for run in runs) for key in runs[0]}
    med["span_names"] = "|".join(sorted(span_names))
    return med


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=4)
    parser.add_argument(
        "--families",
        nargs="+",
        default=["mlp_tail", "attn_tail"],
        choices=["mlp_tail", "attn_tail"],
    )
    parser.add_argument("--include-blocks", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument(
        "--out",
        default=os.path.join(HERE, "llama_context_windows_s128_results.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    workloads = build_workloads(args)

    print(
        "=== Llama-3.1-8B context windows "
        f"S={S} TP=8 dtype=bfloat16 len={args.min_len}..{args.max_len} "
        f"runs={args.runs} ===",
        flush=True,
    )
    print(f"families={','.join(args.families)} workloads={len(workloads)}", flush=True)

    if args.list_only:
        for family, sequence, kind, ops, shape_name in workloads:
            print(
                f"[{family:9s} {kind:6s} {sequence:4s}] "
                f"input={shape_name}:{SHAPES[shape_name]} ops={ops}",
                flush=True,
            )
        return

    dev = torch.device("rngd", 0)
    rows = []
    for family, sequence, kind, ops, shape_name in workloads:
        row = {
            "npe": 8,
            "B": B,
            "S": S,
            "D": D,
            "NH": NH,
            "KV": KV,
            "HD": HD,
            "INTER": INTER,
            "dtype": "bfloat16",
            "family": family,
            "kind": kind,
            "length": len(sequence) if sequence not in ("MLP", "MLPR") else "",
            "start": sequence[0],
            "sequence": sequence,
            "ops": ops,
            "input_shape_name": shape_name,
            "input_shape": str(SHAPES[shape_name]),
            "error": "",
        }
        try:
            row.update(measure(family, sequence, args, dev))
            print(
                f"[{family:9s} {kind:6s} {sequence:4s}] "
                f"task={row['task_us']:.2f}us tu={row['tu_us']:.2f}us "
                f"dma={row['dma_us']:.2f}us sto={row['sto_trf_us']:.2f}us",
                flush=True,
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{family:9s} {kind:6s} {sequence:4s}] FAIL {row['error']}", flush=True)
        rows.append(row)

    fields = [
        "npe", "B", "S", "D", "NH", "KV", "HD", "INTER", "dtype",
        "family", "kind", "length", "start", "sequence", "ops",
        "input_shape_name", "input_shape",
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
