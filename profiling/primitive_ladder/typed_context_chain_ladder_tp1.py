#!/usr/bin/env python3
"""Typed OP4 context-chain ladder with mixed tensor shapes.

This experiment keeps the OP vocabulary small (M/A/S/X) while making shape part
of the context. The chain starts in hidden state H. M toggles between H and F:

  H --M--> F
  F --M--> H

A/S/X are shape-preserving on the current state. This creates valid mixed-shape
chains while keeping the symbolic sequence alphabet fixed.

Train split:
  exhaustive length 1..3

Test split:
  explicit test sequences if provided, otherwise stratified sampled longer
  chains.

Examples:
  python typed_context_chain_ladder_tp1.py --T 128 --D 1024 --F 3584 --runs 3
  python typed_context_chain_ladder_tp1.py --test-sequences MXMA MSMMAX MASXMSM
"""
import argparse
import csv
import itertools
import os
import random
import statistics as st
from collections import defaultdict

import torch
import furiosa.torch  # noqa: F401
from furiosa.torch import native_device as nd

from furiosa.torch.custom_ops import CompileModule  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
DTYPE = torch.bfloat16
OPS = ("M", "A", "S", "X")
H = "H"
F = "F"


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


def next_state(state, op):
    if op == "M":
        return F if state == H else H
    return state


def dim_for_state(args, state):
    if state == H:
        return args.D
    if state == F:
        return args.F
    raise ValueError(state)


def state_path(sequence):
    states = [H]
    state = H
    for op in sequence:
        state = next_state(state, op)
        states.append(state)
    return states


class TypedContextChain(torch.nn.Module):
    def __init__(self, sequence, args):
        super().__init__()
        self.sequence = sequence
        self.b = args.B
        self.t = args.T
        self.d = args.D
        self.f = args.F

        state = H
        for idx, op in enumerate(sequence):
            in_dim = self._dim(state)
            out_state = next_state(state, op)
            out_dim = self._dim(out_state)
            if op == "M":
                self.register_buffer(
                    f"weight_{idx}", torch.randn(in_dim, out_dim, dtype=DTYPE)
                )
            elif op in ("A", "X"):
                self.register_buffer(
                    f"tensor_{idx}",
                    torch.randn(self.b, self.t, in_dim, dtype=DTYPE),
                )
            state = out_state

    def _dim(self, state):
        if state == H:
            return self.d
        if state == F:
            return self.f
        raise ValueError(state)

    def forward(self, x):
        state = H
        for idx, op in enumerate(self.sequence):
            if op == "M":
                x = x @ getattr(self, f"weight_{idx}")
            elif op == "A":
                x = x + getattr(self, f"tensor_{idx}")
            elif op == "S":
                x = torch.softmax(x, dim=-1)
            elif op == "X":
                x = x * getattr(self, f"tensor_{idx}")
            else:
                raise ValueError(op)
            state = next_state(state, op)
        return x


def workload_kind(length):
    return {
        1: "single",
        2: "double",
        3: "triple",
        4: "quad",
    }.get(length, f"len{length}")


def op_name(op):
    return {
        "M": "matmul_toggle",
        "A": "add",
        "S": "softmax",
        "X": "mul",
    }[op]


def build_workloads(args):
    workloads = []
    seen = set()
    for length in range(1, args.train_max_len + 1):
        for seq_tuple in itertools.product(OPS, repeat=length):
            sequence = "".join(seq_tuple)
            workloads.append(("train", sequence))
            seen.add(("train", sequence))

    explicit_tests = load_explicit_tests(args)
    if explicit_tests:
        for sequence in explicit_tests:
            key = ("test", sequence)
            if key not in seen:
                workloads.append(key)
                seen.add(key)
        return workloads

    rng = random.Random(args.seed)
    triples = ["".join(t) for t in itertools.product(OPS, repeat=3)]
    for length in args.test_lens:
        sequences = []
        for triple in triples:
            if len(sequences) >= args.samples_per_test_len:
                break
            pos = rng.randrange(0, length - 2)
            chars = [rng.choice(OPS) for _ in range(length)]
            chars[pos:pos + 3] = list(triple)
            sequences.append("".join(chars))
        while len(sequences) < args.samples_per_test_len:
            sequences.append("".join(rng.choice(OPS) for _ in range(length)))
        for sequence in unique_keep_order(sequences):
            key = ("test", sequence)
            if key not in seen:
                workloads.append(key)
                seen.add(key)
    return workloads


def load_explicit_tests(args):
    sequences = []
    if args.test_sequences:
        sequences.extend(args.test_sequences)
    if args.test_seq_file:
        with open(args.test_seq_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                sequences.append(line.split()[0])
    return unique_keep_order(sequences)


def unique_keep_order(values):
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def measure(sequence, args, dev):
    x = torch.randn(args.B, args.T, args.D, dtype=DTYPE)
    module = TypedContextChain(sequence, args)
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
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--T", type=int, default=128)
    parser.add_argument("--D", type=int, default=1024)
    parser.add_argument("--F", type=int, default=3584)
    parser.add_argument("--npe", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--train-max-len", type=int, default=3)
    parser.add_argument("--test-lens", nargs="+", type=int, default=[4, 5, 6])
    parser.add_argument("--samples-per-test-len", type=int, default=64)
    parser.add_argument("--test-sequences", nargs="*", default=None)
    parser.add_argument("--test-seq-file", default=None)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument(
        "--out",
        default=os.path.join(HERE, "typed_context_chain_ladder_tp1_op4_results.csv"),
    )
    return parser.parse_args()


def validate_args(args):
    explicit_tests = load_explicit_tests(args)
    if explicit_tests:
        bad = sorted(set("".join(explicit_tests)) - set(OPS))
        if bad:
            raise ValueError(
                "explicit test sequence has unsupported ops: "
                f"{''.join(bad)}"
            )
        if any(len(sequence) <= args.train_max_len for sequence in explicit_tests):
            raise ValueError("explicit test sequences must be longer than train max length")
    elif any(length <= args.train_max_len for length in args.test_lens):
        raise ValueError("--test-lens must be greater than --train-max-len")


def main():
    args = parse_args()
    validate_args(args)
    nd.set_fusion(args.npe)
    workloads = build_workloads(args)

    print(
        f"=== typed context chain NPE={args.npe} B={args.B} T={args.T} "
        f"H={args.D} F={args.F} ops={''.join(OPS)} train<=len{args.train_max_len} "
        f"runs={args.runs} workloads={len(workloads)} ===",
        flush=True,
    )

    if args.list_only:
        for split, sequence in workloads:
            states = state_path(sequence)
            transitions = "->".join(states)
            print(
                f"[{split:5s} {workload_kind(len(sequence)):6s} {sequence}] "
                f"states={transitions} ops={'->'.join(op_name(op) for op in sequence)}",
                flush=True,
            )
        return

    dev = torch.device("rngd", 0)
    rows = []
    for idx, (split, sequence) in enumerate(workloads, start=1):
        states = state_path(sequence)
        dims = [dim_for_state(args, state) for state in states]
        row = {
            "npe": args.npe,
            "B": args.B,
            "T": args.T,
            "H_dim": args.D,
            "F_dim": args.F,
            "dtype": "bfloat16",
            "op_vocab": "".join(OPS),
            "split": split,
            "kind": workload_kind(len(sequence)),
            "length": len(sequence),
            "sequence": sequence,
            "ops": "->".join(op_name(op) for op in sequence),
            "state_path": "->".join(states),
            "dim_path": "->".join(str(dim) for dim in dims),
            "error": "",
        }
        try:
            row.update(measure(sequence, args, dev))
            print(
                f"[{idx:04d}/{len(workloads):04d} {split:5s} "
                f"{row['kind']:6s} {sequence:8s}] states={row['state_path']} "
                f"task={row['task_us']:.2f}us tu={row['tu_us']:.2f}us "
                f"dma={row['dma_us']:.2f}us sto={row['sto_trf_us']:.2f}us",
                flush=True,
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(
                f"[{idx:04d}/{len(workloads):04d} {split:5s} "
                f"{row['kind']:6s} {sequence:8s}] FAIL {row['error']}",
                flush=True,
            )
        rows.append(row)

    fields = [
        "npe", "B", "T", "H_dim", "F_dim", "dtype", "op_vocab", "split",
        "kind", "length", "sequence", "ops", "state_path", "dim_path",
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
