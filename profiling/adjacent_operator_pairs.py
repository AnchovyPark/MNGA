#!/usr/bin/env python3
"""Profile adjacent operator pairs on RNGD.

This is the second rung of the operator composition ladder:
compare isolated A + isolated B against joint-compiled A -> B.

Usage:
  python adjacent_operator_pairs.py [decode|prefill|all]
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
OUT_CSV = os.path.join(HERE, "adjacent_operator_pairs_results.csv")

D, NH, HD, INTER = 4096, 32, 128, 14336
DTYPE = torch.bfloat16
N_RUNS = 3
REGIME = sys.argv[1] if len(sys.argv) > 1 else "all"


class PairBlock(torch.nn.Module):
    def __init__(self, ops):
        super().__init__()
        self.kinds = [op[0] for op in ops]
        self.has_aux = []
        for i, (_kind, aux) in enumerate(ops):
            if aux is None:
                self.has_aux.append(False)
            else:
                self.register_buffer(f"aux_{i}", aux)
                self.has_aux.append(True)

    def apply_op(self, kind, x, aux):
        if kind == "rmsnorm":
            v = x.float()
            v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-5)
            return (v * aux.float()).to(x.dtype)
        if kind in ("q_proj", "gate_proj", "down_proj", "av_context"):
            return x @ aux
        if kind == "o_proj":
            bsz, n_heads, seq_len, head_dim = x.shape
            x = x.permute(0, 2, 1, 3).reshape(bsz, seq_len, n_heads * head_dim)
            return x @ aux
        if kind == "activation":
            return F.silu(x)
        if kind == "swiglu_mul":
            return x * aux
        if kind == "residual_add":
            return x + aux
        if kind == "qk_scores":
            return x @ aux
        if kind == "softmax":
            return torch.softmax(x, dim=-1)
        raise ValueError(kind)

    def forward(self, x):
        for i, kind in enumerate(self.kinds):
            aux = getattr(self, f"aux_{i}") if self.has_aux[i] else None
            x = self.apply_op(kind, x, aux)
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


def make_aux(shape):
    return torch.randn(*shape, dtype=DTYPE) if shape else None


def measure(ops, input_shape, dev):
    materialized_ops = [(kind, make_aux(aux_shape)) for kind, aux_shape in ops]
    x = torch.randn(*input_shape, dtype=DTYPE)
    mod = PairBlock(materialized_ops)
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
            if event.name == "Task":
                task += event.time_range.elapsed_us()
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
        "task_us": st.median(r[0] for r in runs),
        "tu_us": st.median(r[1] for r in runs),
        "dma_us": st.median(r[2] for r in runs),
        "envelope_us": st.median(r[3] for r in runs),
    }


def pairs_for(regime):
    if regime == "decode":
        B, S, CTX = 8, 1, 2048
    else:
        B, S, CTX = 1, 2048, 2048

    return [
        dict(
            pair="rmsnorm_to_q_proj",
            input_a=(B, S, D),
            input_b=(B, S, D),
            a=("rmsnorm", (D,)),
            b=("q_proj", (D, D)),
        ),
        dict(
            pair="rmsnorm_to_gate_proj",
            input_a=(B, S, D),
            input_b=(B, S, D),
            a=("rmsnorm", (D,)),
            b=("gate_proj", (D, INTER)),
        ),
        dict(
            pair="gate_proj_to_activation",
            input_a=(B, S, D),
            input_b=(B, S, INTER),
            a=("gate_proj", (D, INTER)),
            b=("activation", None),
        ),
        dict(
            pair="activation_to_swiglu_mul",
            input_a=(B, S, INTER),
            input_b=(B, S, INTER),
            a=("activation", None),
            b=("swiglu_mul", (B, S, INTER)),
        ),
        dict(
            pair="swiglu_mul_to_down_proj",
            input_a=(B, S, INTER),
            input_b=(B, S, INTER),
            a=("swiglu_mul", (B, S, INTER)),
            b=("down_proj", (INTER, D)),
        ),
        dict(
            pair="down_proj_to_residual",
            input_a=(B, S, INTER),
            input_b=(B, S, D),
            a=("down_proj", (INTER, D)),
            b=("residual_add", (B, S, D)),
        ),
        dict(
            pair="qk_scores_to_softmax",
            input_a=(B, NH, S, HD),
            input_b=(B, NH, S, CTX),
            a=("qk_scores", (B, NH, HD, CTX)),
            b=("softmax", None),
        ),
        dict(
            pair="softmax_to_av_context",
            input_a=(B, NH, S, CTX),
            input_b=(B, NH, S, CTX),
            a=("softmax", None),
            b=("av_context", (B, NH, CTX, HD)),
        ),
        dict(
            pair="av_context_to_o_proj",
            input_a=(B, NH, S, CTX),
            input_b=(B, NH, S, HD),
            a=("av_context", (B, NH, CTX, HD)),
            b=("o_proj", (D, D)),
        ),
        dict(
            pair="o_proj_to_residual_add",
            input_a=(B, NH, S, HD),
            input_b=(B, S, D),
            a=("o_proj", (D, D)),
            b=("residual_add", (B, S, D)),
        ),
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
        for spec in pairs_for(regime):
            try:
                a = measure([spec["a"]], spec["input_a"], dev)
                b = measure([spec["b"]], spec["input_b"], dev)
                ab = measure([spec["a"], spec["b"]], spec["input_a"], dev)
                isolated_sum = a["task_us"] + b["task_us"]
                gap = ab["task_us"] - isolated_sum
                ratio = ab["task_us"] / isolated_sum if isolated_sum else 0.0
                row = dict(
                    regime=regime,
                    pair=spec["pair"],
                    task_a_us=round(a["task_us"], 2),
                    task_b_us=round(b["task_us"], 2),
                    task_fused_us=round(ab["task_us"], 2),
                    isolated_sum_us=round(isolated_sum, 2),
                    composition_gap_us=round(gap, 2),
                    fusion_ratio=round(ratio, 4),
                    dma_a_us=round(a["dma_us"], 2),
                    dma_b_us=round(b["dma_us"], 2),
                    dma_fused_us=round(ab["dma_us"], 2),
                    tu_a_us=round(a["tu_us"], 2),
                    tu_b_us=round(b["tu_us"], 2),
                    tu_fused_us=round(ab["tu_us"], 2),
                )
                rows.append(row)
                print(
                    f"[{regime}] {spec['pair']:24s} "
                    f"A+B={isolated_sum:.1f}us fused={ab['task_us']:.1f}us "
                    f"gap={gap:+.1f}us",
                    flush=True,
                )
            except Exception as exc:
                rows.append(dict(regime=regime, pair=spec["pair"],
                                 error=f"{type(exc).__name__}: {exc}"))
                print(f"[{regime}] {spec['pair']} FAIL: {type(exc).__name__}: {exc}",
                      flush=True)

    fields = [
        "regime", "pair", "task_a_us", "task_b_us", "task_fused_us",
        "isolated_sum_us", "composition_gap_us", "fusion_ratio",
        "dma_a_us", "dma_b_us", "dma_fused_us",
        "tu_a_us", "tu_b_us", "tu_fused_us", "error",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[DONE] {OUT_CSV} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
