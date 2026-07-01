#!/usr/bin/env python3
"""Attention-side operator-chain shape sweep on RNGD.

Measures composed attention chains only. Isolated operator costs should come
from stage 2 operator sweeps if needed.

Chains:
  1. qk_softmax_av
     QK scores -> softmax -> AV context

  2. rmsnorm_qproj_qk_softmax
     RMSNorm -> Q projection -> reshape -> QK scores -> softmax

  3. softmax_av_oproj_residual
     softmax -> AV context -> reshape/O projection -> residual add

Usage:
  python attention_chain_shape_sweep.py
  python attention_chain_shape_sweep.py 128,512,2048
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
OUT_CSV = os.path.join(HERE, "attention_chain_shape_sweep_results.csv")

B = 1
NH = 32
HD = 128
D = NH * HD
DTYPE = torch.bfloat16
N_RUNS = 3
DEFAULT_S = [128, 512, 2048]


class AttentionChain(torch.nn.Module):
    def __init__(self, chain, s):
        super().__init__()
        self.chain = chain
        self.s = s
        if chain == "qk_softmax_av":
            self.register_buffer("k", torch.randn(B, NH, HD, s, dtype=DTYPE))
            self.register_buffer("v", torch.randn(B, NH, s, HD, dtype=DTYPE))
        elif chain == "rmsnorm_qproj_qk_softmax":
            self.register_buffer("gain", torch.randn(D, dtype=DTYPE))
            self.register_buffer("q_weight", torch.randn(D, D, dtype=DTYPE))
            self.register_buffer("k", torch.randn(B, NH, HD, s, dtype=DTYPE))
        elif chain == "softmax_av_oproj_residual":
            self.register_buffer("v", torch.randn(B, NH, s, HD, dtype=DTYPE))
            self.register_buffer("o_weight", torch.randn(D, D, dtype=DTYPE))
            self.register_buffer("residual", torch.randn(B, s, D, dtype=DTYPE))
        else:
            raise ValueError(chain)

    def forward(self, x):
        if self.chain == "qk_softmax_av":
            x = x @ self.k
            x = torch.softmax(x, dim=-1)
            return x @ self.v

        if self.chain == "rmsnorm_qproj_qk_softmax":
            y = x.float()
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-5)
            y = (y * self.gain.float()).to(x.dtype)
            y = y @ self.q_weight
            y = y.reshape(B, self.s, NH, HD).permute(0, 2, 1, 3)
            y = y @ self.k
            return torch.softmax(y, dim=-1)

        if self.chain == "softmax_av_oproj_residual":
            y = torch.softmax(x, dim=-1)
            y = y @ self.v
            y = y.permute(0, 2, 1, 3).reshape(B, self.s, D)
            y = y @ self.o_weight
            return y + self.residual

        raise ValueError(self.chain)


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


def input_shape_for(chain, s):
    if chain == "qk_softmax_av":
        return (B, NH, s, HD)
    if chain == "rmsnorm_qproj_qk_softmax":
        return (B, s, D)
    if chain == "softmax_av_oproj_residual":
        return (B, NH, s, s)
    raise ValueError(chain)


def measure(chain, s, dev):
    x = torch.randn(*input_shape_for(chain, s), dtype=DTYPE)
    cm = CompileModule.from_exported(
        torch.export.export(AttentionChain(chain, s), (x,))
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

    chains = [
        "qk_softmax_av",
        "rmsnorm_qproj_qk_softmax",
        "softmax_av_oproj_residual",
    ]
    dev = torch.device("rngd", 0)
    rows = []
    for s in seq_lens:
        for chain in chains:
            result = measure(chain, s, dev)
            row = dict(
                B=B,
                S=s,
                NH=NH,
                HD=HD,
                D=D,
                chain=chain,
                chain_task_us=round(result["task_us"], 2),
                chain_dma_us=round(result["dma_us"], 2),
                chain_tu_us=round(result["tu_us"], 2),
            )
            rows.append(row)
            print(
                f"[S={s}] {chain}={result['task_us']:.1f}us",
                flush=True,
            )

    fields = [
        "B", "S", "NH", "HD", "D", "chain",
        "chain_task_us", "chain_dma_us", "chain_tu_us",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[DONE] {OUT_CSV} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
