#!/usr/bin/env python3
"""Is the 12x gap from bad TP=8 matmul parallelization? Measure gate matmul TP=1 vs TP=8.

gate: (1,S,4096) @ (4096,14336) at S=2048. FLOP = 2*S*4096*14336.
If TP=8 gives ~8x speedup -> sharding works, gap is elsewhere.
If TP=8 ~= TP=1 -> our compile path doesn't parallelize the matmul (the 12x cause).

set_fusion is per-process -> run with NPE arg: python gate_tp_efficiency.py {1|8}
"""
import statistics as st
import sys
from collections import defaultdict

import torch
import furiosa.torch  # noqa: F401
from furiosa.torch import native_device as nd

NPE = int(sys.argv[1]) if len(sys.argv) > 1 else 8
nd.set_fusion(NPE)

from furiosa.torch.custom_ops import CompileModule  # noqa: E402

D, INTER, S = 4096, 14336, 2048
DTYPE = torch.bfloat16
FLOP = 2 * S * D * INTER


def union_us(iv):
    if not iv:
        return 0.0
    iv = sorted(iv); t = 0.0; cs, ce = iv[0]
    for s, e in iv[1:]:
        if s <= ce: ce = max(ce, e)
        else: t += ce - cs; cs, ce = s, e
    return t + (ce - cs)


class Gate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w", torch.randn(D, INTER, dtype=DTYPE))

    def forward(self, x):
        return x @ self.w


def main():
    dev = torch.device("rngd", 0)
    x = torch.randn(1, S, D, dtype=DTYPE)
    cm = CompileModule.from_exported(torch.export.export(Gate(), (x,)))
    cm.to(dev); xd = x.to(dev)
    cm(xd, profiles=None, device=dev)
    tasks, tus = [], []
    for _ in range(3):
        pr = cm.generate_profiles(dev); cm(xd, profiles=pr, device=dev)
        di = [[p.device.index for p in inn] for inn in pr]
        pc = [[p.cpu() for p in inn] for inn in pr]
        spans = cm.edf.npu_node.build_tuc_profile_spans(pc, di, 10**9)
        by = defaultdict(list); task = 0.0
        for e in spans:
            if e.name == "Task": task += e.time_range.elapsed_us()
            else: by[e.name].append((e.time_range.start, e.time_range.end))
        tasks.append(task); tus.append(union_us(by.get("Renegade::TuExec", [])))
    task = st.median(tasks); tu = st.median(tus)
    tflops_task = FLOP / (task * 1e-6) / 1e12
    tflops_tu = FLOP / (tu * 1e-6) / 1e12
    print(f"[NPE={NPE}] gate (1,{S},{D})@({D},{INTER})  {FLOP/1e9:.0f} GFLOP", flush=True)
    print(f"  task={task:.0f}us  -> {tflops_task:.0f} TFLOPS", flush=True)
    print(f"  tu  ={tu:.0f}us  -> {tflops_tu:.0f} TFLOPS (compute only)", flush=True)


if __name__ == "__main__":
    main()
