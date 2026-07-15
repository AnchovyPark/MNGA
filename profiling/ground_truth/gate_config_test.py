#!/usr/bin/env python3
"""Does the LLM compiler config close the 12x gap? Recompile gate matmul with it.

Baseline (None config) gate at S=2048 TP=8 = 2458us. The real furiosa-llm uses an
LLM-tuned config (partial_sum_policy=Bf16ForSplitAndChipAndCluster, tactic_hint=
ForLlmModelComputeBound, ...). Recompile the same gate matmul with that config and
see if the time drops toward the real model's throughput.

python gate_config_test.py   (TP=8, S=2048)
"""
import statistics as st
from collections import defaultdict

import yaml
import torch
import furiosa.torch  # noqa: F401
from furiosa.torch import native_device as nd

nd.set_fusion(8)

from furiosa.torch.custom_ops import CompileModule  # noqa: E402
from furiosa.native_common.compiler import (  # noqa: E402
    create_default_compiler_config, create_llm_compiler_config_with_layer_range)
from furiosa_llm.parallelize.layer_range import LayerRange, TransformerBlock  # noqa: E402

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


def measure(cfg):
    dev = torch.device("rngd", 0)
    x = torch.randn(1, S, D, dtype=DTYPE)
    cm = CompileModule.from_exported(torch.export.export(Gate(), (x,)), compiler_config=cfg)
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
    return st.median(tasks), st.median(tus)


def main():
    default = yaml.safe_load(create_default_compiler_config())
    lr = LayerRange(start=TransformerBlock(idx=0), end=TransformerBlock(idx=0))
    llm = yaml.safe_load(create_llm_compiler_config_with_layer_range(
        "llama", "generate", 0.22, 1, 8, 1, 2048, 2048, lr, True, False, False, False))
    # default + just the two prime-suspect knobs
    two = dict(default)
    two["partial_sum_policy"] = llm["partial_sum_policy"]
    two["tactic_hint"] = llm["tactic_hint"]

    configs = [("None(baseline)", None), ("LLM full", llm),
               ("default+partialsum+tactic", two)]
    print(f"=== gate (1,{S},{D})@({D},{INTER}) {FLOP/1e9:.0f} GFLOP, TP=8 ===", flush=True)
    for name, cfg in configs:
        try:
            task, tu = measure(cfg)
            print(f"  {name:28s} task={task:8.1f}us ({FLOP/(task*1e-6)/1e12:5.0f} TFLOPS)  "
                  f"tu={tu:8.1f}us ({FLOP/(tu*1e-6)/1e12:5.0f} TFLOPS)", flush=True)
        except Exception as e:
            print(f"  {name:28s} FAIL {type(e).__name__}: {str(e)[:110]}", flush=True)


if __name__ == "__main__":
    main()
