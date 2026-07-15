#!/usr/bin/env python3
"""A-workaround: compile OUR kernel with the production LLM config (native_common.compiler.compile).

Goal: get production-quality compilation for our own controllable kernels, so we can
COMPOSE them bottom-up (the research direction), not just decompose the real model.

CompileModule.from_exported wants a limited `Config` object; but the production path uses
`furiosa.native_common.compiler.compile(model, input_args, config=<dict>)` which takes the
FULL LLM config dict. Test whether our bare gate matmul compiles with it, and probe whether
the CompileResult can be run + profiled.

Everything flushed; run AFTER any other device job (no contention).
"""
import sys
import traceback

import yaml
import torch
import furiosa.torch  # noqa: F401
from furiosa.torch import native_device as nd

nd.set_fusion(8)

from furiosa.native_common.compiler import compile as native_compile  # noqa: E402
from furiosa.native_common.compiler import create_llm_compiler_config_with_layer_range  # noqa: E402
from furiosa_llm.parallelize.layer_range import LayerRange, TransformerBlock  # noqa: E402

D, INTER = 4096, 14336
S = int(sys.argv[1]) if len(sys.argv) > 1 else 128
DTYPE = torch.bfloat16


class Gate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w", torch.randn(D, INTER, dtype=DTYPE))

    def forward(self, x):
        return x @ self.w


def p(*a):
    print(*a, flush=True)


def main():
    x = torch.randn(1, S, D, dtype=DTYPE)
    lr = LayerRange(start=TransformerBlock(idx=0), end=TransformerBlock(idx=0))
    llm = yaml.safe_load(create_llm_compiler_config_with_layer_range(
        "llama", "generate", 0.22, 1, 8, 1, S, S, lr, True, False, False, False))
    p("=== config keys:", len(llm), "| partial_sum_policy:", llm.get("partial_sum_policy"))

    p("\n=== native_common.compiler.compile(gate, config=LLM dict) ===")
    try:
        res = native_compile(Gate(), (x,), target_npu="renegade", config=llm, target_ir="edf")
        p("COMPILED OK. type:", type(res).__name__)
        p("attrs:", [a for a in dir(res) if not a.startswith("_")])
        try:
            gs = res.graphs
            p("graphs:", type(gs).__name__, "len", len(gs))
            g = gs[0]
            p("graph[0]:", type(g).__name__)
            p("  attrs:", [a for a in dir(g) if not a.startswith("_")])
        except Exception as e:
            p("graphs access err:", repr(e))
    except Exception as e:
        p("COMPILE FAILED:", type(e).__name__, str(e)[:300])
        traceback.print_exc()


if __name__ == "__main__":
    main()
