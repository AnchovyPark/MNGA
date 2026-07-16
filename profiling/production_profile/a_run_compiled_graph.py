#!/usr/bin/env python3
"""A 뚫기 step 2: RUN + PROFILE a CompiledGraph produced by native_compile(full LLM config).

Chain established:
  native_compile(kernel, config=<full LLM dict>, target_ir="edf") -> CompileResult.graphs[0] = CompiledGraph
  CompiledGraph has: is_edf(), is_pre_command_gen(), serialize()->bytes, deserialize(bytes,tag)
  EdfModule(ir.Edf) runs it + generate_profiles + build_tuc_profile_spans -> Task span (what we already use)

Missing link = CompiledGraph -> ir.Edf. Candidate bridge:  ir.Edf.deserialize(cg.serialize())
(ir.Edf.deserialize(bytes) and CompiledGraph.serialize()->bytes both exist).

If that bridges, we can run OUR kernel compiled with production config and read its real Task time,
then compare to the None-config (CompileModule) number to see the fidelity gain -- and compose bottom-up.
"""
import sys
import traceback

import yaml
import torch
import furiosa.torch  # noqa: F401
from furiosa.torch import native_device as nd
from furiosa.native_torch import ir
from furiosa.torch.custom_ops.edf import EdfModule

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


def profile_edf(edf, x, dev, tag):
    m = EdfModule(edf)
    m.to(dev)
    xd = x.to(dev)
    m._execute_edf([xd], profiles=None)  # warmup
    import statistics as st
    ts = []
    for _ in range(3):
        pr = m.generate_profiles(dev)
        m._execute_edf([xd], profiles=pr)
        di = [[t.device.index for t in inn] for inn in pr]
        pc = [[t.cpu() for t in inn] for inn in pr]
        sp = edf.npu_node.build_tuc_profile_spans(pc, di, 10**9)
        task = sum(e.time_range.elapsed_us() for e in sp if e.name == "Task")
        tu = sum(e.time_range.elapsed_us() for e in sp if e.name == "Renegade::TuExec")
        ts.append((task, tu))
    task = st.median(t for t, _ in ts)
    tu = st.median(u for _, u in ts)
    p(f"  [{tag}] Task={task:.1f}us  TuExec={tu:.1f}us")
    return task


def main():
    x = torch.randn(1, S, D, dtype=DTYPE)
    dev = torch.device("rngd", 0)

    lr = LayerRange(start=TransformerBlock(idx=0), end=TransformerBlock(idx=0))
    llm = yaml.safe_load(create_llm_compiler_config_with_layer_range(
        "llama", "generate", 0.22, 1, 8, 1, S, S, lr, True, False, False, False))
    p(f"=== gate S={S} | LLM config keys={len(llm)} partial_sum_policy={llm.get('partial_sum_policy')}")

    p("\n=== native_compile(gate, full LLM config, target_ir=edf) ===")
    res = native_compile(Gate(), (x,), target_npu="renegade", config=llm, target_ir="edf")
    cg = res.graphs[0]
    p("CompiledGraph:", type(cg).__name__)
    p("  is_edf:", cg.is_edf(), "| is_pre_command_gen:", cg.is_pre_command_gen())
    p("  dir:", [a for a in dir(cg) if not a.startswith("_")])

    p("\n=== bridge: ir.Edf.deserialize(cg.serialize()) ===")
    try:
        raw = cg.serialize()
        p("  serialized bytes:", len(raw))
        edf = ir.Edf.deserialize(raw)
        p("  -> ir.Edf OK:", type(edf).__name__, "| npu_node:", edf.npu_node is not None)
        p("\n=== RUN + PROFILE ===")
        profile_edf(edf, x, dev, f"prod-config S={S}")
    except Exception as e:
        p("  BRIDGE FAILED:", type(e).__name__, str(e)[:400])
        traceback.print_exc()


if __name__ == "__main__":
    main()
