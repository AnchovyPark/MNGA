#!/usr/bin/env python3
"""Ratio-bridge test: is World-1 (bad config) / production a STABLE multiplier across op types?

We have production per-supertask cycles for Llama-3.2-1B (from ~/llama1b_trace, TP=8, S=128 chunk):
  Tokenwise@128 ~362000 cyc  (fused q,k,v,gate,up,down + norms)
  lm_head       1275788 cyc  (single matmul 2048->128256, memory-bound)
  Attention@128 ~40700  cyc  (kv=0: qk+softmax+av)
  Attention@256 ~85600  cyc  (kv=128)

We measure the SAME matmuls in World-1 (CompileModule, None config, TP=8) and compute the gap
factor = world1_us / prod_us per unit. If the gap is ~constant across {Tokenwise, lm_head,
Attention}, the bridge (world1 x k -> prod) is viable. If it swings, it's dead.

Clock ~1.1GHz assumed (cancels out of cross-unit RELATIVE comparison)."""
import statistics as st
import torch, furiosa.torch  # noqa
from furiosa.torch import native_device as nd
from furiosa.torch.custom_ops import CompileModule
nd.set_fusion(8)

D, NH, HD, KV, INTER, VOCAB = 2048, 32, 64, 8, 8192, 128256
S = 128
DT = torch.bfloat16
CLOCK_HZ = 1.1e9

PROD_CYC = {"Tokenwise@128": 362000, "lm_head": 1275788, "Attn@128": 40700, "Attn@256": 85600}


class MM(torch.nn.Module):
    def __init__(self, wsh, reshape=False):
        super().__init__(); self.register_buffer("w", torch.randn(*wsh, dtype=DT)); self.reshape = reshape
    def forward(self, x):
        if self.reshape: x = x.permute(0, 2, 1, 3).reshape(1, S, NH * HD)
        return x @ self.w

def p(*a): print(*a, flush=True)

def measure(insh, wsh, reshape=False):
    dev = torch.device("rngd", 0); x = torch.randn(*insh, dtype=DT)
    cm = CompileModule.from_module(MM(wsh, reshape), (x,))
    cm.to(dev); xd = x.to(dev); cm(xd, profiles=None, device=dev)
    ts = []
    for _ in range(3):
        pr = cm.generate_profiles(dev); cm(xd, profiles=pr, device=dev)
        di = [[q.device.index for q in inn] for inn in pr]; pc = [[q.cpu() for q in inn] for inn in pr]
        sp = cm.edf.npu_node.build_tuc_profile_spans(pc, di, 10**9)
        ts.append(sum(e.time_range.elapsed_us() for e in sp if e.name == "Task"))
    return st.median(ts)

# (name, input_shape, weight_shape, reshape)
UNITS = [
    ("q_proj", (1, S, D), (D, NH * HD), False),
    ("k_proj", (1, S, D), (D, KV * HD), False),
    ("v_proj", (1, S, D), (D, KV * HD), False),
    ("o_proj", (1, NH, S, HD), (NH * HD, D), True),
    ("gate",   (1, S, D), (D, INTER), False),
    ("up",     (1, S, D), (D, INTER), False),
    ("down",   (1, S, INTER), (INTER, D), False),
    ("lm_head",(1, S, D), (D, VOCAB), False),
    ("qk",     (1, NH, S, HD), (1, NH, HD, S), False),
    ("av",     (1, NH, S, S), (1, NH, S, HD), False),
]

def main():
    p(f"=== World-1 (None config, TP=8) 1B matmuls @S={S} ===")
    w = {}
    for name, insh, wsh, rs in UNITS:
        t = measure(insh, wsh, rs); w[name] = t
        p(f"  {name:8s} {t:9.1f}us")

    def prod_us(k): return PROD_CYC[k] / CLOCK_HZ * 1e6
    p("\n=== gap factor = world1_us / prod_us  (per unit) ===")
    tw_core = w["q_proj"] + w["k_proj"] + w["v_proj"] + w["gate"] + w["up"] + w["down"]
    tw_o = tw_core + w["o_proj"]
    attn = w["qk"] + w["av"]
    rows = [
        ("Tokenwise (q,k,v,gate,up,down)", tw_core, prod_us("Tokenwise@128")),
        ("Tokenwise +o_proj",              tw_o,    prod_us("Tokenwise@128")),
        ("lm_head (single matmul)",        w["lm_head"], prod_us("lm_head")),
        ("Attention core (qk+av)",         attn,    prod_us("Attn@128")),
    ]
    p(f"  {'unit':32s} {'world1_us':>10s} {'prod_us':>9s} {'GAP(x)':>8s}")
    gaps = {}
    for name, w1, pu in rows:
        g = w1 / pu; gaps[name] = g
        p(f"  {name:32s} {w1:10.1f} {pu:9.1f} {g:8.1f}")
    p("\n=== VERDICT ===")
    gcore = gaps["Tokenwise (q,k,v,gate,up,down)"]; glm = gaps["lm_head (single matmul)"]; ga = gaps["Attention core (qk+av)"]
    p(f"  Tokenwise gap {gcore:.1f}x vs lm_head gap {glm:.1f}x vs Attention gap {ga:.1f}x")
    spread = max(gcore, glm, ga) / min(gcore, glm, ga)
    p(f"  spread (max/min) = {spread:.1f}x  -> {'CONSTANT-ish (bridge viable)' if spread < 1.5 else 'NOT constant (bridge shaky/dead)'}")


if __name__ == "__main__":
    main()
