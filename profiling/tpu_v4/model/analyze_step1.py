#!/usr/bin/env python3
"""Step 1 표: Module 2 L16 full 비교, weight 사본 artifact, 8B 32층 compile·HBM.

usage: analyze_step1.py  → analyze_step1.log
  layer_w2 = layer.make_layer, x·weight 모두 사본 2개 (Module 2 window 측정과 같은 입력 방식)
  unroll   = make_stack N=1, x만 사본 2개 (TPU_NEXT_MODEL §3 규약)
  unroll_w2 = make_stack N=1, weight도 사본 2개
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from load import NAMES, load_rows  # noqa: E402

M2 = os.path.join(_HERE, "..", "module2", "data_layerpred_tpu_v4.txt")


def main():
    L, M, info = load_rows(os.path.join(_HERE, "data_step1_tpu_v4.txt"), info=True)
    m2 = {}
    for ln in open(M2):
        p = ln.split()
        if p and p[0] == "LPRED":
            m2[(tuple(map(int, p[1:6])), int(p[6]))] = float(p[8])
    print("== Step 1.2 / 1.4: N=1 (sync_min us)")
    print(f"{'model':5s} {'seq':>5s} {'layer_w2':>9s} {'M2 L16':>9s} {'diff':>6s} | {'unroll_w2':>9s} "
          f"{'unroll':>9s} {'artifact':>8s} | {'scan':>9s} {'scan/unr':>8s}")
    for name, dims in NAMES.items():
        for s in (128, 512, 1024, 2048):
            g = lambda mode: L.get((dims, 1, s, mode))  # noqa: E731
            lw, uw, u, sc = g("layer_w2"), g("unroll_w2"), g("unroll"), g("scan")
            ref = m2.get((dims, s))
            d = f"{lw / ref - 1:+6.1%}" if ref else "  -"
            print(f"{name:5s} {s:5d} {lw:9.1f} {ref or 0:9.1f} {d} | {uw:9.1f} {u:9.1f} {1 - u / uw:8.1%} | "
                  f"{sc:9.1f} {sc / u - 1:+8.1%}")
    print("\n   artifact = 1 − unroll/unroll_w2 (weight를 같은 buffer로 반복하면 빨라지는 비율)")
    print("\n== Step 1.3: 8B 32층 make_model (full)")
    for k, v in M.items():
        dims, n, seq, mode, part = k
        i = info.get(("M",) + k, "")
        print(f"   {mode:6s} seq {seq:5d}: {v:10.1f} us  {i}")


if __name__ == "__main__":
    main()
