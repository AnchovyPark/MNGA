#!/usr/bin/env python3
"""Step 2·3: L(N) 표, 구간 marginal, ρ, unroll vs scan, 조립 예측 L_hat(N).

usage: analyze_stack.py [data_stack_tpu_v4.txt]  → analyze_stack.log
  L(N): make_stack, x만 사본 2개 (sync_min us). '_w2' = weight도 사본 2개.
  구간 marginal m(a,b) = (L(b) − L(a)) / (b − a), L(1)로 나눠 보인다 (RNGD: N=2부터 0.86).
  ρ = (L(2) − L(1)) / L(1).
  Step 3 조립:
    simple  N·L(1)
    k=1     L(1) + (N−1)·(L(2) − L(1))           (지시서 L_hat)
    k=2     L(2) + (N−2)·(L(3) − L(2))
    k=3     L(3) + (N−3)·(L(4) − L(3))
    k=4     L(4) + (N−4)·(L(8) − L(4))/4          (N=8은 자기 자신이라 제외)
"""
import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from load import NAMES, NLAYERS, load_rows  # noqa: E402

SEQS = (128, 512, 1024, 2048)


def methods(L, N):
    out = {"simple": N * L[1], "k=1": L[1] + (N - 1) * (L[2] - L[1])}
    if 3 in L:
        out["k=2"] = L[2] + (N - 2) * (L[3] - L[2])
    if 4 in L:
        out["k=3"] = L[3] + (N - 3) * (L[4] - L[3])
    if 8 in L and N > 8:
        out["k=4"] = L[4] + (N - 4) * (L[8] - L[4]) / 4
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "data_stack_tpu_v4.txt")
    R, _, fails = load_rows(path)
    print(f"rows={len(R)} failed={len(fails)} {fails[:5]}")
    err = defaultdict(list)
    for mode in ("unroll", "scan"):
        print(f"\n######## mode = {mode}")
        for name, dims in NAMES.items():
            n = NLAYERS[name]
            print(f"\n== {name} {dims} n={n}")
            for s in SEQS:
                L = {N: v for (d, N, ss, md), v in R.items() if d == dims and ss == s and md == mode}
                if 1 not in L or 2 not in L:
                    continue
                Ns = sorted(L)
                cells = "  ".join(f"L({N})={L[N]:.0f}" for N in Ns)
                marg = "  ".join(f"{a}-{b}:{(L[b] - L[a]) / (b - a) / L[1]:.3f}" for a, b in zip(Ns, Ns[1:]))
                rho = (L[2] - L[1]) / L[1]
                w2 = ""
                l1w, l2w = R.get((dims, 1, s, "unroll_w2")), R.get((dims, 2, s, "unroll_w2"))
                if mode == "unroll" and l2w:
                    l1w = l1w or 0
                    w2 = (f"  | w2: L(2)={l2w:.0f} art(N=2)={l2w - L[2]:.0f}us"
                          + (f" art(N=1)={l1w - L[1]:.0f}us ρ_w2={(l2w - l1w) / l1w:.3f}" if l1w else ""))
                print(f"  seq {s:5d}  {cells}")
                print(f"             ρ={rho:.3f}  marginal/L(1): {marg}{w2}")
                for N in sorted({8, 16, n}):
                    if N not in L:
                        continue
                    for m, v in methods(L, N).items():
                        if m == "k=4" and N <= 8:
                            continue
                        e = v / L[N] - 1
                        err[(mode, m)].append(e)
                        err[(mode, m, name)].append(e)
                        err[(mode, m, "seq", s)].append(e)
                        err[(mode, m, "N", "n" if N == n else N)].append(e)
                print("             L_hat/L−1: " + "  ".join(
                    f"N={N}: " + ",".join(f"{m} {v / L[N] - 1:+.1%}" for m, v in methods(L, N).items()
                                          if not (m == "k=4" and N <= 8))
                    for N in sorted({8, 16, n}) if N in L))
        # unroll vs scan
    print("\n######## scan / unroll − 1")
    for name, dims in NAMES.items():
        for s in SEQS:
            pairs = [(N, R[(dims, N, s, "scan")] / R[(dims, N, s, "unroll")] - 1) for N in range(1, 33)
                     if (dims, N, s, "scan") in R and (dims, N, s, "unroll") in R]
            if pairs:
                print(f"  {name} seq {s:5d}: " + "  ".join(f"N={N}:{r:+.1%}" for N, r in pairs))
    print("\n######## Step 3 조립 오차 (N ∈ {8, 16, n}) — MAPE / max|e| / 평균 부호")
    for mode in ("unroll", "scan"):
        for m in ("simple", "k=1", "k=2", "k=3", "k=4"):
            e = np.array(err.get((mode, m), []))
            if not len(e):
                continue
            by = "  ".join(f"{nm} {np.mean(np.abs(err[(mode, m, nm)])):.1%}" for nm in NAMES if (mode, m, nm) in err)
            bys = "  ".join(f"s{s} {np.mean(np.abs(err[(mode, m, 'seq', s)])):.1%}" for s in SEQS
                            if (mode, m, "seq", s) in err)
            byn = "  ".join(f"N={N} {np.mean(np.abs(err[(mode, m, 'N', N)])):.1%}" for N in (8, 16, "n")
                            if (mode, m, "N", N) in err)
            print(f"  {mode:6s} {m:6s} n={len(e):3d}  MAPE {np.mean(np.abs(e)):6.2%}  max {np.max(np.abs(e)):6.2%}  "
                  f"mean {np.mean(e):+6.2%} | {by} | {bys} | {byn}")


if __name__ == "__main__":
    main()
