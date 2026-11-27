#!/usr/bin/env python3
"""TPU_NEXT_LAYER Step 1·2 측정 telescope 표 (layer_chain.py L16 / L20).

사용: analyze_layer.py L16 data_l16pilot_copies2_tpu_v4.txt [--n 5]
값은 sync_min_us. 크기·seq마다 uni, tele2..n / full, 그리고 덩어리별:
  A·B를 따로 telescope하고 attention은 측정 단독값을 끼운 (A_teleL + U(ATT) + B_teleL) / full.
L20의 ATT 덩어리는 attention 5 token window 측정값(6:11)을 끼운다.
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze_tele import load  # noqa: E402
import layer_chain as lc  # noqa: E402


def tele(get, lo, hi, L):
    L = min(L, hi - lo)
    if L == 1:
        return sum(get(i, 1) for i in range(lo, hi))
    return get(lo, L) + sum(get(i, L) - get(i, L - 1) for i in range(lo + 1, hi - L + 1))


def table(chain, data, nmax):
    n = len(lc.CHAINS[chain])
    sub = lc.SUB[chain]
    A, T, B = sub["A"], sub["ATT"], sub["B"]
    out = {}
    for key in sorted(data, key=lambda k: (tuple(map(int, k[0].replace(",", " ").split())), k[1])):
        d = data[key]
        g = lambda i, L: d[lc.window_name(chain, i, i + L)]  # noqa: E731
        try:
            full = d[lc.window_name(chain, 0, n)]
        except KeyError:
            continue
        row = {}
        for L in range(1, nmax + 1):
            try:
                row[f"t{L}"] = tele(g, 0, n, L) / full
            except KeyError:
                pass
            try:
                row[f"blk{L}"] = (tele(g, A[0], A[1], L) + g(T[0], T[1] - T[0]) + tele(g, B[0], B[1], L)) / full
            except KeyError:
                pass
        try:
            row["blkfull"] = (g(A[0], A[1] - A[0]) + g(T[0], T[1] - T[0]) + g(B[0], B[1] - B[0])) / full
        except KeyError:
            pass
        row["full_us"] = full
        out[key] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chain")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    for path in a.files:
        data = load(path)
        rows = table(a.chain, data, a.n)
        print(f"## {a.chain} {os.path.basename(path)}  (값/full, 1이 정확)")
        cols = [f"t{L}" for L in range(1, a.n + 1)] + [f"blk{L}" for L in range(1, a.n + 1)] + ["blkfull"]
        print(f"{'dims':24s} {'seq':>5s} {'full_us':>9s} " + " ".join(f"{c:>7s}" for c in cols))
        acc = defaultdict(list)
        for key, r in rows.items():
            print(f"{key[0]:24s} {key[1]:5d} {r['full_us']:9.1f} " + " ".join(f"{r[c]:7.3f}" if c in r else f"{'-':>7s}" for c in cols))
            for c in cols:
                if c in r:
                    acc[c].append(abs(r[c] - 1))
                    acc[(c, key[1])].append(abs(r[c] - 1))
        print("MAPE% " + " ".join(f"{c} {100 * np.mean(acc[c]):.2f}" for c in cols if acc[c]))
        for q in sorted({k[1] for k in rows}):
            print(f"  seq {q:5d}: " + " ".join(f"{c} {100 * np.mean(acc[(c, q)]):.2f}" for c in cols if acc[(c, q)]))
        print()


if __name__ == "__main__":
    main()
