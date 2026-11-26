#!/usr/bin/env python3
"""chain을 연결된 덩어리별로 telescope (Step 2 진단).

Tokenwise chain은 attn_out이 외부 입력이라 연결되지 않은 두 덩어리로 나뉜다:
  A = norm1..v (token 0~5),  B = o..add2 (token 6~14)
full(0:15)에서 XLA가 두 덩어리의 실행 순서를 섞기 때문에(memory/2026-11-26-tpu-m2.md Step 2),
덩어리 안에서만 telescope해서 그 덩어리 window(0:6, 6:15)와 비교한다.
사용: analyze_subchain.py <data_windows*.txt> [--n 5]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_tele import load, kn

SUB = {"A": (0, 6), "B": (6, 15)}


def tele_range(d, lo, hi, L):
    """[lo,hi) 안 window만으로 L-gram telescope."""
    g = lambda i, l: d.get(kn.window_name(i, i + l))  # noqa: E731
    L = min(L, hi - lo)
    if L == 1:
        v = [g(i, 1) for i in range(lo, hi)]
        return None if None in v else sum(v)
    terms = [g(lo, L)] + [None if g(i, L) is None or g(i, L - 1) is None else g(i, L) - g(i, L - 1)
                          for i in range(lo + 1, hi - L + 1)]
    return None if None in terms else sum(terms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    for (dims, seq), d in sorted(load(a.path).items()):
        full = d.get(kn.window_name(0, 15))
        parts = {}
        for s, (lo, hi) in SUB.items():
            f = d.get(kn.window_name(lo, hi))
            parts[s] = f
            r = [tele_range(d, lo, hi, L) for L in range(1, a.n + 1)]
            print(f"{dims} {seq:5d} {s} full={f:8.2f}  " + "  ".join(
                f"t{L}={x / f:.4f}" if x else f"t{L}=  -   " for L, x in enumerate(r, 1)))
        print(f"{dims} {seq:5d} A+B={parts['A'] + parts['B']:8.2f} vs full(0:15)={full:8.2f}  ratio={(parts['A'] + parts['B']) / full:.4f}")


if __name__ == "__main__":
    main()
