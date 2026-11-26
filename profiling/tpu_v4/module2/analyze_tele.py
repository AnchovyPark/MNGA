#!/usr/bin/env python3
"""측정한 window로 uni/tele2/tele3 vs full, window별 r2·r3 (TPU_MODULE2_NGRAM.md §2, Step 2).

사용: analyze_tele.py <data_windows*.txt> [--col sync|dev] [--detail]
값은 기본 sync_min_us (core launch skew를 뺀 값, memory/2026-11-26-tpu-m2.md Step 1).
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import kernels as kn  # noqa: E402

N = len(kn.TOKENS)


def load(path, col="sync"):
    """{(dims, seq): {window_name: us}}. nan(실패) 행은 빠진다."""
    idx = {"sync": 12, "dev": 8}[col]
    data = defaultdict(dict)
    for l in open(path):
        p = l.split()
        if not p or p[0] != "WROW":
            continue
        v = float(p[idx])
        if v != v or v <= 0:
            continue
        data[(",".join(p[1:6]), int(p[7]))][p[6]] = v
    return data


def windows_of(d, n=3):
    """d → U, W[L] 배열(L=1..n), full. 하나라도 빠지면 None."""
    W = {}
    for L in range(1, n + 1):
        W[L] = [d.get(kn.window_name(i, i + L)) for i in range(N - L + 1)]
        if None in W[L]:
            return None
    full = d.get(kn.window_name(0, N))
    if full is None:
        return None
    return W, full


def tele(W, L):
    """L-gram telescope: W_L(0) + Σ_{i≥1} [W_L(i) − W_{L−1}(i)]."""
    if L == 1:
        return sum(W[1])
    return W[L][0] + sum(W[L][i] - W[L - 1][i] for i in range(1, N - L + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--col", default="sync")
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    data = load(a.path, a.col)
    hdr = "dims                     seq   full_us  " + "  ".join(f"tele{L}/full" if L > 1 else "uni/full " for L in range(1, a.n + 1))
    print(hdr)
    for (dims, seq), d in sorted(data.items()):
        r = windows_of(d, a.n)
        if r is None:
            print(f"{dims:24s} {seq:5d}  incomplete ({len(d)} windows)")
            continue
        W, full = r
        print(f"{dims:24s} {seq:5d} {full:9.2f}  " + "  ".join(f"{tele(W, L) / full:9.4f}" for L in range(1, a.n + 1)))
        if a.detail:
            U = W[1]
            print("   i  tokens                     U        W2     r2      W3     r3")
            for i in range(N):
                w2 = W[2][i] if i < N - 1 else None
                w3 = W[3][i] if i < N - 2 else None
                r2 = w2 / (U[i] + U[i + 1]) if w2 else None
                r3 = w3 / (U[i] + U[i + 1] + U[i + 2]) if w3 else None
                f = lambda x, fmt: (fmt % x) if x is not None else " " * len(fmt % 0)  # noqa: E731
                print(f"  {i:2d}  {'+'.join(kn.TOKENS[i:i+3]):24s} {U[i]:7.2f}  {f(w2, '%7.2f')} {f(r2, '%5.3f')}  {f(w3, '%7.2f')} {f(r3, '%5.3f')}")


if __name__ == "__main__":
    main()
