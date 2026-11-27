#!/usr/bin/env python3
"""Step 7.2: attention 포함 layer (attn.LAYER, 20 token)에서 측정·예측 telescope.

사용: fit_attn.py [data_attn_copies2_tpu_v4.txt]
- 측정 telescope: uni, tele2, tele3 vs full layer(0:20) 및 덩어리(A, ATT, B).
- 예측: fit_step4의 model(M0, GBM S1·S2)을 LAYER로 바꿔 W2·W3를 E1(크기 하나 빼기)로 예측하고 layer telescope.
- HLO: attention window에 custom-call(online-softmax 등) 치환이 있는지 본다.
fit_step4는 kernels.CHAIN 기준이라, 모듈 전역(kn, N, TYPE_OF, INPUTS)을 LAYER로 바꿔 쓴다.
"""
import os
import sys
import types
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attn as at  # noqa: E402
import fit_step4 as fs  # noqa: E402

shim = types.SimpleNamespace(CHAIN=at.LAYER, TOKENS=at.TOKENS, CHAIN_OUT=at.LAYER_OUT,
                             window_name=at.window_name, window=at.window, shapes=at.shapes)
fs.kn = shim
fs.N = len(at.LAYER)
fs.TYPE_OF = {t[0]: t[1] for t in at.LAYER}
fs.INPUTS = {t[0]: t[2] for t in at.LAYER}
fs._OUTS.clear()
N = fs.N


def dstr(key):
    return ",".join(map(str, key[0]))


def tele(get, lo, hi, L):
    L = min(L, hi - lo)
    if L == 1:
        return sum(get(i, 1) for i in range(lo, hi))
    return get(lo, L) + sum(get(i, L) - get(i, L - 1) for i in range(lo + 1, hi - L + 1))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data_attn_copies2_tpu_v4.txt")
    data = fs.load(path)
    blocks = {"layer": (0, N), **{k: at.SUB[k] for k in ("A", "ATT", "B")}}

    print("## 측정 telescope / full (MAPE %, 크기·seq 전체)")
    acc = {b: {k: [] for k in ("uni", "t2", "t3")} for b in blocks}
    by_seq = {}
    for key, d in sorted(data.items()):
        g = lambda i, L: d[at.window_name(i, i + L)]  # noqa: E731
        for b, (lo, hi) in blocks.items():
            full = d.get(at.window_name(lo, hi))
            if full is None:
                continue
            for k, L in (("uni", 1), ("t2", 2), ("t3", 3)):
                try:
                    v = tele(g, lo, hi, L) / full
                except KeyError:
                    continue
                acc[b][k].append(v)
                if b == "layer":
                    by_seq.setdefault((key[1], k), []).append(v)
        try:
            print(f"  {dstr(key):22s} {key[1]:5d}  layer uni {tele(g, 0, N, 1) / d[at.window_name(0, N)]:.3f}"
                  f"  t3 {tele(g, 0, N, 3) / d[at.window_name(0, N)]:.3f}"
                  f"  | ATT share {d[at.window_name(6, 11)] / d[at.window_name(0, N)]:.2f}"
                  f"  | (A+ATT+B)/layer {(d[at.window_name(0, 6)] + d[at.window_name(6, 11)] + d[at.window_name(11, 20)]) / d[at.window_name(0, N)]:.3f}")
        except KeyError:
            pass
    for b in blocks:
        print(f"  {b:6s} " + "  ".join(f"{k} {100 * np.mean([abs(x - 1) for x in v]):6.2f}%" for k, v in acc[b].items() if v))
    for q in sorted({k[0] for k in by_seq}):
        print(f"  layer seq {q:5d}: " + "  ".join(f"{k} {100 * np.mean([abs(x - 1) for x in by_seq[(q, k)]]):6.2f}%" for k in ("uni", "t2", "t3")))

    rows_by_L = {L: fs.build_rows(data, L) for L in (2, 3)}
    cfgs = [("M0", 0, "r"), ("M3gbm", 0, "r"), ("M3gbm", 1, "r"), ("M3gbm", 2, "r")]
    res = fs.evaluate(rows_by_L, ("E1", "E2"), cfgs)
    for L in (3, 2):
        print(f"\n## W{L} 예측 MAPE (%)  행 {len(rows_by_L[L])}")
        for c in cfgs:
            print(f"  {c[0]:6s} S{c[1]}  E1 {100 * res[(L, *c, 'E1')][0]:6.2f}  E2 {100 * res[(L, *c, 'E2')][0]:6.2f}")
    print("\n## 예측 window로 layer telescope (E1, GBM S2) / full layer")
    pred = {(k, L, i): w for L in (2, 3) for k, i, W, w in res[(L, "M3gbm", 2, "r", "E1")][1]}
    errs = {"uni": [], "t3": [], "t3_hat": []}
    for key, d in sorted(data.items()):
        g = lambda i, L: d[at.window_name(i, i + L)]  # noqa: E731
        h = lambda i, L: g(i, 1) if L == 1 else pred[(key, L, i)]  # noqa: E731
        try:
            full = d[at.window_name(0, N)]
            v = {"uni": tele(g, 0, N, 1) / full, "t3": tele(g, 0, N, 3) / full, "t3_hat": tele(h, 0, N, 3) / full}
        except KeyError:
            continue
        for k in v:
            errs[k].append(abs(v[k] - 1))
        print(f"  {dstr(key):22s} {key[1]:5d}  uni {v['uni']:.3f}  t3 {v['t3']:.3f}  t3_hat {v['t3_hat']:.3f}")
    print("  " + "  ".join(f"{k} MAPE {100 * np.mean(e):.2f}% (max {100 * np.max(e):.1f}%)" for k, e in errs.items()))

    hlo = path.replace("data_", "hlo_")
    if os.path.exists(hlo):
        cc = Counter()
        for l in open(hlo):
            p = l.split()
            if len(p) > 8 and any(t in p[6].split("+") for t in ("qk", "softmax", "av")):
                for op in p[8].split(","):
                    if op.startswith("custom-call"):
                        cc[(op, p[7])] += 1
        print("\n## attention window의 custom-call (치환 확인):", dict(cc) if cc else "없음")


if __name__ == "__main__":
    main()
