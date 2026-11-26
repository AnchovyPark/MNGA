#!/usr/bin/env python3
"""(a) B 덩어리 token 나누기 변형별 측정 telescope (variants.py).

사용: analyze_variants.py [--windows data_windows_copies2_tpu_v4.txt] [--variants data_variants_copies2_tpu_v4.txt]
값은 sync_min_us. variants 파일 값을 우선 쓰고, 없으면 windows 파일(연속 구간)에서 찾는다.
두 파일에 같은 window가 있으면 차이를 측정 재현성으로 보고한다.
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import variants as vr  # noqa: E402
from analyze_tele import load  # noqa: E402
import kernels as kn  # noqa: E402


def tele(get, n, L):
    if L == 1:
        return sum(get(j, 1) for j in range(n))
    return get(0, L) + sum(get(j, L) - get(j, L - 1) for j in range(1, n - L + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default=os.path.join(HERE, "data_windows_copies2_tpu_v4.txt"))
    ap.add_argument("--variants", default=os.path.join(HERE, "data_variants_copies2_tpu_v4.txt"))
    a = ap.parse_args()
    dw, dv = load(a.windows), load(a.variants)
    rep = [abs(dv[k][n] / dw[k][n] - 1) for k in dv for n in dv[k] if n in dw.get(k, {})]
    print(f"# 재현성: 두 파일에 다 있는 window {len(rep)}개, 차이 평균 {100 * np.mean(rep):.2f}% max {100 * np.max(rep):.2f}%")
    fullB = kn.window_name(6, 15)
    res = defaultdict(list)
    print(f"{'dims':22s} {'seq':>5s} " + " ".join(f"{v}:t1/t2/t3".rjust(20) for v in vr.VARIANTS))
    for key in sorted(dv):
        d = {**dw.get(key, {}), **dv[key]}
        full = d[fullB]
        cells = []
        for v, toks in vr.VARIANTS.items():
            n = len(toks)
            get = lambda j, L, toks=toks: d[vr.set_name([o for t in toks[j:j + L] for o in t])]  # noqa: E731
            try:
                r = [tele(get, n, L) / full for L in (1, 2, 3)]
            except KeyError as e:
                cells.append(f"missing {e}".rjust(20))
                continue
            res[v].append((key[1], r))
            cells.append("/".join(f"{x:.3f}" for x in r).rjust(20))
        print(f"{key[0]:22s} {key[1]:5d} " + " ".join(cells))
    print("\n## MAPE (%) of tele_L / full B, 크기 10 × seq 4")
    print(f"{'variant':8s} {'tokens':>6s} {'uni':>7s} {'tele2':>7s} {'tele3':>7s} {'tele3 max':>9s} | tele3 by seq 128/512/1024/2048")
    for v, rows in res.items():
        e = np.array([[abs(x - 1) for x in r] for _, r in rows]) * 100
        bys = [np.mean([abs(r[2] - 1) for q, r in rows if q == s]) * 100 for s in (128, 512, 1024, 2048)]
        print(f"{v:8s} {len(vr.VARIANTS[v]):6d} {e[:, 0].mean():7.2f} {e[:, 1].mean():7.2f} {e[:, 2].mean():7.2f} {e[:, 2].max():9.2f} | "
              + " / ".join(f"{x:.2f}" for x in bys))


if __name__ == "__main__":
    main()
