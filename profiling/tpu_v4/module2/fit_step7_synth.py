#!/usr/bin/env python3
"""Step 7.1: 안 본 type 순서의 window를 chain data로만 학습한 model이 맞추는가.

사용: fit_step7_synth.py [--train data_windows_copies2_tpu_v4.txt] [--synth data_synth_copies2_tpu_v4.txt]
학습: Tokenwise chain의 W2·W3 전부 (크기 10 × seq 4). 평가: synth.SEQS 직선 chain의 W2·W3.
합성 chain의 type 순서는 학습 data에 없다 (M1은 1.0으로, M2는 안 본 쌍을 h=0으로 떨어진다).
"""
import argparse
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fit_step4 as fs  # noqa: E402
import synth as sy  # noqa: E402

CONFIGS = [("M0", 0, "r"), ("M1", 0, "r"), ("M2", 0, "r"), ("M3lin", 1, "r"),
           ("M3gbm", 0, "r"), ("M3gbm", 1, "r"), ("M3gbm", 2, "r"), ("M3gbm", 3, "r")]


def synth_rows(path, L):
    vals = defaultdict(dict)
    for l in open(path):
        p = l.split()
        if not p or p[0] != "SROW":
            continue
        v = float(p[15])
        if v == v and v > 0:
            vals[(tuple(int(x) for x in p[1:6]), p[6], int(p[9]))][(int(p[7]), int(p[8]))] = v
    rows = []
    for (dims, sname, seq), d in vals.items():
        types = tuple(sname.split("-"))
        spec, shp = sy.chain_spec(types, *dims, seq)
        meta = []
        for name, kind, ins, params in spec:
            out = shp[name]
            elems = out[0] * out[1]
            wb = sum(2 * math.prod(shp[q]) for q in params if not q.startswith("y"))
            fl = 2 * seq * shp[ins[0]][1] * out[1] if kind == "gemm" else \
                {"rmsnorm": 4, "rope": 3, "silu": 4, "mul": 1, "add": 1}[kind] * elems
            meta.append({"flops": fl, "wbytes": wb, "obytes": 2 * elems})
        for i in range(len(types) - L + 1):
            if (i, i + L) not in d or any((i + k, i + k + 1) not in d for k in range(L)):
                continue
            rows.append({"dims": dims, "seq": seq, "i": i, "toks": [t[0] for t in spec[i:i + L]],
                         "types": types[i:i + L], "U": [d[(i + k, i + k + 1)] for k in range(L)],
                         "W": d[(i, i + L)], "meta": meta[i:i + L], "direct": [1.0] * (L - 1),
                         "out_bytes": meta[i + L - 1]["obytes"], "sname": sname})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=os.path.join(HERE, "data_windows_copies2_tpu_v4.txt"))
    ap.add_argument("--synth", default=os.path.join(HERE, "data_synth_copies2_tpu_v4.txt"))
    a = ap.parse_args()
    data = fs.load(a.train)
    train = {L: fs.build_rows(data, L) for L in (2, 3)}
    test = {L: synth_rows(a.synth, L) for L in (2, 3)}
    print(f"# 학습 W2 {len(train[2])}, W3 {len(train[3])} / 평가(합성) W2 {len(test[2])}, W3 {len(test[3])}")
    ph = fs.fit_pair_h(train[2])
    preds = {}
    for L in (3, 2):
        tk = sorted({r["types"] for r in train[L]})
        print(f"\n## 합성 W{L} MAPE (%)")
        for c in CONFIGS:
            wh = fs.fit_predict(c[0], c[1], c[2], train[L], test[L], tk, ph)
            e = np.array([abs(w / r["W"] - 1) for r, w in zip(test[L], wh)])
            preds[(L, c)] = wh
            print(f"  {c[0]:6s} S{c[1]} {c[2]:3s}  {100 * e.mean():6.2f}  (max {100 * e.max():6.2f})")
    print("\n## 합성 W3: type 순서별 MAPE (%) — M0 / M2 / GBM S1 / GBM S2, 평균 r3 실측")
    for sname in sorted({r["sname"] for r in test[3]}):
        idx = [k for k, r in enumerate(test[3]) if r["sname"] == sname]
        cells = []
        for c in [("M0", 0, "r"), ("M2", 0, "r"), ("M3gbm", 1, "r"), ("M3gbm", 2, "r")]:
            cells.append(100 * np.mean([abs(preds[(3, c)][k] / test[3][k]["W"] - 1) for k in idx]))
        r3 = np.mean([test[3][k]["W"] / sum(test[3][k]["U"]) for k in idx])
        print(f"  {sname:24s} " + " / ".join(f"{x:6.2f}" for x in cells) + f"   r3={r3:.3f}")


if __name__ == "__main__":
    main()
