#!/usr/bin/env python3
"""(a) B 덩어리 token 나누기 변형별로 W2·W3를 예측(E1)하고 예측 window로 telescope.

사용: fit_variants.py [--windows data_windows_copies2_tpu_v4.txt] [--variants data_variants_copies2_tpu_v4.txt]
변형 token이 여러 op 묶음이면 그 token의 U = 묶음 window 측정값, meta(flops, weight bytes) = op 합,
type = op type을 '.'로 이은 것 (예: gemm.silu.mul). model은 fit_step4 그대로 (M0, GBM S1·S2, target r).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fit_step4 as fs  # noqa: E402
import variants as vr  # noqa: E402
from analyze_tele import load as load_str  # noqa: E402,F401
import kernels as kn  # noqa: E402

CFGS = [("M0", 0, "r"), ("M3gbm", 1, "r"), ("M3gbm", 2, "r")]


def rows_for(var, data, L):
    toks = vr.VARIANTS[var]
    n = len(toks)
    rows = []
    for (dims, seq), d in data.items():
        meta = fs.token_meta(dims, seq)
        tm = []
        for t in toks:
            tm.append({k: sum(meta[o][k] for o in t) for k in ("flops", "wbytes")}
                      | {"obytes": meta[t[-1]]["obytes"]})
        for j in range(n - L + 1):
            ops = [o for t in toks[j:j + L] for o in t]
            name = vr.set_name(ops)
            us = [d.get(vr.set_name(t)) for t in toks[j:j + L]]
            if name not in d or None in us:
                continue
            outs = vr.set_window(ops)[2]
            rows.append({"dims": dims, "seq": seq, "i": j, "toks": [".".join(t) for t in toks[j:j + L]],
                         "types": tuple(".".join(fs.TYPE_OF[o] for o in t) for t in toks[j:j + L]),
                         "U": us, "W": d[name], "meta": tm[j:j + L],
                         "direct": [1.0 if any(o in fs.INPUTS[p] for o in toks[j + k] for p in toks[j + k + 1]) else 0.0
                                    for k in range(L - 1)],
                         "out_bytes": sum(meta[o]["obytes"] for o in outs)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default=os.path.join(HERE, "data_windows_copies2_tpu_v4.txt"))
    ap.add_argument("--variants", default=os.path.join(HERE, "data_variants_copies2_tpu_v4.txt"))
    a = ap.parse_args()
    dw, dv = fs.load(a.windows), fs.load(a.variants)
    data = {k: {**dw.get(k, {}), **dv[k]} for k in dv}
    fullB = kn.window_name(6, 15)
    print(f"{'variant':7s} {'W3 E1 MAPE: M0 / S1 / S2':>28s} | {'B: uni   tele3  tele3_hat(S2)  max':>36s}")
    for var, toks in vr.VARIANTS.items():
        n = len(toks)
        rows_by_L = {L: rows_for(var, data, L) for L in (2, 3)}
        res = fs.evaluate(rows_by_L, ("E1",), CFGS, procs=16)
        pred = {(k, L, i): w for L in (2, 3) for k, i, W, w in res[(L, "M3gbm", 2, "r", "E1")][1]}
        e = {"uni": [], "t3": [], "t3h": []}
        for key, d in data.items():
            g = lambda j, L: d[vr.set_name([o for t in toks[j:j + L] for o in t])]  # noqa: E731
            h = lambda j, L: g(j, 1) if L == 1 else pred[(key, L, j)]  # noqa: E731

            def tel(f, L):
                if L == 1:
                    return sum(f(j, 1) for j in range(n))
                return f(0, L) + sum(f(j, L) - f(j, L - 1) for j in range(1, n - L + 1))
            full = d[fullB]
            e["uni"].append(abs(tel(g, 1) / full - 1))
            e["t3"].append(abs(tel(g, 3) / full - 1))
            e["t3h"].append(abs(tel(h, 3) / full - 1))
        w3 = " / ".join(f"{100 * res[(3, *c, 'E1')][0]:5.2f}" for c in CFGS)
        print(f"{var:7s} {w3:>28s} | {100 * np.mean(e['uni']):6.2f} {100 * np.mean(e['t3']):6.2f}"
              f" {100 * np.mean(e['t3h']):12.2f} {100 * np.max(e['t3h']):8.2f}")


if __name__ == "__main__":
    main()
