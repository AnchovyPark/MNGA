#!/usr/bin/env python3
"""Step 7.2b: attention core(qk..av)를 token 1개 "attn"으로 본 16-token layer의 측정·예측 telescope.

사용: fit_attn1.py [--attn data_attn_copies2_tpu_v4.txt] [--attn1 data_attn1_copies2_tpu_v4.txt]
token: norm1, q, rope_q, k, rope_k, v, attn(=LAYER 6:11), o, add1, norm2, gate, silu, up, mul, down, add2.
token window j..j+L는 LAYER 구간 [시작(j), 끝(j+L-1))이고 이름은 attn.window_name. attn token의 U = ATT window 측정값.
model은 fit_step4 그대로 (M0, GBM S0·S1·S2, target r). attn token의 meta = qk..av op 합 (flops), 출력 = av.
"""
import argparse
import os
import sys
import types

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

SPANS = [(i, i + 1) for i in range(6)] + [at.SUB["ATT"]] + [(i, i + 1) for i in range(11, 20)]
NAMES = [at.TOKENS[lo] if hi - lo == 1 else "attn" for lo, hi in SPANS]
N1 = len(SPANS)
CFGS = [("M0", 0, "r"), ("M3gbm", 0, "r"), ("M3gbm", 1, "r"), ("M3gbm", 2, "r")]


def span(j, L):
    return SPANS[j][0], SPANS[j + L - 1][1]


def wname(j, L):
    return at.window_name(*span(j, L))


def tok_inputs(j):
    lo, hi = SPANS[j]
    inside = set(at.TOKENS[lo:hi])
    return {a for t in at.LAYER[lo:hi] for a in t[2] if a not in inside}


def tok_names(j):
    lo, hi = SPANS[j]
    return set(at.TOKENS[lo:hi])


def rows_for(data, L):
    rows = []
    for (dims, seq), d in data.items():
        meta = fs.token_meta(dims, seq)
        tm = []
        for lo, hi in SPANS:
            ops = at.TOKENS[lo:hi]
            tm.append({"flops": sum(meta[o]["flops"] for o in ops),
                       "wbytes": sum(meta[o]["wbytes"] for o in ops),
                       "obytes": meta[ops[-1]]["obytes"]})
        for j in range(N1 - L + 1):
            W = d.get(wname(j, L))
            U = [d.get(wname(k, 1)) for k in range(j, j + L)]
            if W is None or None in U:
                continue
            lo, hi = span(j, L)
            outs = at.window(lo, hi)[2]
            rows.append({"dims": dims, "seq": seq, "i": j, "toks": NAMES[j:j + L],
                         "types": tuple("attn" if NAMES[k] == "attn" else fs.TYPE_OF[NAMES[k]]
                                        for k in range(j, j + L)),
                         "U": U, "W": W, "meta": tm[j:j + L],
                         "direct": [1.0 if tok_names(k) & tok_inputs(k + 1) else 0.0 for k in range(j, j + L - 1)],
                         "out_bytes": sum(meta[o]["obytes"] for o in outs)})
    return rows


def tele(get, lo, hi, L):
    L = min(L, hi - lo)
    if L == 1:
        return sum(get(j, 1) for j in range(lo, hi))
    return get(lo, L) + sum(get(j, L) - get(j, L - 1) for j in range(lo + 1, hi - L + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", default=os.path.join(HERE, "data_attn_copies2_tpu_v4.txt"))
    ap.add_argument("--attn1", default=os.path.join(HERE, "data_attn1_copies2_tpu_v4.txt"))
    a = ap.parse_args()
    d0, d1 = fs.load(a.attn), fs.load(a.attn1)
    data = {k: {**d0[k], **d1.get(k, {})} for k in d0 if k in d1}
    print(f"# 설정 {len(data)}개 (attn ∩ attn1)")
    full_name = at.window_name(0, len(at.LAYER))

    rows_by_L = {L: rows_for(data, L) for L in (2, 3)}
    res = fs.evaluate(rows_by_L, ("E1", "E2"), CFGS, procs=48)
    for L in (3, 2):
        print(f"\n## 16-token W{L} 예측 MAPE (%)  행 {len(rows_by_L[L])}")
        for c in CFGS:
            print(f"  {c[0]:6s} S{c[1]}  E1 {100 * res[(L, *c, 'E1')][0]:6.2f}  E2 {100 * res[(L, *c, 'E2')][0]:6.2f}")
        for c in CFGS[-2:]:
            att = [abs(w / W - 1) for k, i, W, w in res[(L, *c, "E1")][1]
                   if any(NAMES[j] == "attn" for j in range(i, i + L))]
            print(f"  {c[0]:6s} S{c[1]}  E1 attn 포함 window만 {100 * np.mean(att):6.2f}% (n {len(att)})")

    print("\n## layer telescope / full layer (16 token, attn=1 token). t3_hat = E1 GBM S2(W2,W3 예측), U는 측정")
    pred = {(k, L, i): w for L in (2, 3) for k, i, W, w in res[(L, "M3gbm", 2, "r", "E1")][1]}
    pred1 = {(k, L, i): w for L in (2, 3) for k, i, W, w in res[(L, "M3gbm", 1, "r", "E1")][1]}
    errs = {k: [] for k in ("uni", "t2", "t3", "t3_hat_S1", "t3_hat_S2")}
    by_seq = {}
    for key, d in sorted(data.items()):
        g = lambda j, L: d[wname(j, L)]  # noqa: E731
        h2 = lambda j, L: g(j, 1) if L == 1 else pred[(key, L, j)]  # noqa: E731
        h1 = lambda j, L: g(j, 1) if L == 1 else pred1[(key, L, j)]  # noqa: E731
        full = d[full_name]
        v = {"uni": tele(g, 0, N1, 1) / full, "t2": tele(g, 0, N1, 2) / full, "t3": tele(g, 0, N1, 3) / full,
             "t3_hat_S1": tele(h1, 0, N1, 3) / full, "t3_hat_S2": tele(h2, 0, N1, 3) / full}
        for k in v:
            errs[k].append(abs(v[k] - 1))
            by_seq.setdefault((key[1], k), []).append(abs(v[k] - 1))
        print(f"  {','.join(map(str, key[0])):22s} {key[1]:5d}  " + "  ".join(f"{k} {x:.3f}" for k, x in v.items()))
    print("  " + "  ".join(f"{k} {100 * np.mean(e):.2f}% (max {100 * np.max(e):.1f})" for k, e in errs.items()))
    for q in sorted({k[0] for k in by_seq}):
        print(f"  seq {q:5d}: " + "  ".join(f"{k} {100 * np.mean(by_seq[(q, k)]):6.2f}%" for k in errs))


if __name__ == "__main__":
    main()
