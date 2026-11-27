#!/usr/bin/env python3
"""TPU_NEXT_LAYER Step 3·4: L16 layer의 W2·W3 예측, layer telescope, LPRED 파일.

사용: fit_layer.py [--data data_l16_copies2_tpu_v4.txt] [--seqs 512,2048] [--lpred data_layerpred_tpu_v4.txt]
- model은 fit_step4 그대로 (M0, GBM S1·S2, target r). split E1(크기 빼기)·E2(seq 빼기)·E3(둘 다).
- attn token meta: flops = kernels_attn.feats("attn"), 출력 = [S, NH·HD].
  weight bytes 자리: "none" = 0, "score" = score bytes NH·S²·2 (Module 1에서 attention 단가가 score 128 MiB로 갈림).
- U: 측정값, 그리고 Module 1 예측 U (module1/data_upred_copies8_tpu_v4.txt + data_upred_attn_tpu_v4.txt의 attn).
  예측 U는 test 행에만 쓴다 (학습은 측정 U).
- layer tele3_hat = 예측 W2·W3 window와 U로 telescope (U 측정 / 예측 두 가지).
- LPRED: 크기마다 E1 GBM S1(score) + 예측 U의 tele3_hat. 형식 LPRED D INTER NH HD KV seq l1_pred_us l1_meas_us.
"""
import argparse
import os
import sys
import types
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import fit_step4 as fs  # noqa: E402
import kernels_attn as ka  # noqa: E402
import layer_chain as lc  # noqa: E402

CH = "L16"
SPEC = lc.CHAINS[CH]
N = len(SPEC)
TOK = lc.tokens(CH)
shim = types.SimpleNamespace(CHAIN=SPEC, TOKENS=TOK, CHAIN_OUT=lc.OUT,
                             window_name=lambda lo, hi: lc.window_name(CH, lo, hi),
                             window=lambda lo, hi: lc.window(CH, lo, hi, (64, 64, 2, 32, 1)),
                             shapes=lc.shapes)
fs.kn = shim
fs.N = N
fs.TYPE_OF = {t[0]: t[1] for t in SPEC}
fs.INPUTS = {t[0]: t[2] for t in SPEC}
fs._OUTS.clear()
_orig_meta = fs.token_meta
ATTN_BYTES = {"mode": "score"}


def token_meta(dims, seq):
    D, INTER, NH, HD, KV = dims
    spec_wo = [t for t in SPEC if t[1] != "attn"]
    fs.kn.CHAIN = spec_wo
    try:
        meta = _orig_meta(dims, seq)
    finally:
        fs.kn.CHAIN = SPEC
    fl, _ = ka.feats("attn", seq, NH, KV, HD)
    wb = 2 * NH * seq * seq if ATTN_BYTES["mode"] == "score" else 0
    meta["attn"] = {"flops": fl, "wbytes": wb, "obytes": 2 * seq * NH * HD}
    return meta


fs.token_meta = token_meta
CFGS = [("M0", 0, "r"), ("M3gbm", 1, "r"), ("M3gbm", 2, "r")]


def tele(get, lo, hi, L):
    L = min(L, hi - lo)
    if L == 1:
        return sum(get(i, 1) for i in range(lo, hi))
    return get(lo, L) + sum(get(i, L) - get(i, L - 1) for i in range(lo + 1, hi - L + 1))


def load_upred():
    up = fs.load_upred(os.path.join(HERE, "..", "module1", "data_upred_copies8_tpu_v4.txt"))
    ua = fs.load_upred(os.path.join(HERE, "..", "module1", "data_upred_attn_tpu_v4.txt"))
    for k, v in ua.items():
        if "attn" in v and k in up:
            up[k]["attn"] = v["attn"]
    return {k: v for k, v in up.items() if all(t in v for t in TOK)}


def run(data, up, kinds):
    rows_by_L = {L: fs.build_rows(data, L) for L in (2, 3)}
    res_m = fs.evaluate(rows_by_L, kinds, CFGS)
    res_p = fs.evaluate(rows_by_L, kinds, CFGS, upred=up)
    return rows_by_L, res_m, res_p


def layer_eval(data, up, res, cfg, upred):
    pred = {(k, L, i): w for L in (2, 3) for k, i, W, w in res[(L, *cfg, "E1")][1]}
    out = {}
    for key, d in data.items():
        full = d.get(lc.window_name(CH, 0, N))
        if full is None or (upred and key not in up):
            continue
        u = (lambda i: up[key][TOK[i]]) if upred else (lambda i: d[TOK[i]])  # noqa: E731
        g = lambda i, L: d[lc.window_name(CH, i, i + L)]  # noqa: E731
        h = lambda i, L: u(i) if L == 1 else pred[(key, L, i)]  # noqa: E731
        try:
            out[key] = {"full": full, "uni": tele(h, 0, N, 1), "t3": tele(g, 0, N, 3), "t3_hat": tele(h, 0, N, 3)}
        except KeyError:
            continue
    return out


def mape(vals, k):
    e = [abs(v[k] / v["full"] - 1) for v in vals]
    return 100 * np.mean(e), 100 * np.max(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data_l16_copies2_tpu_v4.txt"))
    ap.add_argument("--seqs", default=None, help="예: 512,2048 (최소 결과)")
    ap.add_argument("--kinds", default="E1,E2,E3")
    ap.add_argument("--lpred", default=None, help="LPRED 파일을 쓸 경로")
    a = ap.parse_args()
    data = fs.load(a.data)
    if a.seqs:
        keep = {int(s) for s in a.seqs.split(",")}
        data = {k: v for k, v in data.items() if k[1] in keep}
    up = load_upred()
    kinds = tuple(a.kinds.split(","))
    print(f"# {os.path.basename(a.data)}  설정 {len(data)}개 (크기 {len({k[0] for k in data})} × seq {sorted({k[1] for k in data})}),"
          f" 예측 U 있는 설정 {sum(k in up for k in data)}")
    errs = [abs(up[k][t] / d[t] - 1) for k, d in data.items() if k in up for t in TOK if t in d]
    ea = [abs(up[k]["attn"] / d["attn"] - 1) for k, d in data.items() if k in up and "attn" in d]
    print(f"# Module 1 예측 U vs 이 측정: token 전체 MAPE {100 * np.mean(errs):.2f}%, attn {100 * np.mean(ea):.2f}% (max {100 * np.max(ea):.1f}%)")

    best = None
    for mode in ("none", "score"):
        ATTN_BYTES["mode"] = mode
        rows_by_L, res_m, res_p = run(data, up, kinds)
        print(f"\n## attn weight-bytes 자리 = {mode}")
        print(f"{'model':12s} | W3 MAPE 측정U " + " ".join(f"{k:>6s}" for k in kinds) + " | 예측U " + " ".join(f"{k:>6s}" for k in kinds)
              + " | W2 E1 측정/예측U")
        for c in CFGS:
            print(f"{c[0]:6s} S{c[1]}    |              " + " ".join(f"{100 * res_m[(3, *c, k)][0]:6.2f}" for k in kinds)
                  + " |       " + " ".join(f"{100 * res_p[(3, *c, k)][0]:6.2f}" for k in kinds)
                  + f" | {100 * res_m[(2, *c, 'E1')][0]:5.2f} / {100 * res_p[(2, *c, 'E1')][0]:5.2f}")
        print(f"\n  layer (E1 window 예측) MAPE % (max): {'':10s} uni          tele3(측정 window)   tele3_hat")
        for c in CFGS[1:]:
            for upred, res in ((False, res_m), (True, res_p)):
                le = layer_eval(data, up, res, c, upred)
                vals = list(le.values())
                u, t3, th = mape(vals, "uni"), mape(vals, "t3"), mape(vals, "t3_hat")
                print(f"  {c[0]} S{c[1]} {'예측U' if upred else '측정U'}  n={len(vals):3d}  {u[0]:6.2f} ({u[1]:5.1f})  {t3[0]:6.2f} ({t3[1]:5.1f})  {th[0]:6.2f} ({th[1]:5.1f})")
                if upred and c == ("M3gbm", 1, "r") and mode == "score":
                    best = le
                if upred:
                    bys = defaultdict(list)
                    for k, v in le.items():
                        bys[k[1]].append(abs(v["t3_hat"] / v["full"] - 1))
                    print("      tele3_hat seq별: " + "  ".join(f"{q} {100 * np.mean(e):.2f}" for q, e in sorted(bys.items())))

    if a.lpred and best:
        with open(a.lpred, "w") as f:
            f.write("# Module 2 layer 1개 예측 (L16 = norm1..v, attn(fused), o..add2). TPU_NEXT_LAYER Step 4.\n"
                    "# l1_pred_us: 그 크기를 빼고 학습한 GBM S1(attn bytes=score), target r, W2·W3 예측 window + Module 1 예측 U"
                    " (copies8 token + attn P3)의 tele3. 완전 bottom-up.\n"
                    "# l1_meas_us: L16 full window 측정 (sync_min, copies2). data_l16_copies2_tpu_v4.txt\n"
                    "# LPRED D INTER NH HD KV seq l1_pred_us l1_meas_us\n")
            for key in sorted(best):
                v = best[key]
                f.write(f"LPRED {' '.join(map(str, key[0]))} {key[1]} {v['t3_hat']:.3f} {v['full']:.3f}\n")
        print(f"\n# LPRED {len(best)}행 → {a.lpred}")


if __name__ == "__main__":
    main()
