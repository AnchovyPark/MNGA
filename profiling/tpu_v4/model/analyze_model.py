#!/usr/bin/env python3
"""Step 5·6: full model prefill 예측 vs 실측.  → analyze_model.log

usage: analyze_model.py
  data: data_model_tpu_v4.txt (MROW full/emb/head), data_stack_tpu_v4.txt (L(N)), data_rho_tpu_v4.txt (ρ̂),
        ../module2/data_layerpred_tpu_v4.txt (l1_pred, l1_meas = L16 full, x·weight 사본 2개)
  F = make_model 실측, E = emb, H = head (둘 다 측정값, mode 무관이라 unroll 행).
  Step 5:
    단순 합   E + n·L(1) + H
    조립      E + L_hat(n) + H,  L_hat(n) = L(1) + (n−1)(L(2) − L(1))
    실측 L(n) E + L(n) + H
  Step 6 (완전 bottom-up):
    full_hat = E + l1_pred·(1 + (n−1)·ρ̂) + H
    ρ̂ = 그 크기를 빼고 학습한 ρ (data_rho). l1_pred는 weight 사본 입력 방식이라 ρ̂_w2(= marginal/L(1)_w2)와 짝.
    비교: ρ̂(x만 사본 정의), l1_meas(측정 layer 1개)로 바꾼 것.
"""
import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from load import NAMES, NLAYERS, load_rows  # noqa: E402

SEQS = (128, 512, 1024, 2048)


def load_rho():
    out = {}
    p = os.path.join(_HERE, "data_rho_tpu_v4.txt")
    if not os.path.exists(p):
        return out
    for ln in open(p):
        q = ln.split()
        if q and q[0] == "RHO":
            dims, s = tuple(map(int, q[1:6])), int(q[6])
            v = list(map(float, q[7:]))
            out[(dims, s)] = dict(zip(["L1", "L2", "L4", "L1w", "rho", "rho_w2", "rho_c", "rho_l", "rhow_c", "rhow_l"], v))
    return out


def load_lpred():
    out = {}
    for ln in open(os.path.join(_HERE, "..", "module2", "data_layerpred_tpu_v4.txt")):
        q = ln.split()
        if q and q[0] == "LPRED":
            out[(tuple(map(int, q[1:6])), int(q[6]))] = (float(q[7]), float(q[8]))
    return out


def main():
    S, _, _ = load_rows(os.path.join(_HERE, "data_stack_tpu_v4.txt"))
    _, M, fails = load_rows(os.path.join(_HERE, "data_model_tpu_v4.txt"))
    print(f"model rows={len(M)} failed={len(fails)} {fails[:5]}")
    rho, lp = load_rho(), load_lpred()
    err = defaultdict(list)
    print("\n== Step 5 (us, 예측/실측 − 1)")
    print(f"{'model':5s} {'mode':6s} {'seq':>5s} {'F':>10s} {'E':>6s} {'H':>7s} {'(E+H)/F':>7s} | "
          f"{'단순합':>7s} {'조립':>7s} {'실측L(n)':>8s}")
    for name, dims in NAMES.items():
        n = NLAYERS[name]
        for mode in ("unroll", "scan"):
            for s in SEQS:
                F = M.get((dims, n, s, mode, "full"))
                E, H = M.get((dims, n, s, "unroll", "emb")), M.get((dims, n, s, "unroll", "head"))
                L = {N: S[(dims, N, s, mode)] for N in (1, 2, n) if (dims, N, s, mode) in S}
                if None in (F, E, H) or len(L) < 3:
                    continue
                pr = {"simple": E + n * L[1] + H, "assembled": E + L[1] + (n - 1) * (L[2] - L[1]) + H,
                      "meas_Ln": E + L[n] + H}
                for k, v in pr.items():
                    err[("5", mode, k)].append(v / F - 1)
                print(f"{name:5s} {mode:6s} {s:5d} {F:10.1f} {E:6.1f} {H:7.1f} {(E + H) / F:7.2%} | "
                      f"{pr['simple'] / F - 1:+7.2%} {pr['assembled'] / F - 1:+7.2%} {pr['meas_Ln'] / F - 1:+8.2%}")
    for mode in ("unroll", "scan"):
        for k in ("simple", "assembled", "meas_Ln"):
            e = np.array(err[("5", mode, k)])
            if len(e):
                print(f"   {mode:6s} {k:10s} n={len(e):2d} MAPE {np.mean(np.abs(e)):6.2%} max {np.max(np.abs(e)):6.2%} "
                      f"mean {np.mean(e):+6.2%}")
    if not rho:
        print("\n(data_rho_tpu_v4.txt 없음 → Step 6 생략)")
        return
    print("\n== Step 6 완전 bottom-up: full_hat = E + l1·(1 + (n−1)·ρ̂) + H  (예측/실측 − 1)")
    combos = [("l1_pred", "rhow_l"), ("l1_pred", "rhow_c"), ("l1_pred", "rho_l"), ("l1_meas", "rhow_l"),
              ("l1_meas", "rho_w2"), ("L1_x", "rho")]
    print(f"{'model':5s} {'mode':6s} {'seq':>5s} " + " ".join(f"{a + '·' + b:>15s}" for a, b in combos))
    for name, dims in NAMES.items():
        n = NLAYERS[name]
        for mode in ("unroll", "scan"):
            for s in SEQS:
                F = M.get((dims, n, s, mode, "full"))
                E, H = M.get((dims, n, s, "unroll", "emb")), M.get((dims, n, s, "unroll", "head"))
                r, l = rho.get((dims, s)), lp.get((dims, s))
                if None in (F, E, H, r, l):
                    continue
                l1 = {"l1_pred": l[0], "l1_meas": l[1], "L1_x": r["L1"]}
                cells = []
                for a, b in combos:
                    e = (E + l1[a] * (1 + (n - 1) * r[b]) + H) / F - 1
                    err[("6", mode, a, b)].append(e)
                    cells.append(f"{e:+15.2%}")
                print(f"{name:5s} {mode:6s} {s:5d} " + " ".join(cells))
    for mode in ("unroll", "scan"):
        for a, b in combos:
            e = np.array(err[("6", mode, a, b)])
            if len(e):
                print(f"   {mode:6s} {a + '·' + b:16s} n={len(e):2d} MAPE {np.mean(np.abs(e)):6.2%} "
                      f"max {np.max(np.abs(e)):6.2%} mean {np.mean(e):+6.2%}")
    print("   (rho = x만 사본 L(1) 기준 실측 ρ, rho_w2 = 실측 marginal/L(1)_w2, rho*_l/_c = 크기 빼고 학습한 lin/const 예측)")


if __name__ == "__main__":
    main()
