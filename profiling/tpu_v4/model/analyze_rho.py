#!/usr/bin/env python3
"""Step 4: ρ 일반화 (크기 10 × seq 4, N ∈ {1, 2, 4}, unroll) → data_rho_tpu_v4.txt, analyze_rho.log

usage: analyze_rho.py [data_rho_meas_tpu_v4.txt]
  ρ    = (L(2) − L(1)) / L(1)          L(1) = x만 사본 (지시서 정의)
  ρ_w2 = (L(2) − L(1)) / L(1)_w2       L(1)_w2 = weight도 사본 (Module 2 l1과 같은 입력 방식, Step 6 짝)
  ρ4   = (L(4) − L(1)) / (3·L(1))      (marginal이 N=2부터 일정한지 확인)
  예측 (크기 1종을 빼고 학습, 같은 seq·다른 seq 모두 학습에 씀):
    const : 다른 크기 ρ의 평균 (같은 seq의 평균 = const_seq 도 같이)
    lin   : a + b·L(1)[ms] + c·weight bytes[GB] + d·flops[TF] + e·seq/1000  (최소제곱)
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from load import load_rows  # noqa: E402

SEQS = (128, 512, 1024, 2048)


def layer_feats(D, I, NH, HD, KV, S):
    q, kv = NH * HD, KV * HD
    w = D * q + 2 * D * kv + q * D + 3 * D * I
    return 2.0 * w, 2.0 * S * w + 4.0 * NH * S * S * HD  # weight bytes, flops


def table(path):
    R, _, fails = load_rows(path)
    rows = []
    for dims in sorted({k[0] for k in R}):
        for s in SEQS:
            g = lambda n, m="unroll": R.get((dims, n, s, m))  # noqa: E731
            if None in (g(1), g(2), g(4)):
                continue
            wb, fl = layer_feats(*dims, s)
            rows.append({"dims": dims, "seq": s, "L1": g(1), "L2": g(2), "L4": g(4), "L1w": g(1, "unroll_w2"),
                         "wb": wb, "fl": fl})
    return rows, fails


def X(r, l1key):
    return [1.0, r[l1key] / 1e3, r["wb"] / 1e9, r["fl"] / 1e12, r["seq"] / 1e3]


def evaluate(rows, target, l1key):
    """크기 1종 빼고 학습 → 그 크기 예측. 반환 pred dict[(dims, seq)] = (const, const_seq, lin)."""
    out = {}
    for d in sorted({r["dims"] for r in rows}):
        tr = [r for r in rows if r["dims"] != d]
        te = [r for r in rows if r["dims"] == d]
        y = np.array([r[target] for r in tr])
        c = np.linalg.lstsq(np.array([X(r, l1key) for r in tr]), y, rcond=None)[0]
        for r in te:
            cs = np.mean([q[target] for q in tr if q["seq"] == r["seq"]])
            out[(d, r["seq"])] = (float(y.mean()), float(cs), float(np.dot(X(r, l1key), c)))
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "data_rho_meas_tpu_v4.txt")
    rows, fails = table(path)
    print(f"points={len(rows)} failed rows={len(fails)} {fails[:5]}")
    for r in rows:
        r["rho"] = (r["L2"] - r["L1"]) / r["L1"]
        r["rho4"] = (r["L4"] - r["L1"]) / (3 * r["L1"])
        r["rho_w2"] = (r["L2"] - r["L1"]) / r["L1w"] if r["L1w"] else float("nan")
    print(f"\n{'dims':28s} {'seq':>5s} {'L1':>8s} {'L1_w2':>8s} {'ρ':>6s} {'ρ4':>6s} {'ρ_w2':>6s}")
    for r in rows:
        print(f"{str(r['dims']):28s} {r['seq']:5d} {r['L1']:8.1f} {r['L1w'] or 0:8.1f} {r['rho']:6.3f} "
              f"{r['rho4']:6.3f} {r['rho_w2']:6.3f}")
    for s in SEQS:
        a = np.array([r["rho"] for r in rows if r["seq"] == s])
        b = np.array([r["rho_w2"] for r in rows if r["seq"] == s])
        print(f"  seq {s:5d}: ρ mean {a.mean():.3f} (min {a.min():.3f} max {a.max():.3f})   "
              f"ρ_w2 mean {np.nanmean(b):.3f} (min {np.nanmin(b):.3f} max {np.nanmax(b):.3f})")
    res = {}
    for target, l1key in (("rho", "L1"), ("rho_w2", "L1w")):
        if any(np.isnan(r[target]) for r in rows):
            continue
        p = evaluate(rows, target, l1key)
        res[target] = p
        print(f"\n== {target} 예측 (크기 1종 빼고 학습)")
        print("   방식        |ρ̂−ρ| 평균   max     L(n)=L1(1+(n−1)ρ̂) 환산 오차 (n=32)")
        for i, nm in enumerate(("const", "const_seq", "lin")):
            e = np.array([p[(r["dims"], r["seq"])][i] - r[target] for r in rows])
            # n=32 층에서 L(32) 상대오차 ≈ 31·Δρ / (1 + 31ρ)
            e32 = np.array([31 * (p[(r["dims"], r["seq"])][i] - r[target]) / (1 + 31 * r[target]) for r in rows])
            print(f"   {nm:10s}  {np.mean(np.abs(e)):.4f}     {np.max(np.abs(e)):.4f}   "
                  f"{np.mean(np.abs(e32)):.2%} (max {np.max(np.abs(e32)):.2%})")
    out = os.path.join(_HERE, "data_rho_tpu_v4.txt")
    with open(out, "w") as f:
        f.write("# Step 4 ρ (analyze_rho.py). L = make_stack unroll sync_min us, x만 사본 2개. L1_w2 = weight도 사본.\n")
        f.write("# rho = (L2−L1)/L1, rho_w2 = (L2−L1)/L1_w2. *_hat_* = 그 크기를 빼고 학습한 예측 (const = 다른 크기 평균, lin = 선형)\n")
        f.write("# RHO D INTER NH HD KV seq L1 L2 L4 L1_w2 rho rho_w2 rho_hat_const rho_hat_lin rho_w2_hat_const rho_w2_hat_lin\n")
        for r in rows:
            k = (r["dims"], r["seq"])
            a = res.get("rho", {}).get(k, (np.nan,) * 3)
            b = res.get("rho_w2", {}).get(k, (np.nan,) * 3)
            f.write(f"RHO {' '.join(map(str, r['dims']))} {r['seq']} {r['L1']:.3f} {r['L2']:.3f} {r['L4']:.3f} "
                    f"{r['L1w'] or 0:.3f} {r['rho']:.4f} {r['rho_w2']:.4f} {a[0]:.4f} {a[2]:.4f} {b[0]:.4f} {b[2]:.4f}\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
