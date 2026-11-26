#!/usr/bin/env python3
"""2차 범위: lm_head (gemm M∈{1,8,128,512}, K=D, N=128256)를 Step 4 gemm P2 계수로 외삽 예측해 비교.
학습 범위 밖: M<128, N=128256 > 학습 최대 N(18944). weight 2·D·N ≥ 262 MB > CMEM 이라 CMEM 항은 쓰이지 않는다.
usage: check_lmhead.py"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..")]
import numpy as np  # noqa: E402

from predict import predict_u  # noqa: E402


def rows(path):
    out = {}
    for ln in open(path):
        p = ln.split()
        if p and p[0] == "KROW":
            out[tuple(map(int, p[2:5]))] = float(p[10])
    return out


same = rows(os.path.join(_HERE, "attn", "data_lmhead_tpu_v4.txt"))
cp = rows(os.path.join(_HERE, "attn", "data_lmhead_copies2_tpu_v4.txt"))
print(f"{'M':>4s} {'D':>5s} {'same':>9s} {'copies2':>9s} {'cp/same':>8s} {'pred_same':>9s} {'err':>7s} "
      f"{'pred_cp8':>9s} {'err':>7s} {'GB/s(w)':>8s}")
es, ec = [], []
for (M, K, N), y in sorted(same.items()):
    yc = cp.get((M, K, N), 0)
    ps = predict_u("gemm", M, K, N, regime="same")
    pc = predict_u("gemm", M, K, N, regime="copies8")
    e1 = ps / y - 1 if y > 0 else float("nan")
    e2 = pc / yc - 1 if yc > 0 else float("nan")
    es.append(abs(e1))
    ec.append(abs(e2))
    print(f"{M:4d} {K:5d} {y:9.1f} {yc:9.1f} {yc / y:8.3f} {ps:9.1f} {e1:+7.1%} {pc:9.1f} {e2:+7.1%} "
          f"{2 * K * N / y / 1e3:8.0f}")
for M in (1, 8, 128, 512):
    idx = [i for i, k in enumerate(sorted(same)) if k[0] == M]
    print(f"M={M:4d}: MAPE same {np.mean([es[i] for i in idx]):.1%}  copies {np.nanmean([ec[i] for i in idx]):.1%}")
print(f"all: MAPE same {np.mean(es):.1%}  copies {np.nanmean(ec):.1%}")
