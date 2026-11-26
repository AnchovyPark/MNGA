#!/usr/bin/env python3
"""Step 4: kernel 단독 latency 예측 model fitting과 평가 (TPU_MODULE1_KERNEL.md §4 Step 4).

usage: fit_step4.py [--data file] [--target sync|dev] [--types gemm,add,...] [--out coef.json]
입력: step3/data_kernels_tpu_v4.txt (KROW), points_step3_meta.txt (출처)
대상 값: sync_min_us (기본, Module 1 대표값). --target dev 이면 §4 정의 dev_min_us.

model (type별로 따로 fitting):
  P0  roofline max(flops/PEAK_FLOPS, bytes/PEAK_BW). fitting 없음.
      spec: TPU v4 chip당 bf16 275 TFLOPS, HBM 1200 GB/s
      (https://docs.cloud.google.com/tpu/docs/v4). profiler plane stats(core당 139.8 TFLOPS,
      614.4 GB/s ×2 core)도 같은 값.
  P1  U = a + b·flops + c·bytes, 상대오차 최소화(가중 최소제곱, weight 1/U).
  P2  Step 2·3에서 찾은 규칙을 넣은 model. shape는 TPU tile(M→8, K·N→128)로 올려서 쓴다.
      gemm   : fitted roofline, weight가 CMEM(128 MiB)에 들어가는지로 weight 대역폭을 나눈다.
               U = a + max(C, Mm) + e·min(C, Mm),  C = flops/Fe,
               Mm = act_bytes/Ba + w_bytes·[w ≤ CMEM]/Bc + w_bytes·[w > CMEM]/Bh
               (같은 입력 반복 측정에서는 CMEM 이하 weight가 이전 실행의 prefetch로 CMEM에 남는다)
      vector : U = a + b·flops + c·bytes + d·bytes·[M < 1024]   (Step 2의 add M=1024 계단)
      rope   : vector + e·M   (행마다 붙는 고정 비용. head 수가 적을수록 원소당 시간이 크다)

split: T1 무작위 80/20 (seed 5개), T2 크기 leave-one-out (10회), T3 seq 외삽 (양방향),
       T4 model 유래 → model 밖.
보고: split × type × model 별 MAPE, p90, max, 그리고 U<20us / U>=20us 구분.
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
PEAK_FLOPS = 275e12
PEAK_BW = 1200e9
B = 2  # bf16 bytes
CMEM = 128 * 2**20  # TPU v4 CMEM 128 MiB


def ceil_to(x, m):
    return int(math.ceil(x / m) * m)


def feats(t, M, K, N, HD, pad=False):
    """(flops, bytes). pad=True 이면 tile 크기로 올린 shape로 계산."""
    if pad:
        M = ceil_to(M, 8)
        N = ceil_to(N, 128)
        K = ceil_to(K, 128) if K else 0
    if t == "gemm":
        return 2.0 * M * K * N, B * (M * K + K * N + M * N)
    if t == "rmsnorm":
        return 4.0 * M * N, B * (2 * M * N + N)
    if t == "rope":
        return 3.0 * M * N, B * (2 * M * N + 2 * HD)
    if t == "silu":
        return 4.0 * M * N, B * (2 * M * N)
    if t in ("mul", "add"):
        return 1.0 * M * N, B * (3 * M * N)
    raise ValueError(t)


def design(model, rows):
    X = []
    for r in rows:
        t, M, K, N, HD = r["key"]
        if model == "P1":
            f, b = feats(t, M, K, N, HD)
            X.append([1.0, f / 1e9, b / 1e6])
        elif model == "P2" and t == "gemm":
            Mp, Kp, Np = ceil_to(M, 8), ceil_to(K, 128), ceil_to(N, 128)
            w = B * Kp * Np
            fits = 1.0 if B * K * N <= CMEM else 0.0
            X.append([2.0 * Mp * Kp * Np / 1e9, B * (Mp * Kp + Mp * Np) / 1e6,
                      w / 1e6 * fits, w / 1e6 * (1 - fits)])
        elif model == "P2":
            f, b = feats(t, M, K, N, HD, pad=True)
            x = [1.0, f / 1e9, b / 1e6, (b / 1e6 if M < 1024 else 0.0)]
            if t == "rope":
                x.append(M / 1e3)
            X.append(x)
    return np.array(X)


def _gemm_p2(p, X):
    comp = X[:, 0] * p[1]
    mem = X[:, 1] * p[2] + X[:, 2] * p[3] + X[:, 3] * p[4]
    return p[0] + np.maximum(comp, mem) + p[5] * np.minimum(comp, mem)


def fit(model, rows):
    if model == "P0":
        return None
    X = design(model, rows)
    y = np.array([r["y"] for r in rows])
    if model == "P2" and rows[0]["key"][0] == "gemm":
        x0 = [2, 1e3 / 275, 1e3 / 1200, 1e3 / 2000, 1e3 / 1200, 0.1]
        return least_squares(lambda p: (_gemm_p2(p, X) - y) / y, x0,
                             bounds=([0] * 6, [np.inf] * 5 + [1])).x
    w = 1.0 / y  # 상대오차 최소화
    coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    return coef


def predict(model, coef, rows):
    if model == "P0":
        out = []
        for r in rows:
            f, b = feats(*r["key"])
            out.append(max(f / PEAK_FLOPS, b / PEAK_BW) * 1e6)
        return np.array(out)
    if model == "P2" and rows[0]["key"][0] == "gemm":
        return _gemm_p2(coef, design(model, rows))
    return design(model, rows) @ coef


def errs(pred, rows):
    y = np.array([r["y"] for r in rows])
    return np.abs(pred / y - 1)


def summarize(e, ys):
    if len(e) == 0:
        return None
    e, ys = np.asarray(e), np.asarray(ys)
    small, big = e[ys < 20], e[ys >= 20]
    return {
        "n": int(len(e)), "mape": float(e.mean()), "p90": float(np.quantile(e, .9)),
        "max": float(e.max()),
        "mape_lt20": float(small.mean()) if len(small) else None, "n_lt20": int(len(small)),
        "mape_ge20": float(big.mean()) if len(big) else None, "n_ge20": int(len(big)),
    }


def load(target, data_path=None):
    src = {}
    for ln in open(os.path.join(HERE, "points_step3_meta.txt")):
        if ln.startswith("#"):
            continue
        p = ln.split()
        src[(p[0], *map(int, p[1:5]))] = p[5].split(",")
    rows, fails = [], []
    for ln in open(data_path or os.path.join(HERE, "step3", "data_kernels_tpu_v4.txt")):
        p = ln.split()
        if not p or p[0] != "KROW":
            continue
        key = (p[1], *map(int, p[2:6]))
        y = float(p[10]) if target == "sync" else float(p[6])
        if y <= 0:
            fails.append(key)
            continue
        s = src.get(key, ["rand"])
        rows.append({"key": key, "y": y, "src": s,
                     "sizes": {int(x.split(":")[1]) for x in s if x.startswith("model:")}})
    return rows, fails


def splits(rows):
    """yield (split name, train rows, test rows)."""
    rng = np.random.default_rng(0)
    for s in range(5):
        idx = rng.permutation(len(rows))
        cut = int(0.8 * len(rows))
        yield "T1", [rows[i] for i in idx[:cut]], [rows[i] for i in idx[cut:]]
    for si in range(10):
        te = [r for r in rows if si in r["sizes"]]
        tr = [r for r in rows if si not in r["sizes"]]
        if te:
            yield "T2", tr, te
    yield "T3_lo2hi", [r for r in rows if r["key"][1] <= 1024], [r for r in rows if r["key"][1] >= 2048]
    yield "T3_hi2lo", [r for r in rows if r["key"][1] >= 2048], [r for r in rows if r["key"][1] <= 1024]
    yield "T4", [r for r in rows if r["sizes"]], [r for r in rows if not r["sizes"]]


def evaluate(rows, types, models):
    """반환 res[split][type][model] = summary, 그리고 행별 오차 (T2 기준)."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: ([], []))))
    worst = defaultdict(list)
    for name, tr, te in splits(rows):
        for t in types:
            trt = [r for r in tr if r["key"][0] == t]
            tet = [r for r in te if r["key"][0] == t]
            if not tet or len(trt) < 5:
                continue
            for m in models:
                coef = fit(m, trt)
                e = errs(predict(m, coef, tet), tet)
                acc[name][t][m][0].extend(e.tolist())
                acc[name][t][m][1].extend([r["y"] for r in tet])
                if name == "T2":
                    worst[(t, m)].extend(zip(e.tolist(), [r["key"] for r in tet], [r["y"] for r in tet]))
    res = {n: {t: {m: summarize(*acc[n][t][m]) for m in acc[n][t]} for t in acc[n]} for n in acc}
    return res, worst


def main():
    target = sys.argv[sys.argv.index("--target") + 1] if "--target" in sys.argv else "sync"
    data = sys.argv[sys.argv.index("--data") + 1] if "--data" in sys.argv else None
    rows, fails = load(target, data)
    types = sorted({r["key"][0] for r in rows})
    if "--types" in sys.argv:
        types = sys.argv[sys.argv.index("--types") + 1].split(",")
    models = ["P0", "P1", "P2"]
    print(f"target={target}  rows={len(rows)}  failed(excluded)={len(fails)} {fails[:10]}")
    res, worst = evaluate(rows, types, models)
    for split in ["T1", "T2", "T3_lo2hi", "T3_hi2lo", "T4"]:
        if split not in res:
            continue
        print(f"\n== {split}")
        print(f"{'type':8s} {'model':5s} {'n':>5s} {'MAPE':>7s} {'p90':>7s} {'max':>7s}  "
              f"{'MAPE<20us':>10s}(n)  {'MAPE>=20us':>10s}(n)")
        for t in types:
            for m in models:
                s = res[split].get(t, {}).get(m)
                if not s:
                    continue
                lt = f"{s['mape_lt20']:.1%}" if s["mape_lt20"] is not None else "-"
                ge = f"{s['mape_ge20']:.1%}" if s["mape_ge20"] is not None else "-"
                print(f"{t:8s} {m:5s} {s['n']:5d} {s['mape']:7.1%} {s['p90']:7.1%} {s['max']:7.1%}  "
                      f"{lt:>10s}({s['n_lt20']})  {ge:>10s}({s['n_ge20']})")
    if "--worst" in sys.argv:
        for (t, m), lst in sorted(worst.items()):
            if m == "P0":
                continue
            lst.sort(reverse=True)
            print(f"\n-- T2 worst {t} {m}")
            for e, k, y in lst[:12]:
                print(f"   {e:6.1%}  {k}  U={y:.2f}")
    # 전체 data로 fitting한 계수 저장 (Step 6 predict.py용)
    if "--out" in sys.argv:
        out = {"target": target, "peak_flops": PEAK_FLOPS, "peak_bw": PEAK_BW, "coef": {}}
        for t in types:
            for m in ("P1", "P2"):
                out["coef"].setdefault(t, {})[m] = fit(m, [r for r in rows if r["key"][0] == t]).tolist()
        json.dump(out, open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1)
    json.dump(res, open(os.path.join(HERE, "step3", f"fit_step4_{target}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
