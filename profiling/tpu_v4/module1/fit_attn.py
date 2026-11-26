#!/usr/bin/env python3
"""2차 범위: attention core kernel 단독 latency fitting과 평가 (Step 4와 같은 방식).

usage: fit_attn.py [--data f1[,f2..]] [--worst] [--out coef.json]
기본 data: attn/data_attn_copies2_tpu_v4.txt (입력 사본 2개 순환). av는 같은 입력 반복이면 score 입력
(≤128 MiB)이 CMEM에서 읽혀 1.2~3.6배 빠르게 잡힌다. 나머지 type은 두 regime이 ±0.5% 안에서 같다.
같은 입력 data: --data attn/data_attn_tpu_v4.txt,attn/data_attn_scale_tpu_v4.txt (scale 첫 측정은
HD=0 버그로 110점 실패 → 실패 행은 첫 파일에 남기고 같은 key의 성공 행이 뒤 파일에 있으면 그것을 쓴다)
key: (type, S, NH, KV, HD). 값 = sync_min_us.

model (type별):
  P0  roofline max(flops/275T, bytes/1200G)  (kernels_attn.feats)
  P1  a + b·flops + c·bytes   (상대오차 최소화)
  P2  모든 type: score 원소 수(NH·S²)가 주된 항이다 (op 분해, attn/oplevel.log: qk·av·attn은 HD 64↔128에서
      시간이 거의 같고 score를 HBM으로 읽고 쓰는 pass 수가 시간을 정한다. qk 단독은 score를 기본 layout으로
      바꾸는 copy가 67%).
      qk, av, attn : a + b·NH·S² + c·(NH+KV)·S·HD + d·NH·S
      scale, mask, softmax : a + b·NH·S² + c·NH·S   (softmax는 행마다 드는 비용이 크다)
  P3  P2의 NH·S² 항을 score가 CMEM(128 MiB)에 들어가는지로 둘로 나눈다 (gemm P2의 CMEM 구분과 같은 생각).
      score bytes 2·NH·S² < 128 MiB 이면 b₁, 아니면 b₂. 사본 순환 data에서도 128 MiB에서 원소당 시간이
      qk ×1.9 (느려짐), av ×0.6, softmax·attn 변화, scale 변화 없음 → 같은 입력 재사용이 아니라 XLA의
      program 안 memory 배치(S(3)=CMEM)다. 경계에 딱 맞는 128 MiB는 다른 buffer와 같이 들어갈 수 없어 '안 들어감'.
split: T1 무작위 80/20 (seed 5), T2 크기 leave-one-out (head 구성이 그 크기의 것인 행을 test),
       T3 S 외삽 (≤1024 → ≥2048, 반대), T4 model 유래 → model 밖 (rand head 구성).
추가: attn 측정값 vs 분해 다섯 kernel 측정값 합 (fusion 이득).
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..")]
import fit_step4 as F  # noqa: E402  (errs, summarize)
import kernels_attn as ka  # noqa: E402

PEAK_FLOPS, PEAK_BW = 275e12, 1200e9
CMEM = 128 * 2**20


def design(model, rows):
    X = []
    for r in rows:
        t, S, NH, KV, HD = r["key"]
        f, b = ka.feats(t, S, NH, KV, HD)
        sc = NH * S * S
        fits = 1.0 if 2 * sc < CMEM else 0.0
        if model == "P1":
            X.append([1.0, f / 1e9, b / 1e6])
        elif model == "P2" and t in ka.ELEMWISE:
            X.append([1.0, sc / 1e6, NH * S / 1e6])
        elif model == "P2":
            X.append([1.0, sc / 1e6, (NH + KV) * S * HD / 1e6, NH * S / 1e6])
        elif t in ka.ELEMWISE:  # P3
            X.append([1.0, sc / 1e6 * (1 - fits), sc / 1e6 * fits, NH * S / 1e6])
        else:
            X.append([1.0, sc / 1e6 * (1 - fits), sc / 1e6 * fits, (NH + KV) * S * HD / 1e6, NH * S / 1e6])
    return np.array(X)


def fit(model, rows):
    if model == "P0":
        return None
    X = design(model, rows)
    y = np.array([r["y"] for r in rows])
    w = 1.0 / y
    coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    return coef


def predict(model, coef, rows):
    if model == "P0":
        return np.array([max(f / PEAK_FLOPS, b / PEAK_BW) * 1e6
                         for f, b in (ka.feats(*r["key"]) for r in rows)])
    return design(model, rows) @ coef


def load(path):
    src = {}
    for ln in open(os.path.join(_HERE, "points_attn_meta.txt")):
        if ln.startswith("#"):
            continue
        p = ln.split()
        src[(p[0], *map(int, p[1:5]))] = p[5].split(",")
    rows, fails = {}, {}
    for ln in (ln for p in path.split(",") if os.path.exists(p) for ln in open(p)):
        p = ln.split()
        if not p or p[0] != "KROW":
            continue
        key = (p[1], *map(int, p[2:6]))
        y = float(p[10])
        if y <= 0:
            fails[key] = 1
            continue
        s = src.get(key, ["rand"])
        rows[key] = {"key": key, "y": y, "src": s,
                     "sizes": {int(x.split(":")[1]) for x in s if x.startswith("model:")}}
    return list(rows.values()), [k for k in fails if k not in rows]


def splits(rows):
    rng = np.random.default_rng(0)
    for _ in range(5):
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


def main():
    path = sys.argv[sys.argv.index("--data") + 1] if "--data" in sys.argv else os.path.join(_HERE, "attn", "data_attn_copies2_tpu_v4.txt")
    rows, fails = load(path)
    types = [t for t in ka.TYPES if any(r["key"][0] == t for r in rows)]
    models = ["P0", "P1", "P2", "P3"]
    print(f"rows={len(rows)} failed={len(fails)} {fails[:10]}")
    acc = defaultdict(lambda: ([], []))
    worst = defaultdict(list)
    for name, tr, te in splits(rows):
        for t in types:
            trt = [r for r in tr if r["key"][0] == t]
            tet = [r for r in te if r["key"][0] == t]
            if not tet or len(trt) < 6:
                continue
            for m in models:
                e = F.errs(predict(m, fit(m, trt), tet), tet)
                acc[(name, t, m)][0].extend(e.tolist())
                acc[(name, t, m)][1].extend(r["y"] for r in tet)
                if name == "T2":
                    worst[(t, m)].extend(zip(e.tolist(), [r["key"] for r in tet], [r["y"] for r in tet]))
    res = {}
    for split in ["T1", "T2", "T3_lo2hi", "T3_hi2lo", "T4"]:
        print(f"\n== {split}")
        print(f"{'type':8s} {'model':5s} {'n':>5s} {'MAPE':>7s} {'p90':>7s} {'max':>7s}  "
              f"{'MAPE<20us':>10s}(n)  {'MAPE>=20us':>10s}(n)")
        for t in types:
            for m in models:
                if (split, t, m) not in acc:
                    continue
                s = F.summarize(*acc[(split, t, m)])
                res.setdefault(split, {}).setdefault(t, {})[m] = s
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
            for e, k, y in lst[:8]:
                print(f"   {e:6.1%}  {k}  U={y:.2f}")
    # fusion 이득: attn vs 다섯 kernel 합
    y = {r["key"]: r["y"] for r in rows}
    print("\n== attn 측정 / 분해 다섯 kernel 측정 합  (S별 평균, min~max)")
    ratio = defaultdict(list)
    for (t, S, NH, KV, HD), v in y.items():
        if t != "attn":
            continue
        parts = [y.get(("qk", S, NH, KV, HD)), y.get(("av", S, NH, KV, HD))] + \
                [y.get((e, S, NH, 0, 0)) for e in ka.ELEMWISE]
        if None in parts:
            continue
        ratio[S].append(v / sum(parts))
    for S in sorted(ratio):
        a = np.array(ratio[S])
        print(f"   S={S:5d}  {a.mean():.3f}  ({a.min():.3f}~{a.max():.3f})  n={len(a)}")
    if "--out" in sys.argv:
        out = {"target": "sync_min_us", "data": path, "coef": {}}
        for t in types:
            tr = [r for r in rows if r["key"][0] == t]
            out["coef"][t] = {m: fit(m, tr).tolist() for m in ("P1", "P2", "P3")}
        json.dump(out, open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1)
    tag = "copies2" if "copies2" in path else "same"
    json.dump(res, open(os.path.join(_HERE, "attn", f"fit_attn_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
