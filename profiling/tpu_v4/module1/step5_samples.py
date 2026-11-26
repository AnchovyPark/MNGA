#!/usr/bin/env python3
"""Step 5: 학습 점 수에 따른 T2(안 본 model 크기) 오차 (TPU_MODULE1_KERNEL.md §4 Step 5).

usage: step5_samples.py [--data file] [--models gemm:P2,rope:P1,...]
T2 fold(크기 leave-one-out)마다 학습 집합에서 n점을 무작위로 뽑아 fitting하고
그 fold의 평가 집합 전체로 MAPE를 낸다. seed 5개의 평균과 범위(min~max).
n ∈ {25, 50, 100, 200, 400, 전체}. 학습 집합이 n보다 작으면 그 n은 건너뛴다.
"""
import sys

import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fit_step4 as F

NS = [25, 50, 100, 200, 400, None]
DEFAULT = {"gemm": "P2", "rope": "P1", "add": "P1", "mul": "P1", "silu": "P1", "rmsnorm": "P1"}


def main():
    data = sys.argv[sys.argv.index("--data") + 1] if "--data" in sys.argv else None
    models = dict(DEFAULT)
    if "--models" in sys.argv:
        for kv in sys.argv[sys.argv.index("--models") + 1].split(","):
            k, v = kv.split(":")
            models[k] = v
    rows, _ = F.load("sync", data)
    folds = [(tr, te) for name, tr, te in F.splits(rows) if name == "T2"]
    print(f"{'type':8s} {'model':5s} " + " ".join(f"{('all' if n is None else n):>18}" for n in NS))
    for t, m in models.items():
        cells = []
        for n in NS:
            per_seed = []
            for seed in range(5):
                rng = np.random.default_rng(seed)
                errs = []
                for tr, te in folds:
                    trt = [r for r in tr if r["key"][0] == t]
                    tet = [r for r in te if r["key"][0] == t]
                    if not tet:
                        continue
                    if n is not None:
                        if len(trt) < n:
                            errs = None
                            break
                        trt = [trt[i] for i in rng.choice(len(trt), n, replace=False)]
                    c = F.fit(m, trt)
                    errs.extend(F.errs(F.predict(m, c, tet), tet).tolist())
                if errs is None:
                    per_seed = None
                    break
                per_seed.append(float(np.mean(errs)))
                if n is None:
                    break  # 전체는 seed와 무관
            if not per_seed:
                cells.append(f"{'-':>18}")
            else:
                cells.append(f"{np.mean(per_seed):6.1%} ({min(per_seed):4.1%}~{max(per_seed):4.1%})")
        print(f"{t:8s} {m:5s} " + " ".join(f"{c:>18}" for c in cells), flush=True)


if __name__ == "__main__":
    main()
