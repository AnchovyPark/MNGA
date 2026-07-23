#!/usr/bin/env python3
"""n-gram ML 예측기 학습·평가 (4단계). ngram_dataset.csv로 학습 → 안 본 seq의 supertask 예측.

핵심 평가:
 (A) leave-one-seq-out: seq 하나 빼고 학습→그 seq window latency 예측. 안 본 seq 일반화 확인.
 (B) 학습모델로 전체 Tokenwise를 telescoping 예측(모델이 예측한 unigram/2-gram으로) → 실측 full + in-model 대조.

물리 특징(flops·weight바이트·input바이트)이 seq를 암묵적으로 encode → 안 본 seq로 외삽 가능.

사용: /home/furiosa/venv3030/bin/python ngram_train_30.py
"""
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CSV = "profiling/production_profile/ngram_dataset.csv"
OPS = ["q", "k", "v", "o", "gate", "up", "down"]
INMODEL = {128: 368000, 512: 827000, 1024: 1429000}
FEATS = ["wsize", "flops", "wbytes", "inbytes"] + [f"has_{o}" for o in OPS]


def mape(pred, true):
    return float(np.mean(np.abs((np.array(pred) - np.array(true)) / np.array(true))) * 100)


def model(kind="linear"):
    if kind == "gbm":
        return GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)
    # 물리 선형: 표준화 후 Ridge (flops/bytes에 선형 → 외삽 가능)
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def feats_for(subset, S):
    D, INTER, NH, HD, KV = 2048, 8192, 32, 64, 8
    spec = {"q": (D, NH*HD), "k": (D, KV*HD), "v": (D, KV*HD), "o": (NH*HD, D),
            "gate": (D, INTER), "up": (D, INTER), "down": (INTER, D)}
    fl = wb = ib = 0
    for op in subset:
        K, N = spec[op]; fl += 2*S*K*N; wb += K*N*2; ib += S*K*2
    row = {"wsize": len(subset), "flops": fl, "wbytes": wb, "inbytes": ib}
    for o in OPS:
        row[f"has_{o}"] = int(o in subset)
    return row


def telescope2(predict, S):
    """모델 predict(subset,S)로 unigram/2-gram 예측해 전체 telescoping."""
    U = {o: predict([o], S) for o in OPS}
    W2 = [predict(OPS[i:i+2], S) for i in range(len(OPS)-1)]
    return W2[0] + sum(W2[i] - U[OPS[i]] for i in range(1, len(W2)))


def main():
    df = pd.read_csv(CSV)
    print(f"데이터 {len(df)}행, seq들={sorted(df.seq.unique())}")

    # (A) leave-one-seq-out — 선형 vs GBM 비교
    print("\n=== (A) leave-one-seq-out: 안 본 seq window 예측 MAPE ===")
    print(f"  {'seq':>5} {'선형(물리)':>10} {'GBM(트리)':>10}")
    for S in sorted(df.seq.unique()):
        tr, te = df[df.seq != S], df[df.seq == S]
        r = {}
        for k in ("linear", "gbm"):
            m = model(k).fit(tr[FEATS], tr["cycle"])
            r[k] = mape(m.predict(te[FEATS]), te["cycle"].values)
        print(f"  {S:5d} {r['linear']:9.1f}% {r['gbm']:9.1f}%")

    # (B) 학습모델(선형)로 전체 Tokenwise telescoping 예측 (held-out seq)
    print("\n=== (B) 안 본 seq의 전체 Tokenwise를 선형모델+telescoping으로 예측 ===")
    print(f"  {'seq':>5} {'예측(telescope)':>16} {'실측 full':>10} {'vs full':>8} {'in-model':>9} {'vs GT':>7}")
    for S in sorted(df.seq.unique()):
        tr = df[df.seq != S]
        m = model("linear").fit(tr[FEATS], tr["cycle"])
        def predict(subset, s):
            x = pd.DataFrame([feats_for(subset, s)])[FEATS]
            return float(m.predict(x)[0])
        pred_full = telescope2(predict, S)
        full_row = df[(df.seq == S) & (df.ops == "|".join(OPS))]
        full = float(full_row["cycle"].iloc[0]) if len(full_row) else float("nan")
        gt = INMODEL.get(S)
        vf = pred_full/full if full == full else float('nan')
        vg = pred_full/gt if gt else float('nan')
        print(f"  {S:5d} {pred_full:16.0f} {full:10.0f} {vf:8.2f} {str(gt):>9} {vg:7.2f}")


if __name__ == "__main__":
    main()
