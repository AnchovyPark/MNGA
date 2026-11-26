#!/usr/bin/env python3
"""Module 2 Step 4·5: W2·W3 예측 (TPU_MODULE2_NGRAM.md §4~7). numpy만 쓴다.

사용: fit_step4.py <data_windows*.txt> [--out result.json] [--seqs 512,2048]

- 값: sync_min_us (memory/2026-11-26-tpu-m2.md Step 1).
- model: M0 단독 합, M1 type 순서별 평균 r, M2 type 쌍별 숨는 상수 h, M3 ridge·GBM 회귀.
- input: S0 type 순서 + U, S1 + flops·weight bytes, S2 + 연결 정보, S3 + shape.
- target: B = r (W/ΣU, 기본), A = log W (M3에서 비교).
- split: E1 크기 하나 빼기, E2 seq 하나 빼기, E3 크기·seq 동시에 빼기.
- Step 5: E1로 예측한 W2·W3로 덩어리별 telescope (A=token 0~5, B=6~14; Step 2 결정 ii).
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import kernels as kn  # noqa: E402

N = len(kn.TOKENS)
TYPES = ["gemm", "rmsnorm", "rope", "silu", "mul", "add"]
TYPE_OF = {t[0]: t[1] for t in kn.CHAIN}
INPUTS = {t[0]: t[2] for t in kn.CHAIN}
PEAK_FLOPS_US = 2 * 139.8e12 / 1e6     # chip(2 core) peak, flop/µs (profiler가 보고한 core spec ×2)
HBM_B_US = 2 * 614.4e9 / 1e6          # byte/µs
SUB = {"A": (0, 6), "B": (6, 15)}


# ── data ────────────────────────────────────────────────────────────────────
def load(path):
    data = defaultdict(dict)
    for l in open(path):
        p = l.split()
        if not p or p[0] != "WROW":
            continue
        v = float(p[12])
        if v == v and v > 0:
            data[(tuple(int(x) for x in p[1:6]), int(p[7]))][p[6]] = v
    return data


def load_upred(path):
    """Module 1 예측 U: {(dims, seq): {token: us}}. 형식 UPRED D INTER NH HD KV token seq u_pred_us."""
    up = defaultdict(dict)
    for l in open(path):
        p = l.split()
        if p and p[0] == "UPRED":
            up[(tuple(int(x) for x in p[1:6]), int(p[7]))][p[6]] = float(p[8])
    return up


def with_upred(r, up):
    """test 행의 U를 예측 U로 바꾼 사본 (Step 6)."""
    u = up[(r["dims"], r["seq"])]
    return {**r, "U": [u[t] for t in r["toks"]]}


def token_meta(dims, seq):
    """token별 flops, weight bytes, 출력 bytes."""
    D, INTER, NH, HD, KV = dims
    shp = kn.shapes(D, INTER, NH, HD, KV, seq)
    meta = {}
    for name, kind, ins, params in kn.CHAIN:
        out = shp[name]
        elems = math.prod(out)
        wb = sum(2 * math.prod(shp[p]) for p in params)
        if kind == "gemm":
            K = shp[ins[0]][1]
            fl = 2 * seq * K * out[1]
        elif kind in ("qk", "av"):     # attention (attn.py): 2·NH·S·S·HD
            fl = 2 * NH * seq * seq * HD
        else:
            fl = {"rmsnorm": 4, "rope": 3, "silu": 4, "mul": 1, "add": 1,
                  "scale": 1, "mask": 1, "softmax": 5}[kind] * elems
        meta[name] = {"flops": fl, "wbytes": wb, "obytes": 2 * elems}
    return meta


def build_rows(data, L):
    """길이 L window 행: dict(dims, seq, i, types, U, W, feats...)."""
    rows = []
    for (dims, seq), d in data.items():
        meta = token_meta(dims, seq)
        for i in range(N - L + 1):
            toks = kn.TOKENS[i:i + L]
            W = d.get(kn.window_name(i, i + L))
            U = [d.get(t) for t in toks]
            if W is None or None in U:
                continue
            rows.append({"dims": dims, "seq": seq, "i": i, "toks": toks,
                         "types": tuple(TYPE_OF[t] for t in toks), "U": U, "W": W,
                         "meta": [meta[t] for t in toks]})
    return rows


# ── features ────────────────────────────────────────────────────────────────
def feats(r, S, type_keys):
    U = np.array(r["U"])
    su = U.sum()
    L = len(U)
    f = []
    # S0: type 순서 one-hot, U 비율, log ΣU, min U 비율
    f += [1.0 if r["types"] == k else 0.0 for k in type_keys]
    f += list(U / su) + [math.log(su), U.min() / su]
    for p in range(L - 1):
        f.append(min(U[p], U[p + 1]) / su)
    if S >= 1:
        for m, u in zip(r["meta"], U):
            tc = m["flops"] / PEAK_FLOPS_US
            tw = m["wbytes"] / HBM_B_US
            f += [tc / u, tw / u, tc / su, tw / su]
    if S >= 2:
        for p in range(L - 1):
            a, b = r["toks"][p], r["toks"][p + 1]
            if "direct" in r:      # 합성 chain 행 (fit_step7_synth.py)
                direct = r["direct"][p]
            else:
                direct = 1.0 if a in INPUTS[b] else 0.0
            f += [direct, direct * r["meta"][p]["obytes"] / HBM_B_US / su]
        # window 밖으로 나가는 bytes (materialize 양)
        if "out_bytes" in r:
            ob = r["out_bytes"]
        else:
            outs = _outs(r["i"], r["i"] + L)
            ob = sum(r["meta"][r["toks"].index(o)]["obytes"] for o in outs)
        f += [ob / HBM_B_US / su]
    if S >= 3:
        D, INTER, NH, HD, KV = r["dims"]
        f += [math.log(r["seq"]), math.log(D), math.log(INTER), NH / 32, HD / 128, KV / 8]
    return f


_OUTS = {}


def _outs(lo, hi):
    if (lo, hi) not in _OUTS:
        _OUTS[(lo, hi)] = kn.window(lo, hi)[2]
    return _OUTS[(lo, hi)]


# ── models ──────────────────────────────────────────────────────────────────
def ridge_fit(X, y, lam=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    A = Z.T @ Z + lam * len(X) * np.diag([1] * (Z.shape[1] - 1) + [0])
    w = np.linalg.solve(A, Z.T @ y)
    return lambda Xn: np.hstack([(Xn - mu) / sd, np.ones((len(Xn), 1))]) @ w


def _tree(X, g, depth, min_leaf):
    """회귀 tree (SSE 최소 split). feature마다 정렬 + 누적합으로 모든 split을 한 번에 본다."""
    n = len(g)
    if depth == 0 or n < 2 * min_leaf:
        return float(g.mean())
    tot, tot2 = g.sum(), (g * g).sum()
    base = tot2 - tot * tot / n
    best = None
    for j in range(X.shape[1]):
        o = np.argsort(X[:, j], kind="stable")
        xs, gs = X[o, j], g[o]
        cs, cs2 = np.cumsum(gs), np.cumsum(gs * gs)
        k = np.arange(min_leaf, n - min_leaf + 1)          # 왼쪽 원소 수
        if len(k) == 0:
            continue
        valid = xs[k - 1] < xs[np.minimum(k, n - 1)]       # 같은 값 사이는 자르지 않는다
        if not valid.any():
            continue
        sl, sl2 = cs[k - 1], cs2[k - 1]
        sse = (sl2 - sl * sl / k) + ((tot2 - sl2) - (tot - sl) ** 2 / (n - k))
        sse = np.where(valid, sse, np.inf)
        m = int(np.argmin(sse))
        if best is None or sse[m] < best[0]:
            best = (sse[m], j, 0.5 * (xs[k[m] - 1] + xs[k[m]]))
    if best is None or best[0] >= base - 1e-12:
        return float(g.mean())
    _, j, t = best
    msk = X[:, j] <= t
    return (j, t, _tree(X[msk], g[msk], depth - 1, min_leaf),
            _tree(X[~msk], g[~msk], depth - 1, min_leaf))


def _tree_pred(node, X):
    """X 전체에 대한 tree 예측 (vector)."""
    if not isinstance(node, tuple):
        return np.full(len(X), node)
    j, t, l, r = node
    m = X[:, j] <= t
    out = np.empty(len(X))
    if m.any():
        out[m] = _tree_pred(l, X[m])
    if (~m).any():
        out[~m] = _tree_pred(r, X[~m])
    return out


def gbm_fit(X, y, n_trees=150, lr=0.1, depth=3, min_leaf=5):
    f0 = float(y.mean())
    pred = np.full(len(y), f0)
    trees = []
    for _ in range(n_trees):
        t = _tree(X, y - pred, depth, min_leaf)
        trees.append(t)
        pred += lr * _tree_pred(t, X)
    return lambda Xn: f0 + lr * sum(_tree_pred(t, Xn) for t in trees)


def fit_predict(model, S, target, train, test, type_keys, pair_h=None):
    """test 행의 W_hat 배열."""
    sU = np.array([sum(r["U"]) for r in test])
    if model == "M0":
        return sU
    if model == "M1":
        acc = defaultdict(list)
        for r in train:
            acc[r["types"]].append(r["W"] / sum(r["U"]))
        return np.array([np.mean(acc[r["types"]]) if acc[r["types"]] else 1.0 for r in test]) * sU
    if model == "M2":
        h = pair_h
        out = []
        for r in test:
            U = r["U"]
            hid = sum(h.get((r["types"][p], r["types"][p + 1]), 0.0) * min(U[p], U[p + 1])
                      for p in range(len(U) - 1))
            out.append(sum(U) - hid)
        return np.array(out)
    Xtr = np.array([feats(r, S, type_keys) for r in train])
    Xte = np.array([feats(r, S, type_keys) for r in test])
    if target == "r":
        ytr = np.array([r["W"] / sum(r["U"]) for r in train])
    else:
        ytr = np.array([math.log(r["W"]) for r in train])
    f = ridge_fit(Xtr, ytr) if model == "M3lin" else gbm_fit(Xtr, ytr)
    yp = f(Xte)
    return yp * sU if target == "r" else np.exp(yp)


def fit_pair_h(rows2):
    """M2: type 쌍별 h. hidden = Ua+Ub−W2 = h·min(Ua,Ub) 최소제곱."""
    num, den = defaultdict(float), defaultdict(float)
    for r in rows2:
        m = min(r["U"])
        num[r["types"]] += (sum(r["U"]) - r["W"]) * m
        den[r["types"]] += m * m
    return {k: num[k] / den[k] for k in num if den[k] > 0}


# ── splits ──────────────────────────────────────────────────────────────────
def splits(rows, kind):
    sizes = sorted({r["dims"] for r in rows})
    seqs = sorted({r["seq"] for r in rows})
    if kind == "E1":
        for s in sizes:
            yield (s, None), (lambda r, s=s: r["dims"] != s), (lambda r, s=s: r["dims"] == s)
    elif kind == "E2":
        for q in seqs:
            yield (None, q), (lambda r, q=q: r["seq"] != q), (lambda r, q=q: r["seq"] == q)
    else:
        for s in sizes:
            for q in seqs:
                yield ((s, q), (lambda r, s=s, q=q: r["dims"] != s and r["seq"] != q),
                       (lambda r, s=s, q=q: r["dims"] == s and r["seq"] == q))


CONFIGS = [("M0", 0, "r"), ("M1", 0, "r"), ("M2", 0, "r"),
           ("M3lin", 0, "r"), ("M3lin", 1, "r"), ("M3lin", 2, "r"), ("M3lin", 3, "r"),
           ("M3gbm", 0, "r"), ("M3gbm", 1, "r"), ("M3gbm", 2, "r"), ("M3gbm", 3, "r"),
           ("M3lin", 2, "logW"), ("M3gbm", 2, "logW")]


_EV = {}


def _eval_task(task):
    L, kind, (model, S, target) = task
    rows_by_L, type_keys = _EV["rows"], _EV["tk"]
    rows = rows_by_L[L]
    errs, preds = [], []
    for key, trf, tef in splits(rows, kind):
        train = [r for r in rows if trf(r)]
        test = [r for r in rows if tef(r)]
        if not test or not train:
            continue
        if _EV.get("upred"):
            test = [with_upred(r, _EV["upred"]) for r in test]
        ph = fit_pair_h([r for r in rows_by_L[2] if trf(r)]) if model == "M2" else None
        wh = fit_predict(model, S, target, train, test, type_keys[L], ph)
        for r, w in zip(test, wh):
            errs.append(abs(w / r["W"] - 1))
            preds.append(((r["dims"], r["seq"]), r["i"], r["W"], float(w)))
    return task, (float(np.mean(errs)) if errs else float("nan"), preds)


def evaluate(rows_by_L, kinds=("E1", "E2", "E3"), configs=CONFIGS, procs=48, upred=None):
    """{(L, model, S, target, split): (MAPE, 행별 (key, i, W, W_hat))}. task별 process 병렬.
    upred가 있으면 test 행의 U를 Module 1 예측값으로 바꾼다 (학습은 측정 U)."""
    import multiprocessing as mp
    _EV["rows"] = rows_by_L
    _EV["upred"] = upred
    _EV["tk"] = {L: sorted({r["types"] for r in rows}) for L, rows in rows_by_L.items()}
    tasks = [(L, k, c) for L in rows_by_L for k in kinds for c in configs]
    with mp.get_context("fork").Pool(min(procs, len(tasks))) as pool:
        out = pool.map(_eval_task, tasks, chunksize=1)
    return {(L, *c, k): v for (L, k, c), v in out}


# ── Step 5: telescope with predicted windows ────────────────────────────────
def tele_range(get, lo, hi, L):
    if L == 1:
        return sum(get(1, i) for i in range(lo, hi))
    return get(L, lo) + sum(get(L, i) - get(L - 1, i) for i in range(lo + 1, hi - L + 1))


def step5(data, res, model, S, target, split="E1", upred=None):
    """덩어리별·합 telescope. 측정(또는 upred면 예측) U와 예측 W2·W3 (split의 held-out 예측)."""
    pred = {}
    for L in (2, 3):
        for key, i, W, wh in res[(L, model, S, target, split)][1]:
            pred[(key, L, i)] = wh
    out = []
    for key, d in sorted(data.items()):
        meas = lambda L, i: d[kn.window_name(i, i + L)]  # noqa: E731
        uh = (lambda i: upred[key][kn.TOKENS[i]]) if upred else (lambda i: d[kn.window_name(i, i + 1)])  # noqa: E731
        hat = lambda L, i: uh(i) if L == 1 else pred[(key, L, i)]  # noqa: E731
        row = {"dims": key[0], "seq": key[1]}
        try:
            for s, (lo, hi) in SUB.items():
                full = d[kn.window_name(lo, hi)]
                row[s] = {"full": full, "uni": tele_range(meas, lo, hi, 1) / full,
                          "tele3": tele_range(meas, lo, hi, 3) / full,
                          "tele3_hat": tele_range(hat, lo, hi, 3) / full,
                          "uni_hat": tele_range(hat, lo, hi, 1) / full,
                          "signs": [float(np.sign(hat(3, i) - meas(3, i))) for i in range(lo, hi - 2)]}
            fAB = row["A"]["full"] + row["B"]["full"]
            for k in ("uni", "uni_hat", "tele3", "tele3_hat"):
                row["AB_" + k] = (row["A"][k] * row["A"]["full"] + row["B"][k] * row["B"]["full"]) / fAB
            row["full15"] = d[kn.window_name(0, N)]
        except KeyError:
            continue
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seqs", default=None, help="쓸 seq 목록 (예: 512,2048)")
    ap.add_argument("--best", default=None, help="Step 5 model, 예: M3gbm,2,r")
    ap.add_argument("--upred", default=None, help="Step 6: Module 1 예측 U 파일 (test 행 U를 바꾼다)")
    ap.add_argument("--configs", default=None, help="평가할 config만, 예: M0,0,r;M3gbm,1,r;M3gbm,2,r")
    ap.add_argument("--kinds", default=None, help="split 목록, 예: E1")
    a = ap.parse_args()
    data = load(a.path)
    if a.seqs:
        keep = {int(x) for x in a.seqs.split(",")}
        data = {k: v for k, v in data.items() if k[1] in keep}
    sizes = sorted({k[0] for k in data})
    seqs = sorted({k[1] for k in data})
    print(f"# {a.path}: 크기 {len(sizes)}종, seq {seqs}")
    rows_by_L = {L: build_rows(data, L) for L in (2, 3)}
    kinds = ("E1", "E2", "E3") if len(seqs) > 1 else ("E1",)
    if a.kinds:
        kinds = tuple(a.kinds.split(","))
    global CONFIGS
    if a.configs:
        CONFIGS = [(m, int(x), t) for m, x, t in (c.split(",") for c in a.configs.split(";"))]
    upred = load_upred(a.upred) if a.upred else None
    if upred:
        errs = [abs(upred[k][t] / d[t] - 1) for k, d in data.items() for t in kn.TOKENS if t in d and t in upred.get(k, {})]
        print(f"# Step 6: U 예측 {a.upred}: 측정 U 대비 MAPE {100 * np.mean(errs):.2f}% (max {100 * np.max(errs):.1f}%, {len(errs)}개)")
    res = evaluate(rows_by_L, kinds, CONFIGS, upred=upred)

    for L in (3, 2):
        print(f"\n## W{L} MAPE (%)  행 {len(rows_by_L[L])}개")
        print(f"{'model':8s} {'S':2s} {'target':6s} " + " ".join(f"{k:>7s}" for k in kinds))
        for model, S, target in CONFIGS:
            print(f"{model:8s} S{S} {target:6s} " + " ".join(
                f"{100 * res[(L, model, S, target, k)][0]:7.2f}" for k in kinds))

    # Step 4 best (E1 W3 기준), 또는 지정
    if a.best:
        m, s, t = a.best.split(",")
        best = (m, int(s), t)
    else:
        best = min(CONFIGS, key=lambda c: res[(3, *c, "E1")][0])
    print(f"\n## 최대 오차 W3 window (best {best}, E1)")
    rows = sorted(res[(3, *best, "E1")][1], key=lambda p: -abs(p[3] / p[2] - 1))[:12]
    for key, i, W, wh in rows:
        print(f"  {key[0]} s{key[1]:5d} {kn.window_name(i, i + 3):22s} W={W:8.2f} hat={wh:8.2f} err={100 * (wh / W - 1):+6.2f}%")

    s5 = step5(data, res, *best, upred=upred)
    print(f"\n## Step 5: 덩어리별 telescope / full (E1 held-out 예측, model {best})")
    print(f"{'dims':24s} {'seq':>5s} | {'A uni':>6s} {'A t3':>6s} {'A t3^':>6s} | {'B uni':>6s} {'B t3':>6s} {'B t3^':>6s} | {'AB uni':>6s} {'AB t3':>6s} {'AB t3^':>6s} | (A+B)/full15")
    for r in s5:
        print(f"{','.join(map(str, r['dims'])):24s} {r['seq']:5d} | "
              f"{r['A']['uni']:6.3f} {r['A']['tele3']:6.3f} {r['A']['tele3_hat']:6.3f} | "
              f"{r['B']['uni']:6.3f} {r['B']['tele3']:6.3f} {r['B']['tele3_hat']:6.3f} | "
              f"{r['AB_uni']:6.3f} {r['AB_tele3']:6.3f} {r['AB_tele3_hat']:6.3f} | "
              f"{(r['A']['full'] + r['B']['full']) / r['full15']:.3f}")
    for k in ("uni", "uni_hat", "tele3", "tele3_hat"):
        for part in ("A", "B"):
            e = [abs(r[part][k] - 1) for r in s5]
            print(f"  {part} {k:9s} MAPE {100 * np.mean(e):6.2f}%  max {100 * np.max(e):6.2f}%")
        e = [abs(r["AB_" + k] - 1) for r in s5]
        print(f"  AB {k:9s} MAPE {100 * np.mean(e):6.2f}%  max {100 * np.max(e):6.2f}%")

    if a.out:
        summ = {f"W{L}|{m}|S{S}|{t}|{k}": v[0] for (L, m, S, t, k), v in res.items()}
        json.dump({"mape": summ, "best": best, "step5": s5}, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
