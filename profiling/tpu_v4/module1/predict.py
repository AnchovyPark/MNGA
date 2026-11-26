#!/usr/bin/env python3
"""Module 1 예측기: kernel type + shape -> 단독 latency U (us, sync_min 기준).

    from predict import predict_u, token_kernel
    predict_u("gemm", M=512, K=2048, N=8192)                 # 같은 입력 반복 기준
    predict_u("gemm", M=512, K=2048, N=8192, regime="copies8")  # 입력 사본 순환 기준
    predict_u("rope", M=512, N=2048, HD=64)

regime
  same    : 같은 입력 반복 측정으로 fitting (공통 문서 §4 기본). gemm weight ≤ 128 MiB는
            이전 실행의 cross-program prefetch로 CMEM에서 읽힌 값이다.
  copies8 : 입력 사본 8개 순환 측정으로 fitting (gemm만 다름. vector는 차이 없음 → same 계수).
            layer마다 weight가 다른 실제 model에 가깝다.

계수: coef_tpu_v4.json (fit_step4.py --out 결과를 regime별로 묶은 것, make_coef()로 생성).
model: gemm = P2 (CMEM 구분 fitted roofline), 나머지 = P1 (a + b·flops + c·bytes).
근거: memory/2026-11-26-tpu-m1.md Step 4.

usage:
  predict.py coef                 # coef_tpu_v4.json 생성 (step3 data로 fitting)
  predict.py upred [regime]       # data_upred[_<regime>]_tpu_v4.txt 생성 (token 15 × 크기 10 × seq 4)
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..")]
import fit_step4 as F  # noqa: E402

COEF_PATH = os.path.join(_HERE, "coef_tpu_v4.json")
MODEL = {"gemm": "P2", "rmsnorm": "P1", "rope": "P1", "silu": "P1", "mul": "P1", "add": "P1"}
DATA = {"same": os.path.join(_HERE, "step3", "data_kernels_tpu_v4.txt"),
        "copies8": os.path.join(_HERE, "step3", "data_kernels_copies8_tpu_v4.txt")}
_COEF = None


def make_coef():
    out = {"target": "sync_min_us", "model": MODEL, "regimes": {}}
    for regime, path in DATA.items():
        if not os.path.exists(path):
            continue
        rows, fails = F.load("sync", path)
        reg = {}
        for t, m in MODEL.items():
            tr = [r for r in rows if r["key"][0] == t]
            if not tr:
                continue
            reg[t] = {"model": m, "coef": [float(c) for c in F.fit(m, tr)], "n_train": len(tr)}
        out["regimes"][regime] = {"data": os.path.relpath(path, _HERE), "n_failed": len(fails),
                                  "types": reg}
    json.dump(out, open(COEF_PATH, "w"), indent=1)
    print("wrote", COEF_PATH, {k: list(v["types"]) for k, v in out["regimes"].items()})


def _coef():
    global _COEF
    if _COEF is None:
        _COEF = json.load(open(COEF_PATH))
    return _COEF


def predict_u(kernel_type, M, K=0, N=0, HD=0, regime="same"):
    """kernel 단독 latency (us). gemm: x[M,K]@w[K,N]. vector: x[M,N] (rope는 HD 필요)."""
    regs = _coef()["regimes"]
    types = regs[regime]["types"]
    if kernel_type not in types:  # copies8은 gemm만 측정. vector는 same과 같다 (Step 1).
        types = regs["same"]["types"]
    c = types[kernel_type]
    import numpy as np
    row = [{"key": (kernel_type, int(M), int(K), int(N), int(HD)), "y": 1.0}]
    return float(F.predict(c["model"], np.array(c["coef"]), row)[0])


def token_kernel(token, D, INTER, NH, HD, KV, S):
    """chain token -> (kernel_type, M, K, N, HD)  (kernels.py CHAIN 정의 그대로)."""
    import kernels as kn
    shp = kn.shapes(D=D, INTER=INTER, NH=NH, HD=HD, KV=KV, S=S)
    _, kind, ins, params = next(c for c in kn.CHAIN if c[0] == token)
    out_n = shp[token][1]
    if kind == "gemm":
        return kind, S, shp[ins[0]][1], out_n, 0
    return kind, S, 0, out_n, (HD if kind == "rope" else 0)


def write_upred(regime="same"):
    import kernels as kn
    from gen_points_step3 import SIZES
    name = "data_upred_tpu_v4.txt" if regime == "same" else f"data_upred_{regime}_tpu_v4.txt"
    path = os.path.join(_HERE, name)
    with open(path, "w") as f:
        f.write(f"# Module 1 예측 U (us, sync_min 기준, regime={regime}). predict.py upred {regime}\n")
        f.write("# UPRED D INTER NH HD KV token seq u_pred_us\n")
        for D, I, NH, HD, KV in SIZES:
            for S in (128, 512, 1024, 2048):
                for tok in kn.TOKENS:
                    t, M, K, N, hd = token_kernel(tok, D, I, NH, HD, KV, S)
                    f.write(f"UPRED {D} {I} {NH} {HD} {KV} {tok} {S} "
                            f"{predict_u(t, M, K, N, hd, regime):.3f}\n")
    print("wrote", path)


if __name__ == "__main__":
    if sys.argv[1] == "coef":
        make_coef()
    elif sys.argv[1] == "upred":
        write_upred(sys.argv[2] if len(sys.argv) > 2 else "same")
