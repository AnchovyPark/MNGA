#!/usr/bin/env python3
"""Module 1 Step 1: 측정 경로 확인 (TPU_MODULES_COMMON.md §5).

1B (2048, 8192, 32, 64, 8), seq 512, token add1(작음) / gate(큼).

usage:
  step1_gate.py check            # TPU 출력 vs CPU 수치 대조 (kernel 6종 + chain window)
  step1_gate.py single <token>   # 이 process에서 token 하나만 측정 (fresh process용)
  step1_gate.py multi            # 한 process에서 여러 window 연달아 측정
  step1_gate.py copies           # 같은 입력 반복 vs 입력 사본 8개 순환
출력 행: S1 <mode> <token> dev_min_us dev_med_us wall_min_us n_samples sync_min_us sync_med_us
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..")]
import jax
import jax.numpy as jnp
import numpy as np

import kernels as kn
import measure as ms

DIMS = dict(D=2048, INTER=8192, NH=32, HD=64, KV=8)
S = 512
SHP = kn.shapes(S=S, **DIMS)
IDX = {t: i for i, t in enumerate(kn.TOKENS)}


def token_window(tok):
    i = IDX[tok]
    return kn.window(i, i + 1)


def row(mode, tok, r):
    print(f"S1 {mode} {tok} {r['dev_min_us']:.3f} {r['dev_med_us']:.3f} "
          f"{r['wall_min_us']:.3f} {r['n_samples']} "
          f"{r['sync_min_us']:.3f} {r['sync_med_us']:.3f}", flush=True)


def run_token(mode, tok):
    fn, an, _ = token_window(tok)
    args = ms.put(kn.make_args(an, SHP))
    r = ms.measure(fn, args)
    row(mode, tok, r)
    return r


def check():
    """작은 크기에서 TPU 결과를 CPU(fp32 기준)와 비교."""
    cpu = jax.devices("cpu")[0]
    shp = kn.shapes(D=256, INTER=512, NH=4, HD=64, KV=2, S=64)
    cases = [(i, i + 1) for i in range(len(kn.CHAIN))] + [(0, 6), (6, 15), (0, 15)]
    worst = 0.0
    for lo, hi in cases:
        fn, an, on = kn.window(lo, hi)
        args = kn.make_args(an, shp)
        tpu = jax.jit(fn)(*ms.put(args))
        ref = jax.jit(fn)(*[jax.device_put(a.astype(jnp.float32), cpu) for a in args])
        for n, a, b in zip(on, tpu, ref):
            a = np.asarray(a, np.float32)
            b = np.asarray(b, np.float32)
            rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
            worst = max(worst, rel)
            print(f"CHECK {kn.window_name(lo, hi)} out={n} shape={a.shape} relerr={rel:.4f}")
    print(f"CHECK worst_relerr={worst:.4f}")


def multi():
    seq = ["add1", "gate", "norm1", "q", "silu", "up", "mul", "down",
           "add1", "gate", "add2", "rope_q", "add1", "gate"]
    for tok in seq:
        run_token("multi", tok)


def copies():
    for tok in ["add1", "gate"]:
        fn, an, _ = token_window(tok)
        same = ms.put(kn.make_args(an, SHP))
        row("same", tok, ms.measure(fn, same))
        sets = [ms.put(kn.make_args(an, SHP, seed=s)) for s in range(8)]
        row("copies8", tok, ms.measure(fn, sets, n_copies=8))
        # kernels.py의 window와 KERNELS[type] 단독 정의가 같은 값을 내는지
        kind = kn.CHAIN[IDX[tok]][1]
        row("kernel_fn", tok, ms.measure(kn.KERNELS[kind], same))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "check":
        check()
    elif mode == "single":
        run_token("fresh", sys.argv[2])
    elif mode == "multi":
        multi()
    elif mode == "copies":
        copies()
    sys.stdout.flush()
    os._exit(0)
