#!/usr/bin/env python3
"""점 목록을 측정해 KROW 행으로 append (이미 있는 key는 건너뜀 = resume 가능).

usage: run_points.py <points.txt> <out_data.txt> [--copies N]
  points.txt 각 행: <type> <M> <K> <N> <HD>   (gemm: x[M,K]@w[K,N], HD=0 /
                                              vector: x[M,N], K=0, rope만 HD)
  --copies N : 입력 사본 N개를 돌려 쓰며 측정 (기본 1 = 같은 입력 반복)

출력 (out_data.txt):
  KROW type M K N HD dev_min_us dev_med_us wall_min_us n_samples sync_min_us sync_med_us
    dev_*  = TPU_MODULES_COMMON.md §4 정의 (core event union, megacore launch skew 포함)
    sync_* = 두 core 모두 시작한 뒤 시간 (Module 1 대표값, memory/2026-11-26-tpu-m1.md Step 1)
  실패 행은 dev/sync = 0으로 남긴다.
HLO 연산 목록: out_data.txt와 같은 dir의 <out stem>_hlo.txt
  HLO type M K N HD <ops>
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..")]
import jax
import jax.numpy as jnp

import kernels as kn
import measure as ms


def arg_shapes(t, M, K, N, HD):
    if t == "gemm":
        return [(M, K), (K, N)]
    if t == "rmsnorm":
        return [(M, N), (N,)]
    if t == "rope":
        return [(M, N), (1, 1, HD), (1, 1, HD)]
    if t == "silu":
        return [(M, N)]
    if t in ("mul", "add"):
        return [(M, N), (M, N)]
    raise ValueError(t)


def make(shps, seed):
    keys = jax.random.split(jax.random.PRNGKey(seed), len(shps))
    return ms.put([jax.random.normal(k, s, jnp.float32).astype(kn.DT)
                   for k, s in zip(keys, shps)])


def key_of(parts):
    return " ".join(parts[1:6])


def main():
    pts_path, out_path = sys.argv[1], sys.argv[2]
    copies = int(sys.argv[sys.argv.index("--copies") + 1]) if "--copies" in sys.argv else 1
    hlo_path = os.path.splitext(out_path)[0] + "_hlo.txt"
    done = set()
    if os.path.exists(out_path):
        for ln in open(out_path):
            p = ln.split()
            if p and p[0] == "KROW":
                done.add(key_of(p))
    pts = [ln.split() for ln in open(pts_path) if ln.strip() and not ln.startswith("#")]
    todo = [p for p in pts if " ".join(p) not in done]
    print(f"[run_points] {len(pts)} points, {len(pts) - len(todo)} done, {len(todo)} todo, copies={copies}",
          flush=True)
    t_start = time.time()
    with open(out_path, "a") as fo, open(hlo_path, "a") as fh:
        for i, p in enumerate(todo):
            t, M, K, N, HD = p[0], *map(int, p[1:])
            shps = arg_shapes(t, M, K, N, HD)
            fn = kn.KERNELS[t]
            r = None
            for attempt in range(2):
                try:
                    if copies > 1:
                        args = [make(shps, s) for s in range(copies)]
                    else:
                        args = make(shps, 0)
                    r = ms.measure(fn, args, n_copies=copies)
                    del args
                    if r["n_samples"] == ms.ITERS and r["sync_min_us"] > 0:
                        break
                except Exception as e:  # noqa: BLE001  실패 행으로 남긴다
                    print(f"[run_points] ERROR {p}: {type(e).__name__}: {e}", flush=True)
                    r = None
            if r is None:
                r = {"dev_min_us": 0.0, "dev_med_us": 0.0, "wall_min_us": 0.0, "n_samples": 0,
                     "sync_min_us": 0.0, "sync_med_us": 0.0, "hlo_ops": "ERROR"}
            fo.write(f"KROW {t} {M} {K} {N} {HD} {r['dev_min_us']:.3f} {r['dev_med_us']:.3f} "
                     f"{r['wall_min_us']:.3f} {r['n_samples']} {r['sync_min_us']:.3f} "
                     f"{r['sync_med_us']:.3f}\n")
            fo.flush()
            fh.write(f"HLO {t} {M} {K} {N} {HD} {r['hlo_ops']}\n")
            fh.flush()
            if (i + 1) % 20 == 0 or i + 1 == len(todo):
                el = time.time() - t_start
                print(f"[run_points] {i + 1}/{len(todo)}  {el:.0f}s  ({el / (i + 1):.2f}s/pt)", flush=True)
            jax.clear_caches()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
