#!/usr/bin/env python3
"""Module 2 측정 라이브러리: window 하나의 device time을 profiler trace로 잰다.

정의 (TPU_MODULES_COMMON.md §4):
- device time = XLA executable 1회 실행이 device에서 차지한 시간.
  jax.profiler.trace의 TPU device plane, 'XLA Modules' line의 event duration.
- TPU v4는 chip 하나에 TensorCore 2개(megacore). plane /device:TPU:0, :1 이
  같은 실행에 대해 event를 하나씩 남긴다. 시간이 겹치는 event를 실행 1회로 묶고
  첫 시작~마지막 종료를 그 실행의 시간으로 본다.
- warmup 20회, 한 trace 안에서 30회 반복. 대표값 min, median과 wall-clock min도 기록.

실행 환경: ~/venvs/vm3_jax (jax 0.11.2). system python의 jax 0.6.2에는
jax.profiler.ProfileData가 없다.
"""
import glob
import os
import re
import shutil
import sys
import tempfile
import time

import jax
import numpy as np
from jax.profiler import ProfileData

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import kernels as kn  # noqa: E402

WARMUP = 20
REPS = 30


def _executions(xplane_path, jit_name):
    """trace에서 jit_name 실행 목록.

    반환: (execs, n_events). execs의 원소는 (union_ns, sync_ns):
    - union_ns: core 사이 겹치는 event를 묶은 첫 시작~마지막 종료 (공통 문서 §4 정의).
    - sync_ns : 마지막 종료 − 가장 늦은 시작. 두 core의 launch 시점 차(skew)를 뺀 값.
      v4에서 core 0이 core 1보다 13~30µs 먼저 시작해 기다리는 일이 무작위로 생겨
      union에 noise로 들어간다 (2026-11-26 확인, memory/2026-11-26-tpu-m2.md).
    """
    pd = ProfileData.from_file(xplane_path)
    ivs = []
    for pl in pd.planes:
        if not pl.name.startswith("/device:TPU:"):
            continue
        for line in pl.lines:
            if line.name != "XLA Modules":
                continue
            for e in line.events:
                # event 이름: "jit_<name>(<fingerprint>)"
                if e.name.split("(")[0] == jit_name:
                    ivs.append((e.start_ns, e.start_ns + e.duration_ns))
    ivs.sort()
    groups = []
    for s, t in ivs:
        if groups and s <= groups[-1][1]:
            g = groups[-1]
            g[1] = max(g[1], t)
            g[2] = max(g[2], s)
        else:
            groups.append([s, t, s])  # [first start, last end, latest start]
    execs = [(t - s, t - ls) for s, t, ls in groups]
    return execs, len(ivs)


def hlo_ops(compiled):
    """compile된 HLO의 연산 종류 목록 (ENTRY 및 fusion 안 포함, 정렬된 고유 이름).

    fusion은 kind까지 (fusion:kLoop 등), custom-call은 target까지 붙인다.
    """
    txt = compiled.as_text()
    ops = set()
    for m in re.finditer(r"=\s*[^=]*?\s([a-z][a-z0-9\-]*)\(", txt):
        ops.add(m.group(1))
    for m in re.finditer(r"kind=(k[A-Za-z]+)", txt):
        ops.add("fusion:" + m.group(1))
    for m in re.finditer(r'custom_call_target="([^"]+)"', txt):
        ops.add("custom-call:" + m.group(1))
    ops.discard("parameter")
    return sorted(ops)


def measure_fn(fn, args, tag, warmup=WARMUP, reps=REPS, n_copies=1):
    """fn을 jit해서 device time과 wall time을 잰다.

    n_copies>1이면 입력 사본을 n_copies개 만들어 돌려 쓴다 (Step 1 비교용).
    반환 dict: dev_us(list), wall_us(list), n_events, hlo_ops.
    """
    dev = jax.devices()[0]
    jit_name = "jit_" + tag
    fn = _named(fn, tag)
    f = jax.jit(fn)
    argsets = [[jax.device_put(a, dev) for a in args]]
    for c in range(1, n_copies):
        argsets.append([jax.device_put(a + 0, dev) for a in args])  # 다른 buffer
    jax.block_until_ready(argsets)
    compiled = f.lower(*argsets[0]).compile()
    ops = hlo_ops(compiled)

    for i in range(warmup):
        jax.block_until_ready(f(*argsets[i % n_copies]))

    # wall-clock (trace 밖)
    wall = []
    for i in range(reps):
        a = argsets[i % n_copies]
        t0 = time.perf_counter()
        jax.block_until_ready(f(*a))
        wall.append((time.perf_counter() - t0) * 1e6)

    d = tempfile.mkdtemp(prefix="m2trace_")
    try:
        with jax.profiler.trace(d):
            for i in range(reps):
                jax.block_until_ready(f(*argsets[i % n_copies]))
        paths = glob.glob(os.path.join(d, "**", "*.xplane.pb"), recursive=True)
        if not paths:
            raise RuntimeError("no xplane.pb")
        execs, n_ev = _executions(paths[0], jit_name)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    dev_us = [u / 1e3 for u, _ in execs]
    sync_us = [y / 1e3 for _, y in execs]
    return {"dev_us": dev_us, "sync_us": sync_us, "wall_us": wall, "n_events": n_ev,
            "n_exec": len(execs), "hlo_ops": ops}


def _named(fn, tag):
    def wrapped(*a):
        return fn(*a)
    wrapped.__name__ = tag
    wrapped.__qualname__ = tag
    return wrapped


def measure_window(lo, hi, shp, reps=REPS, n_copies=1, seed=0):
    fn, arg_names, _ = kn.window(lo, hi)
    args = kn.make_args(arg_names, shp, seed=seed)
    tag = "w%02d_%02d" % (lo, hi)
    r = measure_fn(fn, args, tag, reps=reps, n_copies=n_copies)
    # core 2개가 각각 event를 남기므로 event 수 = 2 × reps 여야 한다
    r["ok"] = (r["n_exec"] == reps and r["n_events"] == 2 * reps
               and min(r["dev_us"]) > 0)
    return r


def summarize(r):
    dv = np.array(r["dev_us"]) if r["dev_us"] else np.array([np.nan])
    sy = np.array(r.get("sync_us") or [np.nan])
    return {"dev_min_us": float(np.min(dv)), "dev_med_us": float(np.median(dv)),
            "sync_min_us": float(np.min(sy)), "sync_med_us": float(np.median(sy)),
            "wall_min_us": float(np.min(r["wall_us"])), "n_samples": r["n_exec"]}


def env_info():
    import importlib.metadata as md
    d = jax.devices()[0]
    v = {}
    for p in ("jax", "jaxlib", "libtpu"):
        try:
            v[p] = md.version(p)
        except Exception:
            v[p] = "?"
    return {"device_kind": d.device_kind, "device_count": jax.device_count(),
            "python": sys.version.split()[0], **v}
