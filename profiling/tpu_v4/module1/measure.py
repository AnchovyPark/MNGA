#!/usr/bin/env python3
"""TPU device-time 측정 (TPU_MODULES_COMMON.md §4).

측정값 = XLA executable 1회 실행이 device에서 차지한 시간.
  - jax.profiler.trace로 trace를 뜨고, /device:TPU:* plane의 'XLA Modules' line에서
    측정 대상 module 이름의 event를 모은다.
  - TPU v4는 chip당 TensorCore 2개(megacore)라 core마다 event가 하나씩 남는다
    (plane /device:TPU:0, /device:TPU:1). 시간이 겹치는 event를 실행 1회로 묶고
    첫 시작 ~ 마지막 종료를 그 실행의 시간으로 본다.
  - warmup 20회 후 한 trace 안에서 iters(30)회. 대표값 min, median도 기록.
  - wall-clock min은 trace 밖에서 같은 횟수를 block_until_ready로 따로 잰다.

NPU-LENS의 measure_hiding_tpu.py에는 이 VM에서 접근할 수 없어 위 정의대로 새로 구현했다.
"""
import glob
import os
import re
import shutil
import statistics
import tempfile
import time

import jax
from jax.profiler import ProfileData

WARMUP = 20
ITERS = 30


def device():
    return jax.devices()[0]


def put(args, dev=None):
    dev = dev or device()
    return [jax.device_put(a, dev) for a in args]


def module_name(compiled):
    """compiled HLO text의 'HloModule <name>' -> name."""
    m = re.match(r"HloModule\s+([^\s,]+)", compiled.as_text())
    return m.group(1) if m else None


_OPC = re.compile(r"=\s*(?:\([^=]*?\)|\S+)\s+([a-z][a-z0-9\-]*)\(")


def hlo_ops(compiled):
    """compiled HLO의 연산 종류 목록 (정렬된 문자열).

    - opcode (fusion/parameter/tuple 등 구조용 제외)
    - fusion kind (kLoop/kInput/kOutput/kCustom ...)
    - custom-call target
    """
    txt = compiled.as_text()
    ops = set()
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith(("HloModule", "}", "ENTRY", "%fused", "fused")) and "=" not in line:
            continue
        m = _OPC.search(line)
        if m:
            op = m.group(1)
            if op not in ("parameter", "tuple", "get-tuple-element", "constant", "bitcast"):
                ops.add(op)
        for k in re.findall(r"kind=(k\w+)", line):
            ops.add("fusion:" + k)
        for t in re.findall(r'custom_call_target="([^"]+)"', line):
            ops.add("cc:" + t)
    return ",".join(sorted(ops))


def _executions(xplane_path, name):
    """trace에서 name module의 실행별 core event 묶음.

    반환: [(union_us, sync_us), ...]
      union_us = 첫 시작 ~ 마지막 종료 (TPU_MODULES_COMMON.md §4 정의)
      sync_us  = 마지막 시작 ~ 마지막 종료 = 두 core가 모두 시작한 뒤의 시간.
                 v4 megacore에서 core0 event는 core1이 뜰 때까지 기다리는
                 launch skew(12~46us, 매번 다름)를 포함한다. sync_us는 이를 뺀다.
    """
    pd = ProfileData.from_file(xplane_path)
    iv = []
    for pl in pd.planes:
        if not pl.name.startswith("/device:TPU:"):
            continue
        for ln in pl.lines:
            if ln.name != "XLA Modules":
                continue
            for e in ln.events:
                if e.name.split("(")[0] == name:
                    iv.append((e.start_ns, e.start_ns + e.duration_ns))
    iv.sort()
    groups = []
    for s, t in iv:
        if groups and s <= groups[-1]["end"]:
            g = groups[-1]
            g["end"] = max(g["end"], t)
            g["last_start"] = max(g["last_start"], s)
        else:
            groups.append({"start": s, "end": t, "last_start": s})
    return [((g["end"] - g["start"]) / 1e3, (g["end"] - g["last_start"]) / 1e3)
            for g in groups]


def measure(fn, args, warmup=WARMUP, iters=ITERS, n_copies=1, keep_trace=None):
    """jit(fn)을 args로 측정. args는 device에 올라간 배열 list.

    n_copies > 1이면 같은 shape의 입력 사본 n_copies개를 돌려가며 쓴다
    (args가 list of list로 주어져야 함).

    반환 dict: dev_min_us, dev_med_us, sync_min_us, sync_med_us, wall_min_us, n_samples, hlo_ops, module.
    측정 실패 시 dev_* = 0.0, n_samples = 실제 잡힌 실행 수.
    """
    arg_sets = args if n_copies > 1 else [args]
    jf = jax.jit(fn)
    compiled = jf.lower(*arg_sets[0]).compile()
    name = module_name(compiled)
    ops = hlo_ops(compiled)

    def run(i):
        out = compiled(*arg_sets[i % len(arg_sets)])
        jax.block_until_ready(out)

    for i in range(warmup):
        run(i)

    # wall-clock
    walls = []
    for i in range(iters):
        t0 = time.perf_counter()
        run(i)
        walls.append((time.perf_counter() - t0) * 1e6)

    # device time
    tdir = keep_trace or tempfile.mkdtemp(prefix="m1tr_")
    try:
        with jax.profiler.trace(tdir):
            for i in range(iters):
                run(i)
        paths = glob.glob(os.path.join(tdir, "**", "*.xplane.pb"), recursive=True)
        ex = _executions(paths[-1], name) if paths else []
    finally:
        if keep_trace is None:
            shutil.rmtree(tdir, ignore_errors=True)

    durs = [u for u, _ in ex]
    syncs = [c for _, c in ex]
    ok = len(durs) == iters and min(durs, default=0) > 0
    return {
        "dev_min_us": min(durs) if ok else 0.0,
        "dev_med_us": statistics.median(durs) if ok else 0.0,
        "sync_min_us": min(syncs) if ok else 0.0,
        "sync_med_us": statistics.median(syncs) if ok else 0.0,
        "wall_min_us": min(walls),
        "n_samples": len(durs),
        "hlo_ops": ops,
        "module": name,
    }
