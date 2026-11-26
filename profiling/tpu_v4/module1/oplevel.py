#!/usr/bin/env python3
"""한 kernel 실행 안의 HLO op별 시간 (core 0 plane의 'XLA Ops' line, 실행 1회 평균).
usage: oplevel.py <type> <M|S> <K|NH> <N|KV> <HD> [...같은 형식 반복]
attention type은 run_points.py와 같이 S NH KV HD."""
import os
import sys
import glob
import tempfile
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..")]
from jax.profiler import ProfileData  # noqa: E402

import kernels as kn  # noqa: E402
import kernels_attn as ka  # noqa: E402
import measure as ms  # noqa: E402
from run_points import arg_shapes, make  # noqa: E402

a = sys.argv[1:]
for i in range(0, len(a), 5):
    t, M, K, N, HD = a[i], *map(int, a[i + 1:i + 5])
    fn = ka.make(t, K, N, HD or 128) if t in ka.TYPES else kn.KERNELS[t]
    tdir = tempfile.mkdtemp(prefix="m1op_")
    r = ms.measure(fn, make(arg_shapes(t, M, K, N, HD), 0), keep_trace=tdir)
    pd = ProfileData.from_file(glob.glob(os.path.join(tdir, "**", "*.xplane.pb"), recursive=True)[-1])
    acc = defaultdict(float)
    for pl in pd.planes:
        if pl.name != "/device:TPU:0":
            continue
        for ln in pl.lines:
            if ln.name == "XLA Ops":
                for e in ln.events:
                    acc[e.name] += e.duration_ns / 1e3 / ms.ITERS
    tot = sum(acc.values())
    print(f"\n{t} {M} {K} {N} {HD}: sync_min {r['sync_min_us']:.1f} us, core0 op sum {tot:.1f} us")
    for k, v in sorted(acc.items(), key=lambda kv: -kv[1])[:8]:
        print(f"   {v:9.1f} us {v / tot:6.1%}  {k}")
sys.stdout.flush()
os._exit(0)
