#!/usr/bin/env python3
"""Step 2 trace: layer 사이 겹침이 어디서 생기나 (core 0 'XLA Ops' line, 실행 1회 = sync_min 실행).

usage: trace_stack.py <model> <seq> <mode> <N> [<N> ...]   → trace_<model>_<seq>_<mode>.log
  N마다 make_stack을 x만 사본 2개로 측정하고 trace를 남겨,
  - op별 시간 (실행 30회 평균, 이름 그대로), 실행 1회의 op 순서·시작 시각
  - HLO의 async copy(copy-start/done)·CMEM(S(3)) 배치된 parameter
  를 출력한다. N=1의 op 시간과 N=2의 layer별 op 시간을 비교하면 어느 op가 짧아졌는지(겹침) 보인다.
"""
import glob
import os
import re
import sys
import tempfile
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import mlib  # noqa: E402
from jax.profiler import ProfileData  # noqa: E402

import jax  # noqa: E402


def weight_prefetch(txt):
    """HLO에서 CMEM(S(3))으로 미리 옮겨지는 layer parameter.
    - copy-start(..., cross_program_prefetch_index): 다음 실행용 prefetch (Step 1 artifact)
    - slice-start(lp_K) → slice-done → ConcatBitcast custom-call (S(3) 출력): program 안 async prefetch
    lp_K = make_stack 인자 순서 K → layer K // 9, PARAM_NAMES[K % 9]."""
    import layer as ly
    defs = {}
    for ln in txt.splitlines():
        m = re.match(r"\s*(?:ROOT )?(%[\w.\-]+) = (.*)", ln)
        if m:
            defs[m.group(1)] = m.group(2)

    def pname(p):
        m = re.match(r"%lp_(\d+)_", p)
        return f"L{int(m.group(1)) // 9}.{ly.PARAM_NAMES[int(m.group(1)) % 9]}" if m else p

    xp = [pname(re.search(r"copy-start\((%[\w.\-]+)\)", v).group(1)) for v in defs.values()
          if "copy-start(" in v and "cross_program_prefetch" in v]
    inprog = []
    for k, v in defs.items():
        if 'custom_call_target="ConcatBitcast"' in v and "S(3)" in v.split("custom-call(")[0]:
            srcs = set()
            for sd in re.findall(r"(%slice-done[\w.\-]*)", v.split("custom_call_target")[0]):
                ss = re.search(r"slice-done\((%[\w.\-]+)\)", defs.get(sd, ""))
                if ss:
                    src = re.search(r"slice-start\((%[\w.\-]+)", defs.get(ss.group(1), ""))
                    if src:
                        srcs.add(pname(src.group(1)))
            inprog.append(f"{k}<-{','.join(sorted(srcs))}")
    ncs = sum("copy-start(" in v for v in defs.values())
    return xp, inprog, ncs


def ops_of(xplane, module, iters):
    pd = ProfileData.from_file(xplane)
    lines = set()
    mods, ops, aops = [], [], []
    for pl in pd.planes:
        if pl.name != "/device:TPU:0":
            continue
        for ln in pl.lines:
            lines.add(ln.name)
            if ln.name == "XLA Modules":
                mods += [(e.start_ns, e.start_ns + e.duration_ns) for e in ln.events
                         if e.name.split("(")[0] == module]
            if ln.name == "XLA Ops":
                ops += [(e.start_ns, e.duration_ns, e.name) for e in ln.events]
            if ln.name == "Async XLA Ops":
                aops += [(e.start_ns, e.duration_ns, e.name) for e in ln.events]
    mods.sort()
    ops.sort()
    per = []
    aper = []
    for s, t in mods:
        per.append([(o - s, d, nm) for o, d, nm in ops if s <= o < t])
        aper.append([(o - s, d, nm) for o, d, nm in aops if s <= o < t])
    return sorted(lines), mods, per, aper


def main():
    name, seq, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    Ns = list(map(int, sys.argv[4:]))
    dims, _ = mlib.MODELS[name]
    out = open(os.path.join(_HERE, f"trace_{name}_{seq}_{mode}.log"), "w")

    def pr(*a):
        print(*a)
        print(*a, file=out)
        out.flush()

    for N in Ns:
        fn, sets = mlib.stack_job(dims, N, seq, mode)
        compiled = jax.jit(fn).lower(*sets[0]).compile()
        txt = compiled.as_text()
        tdir = tempfile.mkdtemp(prefix="mdtr_")
        r = mlib.measure(fn, sets, keep_trace=tdir)
        xp = glob.glob(os.path.join(tdir, "**", "*.xplane.pb"), recursive=True)[-1]
        lines, mods, per, aper = ops_of(xp, r["module"], mlib.ITERS)
        pr(f"\n==== {name} seq {seq} {mode} N={N}: sync_min {r['sync_min_us']:.1f} us  lines={lines}")
        xpf, inprog, ncs = weight_prefetch(txt)
        pr(f"  HLO copy-start {ncs}개, cross-program prefetch: {xpf}")
        pr(f"  HLO program 안 weight prefetch (slice-start→ConcatBitcast, S(3)) {len(inprog)}: " + "; ".join(inprog))
        s3 = [ln.strip()[:140] for ln in txt.splitlines() if "S(3)" in ln and "parameter(" in ln]
        pr(f"  HLO parameter in S(3) {len(s3)}: " + " | ".join(s3[:6]))
        acc = defaultdict(float)
        for ex in per:
            for _, d, nm in ex:
                acc[nm] += d / 1e3 / len(per)
        tot = sum(acc.values())
        pr(f"  core0 op sum {tot:.1f} us (실행 {len(per)}회 평균), op {len(acc)}종")
        # 가장 짧은 실행 1회의 순서
        k = min(range(len(mods)), key=lambda i: mods[i][1] - mods[i][0])
        pr(f"  -- 실행 {k} (module {(mods[k][1] - mods[k][0]) / 1e3:.1f} us) op 순서: start_us dur_us name")
        for o, d, nm in per[k]:
            pr(f"   {o / 1e3:9.1f} {d / 1e3:8.1f}  {nm[:150]}")
        pr(f"  -- 같은 실행의 Async XLA Ops (start_us dur_us name | 그 동안 돈 XLA Ops)")
        for o, d, nm in aper[k]:
            during = sorted({n.split(" = ")[0] for oo, dd, n in per[k] if oo < o + d and oo + dd > o})
            pr(f"   {o / 1e3:9.1f} {d / 1e3:8.1f}  {nm.split(' = ')[0][:40]:40s} | {' '.join(during)[:160]}")
        open(os.path.join(tdir, "hlo.txt"), "w").write(txt)
        pr(f"  (trace dir {tdir})")
        del sets
        mlib.free()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
