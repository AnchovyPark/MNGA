#!/usr/bin/env python3
"""TPU_NEXT_LAYER Step 1 HLO·trace 확인: full layer window에서 XLA가 실행한 op 순서와 각 op이 담은 kernel.

사용: trace_order.py --chain L16 --dims 2048,8192,32,64,8 --seq 512 [--copies 2]
- compile된 HLO의 ENTRY 명령마다 출력 shape과, 그 명령(및 fusion 안)이 온 kernel 함수
  (metadata source_line → kernels.py / kernels_attn.py 함수 이름)를 찾는다.
- profiler trace의 TPU:0 'XLA Ops' line에서 실행 1회의 op 순서와 시간을 뽑아 둘을 잇는다.
- custom-call 여부, attn이 v·o gemm과 같은 fusion에 들어가는지 본다.
"""
import argparse
import glob
import inspect
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))


def func_lines():
    import kernels as kn
    import kernels_attn as ka
    out = []
    for mod in (kn, ka):
        f = os.path.basename(mod.__file__)
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if fn.__module__ != mod.__name__:
                continue
            src, start = inspect.getsourcelines(fn)
            out.append((f, start, start + len(src) - 1, name))
    return out


def kernel_of(file, line, fl):
    base = os.path.basename(file)
    best = None
    for f, a, b, name in fl:
        if f == base and a <= line <= b and (best is None or b - a < best[1] - best[0]):
            best = (a, b, name)
    return best[2] if best else None


def frame_funcs(txt):
    """새 HLO 형식: metadata의 stack_frame_id → 함수 이름 (FunctionNames/FileLocations/StackFrames 표)."""
    def table(title):
        m = re.search(title + r"\n((?:\d+ .*\n)+)", txt)
        return dict(re.findall(r"^(\d+) (.*)$", m.group(1), re.M)) if m else {}
    fnames = {k: v.strip('"') for k, v in table("FunctionNames").items()}
    locs = {k: re.search(r"function_name_id=(\d+)", v).group(1) for k, v in table("FileLocations").items()}
    out = {}
    for k, v in table("StackFrames").items():
        m = re.search(r"file_location_id=(\d+)", v)
        if m and m.group(1) in locs:
            out[k] = fnames.get(locs[m.group(1)], "?").replace("make.<locals>.", "")
    return out


SKIP_FUNCS = {"<module>", "window.<locals>.fn", "fn", "main", "?"}


def parse_hlo(txt, fl):
    frames = frame_funcs(txt)
    comps, cur, entry = {}, None, None
    for line in txt.splitlines():
        m = re.match(r"^(ENTRY\s+)?%?([\w.\-]+)\s.*\{\s*$", line)
        if m and not line.startswith(" "):
            cur = m.group(2)
            comps[cur] = []
            if m.group(1):
                entry = cur
            continue
        if line.startswith("}"):
            cur = None
            continue
        if cur is not None and "=" in line:
            comps[cur].append(line.strip())

    def meta_kernels(ins):
        ks = set()
        for f, l in re.findall(r'source_file="([^"]+)" source_line=(\d+)', ins):
            k = kernel_of(f, int(l), fl)
            if k:
                ks.add(k)
        for fid in re.findall(r"stack_frame_id=(\d+)", ins):
            k = frames.get(fid)
            if k and k not in SKIP_FUNCS:
                ks.add(k)
        return ks

    def comp_kernels(c, depth=0):
        ks = set()
        for ins in comps.get(c, []):
            ks |= meta_kernels(ins)
            if depth < 3:
                for sub in re.findall(r"(?:calls|to_apply)=%?([\w.\-]+)", ins):
                    ks |= comp_kernels(sub, depth + 1)
        return ks

    info = {}
    for ins in comps.get(entry, []):
        m = re.match(r"%?([\w.\-]+)\s*=\s*(\S+(?:\[[^\]]*\])?(?:\{[^}]*\})?)\s+([\w\-]+)\(", ins)
        if not m:
            continue
        name, shape, op = m.groups()
        ks = meta_kernels(ins)
        for sub in re.findall(r"calls=%?([\w.\-]+)", ins):
            ks |= comp_kernels(sub)
        kind = re.search(r"kind=(k\w+)", ins)
        cc = re.search(r'custom_call_target="([^"]+)"', ins)
        info[name] = {"shape": shape, "op": op + (":" + kind.group(1) if kind else "") + (":" + cc.group(1) if cc else ""),
                      "kernels": sorted(ks)}
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default="L16")
    ap.add_argument("--dims", default="2048,8192,32,64,8")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=None)
    a = ap.parse_args()
    import jax
    from jax.profiler import ProfileData
    import layer_chain as lc

    dims = tuple(map(int, a.dims.split(",")))
    hi = a.hi or len(lc.CHAINS[a.chain])
    fn, an, outs = lc.window(a.chain, a.lo, hi, dims)
    shp = lc.shapes(*dims, a.seq)
    args = lc.make_args(an, shp)
    args2 = [x + 0 for x in args]
    f = jax.jit(fn)
    comp = f.lower(*args).compile()
    info = parse_hlo(comp.as_text(), func_lines())
    for _ in range(10):
        jax.block_until_ready(f(*args))
        jax.block_until_ready(f(*args2))
    d = tempfile.mkdtemp()
    with jax.profiler.trace(d):
        for i in range(4):
            jax.block_until_ready(f(*(args if i % 2 == 0 else args2)))
    pb = glob.glob(os.path.join(d, "**", "*.xplane.pb"), recursive=True)[0]
    pd = ProfileData.from_file(pb)
    ops = None
    for pl in pd.planes:
        if pl.name != "/device:TPU:0":
            continue
        mods = [e for ln in pl.lines if ln.name == "XLA Modules" for e in ln.events]
        mods.sort(key=lambda e: e.start_ns)
        last = mods[-1]
        ops = [e for ln in pl.lines if ln.name == "XLA Ops" for e in ln.events
               if last.start_ns <= e.start_ns <= last.start_ns + last.duration_ns]
        ops.sort(key=lambda e: e.start_ns)
        print(f"# {a.chain} {a.dims} seq {a.seq} window {a.lo}:{hi} outs={outs}  module {last.duration_ns / 1e3:.1f} us (TPU:0), op {len(ops)}개")
    shutil.rmtree(d, ignore_errors=True)
    t0 = ops[0].start_ns
    for e in ops:
        nm = re.match(r"%?([\w.\-]+)", e.name).group(1)
        i = info.get(nm, {})
        if e.duration_ns < 500 and not i.get("kernels"):
            continue     # 0.5us 미만이고 kernel 없는 bookkeeping (copy-start 등) 생략
        print(f"  {(e.start_ns - t0) / 1e3:8.1f} {e.duration_ns / 1e3:7.1f}us  {nm:28s} {i.get('op', '?'):28s} "
              f"{i.get('shape', ''):32s} {','.join(i.get('kernels', []))}")


if __name__ == "__main__":
    main()
