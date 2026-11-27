#!/usr/bin/env python3
"""job 목록을 측정해 LROW/MROW 행으로 append (chip 하나 = process 하나, run_jobs.sh가 4개 띄운다).

usage: run_model.py <jobs.txt> <out_part.txt> <claims_dir> <done_glob>
  jobs.txt 각 행 (model = 1B|3B|8B 또는 D,INTER,NH,HD,KV):
    L <model> <n_layers> <seq> <mode> [wcopies]     make_stack (mode unroll|scan) 또는 layer (layer.make_layer)
    M <model> <seq> <mode> <part> [n_layers]        part full (layer.make_model) | emb | head
  wcopies=2: weight까지 사본 2개 (Step 1 artifact 비교). 행의 mode 칸에 '<mode>_w2'로 쓴다.
  여러 process가 같은 jobs를 보고 claims_dir의 파일(O_EXCL)로 job을 나눠 갖는다.
  done_glob에 걸리는 파일에 이미 있는 key는 건너뛴다 (resume).

출력 (TPU_NEXT_MODEL §5):
  LROW D INTER NH HD KV n_layers seq mode sync_min_us sync_med_us wall_min_us n_samples
  MROW D INTER NH HD KV n_layers vocab seq mode part sync_min_us sync_med_us wall_min_us n_samples
  실패 행은 0으로 남긴다. <out>_info.txt: INFO <key> compile_s peak_GiB arg/temp/out GiB n_hlo_ops error
"""
import glob
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import mlib  # noqa: E402


def dims_of(s):
    if s in mlib.MODELS:
        return mlib.MODELS[s]
    return tuple(map(int, s.split(","))), None


def row_key(job):
    """job -> 출력 행의 key 부분 (LROW/MROW 뒤 값 칸 앞까지)."""
    p = job.split()
    dims, n0 = dims_of(p[1])
    d = " ".join(map(str, dims))
    if p[0] == "L":
        n, seq, mode = int(p[2]), int(p[3]), p[4]
        w = int(p[5]) if len(p) > 5 else 1
        return f"LROW {d} {n} {seq} {mode}{'_w2' if w == 2 else ''}"
    seq, mode, part = int(p[2]), p[3], p[4]
    n = int(p[5]) if len(p) > 5 else n0
    return f"MROW {d} {n} {mlib.VOCAB} {seq} {mode} {part}"


def build(job):
    p = job.split()
    dims, n0 = dims_of(p[1])
    if p[0] == "L":
        n, seq, mode = int(p[2]), int(p[3]), p[4]
        w = int(p[5]) if len(p) > 5 else 1
        if mode == "layer":
            fn0, sets = mlib.stack_job(dims, 1, seq, "unroll", w)
            import layer as ly
            return ly.make_layer(*dims), sets
        return mlib.stack_job(dims, n, seq, mode, w)
    seq, mode, part = int(p[2]), p[3], p[4]
    n = int(p[5]) if len(p) > 5 else n0
    return mlib.model_job(dims, n, seq, mode, part)


def main():
    jobs_path, out, claims, done_glob = sys.argv[1:5]
    jobs = [ln.split("#")[0].strip() for ln in open(jobs_path)]
    jobs = [j for j in jobs if j]
    done = set()
    for f in glob.glob(done_glob):
        for ln in open(f):
            p = ln.split()
            if p and p[0] == "LROW":
                done.add(" ".join(p[:9]))
            elif p and p[0] == "MROW":
                done.add(" ".join(p[:11]))
    os.makedirs(claims, exist_ok=True)
    info = os.path.splitext(out)[0] + "_info.txt"
    t0 = time.time()
    with open(out, "a") as fo, open(info, "a") as fi:
        for i, job in enumerate(jobs):
            key = row_key(job)
            if key in done:
                continue
            try:
                os.close(os.open(os.path.join(claims, f"{i:05d}"), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue
            r, err = None, ""
            try:
                fn, sets = build(job)
                r = mlib.measure(fn, sets)
                del sets
            except Exception as e:  # noqa: BLE001  실패 행으로 남긴다
                err = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
                traceback.print_exc()
            if r is None:
                r = {"sync_min_us": 0.0, "sync_med_us": 0.0, "wall_min_us": 0.0, "n_samples": 0,
                     "compile_s": 0.0, "mem": {}, "peak_bytes": None, "hlo_ops": ""}
            fo.write(f"{key} {r['sync_min_us']:.3f} {r['sync_med_us']:.3f} {r['wall_min_us']:.3f} "
                     f"{r['n_samples']}\n")
            fo.flush()
            g = lambda b: f"{b / 2**30:.3f}" if b is not None else "nan"  # noqa: E731
            m = r["mem"]
            fi.write(f"INFO {key} compile_s={r['compile_s']:.1f} peak_GiB={g(r['peak_bytes'])} "
                     f"arg_GiB={g(m.get('argument_size_in_bytes'))} temp_GiB={g(m.get('temp_size_in_bytes'))} "
                     f"out_GiB={g(m.get('output_size_in_bytes'))} n_ops={len(r['hlo_ops'].split(',')) if r['hlo_ops'] else 0}"
                     f"{' ERROR ' + err if err else ''}\n")
            fi.flush()
            print(f"[{time.time() - t0:7.0f}s] {key} -> {r['sync_min_us']:.1f} us "
                  f"(compile {r['compile_s']:.1f}s){' ERR ' + err if err else ''}", flush=True)
            mlib.free()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
