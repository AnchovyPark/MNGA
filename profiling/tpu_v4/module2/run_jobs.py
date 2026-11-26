#!/usr/bin/env python3
"""(크기, seq, copies) job 목록을 chip 4개에 나눠 측정하고 data 파일에 합친다.

chip마다 worker 하나가 queue에서 job을 꺼내 run_windows.py를 process 하나로 돌린다
(TPU_VISIBLE_CHIPS로 process당 chip 1개). chip 사이 차이는 noise floor 안이다
(Module 1 memory/2026-11-26-tpu-m1.md, chip 0~3 차이 ≤0.6%).

사용: run_jobs.py <jobs.txt> [--windows all]
  jobs.txt 한 줄: D,INTER,NH,HD,KV seq copies
출력 (job 순서대로 append):
  copies=1 → data_windows_tpu_v4.txt / hlo_windows_tpu_v4.txt          (공통 문서 기본 방식)
  copies=k → data_windows_copies{k}_tpu_v4.txt / hlo_windows_copies{k}_tpu_v4.txt
이미 data 파일에 있는 (크기, seq) 조합은 건너뛴다 (재실행 가능).
"""
import argparse
import os
import queue
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
PARTS = os.path.join(HERE, "parts")


def files(copies):
    sfx = "" if copies == 1 else f"_copies{copies}"
    return (os.path.join(HERE, f"data_windows{sfx}_tpu_v4.txt"),
            os.path.join(HERE, f"hlo_windows{sfx}_tpu_v4.txt"))


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        for l in open(path):
            p = l.split()
            if p and p[0] == "WROW":
                keys.add((",".join(p[1:6]), p[7]))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--windows", default="all")
    ap.add_argument("--nchips", type=int, default=4)
    ap.add_argument("--noskip", action="store_true", help="이미 있는 (크기, seq)도 잰다 (다른 window 추가용)")
    a = ap.parse_args()
    os.makedirs(PARTS, exist_ok=True)

    jobs = []
    for l in open(a.jobs):
        l = l.split("#")[0].strip()
        if not l:
            continue
        dims, seq, copies = l.split()
        if not a.noskip and (dims, seq) in done_keys(files(int(copies))[0]):
            print("skip", dims, seq, copies, flush=True)
            continue
        jobs.append((dims, seq, int(copies)))
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    lock = threading.Lock()

    def worker(chip):
        env = dict(os.environ, TPU_VISIBLE_CHIPS=str(chip),
                   TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1", TPU_PROCESS_BOUNDS="1,1,1",
                   TPU_PROCESS_PORT=str(8476 + chip))
        while True:
            try:
                dims, seq, copies = q.get_nowait()
            except queue.Empty:
                return
            tag = f"{dims.replace(',', '_')}_s{seq}_c{copies}"
            pd, ph = os.path.join(PARTS, tag + ".txt"), os.path.join(PARTS, tag + "_hlo.txt")
            for p in (pd, ph):
                if os.path.exists(p):
                    os.remove(p)
            cmd = [sys.executable, os.path.join(HERE, "run_windows.py"), "--dims", dims,
                   "--seq", seq, "--windows", a.windows, "--copies", str(copies),
                   "--retry", "--out", pd, "--hlo", ph]
            with open(os.path.join(PARTS, tag + ".log"), "w") as lf:
                rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            with lock:
                fd, fh = files(copies)
                if os.path.exists(pd):
                    with open(fd, "a") as f:
                        f.write(open(pd).read())
                if os.path.exists(ph):
                    with open(fh, "a") as f:
                        f.write(open(ph).read())
                n = sum(1 for _ in open(pd)) if os.path.exists(pd) else 0
                print(f"chip{chip} {tag} rc={rc} rows={n}", flush=True)

    ts = [threading.Thread(target=worker, args=(c,)) for c in range(a.nchips)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


if __name__ == "__main__":
    main()
