#!/usr/bin/env python3
"""Step 1 (TPU_MODULES_COMMON.md §5): 측정 경로 확인 gate.

1B (2048,8192,32,64,8), seq 512, token add1(7:8)·gate(9:10).
 (1) event 수 확인  (2) fresh process 5회 → process 간 min 변동
 (3) 한 process에서 43 window 연달아 잰 뒤 add1·gate를 다시 잰 값과 비교
 (4) device vs wall 차이  (5) 입력 사본 4개 vs 같은 입력 반복
결과 행은 data_step1_tpu_v4.txt에 'S1 <exp> <rep> WROW ...' 형식으로 남긴다.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
DIMS = "2048,8192,32,64,8"
OUT = os.path.join(HERE, "data_step1_tpu_v4.txt")


def run(windows, copies=1):
    cmd = [PY, os.path.join(HERE, "run_windows.py"), "--dims", DIMS, "--seq", "512",
           "--windows", windows, "--copies", str(copies)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    rows = [l for l in p.stdout.splitlines() if l.startswith("WROW")]
    if not rows:
        print(p.stderr[-2000:], file=sys.stderr)
    return rows


def log(exp, rep, rows):
    with open(OUT, "a") as f:
        for r in rows:
            f.write(f"S1 {exp} {rep} {r}\n")
            print(f"S1 {exp} {rep} {r}", flush=True)


def main():
    # (2) fresh process 5회씩
    for rep in range(5):
        log("fresh", rep, run("7:8"))
        log("fresh", rep, run("9:10"))
    # (3) 한 process: add1, gate, 43개 전부, add1, gate
    log("oneproc", 0, run("7:8,9:10"))
    allw = run("7:8,9:10," + ",".join(f"{lo}:{hi}" for lo, hi in _allw()) + ",7:8,9:10")
    log("oneproc_long", 0, allw)
    # (5) 입력 사본 4개
    for rep in range(3):
        log("copies4", rep, run("7:8,9:10", copies=4))
        log("copies1", rep, run("7:8,9:10", copies=1))


def _allw():
    sys.path.insert(0, os.path.join(HERE, ".."))
    import kernels as kn
    return kn.all_windows(3)


if __name__ == "__main__":
    main()
