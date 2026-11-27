#!/bin/bash
# job 목록을 chip NCHIPS(기본 4)개 process가 나눠 측정하고 job 순서대로 합친다.
# usage: run_jobs.sh <jobs.txt> <out.txt>
#   part 파일 <out stem>.part<c>.txt (+ _info.txt, .log). 다시 돌리면 이미 잰 key는 건너뛴다.
set -u
JOBS=$1; OUT=$2
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$HOME/venvs/jaxtpu/bin/python}
NC=${NCHIPS:-4}
STEM=${OUT%.txt}
rm -rf "$STEM.claims"
for c in $(seq 0 $((NC-1))); do
  TPU_VISIBLE_CHIPS=$c TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 \
  TPU_PROCESS_PORT=$((8476+c)) "$PY" -I "$HERE/run_model.py" "$JOBS" "$STEM.part$c.txt" "$STEM.claims" \
    "$STEM.part*.txt" >> "$STEM.part$c.log" 2>&1 &
done
wait
rm -rf "$STEM.claims"
JAX_PLATFORMS=cpu "$PY" -I - "$HERE" "$JOBS" "$STEM" <<'PYEOF'
import glob, sys
sys.path.insert(0, sys.argv[1])
import run_model as rm
jobs, stem = sys.argv[2], sys.argv[3]
keys = [rm.row_key(j) for j in (l.split("#")[0].strip() for l in open(jobs)) if j]
for suf in (".txt", "_info.txt"):
    rows = {}  # key -> 행 list (실패 행도 지우지 않는다. 분석은 key별 마지막 성공 행을 쓴다)
    for f in sorted(glob.glob(f"{stem}.part[0-9]*{suf}")):
        if suf == ".txt" and f.endswith("_info.txt"):
            continue
        for l in open(f):
            p = l.split()
            if not p:
                continue
            q = p[1:] if p[0] == "INFO" else p
            rows.setdefault(" ".join(q[:11 if q[0] == "MROW" else 9]), []).append(l)
    with open(stem + suf, "w") as fo:
        for k in keys:
            fo.writelines(rows.get(k, []))
    print(stem + suf, sum(k in rows for k in keys), "of", len(keys))
PYEOF
