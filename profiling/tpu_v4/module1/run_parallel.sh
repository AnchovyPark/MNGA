#!/bin/bash
# 점 목록을 chip 4개에 round-robin으로 나눠 동시에 측정하고 원래 순서로 합친다.
# chip 사이 차이는 noise floor(0.5%) 안 (memory/2026-11-26-tpu-m1.md Step 2 앞).
# usage: run_parallel.sh <points.txt> <out_data.txt> [run_points 추가 인자...]
set -u
PTS=$1; OUT=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$HOME/venvs/jaxtpu/bin/python}
NC=${NCHIPS:-4}
STEM=${OUT%.txt}
grep -v '^#' "$PTS" | grep . > "$STEM.pts.all"
for c in $(seq 0 $((NC-1))); do
  awk -v c=$c -v n=$NC 'NR%n==c' "$STEM.pts.all" > "$STEM.pts.$c"
  TPU_VISIBLE_CHIPS=$c TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 \
  TPU_PROCESS_PORT=$((8476+c)) "$PY" -I "$HERE/run_points.py" "$STEM.pts.$c" "$STEM.part$c.txt" "$@" \
    > "$STEM.part$c.log" 2>&1 &
done
wait
# 원래 점 순서로 merge
"$PY" -I - "$STEM" "$NC" <<'PYEOF'
import sys
stem, nc = sys.argv[1], int(sys.argv[2])
order = [l.split() for l in open(stem + ".pts.all")]
for suf in (".txt", "_hlo.txt"):
    rows = {}
    for c in range(nc):
        try:
            for l in open(f"{stem}.part{c}{suf}"):
                p = l.split()
                rows[" ".join(p[1:6])] = l
        except FileNotFoundError:
            pass
    with open(stem + suf, "w") as f:
        for p in order:
            k = " ".join(p)
            if k in rows:
                f.write(rows[k])
    print(stem + suf, sum(1 for _ in open(stem + suf)), "rows of", len(order))
PYEOF
