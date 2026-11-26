#!/usr/bin/env python3
"""window 측정 실행기. 한 process에서 주어진 window들을 차례로 잰다.

사용:
  python run_windows.py --dims 2048,8192,32,64,8 --seq 512 --windows all \
      --out data_windows_tpu_v4.txt --hlo hlo_windows_tpu_v4.txt
  --windows: "all" (U+W2+W3+full = 43개), "all5"(W4,W5 포함), "len4,5"(길이 4·5만),
             또는 "lo:hi,lo:hi"
  --copies N: 입력 사본 N개를 돌려 쓴다 (Step 1 비교용, 기본 1)
  --retry: 실패(event 수 불일치, 0) window를 fresh process에서 한 번 더 잰다

출력 행 (TPU_MODULE2_NGRAM.md §8):
  WROW D INTER NH HD KV window seq dev_min_us dev_med_us wall_min_us n_samples sync_min_us sync_med_us
  (끝 두 열은 core launch skew를 뺀 값. measure._executions 참고)
  HROW D INTER NH HD KV window seq op1,op2,...
실패한 행은 dev 값을 nan으로 남기고 지우지 않는다.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def parse_windows(spec, kn):
    if spec == "all":
        return kn.all_windows(3)
    if spec == "all5":
        return kn.all_windows(5)
    if spec.startswith("len"):  # 예: len4,5 → 길이 4, 5 window만
        Ls = [int(x) for x in spec[3:].split(",")]
        n = len(kn.TOKENS)
        return [(i, i + L) for L in Ls for i in range(n - L + 1)]
    out = []
    for p in spec.split(","):
        lo, hi = p.split(":")
        out.append((int(lo), int(hi)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", required=True)
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--windows", default="all")
    ap.add_argument("--out", default=None)
    ap.add_argument("--hlo", default=None)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--copies", type=int, default=1)
    ap.add_argument("--retry", action="store_true")
    ap.add_argument("--json", action="store_true", help="결과를 JSON 한 줄씩 stdout으로")
    a = ap.parse_args()

    import measure as ms
    import kernels as kn

    D, INTER, NH, HD, KV = map(int, a.dims.split(","))
    shp = kn.shapes(D, INTER, NH, HD, KV, a.seq)
    key = f"{D} {INTER} {NH} {HD} {KV}"
    for lo, hi in parse_windows(a.windows, kn):
        name = kn.window_name(lo, hi)
        try:
            r = ms.measure_window(lo, hi, shp, reps=a.reps, n_copies=a.copies)
        except Exception as e:  # noqa: BLE001
            r = {"dev_us": [], "sync_us": [], "wall_us": [float("nan")], "n_exec": 0,
                 "n_events": 0, "hlo_ops": [], "ok": False, "err": repr(e)[:200]}
        if not r["ok"] and a.retry:
            cmd = [sys.executable, __file__, "--dims", a.dims, "--seq", str(a.seq),
                   "--windows", f"{lo}:{hi}", "--reps", str(a.reps),
                   "--copies", str(a.copies), "--json"]
            p = subprocess.run(cmd, capture_output=True, text=True)
            lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
            if lines:
                r2 = json.loads(lines[-1])
                print(f"# retry {name}: ok={r2['ok']}", file=sys.stderr)
                r = r2["raw"]
        s = ms.summarize(r)
        row = (f"WROW {key} {name} {a.seq} {s['dev_min_us']:.3f} "
               f"{s['dev_med_us']:.3f} {s['wall_min_us']:.3f} {s['n_samples']} "
               f"{s['sync_min_us']:.3f} {s['sync_med_us']:.3f}")
        hrow = f"HROW {key} {name} {a.seq} {','.join(r['hlo_ops']) or '-'}"
        if a.json:
            print(json.dumps({"ok": r["ok"], "row": row, "summary": s, "raw": r}))
        else:
            print(row + ("" if r["ok"] else "  # FAIL " + r.get("err", "")), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(row + "\n")
        if a.hlo:
            with open(a.hlo, "a") as f:
                f.write(hrow + "\n")


if __name__ == "__main__":
    main()
