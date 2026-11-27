#!/usr/bin/env python3
"""window 측정 실행기. 한 process에서 주어진 window들을 차례로 잰다.

사용:
  python run_windows.py --dims 2048,8192,32,64,8 --seq 512 --windows all \
      --out data_windows_tpu_v4.txt --hlo hlo_windows_tpu_v4.txt
  --windows: "all" (U+W2+W3+full = 43개), "all5"(W4,W5 포함), "len4,5"(길이 4·5만),
             "m2"(W1~W5 + full + 덩어리 0:6, 6:15 = 68개),
             "variants"(variants.py의 token 나누기 변형에 필요한 op 집합 전부),
             "attn"(attn.py LAYER 20 token의 W1~W3 + full + 덩어리),
             "attn1"(attention core를 token 1개로 볼 때 더 필요한 window, attn.ATT1_EXTRA),
             "L16:n"/"L20:n"(layer_chain.py chain의 W1~Wn + full + 덩어리 A·ATT·B),
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
    if spec == "m2":  # Module 2 Step 3 기본: W1~W5 + full + 덩어리 full (A=0:6, B=6:15)
        return kn.all_windows(5) + [(0, 6), (6, 15)]
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
    if a.windows == "variants":  # (a) token 나누기 변형의 op 집합 window (variants.py)
        import variants as vr
        items = [(vr.set_name(ops), ops) for ops in vr.all_sets()]
    elif a.windows in ("attn", "attn1"):    # Step 7.2 attention 포함 layer (attn.py)
        import attn as at
        shp = at.shapes(D, INTER, NH, HD, KV, a.seq)
        wins = at.all_windows(3) if a.windows == "attn" else at.ATT1_EXTRA
        items = [(at.window_name(lo, hi), (lo, hi)) for lo, hi in wins]
    elif a.windows[:4] in ("L16:", "L20:"):   # TPU_NEXT_LAYER: layer_chain.py, 예: L16:5 = W1~W5 + full + 덩어리
        import layer_chain as lc
        chain, nmax = a.windows[:3], int(a.windows[4:])
        dims = (D, INTER, NH, HD, KV)
        shp = lc.shapes(D, INTER, NH, HD, KV, a.seq)
        items = [(lc.window_name(chain, lo, hi), (lo, hi)) for lo, hi in lc.windows(chain, nmax)]
    else:
        items = [(kn.window_name(lo, hi), (lo, hi)) for lo, hi in parse_windows(a.windows, kn)]
    special = a.windows in ("variants", "attn", "attn1") or a.windows[:4] in ("L16:", "L20:")
    for name, spec in items:
        try:
            if special:
                if a.windows[:4] in ("L16:", "L20:"):
                    fn, arg_names, _ = lc.window(chain, *spec, dims)
                    args = lc.make_args(arg_names, shp)
                    tag = "%s_%02d_%02d" % (chain, *spec)
                elif a.windows == "variants":
                    fn, arg_names, _ = vr.set_window(spec)
                    args = kn.make_args(arg_names, shp)
                    tag = "s_" + name.replace("+", "_")
                else:
                    fn, arg_names, _ = at.window(*spec)
                    args = at.make_args(arg_names, shp)
                    tag = "a%02d_%02d" % spec
                r = ms.measure_fn(fn, args, tag, reps=a.reps, n_copies=a.copies)
                r["ok"] = (r["n_exec"] == a.reps and r["n_events"] == 2 * a.reps and min(r["dev_us"]) > 0)
            else:
                r = ms.measure_window(*spec, shp, reps=a.reps, n_copies=a.copies)
        except Exception as e:  # noqa: BLE001
            r = {"dev_us": [], "sync_us": [], "wall_us": [float("nan")], "n_exec": 0,
                 "n_events": 0, "hlo_ops": [], "ok": False, "err": repr(e)[:200]}
        if not r["ok"] and a.retry and not special:
            lo, hi = spec
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
