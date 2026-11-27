#!/usr/bin/env python3
"""Step 7.1 측정: synth.SEQS의 직선 chain마다 U, W2, W3를 잰다.

사용: run_synth.py --dims D,INTER,NH,HD,KV --seq S --copies 2 --out data_synth_copies2_tpu_v4.txt
행: SROW D INTER NH HD KV <type순서> lo hi seq dev_min_us dev_med_us wall_min_us n_samples sync_min_us sync_med_us
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", required=True)
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", default=None)
    ap.add_argument("--windows", default=None, help="무시 (run_jobs.py 호환)")
    ap.add_argument("--hlo", default=None, help="무시 (run_jobs.py 호환)")
    ap.add_argument("--retry", action="store_true", help="무시 (run_jobs.py 호환)")
    a = ap.parse_args()
    import measure as ms
    import kernels as kn
    import synth as sy

    D, INTER, NH, HD, KV = map(int, a.dims.split(","))
    key = f"{D} {INTER} {NH} {HD} {KV}"
    for si, types in enumerate(sy.SEQS):
        spec, shp = sy.chain_spec(types, D, INTER, NH, HD, KV, a.seq)
        for lo, hi in sy.windows_of(types):
            fn, arg_names, _ = sy.window(spec, lo, hi)
            try:
                r = ms.measure_fn(fn, kn.make_args(arg_names, shp), f"y{si}_{lo}_{hi}",
                                  reps=a.reps, n_copies=a.copies)
                ok = r["n_exec"] == a.reps and r["n_events"] == 2 * a.reps
            except Exception as e:  # noqa: BLE001
                r = {"dev_us": [], "sync_us": [], "wall_us": [float("nan")], "n_exec": 0}
                ok = False
                print("# FAIL", types, lo, hi, repr(e)[:200], file=sys.stderr)
            s = ms.summarize(r)
            row = (f"SROW {key} {sy.seq_name(types)} {lo} {hi} {a.seq} {s['dev_min_us']:.3f} "
                   f"{s['dev_med_us']:.3f} {s['wall_min_us']:.3f} {s['n_samples']} "
                   f"{s['sync_min_us']:.3f} {s['sync_med_us']:.3f}")
            print(row + ("" if ok else "  # FAIL"), flush=True)
            if a.out:
                with open(a.out, "a") as f:
                    f.write(row + "\n")


if __name__ == "__main__":
    main()
