#!/usr/bin/env python3
"""Compare ways of turning measured single/double/triple windows into a chain prediction.

Uses only the already-measured windows in layer_ngram_s128_results.csv (no profiling).
Predicts the 6 test chains with four strategies and reports error vs the measured chain.

Strategies:
  S1 singles (1+1+1)         : sum of standalone op costs (naive)
  S2 pair tiling (2+1 / 1+2) : cover chain with measured pairs, best phase
  S3 telescoping 2-op ctx    : M[first] + sum(M[prev,op] - M[prev])
  S4 telescoping 3-op ctx    : M[first pair] + sum(M[p2,p1,op] - M[p2,p1])
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "layer_ngram_s128_results.csv")

M = {}
TEST = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        if r.get("task_us"):
            M[r["sequence"]] = float(r["task_us"])
        if r["split"] == "test":
            TEST.append(r["sequence"])


def s1_singles(c):
    return sum(M[o] for o in c)


def s2_pair_tiling(c):
    # try both phases, return the one closer to reality is not allowed (no peeking);
    # instead return both and let caller show; here return min-length-cover of two phases.
    def tile(start):
        i, tot = start, 0.0
        if start == 1:
            tot += M[c[0]]
        while i < len(c):
            if i + 1 < len(c):
                tot += M[c[i:i+2]]; i += 2
            else:
                tot += M[c[i]]; i += 1
        return tot
    return tile(0), tile(1)  # (2+2.., 1+2+2..)


def s3_telescope2(c):
    tot = M[c[0]]
    for i in range(1, len(c)):
        tot += M[c[i-1:i+1]] - M[c[i-1]]
    return tot


def s4_telescope3(c):
    if len(c) < 2:
        return M[c]
    tot = M[c[:2]]
    for i in range(2, len(c)):
        tot += M[c[i-2:i+1]] - M[c[i-2:i]]
    return tot


def pct(pred, act):
    return (pred - act) / act * 100


print(f"{'chain':12s} {'len':>3} {'actual':>7} | "
      f"{'S1 sing':>8} {'e%':>6} | {'S2a 2+':>7} {'S2b 1+2':>8} {'e%':>6} | "
      f"{'S3 tel2':>8} {'e%':>6} | {'S4 tel3':>8} {'e%':>6}")
agg = {k: [] for k in ("s1", "s2", "s3", "s4")}
for c in sorted(TEST, key=len):
    act = M[c]
    s1 = s1_singles(c)
    s2a, s2b = s2_pair_tiling(c)
    s2 = s2a  # fair: fixed phase (pairs from the start), no peeking at the answer
    s3 = s3_telescope2(c)
    s4 = s4_telescope3(c)
    agg["s1"].append(abs(pct(s1, act))); agg["s2"].append(abs(pct(s2, act)))
    agg["s3"].append(abs(pct(s3, act))); agg["s4"].append(abs(pct(s4, act)))
    print(f"{c:12s} {len(c):>3} {act:7.1f} | "
          f"{s1:8.1f} {pct(s1,act):+5.0f}% | {s2a:7.1f} {s2b:8.1f} {pct(s2,act):+5.0f}% | "
          f"{s3:8.1f} {pct(s3,act):+5.0f}% | {s4:8.1f} {pct(s4,act):+5.0f}%")

print(f"\nmean |error%|:  "
      f"S1={sum(agg['s1'])/len(agg['s1']):.1f}   "
      f"S2={sum(agg['s2'])/len(agg['s2']):.1f}   "
      f"S3={sum(agg['s3'])/len(agg['s3']):.1f}   "
      f"S4={sum(agg['s4'])/len(agg['s4']):.1f}")
