#!/usr/bin/env python3
"""Step 2: 축별 latency 곡선과 이웃에서 10% 이상 벗어나는 점, HLO 변화.

usage: analyze_step2.py step2/data_step2_tpu_v4.txt
값 = sync_min_us (Module 1 대표값). 출력: 텍스트 표 + step2/step2_curves.png
벗어남 판정: 점 i의 값과 이웃 두 점 평균의 비 |v_i / mean(v_{i-1}, v_{i+1}) − 1| ≥ 10%.
(양 끝 점은 한쪽 이웃만 쓴다.) 이와 별도로 단조성(앞 점보다 작아지는 점)도 적는다.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1]
rows, hlo = [], {}
for ln in open(path):
    p = ln.split()
    if p and p[0] == "KROW":
        rows.append((p[1], *map(int, p[2:6]), float(p[10]), float(p[6])))
for ln in open(os.path.splitext(path)[0] + "_hlo.txt"):
    p = ln.split(maxsplit=6)
    hlo[(p[1], *map(int, p[2:6]))] = p[6].strip() if len(p) > 6 else ""

AXES = [
    ("gemm M (K,N)=(2048,8192)", lambda r: r[0] == "gemm" and r[2:4] == (2048, 8192), 1),
    ("gemm M (K,N)=(4096,4096)", lambda r: r[0] == "gemm" and r[2:4] == (4096, 4096), 1),
    ("gemm N (M,K)=(512,2048)", lambda r: r[0] == "gemm" and r[1] == 512 and r[2] == 2048, 3),
    ("gemm K (M,N)=(512,2048)", lambda r: r[0] == "gemm" and r[1] == 512 and r[3] == 2048, 2),
    ("add M N=2048", lambda r: r[0] == "add", 1),
]

fig, axs = plt.subplots(len(AXES), 2, figsize=(13, 3.2 * len(AXES)))
for ai, (title, sel, xi) in enumerate(AXES):
    pts = sorted({r[xi]: r for r in rows if sel(r)}.values(), key=lambda r: r[xi])
    xs = [r[xi] for r in pts]
    ys = [r[5] for r in pts]
    fails = [x for x, y in zip(xs, ys) if y <= 0]
    good = [(x, y, r) for x, y, r in zip(xs, ys, pts) if y > 0]
    print(f"\n== {title}: {len(pts)} pts, failed {len(fails)} {fails if fails else ''}")
    out, nonmono = [], []
    for i, (x, y, r) in enumerate(good):
        nb = [good[j][1] for j in (i - 1, i + 1) if 0 <= j < len(good)]
        dev = y / (sum(nb) / len(nb)) - 1
        if abs(dev) >= 0.10:
            out.append((x, y, dev))
        if i and y < good[i - 1][1] * 0.995:
            nonmono.append((good[i - 1][0], x, good[i - 1][1], y))
    print(f"  range {good[0][1]:.2f} .. {good[-1][1]:.2f} us")
    print(f"  >=10% from neighbours: {len(out)}")
    for x, y, d in out:
        print(f"    x={x:6d}  {y:9.2f} us  dev={d:+.1%}  x%128={x % 128} x%256={x % 256}  pow2={(x & (x - 1)) == 0}")
    print(f"  non-monotone drops (>0.5%): {len(nonmono)}")
    for a, b, ya, yb in nonmono:
        print(f"    {a}->{b}: {ya:.2f} -> {yb:.2f} ({yb / ya - 1:+.1%})")
    # HLO 연산 목록 변화
    prev = None
    for x, y, r in good:
        h = hlo.get(r[:5], "")
        if prev is not None and h != prev[1]:
            a, b = set(prev[1].split(",")), set(h.split(","))
            print(f"  HLO change at {prev[0]}->{x}: +{sorted(b - a)} -{sorted(a - b)}")
        prev = (x, h)
    ax = axs[ai][0]
    ax.plot([g[0] for g in good], [g[1] for g in good], ".-", ms=3)
    for x, y, d in out:
        ax.plot(x, y, "rx")
    ax.set_title(title + "  (sync_min us)")
    ax.grid(alpha=.3)
    ax2 = axs[ai][1]  # latency / x : 단위 work 당 시간 (계단이 더 잘 보인다)
    ax2.plot([g[0] for g in good], [g[1] / g[0] for g in good], ".-", ms=3)
    ax2.set_title(title + "  (us per unit of x)")
    ax2.grid(alpha=.3)
plt.tight_layout()
png = os.path.join(os.path.dirname(path), "step2_curves.png")
plt.savefig(png, dpi=90)
print("\nsaved", png)
