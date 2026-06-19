#!/usr/bin/env python3
"""Minimum 14 combos (7 pb × 2 ol) — 빠른 검증용.

각 prefill bucket 의 최소/최대 ol 2 point → bucket 당 linear fit (P(b), TBT(b)) 추출 가능한 최소.
"""
import csv
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile_min14.csv")

# (il, [ol_first, ol_last])  — bucket_sweep 의 양 끝
COMBOS_MIN14 = [
    (64,   [5, 60]),                    # pb=128
    (130,  [10, 120]),                  # pb=256
    (260,  [20, 240]),                  # pb=512
    (520,  [40, 480]),                  # pb=1024
    (1030, [80, 960]),                  # pb=2048
    (2050, [160, 1950]),                # pb=4096
    (4100, [300, 3950]),                # pb=8192
]


def main():
    rows = [(il, ol) for il, ols in COMBOS_MIN14 for ol in ols]
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "input_len", "output_len"])
        for i, (il, ol) in enumerate(rows):
            w.writerow([i, il, ol])
    print(f"wrote {len(rows)} combos → {OUT}")


if __name__ == "__main__":
    main()
