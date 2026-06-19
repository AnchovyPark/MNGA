#!/usr/bin/env python3
"""inf2 profiling combo 스펙 → profile.csv 생성.

총 60 combos (combo 당 1 row):
  35 bucket_sweep : 7 prefill bucket × 5 output_len
  10 cross_bucket : decode 가 여러 bucket 통과
  15 boundary     : 각 pb bucket 경계 ±1 지점

사용법:
  python3 build_profile.py           → ./profile.csv 생성
"""
import csv
import os

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile.csv")


# ─── (1) Bucket 내부 sweep ───
BUCKET_SWEEP = [
    # (il,   [ol_list])
    (64,   [5, 15, 30, 45, 60]),           # pb=128
    (130,  [10, 30, 60, 90, 120]),         # pb=256
    (260,  [20, 60, 120, 180, 240]),       # pb=512
    (520,  [40, 120, 240, 360, 480]),      # pb=1024
    (1030, [80, 240, 480, 720, 960]),      # pb=2048
    (2050, [160, 480, 960, 1440, 1950]),   # pb=4096
    (4100, [300, 900, 1800, 2800, 3950]),  # pb=8192
]

# ─── (2) Cross-bucket ───
CROSS_BUCKET = [
    (64,   500),   (64,   2000),  (64,   4000),
    (130,  1000),  (130,  3000),
    (260,  1500),  (260,  3500),
    (520,  3000),
    (1030, 4000),
    (2050, 3000),
]

# ─── (3) Bucket 경계 ───
BOUNDARY = [
    (127, 50),   (128, 50),   (129, 50),        # pb 128/256
    (255, 50),   (256, 50),   (257, 50),        # pb 256/512
    (511, 100),  (512, 100),  (513, 100),       # pb 512/1024
    (1023, 200), (1024, 200), (1025, 200),      # pb 1024/2048
    (2047, 400), (2048, 400), (2049, 400),      # pb 2048/4096
]


def build_rows():
    rows = []
    for il, ol_list in BUCKET_SWEEP:
        for ol in ol_list:
            rows.append((il, ol))
    for il, ol in CROSS_BUCKET:
        rows.append((il, ol))
    for il, ol in BOUNDARY:
        rows.append((il, ol))
    # dedup 확인 (의도적 중복 없어야)
    assert len(set(rows)) == len(rows), \
        f"중복된 (il, ol) 존재: {len(rows)} -> {len(set(rows))}"
    return rows


def main():
    rows = build_rows()
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "input_len", "output_len"])
        for i, (il, ol) in enumerate(rows):
            w.writerow([i, il, ol])
    print(f"wrote {len(rows)} combos → {OUT_PATH}")
    print(f"  bucket_sweep = {sum(len(o) for _, o in BUCKET_SWEEP)}")
    print(f"  cross_bucket = {len(CROSS_BUCKET)}")
    print(f"  boundary     = {len(BOUNDARY)}")


if __name__ == "__main__":
    main()
