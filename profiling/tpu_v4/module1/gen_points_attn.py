#!/usr/bin/env python3
"""2차 범위 측정 점 생성 (TPU_MODULE1_KERNEL.md §3.3). seed 고정, 출력 commit.

attention (../kernels_attn.py): 행 <type> <S> <NH> <KV> <HD>
  head 구성 (NH, KV, HD): 크기 10종에서 나오는 것 (중복 제거 9종) + model 밖 10종
    model 밖: HD ∈ {64,128}, KV ∈ {1,2,4,8}, G=NH/KV ∈ 1..8, 8 ≤ NH ≤ 64
  S: gemm의 M과 같은 10종 (128..4096)
  qk, av, attn: (S, NH, KV, HD) 전부.  scale, mask, softmax: [NH,S,S]만 보므로 (S, NH)로 중복 제거, KV=HD=0
lm_head: gemm M ∈ {1, 8, 128, 512}, K = 크기 10종의 D (중복 제거), N = 128256

출력: points_attn.txt, points_attn_meta.txt (출처 model:<i> | rand), points_lmhead.txt
"""
import numpy as np

from gen_points_step3 import MS, SIZES

SEED = 20261127
VOCAB = 128256


def main():
    rng = np.random.default_rng(SEED)
    cfg_src = {}
    for si, (D, I, NH, HD, KV) in enumerate(SIZES):
        cfg_src.setdefault((NH, KV, HD), []).append(f"model:{si}")
    n_rand = 0
    while n_rand < 10:
        HD = int(rng.choice([64, 128]))
        KV = int(rng.choice([1, 2, 4, 8]))
        NH = KV * int(rng.integers(1, 9))
        if not 8 <= NH <= 64 or (NH, KV, HD) in cfg_src:
            continue
        cfg_src[(NH, KV, HD)] = ["rand"]
        n_rand += 1
    pts = {}
    for (NH, KV, HD), src in cfg_src.items():
        for S in MS:
            for t in ("qk", "av", "attn"):
                pts.setdefault((t, S, NH, KV, HD), []).extend(src)
            for t in ("scale", "mask", "softmax"):
                pts.setdefault((t, S, NH, 0, 0), []).extend(src)
    order = sorted(pts, key=lambda k: (["qk", "scale", "mask", "softmax", "av", "attn"].index(k[0]), k[1:]))
    with open("points_attn.txt", "w") as f, open("points_attn_meta.txt", "w") as fm:
        f.write("# type S NH KV HD   (gen_points_attn.py)\n")
        fm.write("# type S NH KV HD source\n")
        for k in order:
            f.write(" ".join(map(str, k)) + "\n")
            fm.write(" ".join(map(str, k)) + " " + ",".join(sorted(set(pts[k]))) + "\n")
    Ds = sorted({s[0] for s in SIZES})
    with open("points_lmhead.txt", "w") as f:
        f.write("# lm_head: gemm M K=D N=vocab HD=0   (gen_points_attn.py)\n")
        for M in (1, 8, 128, 512):
            for D in Ds:
                f.write(f"gemm {M} {D} {VOCAB} 0\n")
    print(len(cfg_src), "head configs;", len(order), "attn points;", 4 * len(Ds), "lm_head points")
    for c, s in cfg_src.items():
        print("  ", c, s)


if __name__ == "__main__":
    main()
