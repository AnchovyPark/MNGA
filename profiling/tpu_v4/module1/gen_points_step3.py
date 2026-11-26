#!/usr/bin/env python3
"""Step 3 측정 점 생성 (TPU_MODULE1_KERNEL.md §3.1~3.2). seed 고정, 출력 commit.

출력: points_step3_gemm.txt, points_step3_vec.txt   (행: type M K N HD)
      points_step3_meta.txt  (점마다 출처: model:<size idx> 또는 rand)
"""
import numpy as np

SIZES = [  # (D, INTER, NH, HD, KV)  TPU_INSTRUCTIONS.md Step 3
    (1024, 2816, 16, 64, 4), (1536, 4096, 24, 64, 6), (2048, 5504, 16, 128, 4),
    (2048, 8192, 32, 64, 8), (2560, 6912, 20, 128, 4), (3072, 8192, 24, 128, 8),
    (3584, 18944, 28, 128, 4), (4096, 11008, 32, 128, 8), (4096, 14336, 32, 128, 8),
    (5120, 13824, 40, 128, 8),
]
MS = [128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
SEED = 20261126


def rand64(rng, n, lo=256, hi=16384):
    v = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    return [int(max(lo, min(hi, round(x / 64) * 64))) for x in v]


def main():
    rng = np.random.default_rng(SEED)
    # gemm (K,N) -> 출처 (같은 (K,N)이 여러 크기에서 나오면 출처를 모두 적는다)
    kn_src = {}
    for si, (D, I, NH, HD, KV) in enumerate(SIZES):
        for kn in [(D, NH * HD), (D, KV * HD), (NH * HD, D), (D, I), (I, D)]:
            kn_src.setdefault(kn, []).append(f"model:{si}")
    Ks, Ns = rand64(rng, 40), rand64(rng, 40)
    for kn in zip(Ks, Ns):
        kn_src.setdefault(kn, []).append("rand")
    # vector kernel (type, N, HD) -> 출처
    vec_src = {}
    for si, (D, I, NH, HD, KV) in enumerate(SIZES):
        for t in ("rmsnorm", "add"):
            vec_src.setdefault((t, D, 0), []).append(f"model:{si}")
        for t in ("silu", "mul"):
            vec_src.setdefault((t, I, 0), []).append(f"model:{si}")
        for n in (NH * HD, KV * HD):
            vec_src.setdefault(("rope", n, HD), []).append(f"model:{si}")
    for t in ("rmsnorm", "add", "silu", "mul"):
        for n in rand64(rng, 10):
            vec_src.setdefault((t, n, 0), []).append("rand")
    for _ in range(10):
        hd = int(rng.choice([64, 128]))
        n = rand64(rng, 1)[0]
        n = max(hd, (n // hd) * hd)
        vec_src.setdefault(("rope", n, hd), []).append("rand")

    g, v, meta = [], [], []
    for (K, N), src in kn_src.items():
        for M in MS:
            g.append(f"gemm {M} {K} {N} 0")
            meta.append(f"gemm {M} {K} {N} 0 {','.join(src)}")
    for (t, N, HD), src in vec_src.items():
        for M in MS:
            v.append(f"{t} {M} 0 {N} {HD}")
            meta.append(f"{t} {M} 0 {N} {HD} {','.join(src)}")
    hdr = f"# Step 3 points, seed={SEED}. 행: type M K N HD\n"
    open("points_step3_gemm.txt", "w").write(hdr + "\n".join(g) + "\n")
    open("points_step3_vec.txt", "w").write(hdr + "\n".join(v) + "\n")
    open("points_step3_meta.txt", "w").write("# type M K N HD source(model:<SIZES idx> | rand)\n"
                                              + "\n".join(meta) + "\n")
    print(f"gemm (K,N) {len(kn_src)} -> {len(g)} pts;  vector {len(vec_src)} -> {len(v)} pts")


if __name__ == "__main__":
    main()
