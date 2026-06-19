#!/usr/bin/env python3
"""엔진별 구분이 profiler 레벨에서 되는지 확인.

순수 matmul(행렬엔진) vs 순수 vector(elementwise/활성화) 커널을 각각 프로파일해
span name 종류가 갈리는지 본다. 갈리면 엔진 구분 가능, 안 갈리면 TuExec 로 뭉뚱그려짐.
"""
from collections import defaultdict

import torch
import torch.nn.functional as F
import furiosa.torch  # noqa
from furiosa.torch.custom_ops import CompileModule

D = 4096
DTYPE = torch.bfloat16


class PureMatmul(torch.nn.Module):
    def __init__(self, w):
        super().__init__(); self.register_buffer("w", w)
    def forward(self, h):
        return h @ self.w


class PureVector(torch.nn.Module):
    """행렬곱 없이 elementwise/활성화만 잔뜩."""
    def __init__(self, a, b):
        super().__init__(); self.register_buffer("a", a); self.register_buffer("b", b)
    def forward(self, h):
        x = F.silu(h)
        x = x * self.a
        x = x + self.b
        return x * self.a


class MatmulThenVector(torch.nn.Module):
    def __init__(self, w, a):
        super().__init__(); self.register_buffer("w", w); self.register_buffer("a", a)
    def forward(self, h):
        return F.silu(h @ self.w) * self.a


def union_us(iv):
    if not iv: return 0.0
    iv = sorted(iv); t = 0.0; cs, ce = iv[0]
    for s, e in iv[1:]:
        if s <= ce: ce = max(ce, e)
        else: t += ce - cs; cs, ce = s, e
    return t + (ce - cs)


def profile(mod, h, dev, label):
    cm = CompileModule.from_exported(torch.export.export(mod, (h,)))
    cm.to(dev); hd = h.to(dev)
    cm(hd, profiles=None, device=dev)
    profiles = cm.generate_profiles(dev)
    cm(hd, profiles=profiles, device=dev)
    di = [[p.device.index for p in inner] for inner in profiles]
    pc = [[p.cpu() for p in inner] for inner in profiles]
    spans = cm.edf.npu_node.build_tuc_profile_spans(pc, di, 10**9)
    by = defaultdict(list)
    for e in spans:
        by[e.name].append((e.time_range.start, e.time_range.end))
    print(f"\n=== {label} === (span {len(spans)}개)")
    print(f"  {'name':22s} {'cnt':>4s} {'sum_us':>9s} {'union_us':>9s}")
    for name, iv in sorted(by.items(), key=lambda x: -union_us(x[1])):
        s = sum(e - st for st, e in iv)
        print(f"  {name:22s} {len(iv):>4d} {s:>9.1f} {union_us(iv):>9.1f}")


def main():
    dev = torch.device("rngd", 0)
    h = torch.randn(8, 512, D, dtype=DTYPE)
    w = torch.randn(D, D, dtype=DTYPE)
    a = torch.randn(8, 512, D, dtype=DTYPE)
    b = torch.randn(8, 512, D, dtype=DTYPE)
    profile(PureMatmul(w), h, dev, "순수 MATMUL (행렬엔진)")
    profile(PureVector(a, b), h, dev, "순수 VECTOR (elementwise×5)")
    profile(MatmulThenVector(w, a), h, dev, "MATMUL → VECTOR")


if __name__ == "__main__":
    main()
