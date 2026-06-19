#!/usr/bin/env python3
"""span 들의 sum vs union vs envelope 를 계산해, 'TuExec 합 > Task' 미스터리를 푼다.

가설: TuExec/DMA span 이 여러 개이고 서로 겹쳐(병렬 레인) 실행되므로,
      단순 sum 은 과대. 올바른 시간은 union(겹친 구간 병합).
"""
import torch
import torch.nn.functional as F
import furiosa.torch  # noqa
from furiosa.torch.custom_ops import CompileModule

D = 4096
DTYPE = torch.bfloat16


class TwoOp(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.register_buffer("w", w)

    def forward(self, h):
        return F.silu(h @ self.w)


def union_us(intervals):
    """겹치는 구간 병합 후 총 길이(us)."""
    if not intervals:
        return 0.0
    iv = sorted(intervals)
    total = 0.0
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            total += ce - cs
            cs, ce = s, e
    total += ce - cs
    return total


def overlap_us(a, b):
    """두 구간집합 union 의 교집합 총 길이."""
    # a,b 각각 union 후 교집합
    def merge(iv):
        iv = sorted(iv); out = []
        cs, ce = iv[0]
        for s, e in iv[1:]:
            if s <= ce: ce = max(ce, e)
            else: out.append((cs, ce)); cs, ce = s, e
        out.append((cs, ce)); return out
    if not a or not b:
        return 0.0
    A, B = merge(a), merge(b)
    i = j = 0; tot = 0.0
    while i < len(A) and j < len(B):
        s = max(A[i][0], B[j][0]); e = min(A[i][1], B[j][1])
        if s < e: tot += e - s
        if A[i][1] < B[j][1]: i += 1
        else: j += 1
    return tot


def main():
    dev = torch.device("rngd", 0)
    h = torch.randn(8, 512, D, dtype=DTYPE)
    w = torch.randn(D, D, dtype=DTYPE)
    cm = CompileModule.from_exported(torch.export.export(TwoOp(w), (h,)))
    cm.to(dev); hd = h.to(dev)
    cm(hd, profiles=None, device=dev)
    profiles = cm.generate_profiles(dev)
    cm(hd, profiles=profiles, device=dev)
    di = [[p.device.index for p in inner] for inner in profiles]
    pc = [[p.cpu() for p in inner] for inner in profiles]
    spans = cm.edf.npu_node.build_tuc_profile_spans(pc, di, 10**9)

    by = {}
    for e in spans:
        by.setdefault(e.name, []).append((e.time_range.start, e.time_range.end))

    print(f"{'name':18s} {'cnt':>4s} {'sum_us':>10s} {'union_us':>10s} {'envelope_us':>12s}")
    for name, iv in by.items():
        s = sum(e - st for st, e in iv)
        u = union_us(iv)
        env = max(e for _, e in iv) - min(st for st, _ in iv)
        print(f"{name:18s} {len(iv):>4d} {s:>10.1f} {u:>10.1f} {env:>12.1f}")

    # 전체(Task 제외) envelope = 진짜 device latency
    work = [iv for name, ivs in by.items() if name != "Task" for iv in ivs]
    dev_lat = max(e for _, e in work) - min(st for st, _ in work)
    tu = by.get("Renegade::TuExec", [])
    dma = by.get("DMA", [])
    tu_u, dma_u = union_us(tu), union_us(dma)
    ov = overlap_us(tu, dma)
    print(f"\n진짜 device latency (TuExec+DMA envelope) = {dev_lat:.1f} us")
    print(f"  연산 union (compute busy)   = {tu_u:.1f} us")
    print(f"  DMA  union (transfer busy)  = {dma_u:.1f} us")
    print(f"  연산∩DMA 겹침 (동시 실행)    = {ov:.1f} us")
    print(f"  → 둘 중 하나라도 도는 시간    = {union_us(tu+dma):.1f} us")
    idle = dev_lat - union_us(tu + dma)
    print(f"  → 둘 다 노는 시간(idle)       = {idle:.1f} us")
    if "Task" in by:
        ts, te = by["Task"][0]
        print(f"\nTask span: start={ts:.1f} end={te:.1f} dur={te-ts:.1f} "
              f"(end 은 내가 넘긴 task_finished=10^9ns 에 묶임 → 신뢰 X)")


if __name__ == "__main__":
    main()
