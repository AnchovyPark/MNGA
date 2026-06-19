#!/usr/bin/env python3
"""profiler 가 정확히 무엇을 뱉는지 introspection.

2-op 체인을 RNGD 에서 profiled 실행 → build_tuc_profile_spans 가 주는
FunctionEvent 들의 모든 필드를 덤프한다. span 종류(name), 엔진/리소스 구분
(device_index, device_resource_id, thread), 시간(start/end/elapsed) 을 본다.
또 chrome-trace JSON 직렬화가 되는지도 확인.
"""
import json
from collections import defaultdict

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
        return F.silu(h @ self.w)   # matmul → silu (fusion 경계 1개)


def main():
    dev = torch.device("rngd", 0)
    h = torch.randn(8, 512, D, dtype=DTYPE)
    w = torch.randn(D, D, dtype=DTYPE)
    cm = CompileModule.from_exported(torch.export.export(TwoOp(w), (h,)))
    cm.to(dev)
    hd = h.to(dev)
    cm(hd, profiles=None, device=dev)

    profiles = cm.generate_profiles(dev)
    cm(hd, profiles=profiles, device=dev)
    di = [[p.device.index for p in inner] for inner in profiles]
    pc = [[p.cpu() for p in inner] for inner in profiles]
    spans = cm.edf.npu_node.build_tuc_profile_spans(pc, di, 10**9)

    print(f"=== 총 span 개수: {len(spans)} ===\n")

    # 1) FunctionEvent 한 개의 모든 공개 필드
    ev = spans[0]
    print("=== span[0] 의 모든 속성 (dir) ===")
    attrs = {}
    for a in dir(ev):
        if a.startswith("_"):
            continue
        try:
            v = getattr(ev, a)
            if callable(v):
                continue
            attrs[a] = v
        except Exception as e:
            attrs[a] = f"<err {e}>"
    for k, v in attrs.items():
        print(f"  {k:22s} = {v!r}")
    print(f"\n  time_range: start={ev.time_range.start} end={ev.time_range.end} "
          f"elapsed_us={ev.time_range.elapsed_us()}")

    # 2) span name(종류) 별 집계
    print("\n=== name(종류)별 count / sum_us ===")
    by_name = defaultdict(lambda: [0, 0.0])
    for e in spans:
        by_name[e.name][0] += 1
        by_name[e.name][1] += e.time_range.elapsed_us()
    for n, (c, t) in sorted(by_name.items(), key=lambda x: -x[1][1]):
        print(f"  {n:28s} x{c:<4d} sum={t:9.2f}us")

    # 3) 엔진/리소스 구분이 되는지: device_index, device_resource_id, thread 조합
    print("\n=== (device_index, device_resource_id, thread) 조합별 분포 ===")
    by_res = defaultdict(lambda: [0, 0.0])
    for e in spans:
        key = (getattr(e, "device_index", None),
               getattr(e, "device_resource_id", None),
               getattr(e, "thread", None))
        by_res[key][0] += 1
        by_res[key][1] += e.time_range.elapsed_us()
    for k, (c, t) in sorted(by_res.items()):
        print(f"  dev_idx={k[0]} resource_id={k[1]} thread={k[2]}: x{c} sum={t:.2f}us")

    # 4) 처음 12개 span 의 (name, start, end, dur, dev, resource)
    print("\n=== span 타임라인 (처음 15개) ===")
    print(f"  {'name':26s} {'start':>10s} {'end':>10s} {'dur_us':>9s} {'dev':>4s} {'res':>5s}")
    for e in sorted(spans, key=lambda x: x.time_range.start)[:15]:
        print(f"  {e.name:26s} {e.time_range.start:>10.1f} {e.time_range.end:>10.1f} "
              f"{e.time_range.elapsed_us():>9.2f} {getattr(e,'device_index','?'):>4} "
              f"{str(getattr(e,'device_resource_id','?')):>5}")

    # 5) JSON 직렬화 가능한지 (chrome-trace 형태)
    print("\n=== chrome-trace JSON 직렬화 샘플 (span 3개) ===")
    trace = []
    for e in spans[:3]:
        trace.append({
            "name": e.name, "ph": "X",
            "ts": e.time_range.start, "dur": e.time_range.elapsed_us(),
            "pid": getattr(e, "device_index", 0),
            "tid": getattr(e, "device_resource_id", getattr(e, "thread", 0)),
            "args": {"input_shapes": getattr(e, "input_shapes", None)},
        })
    print(json.dumps(trace, indent=2, default=str))


if __name__ == "__main__":
    main()
