#!/usr/bin/env python3
"""RNGD 프로파일러 평가: 단일 커널 + 2-커널 체인의 디바이스 span 측정.

torch 2.10+xpu 빌드의 RNGDProfiler(kineto) 가 Intel PTI 와 충돌하므로,
저수준 경로 (CompileModule → generate_profiles → edf.npu_node.build_tuc_profile_spans)
로 TUC span 을 직접 추출한다.

측정 대상:
  단일: matmul, add, mul, softmax  × S ∈ {128, 512, 1024, 2048}
  체인: matmul→add, matmul→mul, matmul→softmax, where→softmax (mask)

출력: kernel_profiler_eval_results.csv
  (config, kind, op, S, run, wall_us, span name 별 count/sum_us)
"""
import csv
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import torch
import furiosa.torch  # noqa: F401  (rngd device 등록)
from furiosa.torch.custom_ops import CompileModule

HERE = os.path.dirname(os.path.abspath(__file__))

# 사용법: kernel_profiler_eval.py [S1,S2,...] [out_suffix]
SHAPES = ([int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1
          else [128, 512, 1024, 2048])
_SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
OUT_CSV = os.path.join(HERE, f"kernel_profiler_eval_results{_SUFFIX}.csv")
N_RUNS = 3
DTYPE = torch.bfloat16


class MatMul(torch.nn.Module):
    def forward(self, a, b):
        return a @ b


class Add(torch.nn.Module):
    def forward(self, a, b):
        return a + b


class Mul(torch.nn.Module):
    def forward(self, a, b):
        return a * b


class Softmax(torch.nn.Module):
    def forward(self, a):
        return torch.softmax(a, dim=-1)


class MatMulAdd(torch.nn.Module):
    def forward(self, a, b, c):
        return (a @ b) + c


class MatMulMul(torch.nn.Module):
    def forward(self, a, b, c):
        return (a @ b) * c


class MatMulSoftmax(torch.nn.Module):
    def forward(self, a, b):
        return torch.softmax(a @ b, dim=-1)


class WhereSoftmax(torch.nn.Module):
    def forward(self, x, mask):
        return torch.softmax(torch.where(mask, x, torch.tensor(-1e9, dtype=x.dtype)), dim=-1)


def make_inputs(op, S):
    if op in ("matmul", "matmul_add", "matmul_mul", "matmul_softmax"):
        a = torch.randn(S, S, dtype=DTYPE)
        b = torch.randn(S, S, dtype=DTYPE)
        if op in ("matmul_add", "matmul_mul"):
            c = torch.randn(S, S, dtype=DTYPE)
            return (a, b, c)
        return (a, b)
    if op in ("add", "mul"):
        return (torch.randn(S, S, dtype=DTYPE), torch.randn(S, S, dtype=DTYPE))
    if op == "softmax":
        return (torch.randn(S, S, dtype=DTYPE),)
    if op == "where_softmax":
        x = torch.randn(S, S, dtype=DTYPE)
        mask = torch.ones(S, S, dtype=torch.bool).tril()
        return (x, mask)
    raise ValueError(op)


CONFIGS = [
    ("single", "matmul", MatMul),
    ("single", "add", Add),
    ("single", "mul", Mul),
    ("single", "softmax", Softmax),
    ("chain", "matmul_add", MatMulAdd),
    ("chain", "matmul_mul", MatMulMul),
    ("chain", "matmul_softmax", MatMulSoftmax),
    ("chain", "where_softmax", WhereSoftmax),
]


def profile_config(kind, op, mod_cls, S, dev):
    inputs = make_inputs(op, S)
    ep = torch.export.export(mod_cls(), inputs)
    t0 = time.monotonic()
    cm = CompileModule.from_exported(ep)
    compile_s = time.monotonic() - t0

    dev_inputs = tuple(t.to(dev) for t in inputs)
    cm.to(dev)
    cm(*dev_inputs, profiles=None, device=dev)  # warmup

    rows = []
    for run in range(N_RUNS):
        profiles = cm.generate_profiles(dev)
        t0 = time.perf_counter()
        cm(*dev_inputs, profiles=profiles, device=dev)
        wall_us = (time.perf_counter() - t0) * 1e6

        device_indice = [[p.device.index for p in inner] for inner in profiles]
        profiles_cpu = [[p.cpu() for p in inner] for inner in profiles]
        spans = cm.edf.npu_node.build_tuc_profile_spans(profiles_cpu, device_indice, 10**9)

        agg = defaultdict(lambda: [0, 0.0])  # name -> [count, sum_us]
        for ev in spans:
            agg[ev.name][0] += 1
            agg[ev.name][1] += ev.time_range.elapsed_us()

        row = dict(kind=kind, op=op, S=S, run=run,
                   compile_s=round(compile_s, 1),
                   wall_us=round(wall_us, 1),
                   n_spans=len(spans))
        for name, (cnt, total) in agg.items():
            key = name.replace("Renegade::", "").replace("::", "_").lower()
            row[f"{key}_n"] = cnt
            row[f"{key}_us"] = round(total, 2)
        rows.append(row)
        # span 상세는 첫 run 만 stdout 에
        if run == 0:
            print(f"    spans: " + ", ".join(
                f"{n}×{c}={t:.1f}us" for n, (c, t) in sorted(agg.items())), flush=True)
    return rows


def main():
    dev = torch.device("rngd", 0)
    all_rows = []
    for kind, op, mod_cls in CONFIGS:
        for S in SHAPES:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {kind}/{op} S={S}", flush=True)
            try:
                rows = profile_config(kind, op, mod_cls, S, dev)
                all_rows.extend(rows)
            except Exception as e:
                print(f"    FAIL: {str(e)[:200]}", flush=True)
                all_rows.append(dict(kind=kind, op=op, S=S, run=-1,
                                     error=str(e)[:300]))

    fields = sorted({k for r in all_rows for k in r}, key=lambda k: (k not in
                    ("kind", "op", "S", "run", "compile_s", "wall_us", "n_spans"), k))
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n[DONE] {OUT_CSV} ({len(all_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
