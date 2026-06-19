#!/usr/bin/env python3
"""Lv0: decode-attention 을 DMA(HBM 이동) vs 연산(TuExec) 으로 분해.

목적: long-context decode 의 KV 병목이 정말 HBM 전송에서 오는지, 그리고
유효 대역폭이 peak(1.5TB/s) 대비 얼마인지(=SRAM 타일링 헤드룸)를 커널 레벨에서 측정.

decode 1-step attention (q_len=1) 을 context S 별로 컴파일·프로파일:
  scores = q @ k^T   → K 전체 읽음
  probs  = softmax(scores)
  out    = probs @ v → V 전체 읽음
K,V 는 [H,S,D] bf16 → HBM read = 2*H*S*D*2 bytes.

kernel_profiler_eval.py 와 동일한 저수준 경로 사용.
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
OUT_CSV = os.path.join(HERE, "attn_decode_profile_results.csv")

CTXS = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1024, 2048, 4096, 8192, 16384]
N_RUNS = 3
DTYPE = torch.bfloat16
H, D = 32, 128          # heads, head_dim (MHA 형태 — BW 활용률 측정엔 충분)
PEAK_TBS = 1.5          # RNGD HBM peak


class DecodeAttn(torch.nn.Module):
    def forward(self, q, k, v):
        # q:[H,1,D]  k,v:[H,S,D]
        scores = (q @ k.transpose(-1, -2)) * (D ** -0.5)   # [H,1,S]
        probs = torch.softmax(scores, dim=-1)
        return probs @ v                                    # [H,1,D]


def make_inputs(S):
    q = torch.randn(H, 1, D, dtype=DTYPE)
    k = torch.randn(H, S, D, dtype=DTYPE)
    v = torch.randn(H, S, D, dtype=DTYPE)
    return (q, k, v)


def kv_bytes(S):
    return 2 * H * S * D * 2  # K+V, bf16


def profile_ctx(S, dev):
    inputs = make_inputs(S)
    ep = torch.export.export(DecodeAttn(), inputs)
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

        agg = defaultdict(lambda: [0, 0.0])
        for ev in spans:
            agg[ev.name][0] += 1
            agg[ev.name][1] += ev.time_range.elapsed_us()

        row = dict(S=S, run=run, compile_s=round(compile_s, 1),
                   wall_us=round(wall_us, 1), n_spans=len(spans),
                   kv_mb=round(kv_bytes(S) / 1e6, 2))
        for name, (cnt, total) in agg.items():
            key = name.replace("Renegade::", "").replace("::", "_").lower()
            row[f"{key}_n"] = cnt
            row[f"{key}_us"] = round(total, 2)
        rows.append(row)
        if run == 0:
            print(f"    S={S} spans: " + ", ".join(
                f"{n}×{c}={t:.1f}us" for n, (c, t) in sorted(agg.items())), flush=True)
    return rows


def main():
    dev = torch.device("rngd", 0)
    all_rows = []
    for S in CTXS:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ctx={S}", flush=True)
        try:
            all_rows.extend(profile_ctx(S, dev))
        except Exception as e:
            print(f"    FAIL: {str(e)[:200]}", flush=True)
            all_rows.append(dict(S=S, run=-1, error=str(e)[:300]))

    fields = sorted({k for r in all_rows for k in r},
                    key=lambda k: (k not in ("S", "run", "compile_s", "wall_us", "n_spans", "kv_mb"), k))
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n[DONE] {OUT_CSV} ({len(all_rows)} rows)", flush=True)

    # 요약: DMA vs TuExec, 유효 BW
    print("\n=== 요약 (median over runs) ===")
    import statistics as st
    by_s = defaultdict(lambda: defaultdict(list))
    for r in all_rows:
        if r.get("run", -1) < 0:
            continue
        for k in ("dma_us", "tuexec_us", "task_us", "parallelcopy_us"):
            if k in r:
                by_s[r["S"]][k].append(r[k])
    print(f"{'ctx':>6} {'KV(MB)':>8} {'dma_us':>9} {'tuexec_us':>10} {'task_us':>9} {'effBW(TB/s)':>12} {'%peak':>7}")
    for S in CTXS:
        if S not in by_s:
            continue
        m = {k: st.median(v) for k, v in by_s[S].items() if v}
        dma = m.get("dma_us", 0)
        bw = (kv_bytes(S) / 1e12) / (dma / 1e6) if dma else 0
        print(f"{S:>6} {kv_bytes(S)/1e6:>8.1f} {dma:>9.1f} "
              f"{m.get('tuexec_us',0):>10.1f} {m.get('task_us',0):>9.1f} "
              f"{bw:>12.3f} {bw/PEAK_TBS*100:>6.0f}%")


if __name__ == "__main__":
    main()
