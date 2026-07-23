#!/usr/bin/env python3
"""n-gram 학습 데이터 생성 (2026.3.0, device-cycle) — 4단계 ML 예측기용.

Tokenwise op chain [q,k,v,o,gate,up,down]의 연속 window(크기 1~3 + 전체)를 여러 seq에서
device-cycle로 측정 → (특징, latency) 데이터점. 이걸로 ML 학습(특징→latency) → 안 본 조합/seq 예측.

특징: seq, window크기, 시작pos, op set(one-hot), 총 FLOPs, 총 weight바이트, 총 input바이트.
타겟: device-cycle latency (min).

사용: TUC_PROFILE_LEVEL=info RUST_LOG=info,span::tuc=info \
      /home/furiosa/venv3030/bin/python ngram_dataset_30.py
"""
import os, sys, json, csv, statistics as st

import torch
import furiosa.torch as ft
ft.set_fusion(8)
from furiosa.torch._C.config.compiler import CompilerConfig, TacticHintConfig
from furiosa.torch.profiler import Profiler

DEV = "furiosa:0"
D, INTER, NH, HD, KV = 2048, 8192, 32, 64, 8
DT = torch.bfloat16
SDIR = "/tmp/claude-1002/-home-furiosa------pjh-rngd/8464c59e-3432-4d59-9ccc-d34e43519088/scratchpad"
# seq 인자 1개면 그 seq만(fresh 프로세스로 degrade 회피), 없으면 전체
SEQS = [int(sys.argv[1])] if len(sys.argv) > 1 else [128, 256, 512, 1024]
OUT = f"profiling/production_profile/ngram_dataset_s{SEQS[0]}.csv" if len(sys.argv) > 1 else "profiling/production_profile/ngram_dataset.csv"
OPS = ["q", "k", "v", "o", "gate", "up", "down"]
SPEC = {"q": ("x", D, NH*HD), "k": ("x", D, KV*HD), "v": ("x", D, KV*HD),
        "o": ("attn", NH*HD, D), "gate": ("x", D, INTER), "up": ("x", D, INTER),
        "down": ("interin", INTER, D)}


def cfg():
    return CompilerConfig(tactic_hint=TacticHintConfig.ForLlmModelComputeBound,
                          scheduler_beam_search=True, use_attention_kernel=True)


class Window(torch.nn.Module):
    def __init__(self, subset):
        super().__init__(); self.subset = list(subset)
        for op in self.subset:
            _, K, N = SPEC[op]
            self.register_buffer(f"w_{op}", torch.randn(K, N, dtype=DT))
    def forward(self, x, attn, interin):
        src = {"x": x, "attn": attn, "interin": interin}
        return tuple(torch.mm(src[SPEC[op][0]], getattr(self, f"w_{op}")) for op in self.subset)


def _measure_once(cm, xds, pp):
    with Profiler(profile_path=pp):
        for _ in range(20):
            cm(*xds)
    d = json.load(open(pp)); seen = {}
    for e in d:
        if e.get("name") != "Task":
            continue
        a = e["args"]; seen[(a["begin_cycle"], a["end_cycle"], a.get("cluster_index"))] = int(a["cycle_actual"])
    v = sorted(seen.values())
    return min(v) if v else 0


def dev_cyc(subset, S):
    """0(프로파일러 캡처 miss)이면 최대 4회 재측정. 그래도 0이면 None(데이터에서 제외)."""
    mod = Window(subset)
    xs = (torch.randn(S, D, dtype=DT), torch.randn(S, NH*HD, dtype=DT), torch.randn(S, INTER, dtype=DT))
    cm = ft.CompileModule.from_module(mod, xs, compiler_config=cfg()).to(DEV)
    xds = [x.to(DEV) for x in xs]
    for _ in range(8):
        cm(*xds)
    pp = f"{SDIR}/ds_{'_'.join(subset)}_{S}.json"
    for attempt in range(4):
        c = _measure_once(cm, xds, pp)
        if c > 0:
            return c
    return None


def feats(subset, S):
    flops = bytes_w = bytes_in = 0
    for op in subset:
        _, K, N = SPEC[op]
        flops += 2 * S * K * N
        bytes_w += K * N * 2
        bytes_in += S * K * 2
    row = {"seq": S, "wsize": len(subset), "start": OPS.index(subset[0]),
           "flops": flops, "wbytes": bytes_w, "inbytes": bytes_in,
           "ops": "|".join(subset)}
    for op in OPS:
        row[f"has_{op}"] = int(op in subset)
    return row


def main():
    # 연속 window: 크기 1,2,3 + 전체(7)
    windows = []
    for sz in (1, 2, 3):
        for i in range(len(OPS) - sz + 1):
            windows.append(OPS[i:i+sz])
    windows.append(OPS[:])  # 전체
    rows = []; skipped = 0
    for S in SEQS:
        for w in windows:
            cyc = dev_cyc(w, S)
            if cyc is None:
                skipped += 1
                print(f"seq={S} {'|'.join(w):30s} -> SKIP (4회 재측정 실패)", flush=True)
                continue
            r = feats(w, S); r["cycle"] = cyc
            rows.append(r)
            print(f"seq={S} {'|'.join(w):30s} -> {cyc} cyc", flush=True)
    print(f"\n총 {len(rows)}행, 스킵 {skipped}", flush=True)
    with open(OUT, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print(f"\n저장: {OUT} ({len(rows)} rows)", flush=True)
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
