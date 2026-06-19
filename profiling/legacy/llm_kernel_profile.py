#!/usr/bin/env python3
"""LLM 추론 커널 택소노미 프로파일러 (Lv0: 바닥 레이어).

LLM(Llama-style decoder) 추론에 쓰이는 모든 커널을 개별적으로 RNGD 에서
프로파일링한다. 각 커널마다:
  - DMA us (HBM 전송)  vs  TuExec us (연산)
  - 유효 BW (bytes/DMA), 유효 TFLOPS (flops/TuExec)
  - arithmetic intensity (flops/bytes)
  - bound 분류 (memory vs compute)
를 뽑아, 어느 커널에 SRAM-타일링 헤드룸이 있는지 바닥부터 정량화한다.

프로파일러 도구: RNGDProfiler(kineto) 는 torch 2.10+xpu / Intel PTI 충돌 →
검증된 저수준 경로 CompileModule → generate_profiles → build_tuc_profile_spans 사용
(kernel_profiler_eval.py / attn_decode_profile.py 와 동일).

사용법:
  python llm_kernel_profile.py [decode|prefill|all] [ctx1,ctx2,...]
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

# ---- Llama-3.1-8B config ----
D = 4096           # hidden
NH = 32            # query heads
HD = 128           # head dim  (NH*HD = 4096)
NKV = 8            # kv heads (GQA)
KVD = NKV * HD     # 1024
INTER = 14336      # mlp intermediate
VOCAB = 128256
DTYPE = torch.bfloat16
BYTES = 2          # bf16
PEAK_TBS = 1.5     # HBM peak
PEAK_TFLOPS = 256  # bf16

N_RUNS = 3
REGIME = sys.argv[1] if len(sys.argv) > 1 else "all"
CTXS = ([int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2
        else [2048, 8192, 16384])
OUT_CSV = os.path.join(HERE, "llm_kernel_profile_results.csv")


# ============ 커널 모듈들 ============
class MatMul(torch.nn.Module):
    """projection / lm_head: x[B,S,K] @ W[K,N] -> [B,S,N]. W = HBM weight."""
    def forward(self, x, w):
        return x @ w


class RMSNorm(torch.nn.Module):
    def forward(self, x, w):
        v = x.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-5)
        return (v * w.float()).to(x.dtype)


class SiLU(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.silu(x)


class SwiGLUMul(torch.nn.Module):
    def forward(self, g, u):
        return torch.nn.functional.silu(g) * u


class ResidualAdd(torch.nn.Module):
    def forward(self, a, b):
        return a + b


class RoPE(torch.nn.Module):
    """q[B,NH,S,HD] rotate by cos/sin[S,HD]."""
    def forward(self, q, cos, sin):
        q1, q2 = q[..., : HD // 2], q[..., HD // 2:]
        rot = torch.cat((-q2, q1), dim=-1)
        return q * cos + rot * sin


class AttnScores(torch.nn.Module):
    """GQA decode: q[B,NH,Sq,HD] · k[B,NKV,Skv,HD] (k broadcast NKV->NH)."""
    def forward(self, q, k):
        k = k.repeat_interleave(NH // NKV, dim=1)          # on-chip broadcast
        return (q @ k.transpose(-1, -2)) * (HD ** -0.5)    # [B,NH,Sq,Skv]


class Softmax(torch.nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)


class AttnAV(torch.nn.Module):
    """probs[B,NH,Sq,Skv] · v[B,NKV,Skv,HD] (v broadcast)."""
    def forward(self, p, v):
        v = v.repeat_interleave(NH // NKV, dim=1)
        return p @ v                                        # [B,NH,Sq,HD]


class Embedding(torch.nn.Module):
    def forward(self, table, idx):
        return torch.nn.functional.embedding(idx, table)


class Argmax(torch.nn.Module):
    def forward(self, logits):
        return torch.argmax(logits, dim=-1)


# ============ 커널 스펙: (name, module, build(B,Sq,ctx)->inputs, flops, bytes) ============
def lin(name, K, N):
    """linear projection 스펙 헬퍼. x[B,Sq,K] @ W[K,N]."""
    def build(B, Sq, ctx):
        return (torch.randn(B, Sq, K, dtype=DTYPE), torch.randn(K, N, dtype=DTYPE))
    def flops(B, Sq, ctx):
        return 2 * B * Sq * K * N
    def nbytes(B, Sq, ctx):
        return BYTES * (B * Sq * K + K * N + B * Sq * N)  # x + W + out
    return (name, MatMul, build, flops, nbytes)


KERNELS = [
    # --- projections (GEMM) ---
    lin("q_proj",    D, NH * HD),
    lin("k_proj",    D, KVD),
    lin("v_proj",    D, KVD),
    lin("o_proj",    NH * HD, D),
    lin("gate_proj", D, INTER),
    lin("up_proj",   D, INTER),
    lin("down_proj", INTER, D),
    lin("lm_head",   D, VOCAB),

    # --- norm / activation / residual (elementwise) ---
    ("rmsnorm", RMSNorm,
     lambda B, Sq, ctx: (torch.randn(B, Sq, D, dtype=DTYPE), torch.randn(D, dtype=DTYPE)),
     lambda B, Sq, ctx: 4 * B * Sq * D,
     lambda B, Sq, ctx: BYTES * (B * Sq * D + D + B * Sq * D)),
    ("silu", SiLU,
     lambda B, Sq, ctx: (torch.randn(B, Sq, INTER, dtype=DTYPE),),
     lambda B, Sq, ctx: 4 * B * Sq * INTER,
     lambda B, Sq, ctx: BYTES * (2 * B * Sq * INTER)),
    ("swiglu_mul", SwiGLUMul,
     lambda B, Sq, ctx: (torch.randn(B, Sq, INTER, dtype=DTYPE), torch.randn(B, Sq, INTER, dtype=DTYPE)),
     lambda B, Sq, ctx: 5 * B * Sq * INTER,
     lambda B, Sq, ctx: BYTES * (3 * B * Sq * INTER)),
    ("residual_add", ResidualAdd,
     lambda B, Sq, ctx: (torch.randn(B, Sq, D, dtype=DTYPE), torch.randn(B, Sq, D, dtype=DTYPE)),
     lambda B, Sq, ctx: B * Sq * D,
     lambda B, Sq, ctx: BYTES * (3 * B * Sq * D)),

    # --- position ---
    ("rope", RoPE,
     lambda B, Sq, ctx: (torch.randn(B, NH, Sq, HD, dtype=DTYPE),
                         torch.randn(Sq, HD, dtype=DTYPE), torch.randn(Sq, HD, dtype=DTYPE)),
     lambda B, Sq, ctx: 3 * B * NH * Sq * HD,
     lambda B, Sq, ctx: BYTES * (2 * B * NH * Sq * HD + 2 * Sq * HD)),

    # --- embedding / sampling ---
    ("embedding", Embedding,
     lambda B, Sq, ctx: (torch.randn(VOCAB, D, dtype=DTYPE),
                         torch.randint(0, VOCAB, (B, Sq), dtype=torch.int64)),
     lambda B, Sq, ctx: 0,
     lambda B, Sq, ctx: BYTES * (B * Sq * D) + 8 * B * Sq),  # gather: read only selected rows
    ("argmax", Argmax,
     lambda B, Sq, ctx: (torch.randn(B, Sq, VOCAB, dtype=DTYPE),),
     lambda B, Sq, ctx: B * Sq * VOCAB,
     lambda B, Sq, ctx: BYTES * (B * Sq * VOCAB)),
]

# --- attention core (ctx 의존) ---
ATTN_KERNELS = [
    ("attn_scores", AttnScores,
     lambda B, Sq, ctx: (torch.randn(B, NH, Sq, HD, dtype=DTYPE),
                         torch.randn(B, NKV, ctx, HD, dtype=DTYPE)),
     lambda B, Sq, ctx: 2 * B * NH * Sq * ctx * HD,
     lambda B, Sq, ctx: BYTES * (B * NH * Sq * HD + B * NKV * ctx * HD + B * NH * Sq * ctx)),
    ("softmax", Softmax,
     lambda B, Sq, ctx: (torch.randn(B, NH, Sq, ctx, dtype=DTYPE),),
     lambda B, Sq, ctx: 5 * B * NH * Sq * ctx,
     lambda B, Sq, ctx: BYTES * (2 * B * NH * Sq * ctx)),
    ("attn_av", AttnAV,
     lambda B, Sq, ctx: (torch.randn(B, NH, Sq, ctx, dtype=DTYPE),
                         torch.randn(B, NKV, ctx, HD, dtype=DTYPE)),
     lambda B, Sq, ctx: 2 * B * NH * Sq * ctx * HD,
     lambda B, Sq, ctx: BYTES * (B * NH * Sq * ctx + B * NKV * ctx * HD + B * NH * Sq * HD)),
]


def profile_one(name, mod_cls, inputs, dev):
    ep = torch.export.export(mod_cls(), inputs)
    t0 = time.monotonic()
    cm = CompileModule.from_exported(ep)
    compile_s = time.monotonic() - t0
    dev_inputs = tuple(t.to(dev) for t in inputs)
    cm.to(dev)
    cm(*dev_inputs, profiles=None, device=dev)  # warmup

    runs = []
    for run in range(N_RUNS):
        profiles = cm.generate_profiles(dev)
        t0 = time.perf_counter()
        cm(*dev_inputs, profiles=profiles, device=dev)
        wall_us = (time.perf_counter() - t0) * 1e6
        device_indice = [[p.device.index for p in inner] for inner in profiles]
        profiles_cpu = [[p.cpu() for p in inner] for inner in profiles]
        spans = cm.edf.npu_node.build_tuc_profile_spans(profiles_cpu, device_indice, 10**9)
        agg = defaultdict(float)
        for ev in spans:
            key = ev.name.replace("Renegade::", "").replace("::", "_").lower()
            agg[key] += ev.time_range.elapsed_us()
        runs.append((wall_us, dict(agg)))
    return compile_s, runs


def main():
    import statistics as st
    dev = torch.device("rngd", 0)
    regimes = []
    if REGIME in ("decode", "all"):
        for ctx in CTXS:
            regimes.append(("decode", 8, 1, ctx))      # B=8, Sq=1
    if REGIME in ("prefill", "all"):
        regimes.append(("prefill", 1, 2048, 2048))     # B=1, Sq=2048 chunk

    all_rows = []
    for reg, B, Sq, ctx in regimes:
        # ctx-독립 커널은 decode 의 첫 ctx 에서만 1회 (중복 측정 방지)
        ctx_indep_done = (reg == "decode" and ctx != CTXS[0])
        klist = list(ATTN_KERNELS)
        if not ctx_indep_done:
            klist = KERNELS + ATTN_KERNELS
        for name, mod_cls, build, flops_fn, bytes_fn in klist:
            tag = f"{reg} B{B} Sq{Sq} ctx{ctx}"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {name:14s} {tag}", flush=True)
            try:
                inputs = build(B, Sq, ctx)
                compile_s, runs = profile_one(name, mod_cls, inputs, dev)
                fl = flops_fn(B, Sq, ctx)
                by = bytes_fn(B, Sq, ctx)
                dma = st.median([r[1].get("dma", 0.0) for r in runs])
                tu = st.median([r[1].get("tuexec", 0.0) for r in runs])
                task = st.median([r[1].get("task", 0.0) for r in runs])
                wall = st.median([r[0] for r in runs])
                eff_bw = (by / 1e12) / (dma / 1e6) if dma else 0
                eff_tf = (fl / 1e12) / (tu / 1e6) if tu else 0
                ai = fl / by if by else 0
                bound = "mem" if dma >= tu else "cmp"
                row = dict(regime=reg, kernel=name, B=B, Sq=Sq, ctx=ctx,
                           bytes_mb=round(by / 1e6, 2), gflops=round(fl / 1e9, 2),
                           ai=round(ai, 1), dma_us=round(dma, 2), tuexec_us=round(tu, 2),
                           task_us=round(task, 2), wall_us=round(wall, 1),
                           eff_bw=round(eff_bw, 3), pct_bw=round(eff_bw / PEAK_TBS * 100),
                           eff_tflops=round(eff_tf, 1), pct_flops=round(eff_tf / PEAK_TFLOPS * 100),
                           bound=bound, compile_s=round(compile_s, 1))
                all_rows.append(row)
                print(f"    bytes={row['bytes_mb']}MB gflop={row['gflops']} ai={row['ai']} "
                      f"dma={row['dma_us']} tu={row['tuexec_us']} | "
                      f"BW={row['eff_bw']}({row['pct_bw']}%) TF={row['eff_tflops']}({row['pct_flops']}%) "
                      f"[{bound}]", flush=True)
            except Exception as e:
                print(f"    FAIL: {type(e).__name__}: {str(e)[:160]}", flush=True)
                all_rows.append(dict(regime=reg, kernel=name, B=B, Sq=Sq, ctx=ctx,
                                     error=str(e)[:200]))

    fields = ["regime", "kernel", "B", "Sq", "ctx", "bytes_mb", "gflops", "ai",
              "dma_us", "tuexec_us", "task_us", "wall_us", "eff_bw", "pct_bw",
              "eff_tflops", "pct_flops", "bound", "compile_s", "error"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n[DONE] {OUT_CSV} ({len(all_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
