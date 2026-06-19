#!/usr/bin/env python3
"""커널 합성 격차(composition gap) 프로파일러.

연구 질문: 커널 하나를 격리해 잰 프로파일이, 커널 여러 개를 붙여 실행할 때
정확히 어떻게 바뀌는가.

세 가지 실행 모드를 대조한다:
  iso   : 커널 k 를 단독 export→compile→실행 (fresh 입력).      → T_pred = Σ T_iso
  sep   : 각 커널을 따로 compile 후 디바이스에서 back-to-back 실행
          (중간텐서 HBM 왕복, 컴파일러 fusion 없음).            → T_sep
  comp  : 커널 1..k 를 한 그래프로 joint-compile→실행 (fusion). → T_joint

분해:
  ε_sched  = T_sep   − T_pred   (격리→연속실행: dispatch/스케줄/오버헤드 분할상환)
  ε_fusion = T_joint − T_sep    (순수 컴파일러 fusion: 중간텐서 SRAM 잔류+epilogue+overlap)
  ε_dma    = joint_dma − Σiso_dma (fusion 이 줄인 HBM 트래픽, span 기반)

누적 prefix(길이 1→N)로 측정해 커널을 하나씩 더할 때 어떻게 변하는지 추적.

프로파일러: 저수준 경로(CompileModule → generate_profiles → build_tuc_profile_spans).
RNGDProfiler(kineto) 는 torch 2.10+xpu / Intel PTI 충돌로 사용 불가.

대상 체인 = MLP 블록(shape 보존 위해 square 단순화):
  rmsnorm → matmul(Wg) → silu → mul(U) → matmul(Wd) → add(res)

사용법: python kernel_composition.py [decode|prefill|all]
"""
import csv
import os
import statistics as st
import sys
import time
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn.functional as F
import furiosa.torch  # noqa: F401
from furiosa.torch.custom_ops import CompileModule

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "kernel_composition_results.csv")

D = 4096
DTYPE = torch.bfloat16
N_RUNS = 3
REGIME = sys.argv[1] if len(sys.argv) > 1 else "all"

# 체인: (op, aux 종류). aux 는 buffer(=HBM 상수, weight/operand 모사).
#   'w'=matmul weight[D,D]  'g'=rmsnorm gain[D]  'op'=elementwise operand[B,S,D]
CHAIN = [
    ("rmsnorm", "g"),
    ("matmul",  "w"),
    ("silu",    None),
    ("mul",     "op"),
    ("matmul",  "w"),
    ("add",     "op"),
]


def make_aux(kind, B, S):
    if kind == "w":
        return torch.randn(D, D, dtype=DTYPE)
    if kind == "g":
        return torch.randn(D, dtype=DTYPE)
    if kind == "op":
        return torch.randn(B, S, D, dtype=DTYPE)
    return None


class ChainModule(torch.nn.Module):
    """ops 를 순서대로 h[B,S,D] 에 적용. aux 는 buffer 로 보유. (shape 보존)"""
    def __init__(self, ops, auxs):
        super().__init__()
        self.op_names = [o[0] for o in ops]
        self._has_aux = []
        for i, (_op, aux) in enumerate(zip(ops, auxs)):
            if aux is not None:
                self.register_buffer(f"aux_{i}", aux)
                self._has_aux.append(True)
            else:
                self._has_aux.append(False)

    def _apply(self, name, h, aux):
        if name == "matmul":
            return h @ aux
        if name == "add":
            return h + aux
        if name == "mul":
            return h * aux
        if name == "silu":
            return F.silu(h)
        if name == "softmax":
            return torch.softmax(h, dim=-1)
        if name == "rmsnorm":
            v = h.float()
            v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-5)
            return (v * aux.float()).to(h.dtype)
        raise ValueError(name)

    def forward(self, h):
        for i, name in enumerate(self.op_names):
            aux = getattr(self, f"aux_{i}") if self._has_aux[i] else None
            h = self._apply(name, h, aux)
        return h


def compile_warm(mod, h, dev):
    ep = torch.export.export(mod, (h,))
    cm = CompileModule.from_exported(ep)
    cm.to(dev)
    cm(h.to(dev), profiles=None, device=dev)  # warmup
    return cm


def spans_of(cm, hd, dev):
    """profiled 실행 → (wall_us, {span:us}) median."""
    runs = []
    for _ in range(N_RUNS):
        profiles = cm.generate_profiles(dev)
        t0 = time.perf_counter()
        cm(hd, profiles=profiles, device=dev)
        wall = (time.perf_counter() - t0) * 1e6
        di = [[p.device.index for p in inner] for inner in profiles]
        pc = [[p.cpu() for p in inner] for inner in profiles]
        sp = defaultdict(float)
        for ev in cm.edf.npu_node.build_tuc_profile_spans(pc, di, 10**9):
            sp[ev.name.replace("Renegade::", "").replace("::", "_").lower()] += \
                ev.time_range.elapsed_us()
        runs.append((wall, dict(sp)))
    wall = st.median([r[0] for r in runs])
    keys = set().union(*[r[1].keys() for r in runs]) if runs else set()
    spans = {k: st.median([r[1].get(k, 0.0) for r in runs]) for k in keys}
    return wall, spans


def seq_wall(cms, hd, dev):
    """cms 를 디바이스에서 back-to-back 실행한 총 wall_us median (profiling 없음)."""
    runs = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        out = hd
        for cm in cms:
            out = cm(out, profiles=None, device=dev)
        runs.append((time.perf_counter() - t0) * 1e6)
    return st.median(runs)


def main():
    dev = torch.device("rngd", 0)
    regimes = []
    if REGIME in ("decode", "all"):
        regimes.append(("decode", 8, 1))      # 중간텐서 작음 → ε 오버헤드 지배 가설
    if REGIME in ("prefill", "all"):
        regimes.append(("prefill", 8, 512))   # 중간텐서 큼 → fusion 가시 가설

    rows = []
    for reg, B, S in regimes:
        auxs = [make_aux(kind, B, S) for (_n, kind) in CHAIN]
        h0 = torch.randn(B, S, D, dtype=DTYPE)
        hd = h0.to(dev)

        # --- iso: 각 op 단독 (fresh 입력). cm 보관해 sep 에서 재사용 ---
        iso_wall, iso_dma, iso_tu, iso_cms = [], [], [], []
        for i, (name, _k) in enumerate(CHAIN):
            cm = compile_warm(ChainModule([CHAIN[i]], [auxs[i]]), h0, dev)
            w, sp = spans_of(cm, hd, dev)
            iso_cms.append(cm)
            iso_wall.append(w); iso_dma.append(sp.get('dma', 0)); iso_tu.append(sp.get('tuexec', 0))
            print(f"[{datetime.now():%H:%M:%S}] {reg} ISO {i}:{name:8s} "
                  f"wall={w:.1f} dma={sp.get('dma',0):.1f} tu={sp.get('tuexec',0):.1f}", flush=True)
            rows.append(dict(regime=reg, B=B, S=S, kind="iso", length=1, idx=i, op=name,
                             wall_us=round(w, 1), dma_us=round(sp.get('dma', 0), 2),
                             tuexec_us=round(sp.get('tuexec', 0), 2),
                             task_us=round(sp.get('task', 0), 2)))

        # --- sep & comp: 누적 prefix 1..N ---
        for k in range(1, len(CHAIN) + 1):
            pred_w = sum(iso_wall[:k]); pred_dma = sum(iso_dma[:k]); pred_tu = sum(iso_tu[:k])
            # sep: 격리 cm 들을 back-to-back
            t_sep = seq_wall(iso_cms[:k], hd, dev)
            # comp: joint-compile
            cm_j = compile_warm(ChainModule(CHAIN[:k], auxs[:k]), h0, dev)
            t_joint, spj = spans_of(cm_j, hd, dev)
            j_dma, j_tu = spj.get('dma', 0), spj.get('tuexec', 0)
            eps_sched = t_sep - pred_w
            eps_fusion = t_joint - t_sep
            eps_dma = j_dma - pred_dma
            print(f"[{datetime.now():%H:%M:%S}] {reg} k={k} pred(Σiso)={pred_w:.1f} "
                  f"sep={t_sep:.1f}(ε_sched={eps_sched:+.1f}) "
                  f"joint={t_joint:.1f}(ε_fusion={eps_fusion:+.1f}) | "
                  f"dma: Σiso={pred_dma:.1f} joint={j_dma:.1f}(ε_dma={eps_dma:+.1f})", flush=True)
            rows.append(dict(regime=reg, B=B, S=S, kind="chain", length=k, idx=k - 1,
                             op=CHAIN[k - 1][0],
                             pred_wall=round(pred_w, 1), sep_wall=round(t_sep, 1),
                             joint_wall=round(t_joint, 1),
                             eps_sched=round(eps_sched, 1), eps_fusion=round(eps_fusion, 1),
                             eps_total=round(t_joint - pred_w, 1),
                             pred_dma=round(pred_dma, 2), joint_dma=round(j_dma, 2),
                             eps_dma=round(eps_dma, 2),
                             pred_tuexec=round(pred_tu, 2), joint_tuexec=round(j_tu, 2)))

    fields = ["regime", "B", "S", "kind", "length", "idx", "op",
              "wall_us", "dma_us", "tuexec_us", "task_us",
              "pred_wall", "sep_wall", "joint_wall",
              "eps_sched", "eps_fusion", "eps_total",
              "pred_dma", "joint_dma", "eps_dma", "pred_tuexec", "joint_tuexec"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n[DONE] {OUT_CSV} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
