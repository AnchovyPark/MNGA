#!/usr/bin/env python3
"""n-gram≤3 context 예측이 LAYER 길이 체인에서도 먹히나 — 실제 Llama-3.1-8B shape.

context_chain_ladder / analyze_context_attribution 방법을 layer 전체 길이로 확장.
layer critical-path 를 shape-valid 한 선형 체인으로:
  Q(qk) S(softmax) V(av) O(o_proj) R(residual) N(rmsnorm) G(gate) I(silu) X(mul) D(down) R(residual)
R-N-G 에서 attn↔MLP 경계를 넘는다.

train: 모든 ≤triple 연속 윈도우(29개) joint-compile → fused Task.
test : full 11-op 체인 + 경계 넘는 sub-span(6개) joint-compile.
model: op + 좌/우/ctx ridge (analyze_context_attribution 와 동일), ≤triple 학습.
질문 : ≤triple context 가 layer 길이 + 블록경계 체인을 예측하나?

실제 Llama-3.1-8B B=1 S=128 TP=8 bf16. 예측 맞으면 추가 프로파일 없이 Llama 적용.
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
import furiosa.torch  # noqa: F401
from furiosa.torch import native_device as nd

nd.set_fusion(8)

from furiosa.torch.custom_ops import CompileModule  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

D, NH, HD, KV, INTER = 4096, 32, 128, 8, 14336
B = 1
S = int(sys.argv[1]) if len(sys.argv) > 1 else 128
OUT_CSV = os.path.join(HERE, f"layer_ngram_s{S}_results.csv")
DTYPE = torch.bfloat16
N_RUNS = 3
BOS, EOS = "BOS", "EOS"

# op 기호 → 입력 shape (체인 내 고정)
IN_SHAPE = {
    "Q": (B, NH, S, HD), "S": (B, NH, S, S), "V": (B, NH, S, S),
    "O": (B, NH, S, HD), "R": (B, S, D), "N": (B, S, D),
    "G": (B, S, D), "I": (B, S, INTER), "X": (B, S, INTER), "D": (B, S, INTER),
}
OP_NAME = {"Q": "qk", "S": "softmax", "V": "av", "O": "o_proj", "R": "residual",
           "N": "rmsnorm", "G": "gate", "I": "silu", "X": "mul", "D": "down"}

CHAIN = "QSVORNGIXDR"


def union_us(iv):
    if not iv:
        return 0.0
    iv = sorted(iv); t = 0.0; cs, ce = iv[0]
    for s, e in iv[1:]:
        if s <= ce: ce = max(ce, e)
        else: t += ce - cs; cs, ce = s, e
    return t + (ce - cs)


class Window(torch.nn.Module):
    """op 기호 시퀀스를 순서대로 적용. 각 위치별 buffer 등록."""
    def __init__(self, seq):
        super().__init__()
        self.seq = seq
        for i, op in enumerate(seq):
            if op == "Q":
                self.register_buffer(f"b{i}", torch.randn(B, NH, HD, S, dtype=DTYPE))
            elif op == "V":
                self.register_buffer(f"b{i}", torch.randn(B, NH, S, HD, dtype=DTYPE))
            elif op == "O":
                self.register_buffer(f"b{i}", torch.randn(NH * HD, D, dtype=DTYPE))
            elif op == "R":
                self.register_buffer(f"b{i}", torch.randn(B, S, D, dtype=DTYPE))
            elif op == "N":
                self.register_buffer(f"b{i}", torch.randn(D, dtype=DTYPE))
            elif op == "G":
                self.register_buffer(f"b{i}", torch.randn(D, INTER, dtype=DTYPE))
            elif op == "X":
                self.register_buffer(f"b{i}", torch.randn(B, S, INTER, dtype=DTYPE))
            elif op == "D":
                self.register_buffer(f"b{i}", torch.randn(INTER, D, dtype=DTYPE))

    def forward(self, x):
        for i, op in enumerate(self.seq):
            b = getattr(self, f"b{i}", None)
            if op == "Q":
                x = x @ b
            elif op == "S":
                x = torch.softmax(x, dim=-1)
            elif op == "V":
                x = x @ b
            elif op == "O":
                x = x.permute(0, 2, 1, 3).reshape(B, S, NH * HD) @ b
            elif op == "R":
                x = x + b
            elif op == "N":
                v = x.float()
                x = (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-5) * b.float()).to(x.dtype)
            elif op == "G":
                x = x @ b
            elif op == "I":
                x = F.silu(x)
            elif op == "X":
                x = x * b
            elif op == "D":
                x = x @ b
        return x


def measure(seq):
    dev = torch.device("rngd", 0)
    x = torch.randn(*IN_SHAPE[seq[0]], dtype=DTYPE)
    cm = CompileModule.from_exported(torch.export.export(Window(seq), (x,)))
    cm.to(dev); xd = x.to(dev)
    cm(xd, profiles=None, device=dev)
    ts = []
    for _ in range(N_RUNS):
        pr = cm.generate_profiles(dev); cm(xd, profiles=pr, device=dev)
        di = [[p.device.index for p in inn] for inn in pr]
        pc = [[p.cpu() for p in inn] for inn in pr]
        spans = cm.edf.npu_node.build_tuc_profile_spans(pc, di, 10**9)
        ts.append(sum(e.time_range.elapsed_us() for e in spans if e.name == "Task"))
    return st.median(ts)


# ---- attribution (analyze_context_attribution 와 동일) ----
def featurize(seq):
    f = Counter()
    for i, op in enumerate(seq):
        left = seq[i - 1] if i > 0 else BOS
        right = seq[i + 1] if i + 1 < len(seq) else EOS
        f[f"op:{op}"] += 1.0
        f[f"L:{left}->{op}"] += 1.0
        f[f"R:{op}->{right}"] += 1.0
        f[f"C:{left}|{op}|{right}"] += 1.0
    return f


def solve(a, b):
    n = len(b); aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(aug[r][c]))
        if abs(aug[piv][c]) < 1e-12: continue
        aug[c], aug[piv] = aug[piv], aug[c]
        dv = aug[c][c]
        for k in range(c, n + 1): aug[c][k] /= dv
        for r in range(n):
            if r == c: continue
            fct = aug[r][c]
            if fct:
                for k in range(c, n + 1): aug[r][k] -= fct * aug[c][k]
    return [aug[i][n] for i in range(n)]


def fit_ridge(samples, ridge=1.0):
    names = sorted({k for f, _ in samples for k in f})
    idx = {n: i for i, n in enumerate(names)}; n = len(names)
    ata = [[0.0] * n for _ in range(n)]; aty = [0.0] * n
    for f, y in samples:
        items = [(idx[k], v) for k, v in f.items()]
        for i, vi in items:
            aty[i] += vi * y
            for j, vj in items: ata[i][j] += vi * vj
    for i in range(n): ata[i][i] += ridge
    return names, solve(ata, aty)


def predict(seq, names, coef):
    cm = dict(zip(names, coef))
    return sum(cm.get(k, 0.0) * v for k, v in featurize(seq).items())


def main():
    singles = list("QSVORNGIXD")
    doubles = [CHAIN[i:i+2] for i in range(len(CHAIN) - 1)]
    triples = [CHAIN[i:i+3] for i in range(len(CHAIN) - 2)]
    train = singles + doubles + triples
    tests = ["ORNG", "QSVOR", "VORNGI", "NGIXDR", "SVORNGIX", CHAIN]

    print(f"=== layer n-gram windows (Llama-3.1-8B B={B} S={S} TP=8) ===", flush=True)

    task = {}
    rows = []
    for split, seqs in [("train", train), ("test", tests)]:
        for seq in seqs:
            if seq in task:
                continue
            try:
                t = measure(seq)
            except Exception as exc:
                print(f"  [{split:5s} {seq:11s}] FAIL {type(exc).__name__}: {str(exc)[:90]}", flush=True)
                continue
            task[seq] = t
            rows.append(dict(split=split, sequence=seq, length=len(seq),
                             ops="->".join(OP_NAME[o] for o in seq), task_us=round(t, 2)))
            print(f"  [{split:5s} len{len(seq)} {seq:11s}] task={t:8.1f}us", flush=True)

    fields = ["split", "sequence", "length", "ops", "task_us"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\n[DONE] {OUT_CSV} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
