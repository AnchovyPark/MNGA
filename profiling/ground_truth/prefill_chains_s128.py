#!/usr/bin/env python3
"""prefill S=128 을 real-Llama-chain 으로 재구성 → ground truth 20.1ms 수렴 확인.

두 지배 체인을 길이(depth) 늘려가며 fused 측정 (S=128, TP=8):
  MLP:  gate_proj -> silu -> mul(up) -> down_proj -> residual
  ATTN: qk -> softmax -> av -> o_proj(reshape) -> residual
각 depth 의 fused vs 격리합 비교 → fusion 이득 누적.
그 다음 layer 를 fused 체인으로 재구성해 ×32 + lm_head → 예측 prefill 대조.

Llama-3.1-8B: D=4096 NH=32 KV=8 HD=128 INTER=14336 L=32 VOCAB=128256
"""
import statistics as st
import torch
import torch.nn.functional as F
import furiosa.torch  # noqa: F401
from furiosa.torch import native_device as nd

nd.set_fusion(8)
from furiosa.torch.custom_ops import CompileModule  # noqa: E402

D, NH, KV, HD, INTER, L, VOCAB = 4096, 32, 8, 128, 14336, 32, 128256
B, S = 1, 128
DTYPE = torch.bfloat16
N_RUNS = 3


def measure_task(mod, x):
    dev = torch.device("rngd", 0)
    cm = CompileModule.from_exported(torch.export.export(mod, (x,)))
    cm.to(dev); xd = x.to(dev)
    cm(xd, profiles=None, device=dev)
    ts = []
    for _ in range(N_RUNS):
        pr = cm.generate_profiles(dev); cm(xd, profiles=pr, device=dev)
        di = [[p.device.index for p in inner] for inner in pr]
        pc = [[p.cpu() for p in inner] for inner in pr]
        spans = cm.edf.npu_node.build_tuc_profile_spans(pc, di, 10**9)
        ts.append(sum(e.time_range.elapsed_us() for e in spans if e.name == "Task"))
    return st.median(ts)


class MLPChain(torch.nn.Module):
    """gate -> silu -> mul(up) -> down -> residual, 앞 depth 개만 적용."""
    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        self.register_buffer("wg", torch.randn(D, INTER, dtype=DTYPE))
        self.register_buffer("up", torch.randn(B, S, INTER, dtype=DTYPE))
        self.register_buffer("wd", torch.randn(INTER, D, dtype=DTYPE))
        self.register_buffer("res", torch.randn(B, S, D, dtype=DTYPE))

    def forward(self, x):
        x = x @ self.wg                       # 1 gate
        if self.depth >= 2: x = F.silu(x)     # 2 silu
        if self.depth >= 3: x = x * self.up   # 3 mul
        if self.depth >= 4: x = x @ self.wd   # 4 down
        if self.depth >= 5: x = x + self.res  # 5 residual
        return x


class AttnChain(torch.nn.Module):
    """qk -> softmax -> av -> o_proj(reshape) -> residual, 앞 depth 개만."""
    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        self.register_buffer("k", torch.randn(B, NH, HD, S, dtype=DTYPE))
        self.register_buffer("v", torch.randn(B, NH, S, HD, dtype=DTYPE))
        self.register_buffer("wo", torch.randn(NH * HD, D, dtype=DTYPE))
        self.register_buffer("res", torch.randn(B, S, D, dtype=DTYPE))

    def forward(self, q):
        x = q @ self.k                                 # 1 qk -> [B,NH,S,S]
        if self.depth >= 2: x = torch.softmax(x, -1)   # 2 softmax
        if self.depth >= 3: x = x @ self.v             # 3 av -> [B,NH,S,HD]
        if self.depth >= 4:                            # 4 o_proj (reshape+GEMM)
            x = x.permute(0, 2, 1, 3).reshape(B, S, NH * HD) @ self.wo
        if self.depth >= 5: x = x + self.res           # 5 residual
        return x


def main():
    mlp_in = torch.randn(B, S, D, dtype=DTYPE)
    attn_in = torch.randn(B, NH, S, HD, dtype=DTYPE)

    # 격리 단일 조각 (합성 대조용)
    mlp_pieces = {  # depth 순서
        1: measure_task(MLPChain(1), mlp_in),
    }
    attn_pieces = {1: measure_task(AttnChain(1), attn_in)}

    print(f"=== real chain fused vs 격리합 (S={S}, TP=8) ===", flush=True)
    print("[MLP  gate→silu→mul→down→residual]", flush=True)
    mlp_fused = {}
    for depth in (2, 3, 4, 5):
        mlp_fused[depth] = measure_task(MLPChain(depth), mlp_in)
    print("[ATTN qk→softmax→av→o_proj→residual]", flush=True)
    attn_fused = {}
    for depth in (2, 3, 4, 5):
        attn_fused[depth] = measure_task(AttnChain(depth), attn_in)

    # 개별 조각도 재서 격리합 계산 (fused 대조)
    # 조각 시간: MLPChain(depth) - MLPChain(depth-1) 근사 대신, 단일 op 직접 측정
    from math import nan
    # 단일 op (fusion 없이): gate, silu, mul, down, residual / qk, softmax, av, oproj, residual
    class One(torch.nn.Module):
        def __init__(s, k, w=None):
            super().__init__(); s.k = k
            if w is not None: s.register_buffer("w", w)
        def forward(s, x):
            if s.k == "mm": return x @ s.w
            if s.k == "silu": return F.silu(x)
            if s.k == "mul": return x * s.w
            if s.k == "add": return x + s.w
            if s.k == "sm": return torch.softmax(x, -1)
            if s.k == "oproj": return x.permute(0,2,1,3).reshape(B,S,NH*HD) @ s.w
    g = {
        "gate": measure_task(One("mm", torch.randn(D,INTER,dtype=DTYPE)), mlp_in),
        "silu": measure_task(One("silu"), torch.randn(B,S,INTER,dtype=DTYPE)),
        "mul":  measure_task(One("mul", torch.randn(B,S,INTER,dtype=DTYPE)), torch.randn(B,S,INTER,dtype=DTYPE)),
        "down": measure_task(One("mm", torch.randn(INTER,D,dtype=DTYPE)), torch.randn(B,S,INTER,dtype=DTYPE)),
        "res":  measure_task(One("add", torch.randn(B,S,D,dtype=DTYPE)), torch.randn(B,S,D,dtype=DTYPE)),
        "qk":   measure_task(One("mm", torch.randn(B,NH,HD,S,dtype=DTYPE)), attn_in),
        "sm":   measure_task(One("sm"), torch.randn(B,NH,S,S,dtype=DTYPE)),
        "av":   measure_task(One("mm", torch.randn(B,NH,S,HD,dtype=DTYPE)), torch.randn(B,NH,S,S,dtype=DTYPE)),
        "oproj":measure_task(One("oproj", torch.randn(NH*HD,D,dtype=DTYPE)), torch.randn(B,NH,S,HD,dtype=DTYPE)),
    }
    mlp_seq = ["gate","silu","mul","down","res"]
    attn_seq = ["qk","sm","av","oproj","res"]
    def isosum(seq, depth): return sum(g[seq[i]] for i in range(depth))

    print("\n depth | MLP fused / isosum (gap) | ATTN fused / isosum (gap)", flush=True)
    for depth in (2,3,4,5):
        mi, ai = isosum(mlp_seq,depth), isosum(attn_seq,depth)
        print(f"   {depth}   | {mlp_fused[depth]:7.1f} / {mi:7.1f} ({mlp_fused[depth]-mi:+7.1f}) "
              f"| {attn_fused[depth]:7.1f} / {ai:7.1f} ({attn_fused[depth]-ai:+7.1f})", flush=True)

    # ---- layer 재구성 (최대 fusion: 두 체인은 full-depth fused) ----
    rms = measure_task(One("add", torch.randn(B,S,D,dtype=DTYPE)), torch.randn(B,S,D,dtype=DTYPE))  # placeholder
    # 실제 rmsnorm/q/k/v/up 단일값은 이전 스크립트에서: rmsnorm~11-36, q 130, k 49, v 77, up 330
    singles = {"rmsnorm":11.2, "q":130.6, "k":49.2, "v":77.5, "up":330.5, "input_rms":35.7}
    # per-layer (fused chains): input_rms + q+k+v + attn_fused(5) + post_rms + up + mlp_fused(5)
    per_layer_fused = (singles["input_rms"] + singles["q"] + singles["k"] + singles["v"]
                       + attn_fused[5] + singles["rmsnorm"] + singles["up"] + mlp_fused[5])
    # per-layer (single 합, 이전 결과)
    per_layer_single = 1569.0
    total_fused = per_layer_fused * L + 6.7 + 1765.4   # +final_norm +lm_head
    total_single = per_layer_single * L + 6.7 + 1765.4

    print(f"\n===== 재구성 (×{L} + lm_head) =====", flush=True)
    print(f"single 합   per-layer {per_layer_single:.0f}us → {total_single/1000:.2f} ms", flush=True)
    print(f"chain fused per-layer {per_layer_fused:.0f}us → {total_fused/1000:.2f} ms", flush=True)
    print(f"ground truth = 20.10 ms", flush=True)
    print(f"  single 합   : {total_single/1000/20.1:.2f}x", flush=True)
    print(f"  chain fused : {total_fused/1000/20.1:.2f}x", flush=True)


if __name__ == "__main__":
    main()
