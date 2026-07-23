#!/usr/bin/env python3
"""op-identical Tokenwise 격리 재구성 (2026.3.0). in-model Tokenwise supertask와 op 완전 동일.

in-model Tokenwise = QKV proj + RoPE + O proj + MLP + RMSNorm×2 + residual (attention core는 별도 supertask C).
= 컴파일러 supertask A{RMSNorm1, q/k/v proj, RoPE} + B{O proj, residual, RMSNorm2, gate/up/down, silu, residual}.
B의 O proj 입력(attn_out)은 in-model에서 supertask C가 주므로, 여기선 dummy 입력으로 받음(격리이므로).

목표: 이 격리 번들 latency가 in-model Tokenwise 실측(128:334us,512:752us,1024:1299us @1.1GHz)과 맞나.
맞으면 = 격리가 in-model supertask를 충실히 재현 → n-gram 예측의 토대.

사용: /home/furiosa/venv3030/bin/python tokenwise_opidentical_30.py <seq> [check|time]
"""
import os, sys, time, statistics as st

import torch
import furiosa.torch as ft
ft.set_fusion(8)
from furiosa.torch._C.config.compiler import CompilerConfig, TacticHintConfig

S = int(sys.argv[1]) if len(sys.argv) > 1 else 512
MODE = sys.argv[2] if len(sys.argv) > 2 else "time"
DEV = "furiosa:0"
D, INTER, NH, HD, KV = 2048, 8192, 32, 64, 8
DT = torch.bfloat16
EPS = 1e-5
ITERS, WARMUP = 30, 8


def rmsnorm(x, w):
    v = (x.to(torch.float32) ** 2).mean(-1, keepdim=True)
    xn = (x * torch.rsqrt(v.to(DT) + EPS))
    return xn * w


def rotate_half(x):
    x1 = x[..., : HD // 2]
    x2 = x[..., HD // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rope(t, heads, cos, sin):
    # t: (S, heads*HD) -> (S, heads, HD)
    t = t.view(S, heads, HD)
    out = t * cos + rotate_half(t) * sin
    return out.reshape(S, heads * HD)


class Tokenwise(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for n, sh in [("wq",(D,NH*HD)),("wk",(D,KV*HD)),("wv",(D,KV*HD)),("wo",(NH*HD,D)),
                      ("wg",(D,INTER)),("wu",(D,INTER)),("wd",(INTER,D))]:
            self.register_buffer(n, torch.randn(*sh, dtype=DT))
        self.register_buffer("n1", torch.randn(D, dtype=DT))
        self.register_buffer("n2", torch.randn(D, dtype=DT))
        # RoPE cos/sin: (1, HD) broadcast over S,heads (값은 latency 무관, shape만)
        self.register_buffer("cos", torch.randn(1, 1, HD, dtype=DT))
        self.register_buffer("sin", torch.randn(1, 1, HD, dtype=DT))

    def forward(self, x, attn_out):
        # --- supertask A: RMSNorm1 + QKV + RoPE ---
        h = rmsnorm(x, self.n1)
        q = rope(torch.mm(h, self.wq), NH, self.cos, self.sin)
        k = rope(torch.mm(h, self.wk), KV, self.cos, self.sin)
        v = torch.mm(h, self.wv)
        # --- supertask B: O proj + residual + RMSNorm2 + MLP ---
        o = torch.mm(attn_out, self.wo)
        x2 = x + o
        h2 = rmsnorm(x2, self.n2)
        g = torch.mm(h2, self.wg)
        u = torch.mm(h2, self.wu)
        mlp = torch.mm(torch.nn.functional.silu(g) * u, self.wd)
        out = x2 + mlp
        return (q, k, v, out)


def cfg():
    return CompilerConfig(tactic_hint=TacticHintConfig.ForLlmModelComputeBound,
                          scheduler_beam_search=True, use_attention_kernel=True)


def inputs():
    return torch.randn(S, D, dtype=DT), torch.randn(S, NH * HD, dtype=DT)


def check():
    m = Tokenwise(); x, a = inputs()
    # CPU fp32 참조
    def ref():
        def rms(t, w):
            v = (t.float()**2).mean(-1, keepdim=True)
            return (t.float()*torch.rsqrt(v+EPS))*w.float()
        h = rms(x, m.n1)
        out = (rms(x + a.float()@m.wo.float(), m.n2))
        g = out@m.wg.float(); u = out@m.wu.float()
        mlp = (torch.nn.functional.silu(g)*u)@m.wd.float()
        return (x.float()+a.float()@m.wo.float()) + mlp
    r = ref()
    cm = ft.CompileModule.from_module(m, (x, a), compiler_config=cfg()).to(DEV)
    out = cm(x.to(DEV), a.to(DEV))[3].to("cpu").float()
    rel = ((out-r).abs().max()/(r.abs().max()+1e-9)).item()
    print(f"op-identical Tokenwise out 상대오차 = {rel:.4f}", flush=True)


def time_it():
    m = Tokenwise(); x, a = inputs()
    cm = ft.CompileModule.from_module(m, (x, a), compiler_config=cfg()).to(DEV)
    xd, ad = x.to(DEV), a.to(DEV)
    s = ft.current_stream(DEV)
    for _ in range(WARMUP): cm(xd, ad)
    s.synchronize()
    reps = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(ITERS): cm(xd, ad)
        s.synchronize(); reps.append((time.perf_counter()-t0)/ITERS*1e6)
    inmodel = {128: 334, 512: 752, 1024: 1299}.get(S, None)
    L = st.median(reps)
    msg = f"op-identical Tokenwise@{S}: {L:.1f}us"
    if inmodel: msg += f"  | in-model {inmodel}us  → 격리/in-model = {L/inmodel:.2f}x"
    print(msg, flush=True)


if __name__ == "__main__":
    (check if MODE == "check" else time_it)()
    sys.stdout.flush(); os._exit(0)
