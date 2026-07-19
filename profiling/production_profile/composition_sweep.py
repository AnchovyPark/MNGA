#!/usr/bin/env python3
"""Composition sweep: Tokenwise op 뭉치를 하나씩 누적하며 컴파일러의 예상 cost(io/compute/overlap)를 읽어,
op를 추가할 때 얼마나 흡수/겹침되는지 = 컴파일러가 op끼리 어떻게 최적화하는지를 분해한다.

- 신호원 = 컴파일러 자기 cost-model 출력 (dump_lir=True → {tag}.summary/summary.json). 실행 불필요.
- ⚠️ standalone 기준이라 실제 supertask 절대값과 ~2-2.8x 차이(standalone≠in-model fusion).
  절대예측용 아님. RELATIVE 구조(흡수 패턴/overlap 성장)가 유효 신호.

사용: python composition_sweep.py [seq]   (기본 128)
"""
import os, sys, json, csv, yaml
import torch, furiosa.torch  # noqa
from furiosa.torch import native_device as nd
nd.set_fusion(8)
from furiosa.native_common.compiler import compile as native_compile, create_llm_compiler_config_with_layer_range
from furiosa.native_common.compiler import approx_per_layer_params_b
from furiosa_llm.parallelize.layer_range import LayerRange, TransformerBlock

S = int(sys.argv[1]) if len(sys.argv) > 1 else 128
MODEL = sys.argv[2] if len(sys.argv) > 2 else "1b"
# model -> (D, INTER, NH, HD, KV)
D, INTER, NH, HD, KV = {"1b": (2048, 8192, 32, 64, 8),
                        "8b": (4096, 14336, 32, 128, 8)}[MODEL]
APPROX = approx_per_layer_params_b(D, INTER)
DT = torch.bfloat16
OUTDIR = os.environ.get("SD", "/tmp/csweep")
OPS = ["q", "k", "v", "o", "gate", "up", "down"]
# op -> (input_shape, weight_shape)
SPEC = {
    "q":    ((1, S, D),      (D, NH * HD)),
    "k":    ((1, S, D),      (D, KV * HD)),
    "v":    ((1, S, D),      (D, KV * HD)),
    "o":    ((1, S, NH * HD),(NH * HD, D)),
    "gate": ((1, S, D),      (D, INTER)),
    "up":   ((1, S, D),      (D, INTER)),
    "down": ((1, S, INTER),  (INTER, D)),
}


def p(*a): print(*a, flush=True)


def _cfg():
    lr = LayerRange(start=TransformerBlock(idx=0), end=TransformerBlock(idx=0))
    return yaml.safe_load(create_llm_compiler_config_with_layer_range(
        "llama", "generate", APPROX, 1, 8, 1, S, S, lr, True, False, False, False))


def compile_read(mod, inp, tag):
    d = f"{OUTDIR}/cs{S}_{tag}"; os.makedirs(d, exist_ok=True)
    native_compile(mod, (inp,), target_npu="renegade", config=_cfg(),
                   target_ir="edf", dump_lir=True, dump_path=d, dump_tag="c")
    st = json.load(open(f"{d}/c.summary/summary.json"))["lir_stats"]
    return dict(total=st["total_cycle"], io=st["io_cycle"], comp=st["computation_cycle"])


class One(torch.nn.Module):
    def __init__(self, wsh):
        super().__init__(); self.register_buffer("w", torch.randn(*wsh, dtype=DT))
    def forward(self, x): return (x @ self.w,)


class TWPrefix(torch.nn.Module):
    """첫 N개 Tokenwise op만 계산해서 반환 (q,k,v는 attention으로 나가는 실제 output)."""
    def __init__(self, N):
        super().__init__(); self.N = N
        for n, sh in [("wq",(D,NH*HD)),("wk",(D,KV*HD)),("wv",(D,KV*HD)),("wo",(NH*HD,D)),
                      ("wg",(D,INTER)),("wu",(D,INTER)),("wd",(INTER,D))]:
            self.register_buffer(n, torch.randn(*sh, dtype=DT))
    def forward(self, x):
        N = self.N; outs = [x @ self.wq]
        if N >= 2: outs.append(x @ self.wk)
        if N >= 3: outs.append(x @ self.wv)
        if N >= 4: o = outs[0] @ self.wo; outs.append(o)
        if N >= 5: x2 = x + o; g = x2 @ self.wg; outs.append(g)
        if N >= 6: outs.append(x2 @ self.wu)
        if N >= 7: outs.append((torch.nn.functional.silu(g) * outs[5]) @ self.wd)
        return tuple(outs)


def main():
    p(f"=== composition sweep (seq={S}, prod config, 컴파일러 예상 cycle) ===")
    # 1) 누적
    cum = {}
    p(f"{'bundle':>20} {'total':>9} {'io':>9} {'comp':>9} {'overlap':>9} {'Δtotal':>9}")
    prev = 0
    for N in range(1, 8):
        r = compile_read(TWPrefix(N), torch.randn(1, S, D, dtype=DT), f"n{N}")
        cum[N] = r
        ov = r["io"] + r["comp"] - r["total"]; dt = r["total"] - prev; prev = r["total"]
        p(f"{'+'.join(OPS[:N]):>20} {r['total']:9d} {r['io']:9d} {r['comp']:9d} {ov:9d} {dt:9d}")
    # 2) standalone
    single = {}
    for op in OPS:
        insh, wsh = SPEC[op]
        single[op] = compile_read(One(wsh), torch.randn(*insh, dtype=DT), f"s_{op}")["total"]
    # 3) 흡수 분석
    p(f"\n{'op추가':>8} {'marginal':>9} {'standalone':>11} {'흡수':>9} {'흡수율':>7}")
    rows = []; prev = 0
    for N in range(1, 8):
        op = OPS[N - 1]; marg = cum[N]["total"] - prev; prev = cum[N]["total"]
        ab = single[op] - marg; rate = ab / single[op] * 100 if single[op] else 0
        p(f"{op:>8} {marg:9d} {single[op]:11d} {ab:9d} {rate:6.0f}%")
        rows.append(dict(seq=S, op=op, marginal=marg, standalone=single[op], absorbed=ab,
                         absorb_pct=round(rate, 1), cum_total=cum[N]["total"],
                         cum_io=cum[N]["io"], cum_comp=cum[N]["comp"]))
    out = f"profiling/production_profile/composition_sweep_s{S}_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    p(f"\n저장: {out}")


if __name__ == "__main__":
    main()
