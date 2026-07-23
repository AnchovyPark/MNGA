#!/usr/bin/env python3
"""config vs context 분리 실험 (컴파일러 2026.3.0 전용).

배경: 격리 커널 latency와 in-model latency의 격차(1B/8B에서 최대 12x)는
  (a) config 성분 = 격리가 저품질로 컴파일됨  +  (b) context 성분 = cross-op fusion/overlap 상실
이 섞인 것. 2026.2.0에선 격리를 저품질로만 컴파일 가능해 둘을 분리 못 했다.
2026.3.0은 World-1 compile()이 production knob(ForLlmModelComputeBound, scheduler_beam_search,
use_attention_kernel 등 23/100)을 받고 결과를 TpModule로 실행까지 한다 → 처음으로 분리 가능.

이 스크립트: o_proj 단일 matmul을 (1) 저품질(NoConstraint) (2) near-production(ForLlmModelComputeBound
+ production 값) 두 config로 컴파일→실행, latency 비교. 두 숫자의 비율 = config 성분.
near-production 격리 vs 알려진 in-model 실측 잔차 = context(+77-knob 잔여 config) 성분.

⚠️ 첫 실행 프로브다. 2026.3.0 실제 API 동작(Runnable/TpModule.run 인자, weight 처리, 타이밍)을
   하드웨어에서 확인하며 반복 수정할 것. 방어적으로 짜서 실패 지점을 진단 출력한다.

사용: <2026.3.0 venv>/bin/python config_vs_context_30.py [seq] [model]
"""
import sys, time, statistics as st

import torch
import furiosa.torch  # noqa
from furiosa.torch import native_device as nd

S = int(sys.argv[1]) if len(sys.argv) > 1 else 512
MODEL = sys.argv[2] if len(sys.argv) > 2 else "1b"
D, INTER, NH, HD, KV = {"1b": (2048, 8192, 32, 64, 8),
                        "8b": (4096, 14336, 32, 128, 8)}[MODEL]
DT = torch.bfloat16
ITERS, WARMUP = 50, 10


def p(*a):
    print(*a, flush=True)


def version_guard():
    import furiosa.torch.compiler as tc
    knobs = [x for x in dir(tc.Config) if not x.startswith("_")]
    hints = [x for x in dir(tc.TacticHintConfig) if not x.startswith("_")]
    p(f"[env] World-1 Config knob {len(knobs)}개, tactic_hint={hints}")
    ok = "scheduler_beam_search" in knobs and "ForLlmModelComputeBound" in hints
    if not ok:
        p("!! 이 venv는 2026.3.0 아님 (scheduler_beam_search / ForLlmModelComputeBound 없음). 중단.")
        sys.exit(2)
    return tc


def prod_values(seq):
    """production dict에서 노출된 knob의 실제 값을 뽑아 near-production config에 반영."""
    import yaml
    from furiosa.native_common.compiler import create_llm_compiler_config_with_layer_range, approx_per_layer_params_b
    from furiosa_llm.parallelize.layer_range import LayerRange, TransformerBlock
    lr = LayerRange(start=TransformerBlock(idx=0), end=TransformerBlock(idx=0))
    ap = approx_per_layer_params_b(D, INTER)
    d = yaml.safe_load(create_llm_compiler_config_with_layer_range(
        "llama", "generate", ap, 1, 8, 1, seq, seq, lr, True, False, False, False))
    return d


class OProj(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w", torch.randn(NH * HD, D, dtype=DT))

    def forward(self, x):
        return (x @ self.w,)


def make_config(tc, hint_name, pd):
    """World-1 CompilerConfig 구성. hint_name= 'NoConstraint'(저품질) or 'ForLlmModelComputeBound'."""
    H = tc.TacticHintConfig
    kw = dict(tactic_hint=getattr(H, hint_name))
    # near-production일 때만 production 값 반영 (노출된 핵심 knob)
    if hint_name != "NoConstraint":
        for k in ("scheduler_beam_search", "expected_total_beam_states", "use_attention_kernel",
                  "weight_sharding_threshold_in_bytes", "tensor_unit_bridge_threshold_in_page",
                  "local_population_threshold", "sparsify_moe"):
            if k in pd:
                try:
                    kw[k] = pd[k]
                except Exception:
                    pass
    try:
        return tc.Config(**kw)
    except Exception as e:
        p(f"  [config] full kw 실패({e}); tactic_hint만으로 재시도")
        return tc.Config(tactic_hint=getattr(H, hint_name))


def compile_and_time(tc, mod, x, cfg, tag):
    ep = torch.export.export(mod, (x,))
    p(f"  [{tag}] export ok, compile 시작...")
    runnable = tc.compile(ep, cfg)
    p(f"  [{tag}] compile ok -> {type(runnable).__name__}")
    from furiosa.torch._C.module import TpModule
    tp = TpModule.from_runnable(runnable, [x])
    # warmup
    for _ in range(WARMUP):
        tp.run()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        tp.run()
        ts.append((time.perf_counter() - t0) * 1e6)  # us
    return st.median(ts), min(ts)


def main():
    nd.set_fusion(8)
    tc = version_guard()
    pd = prod_values(S)
    x = torch.randn(1, S, NH * HD, dtype=DT)
    mod = OProj()
    p(f"\n=== config-vs-context: o_proj (1,{S},{NH*HD})@({NH*HD},{D}), {MODEL} ===")
    res = {}
    for hint in ("NoConstraint", "ForLlmModelComputeBound"):
        try:
            med, mn = compile_and_time(tc, mod, x, make_config(tc, hint, pd), hint)
            res[hint] = med
            p(f"  [{hint}] latency median={med:.1f}us  min={mn:.1f}us")
        except Exception as e:
            import traceback
            p(f"  [{hint}] 실패: {e}")
            traceback.print_exc()
    if len(res) == 2:
        lo, hi = res["NoConstraint"], res["ForLlmModelComputeBound"]
        p(f"\n[config 성분] 저품질/고품질 = {lo/hi:.2f}x (고품질이 이만큼 빠름)")
        p(f"[다음] 이 고품질 격리값({hi:.1f}us)을 in-model o_proj 실측과 비교 → 잔차 = context 성분")


if __name__ == "__main__":
    main()
