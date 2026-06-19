#!/usr/bin/env python3
"""TacticKernel 스파이크: 커널 레벨 작성이 RNGD 에서 실제 되는지 검증.

3단계로 "kernel-level 연구 가능 여부"를 가린다:
  (1) 커스텀 DSL 커널(elementwise add) 작성 → CPU 실행 정확성
  (2) 같은 커널 → CompileModule.from_module 로 RNGD 컴파일 → 디바이스 실행 정확성
  (3) Furiosa 가 buffer_type: Sram 으로 작성한 YAML(moe) 로드 → SRAM-명세 포맷이 유효/컴파일되는지

prng.py / dfg.py 의 검증된 패턴 그대로 사용.
"""
import os
import torch
import furiosa.torch  # noqa
from furiosa.torch.custom_ops import CompileModule, TacticKernelModule

MOE_YAML = ("/home/furiosa/바탕화면/pjh/npu_chip_project/.venv/lib/python3.10/"
            "site-packages/furiosa/models/core/operators/tk_graphs/"
            "moe_blockwise_compute_wg_idx.yaml")

MY_ADD = """#tactic_kernel_dsl
def my_add(
    a: [B,G]/i32,
    b: [B,G]/i32,
) -> [B,G]/i32:
    out: [B,G]/i32 = tk.Interleaving(
        %0 = read(a)
        %1 = read(b)
        %2 = ve.exec(%0 +_fxp %1)
    )
    return out
"""


def step1_cpu():
    print("=== (1) 커스텀 DSL 커널 CPU 실행 ===")
    m = TacticKernelModule(MY_ADD)
    a = torch.arange(12, dtype=torch.int32).reshape(3, 4)
    b = torch.ones(3, 4, dtype=torch.int32) * 100
    out = m(a, b)
    out = out[0] if isinstance(out, (list, tuple)) else out
    ok = torch.equal(out, a + b)
    print(f"  required_symbolic_params={m.required_symbolic_params}")
    print(f"  out=\n{out}")
    print(f"  CPU 정확성: {'PASS' if ok else 'FAIL'}")
    return m, (a, b)


def step2_rngd(m, cpu_inputs):
    print("\n=== (2) 같은 커널 RNGD 컴파일 + 디바이스 실행 ===")
    dev = torch.device("rngd", 0)
    a, b = cpu_inputs
    try:
        mc = CompileModule.from_module(m, (a, b))
        print("  CompileModule.from_module: OK (RNGD 컴파일 성공)")
        ar, br = a.to(dev), b.to(dev)
        out = mc(ar, br)
        out = out[0] if isinstance(out, (list, tuple)) else out
        out_cpu = out.cpu()
        ok = torch.equal(out_cpu, a + b)
        print(f"  device out=\n{out_cpu}")
        print(f"  RNGD 정확성: {'PASS' if ok else 'FAIL'}")
        return ok
    except Exception as e:
        print(f"  RNGD 컴파일/실행 FAIL: {type(e).__name__}: {str(e)[:300]}")
        return False


def step3_sram_yaml():
    print("\n=== (3) buffer_type: Sram 명세(moe YAML) 로드/컴파일 ===")
    spec = open(MOE_YAML).read()
    n_sram = spec.count("buffer_type: Sram")
    print(f"  YAML 내 'buffer_type: Sram' 텐서 수: {n_sram}")
    try:
        m = TacticKernelModule(spec)
        print(f"  TacticKernelModule(yaml) 파싱: OK  (symbolic={m.required_symbolic_params})")
    except Exception as e:
        print(f"  파싱 FAIL: {type(e).__name__}: {str(e)[:300]}")
        return False
    # CPU 실행 시도 (입력: tensor 0 shape [E] int32, symbolic G)
    try:
        E = torch.arange(8, dtype=torch.int32)  # E=8
        out = m(E, G=4)
        out = out[0] if isinstance(out, (list, tuple)) else out
        print(f"  CPU 실행 OK, out.shape={getattr(out,'shape',None)}")
    except Exception as e:
        print(f"  CPU 실행은 입력 의존적이라 skip/실패 가능: {type(e).__name__}: {str(e)[:150]}")
    # RNGD 컴파일 시도 (SRAM 명세가 컴파일러를 통과하는지)
    try:
        E = torch.arange(8, dtype=torch.int32)
        mc = CompileModule.from_module(m, (E,), {"G": 4})
        print("  SRAM-명세 YAML RNGD 컴파일: OK  ← 커널 레벨 SRAM 배치 컴파일 가능")
        return True
    except Exception as e:
        print(f"  RNGD 컴파일: {type(e).__name__}: {str(e)[:250]}")
        return False


if __name__ == "__main__":
    m, ci = step1_cpu()
    ok2 = step2_rngd(m, ci)
    ok3 = step3_sram_yaml()
    print("\n=== 스파이크 결론 ===")
    print(f"  커스텀 커널 RNGD 실행: {'가능' if ok2 else '불가/미확인'}")
    print(f"  SRAM-명세 커널 컴파일: {'가능' if ok3 else '불가/미확인'}")
