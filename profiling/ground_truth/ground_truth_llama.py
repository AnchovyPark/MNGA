#!/usr/bin/env python3
"""Ground truth: 실제 Llama-3.1-8B (furiosa-llm TP=8 아티팩트) end-to-end latency.

커널 합성 예측(profiling/*)의 검증 대상. 실제 모델을 device에서 돌린 진짜 latency.

아티팩트: furiosa-ai/Llama-3.1-8B-Instruct = tensor_parallel_size=8, PAGED_ATTENTION, bf16.
device 실행 캐시(~/.cache/furiosa/llm) 삭제 상태 → 첫 로드에 재컴파일(오래 걸림).

측정: 두 번의 blocking generate 로 prefill / decode 분리
  t1  = generate(max_tokens=1)   ≈ prefill + 1 decode
  t65 = generate(max_tokens=65)  ≈ prefill + 65 decode
  decode_per_token = (t65 - t1) / 64
  prefill ≈ t1 - decode_per_token

사용법: python ground_truth_llama.py
"""
import sys
import time

import torch  # noqa: F401
import furiosa.torch  # noqa: F401
from furiosa_llm import LLM, SamplingParams

MODEL = "furiosa-ai/Llama-3.1-8B-Instruct"


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t0


def main():
    print(f"[load] {MODEL} 로드/컴파일 시작 (device 재컴파일 가능)...", flush=True)
    llm, t_load = timed(LLM, MODEL)
    print(f"[load] 완료: {t_load:.1f}s", flush=True)

    tok = llm.tokenizer
    base = ("Explain the theory of relativity in detail, covering both special "
            "and general relativity, their historical development, key equations, "
            "experimental confirmations, and modern applications. ") * 60
    base_ids = tok.encode(base)

    def prompt_of_len(n):
        """정확히 ~n 토큰짜리 프롬프트 문자열."""
        ids = base_ids[:n]
        return tok.decode(ids, skip_special_tokens=True)

    RUNS = 3
    def bench(prompt, mt):
        ts = []
        for _ in range(RUNS):
            _, t = timed(llm.generate, prompt, SamplingParams(max_tokens=mt, temperature=0.0))
            ts.append(t)
        ts.sort()
        return ts[len(ts) // 2]

    # warmup
    print("[warmup]...", flush=True)
    bench(prompt_of_len(128), 4)

    # decode/token: 짧은 프롬프트에서 t1 vs t65
    short = prompt_of_len(64)
    n_short = len(tok.encode(short))
    t1s = bench(short, 1)
    t65s = bench(short, 65)
    decode_tok = (t65s - t1s) / 64
    print(f"[decode] prompt~{n_short}tok  t1={t1s*1000:.1f}ms t65={t65s*1000:.1f}ms "
          f"→ decode/token={decode_tok*1000:.2f}ms", flush=True)

    # prefill sweep: 통제된 입력 길이.
    # max_tokens=1 은 prefill forward 그 자체(첫 토큰을 prefill 이 생성) → t1 = prefill.
    # (별도 decode 스텝 없음. decode/token 은 t65-t1 으로 따로 구함.)
    print("\n===== GROUND TRUTH (Llama-3.1-8B, TP=8) =====", flush=True)
    print(f"decode / token ≈ {decode_tok*1000:.2f} ms  (weight-load bound)", flush=True)
    print("prefill (통제 입력길이, = generate max_tokens=1):", flush=True)
    for S in (128, 512, 2048):
        p = prompt_of_len(S)
        n = len(tok.encode(p))
        prefill = bench(p, 1)
        print(f"  S={S:5d} (실제 {n:5d}tok): prefill≈{prefill*1000:7.1f}ms", flush=True)


if __name__ == "__main__":
    main()
