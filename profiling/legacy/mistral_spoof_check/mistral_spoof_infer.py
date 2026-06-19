#!/usr/bin/env python3
"""위장 컴파일된 Mistral artifact 로 실제 텍스트 추론 검증.

이미 빌드된 ~/compiled_models_lens_profiling/Mistral-7B-spoof-llama-check 를
재컴파일 없이 로드. Mistral instruct chat template 을 적용한 진짜 질문들을
넣어 응답이 의미 있게 나오는지 확인한다.
"""
import os

COMPILED = os.path.expanduser("~/compiled_models_lens_profiling/Mistral-7B-spoof-llama-check")
SPOOF_DIR = os.path.expanduser("~/mistral7b_as_llama")
MAX_LEN = 256

from furiosa_llm import LLM, SamplingParams  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from transformers.tokenization_utils_base import BatchEncoding  # noqa: E402

print("[load] artifact (재컴파일 없음)", flush=True)
llm = LLM(COMPILED, devices="npu:0:*", max_model_len=MAX_LEN)

tok = AutoTokenizer.from_pretrained(SPOOF_DIR)
if tok.pad_token is None:
    tok.pad_token_id = tok.eos_token_id

# 다양한 실제 질문 (지식/추론/요약/번역/코딩)
chats = [
    [{"role": "user", "content": "Compare the Size of ther Earth and the Moon."}],
    [{"role": "user", "content": "How many people live in Seoul, South Korea?"}],
    [{"role": "user", "content": "Summarize the symptoms of COVID-19."}],
    [{"role": "user", "content": "Arrange the colors of the rainbow in order."}],
    [{"role": "user", "content": "Write a python fuction to calculate the factorial of a number."}],
]

sp = SamplingParams(temperature=0.0, max_tokens=120, ignore_eos=False)

for i, msgs in enumerate(chats):
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok.encode(text, add_special_tokens=False)  # 템플릿이 이미 <s> 포함
    ids = [int(x) for x in ids]
    # max_model_len 보호: 생성분 남기기
    if len(ids) > MAX_LEN - sp.max_tokens:
        ids = ids[: MAX_LEN - sp.max_tokens]
    enc = BatchEncoding({"input_ids": [list(ids)], "attention_mask": [[1] * len(ids)]})
    out = llm.generate(prompts=[""], sampling_params=sp, prompt_token_ids=enc)
    comp = out[0].outputs[0]
    txt = getattr(comp, "text", None) or tok.decode(comp.token_ids, skip_special_tokens=True)
    print(f"\n{'='*70}", flush=True)
    print(f"[Q{i+1}] {msgs[0]['content']}", flush=True)
    print(f"[A{i+1}] {txt.strip()}", flush=True)

print(f"\n{'='*70}\n[DONE] 실제 텍스트 추론 검증 완료", flush=True)
