#!/usr/bin/env python3
"""이질적(heterogeneous) batch 거동 확인 — bs=32 1B artifact.

run_profiling.py 는 batch 안에 동일 il 을 복제(uniform)하지만, 실제 eval batch 는
sample 마다 il 이 다르다. furiosa-llm 이 batch 내 서로 다른 il(=서로 다른 prefill
bucket)을 어떻게 처리하는지 확인한다.

batch A: 32개 전부 같은 bucket(512) 이지만 il 제각각 (300~500)
batch B: 최대 bucket 은 8192(il=8000) 1개 + 나머지 31개는 bucket128(il=100)
         → 한 batch 안에 여러 bucket 혼재

각 batch 를 ol=1(prefill 거동) 로 돌려 정상 실행/시간 확인.
"""
import os, sys, time

RNGD_DIR = "/home/furiosa/바탕화면/pjh/npu_chip_project/LENS/inference_profiling/RNGD"
LENS_ROOT = "/home/furiosa/바탕화면/pjh/npu_chip_project/LENS"
sys.path.insert(0, RNGD_DIR)
sys.path.insert(0, LENS_ROOT)
import run_profiling as rp  # noqa: E402
from config.buckets import align_to_bucket  # noqa: E402

from furiosa_llm import LLM  # noqa: E402
from furiosa_llm.metadata.config_types import SchedulerConfig  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from transformers.tokenization_utils_base import BatchEncoding  # noqa: E402

ARTIFACT = os.path.expanduser("~/compiled_models_lens_profiling/Llama-3.2-1B-tp8-bs32")
MAX_LEN = 8192


def make_hetero_batch(tok, il_list):
    """il_list 의 각 길이로 dummy prompt 1개씩 — ragged(서로 다른 길이) batch."""
    text = "The quick brown fox jumps over the lazy dog. "
    input_ids, attn = [], []
    for il in il_list:
        ids = tok.encode(text * (il // 8 + 1), add_special_tokens=True)[:il]
        while len(ids) < il:
            ids.append(tok.pad_token_id)
        input_ids.append(list(ids))
        attn.append([1] * il)
    return BatchEncoding({"input_ids": input_ids, "attention_mask": attn})


def run_batch(llm, tok, il_list, ol, label):
    enc = make_hetero_batch(tok, il_list)
    sp = rp._make_sampling_params(ol)
    buckets = sorted(set(align_to_bucket(il) for il in il_list))
    t0 = time.perf_counter()
    outs = llm.generate(prompts=[""] * len(il_list), sampling_params=sp,
                        prompt_token_ids=enc)
    dt = (time.perf_counter() - t0) * 1000
    n_ok = sum(1 for o in outs if getattr(o, "outputs", None))
    print(f"\n[{label}]", flush=True)
    print(f"  il 분포: min={min(il_list)} max={max(il_list)} "
          f"고유개수={len(set(il_list))}", flush=True)
    print(f"  포함 bucket: {buckets}", flush=True)
    print(f"  batch_size={len(il_list)} ol={ol}", flush=True)
    print(f"  -> 실행 {'성공' if n_ok==len(il_list) else '일부실패'} ({n_ok}/{len(il_list)}), "
          f"batch E2E={dt:.1f}ms", flush=True)


print("[load] bs=32 artifact (max_io_memory_mb=8192, max_processing_samples=32)", flush=True)
llm = LLM(ARTIFACT, devices="npu:0:*", max_model_len=MAX_LEN,
          max_io_memory_mb=8192,
          scheduler_config=SchedulerConfig(max_processing_samples=32))
tok = AutoTokenizer.from_pretrained(ARTIFACT)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token_id = tok.eos_token_id

# batch A: 같은 bucket(512), il 제각각
ilA = [300, 350, 400, 450, 500] * 6 + [320, 480]   # 32개, 전부 (256,512] → bucket 512
run_batch(llm, tok, ilA[:32], ol=1, label="A: 동일 bucket(512), il 제각각")

# batch B: max bucket 8192(il=8000) 1개 + 나머지 bucket128(il=100)
ilB = [8000] + [100] * 31                            # 32개, bucket {128, 8192} 혼재
run_batch(llm, tok, ilB, ol=1, label="B: max bucket 8192 1개 + bucket128 31개 (혼재)")

print("\n[DONE]", flush=True)
