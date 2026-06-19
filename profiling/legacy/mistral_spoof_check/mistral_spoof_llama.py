#!/usr/bin/env python3
"""Mistral-7B-v0.3 을 model_type='llama' 로 위장해 RNGD 컴파일 시도.

Mistral 구조는 Llama 와 거의 동일(h=4096,i=14336,32L,GQA8,rope_theta=1e6,
sliding_window=null). 차이: vocab 32768, llama3 rope_scaling 없음.
SDK 화이트리스트는 config 의 model_type 문자열로만 막으므로, 가중치를 받아
config.json 의 model_type 을 'llama' 로 바꾼 로컬 디렉토리로 ArtifactBuilder 를
돌리면 통과하는지(=구조적으로 컴파일되는지) 확인한다.
"""
import json
import os
import shutil
import time

from huggingface_hub import snapshot_download

SRC_ID = "mistralai/Mistral-7B-Instruct-v0.3"
SPOOF_DIR = os.path.expanduser("~/mistral7b_as_llama")
COMPILED = os.path.expanduser("~/compiled_models_lens_profiling/Mistral-7B-spoof-llama-check")
TP = 8
MAX_LEN = 256
BUCKETS = [128, 256]

print(f"[download] {SRC_ID} (weights+tokenizer) ...", flush=True)
t0 = time.monotonic()
snap = snapshot_download(
    SRC_ID,
    allow_patterns=["model-*.safetensors", "model.safetensors.index.json",
                    "config.json", "generation_config.json",
                    "*.model", "tokenizer*", "special_tokens_map.json"],
    ignore_patterns=["consolidated.safetensors"],
)
print(f"  -> {snap}  ({time.monotonic()-t0:.1f}s)", flush=True)

# ── spoof dir 구성: 원본 파일 symlink + config.json 패치 ──
if os.path.exists(SPOOF_DIR):
    shutil.rmtree(SPOOF_DIR)
os.makedirs(SPOOF_DIR)
for fn in os.listdir(snap):
    src = os.path.join(snap, fn)
    if fn == "config.json":
        continue
    os.symlink(os.path.realpath(src), os.path.join(SPOOF_DIR, fn))

cfg = json.load(open(os.path.join(snap, "config.json")))
print(f"[patch] original model_type={cfg.get('model_type')} "
      f"architectures={cfg.get('architectures')}", flush=True)
cfg["model_type"] = "llama"
cfg["architectures"] = ["LlamaForCausalLM"]
cfg.setdefault("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
with open(os.path.join(SPOOF_DIR, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)
print(f"  -> patched config written to {SPOOF_DIR}", flush=True)

# ── 컴파일 ──
from furiosa_llm.artifact import (  # noqa: E402
    ArtifactBuilder, BucketConfig, ModelConfig, ParallelConfig,
)
from furiosa_llm import LLM, SamplingParams  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from transformers.tokenization_utils_base import BatchEncoding  # noqa: E402

prefill_buckets = [(1, b) for b in BUCKETS]
decode_buckets = [(1, b) for b in BUCKETS]
tw = sorted({1} | set(BUCKETS))
print(f"\n[build] spoofed-as-llama  tp={TP} buckets={BUCKETS}", flush=True)
t0 = time.monotonic()
builder = ArtifactBuilder(
    SPOOF_DIR,
    model_config=ModelConfig(max_model_len=MAX_LEN),
    parallel_config=ParallelConfig(tensor_parallel_size=TP, pipeline_parallel_size=1),
    bucket_config=BucketConfig(
        prefill_buckets=prefill_buckets, decode_buckets=decode_buckets,
        tokenwise_seq_lens=tw, skip_validation=True,
    ),
)
builder.build(COMPILED, num_pipeline_builder_workers=2, num_compile_workers=2)
print(f"[build DONE] {time.monotonic()-t0:.1f}s", flush=True)

print("\n[load]", flush=True)
llm = LLM(COMPILED, devices="npu:0:*", max_model_len=MAX_LEN)

tok = AutoTokenizer.from_pretrained(SPOOF_DIR)
if tok.pad_token is None:
    tok.pad_token_id = tok.eos_token_id

print("\n[sanity] 위장 모델이 정상 텍스트를 내는가 (정확도 확인)", flush=True)
sp = SamplingParams(temperature=0.0, max_tokens=20, ignore_eos=False)
for prompt in ["The capital of France is", "Water is made of hydrogen and"]:
    ids = tok.encode(prompt, add_special_tokens=True)
    enc = BatchEncoding({"input_ids": [list(ids)], "attention_mask": [[1] * len(ids)]})
    out = llm.generate(prompts=[""], sampling_params=sp, prompt_token_ids=enc)
    comp = out[0].outputs[0]
    txt = getattr(comp, "text", None) or tok.decode(comp.token_ids)
    print(f"  Q: {prompt!r}\n  A: {txt!r}", flush=True)

print("\n[ALL DONE] spoof 컴파일+추론 완료 — 출력이 말이 되는지 위에서 확인", flush=True)
