#!/usr/bin/env python3
"""실제 LLM 모델을 재빌드하며 각 supertask 컴파일의 cost-estimate(summary.json)를 dump.

FURIOSA_COMPILE_DUMP_PATH env로 dump 위치 지정 → 각 supertask가 {tag}.summary/summary.json 생성
(tag = hash+{batch}_{input}_{attention}). 컴파일 캐시를 우회해야(fresh cache_dir) 재컴파일+dump 됨.
큰 intermediate 파일은 별도 cleaner로 삭제 권장(summary.json만 2KB).

사용: FURIOSA_COMPILE_DUMP_PATH=<dir> python build_with_dump.py <model> <save_dir> <cache_dir> [workers]
"""
import sys
import os

from furiosa_llm.artifact.builder import ArtifactBuilder
try:
    from furiosa_llm.artifact.types.config import CompilerConfig, BucketConfig
except Exception:
    from furiosa_llm.artifact.types import CompilerConfig, BucketConfig

model = sys.argv[1]
save_dir = sys.argv[2]
cache_dir = sys.argv[3]
workers = int(sys.argv[4]) if len(sys.argv) > 4 else 1

# 명시 bucket: prefill seq 128/256/512/1024 + decode 1개(생성모델 필수).
# preset 없는 모델(Qwen 등)도 빌드되고, 유닛 수가 적어 빌드 훨씬 빠름.
BUCKETS = BucketConfig(
    prefill_buckets=[(1, 128), (1, 256), (1, 512), (1, 1024)],
    decode_buckets=[(1, 2048)],
    tokenwise_seq_lens=[128, 256, 512, 1024],
    skip_validation=True,   # prefill+decode+tokenwise만 주고 "4필드 다 필요" 규칙 우회
)

print(f"[build] {model} -> {save_dir}", flush=True)
print(f"[build] cache_dir={cache_dir}  dump={os.environ.get('FURIOSA_COMPILE_DUMP_PATH')}", flush=True)
print(f"[build] buckets: prefill {BUCKETS.prefill_buckets}  decode {BUCKETS.decode_buckets}", flush=True)
b = ArtifactBuilder(
    model,
    bucket_config=BUCKETS,
    compiler_config=CompilerConfig(
        compiler_config_overrides={"enable_tuc_profile": True, "profile_sync": True}),
)
b.build(save_dir, num_compile_workers=workers, num_pipeline_builder_workers=1, cache_dir=cache_dir)
print("[build] DONE", flush=True)
