# inference_profiling/RNGD/

Furiosa RNGD 용 **furiosa-llm direct profiling**.

inf2/ 와 동일한 profile spec (60 combos / 14 combos) 을 RNGD 에서 측정.
Furiosa 공식 SDK (`furiosa-llm`) 의 `ArtifactBuilder` + `LLM` API 를 직접 호출.

## 파일

| 파일 | 역할 |
|------|------|
| `build_profile.py` | 60 combos spec → `profile.csv` 생성 (inf2 와 동일) |
| `profile.csv` | 60 combos × 1 row (id, input_len, output_len) |
| `build_profile_min14.py` | 14 combos min set → `profile_min14.csv` |
| `profile_min14.csv` | 14 combos × 1 row (빠른 검증용) |
| `run_profiling.py` | **furiosa-llm direct runner (기본 경로)** |

## inf2 ↔ RNGD 매핑

| 측면 | inf2 (NxD) | RNGD (furiosa-llm) |
|------|-----------|--------------------|
| SDK | neuronx-distributed-inference | furiosa-llm |
| Config | `NeuronConfig(...)` | `ArtifactBuilder(...)` |
| Prefill bucket | `context_encoding_buckets=[128..8192]` | `prefill_buckets=[(1, b) for b in ...]` |
| Decode bucket | `token_generation_buckets=[128..8192]` | `decode_buckets=[(batch_size, b) for b in ...]` |
| Prefill batch | `ctx_batch_size=1` | `(1, *)` tuple |
| Continuous batching off | `is_continuous_batching=False` | uniform 동시 submit 으로 동치 |
| TP | `tp_degree` | `tensor_parallel_size` (Furiosa: 1/4/8 권장) |
| Compile | `model.compile(dir)` | `builder.build(save_dir, ...)` |
| Load | `model.load(dir)` | `LLM(artifact_dir, devices=...)` |
| Generate | `HFGenerationAdapter.generate(...)` | `llm.generate(prompts, SamplingParams, prompt_token_ids)` |
| Deterministic | `do_sample=False` | `SamplingParams(temperature=0)` |
| 정확히 ol 토큰 | `min_new_tokens=max_new_tokens=ol` | `min_tokens=max_tokens=ol, ignore_eos=True` |

## 핵심 ArtifactBuilder 설정 (`run_profiling.py`)

```python
ArtifactBuilder(
    model_path,
    tensor_parallel_size=tp_degree,         # Llama: 4 또는 8 권장
    pipeline_parallel_size=pp_degree,
    prefill_buckets=[(1, b) for b in buckets],
        # prefill 은 per-sample → inf2 의 ctx_batch_size=1 등가
    decode_buckets=[(batch_size, b) for b in buckets],
        # decode 는 uniform batch=batch_size 로 고정
)
builder.build(compiled_dir,
              num_pipeline_builder_workers=4,
              num_compile_workers=4)
```

## 실행 순서

1. **Artifact build** (compile + save) — 1회
   - tp/buckets 가 바뀌면 다시 build 필요
   - `--skip-compile` 로 재사용 가능
2. **LLM load** — `LLM(compiled_dir, devices=...)` 1회
3. **각 bucket 대표 shape warmup** (engine 캐시 워밍)
4. **각 combo × n_runs 측정**:
   - combo `(input_len, output_len)` 을 **batch_size 개 복제** → uniform batch
   - `SamplingParams(min_tokens=ol, max_tokens=ol, ignore_eos=True)` 로 정확히 ol 토큰 강제
   - `llm.generate(...)` wall-time 측정 (E2E only, dual-call 안 함)

## 사용

```bash
cd ~/npu_chip_project

# 60 combos × 3 runs (Llama-3.1-8B / tp=4)
python3 LENS/inference_profiling/RNGD/run_profiling.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tp-degree 4 --batch-size 32 --max-model-len 8192 --n-runs 3 \
    --devices "npu:0:*"

# 14 combos × 3 runs (빠른 검증)
python3 LENS/inference_profiling/RNGD/run_profiling.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tp-degree 4 --batch-size 32 --max-model-len 8192 --n-runs 3 \
    --profile-csv LENS/inference_profiling/RNGD/profile_min14.csv

# Build 한 번 했으면 재사용
python3 LENS/inference_profiling/RNGD/run_profiling.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tp-degree 4 --batch-size 32 --max-model-len 8192 --n-runs 3 \
    --skip-compile \
    --compiled-dir ~/compiled_models_lens_profiling/Llama-3.1-8B-Instruct
```

## 주요 CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model` | (필수) | HF model id 또는 local path |
| `--tp-degree` | 4 | tensor_parallel_size (Furiosa: 1/4/8 권장, Llama 는 4 또는 8) |
| `--pp-degree` | 1 | pipeline_parallel_size |
| `--batch-size` | 32 | decode batch (Furiosa 권장: power of 2) |
| `--max-model-len` | 8192 | 최대 context 길이 |
| `--devices` | None (auto) | `npu:0:*`, `npu:0:0-3` 등 |
| `--buckets` | (default 7) | comma-list, e.g. `128,256,512,1024,2048,4096,8192` |
| `--compiled-dir` | `~/compiled_models_lens_profiling` | artifact save / load 경로 |
| `--skip-compile` | False | 이미 build 된 artifact 재사용 |
| `--skip-warmup` | False | warmup 건너뜀 |
| `--num-compile-workers` | 4 | 컴파일 병렬도 |
| `--num-pipeline-builder-workers` | 4 | 파이프라인 빌더 병렬도 |

## 출력

`LENS/inference_results/profiling/RNGD/<model>/bs<B>/RNGD_bs<B>_<YYYYMMDD_HHMMSS>.{csv,json}`

결과 CSV 컬럼 (inf2 와 동일):
```
run_id, combo_id, combo_il, combo_ol, batch_size, status,
sample_ids, input_lens, output_lens,      # batch 내 32 sample — 전부 동일 값
max_n_generated, batch_ttft_ms, batch_e2e_ms, error
```

JSON meta 에는 `hardware="RNGD"`, `tp_degree`, `pp_degree`,
`prefill_buckets`, `decode_buckets`, `devices`, `total_sweep_s` 등 기록.

## 주의 사항

- **tp_degree 제약**: Furiosa 공식 — power of 2 (1, 2, 4, 8). Llama 는 4 또는 8 권장.
- **batch_size 제약**: decode_buckets 에 그대로 들어가므로 power of 2 가 자연스러움 (8, 16, 32 등).
- **artifact 재사용**: tp/buckets/batch_size 중 하나라도 바뀌면 `--skip-compile` 사용 금지 (artifact mismatch).
- **device spec**: `npu:0:*` 은 NPU 0 의 모든 PE. tp=4 라면 `npu:0:0-3` 처럼 정확히 PE 수와 일치시키는 게 안전.
- **continuous batching**: furiosa-llm 의 scheduler 는 기본 continuous batching. 본 profiling 은 batch_size 개를 **동시에 한 번** submit 함으로써 같은 step 에서 처리되도록 유도 (inf2 의 `is_continuous_batching=False` 등가 효과).

## 참고

- 공식 LLM API: <https://developer.furiosa.ai/latest/en/furiosa_llm/reference/llm.html>
- 공식 ArtifactBuilder API: <https://developer.furiosa.ai/latest/en/furiosa_llm/reference/artifact_builder.html>
- Build by examples: <https://developer.furiosa.ai/latest/en/furiosa_llm/build-artifacts-by-examples.html>
- SamplingParams: <https://developer.furiosa.ai/latest/en/furiosa_llm/reference/sampling_params.html>
