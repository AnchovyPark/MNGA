# RNGD 프로파일링 세션 기록 (2026-06-10)

## 환경 핵심 사실
- furiosa-llm **2026.2.1** (uv venv `.venv`, python 3.10), RNGD 1장 (npu0, 8 PE)
- `ArtifactBuilder` 는 config 객체 방식만 지원: `parallel_config=ParallelConfig(...)`,
  `bucket_config=BucketConfig(...)`. tp ∈ {4, 8, 32} 만 지원 (tp=칩 수 아니라 **PE 수**, 1칩=8PE → tp=8).
- **furiosa-ai HF org 모델 = prebuilt artifact** (artifact.json + binary_bundle.zip).
  ArtifactBuilder 불가 (raw safetensors 없음) → `LLM()` 으로 직접 로드. bucket 은 컴파일 시 고정.
- `generate()` 의 `prompt_token_ids` 는 **list-of-lists** (torch tensor 넣으면
  "Boolean value of Tensor..." 에러).
- run_profiling.py 에 prebuilt 자동감지 + 신 API 반영 완료 (이 폴더의 버전이 최신).

## 빌드 시 주의 (Qwen3-14B 에서 확인)
- **`uv run` 으로 빌드 금지** — Ray 워커가 다른 env 를 잡아 `No module named 'ray'`.
  반드시 `.venv/bin/python` 직접 실행.
- **워커 수 2/2 로 제한** (`--num-pipeline-builder-workers 2 --num-compile-workers 2`).
  pipeline builder 워커당 모델 전체(~28GB)를 RAM 에 올림 → 기본값 4 는 OOM (RAM 125GB).
- Qwen3-14B, bucket 2개 (128/256) 빌드: 574초. 컴파일 task 수 = 고유 (block종류 × shape).
- 실제 텍스트 sanity check 통과 (Paris / H2O 등 정상 출력).

## 측정 결과 (Llama-3.1-8B prebuilt, bs=1)
- 결과: `LENS/inference_results/profiling/RNGD/Llama-3.1-8B-Instruct/`
  - `bs1/RNGD_bs1_20260610_143535.csv` — min14 (combo 11~13 은 NPU 크래시로 ERROR)
  - `bucket_check/RNGD_bs1_20260610_145111.csv` — 37 combo (ol=1 TTFT 25 + TBT slope 12)
- **TTFT ≈ 평탄**: il 32→1023 에서 19.3~21.7ms (overhead 지배). chunked(>1024) warmed 21~24ms.
  warmup 안 된 chunked 첫 호출은 50~94ms 스파이크.
- **TBT (bucket 별)**: 256:14.57, 384:14.71, 512:14.84, 768:15.02, 1024:15.19, 2048:17.0(hot)
- **min14 예측 검증**: kv≤1024 combo 전부 ±2%. il=1030 combo +10~12% — 원인은 써멀
  (TBT 2048 을 72°C 에서 측정 vs eval 은 식은 상태). TBT(2048)=15.5 면 +1.9%.

## Qwen3-14B 직접 컴파일 검증 (bucket 128/256)
- 무작위 (il,ol) 8개 예측: |오차| 평균 2.05%, 최대 3.23% ✅
- P = {128: 7.2, 256: 25.0}ms (14B 는 prefill 계단 뚜렷), TBT = {128: 30.31, 256: 30.24}ms
- TBT 동일한 이유: decode 는 weight-bound (weight 29GB vs kv 차이 42MB ≈ 0.14%).
  kv 1024+ 에서야 계단 가시화 (8B 데이터와 정합).

## 이슈 / 미해결
- **NPU 크래시**: 8B prebuilt, il=2050 ol=1950 (bs1, kv~4000) 에서
  `Npu npu0pe0-3: Unknown error -5` → 엔진 종료, 디바이스 hang.
  복구: `sudo modprobe -r furiosa_rngd && sudo modprobe furiosa_rngd` 후
  PCI rescan (`echo 1 > /sys/bus/pci/devices/0000:01:00.0/remove; echo 1 > /sys/bus/pci/rescan`).
- **써멀 드리프트**: idle 53°C, sweep 후 72°C. 쿨다운 분당 ~1°C (느림).
  긴 decode 는 run 간 단조 증가 (il=1030 ol=1000: 15.6→17.0→18.5s, +18%).
  → profiling 과 eval 을 같은 열 상태에서 측정해야 함. 온도 로깅 권장.
- TBT(4096) 미측정 (il=2050 류 예측에 필요).

## NPU 메모리 구조 (Qwen3-14B LENS 빌드에서 규명)
- RNGD HBM 51GB. 점유 = weight(29.6GB) + **kv 풀(탐욕적: 용량−io예약 전부 선점)**
  + bucket 워크스페이스(첫 사용 시 lazy 누적, 4096 prefill=1.84GB, 8192≈3.7GB).
- kv 풀을 직접 줄이는 인자 없음 → **`max_io_memory_mb` 를 키우면 (예: 8192)**
  kv 풀이 줄어 워크스페이스 자리가 생김. (줄이면 역효과 — kv 풀이 커짐)
- `SchedulerConfig(max_processing_samples=...)` 는 메모리에 영향 없었음.
- 14B + bucket 7개(128~8192) 동시 활성화는 51GB 초과. **bucket 선택 로드**로 해결:
  artifact 사본 만들고 artifact.json 의 `model.pipeline_metadata_list[0].attention_buckets`
  리스트만 추려내면 됨 (바이너리는 symlink 공유, 재컴파일 불필요).
  → `Qwen3-14B-tp8-lens-large-only` (2048/4096/8192 만) 로 il=8000 까지 동작 확인.

## ⚠ 써멀 스로틀링 (가장 큰 미해결 이슈)
- 14B 지속 부하 시 88~89°C / 97W 에서 **~2.6배 감속** (TBT 30→80ms/tok).
  드리프트가 세션 내내 진행 (il=1030/ol=1010 run별: 42s→64.5s→62.6s).
- 이 때문에 14B min14 예측 검증 1차 시도는 |오차| 평균 47% (프로파일=저온, 평가=고온).
  **공식이 아니라 측정 조건 문제** — 8B 검증(±2%)과 14B bucket128/256 검증(±3.3%)은
  열 상태가 비슷했어서 통과했음.
- 근본 원인 = 데스크탑 섀시 풍량 부족 (RNGD 는 서버용 패시브 쿨링).
  **governor 로 주파수 낮추는 식의 인위적 제한은 쓰지 않는다 — 하드웨어 특성을 그대로
  측정/예측하는 게 목표** (사용자 방침). 대신 측정을 짧게 쪼개고(bucket 일부씩), 식혀가며,
  profiling 과 eval 의 열 상태를 맞추는 방식으로 대응.

## 커널 프로파일러 평가 (2026-06-10 저녁, 완료)
- 도구: `kernel_profiler_eval.py` — torch 2.10+xpu 의 RNGDProfiler(kineto)는 Intel PTI
  충돌로 불가 → **저수준 우회**: `CompileModule.from_exported` → `generate_profiles(dev)`
  → 실행 → `cm.edf.npu_node.build_tuc_profile_spans()` 로 TUC span 추출.
  시간 필드 = `FunctionEvent.time_range.elapsed_us()` (NPU 하드웨어 타임스탬프, wallclock 아님).
- **측정 입도**: Task(커널 envelope) / TuExec(연산 슬라이스, 8 PE 병렬이라 합>Task 가능)
  / DMA(채널별, 병렬). run 간 CV 0~3% (커널이 짧아 써멀 영향 없음).
- **wallclock dispatch overhead ~200~400us/호출** — add 단일 2.1us(device) vs 182us(wall)
  → wallclock 금지 규칙 RNGD 에서 정량 확인.
- 결과 CSV: `kernel_profiler_eval_results*.csv`, `kernel_single_vs_chain.csv` (128~8192).
- **체인은 단일 Task 로 자동 fusion**. 절약(us)은 커널2의 DMA 절감과 비례.
  matmul+add: +1.15(S128) → +386.7(S4096). matmul+softmax 는 절약 정체 후
  **S=4096 에서 역전(-392us)** — fusion 이 더 느림 (tiling 이상치 계열, 예측에 중요).
- **컴파일 한계**: 정사각 matmul 8192³ 는 `UnsupportedOpError` (EinsumByDpe tactic 부재
  → CPU 노드 잔존 → EDF 거부). 4096³ 까지는 OK. 8192 elementwise/softmax OK.
  원인: RNGD=TCP 는 einsum 을 shape 별 tactic 으로 즉석 생성하는데 (cuBLAS 류 라이브러리
  없음), tactic 커버리지가 실전 추론 shape 위주. LLM 8192 bucket 은 K가 작고 TP 분할된
  shape 라서 통과. TPU/XLA 는 128×128 고정 타일이라 임의 shape 보장 — RNGD 단일커널
  sweep 설계는 실전 모델 shape 격자 안에서 할 것.

## main evaluation 목표 구조 (2026-06-12 규명)
- 본 evaluation 의 본체 = `LENS/estimator_results/SUMMARY.csv`. 한 줄 = **(hw, tp, model, bs)**
  한 조합의 예측 검증 결과 (`pooled_abs_mean_pct` + arxiv/cnn/sharegpt/writing 별 오차%).
- 기존 검증 hw: **inf2 / tpu_v4 / tpu_v5e / tpu_v6e**. 공통 모델 = llama3.2-1b, mistral-7b,
  qwen3-14b 등, bs {1,2,4,8,16,32}. 대부분 오차 ±1~6%.
- **RNGD = 새 hw 블록**으로 같은 격자(LLM 3개 × 4 데이터셋 × bs 6단)를 채우는 게 목표.
- 생성 경로: ① profiling CSV → per-bucket P(b)/TBT(b) fit → ② eval CSV 4개(실제 데이터셋
  batch 추론) → ③ `estimator_eval/compare_to_eval.py` → `estimator_results/<hw>/tp<N>/<model>/bs<B>/*.json`
  → ④ JSON 모아 SUMMARY 한 줄. **③④ 는 hw-중립이라 그대로 재사용**.
- RNGD 격차: **eval runner 부재** (지금은 통제 combo 만, 실제 데이터셋 batch 추론 안 함),
  pre-sampled batch (`inference_dataset/RNGD/bs*/`) 부재. → 다음 핵심 작업.

## RNGD 컴파일 지원 아키텍처 화이트리스트 (2026-06-12 규명)
- furiosa-llm SDK 는 컴파일 전 `find_compiler_config(model_type, task, per_layer_params_b)`
  로 **지원 model_type 화이트리스트**를 검사. 없으면 컴파일 시작도 전에
  `ValueError: No compiler configuration available for model_type=...` 로 즉시 거부.
  (GPU 와 달리 NPU 는 구조별 최적화 레시피를 사람이 미리 등록해야 컴파일됨.)
- **generate task 지원 model_type (6개)**: `llama, qwen2, qwen3, qwen3_moe, exaone4, gpt_oss`.
  ❌ 미지원: mistral, gemma/gemma2/gemma3, phi/phi3, mixtral, gptj, gpt_neox, cohere,
  starcoder2, deepseek_v3, falcon 등.
- 7~8B 정식 후보: **Llama-3.1-8B**(llama, prebuilt 확보), **Qwen2.5-7B-Instruct**(qwen2,
  tpu_v5e 격자에 이미 존재), **Qwen3-8B/4B**(qwen3). gpt_oss·exaone4 는 20B+ 라 중형 아님.

## Mistral model_type 위장(spoof) 컴파일 — 성공 (2026-06-12)
- Mistral 은 정식 미지원이나, 구조가 Llama 와 동일(RoPE/GQA/SwiGLU, v0.3 sliding_window=null,
  가중치 키 이름도 동일). config.json 의 `model_type:"mistral"→"llama"`,
  `architectures:["LlamaForCausalLM"]` 로만 바꾸면:
  → 화이트리스트 통과 → Llama 최적화 클래스가 Mistral 가중치 13.5GB 로드 → 컴파일·추론 정상.
- 실제 텍스트 5종(지식/추론/번역/코딩/요약) 추론 검증: 영어·프랑스어·코딩·산수 **전부 정확**.
  한국어 번역만 깨짐("오렵에") — 이는 **Mistral 모델 자체의 한국어 약점**(vocab 32k, 학습데이터
  부족)이지 컴파일/spoof 결함 아님 (영어가 정상이므로). 한국어 중요 시 Qwen 계열 권장.
- 코드/결과: `LENS/inference_profiling/RNGD/mistral_spoof_check/`
  (`mistral_spoof_llama.py` 다운로드+config패치+빌드, `mistral_spoof_infer.py` 실텍스트 추론,
  `mistral_infer_result.log` 결과). 위장 모델 dir `~/mistral7b_as_llama`,
  artifact `~/compiled_models_lens_profiling/Mistral-7B-spoof-llama-check` (bucket 128/256, max_len 256).
- **미결정**: 7B 슬롯을 (a) Mistral spoof 로 갈지 / (b) Qwen2.5-7B 정식으로 갈지. spoof 는
  기술적으론 되나 논문에선 "Llama 호환 경로 실행" 각주 필요(정식 지원 아님).
- HF 다운로드 주의: `allow_patterns=["model-*.safetensors", ...]` + `ignore_patterns=["consolidated.safetensors"]`
  로 HF 샤드만 받을 것 (안 그러면 consolidated 중복본 14.5GB 같이 받음).

## Llama-3.2-1B 정식 컴파일 + 메모리 모델 정밀 측정 (2026-06-12)
- **Llama-3.2-1B (unsloth/Llama-3.2-1B-Instruct, model_type=llama)** RNGD 직접 컴파일·추론 정상.
  meta-llama 는 gated(403) 라 동일사양 재업로드 unsloth 사용 (h=2048,L=16,vocab=128256).
  논문 격자엔 공식 meta-llama 가중치 권장(access 승인 필요).
- **allocation 정밀 측정 (1B, 2-bucket vs 7-bucket 로드 로그 비교)**:
  - Model weights 2.3GiB / Reserved IO memory 2.0GiB / KV cache 43.2GiB — **bucket 2→7개(8192 포함)
    늘려도 셋 다 불변**. 늘어난 건 컴파일 binary 2.4→7.1MiB (~1MiB/bucket) 뿐.
  - → **bucket 개수는 HBM 점유와 거의 무관**. "bucket 별 워크스페이스 lazy 누적(4096=1.84GB,
    8192=3.7GB)" 이라던 이전 memory 기록은 **부정확 → 정정**. IO 워크스페이스는 단일 영역이고
    `max_io_memory_mb`(기본 2048MiB=2.0GiB) 로 정해지는 고정값.
- **paged attention 확정**: furiosa-llm 소스에 `paged_attention_num_blocks/block_size`.
  KV cache 는 단일 블록 풀(block_size 1024~2048)을 모든 요청이 공유. "non-paged 면 KV 비공유"
  우려는 해당 없음 — paged 라 공유됨.

## 14B full 7-bucket 한 번에 로드 — 가능 확인 + min14 통짜 실패 원인 규명 (2026-06-12)
- **이전(06-10) 14B min14 통짜 sweep 실패 기록 (memory 에 없던 사건)**:
  - 16:56~17:00 run: 전 combo `engine terminated` (combo0 il=64 조차) → 통짜 전멸.
  - 이후: max_model_len=4096 으로 낮춰 combo 0~11 OK, combo 12(il4100/ol300)·13(il4100/ol3950)은
    합>4096 이라 `SKIPPED_TOO_LONG` → 18:09 에 large-only+max_len8192 로 12·13 별도 측정.
  - **즉 min14 를 한 번의 로드로 완주 못 하고 분할했었음**.
- **원인 = 컴파일 아님, 런타임 메모리 충돌**: max_len8192 + 전 bucket + 탐욕적 KV 풀이 51GB 초과.
- **해결·재검증 (06-12)**: full 7-bucket `Qwen3-14B-tp8-lens` 를 `--max-io-memory-mb 8192
  --max-processing-samples 4` 로 로드 → KV 풀 43.2→12.8GiB 축소 → **engine terminated 없이 로드 성공**.
  warmup 제거(`--skip-warmup`) + 각 bucket 1회 ol=1 smoke(profile_bucket_smoke.csv): 전 7 bucket OK,
  il=8000(8192 bucket)까지 단일 로드 완주, 온도 72°C 유지. **large-only 분할 불필요 확정**.
- **run_profiling.py 신규 옵션**: `--max-io-memory-mb`, `--max-processing-samples`
  (init_model→_extra_llm_kwargs, 두 LLM 로드 경로 모두 반영). 임시 /tmp 스크립트 대신 본 코드로.
- **warmup 이 발열 주범**: warmup_each_bucket 이 bucket 별 큰 prefill(8192 bucket→il=4097) 수행.
  발열로 진행 불가 → 제거. 단 hot 측정 안정성은 본 측정 때 n_runs≥3 + compare_to_eval 의 median 으로 보완.
  긴 decode(min14 combo11 ol1950, 13 ol3950)도 발열원 → TBT 평탄하므로 ol 축소 검토 가치.

## LENS 전제 검증 (14B, 2026-06-12) — 발열로 일부 bucket 만
- **발열 방침**: 14B sustained decode 가 섀시 냉각 한계 초과(40s 만에 71°C→). governor 등
  인위적 제한은 안 씀(하드웨어 특성 그대로 측정). → **bucket 일부씩 짧게, 식혀가며** 측정.
- **① 동일 bucket → 동일 TTFT** (ttft_check, ol=1, n_runs=3 median): 같은 bucket 안에서
  il 을 바꿔도 TTFT 동일.
  - bucket256: il 130/200/250 → 56.4/56.4/56.1ms (편차 0.56%)
  - bucket1024: il 600/800/1000 → 157.7/156.7/156.4ms (0.85%)
  - bucket4096: il 2200/3200/4000 → 635.9/635.7/636.5ms (0.13%)
  - → padding 으로 같은 bucket = 같은 prefill 비용. CSV: `Qwen3-14B/ttft_check/`.
- **② bucket 내 E2E 가 ol 에 선형** (tbt_b2048, il=1800 고정, ol={20,80,140,200}, n_runs=2):
  - E2E = 80.2 + 37.02·ol, **R²=0.99921** → bucket 내 **TBT 상수(≈37ms/tok)** 확인.
  - CSV: `Qwen3-14B/tbt_b2048/`.
- **TTFT 는 bucket 별로 다름**(56→157→310→636ms) — 정상(prefill 비용은 입력 길이 증가).
- **P/TBT 분리 주의**: decode 구간만으로 그은 직선의 절편(80ms)은 진짜 TTFT(prefill,
  bucket2048≈310ms) 가 아님. E2E≈TTFT+(ol−1)·TBT 라 첫 토큰(prefill)이 훨씬 비싸기 때문.
  → P 는 prefill(ol=1) 측정에서, TBT 는 decode slope 에서. (LENS compare_to_eval 의 fit 과 정합)
- **smoke (전 bucket ol=1, 단일 로드)**: il 100~8000 전 7 bucket OK, TTFT 39→1587ms 단조증가.
  CSV: `Qwen3-14B/bucket_smoke/`. profile CSV: profile_bucket_smoke / _ttft_check / _bucket2048_tbt.

## 1B bs=32 컴파일·추론 + 공식 meta-llama 전환 (2026-06-12)
- **HF 로그인**: 공용 PC 에 hwang2006 토큰 저장돼 있던 걸 본인(anchov) 토큰으로 `hf auth login
  --force` 교체 → **공식 meta-llama/Llama-3.2-1B-Instruct 접근 가능**(gated 승인됨). 격자는 공식 가중치 사용.
  (hf CLI: `huggingface-cli` deprecated → `hf auth login`.)
- **bs=32 × 7-bucket 컴파일 성공** (build 450s). prefill_buckets=(1,b)(prefill 은 항상 batch1),
  decode_buckets=(32,b). artifact: `~/compiled_models_lens_profiling/Llama-3.2-1B-tp8-bs32`.
- **추론 1차 OOM**: bs=32 라 KV 풀이 49GB 선점 → 워크스페이스 2.18GB 할당 실패(engine terminated).
  → `--max-io-memory-mb 8192 --max-processing-samples 32` 로 KV 풀 49→37.2GiB 축소 → **추론 정상**
  (재컴파일 불필요, skip-compile/prebuilt 로드). bs=32 smoke 전 7 bucket OK.
  주의: bs=32 면 max_processing_samples 를 bs 이상(32)으로 둘 것(4로 두면 batch 처리 불가).

## heterogeneous batch 거동 규명 (2026-06-12, 1B bs=32)
- **결론: batch 안에 서로 다른 il(=서로 다른 prefill bucket)을 넣어도 정상 동작**.
  prefill_buckets=(1,b) 라 prefill 은 sample 마다 자기 bucket 으로(배치1), decode 는 (32,b)로 배칭.
  → LENS 의 per-sample dispatch(`batch_TTFT=ΣP(bucket(il_i))`) 모델과 정합. (static 아님 — 앞선 추측 철회)
- 2-샘플 probe: [300,500](동일 bucket 다른 길이), [100,200]·[300,800]·[100,8000](다른 bucket) **전부 OK**.
- 32-샘플 probe: [100]×32 OK, [8000]×32 OK, **[100,8000]×16 교대 OK**, [100,8000,100,8000] OK.
- **유일 실패 = [8000]+[100]×31 (긴 거 1 + 짧은 거 31, 극단 1:31 치우침)** →
  `ValueError: use the same length of input_ids in a batch` (native 엔진의 decode-step 그룹핑
  코너 케이스). bucket 혼재 금지가 아니라 **극단적으로 치우친 길이 비율**이 스케줄러 그룹핑을 깨는 것.
- **eval runner 함의**: 실제 데이터셋 batch 는 길이 분포가 골고루라 대부분 정상. 단 극단 outlier
  대비해 길이별 정렬/그룹핑(또는 비율 균형)으로 코너 케이스 회피 권장. 정확한 깨짐 비율은 추가 규명 여지.
- 실험 코드: `batch_heterogeneous_check.py`, probe 는 /tmp/hetero_probe*.py.

## git
- 2026-06-10: 816f463 + ba95cd4 origin/agent/error-tuning 푸시 완료
  (RNGD run_profiling 신 API, 측정 CSV 전부, kernel eval). 이후 완료마다 커밋·푸시.
- 2026-06-12: main eval 구조 규명 + 컴파일 화이트리스트 + Mistral spoof 검증 기록·코드 커밋.
- 2026-06-12: 1B 정식 컴파일·allocation 측정·paged 확정·14B full bucket 단일로드 검증·
  run_profiling --max-io-memory-mb/--max-processing-samples 추가 커밋.

## 진행 중 (2026-06-10 저녁)
- Qwen3-14B **LENS bucket 전체 본 컴파일** (128~8192, tp=8, bs=1) + min14 sweep:
  - artifact: `~/compiled_models_lens_profiling/Qwen3-14B-tp8-lens`
  - 결과 예정: `LENS/inference_results/profiling/RNGD/Qwen3-14B/bs1/`
  - 로그: `~/rngd_qwen14b_lens_compile.log`, 예상 1~1.5h (캐시 재사용 시 단축)
  - 1차 시도는 16:08 사용자 재부팅(키보드 hang)으로 중단 → 16:4x 재시작.
    `~/.cache/furiosa/llm` (param_files + compiled_graphs 30GB) 캐시는 유효.
  - 재실행 명령은 위 "빌드 시 주의" 의 워커 2/2 + `.venv/bin/python` 형태 그대로.
