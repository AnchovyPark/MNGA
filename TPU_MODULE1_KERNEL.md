# TPU Module 1 지시서 — shape로 kernel 단독 latency 예측

작성일 2026-11-26. 대상: TPU v4 VM에서 Module 1을 맡는 세션.

먼저 [`TPU_MODULES_COMMON.md`](TPU_MODULES_COMMON.md)를 읽는다. token 단위, 공통 코드, 측정 규칙, Step 0·1은 그 문서에 있다.

## 1. 목적

kernel type과 shape가 주어지면 단독(격리 compile) latency `U`를 내는 예측기를 만든다. **측정하지 않은 shape에서 맞아야 한다.**

확인할 것은 두 가지다.

1. 정확도: 안 본 shape, 안 본 model 크기에서 `U`를 얼마나 맞추는가.
2. 측정량: 그 정확도를 얻는 데 측정이 몇 점 필요한가.

**최소 결과**: Step 1 통과, 그리고 `gemm`만으로 Step 3~4의 T1·T2 표.

## 2. 참고 결과

| 출처 | 내용 |
|---|---|
| RNGD (`memory/2026-07-27.md` §3) | generic matmul `(M,K,N)` 50점에서 leave-one-out MAPE 약 9% (Ridge 9.5, GBM 8.9). feature 중요도는 flops가 0.94 |
| RNGD (`memory/2026-07-23.md` §12.2) | tree model은 학습 범위 밖을 외삽하지 못했다. 선형(물리) model이 외삽에서 나았다 |
| LENS, TPU v6e·v5e | 같은 kernel에서 v6e/v5e 속도 비가 `gemm` 2.3~3.8배, vector kernel 1.8~2.2배로 다르다. 공개 spec의 peak 성능 비만으로는 vector kernel을 맞출 수 없다 |
| LENS, RNGD | `gemm` 단독 latency가 seq에 대해 단조롭지 않은 지점이 있었다. TPU에서도 있는지는 확인되지 않았다 |

TPU 수치는 v4가 아니라 v6e·v5e에서 나온 것이다. 경향만 참고한다.

## 3. 대상 kernel과 shape 공간

대상은 `kernels.py`의 6종이다: `gemm`, `rmsnorm`, `rope`, `silu`, `mul`, `add`. 측정 함수는 `kn.KERNELS[type]`을 그대로 쓰고, 입력과 weight는 전부 인자로 넘긴다.

model 크기 10종은 `TPU_INSTRUCTIONS.md` Step 3의 목록 `(D, INTER, NH, HD, KV)`를 쓴다.

### 3.1 `gemm` `(M, K, N)`

- `M` (= seq): 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096
- `(K, N)` model 유래: 크기 10종마다 `(D, NH·HD)`, `(D, KV·HD)`, `(NH·HD, D)`, `(D, INTER)`, `(INTER, D)`. 중복은 한 번만 잰다.
- `(K, N)` model 밖: K, N을 각각 [256, 16384]에서 log-uniform으로 뽑아 64의 배수로 맞춘 40쌍. seed를 고정하고 목록을 commit한다.

약 800점이다.

### 3.2 vector kernel `(M, N)`

- `rmsnorm`, `add`: N은 크기 10종의 D
- `silu`, `mul`: N은 크기 10종의 INTER
- `rope`: N은 크기 10종의 `NH·HD`와 `KV·HD`, HD는 그 크기의 값
- 각 type에 model 밖 N을 10개씩 추가한다 ([256, 16384], 64의 배수).
- `M`은 `gemm`과 같다.

### 3.3 2차 범위

attention core의 kernel(`bmm`, scale, mask, softmax)과 `lm_head`(`M` = 1, `N` = 128256)는 Step 4까지 끝난 뒤에 한다.

## 4. 실험 순서

Step 0·1은 공통 문서 §5를 따른다.

### Step 2. 매끄러운지 확인

sampling 전에 latency가 shape에 대해 매끄럽게 변하는지 먼저 본다. 계단이나 튀는 점이 있으면 sampling 간격과 feature가 달라져야 한다.

- `gemm` `(K, N) = (2048, 8192)`와 `(4096, 4096)`: M을 64부터 4096까지 64 간격으로
- `gemm` `(M, K) = (512, 2048)`: N을 512부터 16384까지 256 간격으로
- `gemm` `(M, N) = (512, 2048)`: K를 512부터 16384까지 256 간격으로
- `add` `N = 2048`: M을 64부터 4096까지 64 간격으로

보고: 축마다 latency 곡선, 그리고 이웃 점에서 10% 이상 벗어나는 점의 위치. 그런 점이 있으면 위치에 규칙(2의 거듭제곱 근방, 특정 배수)이 있는지 적는다. 그 지점에서 HLO 연산 목록이 바뀌는지도 같이 본다.

### Step 3. 측정

§3.1, §3.2의 점을 잰다. Step 2에서 튀는 구간이 나왔으면 그 구간을 더 촘촘히 잰다.

행마다 flops, weight bytes, 입력 bytes, 출력 bytes를 같이 계산해 둔다 (`gemm`: flops = 2·M·K·N, weight bytes = 2·K·N).

### Step 4. fitting과 평가

model은 단순한 것부터 올린다. 목표를 채우는 가장 단순한 model을 채택한다.

| model | 내용 |
|---|---|
| P0 | roofline: `max(flops / peak compute, bytes / memory bandwidth)`. 공개 spec 값만 쓰고 fitting하지 않는다. spec 값의 출처를 적는다 |
| P1 | type별 선형: `U = a + b·flops + c·bytes` |
| P2 | P1의 잔차를 shape feature로 보정 (Step 2에서 찾은 규칙 포함) |

평가 split은 네 가지다.

| split | 학습 | 평가 | 보는 것 |
|---|---|---|---|
| T1 | 무작위 80% | 나머지 20% | 내삽 |
| T2 | 크기 9종에서 나온 shape + model 밖 shape | 남은 크기 1종에서 나온 shape (10번 반복) | 안 본 model 크기 |
| T3 | M ≤ 1024 | M ≥ 2048 (반대 방향도) | seq 외삽 |
| T4 | model 유래 shape만 | model 밖 shape | 안 본 구조 |

보고: split마다 type별 MAPE, 90% 지점 오차, 최대 오차. `U`가 20µs 미만인 행과 그 이상인 행을 나눠서도 낸다. 작은 kernel은 측정 noise의 비중이 크다.

목표(잠정): T1 MAPE 5% 이하, T2 MAPE 10% 이하. 오차는 Step 1의 noise floor보다 작아질 수 없다.

목표에 못 미치는 type이 있으면 오차가 큰 행의 shape를 나열하고 공통점을 적는다.

### Step 5. 측정 수 줄이기

학습 점 수를 25, 50, 100, 200, 400, 전체로 바꿔 가며 T2 오차를 본다. 점은 무작위로 뽑고 seed 5개의 평균과 범위를 낸다.

보고: 점 수에 따른 오차 곡선, 목표를 채우는 최소 점 수, 점 하나를 재는 데 걸린 시간(compile 포함).

### Step 6. Module 2에 넘기기

1. `profiling/tpu_v4/module1/predict.py`에 `predict_u(kernel_type, M, K, N, HD) -> us`를 만든다. fitting된 계수는 json 파일로 commit한다.
2. chain의 token 15개 × 크기 10종 × seq {128, 512, 1024, 2048}에 대해 예측값을 내서 `data_upred_tpu_v4.txt`로 commit한다.
   `UPRED D INTER NH HD KV token seq u_pred_us`
3. Module 2가 push한 data(`profiling/tpu_v4/module2/data_*.txt`)에 같은 token의 측정값이 있으면 비교한다.
   - 예측값과 Module 2 측정값의 오차
   - Module 1이 직접 측정한 값과 Module 2 측정값의 차이. 같은 kernel, 같은 shape이므로 noise floor 안에서 같아야 한다. 다르면 두 VM의 측정 방식이 어긋난 것이니, 다른 작업보다 먼저 원인을 찾는다.

## 5. 산출물

- `profiling/tpu_v4/module1/`: 측정 script, fitting script, `predict.py`, 계수 json
- `profiling/tpu_v4/module1/data_kernels_tpu_v4.txt`:
  `KROW type M K N HD dev_min_us dev_med_us wall_min_us n_samples`
  - `gemm`: `x[M,K] @ w[K,N]`, HD = 0
  - vector kernel: `x[M,N]`, K = 0. `rope`만 HD를 채운다
- HLO 연산 목록은 같은 key로 별도 파일에 둔다.
- `memory/YYYY-MM-DD-tpu-m1.md`: Step별 표와 판정

## 6. 범위 밖

window(kernel 2개 이상)의 latency, fusion·overlap 분석은 Module 2의 일이다. decode, batch > 1, multi-chip도 범위 밖이다.
