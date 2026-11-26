# TPU Module 2 지시서 — kernel 순서와 kernel별 값으로 3-gram window 예측

작성일 2026-11-26. 대상: TPU v4 VM에서 Module 2를 맡는 세션.

먼저 [`TPU_MODULES_COMMON.md`](TPU_MODULES_COMMON.md)를 읽는다. token 단위, 공통 코드, 측정 규칙, Step 0·1은 그 문서에 있다.

## 1. 목적

연속한 kernel 3개의 window latency `W3`를 **측정하지 않고** 예측한다. input은 kernel의 순서와 kernel별 값이다.

이 문서에서 정해야 하는 것은 두 가지다.

1. **output**: 예측기가 무엇을 내야 하는가 (§4).
2. **input**: kernel 순서와 단독 latency `U`만으로 충분한가, 부족하면 무엇을 더 넣어야 하는가 (§5).

우선순위는 **3-gram을 잘 맞추는 것**이다. layer 사이 조립과 full model은 그 다음이다.

**최소 결과**: Step 2의 판정, 그리고 seq {512, 2048}만으로 Step 3~4의 표.

## 2. 정의

token은 공통 문서 §2의 15개다 (`n = 15`). `U(i)` = token i 단독, `W2(i)` = token i, i+1, `W3(i)` = token i, i+1, i+2, `full` = 15개 전체다. 전부 `kernels.py`의 `window(lo, hi)`로 만든다.

```
uni   = Σ_{i=0}^{n-1} U(i)
tele2 = W2(0) + Σ_{i=1}^{n-2} [ W2(i) − U(i)  ]
tele3 = W3(0) + Σ_{i=1}^{n-3} [ W3(i) − W2(i) ]

r2(i) = W2(i) / (U(i) + U(i+1))
r3(i) = W3(i) / (U(i) + U(i+1) + U(i+2))
```

- `r`은 window가 단독 합의 몇 배인지다. 1이면 가산적이고, 1보다 작으면 겹쳐서 숨은 시간이 있다.
- window 수: U 15 + W2 14 + W3 13 + full 1 = (크기, seq)마다 43개.
- 예측값은 `W2_hat`, `W3_hat`으로 쓴다. `tele3_hat`은 위 식에 예측값을 넣은 것이다. `U`는 측정값을 쓰다가 Step 6에서 Module 1의 예측값으로 바꾼다.

## 3. 참고 결과

| 출처 | 내용 |
|---|---|
| RNGD (`memory/2026-07-27.md` §4) | matmul 쌍의 overlap은 상수가 아니다. 앞 kernel의 compute가 크고 뒤 kernel의 weight가 클수록 커졌다 |
| RNGD (같은 곳) | 사이에 낀 연산에 따라 다르다. silu는 overlap을 그대로 두고, rmsnorm은 줄였다 |
| RNGD (`memory/2026-07-27.md` §2) | telescope 오차의 부호가 1B 크기(과소)와 8B 크기(과대)에서 반대였다. 고정 공식 하나로는 둘 다 맞추지 못했다 |
| LENS, TPU v6e | 순서가 결과를 바꾼다. 숨는 비율이 `gemm→silu` 0.68~1.15, `silu→gemm` 0.34~0.53, `rmsnorm→gemm` 0.04~0.15, `gemm→add` 0.82~0.97, `add→gemm` 0.05~0.22였다 (seq 13점의 범위) |
| LENS, TPU v6e | decoder layer 실측은 kernel 단독 합의 1.02배(seq 128)에서 0.34배(seq 4096)까지 변했다. 단, 대부분의 seq에서 softmax와 `AV`가 다른 kernel로 치환돼서 이 비율 전부가 overlap은 아니다 |

TPU 수치는 v4가 아니라 v6e에서 나온 것이고, 숨는 비율의 정확한 정의는 NPU-LENS repo의 `section3/sec3.3`에 있다. 여기서 가져갈 것은 두 가지다. 예측기는 **순서를 구분**해야 하고, 같은 순서라도 **비율이 seq에 따라 변한다**.

## 4. output 후보

| 후보 | 내용 | 장점 | 단점 |
|---|---|---|---|
| A | window 전체 `W3` (과 `W2`) | 직접 측정한 값과 바로 비교된다 | 대부분이 단독 합이라, 예측이 나빠도 오차가 작아 보인다 |
| B | 비율 `r3` (과 `r2`) | 단위가 없어서 크기·seq·device 사이에 비교하기 쉽다. 예측기가 맞춰야 하는 부분(겹침)만 남는다 | A로 되돌릴 때 `U`가 필요하다 |
| C | marginal `W3(i) − W2(i)` | 합하면 바로 layer latency가 된다. "앞 두 kernel 뒤에 붙은 kernel의 시간"으로 읽힌다 | 두 측정값의 차라서 noise가 크다 |

- **학습 target은 B, 평가는 A와 `tele3_hat / full`로 한다.** 이것을 기본으로 두고, Step 4에서 A를 직접 학습한 것과 비교해 더 나은 쪽을 채택한다.
- C는 따로 학습하지 않는다. `W3_hat − W2_hat`으로 얻는다.
- "문맥 안에서 달라진 개별 kernel의 latency"는 output으로 삼지 않는다. 쌍의 overlap을 두 kernel에 어떻게 나눌지가 측정값으로 정해지지 않는다 (`memory/2026-07-27.md` §2).

## 5. input 후보

작은 집합에서 시작해 하나씩 더한다. **목표를 채우는 가장 작은 집합이 Module 2의 input이다.**

| 집합 | 내용 |
|---|---|
| S0 | window 안 token의 type 순서, token별 `U` |
| S1 | S0 + token별 flops, weight bytes |
| S2 | S1 + 이웃 token 사이의 연결 정보: 직접 연결됐는지, 연결 tensor의 bytes |
| S3 | S2 + shape 자체 (seq, D, INTER 등) |

- S0은 "순서와 latency만"이라는 원래 구상이다. S0으로 목표를 채우면 그것이 가장 좋은 결론이다.
- S3은 Module 1의 input과 같다. S3까지 가야만 맞는다면 Module 2가 `U`를 거치지 않고 shape에서 직접 예측하는 것과 다르지 않다. 그 경우 그 사실을 결론으로 보고한다.
- flops와 bytes를 device spec으로 나눈 값(`flops / peak compute`, `bytes / memory bandwidth`)도 같이 시도한다. 다른 device로 옮길 때는 이쪽이 필요하다.

## 6. 예측 model 후보

| model | 내용 | input |
|---|---|---|
| M0 | `W3_hat = U(i) + U(i+1) + U(i+2)` (baseline) | S0 |
| M1 | type 순서별 상수 비율. 학습 data에서 같은 type 순서를 가진 window의 `r3` 평균을 쓴다 | S0 |
| M2 | type 쌍별 상수로 숨는 양을 뺀다. `W2_hat = U_a + U_b − h[type_a, type_b] · min(U_a, U_b)`. `W3_hat`은 단독 합에서 두 이웃 쌍의 숨는 양을 뺀다 | S0 |
| M3 | `r3`를 feature로 회귀 (선형, 그 다음 tree) | S1~S3 |

M1, M2가 목표를 채우면 M3은 필요 없다.

## 7. 실험 순서

Step 0·1은 공통 문서 §5를 따른다.

### Step 2. pilot — kernel 단위 3-gram이 성립하는가 (gate)

1B 크기 `(2048, 8192, 32, 64, 8)`, seq {512, 2048}, window 43개를 **측정**한다.

보고: seq마다 `uni/full`, `tele2/full`, `tele3/full`, 그리고 window별 `r2`, `r3`.

| 결과 | 결론과 다음 행동 |
|---|---|
| `tele3` 오차 ≤ 5%이고 `uni` 오차보다 작다 | kernel 단위 3-gram 성립. Step 3으로 간다 |
| `uni` 오차 ≤ 5% | 가산성 성립. TPU v4에서는 kernel 단독 합으로 충분하다. 이 결론을 보고하고, Step 3은 seq {512, 2048}로 줄여 같은 결론이 크기 10종에서 유지되는지만 확인한다 |
| `tele3` 오차 > 10% | 3개로는 겹침이 닿는 범위를 못 덮는 것이다. `W4`, `W5`를 추가로 재서 몇 gram에서 오차가 5% 아래로 내려오는지 본다. 내려오면 그 n으로 Step 3을 진행하고, 5-gram에서도 안 내려오면 중단하고 보고한다 |
| 그 사이 | 반복 측정으로 noise인지 확인한 뒤 위 셋 중 하나로 판정한다 |

두 번째 줄도 유효한 결론이다. n-gram이 필요하다는 쪽으로 무리하게 해석하지 않는다.

`TPU_INSTRUCTIONS.md` Step 2의 결과(op 묶음 7개 단위)가 `profiling/tpu_v4/`에 올라와 있으면 같은 (크기, seq)에서 나란히 놓고 비교한다.

### Step 3. data 수집

크기 10종 (`TPU_INSTRUCTIONS.md` Step 3의 목록) × seq {128, 512, 1024, 2048} × window 43개 = 1720개를 측정한다.

- 순서는 seq {512, 2048}을 먼저 한다 (최소 결과).
- window마다 HLO 연산 목록을 같이 남긴다.
- 행마다 token별 flops, weight bytes, 연결 tensor bytes를 계산해 붙인다.

### Step 4. 3-gram 맞추기 (핵심)

§6의 model과 §5의 input 집합을 조합해 `W2`, `W3`를 예측한다.

평가 split은 세 가지다.

| split | 평가 대상 | 보는 것 |
|---|---|---|
| E1 | 크기 1종을 통째로 뺀다 (10번 반복) | 안 본 model 크기 |
| E2 | seq 1개를 통째로 뺀다 (4번 반복) | 안 본 seq |
| E3 | 크기 1종과 seq 1개를 동시에 뺀다 | 둘 다 안 본 조합 |

보고:

- (model, input 집합) × (E1, E2, E3)의 `W3` MAPE 표. `W2`도 같은 표로 낸다.
- M0 대비 오차가 얼마나 줄었는지. `W3`의 대부분이 단독 합이라서, M0보다 나아진 폭이 이 module이 한 일이다.
- 오차가 큰 window의 token 순서, 크기, seq. HLO 연산 목록이 다른 행과 다른지도 확인한다.
- target을 `r3`로 둔 것과 `W3`로 둔 것의 비교.

목표(잠정): E1에서 `W3` MAPE 5% 이하이고, M0보다 noise floor 이상 낫다.

이 Step의 결론은 두 문장으로 적는다. "output은 ○○로 한다." "input은 ○○까지 필요하다."

### Step 5. 예측한 window로 telescope

Step 4에서 채택한 model로, 뺀 크기(E1)에 대해 `W2_hat`, `W3_hat`을 만들고 `tele3_hat`을 계산한다.

보고: 크기마다, seq마다 다음 세 값.

| 값 | 의미 |
|---|---|
| `uni / full` | n-gram 없이 단독 합만 쓴 경우 |
| `tele3 / full` | window를 전부 측정한 경우. 예측으로 도달할 수 있는 상한이다 |
| `tele3_hat / full` | window를 예측한 경우. 이 module의 결과다 |

목표(잠정): `tele3_hat` 오차 10% 이하이고 `uni` 오차보다 작다.

`tele3_hat`은 예측값의 차를 12번 더하므로 오차가 쌓일 수 있다. `W3` MAPE는 낮은데 `tele3_hat`이 나쁘면 오차의 방향이 한쪽으로 쏠린 것이다. window별 부호를 같이 보고한다.

### Step 6. Module 1 연결

Module 1이 `profiling/tpu_v4/module1/data_upred_tpu_v4.txt`를 push하면, 측정한 `U` 자리에 예측한 `U`를 넣어 Step 5를 다시 계산한다.

보고: 측정 `U`를 쓴 `tele3_hat / full`과 예측 `U`를 쓴 값. 이 차이가 Module 1의 오차가 최종 결과에 주는 영향이다.

그 전에, Module 1이 직접 측정한 `U`와 이쪽에서 측정한 `U`가 noise floor 안에서 같은지 먼저 확인한다.

### Step 7. 확장 (Step 6까지 끝난 뒤)

1. **안 본 순서**: chain에 없는 type 순서의 window를 만들어 측정하고, Step 4의 model이 맞는지 본다. chain에서 나오는 type 순서는 12가지뿐이라, 여기까지 해야 다른 구조의 model로 일반화된다고 말할 수 있다.
2. **attention core**: `QK^T`, scale, mask, softmax, `AV`를 token으로 추가한다. `online-softmax` 치환이 일어난 window는 HLO의 `op_name`으로 가려내 따로 표시한다.
3. **layer 사이와 full model**: `TPU_INSTRUCTIONS.md` Step 5~6의 `L(1)` 자리에 `tele3_hat`을 넣는다.

## 8. 산출물

- `profiling/tpu_v4/module2/`: 측정 script, fitting script
- `profiling/tpu_v4/module2/data_windows_tpu_v4.txt`:
  `WROW D INTER NH HD KV window seq dev_min_us dev_med_us wall_min_us n_samples`
  - `window`는 token 이름을 `+`로 이은 것이다 (`gate+silu+up`). token 이름에 `_`가 있어서 `_`로 잇지 않는다. `kn.window_name(lo, hi)`를 쓴다.
- HLO 연산 목록은 같은 key로 별도 파일에 둔다.
- `memory/YYYY-MM-DD-tpu-m2.md`: Step별 표, 판정, §4와 §5에 대한 결론

## 9. 범위 밖

shape에서 `U`를 예측하는 것은 Module 1의 일이다. decode, batch > 1, multi-chip도 범위 밖이다.
