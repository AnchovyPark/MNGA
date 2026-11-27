# 용어 규칙

이 repo의 지시서, `memory/` 기록, commit message, 보고는 아래 용어만 쓴다. 처음 보는 사람이 표 하나만 보고 따라올 수 있게 하는 것이 목적이다.

- code의 변수 이름, 함수 이름, data 파일의 행 형식(`WROW` 등)은 바꾸지 않는다.
- 문서에서 그런 이름을 언급할 때는 아래 용어를 먼저 쓰고 code 이름을 괄호로 붙인다. 예: "3-gram latency(`W3`)".
- kernel, latency, layer, profiling, fusion, overlap, gemm, shape, seq 같은 기술 어휘는 영어 그대로 쓴다.

## 1. 기본 용어

| 용어 | 뜻 |
|---|---|
| **kernel** | layer를 이루는 연산 하나. 예: `gate` gemm, `silu` |
| **kernel latency** | kernel 하나만 따로 돌린 시간 |
| **2-gram latency** | 실행 순서상 연속한 kernel 2개를 한 번에 묶어 돌린 시간 |
| **3-gram latency** | 연속한 kernel 3개를 한 번에 묶어 돌린 시간 |
| **layer latency** | layer 하나 전체를 한 번에 돌린 시간 |

**예** (TPU v4, 1B 크기, seq 512, 단위 µs)

| 재는 것 | 시간 |
|---|---|
| gate kernel latency | 101.8 |
| silu kernel latency | 17.7 |
| up kernel latency | 101.6 |
| gate+silu 2-gram latency | 106.0 |
| silu+up 2-gram latency | 118.5 |
| gate+silu+up 3-gram latency | 173.7 |

kernel latency 세 개의 합은 221.1이지만 3-gram latency는 173.7이다. 묶어 돌리면 kernel끼리 overlap되거나 fusion되기 때문이다.

### layer를 이루는 kernel 16개 (TPU v4, 실행 순서)

```
norm1 → q → rope_q → k → rope_k → v → attn → o → add1 → norm2 → gate → silu → up → mul → down → add2
```

`attn`은 attention 연산(qk, scale, mask, softmax, av) 전체를 kernel 하나로 본 것이다.

## 2. 식에 쓰는 표기

`lat(…)` = 괄호 안 kernel들을 한 번에 묶어 돌린 시간이다.

| 표기 | 뜻 |
|---|---|
| `lat(a)` | kernel a의 kernel latency |
| `lat(a,b)` | a, b의 2-gram latency |
| `lat(a,b,c)` | a, b, c의 3-gram latency |
| `lat(layer)` | layer latency |

## 3. 계산 용어

| 용어 | 식 | 뜻 |
|---|---|---|
| **kernel latency 합** | `Σ lat(a)` | kernel을 따로 잰 값을 그냥 더한 것. overlap을 모른다 |
| **2-gram 합산** | `Σ lat(2개 묶음) − Σ lat(안쪽 kernel 1개)` | 2-gram latency로 layer latency를 계산한 값 |
| **3-gram 합산** | `Σ lat(3개 묶음) − Σ lat(안쪽 2개 묶음)` | 3-gram latency로 layer latency를 계산한 값 |
| **쌍 효과** | `lat(a,b) − lat(a) − lat(b)` | a와 b가 붙어서 줄어든 시간. 보통 음수 |
| **가운데 kernel 효과** | `lat(a,b,c) − lat(a,b) − lat(b,c) + lat(b)` | 가운데 b가 a, c와 동시에 붙을 때, 쌍 효과 두 개로 설명되지 않는 나머지 |
| **3-gram 비율** | `lat(a,b,c) / (lat(a) + lat(b) + lat(c))` | Module 2가 예측하는 값 |

- 3-gram 합산은 `kernel latency 합 + Σ 쌍 효과 + Σ 가운데 kernel 효과`와 같은 값이다.
- 2-gram 합산은 `kernel latency 합 + Σ 쌍 효과`다.
- 그래서 3-gram 합산이 2-gram 합산보다 나은 몫은 가운데 kernel 효과다.
- 앞에서부터 더하든 뒤에서부터 더하든 값은 같다. "마지막 kernel의 latency를 예측한다"는 식으로 설명하지 않는다.

## 4. 측정과 예측

| 용어 | 뜻 |
|---|---|
| **측정 latency** | TPU에서 실제로 잰 값 |
| **예측 latency** | model로 계산한 값. 어느 module이 냈는지 같이 쓴다 |
| **Module 1** | kernel 종류와 shape → 예측 kernel latency |
| **Module 2** | kernel 순서와 kernel latency 등 → 예측 2-gram·3-gram latency (3-gram 비율을 예측한 뒤 kernel latency 합을 곱한다) |
| **shape만으로 예측** | Module 1 → Module 2 → 3-gram 합산. 측정 없이 shape만 넣은 결과 |
| **측정 n-gram으로 계산한 값** | 2-gram·3-gram latency를 측정해서 합산한 값. 예측 n-gram을 써서 도달할 수 있는 기준이다 |
| **오차** | 각 조건의 `|예측 / 측정 − 1|`을 평균한 %. 최대 오차를 같이 쓴다 |

평가 방식:

| 용어 | 뜻 |
|---|---|
| **안 본 model 크기** | 그 크기를 빼고 학습하고, 그 크기로 평가 |
| **안 본 seq** | 그 seq를 빼고 학습하고, 그 seq로 평가 |
| **안 본 조합** | 크기와 seq를 동시에 빼고 학습 |
| **안 본 순서** | 학습에 없던 kernel 순서로 평가 |

Module 2의 입력 집합:

| 용어 | 내용 |
|---|---|
| **입력 1** | kernel 순서 + kernel latency |
| **입력 2** | 입력 1 + kernel별 flops, weight bytes (attn은 score bytes 추가) |
| **입력 3** | 입력 2 + kernel 사이 연결 정보, 출력 bytes |
| **입력 4** | 입력 3 + shape 자체 |

## 5. 측정 규칙에 쓰는 용어

| 용어 | 뜻 |
|---|---|
| **TPU 실행 시간** | profiler trace에서 늦게 시작한 core의 시작부터 끝까지. 30회 반복의 최솟값을 쓴다 (`sync_min_us`) |
| **입력 번갈아 쓰기** | 입력 사본 2개를 교대로 넣는 측정 (`copies2`). 주 결과로 쓴다 |
| **같은 입력 반복** | 같은 buffer를 반복해 넣는 측정 (`copies1`). weight가 chip 안 memory에 미리 올라가 실제보다 빠르게 나온다. 비교용으로만 쓴다 |
| **model 크기** | `(D, INTER, NH, HD, KV)` 다섯 숫자. 1B = (2048, 8192, 32, 64, 8), 3B = (3072, 8192, 24, 128, 8), 8B = (4096, 14336, 32, 128, 8) |

## 6. model 단위 용어 (layer를 쌓을 때)

| 용어 | 식 | 뜻 |
|---|---|---|
| **N layer latency** | `lat(N layer)` | layer N개를 쌓아 한 번에 돌린 시간. embedding, lm_head는 없다 |
| **추가 layer latency** | `lat(N layer) − lat(N−1 layer)` | layer 하나를 더 쌓을 때 늘어나는 시간 |
| **추가 layer 비율** | `(lat(2 layer) − lat(1 layer)) / lat(1 layer)` | 추가 layer latency가 layer latency의 몇 배인지. 1보다 작으면 layer 사이 overlap이 있다 |
| **layer 사이 overlap** | — | 다음 layer의 weight load 등이 앞 layer 계산 뒤에 숨는 것 |
| **embedding latency** | — | token id로 embedding을 꺼내는 시간 |
| **head latency** | — | 마지막 norm + lm_head (마지막 token 1개) |
| **model latency** | — | prefill 전체: embedding → layer n개 → 마지막 norm → lm_head |

## 7. 쓰지 않는 표기와 대체

예전 문서와 code에는 아래 표기가 남아 있다. 새로 쓰는 문서에서는 오른쪽을 쓴다.

| 쓰지 않는 표기 | 대체 |
|---|---|
| token | kernel |
| window | n-gram (2-gram, 3-gram) |
| `U` | kernel latency |
| `W2`, `W3` | 2-gram latency, 3-gram latency |
| `full` | layer latency (덩어리 전체를 잰 경우는 "덩어리 latency") |
| `uni`, unigram 합 | kernel latency 합 |
| `tele2`, `tele3` | 2-gram 합산, 3-gram 합산 |
| `tele3_hat` | 예측 n-gram으로 한 3-gram 합산 |
| telescope, telescoping | n-gram 합산 |
| `g`, `t` | 쌍 효과, 가운데 kernel 효과 |
| `r`, `r3` | 3-gram 비율 |
| E1, E2, E3 | 안 본 model 크기, 안 본 seq, 안 본 조합 |
| S0, S1, S2, S3 | 입력 1, 입력 2, 입력 3, 입력 4 |
| `sync_min`, sync | TPU 실행 시간 |
| `copies2`, 사본 순환 | 입력 번갈아 쓰기 |
| `copies1` | 같은 입력 반복 |
| 상한 | 측정 n-gram으로 계산한 값 |
| L16, L20 | kernel 16개 layer, kernel 20개 layer(attention을 5개로 쪼갠 것) |
| chain | kernel 순서 |
| 덩어리 A, B | layer 앞부분(norm1~v), layer 뒷부분(o~add2) |
| `L(N)` | N layer latency |
| marginal | 추가 layer latency |
| `ρ`, rho | 추가 layer 비율 |
| cross-layer | layer 사이 |
| full model, e2e | model latency |
| LPRED, `l1_pred` | 예측 layer latency (shape만으로 예측) |
