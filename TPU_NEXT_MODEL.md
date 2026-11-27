# TPU 다음 단계 B — layer를 쌓았을 때와 model 전체 latency 예측

작성 2026-11-26, 용어 정리 2026-11-27. 대상: **Module 1을 맡은 VM** (`~/venvs/jaxtpu`).

**먼저 [`TERMS.md`](TERMS.md)를 읽는다.** 이 문서와 이 작업의 기록·보고는 그 용어만 쓴다.

그 다음 읽을 것:

- `profiling/tpu_v4/module2/README.md`: 측정 규칙
- `memory/2026-11-27-tpu-layer.md`: Module 2 VM이 만든 예측 layer latency
- `memory/2026-11-26-tpu-m1.md`의 "2차 범위": lm_head 측정

## 1. 무엇을 알아내려는가

지금까지 결과는 다음과 같다 (TPU v4, model 크기 10종 × seq 4종).

| 맞힌 대상 | 방식 | 오차 |
|---|---|---|
| layer latency (attention 포함) | kernel latency 합 | 31.0% |
| layer latency | 측정 n-gram으로 계산한 3-gram 합산 | 6.2% |
| **layer latency** | **shape만으로 예측** | **7.2%** |

실제로 알고 싶은 것은 **model latency**(prefill 전체)다. model은 layer를 n개 쌓고 앞뒤에 embedding과 lm_head를 붙인 것이다. 그런데 layer를 쌓으면 다음 layer의 weight load가 앞 layer 계산 뒤에 숨는다(layer 사이 overlap). 그래서 layer latency × n은 model latency보다 크게 나온다.

이 문서의 질문은 두 가지다.

1. layer 사이 overlap은 몇 layer까지 닿고, model 크기·seq에 따라 어떻게 변하나?
2. 예측 layer latency 하나로 model latency를 몇 % 오차로 맞추나?

**RNGD에서 본 것** (참고, attention 제외, seq 512 하나): 추가 layer latency가 2 layer부터 일정했고, 그 값은 layer latency의 0.86배였다. `lat(1 layer) + 15 × 추가 layer latency`로 16 layer를 0.96배로 맞췄다.

## 2. 쓰는 식

```
lat(N layer) ≈ lat(1 layer) + (N − 1) × 추가 layer latency
             = lat(1 layer) × (1 + (N − 1) × 추가 layer 비율)

model latency ≈ embedding latency + lat(n layer) + head latency
```

**숫자 예** (값은 설명용 가짜 숫자): layer latency가 300 µs이고 추가 layer 비율이 0.86이면, 16 layer는 `300 × (1 + 15 × 0.86) = 4170 µs`로 예측한다. layer latency × 16 = 4800 µs보다 작다.

이 식이 맞으려면 **추가 layer latency가 N에 대해 일정해야** 한다. 그걸 확인하는 것이 Step 2다.

## 3. 공통 code: `profiling/tpu_v4/layer.py`

| 함수 | 만드는 것 |
|---|---|
| `make_layer` | layer 1개 (kernel 16개). Module 2 VM의 layer latency 측정과 같은 계산이다 (그 VM이 출력이 같음을 확인했다) |
| `make_stack(..., n_layers, mode)` | layer N개만 쌓은 것 → N layer latency 측정용 |
| `make_model(..., n_layers, vocab, mode)` | embedding → layer n개 → 마지막 norm → lm_head(마지막 token) → model latency 측정용 |
| `stack_args`, `model_args` | random bf16 인자. layer마다 다른 weight buffer |

`mode`는 두 가지다.

- `"unroll"`: layer를 Python loop로 펼쳐 한 프로그램으로 만든다. XLA가 layer 경계를 넘어 실행 순서를 정할 수 있다.
- `"scan"`: `lax.scan`으로 layer를 한 바퀴씩 돈다. MaxText의 `scan_layers` 방식이다.

이 code는 JAX CPU backend에서만 확인했다. TPU에서는 아직 돌리지 않았다.

## 4. 측정 규칙

- 값은 **TPU 실행 시간**이다 (`TERMS.md` §5).
- **weight는 사본을 만들지 않는다.** 8B 32 layer면 weight가 약 16 GB라 사본 2개는 chip 하나(32 GiB)에 넣기 빠듯하다. layer마다 이미 다른 buffer다.
- **입력**(`x` 또는 token id)만 입력 번갈아 쓰기로 넣는다.
- 같은 weight를 반복 호출하면 첫 weight 하나가 chip 안 memory에 미리 올라가 있을 수 있다. Step 1에서 1 layer에 한해 weight까지 사본 2개를 번갈아 쓴 값과 비교해 그 차이를 기록한다.

## 5. 단계

### Step 1. 확인

1. CPU 작은 크기에서 `layer.py`를 한 번 돌린다.
2. TPU에서 1B `make_layer`의 layer latency를 잰다. Module 2 VM의 측정값(`module2/data_l16_copies2_tpu_v4.txt`의 전체 layer 행)과 noise 안에서 같아야 한다.
3. 8B 32 layer `make_model`이 unroll·scan 각각 compile되고 chip 하나에 올라가는지 확인한다. compile 시간과 HBM 사용량을 기록한다.
4. 위 §4의 weight 사본 차이를 기록한다.

### Step 2. N layer latency 측정

`make_stack`을 unroll·scan 둘 다로 잰다.

| model | 크기 `(D, INTER, NH, HD, KV)` | 실제 layer 수 n | 잴 N |
|---|---|---|---|
| 1B | (2048, 8192, 32, 64, 8) | 16 | 1, 2, 3, 4, 8, 16 |
| 3B | (3072, 8192, 24, 128, 8) | 28 | 1, 2, 3, 4, 8, 16, 28 |
| 8B | (4096, 14336, 32, 128, 8) | 32 | 1, 2, 3, 4, 8, 16, 32 |

seq는 {128, 512, 1024, 2048}이다.

보고:

- N layer latency 표
- 추가 layer latency: 구간마다 `(lat(b layer) − lat(a layer)) / (b − a)`. 몇 layer부터 일정해지는가
- 추가 layer 비율
- trace에서 다음 layer의 weight load가 앞 layer의 어느 kernel과 겹치는가 (attn인지, MLP gemm인지)
- unroll과 scan의 차이

### Step 3. 쌓은 latency 예측

§2의 식으로 N = 8, 16, n의 N layer latency를 예측하고 측정과 비교한다.

추가 layer latency가 2 layer가 아니라 k layer부터 일정해진다면 다음 식도 같이 낸다.

```
lat(N layer) ≈ lat(k layer) + (N − k) × 추가 layer latency(k 이후)
```

### Step 4. 추가 layer 비율을 안 본 model 크기로 일반화

model 크기 10종(Module 2와 같은 목록) × seq 4종에서 N ∈ {1, 2, 4}를 unroll로 잰다. Step 2에서 scan이 unroll과 크게 달랐다면 scan도 잰다.

추가 layer 비율을 두 방식으로 예측하고 **안 본 model 크기**로 평가한다.

- 상수: 다른 크기들의 평균
- 선형 model: layer latency, layer당 weight bytes, layer당 flops, seq를 입력으로

→ `profiling/tpu_v4/model/data_rho_tpu_v4.txt`

### Step 5. model latency

`make_model`로 1B·3B·8B, seq 4종, unroll·scan을 잰다. 부분도 따로 잰다.

- embedding latency
- head latency. vocab 128256에서 lm_head 대역폭이 절반으로 떨어지는 구간이 있어(Module 1 2차 범위) **측정값**을 쓴다.

보고 (예측 / 측정 model latency):

| 방식 | 식 |
|---|---|
| layer latency × n | `embedding + n × lat(1 layer) + head` |
| 쌓은 latency 예측 | `embedding + (§2 식의 lat(n layer)) + head` |
| 측정 N layer latency 사용 | `embedding + 측정 lat(n layer) + head` (embedding·head 경계 효과만 남는다) |

### Step 6. shape만으로 model latency 예측

Module 2 VM이 push한 **예측 layer latency** 파일 `module2/data_layerpred_tpu_v4.txt`를 쓴다.

- 이 파일의 `l1_pred_us`가 shape만으로 예측한 layer latency다.
- `l1_meas_us`는 측정 layer latency다.

```
예측 model latency = embedding latency + l1_pred × (1 + (n − 1) × 예측 추가 layer 비율) + head latency
```

- 예측 추가 layer 비율은 Step 4에서 그 model 크기를 빼고 학습한 값이다.
- embedding·head latency는 측정값이다.

Step 5의 측정 model latency와 비교해 오차를 낸다.

## 6. 산출물

- code·data: `profiling/tpu_v4/model/`
- data 행 (code 형식은 이 이름을 유지한다):

  ```
  LROW D INTER NH HD KV n_layers seq mode sync_min_us sync_med_us wall_min_us n_samples
  MROW D INTER NH HD KV n_layers vocab seq mode part sync_min_us sync_med_us wall_min_us n_samples
  ```

  - `LROW`: N layer latency
  - `MROW`: model latency. `part`는 `full`(model 전체), `emb`(embedding), `head` 중 하나
- 기록: `memory/YYYY-MM-DD-tpu-model.md`에 Step별 결론, 표, 주의점. `TERMS.md` 용어로 쓴다.
- 측정에 실패한 행도 지우지 않는다.

## 7. 순서와 최소 결과

- 논문 마감은 2026-12-08 (AoE)이다.
- **최소 결과**: 1B·8B, seq {512, 2048}, unroll로 한 Step 2·3, 같은 조건의 Step 5.
- 그 다음 3B, seq {128, 1024}, scan, Step 4·6.
- Step마다 commit·push한다 (`CLAUDE.md`의 git 규칙).

## 8. 하지 않는 것

decode, batch > 1, chip 여러 개(TP), 실제 serving stack(MaxText·vLLM)과의 비교. 필요하면 이 문서가 끝난 뒤 따로 정한다.
