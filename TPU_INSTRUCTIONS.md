# TPU v4 n-gram latency 예측 검증 지시서

> 이 문서는 [`TERMS.md`](TERMS.md) 이전에 썼고 이미 끝난 작업이다. 예전 표기(token, window, `U`, `W3`, telescope 등)는 `TERMS.md` §7의 대응표로 읽는다. 새 글에는 `TERMS.md` 용어를 쓴다.

작성일 2026-11-26. 대상: TPU v4 VM에서 이 repo를 받아 실험을 수행하는 세션(사람 또는 agent).

## 1. 목적

RNGD에서 성립한 n-gram(telescoping) latency 예측 방식이 TPU v4에서도 성립하는지 **확정**한다.

"확정"은 §5의 판정 기준에 따라 단계마다 성립/불성립을 결론 내리는 것을 뜻한다. 정확도를 끌어올리는 작업(ML 튜닝 등)은 이 지시서의 범위가 아니다.

## 2. RNGD에서 확정된 것 (재현 대상)

방법: Llama layer의 op chain `[q, k, v, o, gate, up, down]`에서 연속 n개 op 묶음(window)을 따로 compile해 측정하고, 겹치는 부분을 빼며 이어붙여(telescoping) 전체 latency를 예측한다.

| 항목 | RNGD 결과 | 범위 |
|---|---|---|
| unigram 합 / full | 1.23~1.28x | Llama 1B 크기, seq 512·1024 |
| 2-gram telescope / full | 0.97~1.02x | 위와 동일 |
| 3-gram telescope, 크기 일반화 | MAPE 3.1% | seq 512, 크기 8종 |
| 위 + 상수 보정 (leave-one-config-out) | MAPE 1.4% | 위와 동일 |
| N-layer 조립 `L(1) + (N−1)·marginal` | 0.96x | 1B 16층, seq 512, attention 제외 |

- 크기 일반화의 "ML 보정 2.3%"는 `memory/2026-07-27.md`에 기록돼 있으나 fitting script가 남아 있지 않아 재현하지 못했다. 같은 data에서 상수 보정만으로 1.4%가 나온다 (2026-11-26 재계산).
- RNGD에서 하지 못한 것: attention 포함 layer, seq 축 일반화, 8B full model, 예측과 GT의 SW stack 통일(2026.3.0과 2026.2.0의 version 차이가 섞였다). TPU에서는 이 네 가지를 같이 확인한다.

근거: `memory/2026-07-23.md` §11, §15~17, `memory/2026-07-27.md`, `profiling/production_profile/data_sizesweep_s512.txt`.

## 3. 정의

op 순서를 `op_0 .. op_{n-1}` (n=7)이라 하고, `U(i)` = op_i 단독, `W2(i)` = (op_i, op_{i+1}), `W3(i)` = (op_i, op_{i+1}, op_{i+2}), `full` = 전체 n개 window의 latency라 한다.

```
uni   = Σ_{i=0}^{n-1} U(i)
tele2 = W2(0) + Σ_{i=1}^{n-2} [ W2(i) − U(i)  ]
tele3 = W3(0) + Σ_{i=1}^{n-3} [ W3(i) − W2(i) ]
```

- 지표: `pred/full`, 오차 = `|pred/full − 1|`.
- window 수: U 7 + W2 6 + W3 5 + full 1 = (크기, seq)마다 19개.

## 4. TPU로 옮기는 방법

### 4.1 RNGD와 TPU의 대응

| RNGD | TPU v4 |
|---|---|
| window마다 격리 compile (`CompileModule`) | window마다 별도 `jax.jit` 함수 (XLA가 함수 단위로 compile·fusion) |
| device-cycle (`cycle_actual`) | host timing(µs) + `block_until_ready()`. 필요하면 XProf trace |
| TP=8 (`set_fusion(8)`) | chip 1개(device 0), 분산 없음 |
| full·GT가 다른 compiler version | full도 같은 JAX 코드라 version 차이 없음 |

단위와 TP가 다르므로 RNGD 수치와 절대값은 비교하지 않는다. 비교 대상은 비율(`pred/full`)이다.

### 4.2 window 구성 규칙

- 구조는 `profiling/production_profile/ngram_telescope_chained_30.py`의 `WChain`을 그대로 JAX로 옮긴다. 실제 data flow를 보존한다.
  - `q, k, v ← rmsnorm(x)` (q, k에는 RoPE)
  - `o ← attn_out`
  - `gate, up ← rmsnorm(x + o)`
  - `down ← silu(gate) · up`
  - window가 chain 중간에서 시작하면 진입 지점의 값만 외부 dummy 입력으로 준다.
- dtype은 bf16, weight는 random으로 둔다 (latency는 값과 무관).
- weight와 입력은 jit 함수의 **인자**로 넘긴다. closure로 잡으면 큰 상수가 graph에 박혀 compile이 느려진다.
- window 안 모든 op의 출력을 반환한다. 반환하지 않은 연산은 XLA가 지운다.
- 처음 한 번은 작은 크기에서 CPU 계산값과 출력을 비교해 window가 의도한 연산을 하는지 확인한다.

### 4.3 측정 규칙

- compile(첫 호출)과 warmup은 제외한다. 매 호출 뒤 `block_until_ready()`로 device 실행이 끝나기를 기다린다.
- window당 30회 이상 반복하고 min과 median을 둘 다 기록한다.
- **floor를 따로 잰다.** 입력을 그대로 돌려주는 jit 함수의 호출 시간이 호출당 host overhead `F`다. RNGD host timing에는 op당 약 40µs의 floor가 있었다.
  - window당 상수 floor는 telescope에서 상쇄된다 (`tele2`, `tele3`, `full` 모두 floor가 1개씩 남는다).
  - `uni`에는 floor가 n개 쌓인다. 그래서 결과 표는 raw 값으로 만든 것과 모든 window에서 `F`를 뺀 뒤 만든 것 두 개를 낸다. 그러지 않으면 unigram이 실제보다 나쁘게 보인다.
- RNGD에서는 한 process에서 오래 측정하면 값이 망가져서 window마다 fresh process를 썼다. TPU에서도 필요한지는 Step 1에서 확인한다.

## 5. 실험 순서와 판정

Step이 끝날 때마다 결과를 commit·push한다 (VM이 사라져도 결과가 남도록). Step 2가 핵심이고, Step 2를 통과해야 Step 3 이후로 간다.

### Step 0. 환경 기록

- TPU type·topology, `jax.device_count()`, jax·libtpu version, Python version을 기록한다.
- v4 기준 예상값: chip당 HBM 32GiB, `v4-8`은 chip 4개, `jax.device_count()`는 4. 다르면 기록하고 계속한다.
- 측정에는 device 1개만 쓴다.
- TPU를 켜기 전에 script를 JAX CPU backend로 작은 크기에서 돌려 버그를 먼저 잡는다.

### Step 1. 측정 신뢰성 (gate)

크기 1B `(D, INTER, NH, HD, KV) = (2048, 8192, 32, 64, 8)`, seq 512.

1. floor `F`를 측정한다.
2. `full`과 가장 작은 unigram(`v`)을 각각 fresh process로 5번 측정해 process 간 변동을 본다.
3. 같은 두 window를 한 process에서 연달아 측정한 값과 비교해 fresh process가 필요한지 정한다.

통과 조건: `full`의 process 간 변동이 2% 이하이고, 가장 작은 unigram이 `F`와 구분된다 (`F`를 뺀 값이 반복 측정에서 안정적이다).

실패하면 순서대로 시도한다.

1. seq를 2048 이상으로 키운다.
2. jit 안에서 window를 여러 번 실행해 호출당 overhead를 나눈다. iteration 사이에 data 의존을 둬서 연산이 loop 밖으로 빠지지 않게 하고, 모든 window에 같은 wrapper를 쓴다.
3. `jax.profiler` trace와 XProf로 device 시간을 읽는다.

셋 다 안 되면 중단하고 보고한다.

### Step 2. layer 안 telescope (핵심)

크기 2종 × seq {128, 512, 1024, 2048} × window 19개.

- 1B: `(2048, 8192, 32, 64, 8)`
- 8B: `(4096, 14336, 32, 128, 8)`

보고: (크기, seq)마다 `uni/full`, `tele2/full`, `tele3/full`. raw 표와 `F`를 뺀 표 두 개.

판정 (오차는 Step 1의 측정 변동보다 클 때만 의미가 있다):

| 결과 | 결론 |
|---|---|
| `tele2` 또는 `tele3` 오차 ≤ 5%이고 unigram 오차보다 작다 | n-gram 성립 (RNGD와 같은 결론) |
| unigram 합 오차 ≤ 5% | 가산성 성립. TPU에서는 op 사이 overlap이 작아 n-gram 없이 kernel 합으로 충분하다 |
| 둘 다 5~10% | 보류. 반복 측정으로 noise인지 확인한다 |
| 둘 다 > 10% | 4-gram까지 창을 넓혀 한 번 더 본다. 그래도 > 10%면 불성립으로 보고하고 중단한다 |

두 번째 줄도 유효한 결론이다. n-gram이 필요하다는 쪽으로 무리하게 해석하지 않는다.

추가 확인: `full`에서 창 밖으로 나가는 출력(`q, k, v, down`)만 반환한 값을 같이 잰다. 모든 출력을 반환한 값과 3% 이상 다르면 XLA가 중간값을 materialize하는지에 latency가 민감하다는 뜻이다. 이 경우 차이를 기록하고, Step 4 이후는 실제 model과 같은 반환 방식(창 밖으로 나가는 값만)으로 통일한다.

### Step 3. 크기·seq 일반화

- 크기 10종 (RNGD와 동일, `D,INTER,NH,HD,KV`):
  `1024,2816,16,64,4` / `1536,4096,24,64,6` / `2048,5504,16,128,4` / `2048,8192,32,64,8` / `2560,6912,20,128,4` / `3072,8192,24,128,8` / `3584,18944,28,128,4` / `4096,11008,32,128,8` / `4096,14336,32,128,8` / `5120,13824,40,128,8`
- seq {512, 2048}. RNGD는 512만 했다.
- 보고: 크기별 correction(`full/tele`), 그리고 leave-one-config-out MAPE 세 가지.
  - (a) telescope만
  - (b) 상수 보정: 나머지 config의 평균 correction을 곱한다
  - (c) feature 기반 ML 보정
- (c)가 (b)보다 낫지 않으면 "ML 보정 불필요"로 결론 낸다. RNGD는 (a) 3.1%, (b) 1.4%였다.
- fitting script를 반드시 commit한다.

### Step 4. attention 포함 layer

chain을 `[q, k, v, attn, o, gate, up, down]` 8개로 늘린다. `attn`은 causal mask와 GQA를 쓰는 scaled dot-product attention이고, `o`의 입력은 실제 `attn` 출력이다. 같은 방법으로 U/W2/W3/full을 재고 Step 2와 같은 표와 판정을 낸다. 1B·8B 크기, seq {512, 2048}.

### Step 5. layer 사이 (cross-layer)

실제 layer(attention 포함, layer마다 별도 weight)를 N개 이어붙여 `L(N)`을 잰다. N = 1, 2, 3, 4, 8, 16이고 8B 크기는 32까지 잰다.

- marginal(구간 평균 `(L(b) − L(a)) / (b − a)`)이 몇 층에서 일정해지는지 본다. RNGD에서는 N=2부터 일정했고 값은 `0.86·L(1)`이었다.
- 예측 `L(1) + (N−1)·(L(2) − L(1))`과 실측 `L(16)`, `L(32)`를 비교한다.
- `L(1)` 자리에 Step 4의 telescope 예측값을 넣은 완전 bottom-up 예측도 같이 낸다.

### Step 6. full model prefill

같은 JAX 코드로 embedding + N layers + final norm + lm_head(마지막 token)까지 만든 prefill forward를 잰다. weight는 random이다.

- 1B: 16층, D 2048, vocab 128256
- 8B: 32층, D 4096, vocab 128256
- seq {128, 512, 1024, 2048}
- 예측 = Step 5 조립값 + embedding·lm_head 단독 측정값. 실측과 비교한다.
- 8B는 bf16 weight가 약 16GB다. chip 하나(32GiB)에 올라갈 것으로 보지만, 안 올라가면 layer 수를 줄여 측정하고 그 사실을 기록한다.

## 6. 산출물

- code: `profiling/tpu_v4/`에 측정 script와 분석(fitting) script.
- data: `profiling/tpu_v4/data_*.txt`. RNGD와 같은 self-describing 행 형식에 단위를 붙인다. window 이름은 `q_k_v`처럼 `_`로 잇는다.
  `ROW D INTER NH HD KV window seq min_us median_us`
- 측정에 실패한 행도 지우지 않고 남긴다. 분석에서 제외한 data는 이유와 함께 적는다.
- 기록: `memory/YYYY-MM-DD.md`에 Step별 결과 표, 판정, 측정 방법의 caveat.
- commit은 Step 단위로 master에 push한다.

## 7. 범위 밖

decode, multi-chip 분산(TP), batch > 1, 실제 serving stack에서의 GT, 정확도 개선용 ML 튜닝. 필요하면 Step 6까지 끝난 뒤 따로 정한다.

## 8. 참고

- RNGD window 구성: `profiling/production_profile/ngram_telescope_chained_30.py`
- RNGD N-layer 측정: `profiling/production_profile/nlayer_tw_30.py`
- TPU v4 사양: https://docs.cloud.google.com/tpu/docs/v4
- Cloud TPU VM에서 JAX 설치·실행: https://docs.cloud.google.com/tpu/docs/run-calculation-jax
- JAX profiling(XProf): https://docs.jax.dev/en/latest/profiling.html
