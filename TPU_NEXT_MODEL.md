# TPU 다음 단계 B — layer 사이 겹침과 full model prefill 예측

작성일 2026-11-26. 대상: **Module 1을 맡은 VM** (`~/venvs/jaxtpu`). 같은 시기에 Module 2 VM은 [`TPU_NEXT_LAYER.md`](TPU_NEXT_LAYER.md)를 한다.

먼저 읽을 것:

- `profiling/tpu_v4/module2/README.md`의 측정 규약
- `memory/2026-11-26-tpu-m1.md` (특히 2차 범위의 lm_head·gemv)
- `TPU_INSTRUCTIONS.md` Step 5~6 (이 문서가 그 둘을 구체화한다. 다르면 이 문서를 따른다)

## 1. 왜 하나

- Module 2 VM이 layer 1개를 예측해도 model latency는 바로 나오지 않는다. layer를 쌓으면 다음 layer의 weight load가 앞 layer 계산 뒤에 숨는 식의 **layer 사이 겹침**이 생긴다.
- RNGD 결과: 겹침은 바로 앞 layer까지만 닿았다. marginal이 N=2부터 일정했고(1층 latency의 0.86배), `L(1) + (N−1)·marginal`로 16층을 0.96배로 맞췄다. 단 attention 제외, seq 512 하나였다.
- 이 문서의 질문은 두 가지다.
    1. TPU에서 layer 사이 겹침은 몇 층까지 닿고, 크기·seq에 따라 어떻게 변하나?
    2. layer 1개 예측값으로 full model prefill(embedding + N layer + lm_head)을 몇 %로 맞추나?

## 2. 공통 코드: `profiling/tpu_v4/layer.py`

| 함수 | 내용 |
|---|---|
| `make_layer` | attention을 포함한 layer 1개 (Module 2 VM의 L16 full window와 같은 계산) |
| `make_stack(..., n_layers, mode)` | layer만 N개 쌓은 것. layer 사이 측정용 |
| `make_model(..., n_layers, vocab, mode)` | embedding → N layer → final rmsnorm → lm_head(마지막 token) |
| `stack_args`, `model_args` | random bf16 인자. layer마다 다른 seed라 weight buffer가 전부 다르다 |

- `mode="unroll"`: Python loop로 펼친 한 program. XLA가 layer 경계를 넘어 schedule할 수 있다.
- `mode="scan"`: `lax.scan` (MaxText의 `scan_layers` 방식). loop 한 바퀴씩 돈다.
- **JAX CPU backend에서만 확인했다.** unroll과 scan의 출력이 같고, layer를 하나씩 부른 값과도 같다. TPU에서는 아직 돌리지 않았다.

## 3. 측정 규약

- 값은 `sync_min_us`다 (`module2/README.md`).
- **weight는 사본을 만들지 않는다.** 8B 32층이면 weight가 약 16 GB라 사본 2개는 chip 하나(32 GiB)에 넣기 빠듯하다. layer마다 이미 다른 buffer다.
- 입력 `x`(또는 `tokens`)만 사본 2개를 돌려 쓴다.
- 같은 weight를 반복 호출하면 첫 weight 하나가 cross-program prefetch로 CMEM에 남는 artifact가 있을 수 있다. Step 1에서 N=1에 한해 weight까지 사본 2개를 돌려 쓴 값과 비교해 그 크기를 기록한다.

## 4. 단계

### Step 1. 확인

1. CPU 작은 크기에서 `layer.py`를 한 번 돌린다.
2. TPU에서 1B `make_layer` 1개의 값을 잰다. Module 2 VM이 push한 L16 full window와 noise floor 안에서 같아야 한다. 아직 push되지 않았으면 이 비교는 나중에 한다.
3. 8B 32층 `make_model`이 unroll·scan 각각 compile되고 chip 하나에 올라가는지 확인한다. compile 시간과 HBM 사용량을 기록한다.
4. 위 §3의 weight 사본 artifact 크기를 기록한다.

### Step 2. `L(N)` 측정

`make_stack`, mode unroll·scan 둘 다.

| model | dims `(D, INTER, NH, HD, KV)` | 실제 층 수 n | N |
|---|---|---|---|
| 1B | (2048, 8192, 32, 64, 8) | 16 | 1, 2, 3, 4, 8, 16 |
| 3B | (3072, 8192, 24, 128, 8) | 28 | 1, 2, 3, 4, 8, 16, 28 |
| 8B | (4096, 14336, 32, 128, 8) | 32 | 1, 2, 3, 4, 8, 16, 32 |

seq {128, 512, 1024, 2048}.

보고:

- `L(N)` 표
- 구간 marginal `(L(b) − L(a)) / (b − a)`가 몇 층부터 일정해지는지
- `ρ = (L(2) − L(1)) / L(1)`
- trace에서 다음 layer의 weight load가 앞 layer의 어느 부분과 겹치는지 (attention인지, MLP인지)
- unroll과 scan의 차이

### Step 3. 조립 예측

```
L_hat(N) = L(1) + (N − 1) · (L(2) − L(1))
```

N = 8, 16, n에서 실측 `L(N)`과 비교한다. marginal이 N=k부터 일정하다면 `L(k) + (N − k) · m(k)`도 같이 낸다.

### Step 4. `ρ` 일반화

크기 10종(Module 2와 같은 목록) × seq 4종에서 N ∈ {1, 2, 4}를 unroll로 잰다. Step 2에서 scan이 unroll과 크게 다르면 scan도 잰다.

`ρ`를 다음 두 가지로 예측하고, 크기 1종을 빼고 학습하는 방식으로 평가한다.

- 상수 (다른 크기의 평균)
- `L(1)`, layer당 weight bytes, layer당 flops, seq를 feature로 쓴 선형 model

→ `profiling/tpu_v4/model/data_rho_tpu_v4.txt`

### Step 5. full model prefill

`make_model`, 1B·3B·8B, seq 4종, unroll·scan.

다음도 따로 잰다.

- embedding (`embed[tokens]`)
- final rmsnorm + lm_head (마지막 token 1개, `M=1`). Module 1 2차 범위에서 vocab 128256이 gemv 대역폭 cliff에 걸렸으므로 lm_head는 **측정값**을 쓴다.

보고 (예측 / 실측):

| 방식 | 식 |
|---|---|
| 단순 합 | `emb + n · L(1) + head` |
| 조립 | `emb + L_hat(n) + head` |
| 실측 `L(n)` 사용 | `emb + L(n) + head` (embedding·head 경계 효과만 남는다) |

### Step 6. 완전 bottom-up

Module 2 VM이 `module2/data_layerpred_tpu_v4.txt`(`l1_pred_us`)를 push하면 다음을 계산해 Step 5의 실측과 비교한다.

```
full_hat = emb + l1_pred · (1 + (n − 1) · ρ_hat) + head
```

- `ρ_hat`은 Step 4에서 그 크기를 뺀 model로 예측한 값이다.
- emb와 head는 측정값이다.
- 파일이 아직 없으면 Step 1~5를 먼저 push하고, `git pull`로 확인한다.

## 5. 산출물

- code·data: `profiling/tpu_v4/model/`
- data 행:

  ```
  LROW D INTER NH HD KV n_layers seq mode sync_min_us sync_med_us wall_min_us n_samples
  MROW D INTER NH HD KV n_layers vocab seq mode part sync_min_us sync_med_us wall_min_us n_samples
  ```

  `part`는 `full`, `emb`, `head` 중 하나다.
- 기록: `memory/YYYY-MM-DD-tpu-model.md`에 Step별 표, 판정, caveat.
- 측정에 실패한 행도 지우지 않는다.

## 6. 최소 결과와 순서

- 논문 마감은 2026-12-08 (AoE)이다.
- 최소 결과: 1B·8B, seq {512, 2048}, unroll로 한 Step 2·3, 같은 조건의 Step 5.
- 그 다음 3B, seq {128, 1024}, scan, Step 4·6을 채운다.
- Step마다 commit·push한다. push 전에 `git pull --rebase`.

## 7. 하지 않는 것

decode, batch > 1, multi-chip(TP), 실제 serving stack(MaxText·vLLM)과의 비교는 하지 않는다. 필요하면 이 문서가 끝난 뒤 따로 정한다.
