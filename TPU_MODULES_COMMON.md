# TPU module 공통 규칙 (Module 1·2)

작성일 2026-11-26. 대상: TPU v4 VM 두 대에서 각각 Module 1, Module 2를 맡는 세션(사람 또는 agent).

자기 module 지시서보다 이 문서를 먼저 읽는다.

- Module 1: [`TPU_MODULE1_KERNEL.md`](TPU_MODULE1_KERNEL.md) — shape로 kernel 단독 latency를 예측한다.
- Module 2: [`TPU_MODULE2_NGRAM.md`](TPU_MODULE2_NGRAM.md) — kernel 순서와 kernel별 값으로 3-gram window latency를 예측한다.

## 1. 전체 구조

목표는 측정하지 않은 model 크기와 seq에서 LLM prefill latency를 예측하는 것이다. 예측은 두 단계로 나뉜다.

```
kernel 종류 + shape      --[Module 1]-->  kernel 단독 latency U
kernel 순서 + U (+ 추가)  --[Module 2]-->  window latency W2, W3  --telescope-->  layer latency
```

- Module 1이 필요한 이유: n-gram은 kernel별 `U`를 재료로 쓰는데, 모든 shape를 측정할 수는 없다.
- Module 2가 필요한 이유: window(`W2`, `W3`)를 (크기, seq)마다 전부 측정하면 측정량이 kernel 측정보다 더 많아진다. window를 측정하지 않고 예측으로 채운다.
- 두 VM은 서로를 기다리지 않는다. Module 2는 자기가 측정한 `U`로 먼저 진행하고, Module 1의 예측값은 마지막 Step에서 바꿔 끼운다.

### `TPU_INSTRUCTIONS.md`와의 관계

`TPU_INSTRUCTIONS.md`는 **측정한** window로 telescope가 TPU에서 성립하는지 본다. 이 두 module은 그 window를 **예측**으로 대체한다. 용어(`U`, `W2`, `W3`, `full`, `uni`, `tele2`, `tele3`)와 범위 밖 항목은 그 문서 §3, §7을 그대로 따른다.

다음 세 가지는 그 문서와 다르게 정했고, 두 module에서는 이 문서가 우선한다.

| 항목 | `TPU_INSTRUCTIONS.md` | 두 module |
|---|---|---|
| token 단위 | op 묶음 7개 (`q` = rmsnorm + matmul + RoPE 등) | kernel 15개 (§2) |
| 측정값 | host timing에서 floor를 뺀 값 | profiler의 device time (§4) |
| window 반환값 | window 안 모든 op의 출력 | window 밖에서 쓰이는 값만 (§3) |

## 2. token 단위: kernel

Module 1이 shape로 예측하는 대상이 kernel이고, 그 출력이 그대로 Module 2의 input이 되어야 한다. 그래서 두 module 모두 kernel 하나를 token 하나로 본다.

kernel type은 6종이다.

| type | 연산 | shape 인자 |
|---|---|---|
| `gemm` | `x[M,K] @ w[K,N]` | M, K, N |
| `rmsnorm` | `x[M,N]`, gain `g[N]` | M, N |
| `rope` | `t[M, heads·HD]`에 회전 적용 | M, N(= heads·HD), HD |
| `silu` | `x[M,N]` | M, N |
| `mul` | `x[M,N] * y[M,N]` | M, N |
| `add` | `x[M,N] + y[M,N]` | M, N |

Llama layer에서 attention core를 뺀 부분(Tokenwise chain)은 token 15개다. `M`은 seq다.

| idx | token | type | data 입력 | 출력 shape |
|---|---|---|---|---|
| 0 | `norm1` | rmsnorm | `x` (외부) | [S, D] |
| 1 | `q` | gemm | `norm1` | [S, NH·HD] |
| 2 | `rope_q` | rope | `q` | [S, NH·HD] |
| 3 | `k` | gemm | `norm1` | [S, KV·HD] |
| 4 | `rope_k` | rope | `k` | [S, KV·HD] |
| 5 | `v` | gemm | `norm1` | [S, KV·HD] |
| 6 | `o` | gemm | `attn_out` (외부) | [S, D] |
| 7 | `add1` | add | `x`, `o` | [S, D] |
| 8 | `norm2` | rmsnorm | `add1` | [S, D] |
| 9 | `gate` | gemm | `norm2` | [S, INTER] |
| 10 | `silu` | silu | `gate` | [S, INTER] |
| 11 | `up` | gemm | `norm2` | [S, INTER] |
| 12 | `mul` | mul | `silu`, `up` | [S, INTER] |
| 13 | `down` | gemm | `mul` | [S, D] |
| 14 | `add2` | add | `add1`, `down` | [S, D] |

chain 전체가 내보내는 값은 `rope_q`, `rope_k`, `v`, `add2`다.

attention core(`QK^T`, scale, mask, softmax, `AV`)는 두 module 모두 2차 범위다. LENS의 TPU v6e·v5e 측정에서 softmax와 `AV`가 seq에 따라 다른 kernel(`online-softmax` custom-call)로 치환되는 것이 확인됐다. 이 치환은 overlap이 아니라 kernel 교체라서 따로 다뤄야 한다.

## 3. 공통 코드: `profiling/tpu_v4/kernels.py`

kernel 정의, 위 chain, window 생성 함수가 들어 있다. 두 module 모두 이 파일로 측정 대상 함수를 만든다.

```python
import kernels as kn
shp = kn.shapes(D=2048, INTER=8192, NH=32, HD=64, KV=8, S=512)
fn, arg_names, out_names = kn.window(9, 12)     # gate, silu, up
args = kn.make_args(arg_names, shp)
outs = jax.jit(fn)(*args)
```

- **이 파일은 JAX CPU backend에서만 실행을 확인했다.** TPU에서는 아직 돌리지 않았다. Step 1에서 TPU 실행을 확인한다.
- `U(token)`은 `kn.window(i, i+1)`이다. Module 1이 같은 kernel type을 같은 shape로 단독 측정한 값과 같아야 한다.
- window에 속한 token의 입력이 window 밖에서 오면 같은 shape의 dummy 값을 인자로 준다. weight도 전부 인자다.
- **window는 window 밖에서 쓰이는 값만 반환한다.** window 안에서만 쓰이는 중간값은 반환하지 않는다. 실제 model에서 materialize되지 않는 값을 강제로 materialize시키면 XLA의 fusion 결과가 실제와 달라지기 때문이다. 예: `gate, silu, up` window는 `silu`와 `up`의 출력만 반환한다.
- kernel 정의와 `CHAIN`은 바꾸지 않는다. 바꾸면 두 VM의 `U`가 어긋난다. 꼭 바꿔야 하면 이유를 commit message와 기록 파일에 적고, 그 뒤의 data는 이전 data와 섞지 않는다. 함수를 새로 추가하는 것은 괜찮다.

## 4. 측정 규칙

- **측정값은 device time이다.** XLA executable 1회 실행이 device에서 차지한 시간으로 정의한다. `jax.profiler.trace`로 trace를 뜨고, TPU device plane의 `XLA module` event duration을 읽는다.
- host timing을 기본으로 쓰지 않는 이유: LENS의 TPU v6e·v5e 측정에서 host dispatch overhead가 호출당 100~160µs였다. 같은 측정에서 kernel 단독 latency는 v6e 기준 6~51µs였다 (1B 크기, seq 1024). floor를 빼는 방식으로는 이 크기의 kernel을 잴 수 없다.
- TPU v4는 chip 하나에 TensorCore가 2개다. 실행 1회가 core마다 `XLA module` event를 하나씩 남기면, 시간이 겹치는 event를 실행 1회로 묶고 첫 시작부터 마지막 종료까지를 그 실행의 시간으로 본다.
- 검증된 구현이 NPU-LENS repo의 `section3/sec3.3/tools/measure_hiding_tpu.py`에 있다 (`per_execution_device_us`, `measure_profiler`, `measure_wall`, `hlo_ops`). VM에서 그 repo에 접근할 수 있으면 가져다 쓴다. 접근할 수 없으면 위 정의대로 구현한다.
- warmup 20회, 그 뒤 한 trace 안에서 30회 반복한다. 대표값은 min이다. median과 wall-clock min도 같이 기록한다.
- dtype은 bf16, device는 1개만 쓴다.
- 입력은 같은 값을 반복해서 쓴다 (기본). 두 VM이 같은 방식을 써야 한다.
- window마다 compile된 HLO의 연산 종류 목록을 기록한다. 같은 window인데 shape에 따라 목록이 바뀌면 lowering이 바뀐 것이다.
- device time이 0이거나 event 수가 반복 횟수와 맞지 않으면 fresh process에서 한 번 다시 잰다. 그래도 실패하면 실패 행으로 남긴다.

## 5. Step 0·1 (두 VM이 각자 수행)

### Step 0. 환경 기록

`TPU_INSTRUCTIONS.md` Step 0과 같다. TPU type·topology, `jax.device_count()`, `jax.devices()[0].device_kind`, jax·jaxlib·libtpu version을 기록한다. v4가 아니어도 기록하고 그대로 진행한다.

TPU에서 돌리기 전에 script를 JAX CPU backend에서 작은 크기로 먼저 돌려 버그를 잡는다.

### Step 1. 측정 경로 확인 (gate)

1B 크기 `(D, INTER, NH, HD, KV) = (2048, 8192, 32, 64, 8)`, seq 512에서 `add1`(가장 작은 쪽)과 `gate`(큰 쪽) 두 token을 쓴다.

1. profiler 경로로 두 token을 측정한다. `XLA module` event가 반복 횟수만큼 잡히는지 확인한다.
2. 두 token을 각각 fresh process로 5번 측정해 process 사이의 min 값 변동을 본다.
3. 한 process에서 여러 window를 연달아 잰 값과 비교해 fresh process가 필요한지 정한다.
4. device time과 wall-clock min의 차이(dispatch overhead)를 기록한다.
5. 입력 사본을 여러 개 만들어 돌려 쓴 측정과 같은 입력을 반복한 측정을 비교한다. 5% 이상 다르면 차이를 기록한다. 기본 방식(같은 입력 반복)은 바꾸지 않는다.

통과 조건: 두 token 모두 process 사이 변동이 5% 이하다.

이때 얻은 변동 폭이 이후 모든 판정의 noise floor다. 그보다 작은 오차 차이는 의미가 없다고 본다.

실패하면 반복 횟수를 늘리고, 그래도 안 되면 `TPU_INSTRUCTIONS.md` Step 1의 대안(jit 안에서 여러 번 실행)을 시도한다. 다 안 되면 중단하고 보고한다.

## 6. 산출물과 기록

- code: Module 1은 `profiling/tpu_v4/module1/`, Module 2는 `profiling/tpu_v4/module2/`.
- data: self-describing 행 형식의 text 파일. 형식은 각 module 문서에 있다.
- 기록: `memory/YYYY-MM-DD-tpu-m1.md`, `memory/YYYY-MM-DD-tpu-m2.md`. Step별 결과 표, 판정, caveat을 적는다.
- Step이 끝날 때마다 commit·push한다. push 전에 `git pull --rebase`를 한다.
- 다른 module의 directory, `profiling/`의 RNGD 파일, 기존 `memory/` 파일은 건드리지 않는다.
- 측정에 실패한 행은 지우지 않는다. 분석에서 제외한 data는 이유와 함께 적는다.
- fitting script는 반드시 commit한다. RNGD에서는 fitting script가 남지 않아 "ML 보정 2.3%"를 재현하지 못했다.
- token 같은 비밀값은 commit하지 않는다.

## 7. 일정과 판정 기준

- 논문 마감은 2026-12-08 (AoE)이다. 각 module 문서의 "최소 결과"를 먼저 내고, 그 다음에 범위를 넓힌다.
- 각 문서의 목표 수치는 잠정값이다. 목표에 못 미치면 수치를 맞추려고 방법을 바꾸지 말고, 어느 조건에서 깨지는지를 보고한다. 깨지는 지점도 결과다.
