# TPU 다음 단계 A — attention을 포함한 layer 1개의 n-gram 예측

> 이 문서는 [`TERMS.md`](TERMS.md) 이전에 썼고 이미 끝난 작업이다. 예전 표기(token, window, `U`, `W3`, telescope 등)는 `TERMS.md` §7의 대응표로 읽는다. 새 글에는 `TERMS.md` 용어를 쓴다.

작성일 2026-11-26. 대상: **Module 2를 맡은 VM** (`~/venvs/vm3_jax`). 같은 시기에 Module 1 VM은 [`TPU_NEXT_MODEL.md`](TPU_NEXT_MODEL.md)를 한다.

먼저 읽을 것:

- `profiling/tpu_v4/module2/README.md`의 측정 규약 (`sync_min`, 사본 순환)
- `memory/2026-11-26-tpu-m2.md` (Step 2~6, 덩어리 A·B, 변형 V0~V3)
- `memory/2026-11-26-tpu-m1.md`의 "2차 범위" (attention kernel, fused attn)

`TPU_MODULE2_NGRAM.md` Step 7의 "attention core"를 구체화한 문서다. 둘이 다르면 이 문서를 따른다.

## 1. 왜 하나

- 지금 결과: shape만으로 안 본 model 크기의 layer 일부(attention 제외, A + B)를 평균 6.9%로 예측했다. 단독 합은 31%다.
- 빠진 것은 attention core다. 논문의 예측 대상은 layer 전체이므로 attention이 들어가야 한다.
- attention이 들어가면 A(`norm1..v`)와 B(`o..add2`)가 data 의존으로 이어진다. Step 2에서 telescope를 깨뜨린 "연결 없는 두 덩어리의 순서 바꾸기"가 없어질 수 있다. 이것이 첫 확인 대상이다.

## 2. 두 가지 chain

| 이름 | attention 표현 | token 수 | 상태 |
|---|---|---|---|
| L20 | `qk, scale, mask, softmax, av` 5개 token | 20 | 이 VM에서 `attn.py`로 진행 중 (commit 전) |
| L16 | `attn` token 1개 (`kernels_attn.make("attn")`) | 16 | 새로 추가 |

**둘 다 잰다.** 진행 중인 L20은 멈추지 말고 끝낸다. 주 결과는 Step 1의 판정으로 고른다.

L16을 추가하는 이유는 Module 1 결과 때문이다 (`memory/2026-11-26-tpu-m1.md` 2차 범위).

- fused attn은 다섯 kernel 단독 합의 0.23~0.35배다. `qk` 단독 시간의 67%가 score layout을 바꾸는 copy이고, fused에서는 중간 score를 HBM에 반복해 읽고 쓰는 일이 줄어든다.
- 그래서 L20에서는 n-gram이 이 3~4배 차이를 다 담아야 한다. L16은 이 차이가 `attn` token 안에 들어가 있다.
- Module 1은 `attn`의 `U`를 이미 예측한다 (P3, 안 본 크기 오차 3.9%, `module1/data_upred_attn_tpu_v4.txt`).

### L16 token

| idx | token | type | data 입력 |
|---|---|---|---|
| 0 | `norm1` | rmsnorm | `x` (외부) |
| 1 | `q` | gemm | `norm1` |
| 2 | `rope_q` | rope | `q` |
| 3 | `k` | gemm | `norm1` |
| 4 | `rope_k` | rope | `k` |
| 5 | `v` | gemm | `norm1` |
| 6 | `attn` | attn | `rope_q`, `rope_k`, `v` |
| 7 | `o` | gemm | `attn` |
| 8 | `add1` | add | `x`, `o` |
| 9 | `norm2` | rmsnorm | `add1` |
| 10 | `gate` | gemm | `norm2` |
| 11 | `silu` | silu | `gate` |
| 12 | `up` | gemm | `norm2` |
| 13 | `mul` | mul | `silu`, `up` |
| 14 | `down` | gemm | `mul` |
| 15 | `add2` | add | `add1`, `down` |

- chain이 밖으로 내보내는 값은 `rope_k`, `v`(KV cache), `add2`(다음 layer 입력)다. `rope_q`는 `attn`만 쓰므로 내보내지 않는다.
- window 규칙은 지금과 같다. window 밖에서 쓰이는 값만 반환하고, window 밖에서 오는 입력은 같은 shape의 dummy로 준다.
- **L16의 full window(0:16)는 `profiling/tpu_v4/layer.py`의 `make_layer`와 같은 계산이어야 한다.** Module 1 VM이 layer 사이를 잴 때 이 함수를 쓴다. CPU 작은 크기에서 두 출력이 bf16 오차 안에서 같은지 먼저 확인한다.
- `layer.py`는 JAX CPU backend에서만 확인했다. `kernels.py`의 A·B window와 `kernels_attn`의 `attn`을 이어 붙인 값과 일치한다.

## 3. 측정 규약

`module2/README.md` 그대로 쓴다. 값은 `sync_min_us`, 입력은 사본 2개 순환(`copies2`)이 주 결과다.

같은 입력 반복(`copies1`)은 Step 1의 1B에서만 비교용으로 잰다. 나머지는 시간을 아끼기 위해 생략한다.

## 4. 단계

### Step 1. pilot (gate) — chain 고르기

1B `(2048, 8192, 32, 64, 8)`, seq {512, 2048}에서 L16과 L20 각각 W1~W5와 full을 잰다. L16은 덩어리 A(0:6), B(7:16)도 잰다.

보고 표 (chain × seq):

| 값 | 뜻 |
|---|---|
| `uni/full` | 단독 합 |
| `tele2/full` ~ `tele5/full` | 측정 window telescope |
| `(A + U(attn) + B)/full` | attention을 단독값으로 끼워 넣고 덩어리를 따로 telescope한 경우 (L16만) |

HLO에서 다음도 확인한다.

- `attn`이 이웃 gemm(`v`, `o`)과 fusion되는지
- custom-call이 생기는지 (Module 1은 v4에서 online-softmax 치환이 없다고 확인했다)
- full에서 XLA가 token 순서를 바꿔 실행하는지 (trace)

판정:

| 결과 | 다음 |
|---|---|
| L16 또는 L20의 `tele3` 오차 ≤ 5%이고 `uni`보다 작다 | 성립. 오차가 작은 chain을 주 결과로 Step 2. 비슷하면 L16 |
| `tele3`은 5% 초과, `tele4`나 `tele5`는 ≤ 5% | 그 n으로 Step 2 |
| 전부 10% 초과 | trace로 원인을 적고, `(A + U(attn) + B)` 방식으로 갈 수 있는지 판단해 보고 |

### Step 2. data 수집

크기 10종(지금과 같은 목록) × seq {128, 512, 1024, 2048} × 주 chain의 W1~W3(Step 1에서 고른 n까지) + full, `copies2`.

score 크기는 최대 `NH·S²·2B` = 40 × 2048² × 2 ≈ 335 MB이므로 한 chip에 들어간다.

### Step 3. 예측

`fit_step4.py`를 주 chain의 token type으로 확장해 다시 학습한다 (input S1, target `r`, GBM, split E1·E2·E3).

- `token_meta`에 `attn`의 flops·bytes를 넣는다 (`kernels_attn.feats` 사용).
- Module 1 결과에서 attention 단가가 score 128 MiB 경계로 갈렸으므로, score bytes(`NH·S²·2`)를 feature 후보에 넣는다.
- `U`는 측정값과 Module 1 예측값 두 가지로 돌린다. `attn` token은 `module1/data_upred_attn_tpu_v4.txt`, 나머지는 `data_upred_copies8_tpu_v4.txt`를 쓴다.

보고:

| 값 | 측정 U | 예측 U |
|---|---|---|
| W3 MAPE (E1, E2, E3) | | |
| layer `uni` 오차 | | |
| layer `tele3` 오차 (측정 window, 상한) | | |
| **layer `tele3_hat` 오차 (예측 window)** | | |

목표(잠정): layer `tele3_hat` 오차 ≤ 10%이고 `uni`보다 작다.

### Step 4. Module 1 VM에 넘기기

크기마다 E1(그 크기를 빼고 학습) 방식의 layer 1개 예측값을 낸다. 예측 U와 예측 window를 쓴 완전 bottom-up 값이다.

`profiling/tpu_v4/module2/data_layerpred_tpu_v4.txt`:

```
LPRED D INTER NH HD KV seq l1_pred_us l1_meas_us
```

`l1_meas_us`는 주 chain full window의 측정값이다. 이 파일은 만드는 즉시 push한다. Module 1 VM의 Step 6이 이 파일을 기다린다.

## 5. 최소 결과와 순서

- 논문 마감은 2026-12-08 (AoE)이다.
- 최소 결과: Step 1 판정, seq {512, 2048}만으로 한 Step 3 표, Step 4 파일.
- 그 다음 seq {128, 1024}를 채운다.
- Step마다 commit·push한다. push 전에 `git pull --rebase`.

## 6. 하지 않는 것

- B 덩어리 구조 오차의 원인 분석(쌍 항·세 op 항 분해)은 이 VM이 하지 않는다. 이미 push된 data로 다른 곳에서 분석한다.
- layer 사이, full model은 Module 1 VM의 일이다.
- 기록은 `memory/YYYY-MM-DD-tpu-layer.md`에 Step별 표, 판정, caveat으로 남긴다. code·data는 `profiling/tpu_v4/module2/`에서 계속한다.
