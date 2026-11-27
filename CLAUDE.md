# MNGA 작업 규칙

이 repo에서 일하는 모든 Claude 세션(TPU VM 포함)이 따르는 규칙이다.

## 용어

- 지시서, `memory/` 기록, commit message, 사용자에게 하는 보고는 **[`TERMS.md`](TERMS.md)의 용어만** 쓴다. 일을 시작하기 전에 `TERMS.md`를 먼저 읽는다.
- code의 변수·함수 이름과 data 행 형식은 바꾸지 않는다. 문서에서 언급할 때는 용어를 먼저 쓰고 code 이름을 괄호로 붙인다. 예: "3-gram latency(`W3`)".
- `TERMS.md` 이전에 쓴 지시서와 기록에는 예전 표기(token, window, `U`, `W3`, telescope 등)가 남아 있다. 읽을 때는 `TERMS.md` §7의 대응표를 쓰고, 새로 쓰는 글에는 옮기지 않는다.

## 보고 방식

- 결론을 먼저 쓴다. 숫자에는 단위와 조건(model 크기, seq, 측정인지 예측인지)을 붙인다.
- 예시 숫자 하나로 무슨 뜻인지 보여 준다 (`TERMS.md` §1의 예처럼).
- kernel, latency, layer, profiling, fusion, overlap, gemm, shape, seq 같은 기술 어휘는 영어 그대로 쓴다.

## 지시서

| 파일 | 상태 |
|---|---|
| `TPU_NEXT_MODEL.md` | 진행할 것 (Module 1 VM) |
| `TPU_NEXT_LAYER.md` | 끝남 (`memory/2026-11-27-tpu-layer.md`) |
| `TPU_MODULE1_KERNEL.md`, `TPU_MODULE2_NGRAM.md`, `TPU_MODULES_COMMON.md` | 끝남 (`memory/2026-11-26-tpu-m1.md`, `-m2.md`) |
| `TPU_INSTRUCTIONS.md` | 따로 돌리지 않음. 위 지시서들이 대신한다 |

## git

- branch를 만들지 않고 `master`에 바로 push한다. push 전에 `git pull --rebase`.
- 측정이 도는 중에는 `git stash`를 쓰지 않는다 (log가 덮여 잃은 적이 있다).
- Step이 끝날 때마다 commit·push한다. 측정에 실패한 행도 지우지 않는다.
