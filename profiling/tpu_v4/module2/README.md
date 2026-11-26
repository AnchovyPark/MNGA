# Module 2 — kernel 순서와 kernel별 값으로 3-gram window 예측 (TPU v4)

지시서: repo root `TPU_MODULES_COMMON.md`, `TPU_MODULE2_NGRAM.md`. 기록: `memory/2026-11-26-tpu-m2.md`.
실행 환경: `~/venvs/vm3_jax/bin/python` (jax 0.11.2, 측정), `/usr/bin/python3` (numpy, fitting).

## 측정 규약 (두 module 공통으로 확정된 것)

- **값 = `sync_min_us`** (data 행 끝에서 두 번째 열). 두 TensorCore 중 늦게 시작한 core의 시작 ~ 마지막 종료.
  `dev_*` 열은 공통 문서 §4 정의(union)이며 megacore launch skew(13~46µs) 때문에 쓰지 않는다. Module 1과 같은 정의.
- **입력 방식 = 사본 순환이 주 결과.** `data_*_copies2_*` (Module 2, 사본 2개) ↔ Module 1 `copies8` (사본 8개).
  같은 입력 반복(`data_windows_tpu_v4.txt`, 공통 문서 기본)은 cross-program prefetch가 weight ≤ 128 MiB(CMEM)를
  다음 실행용으로 미리 올려 두는 측정 artifact가 있어 gemm을 1.1~2.9배 과소평가한다. 비교용으로만 남긴다.
  사본 2개 = 4개 = 8개 (window·U 모두 noise floor 안에서 같음) → "직전 실행과 다른 buffer"면 충분하다.
- full = 연결된 덩어리별: A = `norm1..v` (token 0~5), B = `o..add2` (6~14). layer = A + B.
  chain 전체(0:15)는 `attn_out`이 외부 입력이라 XLA가 두 덩어리 순서를 섞어 실제 layer와 다르다.

## 파일

| 파일 | 내용 |
|---|---|
| `measure.py` | profiler trace에서 device time (union, sync) 읽기, HLO 연산 목록 |
| `run_windows.py` | 한 process에서 window 목록 측정 → `WROW`/`HROW` 행 |
| `run_jobs.py` | (크기, seq, copies) job을 chip 4개에 나눠 실행 |
| `step1.py` | Step 1 gate 실험 |
| `analyze_tele.py`, `analyze_subchain.py` | 측정 window telescope (chain 전체 / 덩어리별) |
| `fit_step4.py` | Step 4~6: W2·W3 예측 (M0~M3 × S0~S3 × E1~E3), 덩어리별 telescope, `--upred`로 Module 1 U 연결 |
| `data_windows{,_copies2}_tpu_v4.txt` | 크기 10 × seq 4 × window 68 |
| `fit_*.log/json` | fitting 결과 |

data 행: `WROW D INTER NH HD KV window seq dev_min_us dev_med_us wall_min_us n_samples sync_min_us sync_med_us`
