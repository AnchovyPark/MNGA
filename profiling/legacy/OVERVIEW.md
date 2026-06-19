# RNGD LLM 커널 프로파일링 — 개요

이 폴더는 RNGD에서 LLM 추론 커널을 프로파일링하는 스크립트·결과 모음입니다.
(npu_chip_project/LENS/inference_profiling/RNGD 에서 이전 — 2026-06-19, 이후 작업은 여기서 진행)

## 실행 환경

venv는 용량 때문에 옮기지 않았습니다. **절대경로로 호출**하세요:

```bash
VENV=/home/furiosa/바탕화면/pjh/npu_chip_project/.venv/bin/python
cd /home/furiosa/바탕화면/pjh/rngd/profiling
$VENV llm_kernel_profile.py all          # 전체 커널 택소노미 (decode+prefill)
$VENV llm_kernel_profile.py decode 16384 # decode, 특정 ctx
```

스크립트는 출력 CSV를 자기 위치(이 폴더)에 씁니다.

⚠️ 운영 노트(중요)는 `memory.md` 참고: OOM 한계, 써멀 스로틀링, `uv run` 금지(venv python 직접), 프리빌트 아티팩트 사용법.
연구 세션 기록은 상위 `../memory/` 폴더(날짜별).

## 핵심 스크립트

| 스크립트 | 목적 |
|---|---|
| `llm_kernel_profile.py` | **LLM 전체 커널 택소노미 (Lv0)**. 커널별 DMA/연산/유효BW/AI/bound 분류 |
| `attn_decode_profile.py` | decode attention DMA vs 연산 분해 (ctx sweep) |
| `kernel_profiler_eval.py` | 단일/체인 커널 span 측정 (matmul/add/softmax 등) |
| `tactic_kernel_spike.py` | TacticKernel(커스텀 커널) RNGD 작성·컴파일·실행 검증 |
| `run_profiling.py` | E2E TTFT/TBT sweep (furiosa-llm 레벨) — 상세는 `README.md` |

## 프로파일러 도구

공식 `RNGDProfiler`(chrome-trace)는 torch 2.10+xpu / Intel PTI 충돌로 이 셋업에서 사용 불가.
→ 저수준 경로: `CompileModule.from_exported` → `generate_profiles` →
`edf.npu_node.build_tuc_profile_spans` → span(Task/TuExec/DMA/ParallelCopy/StoTrf) 시간(us).
바이트/FLOP는 모델에서 계산해 유효 BW·TFLOPS·arithmetic intensity 도출 (profiler에 byte 카운터 없음).
