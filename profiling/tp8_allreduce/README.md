# TP=8 / all-reduce 측정

목표: RNGD 칩 내 tensor parallelism(TP=8)에서 **PE 간 통신(all-reduce/collective)이
포함되면 latency가 어떻게 달라지는가**를 실측한다.

결론 먼저: **`furiosa.torch.native_device.set_fusion(8)` 로 8 PE 를 융합해 컴파일하면
저수준 `CompileModule.from_exported` 경로에서도 TP=8 이 동작한다.** 통신 비용은
TP=8 에서만 나타나는 `Renegade::ParallelCopy` / `Cluster` span 과, 크게 벌어지는
`task − tu`(전체 device latency − compute union) 간극으로 분리 측정된다.

---

## RNGD 의 TP 구조 (핵심 아키텍처 발견)

RNGD 의 TP=8 은 **8개 PE 를 하나의 device 로 융합**(`rngd:0:8`, 내부적으로 `npu:0:0-7`)
하고, reduce/통신을 **네이티브 컴파일러가 on-chip PE interconnect 로** 처리한다.
그래서 **FX(torch.fx) 레벨엔 `all_reduce` 노드가 생기지 않는다.** mppp/ModelRewriter 는
8-PE 그룹을 *단일* device 로만 보고 pipeline parallelism 만 표현한다
(`furiosa_llm/artifact/builder.py:246` 주석이 이를 명시).

즉 "all_reduce span 을 이름으로 골라낸다"는 접근은 성립하지 않는다. reduce 는
네이티브 TUC 레벨에서 compute + PE 간 copy 로 흩어져 실행되고, 프로파일러엔
`ParallelCopy`(PE 간 데이터 이동) 와 `Cluster`(cluster 조율) span 으로 나타난다.

관련 SDK 위치:
- collective 방출부(FX): `model_rewriter/cc_calculator.py:262` `_same_mesh_partial_to_replicate` → `AllReduce`
- single-device 형태: `model_rewriter/ops/single_device_comm.py:86` `AllReduceSingle` → `SuperTaskKind.ALL_REDUCE`
- 실행 차단: `new_pipeline_builder.py:769` `NotImplementedError("Communication supertasks are not supported yet.")`
- TP 융합 device 문자열: `artifact/resolver.py:162` `_get_tp_group_device` → `npu:0:0-7`
- 융합 API: `furiosa/torch/native_device.py` `set_fusion(num_pe)` (프로세스당 1회), `MeshKind.Single`

---

## 되는 방법 — `set_fusion(8)` 네이티브 TP  ✅

`tp_sweep.py` — `set_fusion(NPE)` 후 workload 를 한 그래프로 compile → 네이티브가
8 PE 로 자동 TP. `set_fusion` 은 프로세스당 1회만 가능하므로 NPE 를 인자로 두 번
실행(1, 8)하고 offline diff.

```
python tp_sweep.py 1     # → tp_sweep_np1_results.csv
python tp_sweep.py 8     # → tp_sweep_np8_results.csv
```

### 결과 (B=1, S=512, Llama-3.1-8B shape)

| workload | TP=1 task | TP=8 task | 전체 속도↑ | compute(tu) 속도↑ | TP=8 ParallelCopy | TP=8 Cluster | TP=8 task−tu |
|---|---|---|---|---|---|---|---|
| matmul (D×D) | 656.5 | 307.0 | 2.1× | 595.7→141.0 (4.2×) | 8.8 | 148.3 | 166.1 |
| mlp (up→silu→down) | 4331.0 | 1593.0 | 2.7× | 4279.9→691.2 (6.2×) | 17.9 | 29.4 | 901.8 |
| attn (qkv→sdpa→o) | 2670.6 | 1100.0 | 2.4× | 2640.3→491.7 (5.4×) | 29.3 | 340.2 | 608.3 |

(us, N_RUNS=3 median)

### 해석
- **compute 는 4~6× 잘 병렬화**되지만 **전체 task 는 2~2.7× 에 그친다.** 차이는 전부
  TP=8 에서 새로 생기는 비용 — PE 간 통신 + reduce + 동기화 bubble.
- TP=1 에선 `task ≈ tu`(간극 30~60us, compute-bound). **TP=8 에선 `task − tu` 가
  166~902us 로 폭증** → 이 간극이 "all-reduce 포함으로 늘어난 latency" 의 실체.
- TP=8 에서만 등장하는 span: **`Renegade::ParallelCopy`**(PE 간 데이터 이동=collective),
  **`Cluster`**(cluster 조율). TP=1 엔 대신 `Renegade::StoTrf` 만 있고 이 둘은 없다.
- ParallelCopy+Cluster 만으로 task−tu 간극이 다 설명되진 않는다(나머지는 병렬화 안 된
  DMA + 직렬화). 즉 TP=8 의 병목은 compute 가 아니라 **통신 + DMA**.
- 요지: **RNGD 칩 내 TP=8 은 이상적 8× 가 아니라 통신·DMA 에 막혀 실효 2~2.7×.**
  all-reduce 는 별도 이름 span 이 아니라 on-chip 에 녹아 있고, TP=1↔TP=8 차분과
  ParallelCopy/Cluster span 으로 정량화된다.

---

## 안 되는 방법 — 손으로 FX all_reduce 삽입  ❌ (문서화용)

`fx_allreduce_probe.py` — `MpppConfig` 를 손으로 author 해서 matmul 출력을 Partial 로
두고 Replicate 요청 → `ModelRewriter` 가 AllReduce 를 넣게 유도하는 시도.

두 변형 모두 **실행 가능한 all_reduce 를 만들지 못한다**:

1. **K(contraction)축 샤딩** (구 `../legacy` 및 초기 `tp_allreduce_derisk.py` 방식):
   `ShardingPropagator` 가 막음 —
   `"only supports cases where all inputs are replicated"`
   (`sharding_prop/sharding_propagator.py:211`). 자동 전파 규칙이 없어 샤딩된 입력을
   거부.
2. **입력 Replicate + 출력 Partial 강제** (이 스크립트): propagator 는 통과하지만
   **AllReduce 노드가 아예 삽입되지 않고** 8개 독립(복제) matmul 만 생성됨.
   (입력이 replicate 라 출력도 replicate 로 처리되어 reduce 가 불필요해짐)
3. 설령 노드가 생겨도 **`new_pipeline_builder.py:769` 에서 실행 차단**
   (`"Communication supertasks are not supported yet."`). FX collective 는 inter-device
   (multi-chip) 용이고 칩 내 TP 실행 경로가 아니다.

→ FX 경로는 collective 방출 메커니즘을 *확인*하는 데만 쓸모 있고, 칩 내 TP=8
   실측에는 부적합. 실측은 반드시 `set_fusion(8)` 네이티브 경로를 쓴다.

---

## 파일
- `tp_sweep.py` — ✅ TP=1 vs TP=8 실측 (set_fusion). 메인.
- `fx_allreduce_probe.py` — ❌ FX all_reduce 삽입 실패 재현 (문서화).
- `tp_sweep_np1_results.csv`, `tp_sweep_np8_results.csv` — 결과.
- `logs/` — 실행 로그.
