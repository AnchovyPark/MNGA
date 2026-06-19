#!/usr/bin/env python3
"""RNGD furiosa-llm direct profiling runner.

Furiosa LLM (RNGD) 공식 API 를 통해 uniform-batch profiling 수행.
inf2/run_profiling.py 와 1:1 대응되도록 설계.

핵심 설정 (Furiosa 공식 권장):
  - prefill_buckets = [(1, b) for b in buckets]
      → prefill 은 per-sample (batch=1) — inf2 의 ctx_batch_size=1 등가
  - decode_buckets  = [(batch_size, b) for b in buckets]
      → decode 는 uniform batch=batch_size — continuous batching 미사용 동치
  - SamplingParams(temperature=0, min_tokens=ol, max_tokens=ol, ignore_eos=True)
      → deterministic + 정확히 ol 토큰 생성
  - E2E only 측정 (dual-call 하지 않음)

입력: profile.csv (combo 당 1 row, 60 combos)
동작:
  1. ArtifactBuilder 로 artifact build (compile + save)  ─ 1회
  2. LLM(artifact_dir, devices=...) 로 load                ─ 1회
  3. bucket 대표 shape 별로 1회 warmup
  4. 각 combo × n_runs 측정 (uniform batch=batch_size)
  5. 결과 CSV 저장

사용법:
  python3 run_profiling.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tp-degree 8 --batch-size 32 --max-model-len 8192 \
    --n-runs 3 \
    --devices "npu:0:*" \
    --output-dir ~/lens_profiling_results/RNGD/Llama-3.1-8B-Instruct \
    --compiled-dir ~/compiled_models_lens_profiling/Llama-3.1-8B-Instruct
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

# LENS bucket list (inf2 와 동일 — 두 NPU 비교를 위해 같은 bucket 사용).
# Furiosa 공식 권장: 각 bucket seq_len 은 power-of-two 가 자연스러움.
RNGD_BUCKETS = [128, 256, 512, 1024, 2048, 4096, 8192]

# 이 파일: LENS/inference_profiling/RNGD/run_profiling.py
HERE = os.path.dirname(os.path.abspath(__file__))
# LENS root = .../LENS/
LENS_ROOT = os.path.dirname(os.path.dirname(HERE))
# 결과는 repo 내부 LENS/inference_results/profiling/RNGD/<model>/bs<B>/ 에
RESULTS_ROOT = os.path.join(LENS_ROOT, "inference_results", "profiling", "RNGD")


def _model_name_from_path(model_path):
    """model_path 에서 사람이 읽기 좋은 model name 추출.

    HF cache snapshot path 인 경우 (예: .../hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/<hash>)
    'Llama-3.1-8B-Instruct' 를 반환. HF model id (org/name) 인 경우 name 만 반환.
    그 외엔 basename.
    """
    p = model_path.rstrip("/")
    if "/snapshots/" in p and "/models--" in p:
        for part in p.split("/"):
            if part.startswith("models--"):
                segs = part[len("models--"):].split("--")
                if len(segs) >= 2:
                    return "--".join(segs[1:])
    # HF model id like "meta-llama/Llama-3.1-8B-Instruct"
    if "/" in p and not os.path.exists(p):
        return p.split("/")[-1]
    return os.path.basename(p)


def default_output_dir(model_path, batch_size):
    model_name = _model_name_from_path(model_path)
    return os.path.join(RESULTS_ROOT, model_name, f"bs{batch_size}")


def parse_buckets_arg(value):
    """Parse comma-separated bucket list, e.g. '32,64,128,256'."""
    if value is None or value.strip() == "":
        return None
    buckets = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not buckets:
        raise ValueError("--buckets must contain at least one integer")
    return buckets


def _buckets_for(max_model_len, bucket_list=None):
    base = bucket_list or RNGD_BUCKETS
    bs = [b for b in base if b <= max_model_len]
    if max_model_len not in bs:
        bs.append(max_model_len)
    return sorted(set(bs))


def _find_prebuilt_artifact(model_path):
    """model_path 가 furiosa-llm prebuilt artifact 인지 감지.

    furiosa-ai HF org 모델들 (예: furiosa-ai/Llama-3.1-8B-Instruct) 은
    raw HF 모델이 아니라 이미 컴파일된 artifact (artifact.json +
    binary_bundle.zip) 이므로 ArtifactBuilder 없이 LLM() 으로 바로 로드한다.

    Returns:
        (artifact_src, artifact_json_path) — prebuilt 면 LLM 에 넘길 경로와
        artifact.json 의 로컬 경로. 아니면 (None, None).
    """
    local = os.path.expanduser(model_path)
    if os.path.isdir(local):
        aj = os.path.join(local, "artifact.json")
        if os.path.exists(aj):
            return local, aj
        return None, None
    try:
        from huggingface_hub import hf_hub_download
        aj = hf_hub_download(model_path, "artifact.json")
        return model_path, aj
    except Exception:
        return None, None


def _buckets_from_artifact(artifact_json_path, max_model_len):
    """prebuilt artifact.json 에서 prefill bucket attention size 목록 추출."""
    with open(artifact_json_path) as f:
        artifact = json.load(f)
    sizes = set()
    for pm in artifact.get("model", {}).get("pipeline_metadata_list", []):
        for b in pm.get("attention_buckets", []):
            # prefill bucket: kv_cache_size == 0
            if b.get("kv_cache_size", -1) == 0:
                sizes.add(b["attention_size"])
    return sorted(s for s in sizes if s <= max_model_len) or sorted(sizes)


def load_profile(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["id"]), int(r["input_len"]), int(r["output_len"])))
    return rows


def _extra_llm_kwargs(max_io_memory_mb, max_processing_samples):
    """LLM 로드 시 메모리 거동 제어 인자 구성.

    max_io_memory_mb: IO(워크스페이스) 예약 영역 상한. KV 풀이 HBM 을 탐욕적으로
      선점하므로, 큰 bucket(8192)/긴 시퀀스의 워크스페이스 자리를 확보하려면 이 값을
      키워 KV 풀을 줄인다 (예: 14B full bucket + max_len 8192 에서 8192 권장).
    max_processing_samples: 동시 처리 샘플 수. bs=1 profiling 에선 작게(예: 4) 두면
      KV 풀 예약이 줄어 메모리 여유가 생긴다.
    """
    kwargs = {}
    if max_io_memory_mb is not None:
        kwargs["max_io_memory_mb"] = max_io_memory_mb
    if max_processing_samples is not None:
        from furiosa_llm.metadata.config_types import SchedulerConfig
        kwargs["scheduler_config"] = SchedulerConfig(
            max_processing_samples=max_processing_samples)
    return kwargs


def init_model(model_path, tp_degree, pp_degree, batch_size, max_model_len,
               compiled_dir, devices,
               skip_compile=False, bucket_list=None,
               num_pipeline_builder_workers=4, num_compile_workers=4,
               max_io_memory_mb=None, max_processing_samples=None):
    """ArtifactBuilder 로 build → LLM 으로 load.

    Furiosa 공식 권장:
      - tp_degree ∈ {1, 4, 8} (power of two)  ─ Llama 는 4 또는 8 권장
      - prefill batch=1, decode batch=batch_size (uniform 측정용)
    """
    from furiosa_llm import LLM
    from transformers import AutoTokenizer

    extra_kwargs = _extra_llm_kwargs(max_io_memory_mb, max_processing_samples)

    # ── prebuilt artifact 경로 (furiosa-ai HF org 모델 등) ──────────────
    artifact_src, artifact_json = _find_prebuilt_artifact(model_path)
    if artifact_src is not None:
        buckets = _buckets_from_artifact(artifact_json, max_model_len)
        print(f"[init_model] prebuilt furiosa artifact 감지: {model_path}")
        print("  → ArtifactBuilder build 생략, LLM 으로 직접 로드")
        print(f"  artifact prefill buckets={buckets}")
        if max_model_len > max(buckets):
            print(f"  [WARN] max_model_len={max_model_len} 이 artifact 최대 "
                  f"bucket({max(buckets)}) 보다 큼 — 초과 combo 는 실패할 수 있음")
        llm_kwargs = dict(extra_kwargs)
        if devices:
            llm_kwargs["devices"] = devices
        print("\n[load] loading artifact to RNGD ...")
        if extra_kwargs:
            print(f"  extra LLM kwargs: {extra_kwargs}")
        t0 = time.monotonic()
        llm = LLM(artifact_src, **llm_kwargs)
        print(f"  load time: {time.monotonic()-t0:.1f}s")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return llm, tokenizer, buckets

    # ── raw HF 모델 경로: ArtifactBuilder 로 직접 build ────────────────
    buckets = _buckets_for(max_model_len, bucket_list=bucket_list)
    prefill_buckets = [(1, b) for b in buckets]
    decode_buckets = [(batch_size, b) for b in buckets]
    # composable kernel 의 tokenwise(비-attention) layer 는 총 토큰 수
    # (= batch * input_ids_len) 단위로 컴파일된다.
    #   prefill (1, b)          → query len = b   (buckets 와 동일)
    #   decode  (batch_size, 1) → query len = batch_size
    # 공식 preset (Qwen2.5-0.5B 등) 과 동일하게 두 집합의 합집합을 사용.
    tokenwise_seq_lens = sorted({batch_size} | set(buckets))

    print(f"[init_model] model={model_path}")
    print(f"  tp={tp_degree} pp={pp_degree} batch={batch_size} max_model_len={max_model_len}")
    print(f"  buckets={buckets}")
    print(f"  prefill_buckets={prefill_buckets}")
    print(f"  decode_buckets={decode_buckets}")
    print(f"  tokenwise_seq_lens={tokenwise_seq_lens}")
    print(f"  devices={devices}  compiled_dir={compiled_dir}")

    compiled_dir = os.path.expanduser(compiled_dir)
    os.makedirs(os.path.dirname(compiled_dir.rstrip("/")) or ".", exist_ok=True)

    if not skip_compile:
        from furiosa_llm.artifact import (
            ArtifactBuilder,
            BucketConfig,
            ModelConfig,
            ParallelConfig,
        )

        print("\n[build] building artifact (this may take many minutes) ...")
        # furiosa-llm 2026.x API: 개별 kwargs 대신 config 객체로 전달.
        # append_buckets(chunked prefill 용) 는 uniform-batch 측정에 불필요해
        # 비워둔다 — 공식 Qwen2.5-0.5B preset 도 append 없이 동작.
        # 생성 모델에서 bucket 필드를 일부만 채우면 partial-config 검증에
        # 걸리므로 skip_validation=True 로 그 검사만 우회한다
        # (decode bucket 존재 / max_model_len 한도 검증은 그대로 수행됨).
        builder = ArtifactBuilder(
            model_path,
            model_config=ModelConfig(max_model_len=max_model_len),
            parallel_config=ParallelConfig(
                tensor_parallel_size=tp_degree,
                pipeline_parallel_size=pp_degree,
            ),
            bucket_config=BucketConfig(
                prefill_buckets=prefill_buckets,
                decode_buckets=decode_buckets,
                tokenwise_seq_lens=tokenwise_seq_lens,
                skip_validation=True,
            ),
        )
        t0 = time.monotonic()
        builder.build(
            compiled_dir,
            num_pipeline_builder_workers=num_pipeline_builder_workers,
            num_compile_workers=num_compile_workers,
        )
        print(f"  build time: {time.monotonic()-t0:.1f}s")

    print("\n[load] loading artifact to RNGD ...")
    t0 = time.monotonic()
    llm_kwargs = dict(max_model_len=max_model_len, **extra_kwargs)
    if devices:
        llm_kwargs["devices"] = devices
    if extra_kwargs:
        print(f"  extra LLM kwargs: {extra_kwargs}")
    llm = LLM(compiled_dir, **llm_kwargs)
    print(f"  load time: {time.monotonic()-t0:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return llm, tokenizer, buckets


def make_prompt_token_ids(tokenizer, target_len, batch_size):
    """길이 target_len 의 dummy prompt 를 batch_size 개 복제 → BatchEncoding.

    furiosa-llm 2026.x 의 generate() 는 input_ids 를 torch tensor 가 아닌
    python list-of-lists 로 기대한다 (`if input_ids and isinstance(input_ids[0], list)`).
    """
    from transformers.tokenization_utils_base import BatchEncoding

    if target_len <= 0:
        target_len = 1
    text = "The quick brown fox jumps over the lazy dog. " * (target_len // 8 + 1)
    ids = tokenizer.encode(text, add_special_tokens=True)[:target_len]
    while len(ids) < target_len:
        ids.append(tokenizer.pad_token_id)
    input_ids = [list(ids) for _ in range(batch_size)]
    attention_mask = [[1] * target_len for _ in range(batch_size)]
    return BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})


def _make_sampling_params(ol):
    """deterministic + 정확히 ol 토큰 생성 (EOS 로 일찍 종료 금지)."""
    from furiosa_llm import SamplingParams
    return SamplingParams(
        temperature=0.0,
        min_tokens=ol,
        max_tokens=ol,
        ignore_eos=True,
    )


def _max_generated(outputs):
    """RequestOutput list 에서 가장 긴 생성 토큰 수."""
    n = 0
    for out in outputs:
        if not getattr(out, "outputs", None):
            continue
        for completion in out.outputs:
            tok_ids = getattr(completion, "token_ids", None)
            if tok_ids is None:
                continue
            n = max(n, len(tok_ids))
    return n


def measure_combo(llm, tokenizer, il, ol, batch_size, n_runs, combo_id,
                  max_model_len):
    """한 combo × n_runs 측정 (E2E only)."""
    if il + ol > max_model_len:
        return [{
            "run_id": r, "combo_id": combo_id,
            "combo_il": il, "combo_ol": ol,
            "batch_size": batch_size, "status": "SKIPPED_TOO_LONG",
            "batch_e2e_ms": "", "error": "",
        } for r in range(n_runs)]

    encoded = make_prompt_token_ids(tokenizer, il, batch_size)
    sp = _make_sampling_params(ol)

    rows = []
    for run_id in range(n_runs):
        try:
            t0 = time.perf_counter()
            outputs = llm.generate(
                prompts=[""] * batch_size,
                sampling_params=sp,
                prompt_token_ids=encoded,
            )
            e2e_s = time.perf_counter() - t0
            max_n = _max_generated(outputs)

            rows.append({
                "run_id": run_id,
                "combo_id": combo_id,
                "combo_il": il,
                "combo_ol": ol,
                "batch_size": batch_size,
                "status": "OK",
                "sample_ids": json.dumps(list(range(batch_size))),
                "input_lens": json.dumps([il] * batch_size),
                "output_lens": json.dumps([ol] * batch_size),
                "max_n_generated": max_n,
                "batch_ttft_ms": "",
                "batch_e2e_ms": round(e2e_s * 1000, 3),
                "error": "",
            })
        except Exception as e:
            rows.append({
                "run_id": run_id,
                "combo_id": combo_id,
                "combo_il": il,
                "combo_ol": ol,
                "batch_size": batch_size,
                "status": "ERROR",
                "batch_e2e_ms": "",
                "error": str(e),
            })
    return rows


def warmup_each_bucket(llm, tokenizer, batch_size, buckets, max_model_len):
    """각 prefill bucket 대표 shape 1회씩 warmup (engine 캐시 워밍)."""
    prev = 0
    WARMUP_SHAPES = []
    for b in buckets:
        il = max(1, prev + 1 if prev else min(8, b))
        ol = max(1, min(20, b - il))
        WARMUP_SHAPES.append((il, ol))
        prev = b

    for il, ol in WARMUP_SHAPES:
        if il + ol > max_model_len:
            continue
        encoded = make_prompt_token_ids(tokenizer, il, batch_size)
        sp = _make_sampling_params(ol)
        try:
            t0 = time.perf_counter()
            llm.generate(
                prompts=[""] * batch_size,
                sampling_params=sp,
                prompt_token_ids=encoded,
            )
            print(f"  warmup (il={il}, ol={ol}): {time.perf_counter()-t0:.2f}s")
        except Exception as e:
            print(f"  [WARN] warmup (il={il}, ol={ol}) 실패: {e}")


def write_results(out_path, rows):
    fields = [
        "run_id", "combo_id", "combo_il", "combo_ol",
        "batch_size", "status", "sample_ids", "input_lens", "output_lens",
        "max_n_generated", "batch_ttft_ms", "batch_e2e_ms", "error",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="HF model id or local path (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    p.add_argument("--tp-degree", type=int, default=8,
                   help="tensor_parallel_size — RNGD 1 장 = 8 PE 이므로 "
                        "단일 device 전체 사용 시 8 (furiosa-llm 기본값)")
    p.add_argument("--pp-degree", type=int, default=1,
                   help="pipeline_parallel_size")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--devices", default=None,
                   help="device spec, e.g. 'npu:0:*' or 'npu:0:0-3'. None → auto-detect")
    p.add_argument("--profile-csv",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "profile.csv"))
    p.add_argument("--output-dir", default=None,
                   help=f"default: {RESULTS_ROOT}/<model>/bs<B>/")
    p.add_argument("--compiled-dir",
                   default="~/compiled_models_lens_profiling",
                   help="ArtifactBuilder save_dir (and LLM load path)")
    p.add_argument("--skip-compile", action="store_true",
                   help="이미 build 된 artifact 재사용")
    p.add_argument("--skip-warmup", action="store_true")
    p.add_argument("--buckets", default=None,
                   help="comma-separated bucket list, e.g. 128,256,512,1024,2048,4096,8192")
    p.add_argument("--num-pipeline-builder-workers", type=int, default=4)
    p.add_argument("--num-compile-workers", type=int, default=4)
    p.add_argument("--max-io-memory-mb", type=int, default=None,
                   help="LLM 로드 시 IO(워크스페이스) 예약 상한. KV 풀이 HBM 을 탐욕적으로 "
                        "선점하므로 큰 bucket/긴 시퀀스 워크스페이스 확보용으로 키운다 "
                        "(14B full bucket + max_len 8192 → 8192 권장)")
    p.add_argument("--max-processing-samples", type=int, default=None,
                   help="scheduler 동시 처리 샘플 수. bs=1 profiling 에선 작게(예: 4) 두면 "
                        "KV 풀 예약이 줄어 메모리 여유 확보")
    args = p.parse_args()

    bucket_list = parse_buckets_arg(args.buckets)

    if args.output_dir is None:
        args.output_dir = default_output_dir(args.model, args.batch_size)
    args.output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[output] {args.output_dir}")

    combos = load_profile(args.profile_csv)
    print(f"[{datetime.now()}] loaded {len(combos)} combos from {args.profile_csv}\n")

    llm, tokenizer, buckets = init_model(
        args.model, args.tp_degree, args.pp_degree,
        args.batch_size, args.max_model_len,
        args.compiled_dir, args.devices,
        skip_compile=args.skip_compile, bucket_list=bucket_list,
        num_pipeline_builder_workers=args.num_pipeline_builder_workers,
        num_compile_workers=args.num_compile_workers,
        max_io_memory_mb=args.max_io_memory_mb,
        max_processing_samples=args.max_processing_samples,
    )

    if not args.skip_warmup:
        print(f"\n[{datetime.now()}] warming up each bucket ...")
        warmup_each_bucket(llm, tokenizer, args.batch_size, buckets, args.max_model_len)

    print(f"\n[{datetime.now()}] starting sweep: {len(combos)} combos × "
          f"{args.n_runs} runs × batch={args.batch_size}")
    all_rows = []
    t_sweep_start = time.perf_counter()
    for combo_id, il, ol in combos:
        t0 = time.perf_counter()
        rows = measure_combo(
            llm, tokenizer, il, ol,
            args.batch_size, args.n_runs, combo_id, args.max_model_len,
        )
        all_rows.extend(rows)
        ok = sum(1 for r in rows if r.get("status") == "OK")
        print(f"  [{combo_id:>2d}/{len(combos)-1}] il={il:>5} ol={ol:>5}  "
              f"OK={ok}/{args.n_runs}  {time.perf_counter()-t0:.1f}s")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"RNGD_bs{args.batch_size}_{ts}"
    out_csv = os.path.join(args.output_dir, f"{tag}.csv")
    out_json = os.path.join(args.output_dir, f"{tag}.json")
    write_results(out_csv, all_rows)

    meta = dict(
        hardware="RNGD",
        model=args.model,
        tp_degree=args.tp_degree, pp_degree=args.pp_degree,
        batch_size=args.batch_size, max_model_len=args.max_model_len,
        n_runs=args.n_runs, n_combos=len(combos),
        profile_csv=args.profile_csv,
        buckets=buckets,
        prefill_buckets=[(1, b) for b in buckets],
        decode_buckets=[(args.batch_size, b) for b in buckets],
        tokenwise_seq_lens=sorted({args.batch_size} | set(buckets)),
        devices=args.devices,
        run_timestamp=datetime.now().isoformat(),
        total_sweep_s=round(time.perf_counter() - t_sweep_start, 1),
    )
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[DONE]")
    print(f"  {out_csv}  ({len(all_rows)} rows)")
    print(f"  {out_json}")
    print(f"  sweep time: {meta['total_sweep_s']}s")


if __name__ == "__main__":
    main()
