#!/usr/bin/env python3
"""Rebuild a furiosa-llm artifact with per-kernel profiling ENABLED.

furiosa-llm forces enable_tuc_profile=False at build time (compiler_config.py:173),
so the stock artifact emits no per-kernel NPU spans. Override it so the compiled
program allocates profile buffers; then a run with FURIOSA_PROFILER_OUTPUT_PATHS
(+ TUC_PROFILE_LEVEL, RUST_LOG=span::tuc=info) dumps a Chrome trace with per-kernel
matmul/attention timings -- the REAL production-compiled latencies.

Usage:
  python build_profiled_artifact.py <model_id_or_path> <save_dir> [num_compile_workers]
"""
import sys

from furiosa_llm.artifact.builder import ArtifactBuilder

try:
    from furiosa_llm.artifact.types.config import CompilerConfig
except Exception:
    from furiosa_llm.artifact.types import CompilerConfig  # fallback

model = sys.argv[1]
save_dir = sys.argv[2]
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8

print(f"[build] model={model} -> {save_dir} (workers={workers})", flush=True)
print(f"[build] overrides: enable_tuc_profile=True, profile_sync=True", flush=True)

builder = ArtifactBuilder(
    model,
    compiler_config=CompilerConfig(
        compiler_config_overrides={"enable_tuc_profile": True, "profile_sync": True},
    ),
)
builder.build(save_dir, num_compile_workers=workers, num_pipeline_builder_workers=1)
print(f"[build] DONE -> {save_dir}", flush=True)
