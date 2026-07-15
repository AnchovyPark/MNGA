#!/usr/bin/env python3
"""Run a profiling-enabled furiosa-llm artifact and dump per-kernel Chrome trace.

The artifact must have been built with enable_tuc_profile=True (build_profiled_artifact.py).
Per-kernel NPU spans are emitted by the native telemetry layer, controlled by env vars set
BEFORE launch (they are read at runtime init):

  FURIOSA_PROFILER_OUTPUT_PATHS=<dir_or_file>
  TUC_PROFILE_LEVEL=info
  RUST_LOG=span::tuc=info

Usage (env vars in the launching shell):
  FURIOSA_PROFILER_OUTPUT_PATHS=~/trace_out TUC_PROFILE_LEVEL=info RUST_LOG=span::tuc=info \
    python run_profiled_dump.py <artifact_dir> [prompt_tokens]
"""
import os
import sys
import time

import torch  # noqa: F401
import furiosa.torch  # noqa: F401
from furiosa_llm import LLM, SamplingParams

art = sys.argv[1]
n_tok = int(sys.argv[2]) if len(sys.argv) > 2 else 128

print(f"[run] artifact={art}", flush=True)
print(f"[run] FURIOSA_PROFILER_OUTPUT_PATHS={os.environ.get('FURIOSA_PROFILER_OUTPUT_PATHS')}", flush=True)
print(f"[run] TUC_PROFILE_LEVEL={os.environ.get('TUC_PROFILE_LEVEL')}", flush=True)

llm = LLM(art)
tok = llm.tokenizer

# controlled prefill prompt of ~n_tok tokens
base = ("The theory of relativity fundamentally changed physics by unifying space and "
        "time into a single continuum and describing gravity as curvature. ") * 40
ids = tok.encode(base)[:n_tok]
prompt = tok.decode(ids, skip_special_tokens=True)
print(f"[run] prompt ~{len(tok.encode(prompt))} tokens", flush=True)

# warmup
llm.generate(prompt, SamplingParams(max_tokens=1, temperature=0.0))
# timed (prefill = max_tokens=1)
t0 = time.perf_counter()
llm.generate(prompt, SamplingParams(max_tokens=1, temperature=0.0))
print(f"[run] prefill generate: {(time.perf_counter()-t0)*1000:.1f} ms", flush=True)

print("[run] DONE — check trace dir for per-kernel spans", flush=True)
