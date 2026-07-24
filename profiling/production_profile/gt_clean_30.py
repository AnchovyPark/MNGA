#!/usr/bin/env python3
"""실제 1B prefill GT를 깨끗이: prefill supertask cycle 합 + wall-clock. (getting-started venv)

정확한 토큰수(bucket 딱 맞게)로 generate → 프로파일러 CSV에서 prefill supertask만 골라 합.
클럭 = cycle합 / device시간. 여러 seq에서 일관되면 겹침 없음(naive 정확), 안 맞으면 cross-layer 겹침.

사용: EDF_PROFILER_OUTPUT_PATH=<file> TUC_PROFILE_LEVEL=info RUST_LOG=info,span::tuc=info \
      <getting-started venv>/python gt_clean_30.py <artifact> <n_tok>
"""
import os, sys, time, csv, statistics as st

import torch  # noqa
import furiosa.torch  # noqa
from furiosa_llm import LLM, SamplingParams

art = sys.argv[1]
n_tok = int(sys.argv[2])
trace = os.environ["EDF_PROFILER_OUTPUT_PATH"]

llm = LLM(art)
tok = llm.tokenizer
base = ("The theory of relativity fundamentally changed physics by unifying space and "
        "time into a single continuum. ") * 200
ids = tok.encode(base)[:n_tok]
prompt = tok.decode(ids, skip_special_tokens=True)
ntok_real = len(tok.encode(prompt))
sp = SamplingParams(max_tokens=1, temperature=0.0)

llm.generate(prompt, sp)  # warmup (프로파일 파일 덮어씀)
# trace 비우고 timed 1회만
open(trace, "w").close()
ts = []
for _ in range(3):
    t0 = time.perf_counter(); llm.generate(prompt, sp); ts.append(time.perf_counter() - t0)
wall = min(ts) * 1000

# trace 파싱: prefill supertask (input_size == n_tok bucket) 만
rows = [r for r in csv.reader(open(trace)) if len(r) == 3 and r[0] != "leader_device"]
# 마지막 timed run만 반영되게 — prefill = 큰 Tokenwise/Attention. bucket 크기로 필터.
tw = [int(c) for _, n, c in rows if n.startswith("Tokenwise") and "input_size" in n]
at = [int(c) for _, n, c in rows if n.startswith("Attention")]
# prefill supertask는 값이 큼(decode는 seq=1이라 작음). 상위값=prefill.
tw_p = sorted(tw)[-16:] if len(tw) >= 16 else tw   # 큰 16개 = prefill 16층
at_p = sorted(at)[-16:] if len(at) >= 16 else at
tw_med = int(st.median(tw_p)); at_med = int(st.median(at_p))
one_fwd = 16 * tw_med + 16 * at_med
print(f"RESULT ntok={ntok_real} wall={wall:.1f}ms  TW_med={tw_med} Attn_med={at_med}  "
      f"prefill_cyc합={one_fwd}  클럭추정(합/wall)={one_fwd/(wall/1000)/1e9:.2f}GHz", flush=True)
sys.stdout.flush(); os._exit(0)
