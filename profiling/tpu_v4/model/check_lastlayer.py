#!/usr/bin/env python3
"""Step 5 보조: make_model(unroll)에서 XLA가 마지막 layer를 마지막 token 1행으로 줄이는지 (HLO shape 확인).
usage: check_lastlayer.py → check_lastlayer.log.  head가 x[-1:]만 쓰므로 마지막 layer의 MLP·o·q·attention 출력은
마지막 행만 필요하다 (rope_k, v는 KV cache로 전부 나간다)."""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import jax  # noqa: E402
import mlib  # noqa: E402
import layer as ly  # noqa: E402

out = open(os.path.join(_HERE, "check_lastlayer.log"), "w")
for name in ("1B", "3B", "8B"):
    dims, n = mlib.MODELS[name]
    D, I = dims[0], dims[1]
    for seq in (128, 512, 1024, 2048):
        for mode in ("unroll", "scan"):
            fn, sets = mlib.model_job(dims, n, seq, mode, "full")
            txt = jax.jit(fn).lower(*sets[0]).compile().as_text()
            # 1행짜리 MLP 중간값 bf16[1,INTER] 과 1행 attention 출력이 있는지
            n_i1 = len(re.findall(rf"bf16\[1,{I}\]", txt))
            n_is = len(re.findall(rf"bf16\[{seq},{I}\]", txt))
            line = f"{name} seq {seq:5d} {mode:6s}: HLO에 bf16[1,{I}] {n_i1}회, bf16[{seq},{I}] {n_is}회"
            print(line)
            out.write(line + "\n")
            del sets
            mlib.free()
