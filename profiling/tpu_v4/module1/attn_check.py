#!/usr/bin/env python3
"""2차 범위 gate: kernels_attn 정의를 TPU에서 fp32 참조(GQA repeat, 직접 계산)와 비교하고 HLO 연산 목록을 본다.
usage: attn_check.py"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..")]
import jax
import jax.numpy as jnp
import numpy as np

import kernels_attn as ka
import measure as ms


def ref(q, k, v, NH, KV, HD):
    S = q.shape[0]
    q = q.astype(jnp.float32).reshape(S, NH, HD).transpose(1, 0, 2)
    k = jnp.repeat(k.astype(jnp.float32).reshape(S, KV, HD).transpose(1, 0, 2), NH // KV, axis=0)
    v = jnp.repeat(v.astype(jnp.float32).reshape(S, KV, HD).transpose(1, 0, 2), NH // KV, axis=0)
    s = q @ k.transpose(0, 2, 1) / np.sqrt(HD)
    s = jnp.where(np.tril(np.ones((S, S), bool)), s, -1e30)
    p = jax.nn.softmax(s, axis=-1)
    return (p @ v).transpose(1, 0, 2).reshape(S, NH * HD)


for NH, KV, HD, S in [(16, 4, 64, 256), (40, 8, 128, 512), (24, 6, 64, 384)]:
    keys = jax.random.split(jax.random.PRNGKey(1), 3)
    q, k, v = [jax.random.normal(kk, s, jnp.float32).astype(ka.DT)
               for kk, s in zip(keys, ka.arg_shapes("attn", S, NH, KV, HD))]
    out = jax.jit(ka.make("attn", NH, KV, HD))(q, k, v)
    r = jax.jit(lambda a, b, c: ref(a, b, c, NH, KV, HD))(q, k, v)
    err = float(jnp.max(jnp.abs(out.astype(jnp.float32) - r)))
    # 분해 경로도 같아야 한다
    s = ka.make("qk", NH, KV, HD)(q, k)
    for t in ("scale", "mask", "softmax"):
        s = jax.jit(ka.make(t, NH, KV, HD))(s)
    o2 = jax.jit(ka.make("av", NH, KV, HD))(s, v)
    err2 = float(jnp.max(jnp.abs(o2.astype(jnp.float32) - r)))
    print(f"NH={NH} KV={KV} HD={HD} S={S}: attn max|err|={err:.4f}  split max|err|={err2:.4f}  "
          f"|ref|max={float(jnp.max(jnp.abs(r))):.2f}")
    for t in ka.TYPES:
        args = [jnp.zeros(sh, ka.DT) for sh in ka.arg_shapes(t, S, NH, KV, HD)]
        c = jax.jit(ka.make(t, NH, KV, HD)).lower(*args).compile()
        print(f"   {t:8s} {ms.hlo_ops(c)}")
sys.stdout.flush()
os._exit(0)
