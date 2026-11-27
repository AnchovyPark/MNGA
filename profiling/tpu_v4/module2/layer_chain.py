#!/usr/bin/env python3
"""TPU_NEXT_LAYER: attention을 포함한 layer 1개의 token chain 두 가지 (공통 정의만 조합).

  L16: norm1..v (CHAIN 0:6) + attn (kernels_attn "attn", fused) + o..add2 (CHAIN 6:15)
  L20: norm1..v + qk, scale, mask, softmax, av (kernels_attn 분해 kernel) + o..add2

kernel 정의는 kernels.py와 kernels_attn.py 그대로다. o의 입력 attn_out은 attention 출력으로 바뀐다.
layer가 밖으로 내보내는 값은 layer.py의 make_layer와 같다: rope_k, v (KV cache), add2.
window 규칙은 kn.window와 같다: window 밖에서 쓰이는 값과 OUT만 반환한다.
(attn.py는 Step 7.2에서 먼저 만든 L20 변형이다: qk는 repeat 방식, mask -inf, OUT = add2만.)
"""
import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import kernels as kn  # noqa: E402
import kernels_attn as ka  # noqa: E402

DT = kn.DT
OUT = ["rope_k", "v", "add2"]

_ATT20 = [("qk", "qk", ["rope_q", "rope_k"], []),
          ("scale", "scale", ["qk"], []),
          ("mask", "mask", ["scale"], []),
          ("softmax", "softmax", ["mask"], []),
          ("av", "av", ["softmax", "v"], [])]
_ATT16 = [("attn", "attn", ["rope_q", "rope_k", "v"], [])]


def _join(att):
    last = att[-1][0]
    return kn.CHAIN[:6] + att + [(n, k, [last] if ins == ["attn_out"] else ins, p) for n, k, ins, p in kn.CHAIN[6:]]


CHAINS = {"L16": _join(_ATT16), "L20": _join(_ATT20)}
# 덩어리: A = norm1..v, ATT = attention, B = o..add2
SUB = {"L16": {"A": (0, 6), "ATT": (6, 7), "B": (7, 16)},
       "L20": {"A": (0, 6), "ATT": (6, 11), "B": (11, 20)}}


def tokens(chain):
    return [t[0] for t in CHAINS[chain]]


def shapes(D, INTER, NH, HD, KV, S):
    shp = kn.shapes(D, INTER, NH, HD, KV, S)
    sc = (NH, S, S)
    shp.update({"qk": sc, "scale": sc, "mask": sc, "softmax": sc, "av": (S, NH * HD), "attn": (S, NH * HD)})
    return shp


def kfun(kind, dims):
    D, INTER, NH, HD, KV = dims
    if kind in ka.TYPES:
        return ka.make(kind, NH, KV, HD)
    return kn.KERNELS[kind]


def window(chain, lo, hi, dims):
    spec = CHAINS[chain]
    toks = spec[lo:hi]
    inside = [t[0] for t in toks]
    arg_names = []
    for _, _, ins, params in toks:
        for a in ins + params:
            if a not in inside and a not in arg_names:
                arg_names.append(a)
    out_names = [n for n in inside if n in OUT or any(n in t[2] for t in spec if t[0] not in inside)]
    fns = [kfun(kind, dims) for _, kind, _, _ in toks]

    def fn(*args):
        env = dict(zip(arg_names, args))
        for (name, _, ins, params), f in zip(toks, fns):
            env[name] = f(*[env[a] for a in ins + params])
        return tuple(env[n] for n in out_names)

    return fn, arg_names, out_names


def window_name(chain, lo, hi):
    return "+".join(tokens(chain)[lo:hi])


def windows(chain, max_n):
    n = len(CHAINS[chain])
    wins = [(i, i + L) for L in range(1, max_n + 1) for i in range(n - L + 1)] + [(0, n)]
    return wins + [v for v in SUB[chain].values() if v not in wins]


def make_args(arg_names, shp, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), len(arg_names))
    out = []
    for k, n in zip(keys, arg_names):
        x = jax.random.normal(k, shp[n], dtype=jnp.float32)
        if n == "softmax":            # 확률 자리의 dummy는 확률처럼
            x = jax.nn.softmax(x, axis=-1)
        out.append(x.astype(DT))
    return out
