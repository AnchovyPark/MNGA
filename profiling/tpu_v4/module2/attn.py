#!/usr/bin/env python3
"""Step 7.2 attention core: token 5개를 chain에 넣어 실제 layer처럼 연결한다.

LAYER = CHAIN의 norm1..v (0~5) + [qk, scale, mask, softmax, av] + CHAIN의 o..add2 (o의 입력 = av).
kernels.py의 kernel 정의와 CHAIN은 그대로 쓰고, attention kernel만 여기서 추가한다.

attention kernel (causal, GQA, bf16):
  qk      : q[S, NH·HD], k[S, KV·HD] → scores[NH, S, S] (GQA: head h는 kv head h // (NH/KV))
  scale   : scores · 1/sqrt(HD)
  mask    : causal (j > i 위치를 -inf로)
  softmax : 마지막 축 softmax (fp32로 계산 후 bf16)
  av      : probs[NH, S, S] @ v[S, KV·HD] → [S, NH·HD]
window 반환 규칙은 kn.window와 같다 (window 밖에서 쓰이는 값과 LAYER_OUT만 반환).
"""
import math
import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import kernels as kn  # noqa: E402

DT = kn.DT


def _heads(t, hd):
    return t.reshape(t.shape[0], -1, hd)            # [S, H, HD]


def qk(q, k, cos):
    """cos는 HD를 알기 위한 인자 (shape만 쓴다)."""
    hd = cos.shape[-1]
    qh, kh = _heads(q, hd), _heads(k, hd)            # [S,NH,HD], [S,KV,HD]
    g = qh.shape[1] // kh.shape[1]
    kh = jnp.repeat(kh, g, axis=1)                   # [S,NH,HD]
    return jnp.einsum("qhd,khd->hqk", qh, kh)        # [NH,S,S]


def scale(s, cos):
    return s * jnp.asarray(1.0 / math.sqrt(cos.shape[-1]), DT)


def mask(s):
    S = s.shape[-1]
    m = jnp.tril(jnp.ones((S, S), dtype=bool))
    return jnp.where(m[None], s, jnp.asarray(-jnp.inf, DT))


def softmax(s):
    return jax.nn.softmax(s.astype(jnp.float32), axis=-1).astype(DT)


def av(p, v, cos):
    hd = cos.shape[-1]
    vh = _heads(v, hd)                               # [S,KV,HD]
    g = p.shape[0] // vh.shape[1]
    vh = jnp.repeat(vh, g, axis=1)                   # [S,NH,HD]
    o = jnp.einsum("hqk,khd->qhd", p, vh)            # [S,NH,HD]
    return o.reshape(o.shape[0], -1)


KERNELS = {**kn.KERNELS, "qk": qk, "scale": scale, "mask": mask, "softmax": softmax, "av": av}

LAYER = (kn.CHAIN[:6] + [
    ("qk",      "qk",      ["rope_q", "rope_k"], ["cos"]),
    ("scale",   "scale",   ["qk"],               ["cos"]),
    ("mask",    "mask",    ["scale"],            []),
    ("softmax", "softmax", ["mask"],             []),
    ("av",      "av",      ["softmax", "v"],     ["cos"]),
] + [(n, k, ["av"] if ins == ["attn_out"] else ins, p) for n, k, ins, p in kn.CHAIN[6:]])
TOKENS = [t[0] for t in LAYER]
LAYER_OUT = ["add2"]     # layer 전체가 내보내는 값 (다음 layer 입력)
SUB = {"A": (0, 6), "ATT": (6, 11), "B": (11, 20), "QKV_ATT": (0, 11), "ATT_B": (6, 20)}
# attention core(qk..av, 6:11)를 token 1개 "attn"으로 볼 때 (16 token) W2·W3에 더 필요한 window
ATT1_EXTRA = [(5, 11), (6, 12), (4, 11), (5, 12), (6, 13)]


def shapes(D, INTER, NH, HD, KV, S):
    shp = kn.shapes(D, INTER, NH, HD, KV, S)
    shp.update({"qk": (NH, S, S), "scale": (NH, S, S), "mask": (NH, S, S),
                "softmax": (NH, S, S), "av": (S, NH * HD)})
    return shp


def window(lo, hi):
    toks = LAYER[lo:hi]
    inside = [t[0] for t in toks]
    arg_names = []
    for _, _, ins, params in toks:
        for a in ins + params:
            if a not in inside and a not in arg_names:
                arg_names.append(a)
    out_names = [n for n in inside
                 if n in LAYER_OUT or any(n in t[2] for t in LAYER if t[0] not in inside)]

    def fn(*args):
        env = dict(zip(arg_names, args))
        for name, kind, ins, params in toks:
            env[name] = KERNELS[kind](*[env[a] for a in ins + params])
        return tuple(env[n] for n in out_names)

    return fn, arg_names, out_names


def window_name(lo, hi):
    return "+".join(TOKENS[lo:hi])


def all_windows(max_n=3):
    n = len(LAYER)
    wins = [(i, i + L) for L in range(1, max_n + 1) for i in range(n - L + 1)]
    return wins + [(0, n)] + [v for v in SUB.values() if v not in wins]


def make_args(arg_names, shp, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), len(arg_names))
    out = []
    for k, n in zip(keys, arg_names):
        x = jax.random.normal(k, shp[n], dtype=jnp.float32)
        if n in ("softmax",):            # softmax 출력 자리의 dummy는 확률처럼
            x = jax.nn.softmax(x, axis=-1)
        out.append(x.astype(DT))
    return out
