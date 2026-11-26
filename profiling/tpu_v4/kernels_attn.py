#!/usr/bin/env python3
"""TPU module 공통 (2차 범위): attention core kernel 정의.

kernels.py의 kernel 정의와 CHAIN은 바꾸지 않는다 (TPU_MODULES_COMMON.md §3). 이 파일은
새 함수만 더한다. Module 2가 attention token을 chain에 붙일 때도 이 정의를 쓰면 두 VM의 U가
같은 정의가 된다.

layout은 CHAIN과 맞춘다: q = rope_q [S, NH·HD], k = rope_k [S, KV·HD], v [S, KV·HD],
attention 출력은 attn_out [S, NH·HD] (CHAIN의 `o` 입력).
GQA: head h는 kv head h // (NH/KV)를 쓴다. score는 [NH, S, S] bf16.

  qk      : q, k      -> s [NH,S,S]       s = q·kᵀ (head별, bf16 출력)
  scale   : s         -> s · HD^-0.5
  mask    : s         -> causal (j > i 자리를 bf16 최솟값으로). mask는 iota로 만든다 (입력 없음)
  softmax : s         -> softmax(s) (fp32로 계산 후 bf16)
  av      : p, v      -> o [S, NH·HD]
  attn    : q, k, v   -> 위 다섯 개를 한 함수로 (TPU_INSTRUCTIONS.md Step 4의 `attn` token)

HD 같은 static 값이 필요하므로 make(type, NH, KV, HD)가 배열 인자만 받는 함수를 돌려준다.

    import kernels_attn as ka
    fn = ka.make("attn", NH=32, KV=8, HD=128)
    shps = ka.arg_shapes("attn", S=512, NH=32, KV=8, HD=128)
"""
import jax
import jax.numpy as jnp

DT = jnp.bfloat16
TYPES = ["qk", "scale", "mask", "softmax", "av", "attn"]
ELEMWISE = ("scale", "mask", "softmax")  # [NH,S,S]만 받는다. KV·HD와 무관


def _qk(q, k, NH, KV, HD):
    S, T = q.shape[0], k.shape[0]
    G = NH // KV
    q4 = q.reshape(S, KV, G, HD)
    k3 = k.reshape(T, KV, HD)
    return jnp.einsum("skgd,tkd->kgst", q4, k3).reshape(NH, S, T)


def _scale(s, HD):
    return s * jnp.asarray(HD ** -0.5, s.dtype)


def _mask(s):
    S, T = s.shape[-2:]
    i = jax.lax.broadcasted_iota(jnp.int32, (S, T), 0)
    j = jax.lax.broadcasted_iota(jnp.int32, (S, T), 1)
    return jnp.where(j <= i, s, jnp.finfo(s.dtype).min)


def _softmax(s):
    return jax.nn.softmax(s.astype(jnp.float32), axis=-1).astype(s.dtype)


def _av(p, v, NH, KV, HD):
    T = v.shape[0]
    S = p.shape[1]
    G = NH // KV
    p4 = p.reshape(KV, G, S, T)
    v3 = v.reshape(T, KV, HD)
    return jnp.einsum("kgst,tkd->skgd", p4, v3).reshape(S, NH * HD)


def make(t, NH, KV, HD):
    if t == "qk":
        return lambda q, k: _qk(q, k, NH, KV, HD)
    if t == "scale":
        return lambda s: _scale(s, HD)
    if t == "mask":
        return _mask
    if t == "softmax":
        return _softmax
    if t == "av":
        return lambda p, v: _av(p, v, NH, KV, HD)
    if t == "attn":
        def attn(q, k, v):
            s = _qk(q, k, NH, KV, HD)
            return _av(_softmax(_mask(_scale(s, HD))), v, NH, KV, HD)
        return attn
    raise ValueError(t)


def arg_shapes(t, S, NH, KV, HD):
    q, kv, sc = (S, NH * HD), (S, KV * HD), (NH, S, S)
    return {"qk": [q, kv], "scale": [sc], "mask": [sc], "softmax": [sc],
            "av": [sc, kv], "attn": [q, kv, kv]}[t]


def feats(t, S, NH, KV, HD):
    """(flops, bytes) — 각 kernel이 HBM에서 읽고 쓰는 배열 기준 (bf16)."""
    B = 2
    sc = NH * S * S
    if t == "qk":
        return 2.0 * sc * HD, B * (NH * S * HD + KV * S * HD + sc)
    if t == "av":
        return 2.0 * sc * HD, B * (sc + KV * S * HD + NH * S * HD)
    if t in ("scale", "mask"):
        return 1.0 * sc, B * 2 * sc
    if t == "softmax":
        return 5.0 * sc, B * 2 * sc
    if t == "attn":  # score를 HBM에 쓰지 않는다고 볼 때의 최소 bytes
        return 4.0 * sc * HD + 7.0 * sc, B * (2 * NH * S * HD + 2 * KV * S * HD)
    raise ValueError(t)
