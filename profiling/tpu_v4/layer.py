#!/usr/bin/env python3
"""TPU 공통 (3차 범위): attention을 포함한 layer 1개와 prefill model 전체.

kernels.py, kernels_attn.py의 정의만 조합한다. 새 kernel 정의는 없다.
두 VM이 같은 layer를 재도록, layer와 model은 이 파일의 함수로만 만든다.
지시서: TPU_NEXT_LAYER.md (layer 안 n-gram), TPU_NEXT_MODEL.md (layer 사이, full model).

상태: JAX CPU backend에서만 실행을 확인했다 (2026-11-26). TPU에서는 아직 실행하지 않았다.

layer 1개 = CHAIN 앞 덩어리 A (norm1 .. v) -> attn -> CHAIN 뒤 덩어리 B (o .. add2).
attn은 kernels_attn의 fused attn(qk, scale, mask, softmax, av)이다. B의 `o` 입력이 attn 출력이다.
layer가 밖으로 내보내는 값: rope_k, v (KV cache로 HBM에 쓰인다), add2 (다음 layer 입력).

    import layer as ly
    dims = (2048, 8192, 32, 64, 8)                 # D, INTER, NH, HD, KV
    f = ly.make_layer(*dims)                       # f(*ly.layer_args(*dims, seq)) -> (rope_k, v, add2)
    s = ly.make_stack(*dims, n_layers=4, mode="unroll")   # s(*ly.stack_args(*dims, 4, seq)) -> (x, kv)
    m = ly.make_model(*dims, n_layers=16, vocab=128256, mode="unroll")
    args = ly.model_args(*dims, n_layers=16, vocab=128256, seq=512, mode="unroll")
    logits, kv = jax.jit(m)(*args)
"""
import jax
import jax.numpy as jnp

import kernels as kn
import kernels_attn as ka

DT = kn.DT

# layer 1개의 parameter 이름과 순서 (layer_params, make_layer 인자 순서)
PARAM_NAMES = ["g1", "wq", "wk", "wv", "wo", "g2", "wg", "wu", "wd"]
# 모든 layer가 같이 쓰는 RoPE 표
ROPE_NAMES = ["cos", "sin"]


def param_shapes(D, INTER, NH, HD, KV):
    q, kv = NH * HD, KV * HD
    return {"g1": (D,), "wq": (D, q), "wk": (D, kv), "wv": (D, kv), "wo": (q, D),
            "g2": (D,), "wg": (D, INTER), "wu": (D, INTER), "wd": (INTER, D),
            "cos": (1, 1, HD), "sin": (1, 1, HD)}


def make_layer(D, INTER, NH, HD, KV):
    """layer 1개. f(x, cos, sin, g1, wq, wk, wv, wo, g2, wg, wu, wd) -> (rope_k, v, add2)."""
    attn = ka.make("attn", NH, KV, HD)

    def f(x, cos, sin, g1, wq, wk, wv, wo, g2, wg, wu, wd):
        norm1 = kn.rmsnorm(x, g1)
        rope_q = kn.rope(kn.gemm(norm1, wq), cos, sin)
        rope_k = kn.rope(kn.gemm(norm1, wk), cos, sin)
        v = kn.gemm(norm1, wv)
        a = attn(rope_q, rope_k, v)
        add1 = kn.add(x, kn.gemm(a, wo))
        norm2 = kn.rmsnorm(add1, g2)
        mul = kn.mul(kn.silu(kn.gemm(norm2, wg)), kn.gemm(norm2, wu))
        add2 = kn.add(add1, kn.gemm(mul, wd))
        return rope_k, v, add2

    return f


def make_stack(D, INTER, NH, HD, KV, n_layers, mode="unroll"):
    """layer n_layers개만 쌓은 것 (embedding, lm_head 없음). layer 사이 겹침 측정용.

    f(x, cos, sin, *layer params) -> (x_out, kv)
    mode="unroll": layer마다 parameter를 따로 받고 Python loop로 펼친다 (한 program).
                   layer params = 9 * n_layers개 배열을 layer 순서대로 펼친 것.
    mode="scan":   layer parameter를 첫 축으로 쌓아 lax.scan으로 돈다 (MaxText scan_layers 방식).
                   layer params = 9개 배열, 각각 [n_layers, ...].
    kv: unroll이면 layer마다 (rope_k, v)를 펼친 tuple, scan이면 [n_layers, S, KV*HD] 두 개.
    """
    layer = make_layer(D, INTER, NH, HD, KV)
    P = len(PARAM_NAMES)

    if mode == "unroll":
        def stack(x, cos, sin, *lp):
            kv = []
            for i in range(n_layers):
                k, v, x = layer(x, cos, sin, *lp[i * P:(i + 1) * P])
                kv += [k, v]
            return x, tuple(kv)
        return stack

    if mode == "scan":
        def stack(x, cos, sin, *lp):
            def body(h, p):
                k, v, y = layer(h, cos, sin, *p)
                return y, (k, v)
            return jax.lax.scan(body, x, tuple(lp))
        return stack

    raise ValueError(mode)


def make_model(D, INTER, NH, HD, KV, n_layers, vocab, mode="unroll"):
    """prefill 전체: embedding -> layer n_layers개 -> final rmsnorm -> lm_head(마지막 token).

    인자: tokens[S] int32, embed[vocab, D], cos, sin, layer params(make_stack과 같음), gf[D], lm_head[D, vocab]
    반환: (logits[1, vocab], kv)
    """
    stack = make_stack(D, INTER, NH, HD, KV, n_layers, mode)

    def model(tokens, embed, cos, sin, *rest):
        lp, gf, lm_head = rest[:-2], rest[-2], rest[-1]
        x, kv = stack(embed[tokens], cos, sin, *lp)
        return kn.gemm(kn.rmsnorm(x[-1:], gf), lm_head), kv

    return model


def _rand(key, shape):
    return jax.random.normal(key, shape, dtype=jnp.float32).astype(DT)


def layer_params(D, INTER, NH, HD, KV, seed=0):
    """layer 1개의 parameter (PARAM_NAMES 순서). random bf16."""
    shp = param_shapes(D, INTER, NH, HD, KV)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(PARAM_NAMES))
    return [_rand(k, shp[n]) for k, n in zip(keys, PARAM_NAMES)]


def layer_args(D, INTER, NH, HD, KV, seq, seed=0):
    """make_layer 함수의 인자 전체: x, cos, sin, layer parameter."""
    shp = param_shapes(D, INTER, NH, HD, KV)
    k = jax.random.split(jax.random.PRNGKey(10_000 + seed), 3)
    return ([_rand(k[0], (seq, D)), _rand(k[1], shp["cos"]), _rand(k[2], shp["sin"])]
            + layer_params(D, INTER, NH, HD, KV, seed))


def stack_params(D, INTER, NH, HD, KV, n_layers, mode="unroll", seed=0):
    """make_stack의 layer params. layer마다 다른 seed라 weight buffer가 전부 다르다."""
    per = [layer_params(D, INTER, NH, HD, KV, seed=seed * 1000 + i + 1) for i in range(n_layers)]
    if mode == "unroll":
        return [a for p in per for a in p]
    if mode == "scan":
        return [jnp.stack([p[j] for p in per]) for j in range(len(PARAM_NAMES))]
    raise ValueError(mode)


def stack_args(D, INTER, NH, HD, KV, n_layers, seq, mode="unroll", seed=0):
    """make_stack 함수의 인자 전체: x, cos, sin, layer params."""
    shp = param_shapes(D, INTER, NH, HD, KV)
    k = jax.random.split(jax.random.PRNGKey(30_000 + seed), 3)
    return ([_rand(k[0], (seq, D)), _rand(k[1], shp["cos"]), _rand(k[2], shp["sin"])]
            + stack_params(D, INTER, NH, HD, KV, n_layers, mode, seed))


def model_args(D, INTER, NH, HD, KV, n_layers, vocab, seq, mode="unroll", seed=0):
    """make_model 함수의 인자 전체. layer마다 다른 seed라 weight buffer가 전부 다르다.

    8B 32층이면 weight가 약 16 GB다. 배열은 jax.random으로 기본 device에서 바로 만든다.
    """
    shp = param_shapes(D, INTER, NH, HD, KV)
    k = jax.random.split(jax.random.PRNGKey(20_000 + seed), 6)
    tokens = jax.random.randint(k[0], (seq,), 0, vocab, dtype=jnp.int32)
    embed = _rand(k[1], (vocab, D))
    cos, sin = _rand(k[2], shp["cos"]), _rand(k[3], shp["sin"])
    gf, lm_head = _rand(k[4], (D,)), _rand(k[5], (D, vocab))
    lp = stack_params(D, INTER, NH, HD, KV, n_layers, mode, seed)
    return [tokens, embed, cos, sin] + lp + [gf, lm_head]
