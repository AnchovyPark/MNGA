#!/usr/bin/env python3
"""TPU module 공통: kernel 정의, Tokenwise chain, window 생성.

규칙은 repo root의 TPU_MODULES_COMMON.md §2~3에 있다. Module 1과 Module 2가
같은 정의로 측정해야 U가 서로 맞으므로, kernel 정의와 CHAIN은 바꾸지 않는다.

상태: JAX CPU backend에서만 실행을 확인했다 (2026-11-26). TPU에서는 아직
실행하지 않았다.

사용:
    import kernels as kn
    shp = kn.shapes(D=2048, INTER=8192, NH=32, HD=64, KV=8, S=512)
    fn, arg_names, out_names = kn.window(9, 12)      # gate, silu, up
    args = kn.make_args(arg_names, shp)
    outs = jax.jit(fn)(*args)                        # out_names 순서의 tuple
"""
import jax
import jax.numpy as jnp

DT = jnp.bfloat16
EPS = 1e-5


# ── kernel 정의 ──────────────────────────────────────────────────────────────
def gemm(x, w):
    """x[M,K] @ w[K,N]."""
    return x @ w


def rmsnorm(x, g):
    """x[M,N], g[N]. 제곱 평균은 fp32로 누산한다 (RNGD WChain과 같은 정의)."""
    v = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(v.astype(DT) + EPS) * g


def rope(t, cos, sin):
    """t[S, heads*HD], cos·sin[1,1,HD]."""
    hd = cos.shape[-1]
    t3 = t.reshape(t.shape[0], -1, hd)
    x1, x2 = t3[..., : hd // 2], t3[..., hd // 2:]
    return (t3 * cos + jnp.concatenate((-x2, x1), axis=-1) * sin).reshape(t.shape)


def silu(x):
    return jax.nn.silu(x)


def mul(x, y):
    return x * y


def add(x, y):
    return x + y


KERNELS = {"gemm": gemm, "rmsnorm": rmsnorm, "rope": rope,
           "silu": silu, "mul": mul, "add": add}


# ── Tokenwise chain (Llama layer에서 attention core를 뺀 부분) ───────────────
# (token, kernel type, data 입력, parameter)
# data 입력 이름이 CHAIN의 token이 아니면 chain 밖에서 오는 값이다 (x, attn_out).
CHAIN = [
    ("norm1",  "rmsnorm", ["x"],             ["g1"]),
    ("q",      "gemm",    ["norm1"],         ["wq"]),
    ("rope_q", "rope",    ["q"],             ["cos", "sin"]),
    ("k",      "gemm",    ["norm1"],         ["wk"]),
    ("rope_k", "rope",    ["k"],             ["cos", "sin"]),
    ("v",      "gemm",    ["norm1"],         ["wv"]),
    ("o",      "gemm",    ["attn_out"],      ["wo"]),
    ("add1",   "add",     ["x", "o"],        []),
    ("norm2",  "rmsnorm", ["add1"],          ["g2"]),
    ("gate",   "gemm",    ["norm2"],         ["wg"]),
    ("silu",   "silu",    ["gate"],          []),
    ("up",     "gemm",    ["norm2"],         ["wu"]),
    ("mul",    "mul",     ["silu", "up"],    []),
    ("down",   "gemm",    ["mul"],           ["wd"]),
    ("add2",   "add",     ["add1", "down"],  []),
]
TOKENS = [t[0] for t in CHAIN]
# chain 전체를 빠져나가 다음 단계(attention core, 다음 layer)가 쓰는 값
CHAIN_OUT = ["rope_q", "rope_k", "v", "add2"]


def shapes(D, INTER, NH, HD, KV, S):
    """chain에 나오는 모든 값(외부 입력, token 출력, parameter)의 shape."""
    q, kv = NH * HD, KV * HD
    return {
        "x": (S, D), "attn_out": (S, q),
        "norm1": (S, D), "q": (S, q), "rope_q": (S, q),
        "k": (S, kv), "rope_k": (S, kv), "v": (S, kv),
        "o": (S, D), "add1": (S, D), "norm2": (S, D),
        "gate": (S, INTER), "silu": (S, INTER), "up": (S, INTER),
        "mul": (S, INTER), "down": (S, D), "add2": (S, D),
        "g1": (D,), "g2": (D,),
        "wq": (D, q), "wk": (D, kv), "wv": (D, kv), "wo": (q, D),
        "wg": (D, INTER), "wu": (D, INTER), "wd": (INTER, D),
        "cos": (1, 1, HD), "sin": (1, 1, HD),
    }


def window(lo, hi):
    """CHAIN[lo:hi]를 하나의 함수로 묶는다. 반환: (fn, arg_names, out_names).

    - arg_names: window 밖에서 오는 data 입력과 window 안 kernel의 parameter.
      전부 fn의 인자다 (closure로 잡지 않는다).
    - out_names: window 안 token 중 window 밖에서 쓰이는 값. window 안에서만
      쓰이는 중간값은 반환하지 않는다 (실제 model에서 materialize되지 않는 값을
      강제로 materialize시키지 않기 위해서다).
    - fn(*args)는 out_names 순서의 tuple을 돌려준다.
    """
    toks = CHAIN[lo:hi]
    if not toks:
        raise ValueError(f"empty window: [{lo}:{hi}]")
    inside = [t[0] for t in toks]
    arg_names = []
    for _, _, ins, params in toks:
        for a in ins + params:
            if a not in inside and a not in arg_names:
                arg_names.append(a)
    out_names = [
        n for n in inside
        if n in CHAIN_OUT or any(n in t[2] for t in CHAIN if t[0] not in inside)
    ]

    def fn(*args):
        env = dict(zip(arg_names, args))
        for name, kind, ins, params in toks:
            env[name] = KERNELS[kind](*[env[a] for a in ins + params])
        return tuple(env[n] for n in out_names)

    return fn, arg_names, out_names


def make_args(arg_names, shp, seed=0):
    """arg_names 순서의 random bf16 입력. latency는 값과 무관하다."""
    keys = jax.random.split(jax.random.PRNGKey(seed), len(arg_names))
    return [jax.random.normal(k, shp[n], dtype=jnp.float32).astype(DT)
            for k, n in zip(keys, arg_names)]


def all_windows(max_n=3):
    """측정할 window의 (lo, hi) 목록: 길이 1..max_n 전부와 full."""
    n = len(CHAIN)
    wins = [(i, i + L) for L in range(1, max_n + 1) for i in range(n - L + 1)]
    return wins + [(0, n)]


def window_name(lo, hi):
    """data 행에 쓰는 이름. token 이름에 '_'가 있어서 '+'로 잇는다."""
    return "+".join(TOKENS[lo:hi])
