#!/usr/bin/env python3
"""Step 7.1 안 본 순서: chain에 없는 kernel type 순서의 3-token 직선 chain.

kernel 정의는 kernels.KERNELS를 그대로 쓴다. data는 일직선으로 흐른다 (token k의 출력을 token k+1이 쓴다).
mul·add의 두 번째 operand와 rope의 cos·sin, gemm weight, rmsnorm gain은 인자로 준다.

shape 규칙 (크기 D, INTER, HD; seq S): 입력 x[S, D]에서 시작한다.
  gemm: 현재 폭이 D면 → INTER, 아니면 → D  (Llama의 up/down 폭을 번갈아 쓴다)
  rmsnorm·silu·rope·mul·add: 폭 유지 (rope는 폭/HD개 head)

window 반환 규칙은 kn.window와 같다: window 안에서만 쓰이는 값은 반환하지 않는다
(직선 chain이므로 window의 마지막 token 출력만 반환된다).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import kernels as kn  # noqa: E402

# chain(Tokenwise)의 W3 type 순서 12가지에 없는 순서. 모든 type 쌍이 고루 나오도록 골랐다.
SEQS = [
    ("gemm", "rmsnorm", "gemm"),
    ("gemm", "add", "gemm"),
    ("gemm", "gemm", "gemm"),
    ("add", "gemm", "silu"),
    ("silu", "gemm", "add"),
    ("rmsnorm", "gemm", "gemm"),
    ("mul", "gemm", "rmsnorm"),
    ("rope", "gemm", "add"),
    ("gemm", "silu", "mul"),
    ("add", "rmsnorm", "rope"),
    ("rmsnorm", "silu", "gemm"),
    ("gemm", "mul", "add"),
]


def chain_spec(types, D, INTER, NH, HD, KV, S):
    """(spec, shapes). spec = [(name, kind, data_inputs, params)], 이름 t0,t1,t2."""
    shp = {"x": (S, D), "cos": (1, 1, HD), "sin": (1, 1, HD)}
    spec = []
    cur, width = "x", D
    for k, kind in enumerate(types):
        name = f"t{k}"
        params, ins = [], [cur]
        if kind == "gemm":
            n = INTER if width == D else D
            params = [f"w{k}"]
            shp[f"w{k}"] = (width, n)
            width = n
        elif kind == "rmsnorm":
            params = [f"g{k}"]
            shp[f"g{k}"] = (width,)
        elif kind == "rope":
            params = ["cos", "sin"]
        elif kind in ("mul", "add"):
            ins = [cur, f"y{k}"]
            shp[f"y{k}"] = (S, width)
        shp[name] = (S, width)
        spec.append((name, kind, ins, params))
        cur = name
    return spec, shp


def window(spec, lo, hi):
    """spec[lo:hi]를 함수로. 반환 (fn, arg_names, out_names)."""
    toks = spec[lo:hi]
    inside = [t[0] for t in toks]
    arg_names = []
    for _, _, ins, params in toks:
        for a in ins + params:
            if a not in inside and a not in arg_names:
                arg_names.append(a)
    out_names = [n for n in inside
                 if n == spec[-1][0] or any(n in t[2] for t in spec if t[0] not in inside)]

    def fn(*args):
        env = dict(zip(arg_names, args))
        for name, kind, ins, params in toks:
            env[name] = kn.KERNELS[kind](*[env[a] for a in ins + params])
        return tuple(env[n] for n in out_names)

    return fn, arg_names, out_names


def seq_name(types):
    return "-".join(types)


def windows_of(types):
    """[(lo, hi)]: U 3개, W2 2개, W3 1개."""
    n = len(types)
    return [(i, i + L) for L in range(1, n + 1) for i in range(n - L + 1)]
