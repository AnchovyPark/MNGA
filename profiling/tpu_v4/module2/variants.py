#!/usr/bin/env python3
"""(a) token 나누기 변형: B 덩어리(o..add2)를 다른 token 단위·순서로 나눠 telescope를 본다.

kernels.py의 kernel 정의와 CHAIN은 그대로 쓴다. 변형 token은 CHAIN token(op)들의 묶음이고,
window = 그 묶음들의 op 합집합이다. window 함수는 op 집합으로 만든다 (CHAIN 순서 = 위상 순서).
반환값 규칙은 kn.window와 같다: 집합 밖 token이 쓰거나 CHAIN_OUT인 값만 반환한다.

변형 (Step 2 진단과 full B의 XLA fusion: {o,add1,norm2 reduce} {norm2·gate} {norm2·up·silu·mul} {down,add2}):
  V0  o | add1 | norm2 | gate | silu | up | mul | down | add2          (원래 순서, kernel 단위)
  V1  o | add1 | norm2 | gate | up | silu | mul | down | add2          (순서만: silu를 up 뒤로)
  V2  o | add1.norm2 | gate | up | silu.mul | down.add2                (elementwise를 이웃에 붙임)
  V3  o.add1.norm2 | gate | up.silu.mul | down.add2                    (XLA fusion 단위)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import kernels as kn  # noqa: E402

VARIANTS = {
    "V0": [("o",), ("add1",), ("norm2",), ("gate",), ("silu",), ("up",), ("mul",), ("down",), ("add2",)],
    "V1": [("o",), ("add1",), ("norm2",), ("gate",), ("up",), ("silu",), ("mul",), ("down",), ("add2",)],
    "V2": [("o",), ("add1", "norm2"), ("gate",), ("up",), ("silu", "mul"), ("down", "add2")],
    "V3": [("o", "add1", "norm2"), ("gate",), ("up", "silu", "mul"), ("down", "add2")],
}
ORDER = {t: i for i, t in enumerate(kn.TOKENS)}


def set_name(ops):
    """op 집합의 이름: CHAIN 순서로 '+' 연결. 연속 구간이면 kn.window_name과 같다."""
    return "+".join(sorted(set(ops), key=ORDER.__getitem__))


def set_window(ops):
    """op 집합을 하나의 함수로. 반환 (fn, arg_names, out_names) — kn.window와 같은 규약."""
    inside = sorted(set(ops), key=ORDER.__getitem__)
    toks = [t for t in kn.CHAIN if t[0] in inside]
    arg_names = []
    for _, _, ins, params in toks:
        for a in ins + params:
            if a not in inside and a not in arg_names:
                arg_names.append(a)
    out_names = [n for n in inside
                 if n in kn.CHAIN_OUT or any(n in t[2] for t in kn.CHAIN if t[0] not in inside)]

    def fn(*args):
        env = dict(zip(arg_names, args))
        for name, kind, ins, params in toks:
            env[name] = kn.KERNELS[kind](*[env[a] for a in ins + params])
        return tuple(env[n] for n in out_names)

    return fn, arg_names, out_names


def variant_windows(var, max_n=3):
    """변형의 window 목록: [(j, L, op 집합)] (j = 변형 token index, L = 길이) + full (j=0, L=n)."""
    toks = VARIANTS[var]
    n = len(toks)
    out = []
    for L in range(1, min(max_n, n) + 1):
        for j in range(n - L + 1):
            out.append((j, L, tuple(o for t in toks[j:j + L] for o in t)))
    if max_n < n:
        out.append((0, n, tuple(o for t in toks for o in t)))
    return out


def all_sets(max_n=3):
    """모든 변형에 필요한 op 집합 (이름으로 중복 제거)."""
    seen = {}
    for v in VARIANTS:
        for _, _, ops in variant_windows(v, max_n):
            seen.setdefault(set_name(ops), ops)
    return list(seen.values())
