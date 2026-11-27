#!/usr/bin/env python3
"""TPU_NEXT_LAYER §2: L16·L20 full window가 layer.make_layer와 같은 계산인지 CPU 작은 크기로 확인."""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import layer as ly  # noqa: E402
import layer_chain as lc  # noqa: E402

for dims, S in [((256, 512, 4, 64, 2), 64), ((384, 1024, 6, 64, 3), 128)]:
    args = ly.layer_args(*dims, S)
    names = ["x", "cos", "sin"] + ly.PARAM_NAMES
    byname = dict(zip(names, args))
    ref = jax.jit(ly.make_layer(*dims))(*args)
    for ch in ("L16", "L20"):
        n = len(lc.CHAINS[ch])
        fn, an, outs = lc.window(ch, 0, n, dims)
        got = jax.jit(fn)(*[byname[a] for a in an])
        assert outs == ["rope_k", "v", "add2"], outs
        for o, r, g in zip(outs, ref, got):
            r32, g32 = r.astype(jnp.float32), g.astype(jnp.float32)
            err = float(jnp.max(jnp.abs(r32 - g32)) / (jnp.max(jnp.abs(r32)) + 1e-9))
            print(dims, S, ch, o, f"max|err|/max|ref| = {err:.2e}", "OK" if err < 2e-2 else "MISMATCH")
