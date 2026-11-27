#!/usr/bin/env python3
"""Step 1.1: CPU 작은 크기에서 layer.py 확인 (unroll = scan = layer 하나씩, model = 직접 조립).

usage: JAX_PLATFORMS=cpu check_cpu.py   → check_cpu.log
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..")]
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import kernels as kn  # noqa: E402
import layer as ly  # noqa: E402


def diff(a, b):
    """상대 오차 max|a-b| / max|b|. random N(0,1) weight라 activation 크기가 층마다 커진다 (x ~ 1e4~1e5).
    bf16(유효 숫자 8 bit)에서 fusion 경계가 다르면 반올림이 달라지므로 1e-2 수준이면 같은 계산이다."""
    a, b = a.astype(jnp.float32), b.astype(jnp.float32)
    return float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))


def main():
    print("backend", jax.default_backend())
    for dims, n, seq, vocab in [((256, 512, 4, 64, 2), 3, 64, 1000), ((384, 1024, 6, 64, 3), 4, 128, 2048)]:
        P = len(ly.PARAM_NAMES)
        au = ly.stack_args(*dims, n_layers=n, seq=seq, mode="unroll")
        asc = ly.stack_args(*dims, n_layers=n, seq=seq, mode="scan")
        xu, kvu = jax.jit(ly.make_stack(*dims, n_layers=n, mode="unroll"))(*au)
        xs, (ks, vs) = jax.jit(ly.make_stack(*dims, n_layers=n, mode="scan"))(*asc)
        lay = jax.jit(ly.make_layer(*dims))
        x = au[0]
        dk = []
        for i in range(n):
            k, v, x = lay(x, au[1], au[2], *au[3 + i * P:3 + (i + 1) * P])
            dk += [diff(k, kvu[2 * i]), diff(v, kvu[2 * i + 1]), diff(k, ks[i]), diff(v, vs[i])]
        print(f"dims={dims} n={n} seq={seq} max|x|={float(jnp.max(jnp.abs(xu.astype(jnp.float32)))):.3g}: rel |unroll-scan| x={diff(xu, xs):.3g}  "
              f"|unroll-layerwise| x={diff(xu, x):.3g}  kv max={max(dk):.3g}")
        ma = ly.model_args(*dims, n_layers=n, vocab=vocab, seq=seq, mode="unroll")
        mas = ly.model_args(*dims, n_layers=n, vocab=vocab, seq=seq, mode="scan")
        lu, _ = jax.jit(ly.make_model(*dims, n_layers=n, vocab=vocab, mode="unroll"))(*ma)
        ls, _ = jax.jit(ly.make_model(*dims, n_layers=n, vocab=vocab, mode="scan"))(*mas)
        tokens, embed, cos, sin, lp, gf, lm = ma[0], ma[1], ma[2], ma[3], ma[4:-2], ma[-2], ma[-1]
        h, _ = jax.jit(ly.make_stack(*dims, n_layers=n, mode="unroll"))(embed[tokens], cos, sin, *lp)
        ref = kn.gemm(kn.rmsnorm(h[-1:], gf), lm)
        print(f"   model logits {lu.shape}: rel |unroll-scan|={diff(lu, ls):.3g}  |unroll-(emb+stack+head)|={diff(lu, ref):.3g}"
              f"  finite={bool(np.isfinite(np.asarray(lu, np.float32)).all())}")


if __name__ == "__main__":
    main()
