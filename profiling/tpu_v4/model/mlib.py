#!/usr/bin/env python3
"""TPU_NEXT_MODEL 공통: model 목록, 인자 만들기, 측정 (Module 1 measure.py의 sync 정의 그대로).

인자 값: layer parameter는 ../layer.py layer_params(seed = layer 번호 + 1)와 같은 값이다
(stack_params(seed=0)과 같다). host(CPU backend)에서 만들어 numpy로 두고 측정마다 device_put 한다.
  - 같은 크기에서 N을 바꿔 잴 때 다시 만들지 않는다 (host cache).
  - scan의 [n_layers, ...] 배열은 host에서 쌓는다. device에서 jnp.stack 하면 8B 32층은 weight가 두 벌(28 GB)이 된다.
입력 사본 (TPU_NEXT_MODEL §3): x(또는 tokens)만 사본 2개를 돌려 쓴다. weight는 사본을 만들지 않는다.
  wcopies=2 이면 weight까지 사본 2개 (Step 1의 artifact 비교, N=1만).
"""
import gc
import glob
import os
import shutil
import statistics
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "module1")]
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import kernels as kn  # noqa: E402
import layer as ly  # noqa: E402
import measure as ms  # noqa: E402  (module1: _executions, module_name, hlo_ops)

VOCAB = 128256
MODELS = {  # name: (dims (D, INTER, NH, HD, KV), 실제 층 수 n)
    "1B": ((2048, 8192, 32, 64, 8), 16),
    "3B": ((3072, 8192, 24, 128, 8), 28),
    "8B": ((4096, 14336, 32, 128, 8), 32),
}
WARMUP, ITERS = ms.WARMUP, ms.ITERS
N_X = 2  # 입력 사본 수

_CPU = jax.devices("cpu")[0]
_LCACHE = {}  # (dims, layer seed, copy) -> [np arrays]


def _host(fn, *a, **k):
    with jax.default_device(_CPU):
        out = fn(*a, **k)
    return [np.asarray(x) for x in out] if isinstance(out, (list, tuple)) else np.asarray(out)


def host_layer(dims, i, wcopy=0):
    """layer i (0부터)의 parameter. layer.stack_params(seed=0)의 i번째 = layer_params(seed=i+1).
    wcopy>0 인 weight 사본은 seed를 100000·wcopy 만큼 옮긴다."""
    key = (dims, i, wcopy)
    if key not in _LCACHE:
        if any(k[0] != dims for k in _LCACHE):
            _LCACHE.clear()
            gc.collect()
        _LCACHE[key] = _host(ly.layer_params, *dims, seed=i + 1 + 100_000 * wcopy)
    return _LCACHE[key]


def _rand_host(seed, shape):
    return _host(lambda: ly._rand(jax.random.PRNGKey(seed), shape))


def _put(a):
    return jax.device_put(a, ms.device())


def stack_lp(dims, n, mode, wcopy=0):
    per = [host_layer(dims, i, wcopy) for i in range(n)]
    if mode == "unroll":
        return [_put(a) for p in per for a in p]
    return [_put(np.stack([p[j] for p in per])) for j in range(len(ly.PARAM_NAMES))]


def rope_tables(dims):
    shp = ly.param_shapes(*dims)
    return _put(_rand_host(31_001, shp["cos"])), _put(_rand_host(31_002, shp["sin"]))


def xs(dims, seq):
    return [_put(_rand_host(40_000 + c, (seq, dims[0]))) for c in range(N_X)]


def stack_job(dims, n, seq, mode, wcopies=1):
    """make_stack 함수와 arg_sets (사본마다 인자 list)."""
    fn = ly.make_stack(*dims, n_layers=n, mode=mode)
    cos, sin = rope_tables(dims)
    x = xs(dims, seq)
    lps = [stack_lp(dims, n, mode, w) for w in range(wcopies)]
    sets = [[x[c % N_X], cos, sin, *lps[c % wcopies]] for c in range(max(N_X, wcopies))]
    return fn, sets


def _emb(tokens, embed):
    return embed[tokens]


def _head(x, gf, lm_head):
    return kn.gemm(kn.rmsnorm(x[-1:], gf), lm_head)


def model_job(dims, n, seq, mode, part):
    """part: full = layer.make_model, emb = embed[tokens], head = final rmsnorm + lm_head (마지막 token)."""
    D = dims[0]
    toks = [_put(_host(lambda c=c: jax.random.randint(jax.random.PRNGKey(41_000 + c), (seq,), 0, VOCAB,
                                                        dtype=jnp.int32))) for c in range(N_X)]
    if part == "emb":
        embed = _put(_rand_host(42_001, (VOCAB, D)))
        return _emb, [[t, embed] for t in toks]
    gf, lm_head = _put(_rand_host(42_002, (D,))), _put(_rand_host(42_003, (D, VOCAB)))
    if part == "head":
        return _head, [[x, gf, lm_head] for x in xs(dims, seq)]
    embed = _put(_rand_host(42_001, (VOCAB, D)))
    cos, sin = rope_tables(dims)
    lp = stack_lp(dims, n, mode)
    fn = ly.make_model(*dims, n_layers=n, vocab=VOCAB, mode=mode)
    return fn, [[t, embed, cos, sin, *lp, gf, lm_head] for t in toks]


def measure(fn, arg_sets, warmup=WARMUP, iters=ITERS, keep_trace=None):
    """module1 measure.measure와 같은 정의 (sync_min = 두 core 모두 시작한 뒤 ~ 마지막 종료).
    추가로 compile 시간과 memory analysis를 돌려준다."""
    t0 = time.perf_counter()
    compiled = jax.jit(fn).lower(*arg_sets[0]).compile()
    compile_s = time.perf_counter() - t0
    name = ms.module_name(compiled)
    mem = {}
    try:
        ma = compiled.memory_analysis()
        for k in ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes",
                  "generated_code_size_in_bytes", "alias_size_in_bytes"):
            mem[k] = getattr(ma, k, None)
    except Exception:  # noqa: BLE001
        pass

    def run(i):
        jax.block_until_ready(compiled(*arg_sets[i % len(arg_sets)]))

    for i in range(warmup):
        run(i)
    walls = []
    for i in range(iters):
        t = time.perf_counter()
        run(i)
        walls.append((time.perf_counter() - t) * 1e6)
    tdir = keep_trace or tempfile.mkdtemp(prefix="mdtr_")
    try:
        with jax.profiler.trace(tdir):
            for i in range(iters):
                run(i)
        paths = glob.glob(os.path.join(tdir, "**", "*.xplane.pb"), recursive=True)
        ex = ms._executions(paths[-1], name) if paths else []
    finally:
        if keep_trace is None:
            shutil.rmtree(tdir, ignore_errors=True)
    syncs = [c for _, c in ex]
    ok = len(syncs) == iters and min(syncs, default=0) > 0
    try:
        peak = ms.device().memory_stats().get("peak_bytes_in_use")
    except Exception:  # noqa: BLE001
        peak = None
    return {"sync_min_us": min(syncs) if ok else 0.0, "sync_med_us": statistics.median(syncs) if ok else 0.0,
            "wall_min_us": min(walls), "n_samples": len(syncs), "compile_s": compile_s, "mem": mem,
            "peak_bytes": peak, "hlo_ops": ms.hlo_ops(compiled), "module": name}


def free():
    gc.collect()
    jax.clear_caches()
