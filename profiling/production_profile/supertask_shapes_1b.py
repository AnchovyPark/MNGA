#!/usr/bin/env python3
"""1B supertask들의 shape + 이론 roofline 추출 (비교의 '이론적 합산' 쪽).

- 아티팩트(~/llama1b_profiled/artifact.json)에서 실제 supertask I/O shape를 (가정 없이) 확인.
- 각 (supertask, seq)마다 op 구성/FLOPs/bytes/roofline floor 계산.
- 다음 단계에서 실측 supertask 시간과 비교할 것.

Llama-3.2-1B: D=2048, INTER=8192, NH=32, KV=8, HD=64, L=16, VOCAB=128256, TP=8.
Tokenwise(per-layer 비-attention) = norm + q,k,v,o proj + gate,up,down + silu + residual.
Attention(per-layer) = qk + softmax + av. (prefill: q_len=kv_len=S, kv_cache=0)
"""
import json
import csv
import sys

D, INTER, VOCAB, NH, HD, KV, L = 2048, 8192, 128256, 32, 64, 8, 16
PEAK = 2e15          # 2 PFLOPS bf16 (8 PE aggregate)
BW = 1e12            # HBM ~1 TB/s aggregate (가정; memory-bound 실측으로 나중에 보정)
ART = "/home/furiosa/llama1b_profiled/artifact.json"
SEQS = [128, 256, 384, 512, 1024]
BYTES = 2            # bf16


def p(*a): print(*a, flush=True)


def cfloor_us(flops): return flops / PEAK * 1e6
def mfloor_us(bytes_): return bytes_ / BW * 1e6


def tokenwise_roofline(S):
    """per-layer 비-attention: q,k,v,o proj + gate,up,down. (norm/silu/residual = 벡터, roofline엔 미미)"""
    mats = [  # (K, N)  — x@W
        (D, NH * HD),   # q
        (D, KV * HD),   # k
        (D, KV * HD),   # v
        (NH * HD, D),   # o
        (D, INTER),     # gate
        (D, INTER),     # up
        (INTER, D),     # down
    ]
    flops = sum(2 * S * k * n for k, n in mats)
    wbytes = sum(k * n * BYTES for k, n in mats)          # weight load (fusion이라 블록당 1회)
    # activation: 입력/출력만 HBM (중간은 on-chip 가정)
    abytes = (S * D + S * INTER) * BYTES                   # 대략 입출력
    bytes_ = wbytes + abytes
    c, m = cfloor_us(flops), mfloor_us(bytes_)
    return dict(flops=flops, wbytes=wbytes, bytes=bytes_, cfloor=c, mfloor=m, roofline=max(c, m))


def attention_roofline(S):
    """per-layer attention core: qk + av (softmax는 벡터). prefill q=kv=S, no cache. NH heads."""
    qk = 2 * NH * S * S * HD
    av = 2 * NH * S * S * HD
    flops = qk + av
    # bytes: scores(S*S) write/read + q,k,v 활성 (weight 없음)
    scores = NH * S * S * BYTES
    qkv_act = 3 * NH * S * HD * BYTES
    bytes_ = scores + qkv_act
    c, m = cfloor_us(flops), mfloor_us(bytes_)
    return dict(flops=flops, wbytes=0, bytes=bytes_, cfloor=c, mfloor=m, roofline=max(c, m))


def dump_artifact_shapes():
    a = json.load(open(ART))
    pipe = a["model"]["pipelines"][0]
    T = pipe["tensors"]
    p("=== 아티팩트 실제 supertask I/O shape (가정 검증용) ===")
    for kind in ("tokenwise", "attention"):
        for sname, s in pipe["stages"].items():
            if isinstance(s, dict) and s.get("kind") == kind:
                tasks = s.get("tasks", {})
                bkey = "128" if "128" in tasks else ("1,128,128" if "1,128,128" in tasks else list(tasks)[0])
                t = tasks[bkey]
                ins = [T[i]["shape"] for i in t["inputs"] if i in T]
                outs = [T[o]["shape"] for o in t["outputs"] if o in T]
                p(f"  [{kind} {sname} bucket={bkey}] in={ins}  out={outs}")
                break


def main():
    dump_artifact_shapes()
    p(f"\n=== 이론 roofline (peak={PEAK/1e15:.0f}PF, BW={BW/1e12:.0f}TB/s) ===")
    rows = []
    p(f"{'supertask':10s} {'seq':>5s} {'FLOP(G)':>8s} {'wt(MB)':>7s} {'cfloor':>8s} {'mfloor':>8s} {'roofline_us':>11s} {'bound':>7s}")
    for S in SEQS:
        for name, fn in [("Tokenwise", tokenwise_roofline), ("Attention", attention_roofline)]:
            r = fn(S)
            bound = "compute" if r["cfloor"] >= r["mfloor"] else "memory"
            p(f"{name:10s} {S:5d} {r['flops']/1e9:8.1f} {r['wbytes']/1e6:7.1f} {r['cfloor']:8.1f} {r['mfloor']:8.1f} {r['roofline']:11.1f} {bound:>7s}")
            rows.append(dict(supertask=name, seq=S, **{k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()}, bound=bound))
    # per-layer 합 & e2e 이론치
    p(f"\n=== per-layer 합 (Tokenwise+Attention) & e2e 이론 roofline ===")
    for S in SEQS:
        tw = tokenwise_roofline(S)["roofline"]
        at = attention_roofline(S)["roofline"]
        per_layer = tw + at
        e2e = per_layer * L / 1000
        p(f"  S={S:5d}: per-layer {per_layer:7.1f}us  x{L} = {e2e:6.2f}ms  (Tokenwise {tw:.0f} + Attention {at:.0f})")
    out = "profiling/production_profile/supertask_shapes_1b_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    p(f"\n저장: {out}")


if __name__ == "__main__":
    main()
