#!/usr/bin/env python3
"""de-risk: ModelRewriter 가 K축 샤딩 matmul 에 all_reduce 를 자동 삽입하는지 확인.

목표(최종): 칩 내 TP 에서 all-reduce 가 포함될 때 latency 가 얼마나 늘어나는지 측정.
여기선 그 1단계 — 샤딩 config 를 손으로 author 해서 rewrite 결과 그래프에
all_reduce/Reduce 노드가 생기는지 CPU 레벨로 검증. (디바이스 실행 X)

matmul y = x @ w,  x:[B,S,K]  w:[K,N]
K(contraction) 를 2-way 샤딩 → 각 device 부분합(Partial) → 최종 Replicate = all_reduce.
"""
import torch
import furiosa.torch  # noqa

from furiosa_llm.parallelize.mppp.config import (
    MpppConfig, ShardSpec, DeviceMesh, Shard, Replicate, Partial, ReduceOp,
    DeviceId, Device, DynamicTensorSpec,
)
from furiosa_llm.parallelize.model_rewriter.api import ModelRewriter
from furiosa_llm.parallelize.trace import get_aten_graph_with_metadata

K, N = 512, 512


class MM(torch.nn.Module):
    def forward(self, x, w):
        return x @ w


def main():
    x = torch.randn(2, 8, K)
    w = torch.randn(K, N)
    model = MM()
    args = (x, w)

    gm, _ = get_aten_graph_with_metadata(model, args, {})
    print("=== aten graph 노드 (op / name / target) ===")
    placeholders, mm_node, out_node = [], None, None
    for nd in gm.graph.nodes:
        print(f"  {nd.op:14s} {nd.name:20s} {nd.target}")
        if nd.op == "placeholder":
            placeholders.append(nd)
        if nd.op == "call_function" and "mm" in str(nd.target).lower():
            mm_node = nd
        if nd.op == "output":
            out_node = nd

    # 2-device mesh (칩 내 2 PE)
    devs = {DeviceId("0"): Device("rngd:0:1"), DeviceId("1"): Device("rngd:0:1")}
    mesh = DeviceMesh([DeviceId("0"), DeviceId("1")])

    # placeholder 샤딩: 입력텐서(가장 큰 last dim=K) 와 weight(dim0=K) 를 K로 shard
    static = {}
    for nd in placeholders:
        meta = nd.meta.get("val")
        shape = tuple(meta.shape) if meta is not None else None
        # K 차원을 찾아 그 축으로 shard
        if shape and shape[-1] == K and len(shape) >= 2 and shape[0] != K:
            static[nd.name] = ShardSpec([Shard(len(shape) - 1)], mesh)  # x: last dim
            print(f"  -> shard {nd.name} dim={len(shape)-1} (x, K=last)")
        elif shape and shape[0] == K:
            static[nd.name] = ShardSpec([Shard(0)], mesh)               # w: dim0
            print(f"  -> shard {nd.name} dim=0 (w, K)")
        else:
            static[nd.name] = ShardSpec([Replicate()], mesh)
            print(f"  -> replicate {nd.name} shape={shape}")

    # matmul 출력은 Partial(SUM) 이 됨 → 최종 Replicate 요청 = all_reduce 유도
    dynamic = []
    if mm_node is not None:
        dynamic.append(DynamicTensorSpec(
            mm_node.name,
            out_node.name if out_node else mm_node.name,
            ShardSpec([Replicate()], mesh),
        ))
        print(f"  -> dynamic: {mm_node.name} -> Replicate (all_reduce 유도)")

    cfg = MpppConfig("mm-tp2", devices=devs, static_tensors=static, dynamic_tensors=dynamic)

    print("\n=== ModelRewriter.rewrite 실행 ===")
    try:
        mr = ModelRewriter(gm, cfg)
        gm2 = mr.rewrite(args)
        print("rewrite OK. 결과 그래프 노드:")
        has_cc = False
        for nd in gm2.graph.nodes:
            t = str(nd.target).lower()
            mark = ""
            if any(k in t or k in nd.name.lower() for k in ("all_reduce", "reduce", "all_gather", "comm")):
                mark = "   <<< COLLECTIVE"; has_cc = True
            print(f"  {nd.op:14s} {nd.name:22s} {nd.target}{mark}")
        print(f"\n>>> all_reduce/collective 삽입됨: {'YES' if has_cc else 'NO'}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"\nrewrite FAIL: {type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    main()
