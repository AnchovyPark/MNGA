#!/usr/bin/env python3
"""[실패 경로 문서화] FX 레벨에서 all_reduce 노드를 손으로 삽입 시도.

이 스크립트는 "왜 손으로 MpppConfig 를 짜서 all_reduce 를 넣는 방식이
안 되는가"를 재현/기록하기 위한 것이다. (실측용 아님 — 실측은 tp_sweep.py)

경로:
  matmul y=x@w 의 contraction(K) 축을 2-way Partial 로 두고 → Replicate 요청
  → ModelRewriter.cc_calculator._same_mesh_partial_to_replicate 가 AllReduce 삽입.

두 가지가 막는다(README §approach-B 참고):
  1. ShardingPropagator: 자동 전파 규칙이 없어 "모든 입력이 Replicate"만 허용.
     → matmul 노드 spec 을 static_tensors 로 직접 Partial 지정해 우회 가능.
  2. new_pipeline_builder: AllReduce supertask 실행 경로가 NotImplementedError.
     → 그래프엔 노드가 생겨도 device 실행/프로파일 불가.

여기선 (1)을 우회해 AllReduce 노드가 실제로 삽입되는지 그래프를 덤프하고,
(2)로 인해 실행이 막히는 지점을 표시한다.
"""
import torch
import furiosa.torch  # noqa: F401

from furiosa_llm.parallelize.mppp.config import (
    MpppConfig, ShardSpec, DeviceMesh, Replicate, Partial, ReduceOp,
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
    args = (x, w)
    gm, _ = get_aten_graph_with_metadata(MM(), args, {})

    ph, mm_node, out_node = [], None, None
    print("=== aten graph ===")
    for nd in gm.graph.nodes:
        print(f"  {nd.op:14s} {nd.name:20s} {nd.target}")
        if nd.op == "placeholder":
            ph.append(nd)
        if nd.op == "call_function" and "mm" in str(nd.target).lower():
            mm_node = nd
        if nd.op == "output":
            out_node = nd

    # 8 single-PE device mesh
    devs = {DeviceId(str(i)): Device(f"npu:0:{i}") for i in range(8)}
    mesh = DeviceMesh([DeviceId(str(i)) for i in range(8)])

    # 입력은 Replicate (propagator 우회), matmul 출력을 직접 Partial(SUM) 로 고정
    static = {nd.name: ShardSpec([Replicate()], mesh) for nd in ph}
    if mm_node is not None:
        static[mm_node.name] = ShardSpec([Partial(ReduceOp.SUM)], mesh)

    dynamic = []
    if mm_node is not None and out_node is not None:
        dynamic.append(DynamicTensorSpec(
            mm_node.name, out_node.name, ShardSpec([Replicate()], mesh)))

    cfg = MpppConfig("mm-tp8-allreduce", devices=devs,
                     static_tensors=static, dynamic_tensors=dynamic)

    print("\n=== ModelRewriter.rewrite (AllReduce 삽입 시도) ===")
    try:
        gm2 = ModelRewriter(gm, cfg).rewrite(args)
        has_cc = False
        for nd in gm2.graph.nodes:
            t = (str(nd.target) + nd.name).lower()
            mark = ""
            if any(k in t for k in ("allreduce", "all_reduce", "reduce", "comm")):
                mark = "   <<< COLLECTIVE"; has_cc = True
            print(f"  {nd.op:14s} {nd.name:24s} {nd.target}{mark}")
        print(f"\n>>> AllReduce 삽입됨: {'YES' if has_cc else 'NO'}")
        print(">>> 다음 단계(new_pipeline_builder)에서 실행 시 "
              "'Communication supertasks are not supported yet' 로 막힘.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nrewrite FAIL: {type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    main()
