#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Description: A2E and E2A test code for vllm-ascend
#

import torch
import torch_npu
import numpy as np
import torch.distributed as dist
import os
import torch.multiprocessing as mp

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

class A2E_E2A_Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, expert_ids, scales, batch_size, hidden_size, topk,
                expert_rank_size, atten_rank_size, rank, group_ep, aiv_num):
        is_moe_side = rank < expert_rank_size
        is_attention_side = rank >= expert_rank_size

        if is_attention_side:
            a2e_output = torch.ops._C_ascend.a2e(
                x=x,
                expert_ids=expert_ids,
                scales=scales,
                batch_size=batch_size,
                hidden_size=hidden_size,
                topk=topk,
                expert_rank_size=expert_rank_size,
                attention_rank_size=atten_rank_size,
                rank=rank,
                group_ep=group_ep,
                aiv_num=aiv_num,
                compute_gate=1)

            expand_x, simulate_expert_ids, simulate_expert_scales, atten_batch_size, x_active_mask_out = a2e_output

            e2a_output = torch.ops._C_ascend.e2a(
                expand_x=x,
                atten_batch_size=atten_batch_size,
                batch_size=batch_size,
                hidden_size=hidden_size,
                topk=topk,
                expert_rank_size=expert_rank_size,
                attention_rank_size=atten_rank_size,
                rank=rank,
                group_ep=group_ep,
                aiv_num=aiv_num)

            return e2a_output, expand_x, simulate_expert_ids, simulate_expert_scales, atten_batch_size, x_active_mask_out
        else:
            dummy_x = torch.empty(0, hidden_size, dtype=x.dtype, device=x.device)
            dummy_expert_ids = torch.empty(0, topk, dtype=torch.int32, device=x.device)
            dummy_scales = torch.empty(0, topk, dtype=torch.float, device=x.device)

            a2e_output = torch.ops._C_ascend.a2e(
                x=dummy_x,
                expert_ids=dummy_expert_ids,
                scales=dummy_scales,
                batch_size=batch_size,
                hidden_size=hidden_size,
                topk=topk,
                expert_rank_size=expert_rank_size,
                attention_rank_size=atten_rank_size,
                rank=rank,
                group_ep=group_ep,
                aiv_num=aiv_num,
                compute_gate=1)

            expand_x, simulate_expert_ids, simulate_expert_scales, atten_batch_size, x_active_mask_out = a2e_output

            e2a_output = torch.ops._C_ascend.e2a(
                expand_x=expand_x,
                atten_batch_size=atten_batch_size,
                batch_size=batch_size,
                hidden_size=hidden_size,
                topk=topk,
                expert_rank_size=expert_rank_size,
                attention_rank_size=atten_rank_size,
                rank=rank,
                group_ep=group_ep,
                aiv_num=aiv_num)

            return e2a_output, expand_x, simulate_expert_ids, simulate_expert_scales, atten_batch_size, x_active_mask_out


def gen_x(rank, batch_size, hidden_size):
    arr = [rank * batch_size + i + 1 for i in range(batch_size)
           for j in range(hidden_size)]
    return arr


def gen_expert_ids(rank, batch_size, topk, expert_rank_size):
    arr = [0] * (batch_size * topk)
    for i in range(batch_size):
        for j in range(topk):
            arr[i * topk + j] = (rank + i + j) % expert_rank_size
    return arr


def gen_scales(batch_size, topk):
    arr = [0.0] * (batch_size * topk)
    for i in range(batch_size):
        for j in range(topk):
            arr[i * topk + j] = 1.0 / topk
    return arr


def run_once(local_rank_id, ep_world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29600"
    rank = local_rank_id
    world_size = ep_world_size

    torch.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)

    batch_size = 16
    hidden_size = 512
    topk = 2
    expert_rank_size = 4
    atten_rank_size = 4
    aiv_num = 4
    data_type = torch.bfloat16

    is_moe_side = rank < expert_rank_size
    is_attention_side = rank >= expert_rank_size

    ep_ranks_list = list(range(0, world_size))
    ep_group = dist.new_group(backend="hccl", ranks=ep_ranks_list)
    ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    torch_npu.npu.synchronize()

    if is_attention_side:
        x_data = np.array(gen_x(rank, batch_size, hidden_size))
        x_data = x_data.reshape(batch_size, hidden_size)
        x_tensor = torch.tensor(x_data, dtype=data_type, device='npu')

        expert_ids_data = np.array(gen_expert_ids(rank, batch_size, topk, expert_rank_size))
        expert_ids_data = expert_ids_data.reshape(batch_size, topk)
        expert_ids_tensor = torch.tensor(expert_ids_data, dtype=torch.int32, device='npu')

        scales_data = np.array(gen_scales(batch_size, topk))
        scales_data = scales_data.reshape(batch_size, topk)
        scales_tensor = torch.tensor(scales_data, dtype=torch.float, device='npu')
    else:
        x_tensor = torch.empty(0, hidden_size, dtype=data_type, device='npu')
        expert_ids_tensor = torch.empty(0, topk, dtype=torch.int32, device='npu')
        scales_tensor = torch.empty(0, topk, dtype=torch.float, device='npu')

    mod = A2E_E2A_Module().npu()

    e2a_output, expand_x, simulate_expert_ids, simulate_expert_scales, atten_batch_size, x_active_mask_out = mod(
        x=x_tensor,
        expert_ids=expert_ids_tensor,
        scales=scales_tensor,
        batch_size=batch_size,
        hidden_size=hidden_size,
        topk=topk,
        expert_rank_size=expert_rank_size,
        atten_rank_size=atten_rank_size,
        rank=rank,
        group_ep=ep_hcomm_info,
        aiv_num=aiv_num)

    torch.npu.synchronize()

    if is_attention_side:
        print(f"Attention Side Rank {rank}: A2E-E2A test run completed!")
        print(f"  Input shape: {x_tensor.shape}")
        print(f"  E2A output shape: {e2a_output.shape}")

        assert e2a_output.shape == x_tensor.shape, \
            f"E2A output shape mismatch: {e2a_output.shape} vs {x_tensor.shape}"
        assert torch.allclose(e2a_output, x_tensor, atol=1e-3), \
            "E2A output does not match input x"
        print(f"  Input and output are consistent!")
    else:
        print(f"MOE Side Rank {rank}: A2E-E2A test run completed!")
        print(f"  A2E expand_x shape: {expand_x.shape}")
        print(f"  A2E simulate_expert_ids shape: {simulate_expert_ids.shape}")
        print(f"  A2E simulate_expert_scales shape: {simulate_expert_scales.shape}")
        print(f"  A2E atten_batch_size shape: {atten_batch_size.shape}")
        print(f"  A2E x_active_mask_out shape: {x_active_mask_out.shape}")

    dist.destroy_process_group()


def test_a2e_e2a():
    ep_world_size = 8

    print("A2E-E2A test started!")
    print(f"Running with {ep_world_size} ranks")

    mp.spawn(run_once, args=(ep_world_size, ), nprocs=ep_world_size, join=True)

    print("A2E-E2A test completed successfully!")


if __name__ == "__main__":
    test_a2e_e2a()
