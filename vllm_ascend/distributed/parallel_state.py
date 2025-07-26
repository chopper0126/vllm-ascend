from typing import Optional

import torch
from vllm.distributed.parallel_state import (GroupCoordinator, get_world_group,
                                             init_model_parallel_group)

# vllm-ascend will maintain its own EP GroupCoordinator and ETP GroupCoordinator for
# customize parallel solution
_EP: Optional[GroupCoordinator] = None
_ETP: Optional[GroupCoordinator] = None
_AE: Optional[GroupCoordinator] = None


def get_ep_group() -> GroupCoordinator:
    assert _EP is not None, ("expert model parallel group is not initialized")
    return _EP


def get_etp_group() -> GroupCoordinator:
    assert _ETP is not None, (
        "expert tensor parallel group is not initialized")
    return _ETP

def get_ae_group() -> GroupCoordinator:
    assert _AE is not None, ("tensor model parallel group is not initialized")
    return _AE  


def model_parallel_initialized():
    return (_ETP is not None and _EP is not None)


def init_ascend_model_parallel(
    expert_parallel_size: int = 1,
    expert_tensor_parallel_size: int = 1,
    world_size: Optional[int] = None,
    backend: Optional[str] = None,
):
    if model_parallel_initialized():
        return
    assert torch.distributed.is_initialized()
    world_size = world_size or torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)
    num_expert_parallel_groups = expert_tensor_parallel_size
    num_expert_tensor_parallel_groups = expert_parallel_size

    global _EP
    group_ranks = []
    for i in range(num_expert_parallel_groups):
        ranks = list(range(i, world_size, num_expert_parallel_groups))
        group_ranks.append(ranks)

    _EP = init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="ep")

    group_ranks = []
    global _ETP
    for i in range(num_expert_tensor_parallel_groups):
        ranks = list(
            range(i * expert_tensor_parallel_size,
                  (i + 1) * expert_tensor_parallel_size))
        group_ranks.append(ranks)

    _ETP = init_model_parallel_group(group_ranks,
                                     get_world_group().local_rank,
                                     backend,
                                     group_name="etp")

def init_ascend_model_parallel_for_AE_split(
    expert_parallel_size: int = 1,
    expert_tensor_parallel_size: int = 1,
    world_size: Optional[int] = None,
    backend: Optional[str] = None,
):
    # print("in init_ascend_model_parallel")
    # print(f"tensor_parallel_size is == {expert_parallel_size}")
    all_ranks = torch.arange(world_size).reshape(
        -1, 1, 1,
        expert_parallel_size) 
    if model_parallel_initialized():
        return
    assert torch.distributed.is_initialized()
    world_size = expert_parallel_size
    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)
    num_expert_parallel_groups = expert_tensor_parallel_size
    num_expert_tensor_parallel_groups = expert_parallel_size

    # global _EP
    # group_ranks = []
    # for i in range(num_expert_parallel_groups):
    #     ranks = list(range(i + world_size, world_size + world_size, num_expert_parallel_groups))
    #     group_ranks.append(ranks)
    # print(f"EP group_ranks is === {group_ranks}")
    # print(f"init_ascend_model_parallel_for_AE_split get_world_group().local_rank is === {get_world_group().local_rank}")
    # if get_world_group().local_rank in group_ranks[0]:
    #     _EP = init_model_parallel_group(group_ranks,
    #                                     get_world_group().local_rank,
    #                                     backend,
    #                                     group_name="ep")
    
    global _EP
    assert _EP is None, ("expert parallel group is already initialized")
    group_ranks = all_ranks.transpose(1, 2).reshape(
        -1, 1 * 4).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    #print(f"vllm ascend _EP group_ranks is === {group_ranks}")
    
    _EP = init_model_parallel_group(group_ranks,
                                get_world_group().local_rank,
                                backend,
                                group_name="ep")
    
    # global _TP
    # group_ranks = []
    # for i in range(num_expert_parallel_groups):
    #     ranks = list(range(i, world_size, num_expert_parallel_groups))
    #     group_ranks.append(ranks)
    # print(f"TP group_ranks is === {group_ranks}")
    # print(f"init_ascend_model_parallel_for_AE_split get_world_group().local_rank is === {get_world_group().local_rank}")
    # if get_world_group().local_rank in group_ranks[0]:
    #     _TP = init_model_parallel_group(group_ranks,
    #                                     get_world_group().local_rank,
    #                                     backend,
    #                                     group_name="tp")
    
    global _AE
    group_ranks = []
    for i in range(expert_parallel_size):
        ranks = list(range(i, expert_parallel_size * 2, expert_parallel_size))
        group_ranks.append(ranks)
    #print(f"_AE group_ranks is === {group_ranks}")
    _AE = init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="ae")

    group_ranks = []
    global _ETP
    for i in range(num_expert_tensor_parallel_groups * 2):
        ranks = list(
            range(i * expert_tensor_parallel_size,
                  (i + 1) * expert_tensor_parallel_size))
        group_ranks.append(ranks)
    # print(f"_ETP group_ranks is === {group_ranks}")
    
    _ETP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        group_name="etp")





def destory_ascend_model_parallel():
    global _EP
    if _EP:
        _EP.destroy()
    _EP = None

    global _ETP
    if _ETP:
        _ETP.destroy()
    _ETP = None
