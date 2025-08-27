import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os
import torch

from datetime import timedelta
from typing import Any, Optional, Union

import torch
import torch.distributed
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
    _get_default_group,
    _update_default_pg,
)

def init_process_group(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = None,
    pg_options: Optional[Any] = None,
):
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)
        store = PrefixStore(group_name, store)

    pg_options_param_name = "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


class DefaultProcessGroupSwitcher:
    def __init__(self, default_group, new_default_group):
        self.default_group = default_group
        self.new_default_group = new_default_group

    def __enter__(self):
        _update_default_pg(self.new_default_group)

    def __exit__(self, exc_type, exc_value, traceback):
        _update_default_pg(self.default_group)


def run_hccl_process(rank, world_size, attn_size, ffn_size):
    is_train = rank < attn_size
    print(f'rank={rank}, world_size={world_size}, is_train={is_train}')
    if rank == 0: time.sleep(1); print('======================================================================')
    import torch
    torch.npu.set_device(rank)

    if is_train:
        # train
        new_default_group = init_process_group(
            init_method='tcp://127.0.0.1:29500',
            backend='gloo', 
            rank=rank, 
            world_size=world_size, 
            group_name="new_hccl"
        )

        default_group = dist.init_process_group(
            init_method='tcp://127.0.0.1:29502',
            backend='hccl', 
            rank=rank, 
            world_size=attn_size
        )
    else:
        # infer
        new_default_group = init_process_group(
            init_method='tcp://127.0.0.1:29500',
            backend='gloo', 
            rank=rank, 
            world_size=world_size, 
            group_name="new_hccl"
        )

        default_group = dist.init_process_group(
            init_method='tcp://127.0.0.1:29503',
            backend='hccl', 
            rank=rank % attn_size, 
            world_size=ffn_size
        )

    # Default Group: [[0, 1], [2, 3]] (attn/ffn)
    # New Default Group: [0, 1, 2, 3] (attn + ffn)
    # Sub Group: [[0, 2], [1, 3]] (attn-ffn weight update)

    # switcher, update default group to new_default_group
    default_pg_switcher = DefaultProcessGroupSwitcher(_get_default_group(), new_default_group)
    # create sub_group in new_default_group
    with default_pg_switcher:
        sub_group_ranks = [[0, 2], [1, 3]]
        for sub_group_rank in sub_group_ranks:
            tmp_group = dist.new_group(sub_group_rank)
            if rank in sub_group_rank:
                sub_group = tmp_group

    # all-reduce in default_group   [[0, 1], [2, 3]]
    data = torch.tensor([rank]).npu()
    print(f'Default Group all-reduce Before: rank={rank}, data={data}') # [0, 1, 2, 3]
    dist.all_reduce(data, group=default_group)
    print(f'Default Group all-reduce After: rank={rank}, data={data}')  # [1, 1, 5, 5]
    dist.barrier(group=new_default_group)

    # all-reduce in sub_group       [[0, 2], [1, 3]]
    data = torch.tensor([rank])
    with default_pg_switcher:
        print(f'Sub Group all-reduce Before: rank={rank}, data={data}') # [0, 1, 2, 3]
        dist.all_reduce(data, group=sub_group)
        dist.barrier(group=new_default_group)
        print(f'Sub Group all-reduce After: rank={rank}, data={data}')  # [2, 4, 2, 4]

    # send/recv in sub_group       [[0, 2], [1, 3]]
    data = torch.tensor([rank])
    with default_pg_switcher:
        print(f'Sub Group send/recv Before: rank={rank}, data={data}') # [0, 1, 2, 3]
        # tensor = torch.zeros(rank)
        # print(f'Sub Group send/recv Before: tensor={tensor}') # 
        if rank == 0 or rank == 1:
            data += 1
            # Send the tensor to process 1
            dist.send(tensor=data, dst=rank + 2)
        else:
            # Receive tensor from process 0
            dist.recv(tensor=data, src=rank - 2)

        # dist.all_reduce(data, group=sub_group)
        dist.barrier(group=new_default_group)
        print(f'Sub Group send/recv After: rank={rank}, data={data}')  # 

    # broadcast_object in sub_group
    data = [{'key0': torch.tensor([rank]), 'key1': torch.tensor([rank * 10])}]
    with default_pg_switcher:
        print(f'Sub Group broadcast_object Before: rank={rank}, data={data}') # [{key0:0, key1:0}, {key0:1, key1:10}, {key0:2, key1:20}, {key0:3, key1:30}]
        dist.broadcast_object_list(data, src=sub_group_ranks[rank%2][0], group=sub_group)
        dist.barrier(group=new_default_group)
        print(f'Sub Group broadcast_object After: rank={rank}, data={data}')  # [{key0:0, key1:0}, {key0:1, key1:10}, {key0:0, key1:0}, {key0:1, key1:10}]


if __name__ == '__main__':
    hccl_world_size = 4
    attn_size = 2
    ffn_size = 2

    hccl_processes = []
    for rank in range(hccl_world_size):
        p = mp.Process(target=run_hccl_process, args=(rank, hccl_world_size, attn_size, ffn_size))
        hccl_processes.append(p)
        p.start()


    for p in hccl_processes:
        p.join()

    print("All processes finished")