from dataclasses import dataclass
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os
import torch
import torch_npu

from datetime import timedelta
from typing import Any, Optional, Union
from abc import ABC, abstractmethod

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

from .AFDConnector import *

from vllm.distributed.parallel_state import (get_dp_group,get_world_group,get_new_default_group,GroupCoordinator, 
                                                creat_hccl_process_group, init_model_parallel_group)

class FFNNeedMetadata():

    def __init__(self,
                is_prefill: bool = False,
                enable_force_load_balance: bool = False,
                is_ffn: bool = False):
        self.is_prefill = is_prefill
        self.enable_force_load_balance = enable_force_load_balance
        self.is_ffn = is_ffn

@dataclass
class P2PAFDConnectorMetadata(AFDConnectorMetadata):
    layer_idx: int              # Layer index for computation
    stage_idx: int              # Pipeline stage index  
    seq_lens: list[int]         # Sequence lengths for each request
    dtype: torch.dtype          # Tensor data type
    device: torch.device        # Compute device
    request_id: Optional[str]   # Request identifier
    timestamp: Optional[float]  # Timestamp for debugging
    group : dist.ProcessGroup # communication domain    
    topk_idx: Optional[torch.Tensor] # indices token which expert to be sended
    topk_weights: Optional[torch.Tensor] # the expert weights
    moe_expert_num: Optional[int] # number of moe experts
    shared_expert_num: Optional[int] # number of share experts
    handle: Optional[torch.Tensor] # the communication handle given by the recv_attn_output
    def __init__(self, layer_idx: int, stage_idx: int, seq_lens: list[int]):
        self.layer_idx = layer_idx
        self.stage_idx = stage_idx
        self.seq_lens = seq_lens

    def set_attn_metadata(self, attn_metadata):
        self.attn_metadata = attn_metadata

    def set_ffn_need_metadata(self, ffn_need_metadata):
        self.ffn_need_metadata = ffn_need_metadata

def get_comm1_group_new() -> GroupCoordinator:
    assert _COMM1_GROUP is not None, ("afd group is not initialized")
    return _COMM1_GROUP

def get_comm2_group_new() -> GroupCoordinator:
    assert _COMM2_GROUP is not None, ("afd group is not initialized")
    return _COMM2_GROUP

tensor1 = torch.randn(5120, 5120).npu()

class P2PAFDConnector(AFDConnectorBase):
    def __init__(self, rank: int, attn_size: int, ffn_size: int, is_ffn: bool):
        backend = "hccl"
        global _NEW_DEFAULT_GROUP
        if is_ffn:
            rank = rank+attn_size
        _NEW_DEFAULT_GROUP = creat_hccl_process_group(rank, ffn_size+attn_size, backend=backend)
        self.default_group = _NEW_DEFAULT_GROUP
        default_pg_switcher = DefaultProcessGroupSwitcher(_get_default_group(), _NEW_DEFAULT_GROUP)
        self.comm_stream = torch_npu.npu.Stream()
        # create sub_group in new_default_group
        with default_pg_switcher:
            sub_group_ranks = []
            for i in range(ffn_size):
                ranks = list([i, ffn_size+i])
                sub_group_ranks.append(ranks)
            global _COMM1_GROUP
            _COMM1_GROUP = init_model_parallel_group(sub_group_ranks,
                                    rank,
                                    backend,
                                    group_name="ae")
            global _COMM2_GROUP
            _COMM2_GROUP = init_model_parallel_group(sub_group_ranks,
                                    rank,
                                    backend,
                                    group_name="ae")
                                    
    # attn send to moe  当前无需wait，在下个stage发送之前wait
    def send_attn_output(self, hidden_states: torch.Tensor, metadata: AFDConnectorMetadata) -> Any:    
        self.comm_stream.wait_stream(torch_npu.npu.current_stream())
        with torch_npu.npu.stream(self.comm_stream):
            ae_group = get_comm2_group_new()
            dst = (ae_group.rank_in_group + 1) % ae_group.world_size
            ffn_need_metadata = metadata.ffn_need_metadata
            attn_metadata = metadata.attn_metadata
            size_tensor = torch.tensor(hidden_states.size()).npu()
            handle1 = dist.isend(tensor=size_tensor, dst=dst, group=ae_group.device_group)
            handle2 = dist.isend(tensor=hidden_states, dst=dst, group=ae_group.device_group)
        return handle1, handle2

    # attn recv from moe
    def recv_ffn_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with torch_npu.npu.stream(self.comm_stream):
            ae_group = get_comm1_group_new()
            src = (ae_group.rank_in_group - 1) % ae_group.world_size
            hidden_states = torch.empty(hidden_states.size(), dtype=hidden_states.dtype, device="npu")
            handle = dist.irecv(tensor=hidden_states, group=ae_group.device_group, src=src)
        return hidden_states, handle
    
    # moe send to attn  当前无需wait，在下个stage发送之前wait
    def send_ffn_output(self, ffn_output: torch.Tensor):
        self.comm_stream.wait_stream(torch_npu.npu.current_stream())
        with torch_npu.npu.stream(self.comm_stream):
            ae_group = get_comm1_group_new()
            dst = (ae_group.rank_in_group + 1) % ae_group.world_size
            handle = dist.isend(tensor=ffn_output, group=ae_group.device_group, dst=dst)
        return handle

    # moe recv from attn
    def recv_attn_output(self, timeout_ms: Optional[int] = None) -> Any:
        with torch_npu.npu.stream(self.recv_stream):
            ae_group = get_comm2_group_new()
            src = (ae_group.rank_in_group - 1) % ae_group.world_size
            buffer = torch.empty(2, dtype=torch.int64, device="npu")
            handle1 = dist.irecv(tensor=buffer, group=ae_group.device_group, src=src)
            handle1.wait()
            hidden_states = torch.empty((buffer[0], buffer[1]), dtype=torch.bfloat16, device="npu")
            handle2 = dist.irecv(tensor=hidden_states, group=ae_group.device_group, src=src)
        return None, None, hidden_states, handle2