from dataclasses import dataclass
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os
import torch

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

from vllm.distributed.parallel_state import (get_dp_group,get_world_group,get_new_default_group,GroupCoordinator, 
                                                creat_hccl_process_group, init_model_parallel_group)
from .AFDConnector import *

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

def get_ae_group_new() -> GroupCoordinator:
    assert _AE_GROUP is not None, ("afd group is not initialized")
    return _AE_GROUP

class P2PAFDConnector(AFDConnectorBase):
    def __init__(self, rank: int, attn_size: int, ffn_size: int, is_ffn: bool):
        backend = "hccl"
        global _NEW_DEFAULT_GROUP
        if is_ffn:
            rank = rank+attn_size
        _NEW_DEFAULT_GROUP = creat_hccl_process_group(rank, ffn_size+attn_size)
        self.default_group = _NEW_DEFAULT_GROUP
        default_pg_switcher = DefaultProcessGroupSwitcher(_get_default_group(), _NEW_DEFAULT_GROUP)
        # create sub_group in new_default_group
        with default_pg_switcher:
            sub_group_ranks = []
            for i in range(ffn_size):
                ranks = list([i, ffn_size+i])
                sub_group_ranks.append(ranks)
            global _AE_GROUP
            _AE_GROUP = init_model_parallel_group(sub_group_ranks,
                                    rank,
                                    backend,
                                    group_name="ae")
                                    
    # ATTN发给MOE（ATTN发送）
    def send_attn_output(self, hidden_states: torch.Tensor, metadata: AFDConnectorMetadata) -> Any:    
        default_pg_switcher = DefaultProcessGroupSwitcher(_get_default_group(), self.default_group)
        topk_weights = metadata.ffn_need_metadata.topk_weights
        topk_ids = metadata.ffn_need_metadata.topk_ids
        with default_pg_switcher:
            ae_group = get_ae_group_new()
            dst = (ae_group.rank_in_group + 1) % ae_group.world_size
            ffn_need_metadata = metadata.ffn_need_metadata
            ae_group.send_object(ffn_need_metadata, dst=dst)
            attn_metadata = metadata.attn_metadata
            ae_group.send_object(attn_metadata, dst=dst)
            size_tensor = torch.tensor(hidden_states.size()).npu()
            topk_weights_size = torch.tensor(topk_weights.size()).npu()
            topk_ids_size = torch.tensor(topk_ids.size()).npu()
            ae_group.send(size_tensor)
            ae_group.send(hidden_states)

            ae_group.send(topk_weights_size)            
            ae_group.send(topk_weights)
            print(topk_weights_size, topk_weights.size())

            ae_group.send(topk_ids_size)
            ae_group.send(topk_ids)
            print(topk_ids_size, topk_ids.size())
        return

    # MOE发给ATTN（ATTN接收）hidden_states只负责提供shape和dtype
    def recv_ffn_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        default_pg_switcher = DefaultProcessGroupSwitcher(_get_default_group(), self.default_group)
        with default_pg_switcher:
            ae_group = get_ae_group_new()
            hidden_states = ae_group.recv(hidden_states.size(),dtype=hidden_states.dtype)
        return hidden_states
    
    # MOE发给ATTN(MOE发送) 
    def send_ffn_output(self, ffn_output: torch.Tensor):
        default_pg_switcher = DefaultProcessGroupSwitcher(_get_default_group(), self.default_group)
        with default_pg_switcher:
            ae_group = get_ae_group_new()
            ae_group.send(ffn_output)
        return
    
    # ATTN发给MOE(MOE接收)
    def recv_attn_output(self, timeout_ms: Optional[int] = None) -> Any:
        default_pg_switcher = DefaultProcessGroupSwitcher(_get_default_group(), self.default_group)
        with default_pg_switcher:
            ae_group = get_ae_group_new()
            src = (ae_group.rank_in_group - 1) % ae_group.world_size
            ffn_need_metadata_obj = ae_group.recv_object(src=src)
            attn_metadata = ae_group.recv_object(src=src)
            size_tensor = ae_group.recv(2,dtype=torch.int64)
            size_tensor = torch.zeros([size_tensor[0],size_tensor[1]])
            hidden_states = ae_group.recv(size_tensor.size(),dtype=torch.bfloat16)

            topk_weights_size = ae_group.recv(2,dtype=torch.int64)
            topk_weights = torch.zeros([topk_weights_size[0],topk_weights_size[1]])
            topk_weights = ae_group.recv(topk_weights.size(),dtype=torch.bfloat16)

            topk_ids_size = ae_group.recv(2,dtype=torch.int64)
            topk_ids = torch.zeros([topk_ids_size[0],topk_ids_size[1]])
            topk_ids = ae_group.recv(topk_ids.size(),dtype=torch.int32)
            print(topk_ids_size, topk_ids.size())

            ffn_need_metadata_obj.topk_weights = topk_weights
            ffn_need_metadata_obj.topk_ids = topk_ids
            
        
        return ffn_need_metadata_obj, attn_metadata, hidden_states