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

class FFNNeedMetadata():

    def __init__(self,
                is_prefill: bool = False,
                enable_force_load_balance: bool = False,
                is_ffn: bool = False):
        self.is_prefill = is_prefill
        self.enable_force_load_balance = enable_force_load_balance
        self.is_ffn = is_ffn

@dataclass
class AFDConnectorMetadata:
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

class DefaultProcessGroupSwitcher:
    def __init__(self, default_group, new_default_group):
        self.default_group = default_group
        self.new_default_group = new_default_group

    def __enter__(self):
        _update_default_pg(self.new_default_group)

    def __exit__(self, exc_type, exc_value, traceback):
        _update_default_pg(self.default_group)  

def creat_hccl_process_group(rank, world_size):
    import torch
    torch.npu.set_device(rank)
    new_default_group = init_process_group(
        init_method='tcp://127.0.0.1:29500',
        backend='gloo', 
        rank=rank, 
        world_size=world_size, 
        group_name="new_hccl"
    )
    return new_default_group

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

def get_ae_group_new() -> GroupCoordinator:
    assert _AE_GROUP is not None, ("afd group is not initialized")
    return _AE_GROUP

class AFDConnectorBase(ABC):
    # Attention Worker Interface (sends FFN requests)
    @abstractmethod
    def send_attn_output(self, hidden_states: torch.Tensor, ffn_need_metadata, attn_metadata) -> Any:
        pass

    @abstractmethod
    def recv_ffn_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass

    # FFN Server Interface (receives and responds to requests)
    @abstractmethod
    def recv_attn_output(self, timeout_ms: Optional[int] = None) -> Any:
        pass

    @abstractmethod
    def send_ffn_output(self, ffn_output: torch.Tensor) -> None:
        pass

class CAMAFDConnector(AFDConnectorBase):
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
        with default_pg_switcher:
            ae_group = get_ae_group_new()
            dst = (ae_group.rank_in_group + 1) % ae_group.world_size
            ffn_need_metadata = metadata.ffn_need_metadata
            ae_group.send_object(ffn_need_metadata, dst=dst)
            attn_metadata = metadata.attn_metadata
            ae_group.send_object(attn_metadata, dst=dst)
            size_tensor = torch.tensor(hidden_states.size()).npu()
            ae_group.send(size_tensor)
            ae_group.send(hidden_states)
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
        
        return ffn_need_metadata_obj, attn_metadata, hidden_states
