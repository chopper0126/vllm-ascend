from dataclasses import dataclass

from .AFDConnector import *

import torch_npu

from torch.distributed.distributed_c10d import _get_default_group

@dataclass
class M2NAFDConnectorMetadata:
    def __init__(self):
        self.topk_idx = None
        self.topk_weights = None
        self.moe_expert_num = 0
        self.scale = None
        self.handle = None
        self.quant_mode = 0
        self.aiv_num = 0
        self.batch_size = 0
        self.h = 0
        self.k = 0
        self.expert_token_nums_type = 0
        self.expand_x_type = torch.float16

class M2NAFDConnector(AFDConnectorBase):
    def __init__(self, rank: int, attn_size: int, ffn_size: int, is_ffn: bool):
        backend = "hccl"
        global _NEW_DEFAULT_GROUP
        if is_ffn:
            rank = rank + attn_size
        _NEW_DEFAULT_GROUP = creat_hccl_process_group(rank, ffn_size+attn_size)
        self.default_group = _NEW_DEFAULT_GROUP
        self.rank = rank
        self.attn_size = attn_size
        self.ffn_size = ffn_size
                                    
    # ATTN发给MOE（ATTN发送）
    # TODO:metadata的获取，最好从框架侧去拿
    def send_attn_output(self, hidden_states: torch.Tensor, metadata: M2NAFDConnectorMetadata) -> Any:
        dynamic_scales = metadata.scale
        if dynamic_scales is None:
            dynamic_scales = torch.tensor([], dtype=torch.float32, device='npu')
        recv_counts = torch_npu.npu_m2n_distribute_send(x=hidden_states,
                                                        expert_ids=metadata.topk_idx,
                                                        expert_scales=metadata.topk_weights,
                                                        group_ep=self.default_group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank),
                                                        world_size=self.attn_size + self.ffn_size,
                                                        moe_world_size=self.ffn_size,
                                                        ep_rank_id=self.rank,
                                                        moe_expert_num=metadata.moe_expert_num,
                                                        quant_mode=metadata.quant_mode,
                                                        aiv_num=metadata.aiv_num,
                                                        dynamic_scales=metadata.scale)
        return recv_counts

    # MOE发给ATTN（ATTN接收）
    def recv_ffn_output(self, hidden_states: torch.Tensor, metadata: M2NAFDConnectorMetadata) -> torch.Tensor:
        xOut = torch_npu.npu_n2m_distribute_recv(x=hidden_states,
                                                ep_recv_counts=metadata.handle,
                                                group_ep=self.default_group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank),
                                                world_size=self.attn_size + self.ffn_size,
                                                moe_world_size=self.ffn_size,
                                                ep_rank_id=self.rank,
                                                moe_expert_num=metadata.moe_expert_num,
                                                aiv_num=metadata.aiv_num)
        return xOut
    
    # MOE发给ATTN(MOE发送) 
    def send_ffn_output(self, ffn_output: torch.Tensor, metadata: M2NAFDConnectorMetadata):
        torch_npu.npu_n2m_distribute_send(expandX=ffn_output,
                                        ep_send_counts=metadata.handle,
                                        expert_scales=metadata.topk_weights,
                                        group_ep=self.default_group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank),
                                        world_size=self.attn_size + self.ffn_size,
                                        moe_world_size=self.ffn_size,
                                        ep_rank_id=self.rank,
                                        moe_expert_num=metadata.moe_expert_num,
                                        batch_size=metadata.batch_size,
                                        k=metadata.k,
                                        aiv_num=metadata.aiv_num)
        return
    
    # ATTN发给MOE(MOE接收)
    def recv_attn_output(self, metadata: M2NAFDConnectorMetadata) -> Any: 
        x_type = torch.int8
        if metadata.quant_mode == 0 :
            x_type = metadata.expand_x_type
        expand_x, dynamic_scales, expert_token_nums, recv_counts, expand_scales = torch_npu.npu_m2n_distribute_recv(x = torch.tensor([], dtype=x_type, device='npu'),
                                                                                group_ep=self.default_group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank),
                                                                                world_size=self.attn_size + self.ffn_size,
                                                                                moe_world_size=self.ffn_size,
                                                                                ep_rank_id=self.rank,
                                                                                moe_expert_num=metadata.moe_expert_num,
                                                                                quant_mode=metadata.quant_mode,
                                                                                batch_size=metadata.batch_size,
                                                                                h=metadata.h,
                                                                                k=metadata.k,
                                                                                expert_token_nums_type=metadata.expert_token_nums_type,
                                                                                aiv_num=metadata.aiv_num)
        return expand_x, dynamic_scales, expert_token_nums, recv_counts, expand_scales