from dataclasses import dataclass

from .AFDConnector import *

@dataclass
class CAMAFDConnectorMetadata:
    batch_size: int
    hidden_dim: int
    top_k: int

class CAMAFDConnector(AFDConnectorBase):
    def __init__(self, rank: int, attn_size: int, ffn_size: int, is_ffn: bool):
        backend = "hccl"
        global _NEW_DEFAULT_GROUP
        if not is_ffn:
            rank = rank+ffn_size
        _NEW_DEFAULT_GROUP = creat_hccl_process_group(rank, ffn_size+attn_size)
        self.default_group = _NEW_DEFAULT_GROUP
        self.rank = rank
        self.attn_size = attn_size
        self.ffn_size = ffn_size
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
    # TODO:metadata的获取，最好从框架侧去拿
    def send_attn_output(self, hidden_states: torch.Tensor, metadata: CAMAFDConnectorMetadata) -> Any:
        # self.batch_size = hidden_states.shape[0]
        # self.hidden_dim = hidden_state.shape[1]
        # self.top_k = expertIds.shape[1]
        self.batch_size = metadata.batch_size
        self.hidden_dim = metadata.hidden_dim
        self.top_k = metadata.top_k
        if metadata.scale:
            self.expandXOutDType = int8
            self.dynamicQuant = True
        else:
            self.expandXOutDType = float16
            self.dynamicQuant = False
        torch.ops.umdk_cam_op_lib.cam_a2e(expandX = hidden_states, expertIds = metadata.topk_idx, 
                                          scales = metadata.topk_weights, commArgs0 = torch.tensor([], dtype=torch.float16, device='npu'), 
                                          expandXOutDType = self.expandXOutDType, 
                                          commId0=None, batchSize = self.batch_size, hiddenSize = self.hidden_dim, topk = self.top_k, 
                                          expertRankSize = self.ffn_size, attentionRankSize = self.attn_size,
                                          sharedExpertNum = metadata.shared_expert_num, totalExpertNum = metadata.moe_expert_num, rank = self.rank,
                                          loadBalancingRankNum=0, loadBalancingThreshold=1, dynamicQuant = self.dynamicQuant, 
                                          ep_hcomm_info = self.default_group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank))
        return

    # MOE发给ATTN（ATTN接收）
    def recv_ffn_output(self, metadata: CAMAFDConnectorMetadata) -> torch.Tensor:
        output2 = torch.ops.umdk_cam_op_lib.cam_e2a(expandXOut, simulateExpertIds, simulateExpertScales, expandIdx, epRecvCounts,                                         
                                          commArgs = torch.tensor([], dtype=torch.float16, device='npu'), 
                                          commId=None, 
                                          batchSize = self.batch_size, hiddenSize = self.hidden_dim, topk = self.topk,
                                          expertRankSize = self.ffn_size, attentionRankSize = self.attn_size,
                                          sharedExpertNum = metadata.shared_expert_num, totalExpertNum = metadata.moe_expert_num,
                                          rank = self.rank, 
                                          loadBalancingRankNum=0, loadBalancingThreshold=1, 
                                          ep_hcomm_info = self.default_group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank))
        return output2
    
    # MOE发给ATTN(MOE发送) 
    def send_ffn_output(self, ffn_output: torch.Tensor, metadata: CAMAFDConnectorMetadata):
        torch.ops.umdk_cam_op_lib.cam_e2a(expandXOut = ffn_output, simulateExpertIds = self.simulateExpertIds,
                                          simulateExpertScales = self.simulateExpertScales, 
                                          expandIdx = self.expandIdx, epRecvCounts = self.epRecvCounts,
                                          commArgs = torch.tensor([], dtype=torch.float16, device='npu'), 
                                          batchSize = self.batch_size, hiddenSize = self.hidden_dim, topk = self.topk,
                                          expertRankSize = self.ffn_size, attentionRankSize = self.attn_size,
                                          sharedExpertNum = metadata.shared_expert_num, totalExpertNum = metadata.moe_expert_num,
                                          rank = self.rank, 
                                          loadBalancingRankNum=0, loadBalancingThreshold=1, 
                                          ep_hcomm_info = self.default_group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank))
        return
    
    # ATTN发给MOE(MOE接收)
    def recv_attn_output(self, metadata: CAMAFDConnectorMetadata) -> Any: 
        self.batch_size = metadata.batch_size
        self.hidden_dim = metadata.hidden_dim
        self.top_k = metadata.top_k
        output1 = torch.ops.umdk_cam_op_lib.cam_a2e(expandX, expertIds, expertScales, commArgs0, expandXOutDType, commId0,                                                    
                                                    batchSize = self.batch_size, hiddenSize = self.hidden_dim, topk = self.top_k,
                                                    expertRankSize = self.ffn_size, attentionRankSize = self.attn_size,
                                                    rank = self.rank,
                                                    loadBalancingRankNum=0, loadBalancingThreshold=1, 
                                                    ep_hcomm_info = self.default_group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank))
        expandX, dynamicScales, expandIdx, expertTokenNums, epRecvCounts, simulateExpertIds, simulateExpertScales = output1[0:7]
        self.dynamicScales = dynamicScales
        self.expandIdx = expandIdx
        self.expertTokenNums = expertTokenNums
        self.epRecvCounts = epRecvCounts
        self.simulateExpertIds = simulateExpertIds
        self.simulateExpertScales = simulateExpertScales
        # self.batch_size = self.simulateExpertIds.shape[0]
        # self.topk = self.simulateExpertIds.shape[1]
        # self.hidden_dim = expandX.shape[0]
        return expandX