import torch.distributed as dist
import torch.multiprocessing as mp
import time
import torch

from typing import Callable, Optional, TypeVar, Union


from vllm.engine.arg_utils import EngineArgs


from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.worker.model_runner import ModelRunner
from vllm.worker.worker import Worker

T = TypeVar("T", bound=Worker)

# from torch import nn
from vllm.distributed.parallel_state import (
                                             init_model_parallel_group
                                             )


from datetime import timedelta
from typing import Any, Optional, Union

import torch
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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from vllm.config import  VllmConfig


from vllm_ascend.quantization.quant_config import AscendLinearMethod
from vllm_ascend.worker.worker_v1 import NPUWorker
from vllm.distributed import parallel_state as ps


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
    import torch_npu
    import os
    torch.npu.set_device(rank)
    new_default_group = init_process_group(
        init_method='tcp://127.0.0.1:29500',
        backend='gloo', 
        rank=rank, 
        world_size=world_size, 
        group_name="new_hccl"
    )
    return new_default_group
    
def create_ffn_process_group(rank, world_size,attn_size, ffn_size):
    print(f"进程 {rank} 启动，参数: world_size={world_size}, "
          f"attn_size={attn_size}, ffn_size={ffn_size}")
    if rank == 1: time.sleep(1); print('======================================================================')
    import torch
    torch.npu.set_device(rank)
    init_method = 'tcp://127.0.0.1:29503'
    ffn_default_group = dist.init_process_group(
            init_method=init_method,
            backend='hccl', 
            rank=rank % attn_size, 
            world_size=ffn_size
        )
    return ffn_default_group

def run_ffn(rank, world_size,attn_size, ffn_size):

    ffn_default_group = create_ffn_process_group(rank, world_size,attn_size, ffn_size)
    
    config = create_config()
    attn_ranks = list(config.additional_config.get("attn_ranks"))
    ffn_ranks = list(config.additional_config.get("ffn_ranks"))
    role = config.additional_config.get("role")
    print(f'attn_ranks is {attn_ranks}')
    # new_default_group
    # global _NEW_DEFAULT_GROUP
    ps._NEW_DEFAULT_GROUP = creat_hccl_process_group(rank, len(attn_ranks) + len(ffn_ranks))
    # switcher, update default group to new_default_group
    default_pg_switcher = DefaultProcessGroupSwitcher(_get_default_group(), ps._NEW_DEFAULT_GROUP)
    # create sub_group in new_default_group
    with default_pg_switcher:
        sub_group_ranks = []
        for i in range(len(ffn_ranks)):
            ranks = list([attn_ranks[i],ffn_ranks[i]])
            sub_group_ranks.append(ranks)
        ps._AE_GROUP = init_model_parallel_group(sub_group_ranks,
                                    rank,
                                    backend='hccl', 
                                    group_name="ae")
    print(f'rank={rank},create global process group success')   
    print(f'rank={rank},start to run model') 
    
    """Initialize the worker for Ascend."""
    # register patch for vllm
    from vllm_ascend.utils import adapt_patch
    from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions
    from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
    adapt_patch()
    # Register ops when worker init.
    from vllm_ascend import ops
    ops.register_dummy_fusion_op()
    _register_atb_extensions()
    # # init ascend config
    # init_ascend_config(vllm_config)

    
    
    ffn_worker = create_worker(
        FFNWorker,
        model_runner_cls=FFNModelRunner,
        engine_config = config,
        rank = rank,
        ffn_size = ffn_size
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


from vllm_ascend.platform import NPUPlatform
import threading

class FFNWorker(NPUWorker):

    def __init__(
            self,
            vllm_config: VllmConfig,
            local_rank: int,
            rank: int,
            distributed_init_method: str,
            is_driver_worker: bool = False,
            # Additional parameters for compatibility with vllm
            **kwargs):
        
        """Initialize the worker for Ascend."""
        # register patch for vllm
        from vllm_ascend.utils import adapt_patch
        from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions
        from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
        adapt_patch()
        # Register ops when worker init.
        from vllm_ascend import ops
        ops.register_dummy_fusion_op()
        _register_atb_extensions()
        # init ascend config
        init_ascend_config(vllm_config)

        super().__init__(vllm_config=vllm_config,
                         local_rank=local_rank,
                         rank=rank,
                         distributed_init_method=distributed_init_method,
                         is_driver_worker=is_driver_worker)

    def init_device(self):
        # TODO: when cross machine,use self.local_rank
        # from vllm_ascend.ascend_config import get_ascend_config
        # ascend_config = get_ascend_config()
        # ffn_ranks = ascend_config.ffn_ranks
        # device = torch.device(f"npu:{self.local_rank + len(ffn_ranks)}")
        device = torch.device(f"npu:{self.rank}")
        NPUPlatform.set_device(device)
        NPUPlatform.empty_cache()
        self.init_npu_memory = NPUPlatform.mem_get_info()[0]

        # Initialize the distributed environment.
        self._init_worker_distributed_environment()
        # Set random seed.
        NPUPlatform.seed_everything(self.model_config.seed)

        # Init ModelRunner here, so that we have access to self.device.
        self.model_runner = FFNModelRunner(self.vllm_config, device)

    def execute_model(
        self,
    ):

        output = self.model_runner.execute_model()
        return output

def create_config() -> VllmConfig:
    
    engine_args = EngineArgs(
        model="/data/weight/DeepSeek-V2-Lite",
        enforce_eager=True,
        trust_remote_code=True,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        additional_config={
            # 关闭chunked_prefill ,调度器走vllm-ascend 重写的调度器，V0
            'ascend_scheduler_config':{
                'enabled': True,},
            "enable_afd":True,
            "enable_ms_afd":False,
            "attn_ranks": [0,1],
            "ffn_ranks": [2,3],
            "role":"ffn"
            }
    )
    engine_config = engine_args.create_engine_config()  
    return engine_config
    
def create_worker(cls: Callable[..., T],
                  model_runner_cls: Optional[ModelRunner] = None,
                  engine_config: VllmConfig = None,
                  **kargs) -> T:
    rank = kargs.get('rank')
    ffn_size = kargs.get('ffn_size')

    distributed_init_method = get_distributed_init_method(
        get_ip(), get_open_port())
    print(f'create worker rank is ========= {rank}')
    worker = cls(
        vllm_config = engine_config,
        local_rank = rank % ffn_size ,
        rank= rank,
        distributed_init_method = distributed_init_method,
        is_driver_worker = False,
        model_runner_cls = model_runner_cls,
    )

    worker.init_device()
    worker.load_model()
    worker.execute_model()


from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

class FFNModelRunner(NPUModelRunner):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):

        super().__init__(vllm_config=vllm_config,
                         device=device)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
    
    def execute_model(self):
        """Execute FFN computation for a single request batch"""
        print('ffn forward begain')
        # TODO: use event replace
        while True:
            layers_num = len(self.model.model.layers)
            for i in range(layers_num):
                self.model.model.layers[i].ffn_forward()
        print('ffn forward finished')
     
if __name__ == '__main__':
    hccl_world_size = 4
    attn_size = 2
    ffn_size = 2

    hccl_processes = []
    for rank in range(ffn_size,hccl_world_size):
        p = mp.Process(target=run_ffn, args=(rank, hccl_world_size, attn_size, ffn_size))
        hccl_processes.append(p)
        p.start()


    for p in hccl_processes:
        p.join()

    print("All processes finished")
