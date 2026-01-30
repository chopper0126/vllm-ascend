#!/bin/bash

   unset http_proxy
   unset https_proxy
   clear
   ulimit -u unlimited

   # (需配置项)权重路径
   MODEL_PATH="/home/skf/weight"
   # MODEL_PATH="/home/lxf/DSV2LiteWeight"

   IF_NAME="enp8s0f4u1"
   LOCAL_IP="141.61.73.133"

   export HCCL_IF_IP=${LOCAL_IP}
   export HCCL_SOCKET_IFNAME=${IF_NAME}
   export GLOO_SOCKET_IFNAME=${IF_NAME}
   export TP_SOCKET_IFNAME=${IF_NAME}
   export OMP_PROC_BIND=false
   export OMP_NUM_THREADS=10
   export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
   export HCCL_BUFFSIZE=600
   export TASK_QUEUE_ENABLE=1
   export HCCL_OP_EXPANSION_MODE="AIV"
   export VLLM_USE_V1=1
   export ASCEND_RT_VISIBLE_DEVICES=$1
   export HCCL_EXEC_TIMEOUT=10000
   export ASCEND_LAUNCH_BLOCKING=0
   export TORCHDYNAMO_VERBOSE=1

   export VLLM_USE_V1=1
   export VLLM_VERSION="v0.11.0"
   export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
   source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/CAM/bin/set_env.bash

   # 日志设置
   # (需配置项)基础日志路径设置
   timestamp=$(date +"%Y-%m-%d-%H-%M-%S")
   ALL_LOGS=/home/y00889327/workspace-afd/vllm-logs/${timestamp}
   # mkdir -p "${ALL_LOGS}"

   # # CANN日志设置
   mkdir -p "${ALL_LOGS}"/CANN/"${HCCL_IF_IP}"
   export ASCEND_PROCESS_LOG_PATH=${ALL_LOGS}/CANN/${HCCL_IF_IP}
   # 是否开启日志打屏。开启后，日志将不会保存在log文件中，而是将产生的日志直接打屏显示。
   export ASCEND_SLOG_PRINT_TO_STDOUT=0
   # 设置日志级别。1为INFO，2为WARNING
   export ASCEND_GLOBAL_LOG_LEVEL=3
   # 设置应用类日志是否开启Event日志。
   export ASCEND_GLOBAL_EVENT_ENABLE=1
   # 指定Device侧应用类日志回传到Host侧的延时时间。
   # export ASCEND_LOG_DEVICE_FLUSH_TIMEOUT=2000
   # 指定日志拥塞处理方式。0：默认处理方式，在日志拥塞或IO访问性能差的情况下，为保证业务性能不劣化，系统可能会丢失日志。1：在日志拥塞或IO访问性能差的情况下，不丢失日志。该方式下，为便于问题定位，建议配置为1。
   export ASCEND_LOG_SYNC_SAVE=0

   # 应用日志设置
   export VLLM_LOGGING_LEVEL=WARNING
   APP_LOG_PATH=${ALL_LOGS}/"$LOCAL_IP".log
   #    --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
   #    --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": true,"finegrained_tp_config": {"lmhead_tensor_parallel_size":16}}' \
   # MooncakeLayerwiseConnector
   vllm serve $MODEL_PATH \
       --host 0.0.0.0 \
       --port $2 \
       --data-parallel-size $3 \
       --data-parallel-rank $4 \
       --data-parallel-address $5 \
       --data-parallel-rpc-port $6 \
       --tensor-parallel-size $7 \
       --enable-expert-parallel \
       --seed 1024 \
       --max-model-len 4096 \
       --max-num-batched-tokens 20 \
       --max-num-seqs 20 \
       --trust-remote-code \
       --gpu-memory-utilization 0.90  \
       --quantization ascend \
       --no-enable-prefix-caching \
       --additional-config '{"multistream_overlap_shared_expert": false}' \
       --kv-transfer-config \
       '{"kv_connector": "SharedStorageConnector",
           "kv_role": "kv_consumer",
           "kv_port": "30200",
           "engine_id": "2",
           "kv_connector_extra_config": {
                       "prefill": {
                               "dp_size": 2,
                               "tp_size": 8
                       },
                       "decode": {
                               "dp_size": 32,
                               "tp_size": 1
                       }
               }
       }'\
       --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[20]}' \
       --afd-config \
             '{
                "afd_connector": "camm2nconnector",
                "afd_role": "attention",
                "num_afd_stages": "2",
                "afd_extra_config": {
                  "afd_size": "32A16F"
                },
                "compute_gate_on_attention": "True",
     	         "afd_host": "141.61.73.131",
                "afd_port": "23961",
                "quant_mode":"1"
              }' 2>&1 | tee "$APP_LOG_PATH"