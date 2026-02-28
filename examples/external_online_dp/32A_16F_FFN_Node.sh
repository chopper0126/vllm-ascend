#!/bin/bash
if [[ "$1" != "attention" && "$1" != "ffn" ]]; then
    echo -e "\033[31m无效的命令,使用方法: bash single_afd_A3_32A16F.sh [attention/ffn] ip\033[0m"
    exit 1
fi
# 检查第二个参数是否为IP地址格式
if ! [[ "$2" =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
    echo -e "\033[31m错误：第二个参数必须是有效的IP地址\033[0m"
    exit 1
fi

unset http_proxy
unset https_proxy
clear
ulimit -u unlimited
pkill -9 vllm
pkill -9 VLLM
pkill -9 python

# (需配置项)默认参数，可通过入参覆盖：
# $3: MODEL_PATH
# $4: --max-num-seqs
# $5: cudagraph_capture_sizes（逗号分隔，如 "20" 或 "20,40"）
DEFAULT_MODEL_PATH="/home/c00945949/weight/DeepSeek-V3.1_w8a8mix_mtp/"
DEFAULT_MAX_NUM_SEQS=20
DEFAULT_CUDAGRAPH_CAPTURE_SIZES="20"

MODEL_PATH="${3:-$DEFAULT_MODEL_PATH}"
MAX_NUM_SEQS="${4:-$DEFAULT_MAX_NUM_SEQS}"
CUDAGRAPH_CAPTURE_SIZES="${5:-$DEFAULT_CUDAGRAPH_CAPTURE_SIZES}"

IF_NAME="enp8s0f4u1"
LOCAL_IP="$2"

export HCCL_IF_IP=${LOCAL_IP}
export HCCL_SOCKET_IFNAME=${IF_NAME}
export GLOO_SOCKET_IFNAME=${IF_NAME}
export TP_SOCKET_IFNAME=${IF_NAME}
export HCCL_BUFFSIZE=2048
export HCCL_EXEC_TIMEOUT=10000
export ASCEND_LAUNCH_BLOCKING=0
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

export TORCHDYNAMO_VERBOSE=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100

export VLLM_USE_V1=1
export VLLM_VERSION="v0.11.0"
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600

source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/CAM/bin/set_env.bash

timestamp=$(date +"%Y-%m-%d-%H-%M-%S")
ALL_LOGS=/home/y00889327/workspace-afd/vllm-logs/${timestamp}
mkdir -p "${ALL_LOGS}"/CANN/"${HCCL_IF_IP}"
export ASCEND_PROCESS_LOG_PATH=${ALL_LOGS}/CANN/${HCCL_IF_IP}
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=1
export ASCEND_LOG_SYNC_SAVE=0

export VLLM_LOGGING_LEVEL=WARNING
APP_LOG_PATH=${ALL_LOGS}/"$1".log

# (需配置项)应用启动参数配置
if [ "$1" == 'attention' ]; then
    vllm serve $MODEL_PATH \
        --host 0.0.0.0 \
        --port 8006 \
        --quantization ascend \
        --data-parallel-size 16 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --enable-expert-parallel \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-model-len 4096 \
        --max-num-batched-tokens 20 \
        --trust-remote-code \
        --no-enable-prefix-caching \
        --gpu-memory-utilization 0.9 \
        --enable-dbo \
        --dbo-prefill-token-threshold 12 \
        --dbo-decode-token-threshold 2 \
        --compilation-config "{\"cudagraph_mode\": \"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${CUDAGRAPH_CAPTURE_SIZES}]}" \
        --kv-transfer-config \
        '{
            "kv_connector": "SharedStorageConnector",
            "kv_role": "kv_consumer",
            "kv_port": "30200",
            "engine_id": "2",
            "kv_connector_extra_config": {
                "prefill": {
                    "dp_size": 1,
                    "tp_size": 16
                },
                "decode": {
                    "dp_size": 16,
                    "tp_size": 1
                }
            }
        }' \
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
else
    python -m vllm.entrypoints.afd_ffn_server $MODEL_PATH \
        --data-parallel-size 1 \
        --tensor-parallel-size 16 \
        --seed 1024 \
        --enable-expert-parallel \
        --quantization ascend \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-model-len 4096 \
        --max-num-batched-tokens 20 \
        --trust-remote-code \
        --no-enable-prefix-caching \
        --gpu-memory-utilization 0.9 \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[20]}' \
        --enable-dbo \
        --dbo-prefill-token-threshold 12 \
        --dbo-decode-token-threshold 2 \
        --kv-transfer-config \
        '{
            "kv_connector": "SharedStorageConnector",
            "kv_role": "kv_consumer",
            "kv_port": "30200",
            "engine_id": "2",
            "kv_connector_extra_config": {
                "prefill": {
                    "dp_size": 1,
                    "tp_size": 16
                },
                "decode": {
                    "dp_size": 1,
                    "tp_size": 16
                }
            }
        }' \
        --afd-config \
        '{
            "afd_connector": "camm2nconnector",
            "num_afd_stages": "2",
            "afd_role": "ffn",
            "afd_extra_config": {
                "afd_size": "32A16F"
            },
            "compute_gate_on_attention": "True",
            "afd_host": "141.61.73.131",
            "afd_port": "23961",
            "quant_mode":"1"
        }' 2>&1 | tee "$APP_LOG_PATH"
fi