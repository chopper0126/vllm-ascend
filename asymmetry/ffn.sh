#ffn
#!/bin/bash
# 用法: ./start_ffn.sh [-d DEVICES] [-h]
# 示例: ./start_ffn.sh -d 12,13
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/CAM/bin/set_env.bash
export PYTHONPATH="/home/cyj/code/v13/vllm-ascend:/home/cyj/code/v13/vllm:$PYTHONPATH"
# 默认值
DEVICES="0,1,2,3"
AFD_PORT=29666
LOG_DIR="/home/cyj/run_vllm_v13/log"
MODEL_PATH="/home/cyj/weight/DSV2LiteWeight"

# 跨机通信配置 - FFN服务器作为从节点
MASTER_ADDR="90.90.97.37"  # 主节点IP（Attention服务器的IP）
MASTER_PORT="29500"
NETWORK_INTERFACE="enp194s0f0"  # 网络接口名
BSIZE=40
AFD_SIZE="8A8F"
NUM_DEVICES=8
# 解析命令行参数
   usage() {
       echo "用法: $0 [-d DEVICES] [-p PORT] [-a AFD_PORT] [-l LOG_DIR] [-m MASTER_ADDR] [-t MASTER_PORT] [-i INTERFACE] [-h]"
       echo "  参数:"
       echo "    -d DEVICES         使用的卡号（默认: 14,15）"
       echo "    -p PORT            HTTP服务端口（默认: 8006）"
       echo "    -a AFD_PORT        AFD通信端口（默认: 29666）"
       echo "    -l LOG_DIR         日志目录（默认: $LOG_DIR）"
       echo "    -m MASTER_ADDR     主节点IP地址（默认: $MASTER_ADDR）"
       echo "    -t MASTER_PORT     主节点端口（默认: $MASTER_PORT）"
       echo "    -i INTERFACE       网络接口名（默认: $NETWORK_INTERFACE）"
       echo "    -h                 显示帮助信息"
       exit 1
   }

   while getopts "d:p:a:l:m:t:i:s:n:c:h" opt; do
       case $opt in
           d) DEVICES="$OPTARG" ;;
           p) PORT="$OPTARG" ;;
           a) AFD_PORT="$OPTARG" ;;
           l) LOG_DIR="$OPTARG" ;;
           m) MASTER_ADDR="$OPTARG" ;;
           t) MASTER_PORT="$OPTARG" ;;
           i) NETWORK_INTERFACE="$OPTARG" ;;
           s) BSIZE="$OPTARG" ;;
           n) NUM_DEVICES="$OPTARG" ;;
           c) AFD_SIZE="$OPTARG" ;;
           h) usage ;;
           \?) echo "无效选项: -$OPTARG" >&2; usage ;;
           :) echo "选项 -$OPTARG 需要参数" >&2; usage ;;
       esac
   done

# 设置环境变量
export HCCL_BUFFSIZE=2048
export VLLM_LOGGING_LEVEL=DEBUG
export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"

# 设置跨机通信环境变量
export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT="$MASTER_PORT"
export GLOO_SOCKET_IFNAME="$NETWORK_INTERFACE"
export HCCL_SOCKET_IFNAME="$NETWORK_INTERFACE"

# 确保日志目录存在
mkdir -p "$LOG_DIR"
echo "DEVICES:$DEVICES"
echo "ASCEND_RT_VISIBLE_DEVICES:$ASCEND_RT_VISIBLE_DEVICES"
# 日志文件名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/ffn_${DEVICES//,/_}_${TIMESTAMP}.log"
echo "LOG_FILE: $LOG_FILE"
# 构建AFD配置JSON
AFD_CONFIG='{
  "afd_connector": "camp2pconnector",
  "num_afd_stages": "2",
  "afd_role": "ffn",
  "afd_extra_config": {
    "afd_size": "'$AFD_SIZE'"
  },
  "compute_gate_on_attention": "True",
  "afd_port": "'"$AFD_PORT"'"
}'

echo "AFD_CONFIG:$AFD_CONFIG"
   COMPILATION_CONFIG='{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": ['$BSIZE']}'
# 启动ffn服务器
echo "BSIZE:$BSIZE NUM_DEVICES:$NUM_DEVICES "
python -m vllm.entrypoints.afd_ffn_server "$MODEL_PATH" \
    --tensor-parallel-size $NUM_DEVICES \
    --enable_expert_parallel \
    --max_num_batched_tokens $BSIZE \
    --compilation-config "$COMPILATION_CONFIG"  \
    --enable-dbo \
    --dbo-prefill-token-threshold 12 \
    --dbo-decode-token-threshold 2 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --ubatch-size 2 \
    --max_num_seqs $BSIZE \
    --max-model-len 8192 \
    --afd-config "$AFD_CONFIG" \
    --additional-config '{"enable_force_load_balance": "True"}' \
    --kv-transfer-config '{
        "kv_connector": "DecodeBenchConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "fill_mean": 0.015,
            "fill_std": 0.0
        }
    }' \
   2>&1  > "$LOG_FILE"




