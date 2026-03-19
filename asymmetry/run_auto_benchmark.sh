#!/bin/bash

# ==============================================================================
# 自动化性能基准测试脚本
# ==============================================================================

# ================= 1. 用户配置区域 =================
# BSIZE_LIST="56 72 88 104 120 136 152 168 184 200 216 232 248 264 280 296 312"
BSIZE_LIST="24 32 40 48"
START_DEVICE1=0
START_DEVICE2=0
ATTN_ARR=(4 6 4 8 12)
FFN_ARR=(2 2 4 4 4)
DP=4
DATA_MULTIPLIER=16 #数据量
# TIMES=4 # bs > 160 ,并发太小导致触发不了dbo,导致触发不了入图

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PY="/home/cyj/code/benchmark/ais_bench/datasets/synthetic/synthetic_config.py"
MODEL_PY="/home/cyj/code/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py"
ARCHIVE_ROOT="${SCRIPT_DIR}/benchmark_results"

WARMUP_REQUEST_COUNT=1536
WARMUP_MAX_OUT_LEN=2
FORMAL_MAX_OUT_LEN=1024

GLOBAL_SUMMARY="${ARCHIVE_ROOT}/global_summary.csv"

# 日志打印
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

cleanup_vllm() {
    log "正在清理残留 VLLM 进程..."
    pkill -9 -f "vllm" 2>/dev/null || true
    pkill -9 -f "VLLM" 2>/dev/null || true
    sleep 3
}

update_configs() {
    local req_count=$1
    local batch_val=$2
    local out_len=$3
    log "req_count:$1 batch_val:$2 out_len:$3"
    sed -i "s/\"RequestCount\":[[:space:]]*[0-9]\+/\"RequestCount\": $req_count/" "$CONFIG_PY"
    sed -i "s/batch_size=[[:space:]]*[0-9]\+/batch_size=$batch_val/" "$MODEL_PY"
    sed -i "s/max_out_len=[[:space:]]*[0-9]\+/max_out_len=$out_len/" "$MODEL_PY"
    if ! grep -q "\"RequestCount\": $req_count" "$CONFIG_PY"; then
        log "错误：修改 RequestCount 失败"
        return 1
    fi
    return 0
}

archive_scenario() {
    local run_dir=$1
    local bsize=$2
    local total_batch=$3
    local req_count=$4

    local script_dir="${run_dir}/script"
    local log_dir="${run_dir}/log"
    local profile_dir="${run_dir}/profile"

    mkdir -p "$script_dir" "$log_dir" "$profile_dir"

    cp "$CONFIG_PY" "${script_dir}/config_snapshot_synthetic.py"
    cp "$MODEL_PY" "${script_dir}/config_snapshot_model.py"

    cat > "${script_dir}/run_params.txt" << EOF
Run Timestamp: $(date)
Test Scenario: BSIZE=${bsize}, DP=${DP}
Calculated Parameters:
  - Per Card Batch Size (BSIZE): $bsize
  - Data Parallel (DP): $DP
  - Total Batch Size (DP * BSIZE): $total_batch
  - Warmup Request Count: $WARMUP_REQUEST_COUNT
  - Formal Request Count: $req_count
  - Formal Max Out Len: $FORMAL_MAX_OUT_LEN
  - Data Multiplier: $DATA_MULTIPLIER
Directory Structure:
  - script/: Configuration snapshots & params
  - log/: Execution logs
  - profile/: Performance profiling data (Moved after test)
EOF
    log "场景已归档至: $run_dir"
}

generate_report() {
    local log_file=$1
    local report_file=$2
    local status="SUCCESS"
    local tpot_avg="N/A"
    local out_tok_thru_avg="N/A"
    local total_tok_thru_avg="N/A"

    tpot_raw=$(grep -i -E "TPOT" "$log_file" | head -1)
    if [ -n "$tpot_raw" ]; then
        if [[ "$tpot_raw" =~ TPOT[^│]*│[^│]*total[^│]*│[[:space:]]*([0-9.]+) ]]; then
            tpot_avg="${BASH_REMATCH[1]}"
        fi
    fi

    out_tok_raw=$(grep -i -E "output token throughput" "$log_file" | head -1)
    if [ -n "$out_tok_raw" ]; then
        out_tok_thru_avg=$(echo "$out_tok_raw" | grep -oE '[0-9.]+')
        [ -z "$out_tok_thru_avg" ] && out_tok_thru_avg="N/A"
    fi

    total_tok_raw=$(grep -i -E "total token throughput" "$log_file" | head -1)
    if [ -n "$total_tok_raw" ]; then
        total_tok_thru_avg=$(echo "$total_tok_raw" | grep -oE '[0-9.]+')
        [ -z "$total_tok_thru_avg" ] && total_tok_thru_avg="N/A"
    fi

    if [ "$tpot_avg" == "N/A" ] || [ "$out_tok_thru_avg" == "N/A" ] || [ "$total_tok_thru_avg" == "N/A" ]; then
        status="PARTIAL_PARSE"
        log "警告：指标解析失败！TPOT:$tpot_avg, OutThru:$out_tok_thru_avg, TotalThru:$total_tok_thru_avg"
    fi

    cat > "$report_file" << EOF
================================================================
Benchmark Summary Report
Generated: $(date)
Status: $status
================================================================

Key Metrics (Average Values):
  1. TPOT (Time Per Output Token):       $tpot_avg ms
  2. Output Token Throughput:            $out_tok_thru_avg token/s
  3. Total Token Throughput:             $total_tok_thru_avg token/s

File Locations:
  - Full Log: log/benchmark.log
  - Configs:  script/config_snapshot_*.py
  - Params:   script/run_params.txt
  - Profile:  profile/ (Check for profiling data)
================================================================
EOF
    echo "$status,$tpot_avg,$out_tok_thru_avg,$total_tok_thru_avg"
}

# ================= 2. 主执行流程 =================

log "=========================================="
log "启动自动化基准测试 (修复 Profile 执行问题)"
log "=========================================="

mkdir -p "$ARCHIVE_ROOT"

if [ ! -f "$GLOBAL_SUMMARY" ]; then
    echo "Timestamp,Run_Dir,BSIZE,DP,Total_Batch_Size,Request_Count,Status,TPOT(ms),OutTokenThru(token/s),TotalTokenThru(token/s)" > "$GLOBAL_SUMMARY"
fi

NUM_PAIRS=${#ATTN_ARR[@]}
for BSIZE in $BSIZE_LIST; do
    for (( i=0; i<$NUM_PAIRS; i++ )); do
        ATTN=${ATTN_ARR[$i]}
        FFN=${FFN_ARR[$i]}
        log "配置中: B=$BSIZE, A=$ATTN, F=$FFN"
        #update_configs "$BSIZE" "$ATTN" "$FFN" || { log "配置失败，跳过"; continue; }

        cleanup_vllm

        DP=${ATTN_ARR[$i]}
        batch_size_value=$((DP * BSIZE))
        target_count=$((DATA_MULTIPLIER * batch_size_value))

        # batch_size_value=$((DP * BSIZE * TIMES))
        # target_count=6144

        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        RUN_DIR="${ARCHIVE_ROOT}/BSIZE_${BSIZE}_${ATTN}A${FFN}F${TIMESTAMP}"

        # 计算 FFN 的设备字符串 (0 到)
	FFN_START=$((START_DEVICE1))
        FFN_END=$((START_DEVICE1 + FFN - 1))
        FFN_DEVICES=$(seq -s, $FFN_START $FFN_END)

        # 计算 ATTN 的设备字符串 (FFN 到 FFN+ATTN-1)
	ATTN_START=$((START_DEVICE2 + FFN))
        ATTN_END=$((START_DEVICE2 + FFN + ATTN - 1))
        ATTN_DEVICES=$(seq -s, $ATTN_START $ATTN_END)

        AFD_SIZE=${ATTN}A${FFN}F

        log "启动后端服务..."
        # 添加日志目录参数 -l (如果内部脚本支持)
	bash "${SCRIPT_DIR}/attn.sh" -d $ATTN_DEVICES -s $BSIZE -l "${RUN_DIR}/log" -n $ATTN -c "$AFD_SIZE" &
        PID_ATT=$!

        bash "${SCRIPT_DIR}/ffn.sh" -d $FFN_DEVICES -s $BSIZE -l "${RUN_DIR}/log" -n $FFN -c "$AFD_SIZE" &
        PID_FFN=$!

        log "等待服务启动 (90秒)..."
        sleep 90

        if ! kill -0 $PID_ATT 2>/dev/null || ! kill -0 $PID_FFN 2>/dev/null; then
            log "错误：后端服务启动失败！"
            echo "Service Start Failed" > "${RUN_DIR}/log/error.log"
            echo "$(date),${RUN_DIR},$BSIZE,$DP,$batch_size_value,$target_count,START_FAILED,N/A,N/A,N/A" >> "$GLOBAL_SUMMARY"
            continue
        fi
        log "后端服务启动成功"

        # === 预热 ===
        log "--- 阶段 1: 预热 ---"
        WARMUP_LOG="${RUN_DIR}/log/warmup.log"
	log "WARMUP_REQUEST_COUNT:$WARMUP_REQUEST_COUNT batch_size_value:$batch_size_value WARMUP_MAX_OUT_LEN:$WARMUP_MAX_OUT_LEN"
        if update_configs $WARMUP_REQUEST_COUNT $batch_size_value $WARMUP_MAX_OUT_LEN; then
            ais_bench --models vllm_api_stream_chat --datasets synthetic_gen --mode perf --debug > "$WARMUP_LOG" 2>&1
            log "预热完成。"
        else
            log "预热配置失败，跳过。"
            kill -9 $PID_ATT $PID_FFN 2>/dev/null
            continue
        fi

        # === 正式测试 ===
        log "--- 阶段 2: 正式测试 ---"
	log "target_count:$target_count batch_size_value:$batch_size_value FORMAL_MAX_OUT_LEN:$FORMAL_MAX_OUT_LEN"
        if ! update_configs $target_count $batch_size_value $FORMAL_MAX_OUT_LEN; then
            log "正式配置失败，跳过。"
            kill -9 $PID_ATT $PID_FFN 2>/dev/null
            continue
        fi

        archive_scenario "$RUN_DIR" "$BSIZE" "$batch_size_value" "$target_count"

        FORMAL_LOG="${RUN_DIR}/log/benchmark.log"
        log "运行 ais_bench..."

        START_TIME=$(date +%s)
        ais_bench --models vllm_api_stream_chat --datasets synthetic_gen --mode perf --debug > "$FORMAL_LOG" 2>&1
        EXIT_CODE=$?
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        if [ $EXIT_CODE -ne 0 ]; then
            log "警告：ais_bench 返回非零代码 ($EXIT_CODE)，继续尝试提取数据。"
        fi

        # === 生成报告 ===
        REPORT_FILE="${RUN_DIR}/summary.txt"
        METRICS=$(generate_report "$FORMAL_LOG" "$REPORT_FILE")
        IFS=',' read -r STATUS TPOT_VAL OUT_THRU_VAL TOTAL_THRU_VAL <<< "$METRICS"

        echo "$(date +%Y-%m-%d_%H:%M:%S),${RUN_DIR},$BSIZE,$DP,$batch_size_value,$target_count,$STATUS,$TPOT_VAL,$OUT_THRU_VAL,$TOTAL_THRU_VAL" >> "$GLOBAL_SUMMARY"

        log "测试完成。耗时: ${DURATION}s"
        log "结果摘要:"
        cat "$REPORT_FILE"

        # ==========================================================
        # 🚀 [修复重点] Profile 移动与解析 (增加详细日志和容错)
        # ==========================================================
        log ">>> 开始处理 Profile 数据..."

        PROFILE_FFN_DIR="${RUN_DIR}/profile/ffn"
        PROFILE_ATTENTION_DIR="${RUN_DIR}/profile/attention"
        mkdir -p "$PROFILE_FFN_DIR"
        mkdir -p "$PROFILE_ATTENTION_DIR"

        # 1. 处理 FFN Profile
        log "  [FFN] 查找并处理 Profile 数据..."
        FFN_SRC="/home/cyj/run_vllm_v13/ffn_prof"
        if [ -d "$FFN_SRC" ]; then
            # 获取最新的模式串
            FFN_PATTERN=$(ls -d "$FFN_SRC"/*/ 2>/dev/null | xargs -I {} basename {} | grep -oE '[0-9]{10,}' | head -n 1 | cut -c1-11)

            if [ -n "$FFN_PATTERN" ]; then
                log "  [FFN] 找到 Pattern: $FFN_PATTERN"
                log "  [FFN] 执行 Python 解析..."
                # 使用 || true 防止 Python 错误导致脚本退出
                python profileabc.py "$FFN_SRC" "$FFN_PATTERN" || log "  [FFN] Python 脚本执行完成 (可能有警告)"

                log "  [FFN] 开始移动文件到 $PROFILE_FFN_DIR..."
                shopt -s dotglob nullglob
                count=0
                for item in "$FFN_SRC"/*; do
                    if [ -e "$item" ]; then
                        mv "$item" "$PROFILE_FFN_DIR"/ && ((count++))
                    fi
                done
                shopt -u dotglob nullglob
                log "  [FFN] 成功移动 $count 个项目"
            else
                log "  [FFN] 未找到有效的 Pattern，跳过解析。"
            fi
        else
            log "  [FFN] 源目录 $FFN_SRC 不存在，跳过。"
        fi

        # 2. 处理 Attention Profile
        log "  [ATTN] 查找并处理 Profile 数据..."
        ATTN_SRC="/home/cyj/run_vllm_v13/profile"
        if [ -d "$ATTN_SRC" ]; then
            ATTN_PATTERN=$(ls -d "$ATTN_SRC"/*/ 2>/dev/null | xargs -I {} basename {} | grep -oE '[0-9]{10,}' | head -n 1 | cut -c1-11)

            if [ -n "$ATTN_PATTERN" ]; then
                log "  [ATTN] 找到 Pattern: $ATTN_PATTERN"
                # 如果有对应的 Python 脚本也可以在这里调用，目前只移动
                log "  [ATTN] 开始移动文件到 $PROFILE_ATTENTION_DIR..."
                shopt -s dotglob nullglob
                count=0
                for item in "$ATTN_SRC"/*; do
                    if [ -e "$item" ]; then
                        mv "$item" "$PROFILE_ATTENTION_DIR"/ && ((count++))
                    fi
                done
                shopt -u dotglob nullglob
                log "  [ATTN] 成功移动 $count 个项目"
            else
                log "  [ATTN] 未找到有效的 Pattern，跳过解析。"
            fi
        else
            log "  [ATTN] 源目录 $ATTN_SRC 不存在，跳过。"
        fi

        log ">>> Profile 处理全部结束！数据位于: $PROFILE_FFN_DIR"
        log ">>> Profile 处理全部结束！数据位于: $PROFILE_ATTENTION_DIR"
        # ==========================================================

        # === 清理 ===
        log "清理进程..."
        kill -9 $PID_ATT $PID_FFN 2>/dev/null
        cleanup_vllm

        log "冷却 10 秒..."
        sleep 10
    done
done

log "=========================================="
log "所有测试完成！"
log "全局汇总文件: $GLOBAL_SUMMARY"

if command -v column &> /dev/null; then
    echo "=== 全局测试结果预览 ==="
    tail -n +2 "$GLOBAL_SUMMARY" | column -t -s,
fi

