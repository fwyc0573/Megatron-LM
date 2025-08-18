#! /bin/bash

# Setting the environment variables
# export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=WARN # WARN INFO
# export NCCL_ALGO=RING #Ring
# export GLOO_SOCKET_IFNAME="bond4"

export CUDA_VISIBLE_DEVICES=4 #0,1,2,3

# export TORCH_CUDA_ARCH_LIST=Ampere

# Distributed training variables
NNODES=1
GPUS_PER_NODE=1 # 修改为1，因为我们每次只使用一个GPU模拟一个rank
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
NODE_RANK=0
MASTER_PORT=6002
MASTER_ADDR="localhost" #"localhost"


# Parallelism variables 
FIXED_PP=1 # 实际使用的PP大小，设为1
FIXED_TP=1 # 实际使用的TP大小，设为1
FIXED_DP=$((${GPU_NUM}/${TP}/${PP}))

BASE_PATH=/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM #/data/ytyang/yichengfeng/Megatron-LM

# 额外需要测试的3个配置
BATCH_CONFIGS=(
    "8192 32 4"   # world_size=8192, pp=32, tp=4
    "8192 64 4"
    "8192 64 2"
)

# 单一配置模式的默认设置（向后兼容）
DEFAULT_FAKE_PP=2
DEFAULT_FAKE_TP=2
DEFAULT_FAKE_WORLD_SIZE=8192

# 模型和批次设置
MODEL_SIZE=485 # 使用原脚本中的模型大小
# NUM_MICBATCH=1
MICRO_BATCH_SIZE=1

# 函数：验证配置的有效性
validate_config() {
    local world_size=$1
    local pp_size=$2
    local tp_size=$3
    local dp_size=$4

    if [ "$((dp_size * pp_size * tp_size))" -ne "$world_size" ]; then
        echo "ERROR: Invalid configuration!"
        echo "  WORLD_SIZE=${world_size}, PP=${pp_size}, TP=${tp_size}, DP=${dp_size}"
        echo "  DP * PP * TP = $((dp_size * pp_size * tp_size)) != WORLD_SIZE = ${world_size}"
        return 1
    fi

    if [ "$dp_size" -le 0 ]; then
        echo "ERROR: DP_SIZE must be positive, got ${dp_size}"
        return 1
    fi

    return 0
}


# size variables
# MODEL_SIZE="tiny" # "tiny" 6.7

if   [[ ${MODEL_SIZE} == 13 ]];   then HIDDEN_SIZE=5120;  NUM_HEAD=32; NUM_LAYERS=40;
elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_LAYERS=80;
elif [[ ${MODEL_SIZE} == 175 ]];  then HIDDEN_SIZE=12288;  NUM_HEAD=96; NUM_LAYERS=96;
elif [[ ${MODEL_SIZE} == "tiny" ]]; then HIDDEN_SIZE=128;  NUM_HEAD=8; NUM_LAYERS=4;
elif [[ ${MODEL_SIZE} == "2T" ]];  then HIDDEN_SIZE=25600;  NUM_HEAD=160; NUM_LAYERS=256;
elif [[ ${MODEL_SIZE} == "1T" ]];  then HIDDEN_SIZE=25600;  NUM_HEAD=160; NUM_LAYERS=128;
elif [[ ${MODEL_SIZE} == 485 ]];  then HIDDEN_SIZE=20480;  NUM_HEAD=160; NUM_LAYERS=96;
elif [[ ${MODEL_SIZE} == 30 ]];   then HIDDEN_SIZE=7680;  NUM_HEAD=48; NUM_LAYERS=40;
elif [[ ${MODEL_SIZE} == 40 ]];   then HIDDEN_SIZE=9216;  NUM_HEAD=72; NUM_LAYERS=40;
elif [[ ${MODEL_SIZE} == 6.7 ]];  then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_LAYERS=32;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi


DO_TRACE=True
# TRACE控制参数
TRAIN_ITERS=10
TRACE_ITER_NUM=1 # trace_iter_num的范围<=train_iters-1（除去第一次）
TRACE_START=$(($TRAIN_ITERS-$TRACE_ITER_NUM+1)) # [start, train_iters]
NSIGHT_START=$(($TRAIN_ITERS)) # [start, train_iters)


MAX_SEQ_LEN=2048 # 4096 2048 1024
MAX_POSITION_EMBEDDINGS=2048 # 4096 2048 1024

# 检查trace_iter_num是否在合理的范围内
if [ $TRACE_ITER_NUM -gt $((TRAIN_ITERS - 1)) ]; then
  echo "Error: trace_iter_num must be less than or equal to train_iters - 2"
  exit 1
fi

TRACE_ARGS=" \
       --do-trace $DO_TRACE \
       --trace-start $TRACE_START \
       --nsight-start $NSIGHT_START \
       "

FAKE_GPUS_PER_NODE=8
FAKE_WRANK=0
FAKE_LOCAL_RANK=0
# IS_SCALING_MODE=Falsef

# 全局常量定义（这些参数现在在execute_single_config函数中动态构建）
SEQ_LEN=$MAX_SEQ_LEN

# CHECKPOINT_PATH=<Specify path>
# BASE_PATH=/research/d1/gds/ytyang/yichengfeng/Megatron-LM
VOCAB_FILE=${BASE_PATH}/data/output_prefix_gpt2/gpt2-vocab.json
MERGE_FILE=${BASE_PATH}/data/output_prefix_gpt2/gpt2-merges.txt
DATA_PATH=${BASE_PATH}/data/output_prefix_gpt2/my-gpt2_text_document
DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1 \
    --vocab-size 51200 
"
# --vocab-size 3200

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters 1
"

# 函数：执行单个配置的模拟
execute_single_config() {
    local world_size=$1
    local pp_size=$2
    local tp_size=$3
    local config_index=$4

    # 计算DP大小
    local dp_size=$((world_size / (pp_size * tp_size)))

    echo ""
    echo "============================================================"
    echo "CONFIGURATION ${config_index}: WORLD_SIZE=${world_size}, PP=${pp_size}, TP=${tp_size}, DP=${dp_size}"
    echo "============================================================"

    # 验证配置
    if ! validate_config "$world_size" "$pp_size" "$tp_size" "$dp_size"; then
        echo "Skipping invalid configuration ${config_index}"
        return 1
    fi

    # 计算全局批次大小
    local global_batch_size=6144
    if [ $((global_batch_size % (MICRO_BATCH_SIZE * dp_size))) -ne 0 ]; then
        echo "ERROR: global_batch_size (${global_batch_size}) must be divisible by MICRO_BATCH_SIZE * dp_size (${MICRO_BATCH_SIZE} * ${dp_size})"
        return 1
    fi
    local NUM_MICBATCH=$((global_batch_size / (MICRO_BATCH_SIZE * dp_size)))
    # local global_batch_size=$((NUM_MICBATCH * MICRO_BATCH_SIZE * dp_size))

    # 创建配置特定的日志目录
    local log_name="SIM_GPT_${MODEL_SIZE}_Config${config_index}_WS${world_size}_PP${pp_size}_TP${tp_size}_DP${dp_size}_nMICROB${NUM_MICBATCH}_MICROB_SIZE${MICRO_BATCH_SIZE}_GLOBAL_BATCH${global_batch_size}"
    local log_dir="${BASE_PATH}/logs/${log_name}"

    echo "Creating log directory: ${log_dir}"
    mkdir -p ${log_dir}

    # 计算需要运行的特定ranks
    # 根据Megatron rank映射逻辑 (order="tp-cp-ep-dp-pp")
    # 对于dense模型，我们只需要运行以下特定ranks：
    # - PP rank = 0, 1, 2, ..., PP_SIZE-1 (所有PP stages)
    # - TP rank = 0 (固定，代表TP local rank 0)
    # - DP rank = 0 (固定，代表DP local rank 0)

    echo "Calculating selected ranks for dense model simulation..."
    echo "Configuration: PP=${pp_size}, TP=${tp_size}, DP=${dp_size}, WORLD_SIZE=${world_size}"

    # 根据rank映射公式计算特定ranks
    local selected_ranks=()

    # 计算每个PP stage在TP rank=0, DP rank=0时对应的world rank
    for pp_stage in $(seq 0 $((pp_size - 1)))
    do
        # 根据Megatron rank映射: global_rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size
        # 这里: tp_rank=0, dp_rank=0
        local rank=$((0 + 0 * tp_size + pp_stage * tp_size * dp_size))
        selected_ranks+=(${rank})
        echo "PP stage ${pp_stage} -> world rank ${rank}"
    done

    echo ""
    echo "============================================================"
    echo "Starting Megatron-LM Single GPU Simulation for selected ranks"
    echo "Model: GPT-${MODEL_SIZE}, PP=${pp_size}, TP=${tp_size}, DP=${dp_size}"
    echo "Selected ranks (PP stages 0-$((pp_size-1)) with TP=0, DP=0): ${selected_ranks[@]}"
    echo "Total ranks to simulate: ${#selected_ranks[@]} (instead of ${world_size})"
    echo "Time savings: ~$((100 - 100 * ${#selected_ranks[@]} / world_size))% reduction in execution time"
    echo "============================================================"

    # 构建配置特定的参数
    local gpt_args="
        --tensor-model-parallel-size ${FIXED_TP} \
        --pipeline-model-parallel-size ${FIXED_PP} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_HEAD} \
        --micro-batch-size ${MICRO_BATCH_SIZE} \
        --global-batch-size ${global_batch_size} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --train-iters ${TRAIN_ITERS} \
        --lr-decay-iters ${TRAIN_ITERS} \
        --tokenizer-type GPT2BPETokenizer \
        --distributed-backend nccl \
        --lr 0.00015 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --lr-warmup-fraction .01 \
        --clip-grad 1.0 \
        --fp16 \
    "

    local sim_args="
        --is-scaling-mode \
        --fake-pp ${pp_size} \
        --fake-tp ${tp_size} \
        --fake-dp ${dp_size} \
        --fake-world-size ${world_size} \
        --fake-wrank ${FAKE_WRANK} \
        --fake-gpus-per-node ${FAKE_GPUS_PER_NODE} \
        --fake-local-rank ${FAKE_LOCAL_RANK} \
    "

    # 迭代选定的ranks
    for current_fake_rank_id in "${selected_ranks[@]}"
    do
        echo "------------------------------------------------------------"
        echo ">>> Simulating FAKE RANK ID: ${current_fake_rank_id} (PP stage $((${current_fake_rank_id} / (${tp_size} * ${dp_size})))) <<<"
        echo "------------------------------------------------------------"

        # 为当前rank创建日志文件
        local rank_log_path="${log_dir}/fake_rank_${current_fake_rank_id}.log"

        echo "Running simulation for rank ${current_fake_rank_id}, log: ${rank_log_path}"

        # 使用torchrun执行Python脚本，添加fake-current-rank-id参数
        torchrun --nproc_per_node=${GPUS_PER_NODE} --nnodes=${NNODES} --node-rank ${NODE_RANK} --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT} ${BASE_PATH}/pretrain_llama.py \
            $gpt_args \
            $DATA_ARGS \
            $OUTPUT_ARGS \
            $sim_args \
            $TRACE_ARGS \
            --fake-current-rank-id ${current_fake_rank_id} \
            --distributed-backend nccl \
            --use-mcore-models \
            --seed 42 2>&1 | tee ${rank_log_path}

        # 检查执行状态
        exit_status=${PIPESTATUS[0]}
        if [ ${exit_status} -ne 0 ]; then
            echo "ERROR: Python script failed for rank ${current_fake_rank_id} with status ${exit_status}"
            echo "Check log: ${rank_log_path}"
            return 1
        fi

        echo ">>> Finished simulation for FAKE RANK ID: ${current_fake_rank_id} <<<"
    done

    # 配置完成总结
    local time_reduction=$((100 - 100 * ${#selected_ranks[@]} / world_size))
    echo ""
    echo "============================================================"
    echo "Configuration ${config_index} completed successfully!"
    echo ""
    echo "OPTIMIZATION SUMMARY:"
    echo "- Original ranks to simulate: ${world_size}"
    echo "- Optimized ranks simulated: ${#selected_ranks[@]}"
    echo "- Time reduction: ~${time_reduction}%"
    echo "- Simulated ranks: ${selected_ranks[@]}"
    echo ""
    echo "OUTPUT LOCATIONS:"
    echo "- Log directory: ${log_dir}"
    echo "- Profiler logs: ${BASE_PATH}/profiler_log/"
    echo ""
    echo "NOTE: For dense models, only PP stages 0-$((pp_size-1)) with TP=0, DP=0"
    echo "      are needed to capture the complete workload pattern."
    echo "============================================================"

    # 返回统计信息 (通过全局变量)
    CONFIG_STATS["config_${config_index}_world_size"]=$world_size
    CONFIG_STATS["config_${config_index}_pp_size"]=$pp_size
    CONFIG_STATS["config_${config_index}_tp_size"]=$tp_size
    CONFIG_STATS["config_${config_index}_dp_size"]=$dp_size
    CONFIG_STATS["config_${config_index}_original_ranks"]=$world_size
    CONFIG_STATS["config_${config_index}_optimized_ranks"]=${#selected_ranks[@]}
    CONFIG_STATS["config_${config_index}_time_reduction"]=$time_reduction
    CONFIG_STATS["config_${config_index}_log_dir"]=$log_dir

    return 0
}

# 主执行逻辑
echo "============================================================"
echo "Megatron-LM Workload Tracer - Batch Configuration Support"
echo "============================================================"

# 初始化统计信息存储
declare -A CONFIG_STATS

# 检查是否使用批量配置模式
if [ ${#BATCH_CONFIGS[@]} -gt 0 ]; then
    echo "Running in BATCH CONFIGURATION mode"
    echo "Total configurations to process: ${#BATCH_CONFIGS[@]}"
    echo ""

    # 显示所有配置
    echo "Configurations to process:"
    for i in "${!BATCH_CONFIGS[@]}"; do
        config="${BATCH_CONFIGS[$i]}"
        read -r world_size pp_size tp_size <<< "$config"
        dp_size=$((world_size / (pp_size * tp_size)))
        echo "  Config $((i+1)): WORLD_SIZE=${world_size}, PP=${pp_size}, TP=${tp_size}, DP=${dp_size}"
    done
    echo ""

    # 执行批量配置
    successful_configs=0
    failed_configs=0

    for i in "${!BATCH_CONFIGS[@]}"; do
        config="${BATCH_CONFIGS[$i]}"
        read -r world_size pp_size tp_size <<< "$config"
        config_index=$((i+1))

        echo ""
        echo "Processing configuration ${config_index}/${#BATCH_CONFIGS[@]}..."

        if execute_single_config "$world_size" "$pp_size" "$tp_size" "$config_index"; then
            ((successful_configs++))
            echo "✓ Configuration ${config_index} completed successfully"
        else
            ((failed_configs++))
            echo "✗ Configuration ${config_index} failed"
        fi
    done

    # 批量配置总结
    echo ""
    echo "============================================================"
    echo "BATCH CONFIGURATION SUMMARY"
    echo "============================================================"
    echo "Total configurations processed: ${#BATCH_CONFIGS[@]}"
    echo "Successful configurations: ${successful_configs}"
    echo "Failed configurations: ${failed_configs}"
    echo ""

    # 显示每个配置的优化效果
    echo "Optimization results by configuration:"
    for i in "${!BATCH_CONFIGS[@]}"; do
        config_index=$((i+1))
        if [[ -n "${CONFIG_STATS[config_${config_index}_world_size]}" ]]; then
            world_size=${CONFIG_STATS[config_${config_index}_world_size]}
            pp_size=${CONFIG_STATS[config_${config_index}_pp_size]}
            tp_size=${CONFIG_STATS[config_${config_index}_tp_size]}
            dp_size=${CONFIG_STATS[config_${config_index}_dp_size]}
            original_ranks=${CONFIG_STATS[config_${config_index}_original_ranks]}
            optimized_ranks=${CONFIG_STATS[config_${config_index}_optimized_ranks]}
            time_reduction=${CONFIG_STATS[config_${config_index}_time_reduction]}
            log_dir=${CONFIG_STATS[config_${config_index}_log_dir]}

            echo "  Config ${config_index}: WS=${world_size}, PP=${pp_size}, TP=${tp_size}, DP=${dp_size}"
            echo "    Ranks: ${optimized_ranks}/${original_ranks} (${time_reduction}% reduction)"
            echo "    Logs: ${log_dir}"
        else
            echo "  Config ${config_index}: Failed or skipped"
        fi
    done

else
    # 单一配置模式（向后兼容）
    echo "Running in SINGLE CONFIGURATION mode (backward compatibility)"
    echo "Using default configuration: WORLD_SIZE=${DEFAULT_FAKE_WORLD_SIZE}, PP=${DEFAULT_FAKE_PP}, TP=${DEFAULT_FAKE_TP}"

    if execute_single_config "$DEFAULT_FAKE_WORLD_SIZE" "$DEFAULT_FAKE_PP" "$DEFAULT_FAKE_TP" "1"; then
        echo "✓ Single configuration completed successfully"
    else
        echo "✗ Single configuration failed"
        exit 1
    fi
fi

echo ""
echo "============================================================"
echo "ALL SIMULATIONS COMPLETED!"
echo "============================================================"
echo "Timestamp: $(date)"
echo "Base path: ${BASE_PATH}"
echo "Model size: GPT-${MODEL_SIZE}"
echo ""
echo "Key benefits of this optimization:"
echo "- Only simulates essential ranks (PP stages 0 to PP_SIZE-1 with TP=0, DP=0)"
echo "- Captures complete workload patterns for dense models"
echo "- Significant time savings compared to full rank simulation"
echo "- Supports both single and batch configuration modes"
echo ""
echo "For detailed logs and profiler output, check the respective log directories."
echo "============================================================"