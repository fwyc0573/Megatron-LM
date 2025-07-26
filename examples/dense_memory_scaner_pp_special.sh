#! /bin/bash

# Setting the environment variables
# export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=WARN # WARN INFO
# export NCCL_ALGO=RING #Ring
# export GLOO_SOCKET_IFNAME="bond4"

export CUDA_VISIBLE_DEVICES=1 #0,1,2,3

# export TORCH_CUDA_ARCH_LIST=Ampere

# Distributed training variables
NNODES=1
GPUS_PER_NODE=1 # 修改为1，因为我们每次只使用一个GPU模拟一个rank
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
NODE_RANK=0
MASTER_PORT=6004
MASTER_ADDR="localhost" #"localhost"

# Parallelism variables 
PP=1 # 实际使用的PP大小，设为1
TP=1 # 实际使用的TP大小，设为1
DP=$((${GPU_NUM}/${TP}/${PP}))

BASE_PATH=/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM #/data/ytyang/yichengfeng/Megatron-LM

# 模拟的并行度设置 - 现在将在循环中动态设置
FAKE_WORLD_SIZE=8192

# 创建主日志目录
MODEL_SIZE=540 # 使用原脚本中的模型大小
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG_DIR=${BASE_PATH}/log/CONFIG_SWEEP_WS${FAKE_WORLD_SIZE}_${MODEL_SIZE}_${TIMESTAMP}
mkdir -p ${MAIN_LOG_DIR}

# 创建配置组合记录文件
CONFIG_LOG=${MAIN_LOG_DIR}/config_combinations.log
SUCCESS_LOG=${MAIN_LOG_DIR}/successful_configs.log
FAILED_LOG=${MAIN_LOG_DIR}/failed_configs.log

echo "Configuration sweep started at $(date)" > ${CONFIG_LOG}
echo "Successful configurations:" > ${SUCCESS_LOG}
echo "Failed configurations:" > ${FAILED_LOG}

MICRO_BATCH_SIZE=1

# size variables
# MODEL_SIZE="1T" # "tiny" 6.7

if   [[ ${MODEL_SIZE} == 13 ]];   then HIDDEN_SIZE=5120;  NUM_HEAD=32; NUM_LAYERS=40;
elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_LAYERS=80;
elif [[ ${MODEL_SIZE} == 175 ]];  then HIDDEN_SIZE=12288;  NUM_HEAD=96; NUM_LAYERS=96;
elif [[ ${MODEL_SIZE} == "tiny" ]]; then HIDDEN_SIZE=128;  NUM_HEAD=8; NUM_LAYERS=4;
elif [[ ${MODEL_SIZE} == "2T" ]];  then HIDDEN_SIZE=25600;  NUM_HEAD=160; NUM_LAYERS=256;
elif [[ ${MODEL_SIZE} == "1T" ]];  then HIDDEN_SIZE=25600;  NUM_HEAD=160; NUM_LAYERS=128;
elif [[ ${MODEL_SIZE} == 485 ]];  then HIDDEN_SIZE=20480;  NUM_HEAD=160; NUM_LAYERS=96;
elif [[ ${MODEL_SIZE} == 540 ]];  then HIDDEN_SIZE=18432;  NUM_HEAD=48; NUM_LAYERS=118; KV_CHANNELS=256;
elif [[ ${MODEL_SIZE} == 30 ]];   then HIDDEN_SIZE=7680;  NUM_HEAD=48; NUM_LAYERS=40;
elif [[ ${MODEL_SIZE} == 40 ]];   then HIDDEN_SIZE=9216;  NUM_HEAD=72; NUM_LAYERS=40;
elif [[ ${MODEL_SIZE} == 6.7 ]];  then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_LAYERS=32;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi


DO_TRACE=True
# TRACE控制参数
TRAIN_ITERS=5
TRACE_ITER_NUM=1 # trace_iter_num的范围<=train_iters-1（除去第一次）
TRACE_START=$(($TRAIN_ITERS-$TRACE_ITER_NUM+1)) # [start, train_iters]
NSIGHT_START=$(($TRAIN_ITERS)) # [start, train_iters]


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

echo "============================================================"
echo "Starting Megatron-LM Configuration Sweep for ${FAKE_WORLD_SIZE} cards"
echo "Model: GPT-${MODEL_SIZE}"
echo "Testing all valid PP/TP/DP combinations"
echo "============================================================"

# 计数器
total_configs=0
successful_configs=0
failed_configs=0

# 遍历所有可能的FAKE_TP值
for FAKE_TP in 1 2 4 8; do
    echo "Testing configurations with TP=${FAKE_TP}..."
    
    # 计算剩余卡数
    remaining=$((FAKE_WORLD_SIZE / FAKE_TP))
    
    for i in $(seq 1 $((remaining/2))); do
        # 检查FAKE_PP是否超过NUM_LAYERS
        FAKE_PP=$((i * 2))
        if [ $FAKE_PP -gt $NUM_LAYERS ]; then
            echo "Skipping PP=${FAKE_PP} (exceeds NUM_LAYERS=${NUM_LAYERS})"
            continue
        fi
        
        # 检查是否能整除
        if [ $((remaining % FAKE_PP)) -eq 0 ]; then
            FAKE_DP=$((remaining / FAKE_PP))
            
            # 检查FAKE_DP是否为1或2的倍数
            if [ $FAKE_DP -eq 1 ] || [ $((FAKE_DP % 2)) -eq 0 ]; then
                total_configs=$((total_configs + 1))
                
                # 设置NUM_MICBATCH = 6 * FAKE_PP
                NUM_MICBATCH=$((6 * FAKE_PP))
                GLOBAL_BATCH_SIZE=$((NUM_MICBATCH * MICRO_BATCH_SIZE * FAKE_DP))
                
                # 创建配置特定的日志目录
                CONFIG_NAME="TP${FAKE_TP}_PP${FAKE_PP}_DP${FAKE_DP}_nMICROB${NUM_MICBATCH}_MICROB_SIZE${MICRO_BATCH_SIZE}_GLOBAL_BATCH${GLOBAL_BATCH_SIZE}"
                CONFIG_LOG_DIR=${MAIN_LOG_DIR}/${CONFIG_NAME}
                mkdir -p ${CONFIG_LOG_DIR}
                
                echo "------------------------------------------------------------"
                echo ">>> Config ${total_configs}: TP=${FAKE_TP}, PP=${FAKE_PP}, DP=${FAKE_DP} <<<"
                echo ">>> NUM_MICBATCH=${NUM_MICBATCH}, GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} <<<"
                echo "------------------------------------------------------------"
                
                # 记录配置信息
                echo "Config ${total_configs}: TP=${FAKE_TP}, PP=${FAKE_PP}, DP=${FAKE_DP}, NUM_MICBATCH=${NUM_MICBATCH}, GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}" >> ${CONFIG_LOG}
                
                # 设置SIM_ARGS
                SIM_ARGS=" \
                       --is-scaling-mode \
                       --fake-world-size $FAKE_WORLD_SIZE \
                       --fake-wrank $FAKE_WRANK \
                       --fake-gpus-per-node $FAKE_GPUS_PER_NODE \
                       --fake-local-rank $FAKE_LOCAL_RANK \
                       --fake-pp $FAKE_PP \
                       --fake-dp $FAKE_DP \
                       --fake-tp $FAKE_TP \
                       --trace-memory \
                       --trace-memory-interval 0.1 \
                       "
                
                # 设置GPT_ARGS
                GPT_ARGS="
                    --main-tokenizer-type GPT2BPETokenizer \
                    --tensor-model-parallel-size $TP \
                    --pipeline-model-parallel-size $PP \
                    --num-layers $NUM_LAYERS \
                    --hidden-size $HIDDEN_SIZE \
                    --kv-channels $KV_CHANNELS \
                    --num-attention-heads $NUM_HEAD \
                    --seq-length $MAX_SEQ_LEN \
                    --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
                    --micro-batch-size $MICRO_BATCH_SIZE \
                    --global-batch-size $GLOBAL_BATCH_SIZE \
                    --lr 0.00015 \
                    --train-iters $TRAIN_ITERS \
                    --lr-decay-iters 320000 \
                    --lr-decay-style cosine \
                    --min-lr 1.0e-5 \
                    --weight-decay 1e-2 \
                    --lr-warmup-fraction .01 \
                    --clip-grad 1.0 \
                    --fp16 \
                "
                
                # 为当前配置创建日志文件
                RANK_LOG_PATH="${CONFIG_LOG_DIR}/global_rank_0.log"
                
                echo "Running configuration: TP=${FAKE_TP}, PP=${FAKE_PP}, DP=${FAKE_DP}, log: ${RANK_LOG_PATH}"
                
                # 只运行rank 0，添加错误处理
                set +e  # 允许命令失败而不退出脚本
                torchrun --nproc_per_node=${GPUS_PER_NODE} --nnodes=${NNODES} --node-rank ${NODE_RANK} --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT} ${BASE_PATH}/pretrain_llama.py \
                    $GPT_ARGS \
                    $DATA_ARGS \
                    $OUTPUT_ARGS \
                    $SIM_ARGS \
                    $TRACE_ARGS \
                    --fake-current-rank-id 0 \
                    --distributed-backend nccl \
                    --use-mcore-models \
                    --seed 42 2>&1 | tee ${RANK_LOG_PATH}
                
                exit_status=${PIPESTATUS[0]}
                set -e  # 重新启用错误时退出
                
                if [ ${exit_status} -eq 0 ]; then
                    successful_configs=$((successful_configs + 1))
                    echo "✓ SUCCESS: Configuration TP=${FAKE_TP}, PP=${FAKE_PP}, DP=${FAKE_DP}"
                    echo "Config ${total_configs}: TP=${FAKE_TP}, PP=${FAKE_PP}, DP=${FAKE_DP}, NUM_MICBATCH=${NUM_MICBATCH}, GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} - SUCCESS" >> ${SUCCESS_LOG}
                else
                    failed_configs=$((failed_configs + 1))
                    echo "✗ FAILED: Configuration TP=${FAKE_TP}, PP=${FAKE_PP}, DP=${FAKE_DP} with status ${exit_status}"
                    echo "Config ${total_configs}: TP=${FAKE_TP}, PP=${FAKE_PP}, DP=${FAKE_DP}, NUM_MICBATCH=${NUM_MICBATCH}, GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} - FAILED (status: ${exit_status})" >> ${FAILED_LOG}
                    
                    # 检查是否是OOM错误
                    if grep -q "out of memory\|OutOfMemoryError\|CUDA out of memory" ${RANK_LOG_PATH}; then
                        echo "  → OOM detected, continuing to next configuration..."
                    fi
                fi
                
                echo ">>> Finished config ${total_configs}: TP=${FAKE_TP}, PP=${FAKE_PP}, DP=${FAKE_DP} <<<"
                echo ""
            fi
        fi
    done
done

# 创建总结日志文件
SUMMARY_LOG=${MAIN_LOG_DIR}/summary.log

echo "============================================================"
echo "Configuration sweep completed!"
echo "Total configurations tested: ${total_configs}"
echo "Successful configurations: ${successful_configs}"
echo "Failed configurations: ${failed_configs}"
echo "Success rate: $(awk "BEGIN {printf \"%.2f%%\", ${successful_configs}/${total_configs}*100}")"
echo ""
echo "Main log directory: ${MAIN_LOG_DIR}"
echo "Configuration summary: ${CONFIG_LOG}"
echo "Successful configs: ${SUCCESS_LOG}"
echo "Failed configs: ${FAILED_LOG}"
echo "Profiler logs in: ${BASE_PATH}/profiler_log/"
echo "============================================================"

# 将总结信息写入summary.log
echo "============================================================" > ${SUMMARY_LOG}
echo "Megatron-LM Configuration Sweep Summary" >> ${SUMMARY_LOG}
echo "Completed at: $(date)" >> ${SUMMARY_LOG}
echo "============================================================" >> ${SUMMARY_LOG}
echo "" >> ${SUMMARY_LOG}

# 记录关键配置设置
echo "KEY CONFIGURATION SETTINGS:" >> ${SUMMARY_LOG}
echo "  Total GPUs (FAKE_WORLD_SIZE): ${FAKE_WORLD_SIZE}" >> ${SUMMARY_LOG}
echo "  Model Size: ${MODEL_SIZE}" >> ${SUMMARY_LOG}
echo "  Hidden Size: ${HIDDEN_SIZE}" >> ${SUMMARY_LOG}
echo "  Number of Layers: ${NUM_LAYERS}" >> ${SUMMARY_LOG}
echo "  Number of Attention Heads: ${NUM_HEAD}" >> ${SUMMARY_LOG}
echo "  Max Sequence Length: ${MAX_SEQ_LEN}" >> ${SUMMARY_LOG}
echo "  Micro Batch Size: ${MICRO_BATCH_SIZE}" >> ${SUMMARY_LOG}
echo "  Training Iterations: ${TRAIN_ITERS}" >> ${SUMMARY_LOG}
echo "  Base Path: ${BASE_PATH}" >> ${SUMMARY_LOG}
echo "  CUDA Visible Devices: ${CUDA_VISIBLE_DEVICES}" >> ${SUMMARY_LOG}
echo "" >> ${SUMMARY_LOG}

# 记录测试结果统计
echo "TEST RESULTS SUMMARY:" >> ${SUMMARY_LOG}
echo "  Total configurations tested: ${total_configs}" >> ${SUMMARY_LOG}
echo "  Successful configurations: ${successful_configs}" >> ${SUMMARY_LOG}
echo "  Failed configurations: ${failed_configs}" >> ${SUMMARY_LOG}
echo "  Success rate: $(awk "BEGIN {printf \"%.2f%%\", ${successful_configs}/${total_configs}*100}")" >> ${SUMMARY_LOG}
echo "" >> ${SUMMARY_LOG}

# 记录文件路径信息
echo "OUTPUT FILES:" >> ${SUMMARY_LOG}
echo "  Main log directory: ${MAIN_LOG_DIR}" >> ${SUMMARY_LOG}
echo "  Configuration summary: ${CONFIG_LOG}" >> ${SUMMARY_LOG}
echo "  Successful configs: ${SUCCESS_LOG}" >> ${SUMMARY_LOG}
echo "  Failed configs: ${FAILED_LOG}" >> ${SUMMARY_LOG}
echo "  Profiler logs: ${BASE_PATH}/profiler_log/" >> ${SUMMARY_LOG}
echo "  Memory traces: ${MAIN_LOG_DIR}/*/memory_traces_scaling/" >> ${SUMMARY_LOG}
echo "" >> ${SUMMARY_LOG}

# 记录测试的并行配置范围
echo "TESTED CONFIGURATION RANGES:" >> ${SUMMARY_LOG}
echo "  TP values tested: 1, 2, 4, 8" >> ${SUMMARY_LOG}
echo "  PP values tested: 2, 4, 6, 8, ... (even numbers up to remaining cards)" >> ${SUMMARY_LOG}
echo "  DP values: Calculated as remaining/(TP*PP), must be 1 or even" >> ${SUMMARY_LOG}
echo "  NUM_MICBATCH formula: 6 * PP" >> ${SUMMARY_LOG}
echo "  GLOBAL_BATCH_SIZE formula: NUM_MICBATCH * MICRO_BATCH_SIZE * DP" >> ${SUMMARY_LOG}
echo "" >> ${SUMMARY_LOG}

echo "============================================================" >> ${SUMMARY_LOG}

echo ""
echo "Summary report saved to: ${SUMMARY_LOG}"


SCRIPT_END_TIME=$(date +%s)
TOTAL_RUNTIME=$((SCRIPT_END_TIME - $(date -d "${TIMESTAMP:0:8} ${TIMESTAMP:9:2}:${TIMESTAMP:11:2}:${TIMESTAMP:13:2}" +%s)))
TOTAL_RUNTIME_HOURS=$((TOTAL_RUNTIME / 3600))
TOTAL_RUNTIME_MINUTES=$(((TOTAL_RUNTIME % 3600) / 60))
TOTAL_RUNTIME_SECONDS=$((TOTAL_RUNTIME % 60))

echo ""
echo "============================================================"
echo "PROFILING PERFORMANCE SUMMARY"
echo "============================================================"
echo "Total profiling runtime: ${TOTAL_RUNTIME_HOURS}h ${TOTAL_RUNTIME_MINUTES}m ${TOTAL_RUNTIME_SECONDS}s"
echo "Average time per configuration: $(awk "BEGIN {printf \"%.2f\", ${TOTAL_RUNTIME}/${total_configs}}")s"
echo "Configurations per hour: $(awk "BEGIN {printf \"%.2f\", ${total_configs}*3600/${TOTAL_RUNTIME}}")"
echo "============================================================"


echo "" >> ${SUMMARY_LOG}
echo "PROFILING PERFORMANCE SUMMARY:" >> ${SUMMARY_LOG}
echo "  Script started at: ${TIMESTAMP}" >> ${SUMMARY_LOG}
echo "  Script completed at: $(date +"%Y%m%d_%H%M%S")" >> ${SUMMARY_LOG}
echo "  Total profiling runtime: ${TOTAL_RUNTIME_HOURS}h ${TOTAL_RUNTIME_MINUTES}m ${TOTAL_RUNTIME_SECONDS}s" >> ${SUMMARY_LOG}
echo "  Average time per configuration: $(awk "BEGIN {printf \"%.2f\", ${TOTAL_RUNTIME}/${total_configs}}")s" >> ${SUMMARY_LOG}
echo "  Configurations per hour: $(awk "BEGIN {printf \"%.2f\", ${total_configs}*3600/${TOTAL_RUNTIME}}")" >> ${SUMMARY_LOG}
echo "  Success rate: $(awk "BEGIN {printf \"%.2f%%\", ${successful_configs}/${total_configs}*100}")" >> ${SUMMARY_LOG}
echo "" >> ${SUMMARY_LOG}


echo "CONFIGURATION DISTRIBUTION:" >> ${SUMMARY_LOG}
echo "  Tested TP values: 1, 2, 4, 8" >> ${SUMMARY_LOG}
echo "  PP range: 2 to min(${NUM_LAYERS}, remaining_cards)" >> ${SUMMARY_LOG}
echo "  DP constraint: Must be 1 or even number" >> ${SUMMARY_LOG}
echo "  Total world size: ${FAKE_WORLD_SIZE}" >> ${SUMMARY_LOG}
echo "" >> ${SUMMARY_LOG}

echo "============================================================" >> ${SUMMARY_LOG}
